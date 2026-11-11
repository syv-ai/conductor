"""How a run fails and what ``run.again`` reuses: the behaviours a host's side effects rest on.

A node's own failure is a ``Problem`` on the run, never an exception out of
it; only the outside world's failure is retried, and a failure never quotes a
value. ``again`` calls a node only for an item whose inputs differ from what an
earlier unit of that node received, so a CSV edited on one row sends one mail.
"""

import asyncio
import io
import json
import logging
import random
import time
from typing import Annotated, Any, ClassVar, Literal, TypedDict

import polars as pl
import pydantic
import pytest
from conductor import (
    SKIPPED,
    Ask,
    ExternalError,
    FromRun,
    GraphRecord,
    InvalidCallError,
    NodeError,
    NodeRecord,
    Param,
    Policy,
    Series,
    Table,
    make_graph,
    node,
)
from pydantic import BaseModel

from tests.test_core.nodes import place, registry_of


def _alone(definition: Any, **values: Any) -> Any:
    """A graph of one node, ``n``, with these values typed in."""
    return registry_of(definition).load(GraphRecord(nodes=[NodeRecord(id="n", node=definition.id, values=values)]))


@node("each", title="Each", description="")
def _each(items: list[str]) -> Series[str]:
    """The caller's list as a Series, so what reads it runs once per item."""
    return items


_EACH = NodeRecord(id="each", node="each")


def _events(run: Any, **kwargs: Any) -> list[Any]:
    async def collect() -> list[Any]:
        return [event async for event in run.stream(**kwargs)]

    return asyncio.run(collect())


# -- failures ---------------------------------------------------------------------


def test_what_a_units_result_rests_on_names_its_scope_by_address_not_by_how_a_class_prints(monkeypatch: pytest.MonkeyPatch):
    """A waiting id and a saved run hold the digest, so it is made of the scope's addresses as an author writes
    them (``words.result``), whatever the engine's class for an address is called or prints as."""
    from conductor._internal.checked_graph.input_plans import OutputAddress

    @node("words", title="Words", description="")
    def words(text: str) -> Series[str]:
        return text.split()

    @node("upper", title="Upper", description="")
    def upper(text: str) -> str:
        return text.upper()

    def digest() -> str:
        graph = registry_of(words, upper).load(make_graph(words, upper))
        return graph._functions[("upper",)].reuse_digest

    before = digest()
    monkeypatch.setattr(OutputAddress, "__repr__", lambda self: f"Renamed({self.node_id}, {self.output})")
    assert digest() == before


def test_a_foreign_exception_runs_once_and_is_execution_failed_and_quotes_nothing(caplog):
    """``{}['missing']`` under ``Policy(retries=3)`` is a bug, not weather: one call, no retry, and the problem says only that the node failed."""
    calls: list[int] = []

    @node("buggy", title="Buggy", description="", policy=Policy(retries=3, retry_delay=0.01))
    def buggy(text: str = "") -> str:
        calls.append(1)
        return {}["the secret"]  # type: ignore[no-any-return]

    with caplog.at_level(logging.ERROR, logger="conductor"):
        events = _events(_alone(buggy).with_inputs())

    assert len(calls) == 1 and not [e for e in events if e.type == "node_retry"]
    (failed,) = [e for e in events if e.type == "node_failed"]
    assert (failed.cause.type, failed.cause.msg, failed.cause.loc, failed.cause.ctx) == ("execution_failed", "The node failed; its developer finds why in the conductor log.", ("n",), {"error": "KeyError"})
    assert "secret" not in json.dumps([e.model_dump(mode="json") for e in events])
    assert "the secret" in caplog.text  # the text goes to the log, with its traceback
    assert events[-1].run.status == "failed" and events[-1].run.state.cause == failed.cause


@pytest.mark.parametrize("concurrency", [1, 2])
def test_a_node_error_a_wrong_output_and_an_ask_are_never_retried_whatever_retry_on_names(concurrency):
    calls: list[str] = []
    policy = Policy(retries=3, retry_delay=0.01, retry_on=(BaseException,), concurrency=concurrency)

    @node("own", title="Own", description="", policy=policy)
    def own(text: str = "") -> str:
        calls.append("own")
        raise NodeError("The document could not be read.", code="unreadable")

    @node("wrong", title="Wrong", description="", policy=policy)
    def wrong(text: str = "") -> int:
        calls.append("wrong")
        return "not a number"  # type: ignore[return-value]

    @node("asks", title="Asks", description="", policy=policy)
    def asks(text: str = "") -> str:
        calls.append("asks")
        raise Ask("Which?")

    causes = [_alone(definition).run().state.cause for definition in (own, wrong)]

    assert [c.type for c in causes] == ["unreadable", "invalid_output"] and _alone(asks).run().status == "waiting"
    assert calls == ["own", "wrong", "asks"]


def test_a_foreign_exception_named_in_retry_on_is_external_and_retries():
    calls: list[int] = []

    @node("flaky", title="Flaky", description="", policy=Policy(retries=2, retry_delay=0.01, retry_on=(ConnectionError,)))
    def flaky(text: str = "") -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("socket closed")
        return "ok"

    events = _events(_alone(flaky).with_inputs())

    retries = [e for e in events if e.type == "node_retry"]
    assert len(calls) == 3 and [(e.attempt, e.retries, e.loc[0]) for e in retries] == [(1, 2, "n"), (2, 2, "n")]
    assert retries[0].cause.type == "external_failed" and retries[0].cause.msg == "An outside service did not answer."
    assert events[-1].run.outputs == {"n.result": "ok"}


def test_an_exhausted_external_failure_is_external_failed_without_its_text():
    calls: list[int] = []

    @node("down", title="Down", description="", policy=Policy(retries=2, retry_delay=0.01, retry_on=(ConnectionError,)))
    def down(text: str = "") -> str:
        calls.append(1)
        raise ConnectionError("socket closed")

    run = _alone(down).run()

    assert len(calls) == 3 and run.status == "failed"
    assert (run.state.cause.type, run.state.cause.msg) == ("external_failed", "An outside service did not answer.")


def test_an_external_failure_the_node_raised_retries_and_keeps_its_message():
    """A node that knows the outside world failed says so with ``ExternalError``; the message it wrote is what people read."""
    calls: list[int] = []

    @node("bank", title="Bank", description="", policy=Policy(retries=1, retry_delay=0.01))
    def bank(text: str = "") -> str:
        calls.append(1)
        raise ExternalError("Nationalbanken svarede ikke.")

    run = _alone(bank).run()

    assert len(calls) == 2
    assert (run.state.cause.type, run.state.cause.msg) == ("external_failed", "Nationalbanken svarede ikke.")


def test_a_node_error_runs_once_and_keeps_its_message_code_and_ctx():
    calls: list[int] = []

    @node("strict", title="Strict", description="", policy=Policy(retries=3, retry_delay=0.01))
    def strict(text: str = "") -> str:
        calls.append(1)
        raise NodeError("Dokumentet kunne ikke læses.", code="unreadable", ctx={"pages": 0})

    events = _events(_alone(strict).with_inputs())

    (failed,) = [e for e in events if e.type == "node_failed"]
    assert len(calls) == 1
    assert failed.cause.model_dump(mode="json") == {"type": "unreadable", "loc": ["n"], "msg": "Dokumentet kunne ikke læses.", "ctx": {"pages": 0}}


@pytest.mark.parametrize("ctx", [{"handle": object()}, {"pages": {1, 2}}, {"pages": (1, 2)}])
def test_a_node_error_whose_ctx_has_no_json_form_is_refused_where_the_node_raises_it_so_the_run_still_saves(ctx: dict[str, Any]) -> None:
    @node("read", title="Read", description="")
    def read(path: str = "") -> str:
        raise NodeError("The document could not be read.", code="unreadable", ctx=ctx)

    with pytest.raises(pydantic.ValidationError):
        NodeError("The document could not be read.", ctx=ctx)
    with pytest.raises(pydantic.ValidationError):
        ExternalError("The service did not answer.", ctx=ctx)

    events = _events(_alone(read).with_inputs())
    run = _alone(read).run()

    assert (run.status, run.state.cause.type, run.state.cause.loc) == ("failed", "execution_failed", ("n",))
    assert json.loads(run.to_json())["state"]["cause"]["ctx"] == {"error": "ValidationError"}
    assert [json.loads(event.model_dump_json())["type"] for event in events][-2:] == ["node_failed", "run_end"]


def test_a_validation_error_the_node_raised_is_never_retried():
    calls: list[int] = []

    class Strict(BaseModel):
        n: int

    @node("parse", title="Parse", description="", policy=Policy(retries=3, retry_delay=0.01))
    def parse(text: str = "") -> int:
        calls.append(1)
        return Strict.model_validate({"n": text}).n

    run = _alone(parse, text="not a number").run()

    assert len(calls) == 1 and (run.state.cause.type, run.state.cause.ctx) == ("execution_failed", {"error": "ValidationError"})


def test_a_policy_out_of_bounds_or_of_another_type_is_refused_and_its_dump_leaves_retry_on_out():
    nan, inf = float("nan"), float("inf")
    for wrong in (
        {"concurrency": 0}, {"retries": -1}, {"retries": 1.5}, {"retry_delay": -1}, {"attempt_timeout": 0},
        {"attempt_timeout": nan}, {"attempt_timeout": inf}, {"retry_delay": nan}, {"retry_delay": inf},
        {"retries": True}, {"retries": "2"}, {"concurrency": 1.0}, {"retry_delay": "1"}, {"reuse_when_inputs_match": 1},
    ):
        with pytest.raises(pydantic.ValidationError):
            Policy(**wrong)
    assert Policy(retry_delay=2, attempt_timeout=5).attempt_timeout == 5  # a whole number of seconds is a number
    assert "retry_on" not in Policy(retry_on=(ConnectionError,)).model_dump()


def test_a_failed_item_is_retried_alone():
    """A node that runs per item retries the item that failed; the items that completed are not run again."""
    calls: list[str] = []

    @node("split", title="Split", description="")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @node("flaky-on-b", title="Flaky on b", description="", policy=Policy(retries=2, retry_delay=0.01))
    def flaky_on_b(text: str) -> str:
        calls.append(text)
        if text == "b" and calls.count("b") == 1:
            raise ExternalError("transient")
        return text.upper()

    graph = registry_of(split, flaky_on_b).load(make_graph(split, flaky_on_b, values={"split.text": "a,b,c"}))

    assert graph.run().results["flaky-on-b"]["result"] == ["A", "B", "C"]
    assert sorted(calls) == ["a", "b", "b", "c"]


def test_a_flaky_node_in_one_branch_retries_while_the_other_branch_completes():
    calls = {"a": 0, "b": 0}

    @node("flaky-a", title="Flaky a", description="", policy=Policy(retries=2, retry_delay=0.01))
    def flaky_a(text: str) -> str:
        calls["a"] += 1
        if calls["a"] == 1:
            raise ExternalError("first try")
        return f"A:{text}"

    @node("fast-b", title="Fast b", description="")
    def fast_b(text: str) -> str:
        calls["b"] += 1
        return f"B:{text}"

    @node("join-two", title="Join two", description="")
    def join_two(a: str, b: str) -> str:
        return f"{a}+{b}"

    graph = registry_of(flaky_a, fast_b, join_two).load(GraphRecord(
        nodes=[NodeRecord(id="n1", node="flaky-a", values={"text": "x"}), NodeRecord(id="n2", node="fast-b", values={"text": "y"}), NodeRecord(id="n3", node="join-two")],
        edges=[("n1.result", "n3.a"), ("n2.result", "n3.b")],
    ))

    assert graph.run().outputs == {"n3.result": "A:x+B:y"}
    assert calls == {"a": 2, "b": 1}


def test_a_run_that_failed_resumes_from_its_json_calling_only_what_had_not_finished():
    calls: list[str] = []
    broken = {"on": True}

    @node("split", title="Split", description="")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @node("mail", title="Mail", description="")
    def mail(to: str) -> str:
        calls.append(to)
        if to == "b" and broken["on"]:
            raise ValueError("smtp down")
        return f"sent {to}"

    graph = registry_of(split, mail).load(make_graph(split, mail, values={"split.text": "a,b,c"}))
    failed = graph.run()
    broken["on"] = False
    resumed = graph.load_run(failed.to_json()).resume()

    assert failed.status == "failed" and failed.state.cause.loc == ("mail", 1)
    assert resumed.results["mail"]["result"] == ["sent a", "sent b", "sent c"]
    assert calls.count("a") == 1 and calls.count("b") == 2


def test_a_cancelled_leg_keeps_what_it_finished_and_resuming_calls_none_of_it_again():
    calls: list[str] = []

    @node("split", title="Split", description="")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @node("count-calls", title="Count calls", description="")
    def count_calls(text: str) -> str:
        calls.append(text)
        return text

    graph = registry_of(split, count_calls).load(make_graph(split, count_calls, values={"split.text": "a,b,c,d"}))

    async def cancel_after_two() -> Any:
        cancel = asyncio.Event()
        last = None
        async for event in graph.with_inputs().stream(cancel=cancel):
            if len(calls) >= 2:
                cancel.set()
            last = event
        return last.run  # type: ignore[union-attr]

    stopped = asyncio.run(cancel_after_two())
    finished = graph.load_run(stopped.to_json()).resume()

    assert finished.results["count-calls"]["result"] == ["a", "b", "c", "d"]
    assert sorted(calls) == ["a", "b", "c", "d"]


# -- running again ----------------------------------------------------------------


def _mails() -> tuple[Any, list[str]]:
    """A graph that mails each address of a list, and the calls it made."""
    sent: list[str] = []

    @node("mail", title="Mail", description="")
    def mail(to: str) -> str:
        sent.append(to)
        return f"sent {to}"

    @node("confirm", title="Confirm", description="")
    def confirm(receipt: str) -> str:
        sent.append(f"confirm {receipt}")
        return receipt.upper()

    graph = registry_of(_each, mail, confirm).load(GraphRecord(
        nodes=[_EACH, NodeRecord(id="mail", node="mail"), NodeRecord(id="confirm", node="confirm")],
        edges=[("each.result", "mail.to"), ("mail.result", "confirm.receipt")],
    ))
    return graph, sent


def test_a_changed_item_is_new_work_all_the_way_down_and_nothing_else_is():
    graph, sent = _mails()
    first = graph.run(items=["ida", "bo", "al"])
    sent.clear()

    again = first.again(items=["ida", "bob", "al"])

    assert sent == ["bob", "confirm sent bob"]
    assert again.results["confirm"]["result"] == ["SENT IDA", "SENT BOB", "SENT AL"]


def test_an_item_inserted_above_or_a_duplicate_taken_out_moves_the_rest_without_calls():
    graph, sent = _mails()
    first = graph.run(items=["ida", "ida", "bo"])
    sent.clear()

    assert first.again(items=["eva", "ida", "bo"]).results["mail"]["result"] == ["sent eva", "sent ida", "sent bo"]
    assert sent == ["eva", "confirm sent eva"]
    sent.clear()
    first.again(items=["ida", "bo"])
    assert sent == []


def test_a_run_with_changed_inputs_streams_and_a_node_entirely_reused_says_how_much():
    graph, sent = _mails()
    first = graph.run(items=["ida", "bo"])
    sent.clear()

    events = _events(first.with_inputs(items=["ida", "bo", "al"]))

    completed = {e.loc[0]: e.reused_units for e in events if e.type == "node_complete"}
    assert sent == ["al", "confirm sent al"] and completed == {"each": 0, "mail": 2, "confirm": 2}
    assert events[-1].run.results["confirm"]["result"] == ["SENT IDA", "SENT BO", "SENT AL"]


def test_a_series_reordered_is_new_work_and_one_unchanged_is_not():
    calls: list[list[str]] = []

    @node("join-texts", title="Join texts", description="")
    def join_texts(texts: Series[str]) -> str:
        calls.append(list(texts))
        return "+".join(texts)

    graph = registry_of(join_texts).load(GraphRecord(nodes=[NodeRecord(id="j", node="join-texts")]))
    first = graph.run(texts=["a", "b"])

    first.again(texts=["a", "b"])
    first.again(texts=["b", "a"])

    assert calls == [["a", "b"], ["b", "a"]]


def test_a_dict_given_again_in_another_key_order_is_the_same_work():
    calls: list[Any] = []

    @node("take-mapping", title="Take mapping", description="")
    def take_mapping(value: dict[str, Any] | None) -> int:
        calls.append(value)
        return 1

    graph = registry_of(take_mapping).load(GraphRecord(nodes=[NodeRecord(id="t", node="take-mapping")]))
    first = graph.load_run(graph.run(value={"a": 1, "b": 2}).to_json())
    first.again(value={"b": 2, "a": 1})

    assert calls == [{"a": 1, "b": 2}]


def test_what_a_value_is_stays_apart_in_what_a_unit_received():
    calls: list[Any] = []

    @node("echo-any", title="Echo any", description="")
    def echo_any(value: Any) -> str:
        calls.append(value)
        return repr(value)

    graph = registry_of(echo_any).load(GraphRecord(nodes=[NodeRecord(id="e", node="echo-any")]))
    run = graph.run(value=1)
    for value in (1, "1", True, 1.0, [1], {"a": 1}):
        run = run.again(value=value)

    assert calls == [1, "1", True, 1.0, [1], {"a": 1}]


def test_a_from_run_value_is_not_part_of_what_a_unit_received():
    calls: list[str] = []

    class Caller:
        def __init__(self, name: str) -> None:
            self.name = name

    @node("signed", title="Signed", description="")
    def signed(text: str, who: FromRun[Caller]) -> str:
        calls.append(who.name)
        return f"{text} by {who.name}"

    graph = registry_of(signed).load(GraphRecord(nodes=[NodeRecord(id="s", node="signed")]))
    first = graph.run(text="hi", from_run={Caller: Caller("ida")})
    again = first.again(from_run={Caller: Caller("bo")})

    assert calls == ["ida"] and again.outputs == {"s.result": "hi by ida"}


def test_a_value_with_no_json_form_is_called_again_and_never_fails_the_run():
    calls: list[Any] = []

    class Opaque:
        """A host's value pydantic carries but has no JSON form for."""
        type_id: ClassVar[str] = "Opaque"
        type_title: ClassVar[str] = "Opaque"

        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
            from pydantic_core import core_schema

            return core_schema.is_instance_schema(cls)

    @node("make", title="Make", description="")
    def make(text: str) -> Opaque:
        return Opaque()

    @node("use", title="Use", description="")
    def use(thing: Opaque) -> str:
        calls.append(thing)
        return "used"

    graph = registry_of(make, use).load(make_graph(make, use))
    first = graph.run(text="a")
    again = first.again(text="a")

    assert again.status == "complete" and again.outputs == {"use.result": "used"}
    assert len(calls) == 2


def test_an_answer_stays_with_its_item_when_an_item_is_inserted_above_and_an_edited_item_asks_again():
    @node("approve", title="Approve", description="")
    def approve(name: str) -> bool:
        raise Ask(f"Approve {name}?")

    graph = registry_of(_each, approve).load(GraphRecord(nodes=[_EACH, NodeRecord(id="a", node="approve")], edges=[("each.result", "a.name")]))
    run = graph.run(items=["ida", "bo"])
    run = run.answer("a@0", True).answer("a@1", False).resume()

    moved = run.again(items=["eva", "ida", "bo"])
    edited = run.again(items=["ida", "bob"])

    assert [place(w) for w in moved.waiting] == ["a@0"] and moved.results["a"]["result"] == [None, True, False]
    assert [w.inputs["name"] for w in edited.waiting] == ["bob"] and edited.results["a"]["result"] == [True, None]


def test_a_table_output_reads_back_by_its_own_column_types_whatever_graphs_came_before():
    """A column's check belongs to its field, so graphs loaded and dropped before never lend theirs."""
    import gc
    from datetime import date
    from typing import TypedDict

    import polars as pl

    class Dated(TypedDict):
        v: date

    class Texts(TypedDict):
        v: str

    def outs(kind: str) -> dict[str, Any]:
        from conductor import Table

        return {"result": Table[Dated] if kind == "date" else Table[Texts]}

    @node("mk", title="Mk", description="", compute_outputs=outs)
    def mk(kind: str) -> dict[str, Any]:
        return {"result": pl.DataFrame({"v": [date(2020, 1, 1)] if kind == "date" else ["2020-01-01"]})}

    registry = registry_of(mk)
    for i in range(60):
        kind = "date" if i % 2 == 0 else "str"
        graph = registry.load(GraphRecord(nodes=[NodeRecord(id="m", node="mk", values={"kind": kind})]))
        value = graph.load_run(graph.run().to_json()).results["m"]["result"]["v"][0]
        assert isinstance(value, date) == (kind == "date"), (i, value)
        del graph
        gc.collect()


# -- what again reuses, item by item -----------------------------------------------------------


@node("docs", title="Docs", description="")
def _docs(text: str) -> Series[str]:
    return text.split("|") if text else []


class _Line(TypedDict):
    word: str


@node("line-records", title="Line records", description="")
def _line_records(table: Table[_Line]) -> Series[_Line]:
    return table.to_dicts()


def test_a_document_inserted_at_the_top_reads_only_the_new_document_and_its_lines_are_read_where_they_now_are():
    read: list[str] = []
    upper_calls: list[str] = []

    @node("lines", title="Lines", description="")
    def lines(text: str) -> Table[_Line]:
        read.append(text)
        return pl.DataFrame({"word": text.split(",")})

    @node("shout", title="Shout", description="")
    def shout(word: str) -> str:
        upper_calls.append(word)
        return word.upper()

    graph = registry_of(_docs, lines, _line_records, shout).load(GraphRecord(
        nodes=[NodeRecord(id="d", node="docs"), NodeRecord(id="l", node="lines"), NodeRecord(id="r", node="line-records"), NodeRecord(id="s", node="shout")],
        edges=[("d.result", "l.text"), ("l.result", "r.table"), ("r.result.word", "s.word")],
    ))
    first = graph.run(text="a,b|c")
    read.clear()
    upper_calls.clear()

    again = first.again(text="x|a,b|c")

    assert read == ["x"] and upper_calls == ["x"]
    assert again.results["s"]["result"] == [["X"], ["A", "B"], ["C"]]


def _filtered(threshold_default: int = 2) -> tuple[Any, list[str], list[str]]:
    """CSV text, read to a table, filtered by a ``min`` the caller gives, its names read whole; the calls of the filter and the reader."""
    filtered: list[str] = []
    counted: list[str] = []

    class Row(TypedDict):
        name: str
        age: int

    @node("read-rows", title="Read rows", description="")
    def read_rows(text: str) -> Table[Row]:
        return pl.read_csv(io.StringIO(text))

    @node("threshold", title="Threshold", description="")
    def threshold(min: int = threshold_default) -> int:
        return min

    @node("older-than", title="Older than", description="")
    def older_than(rows: pl.DataFrame, min: int) -> Table[Row]:
        filtered.append("called")
        return rows.filter(pl.col("age") > min)

    @node("names", title="Names", description="")
    def names(rows: pl.DataFrame) -> str:
        counted.append("called")
        return ",".join(rows["name"])

    graph = registry_of(read_rows, threshold, older_than, names).load(GraphRecord(
        nodes=[NodeRecord(id="r", node="read-rows"), NodeRecord(id="t", node="threshold"), NodeRecord(id="f", node="older-than"), NodeRecord(id="n", node="names")],
        edges=[("r.result", "f.rows"), ("t.result", "f.min"), ("f.result", "n.rows")],
    ))
    return graph, filtered, counted


def test_a_row_a_filter_drops_inserted_above_leaves_the_whole_table_reader_uncalled():
    graph, filtered, counted = _filtered()
    first = graph.run(text="name,age\nida,30\nbo,40\n")
    filtered.clear()
    counted.clear()

    again = first.again(text="name,age\neva,1\nida,30\nbo,40\n")

    assert filtered == ["called"] and counted == []
    assert again.outputs == {"n.result": "ida,bo"}


def test_equal_tables_made_by_different_filters_are_the_same_work_for_the_reader_below():
    graph, filtered, counted = _filtered()
    first = graph.run(text="name,age\nida,30\nbo,40\n", min=2)
    counted.clear()

    again = first.again(min=10)

    assert filtered == ["called", "called"] and counted == []
    assert again.outputs == {"n.result": "ida,bo"}


def test_a_reused_series_and_items_called_afresh_line_up_item_by_item_after_a_document_is_inserted_above():
    @node("words", title="Words", description="")
    def words(text: str) -> Series[str]:
        return text.split(",")

    @node("stamp", title="Stamp", description="", policy=Policy(reuse_when_inputs_match=False))
    def stamp(word: str) -> str:
        return f"<{word}>"

    @node("show", title="Show", description="")
    def show(words: Series[str], stamps: Series[str]) -> str:
        return ";".join(f"{w}={s}" for w, s in zip(words, stamps, strict=True))

    graph = registry_of(_docs, words, stamp, show).load(GraphRecord(
        nodes=[NodeRecord(id="d", node="docs"), NodeRecord(id="w", node="words"), NodeRecord(id="t", node="stamp"), NodeRecord(id="s", node="show")],
        edges=[("d.result", "w.text"), ("w.result", "t.word"), ("w.result", "s.words"), ("t.result", "s.stamps")],
    ))
    first = graph.run(text="a,b|c")

    again = first.again(text="x,y,z|a,b|c")

    assert first.results["s"]["result"] == ["a=<a>;b=<b>", "c=<c>"]
    assert again.results["s"]["result"] == ["x=<x>;y=<y>;z=<z>", "a=<a>;b=<b>", "c=<c>"]


@pytest.mark.parametrize("rows", ["a,b", "a,b,c,d", "x,y,z"], ids=["fewer", "more", "same-count"])
def test_again_with_changed_items_above_an_embedded_graph_gives_its_inner_fixed_value_node_on_every_item(rows):
    @node("label", title="Label", description="")
    def label(prefix: str) -> str:
        return prefix + ":"

    @node("pair", title="Pair", description="")
    def pair(text: str, tag: str) -> str:
        return tag + text

    @node("same", title="Same", description="")
    def same(text: str) -> str:
        return text

    inner = node("tagged", title="Tagged", description="")(GraphRecord(
        nodes=[NodeRecord(id="same", node="same"), NodeRecord(id="label", node="label", values={"prefix": "p"}), NodeRecord(id="pair", node="pair")],
        edges=[("same.result", "pair.text"), ("label.result", "pair.tag")],
    ))
    graph = registry_of(_docs, same, label, pair, inner).load(GraphRecord(
        nodes=[NodeRecord(id="d", node="docs"), NodeRecord(id="t", node="tagged")], edges=[("d.result", "t.text")],
    ))
    first = graph.run(text="a|b|c")

    again = first.again(text=rows.replace(",", "|"))
    direct = graph.run(text=rows.replace(",", "|"))

    assert again.status == "complete" and again.results == direct.results
    assert again.results["t"]["pair.result"] == [f"p:{r}" for r in rows.split(",")]


def test_an_input_filled_again_overrides_the_first_value_and_the_earlier_run_is_untouched():
    graph, sent = _mails()
    first = graph.run(items=["ida"])

    prepared = first.with_inputs(items=["bo"])
    again = prepared.resume()

    assert "items" in graph.inputs
    assert first.inputs != prepared.inputs and first.results["mail"]["result"] == ["sent ida"]
    assert again.results["mail"]["result"] == ["sent bo"]


def _csv_mails() -> tuple[Any, list[str]]:
    """A CSV of names and emails, and a ``body`` (default ``hej``) sent to each email; the mails sent."""
    sent: list[str] = []

    class Person(TypedDict):
        name: str
        email: str

    @node("read-people", title="Read people", description="")
    def read_people(text: str) -> Table[Person]:
        return pl.read_csv(io.StringIO(text))

    @node("people", title="People", description="")
    def people(table: Table[Person]) -> Series[Person]:
        return table.to_dicts()

    @node("body", title="Body text", description="")
    def body_text(body: str = "hej") -> str:
        return body

    @node("send-body", title="Send body", description="")
    def send_body(email: str, body: str) -> str:
        sent.append(email)
        return f"{body} {email}"

    graph = registry_of(read_people, people, body_text, send_body).load(GraphRecord(
        nodes=[NodeRecord(id="r", node="read-people"), NodeRecord(id="p", node="people"), NodeRecord(id="b", node="body"), NodeRecord(id="m", node="send-body")],
        edges=[("r.result", "p.table"), ("p.result.email", "m.email"), ("b.result", "m.body")],
    ))
    return graph, sent


def test_a_sorted_csv_or_an_edit_to_a_column_nothing_reads_sends_no_mail_and_a_changed_body_sends_every_one():
    graph, sent = _csv_mails()
    first = graph.run(text="name,email\nida,i@a.dk\nbo,b@a.dk\n")
    sent.clear()

    sorted_ = first.again(text="name,email\nbo,b@a.dk\nida,i@a.dk\n")
    renamed = first.again(text="name,email\nIda,i@a.dk\nBo,b@a.dk\n")
    assert sent == []
    assert sorted_.results["m"]["result"] == ["hej b@a.dk", "hej i@a.dk"]
    assert renamed.results["m"]["result"] == ["hej i@a.dk", "hej b@a.dk"]

    new_body = first.again(body="hi")

    assert sent == ["i@a.dk", "b@a.dk"] and new_body.results["m"]["result"] == ["hi i@a.dk", "hi b@a.dk"]


def test_a_document_inserted_above_items_split_from_items_calls_only_the_new_items():
    calls: list[str] = []

    @node("words", title="Words", description="")
    def words(text: str) -> Series[str]:
        return text.split(",")

    @node("shout", title="Shout", description="")
    def shout(text: str) -> str:
        calls.append(text)
        return text.upper()

    graph = registry_of(_docs, words, shout).load(make_graph(_docs, words, shout))
    first = graph.run(text="a,b|c")
    calls.clear()

    again = first.again(text="x,y|a,b|c")

    assert calls == ["x", "y"]
    assert again.results["shout"]["result"] == [["X", "Y"], ["A", "B"], ["C"]]


def _approvals() -> Any:
    @node("approve", title="Approve", description="")
    def approve(name: str) -> bool:
        raise Ask(f"Approve {name}?")

    return registry_of(_each, approve).load(GraphRecord(nodes=[_EACH, NodeRecord(id="a", node="approve")], edges=[("each.result", "a.name")]))


def test_each_item_still_waiting_after_again_is_announced_on_the_stream():
    run = _approvals().run(items=["ida", "bo"])

    events = _events(run.with_inputs(items=["eva", "ida", "bo"]))

    assert [place(e.unit) for e in events if e.type == "unit_waiting"] == ["a@0", "a@1", "a@2"]
    assert events[-1].status == "waiting"


def test_an_again_that_fails_before_it_reaches_the_old_items_still_reuses_them_on_resume():
    broken = {"on": True}
    sent: list[str] = []

    @node("mail-or-fail", title="Mail or fail", description="")
    def mail_or_fail(to: str) -> str:
        sent.append(to)
        if to == "eva" and broken["on"]:
            raise ValueError("smtp down")
        return f"sent {to}"

    graph = registry_of(_each, mail_or_fail).load(GraphRecord(nodes=[_EACH, NodeRecord(id="m", node="mail-or-fail")], edges=[("each.result", "m.to")]))
    first = graph.run(items=["ida", "bo"])
    sent.clear()

    failed = first.again(items=["eva", "ida", "bo"])
    assert failed.status == "failed" and sent == ["eva"]
    broken["on"] = False
    resumed = graph.load_run(failed.to_json()).resume()

    assert sent == ["eva", "eva"] and resumed.results["m"]["result"] == ["sent eva", "sent ida", "sent bo"]


def test_an_again_cancelled_before_the_node_runs_then_resumed_sends_only_the_new_mail():
    graph, sent = _mails()
    first = graph.run(items=["ida", "bo"])
    sent.clear()
    cancel = asyncio.Event()
    cancel.set()

    stopped = _events(first.with_inputs(items=["ida", "bo", "al"]), cancel=cancel)[-1].run
    assert stopped.status == "cancelled"
    resumed = graph.load_run(stopped.to_json()).resume()

    assert sent == ["al", "confirm sent al"] and resumed.results["confirm"]["result"] == ["SENT IDA", "SENT BO", "SENT AL"]


def test_after_an_answered_question_again_with_one_item_added_sends_one_mail():
    sent: list[str] = []

    @node("approve", title="Approve", description="")
    def approve(name: str) -> str:
        raise Ask(f"Mail {name}?")

    @node("mail", title="Mail", description="")
    def mail(to: str) -> str:
        sent.append(to)
        return f"sent {to}"

    graph = registry_of(_each, approve, mail).load(GraphRecord(
        nodes=[_EACH, NodeRecord(id="a", node="approve"), NodeRecord(id="m", node="mail")], edges=[("each.result", "a.name"), ("a.result", "m.to")],
    ))
    run = graph.run(items=["ida", "bo"]).answer("a@0", "ida@a.dk").answer("a@1", "bo@a.dk").resume()
    sent.clear()

    again = run.again(items=["ida", "bo", "al"])
    assert [place(w) for w in again.waiting] == ["a@2"]
    done = again.answer("a@2", "al@a.dk").resume()

    assert sent == ["al@a.dk"] and done.results["m"]["result"] == ["sent ida@a.dk", "sent bo@a.dk", "sent al@a.dk"]


def test_a_third_run_reuses_what_the_second_had_itself_reused():
    graph, sent = _mails()
    second = graph.run(items=["ida", "bo"]).again(items=["ida", "bo", "al"])
    sent.clear()

    third = second.again(items=["ida", "bo", "al", "eva"])

    assert sent == ["eva", "confirm sent eva"] and third.results["mail"]["result"][:2] == ["sent ida", "sent bo"]


def test_a_cancelled_leg_between_two_runs_reuses_what_one_direct_leg_does():
    rng = random.Random(7)
    graph, sent = _mails()
    names = ["ida", "bo", "al", "eva", "kim"]
    items = ["ida", "bo"]
    run = graph.run(items=items)
    cancel = asyncio.Event()
    cancel.set()
    for _ in range(20):
        edited = list(items)
        choice = rng.randrange(3)
        if choice == 0:
            edited.insert(rng.randrange(len(edited) + 1), rng.choice(names))
        elif choice == 1 and edited:
            edited.pop(rng.randrange(len(edited)))
        else:
            rng.shuffle(edited)
        sent.clear()
        direct = run.again(items=edited)
        direct_calls = sorted(sent)
        sent.clear()
        stopped = _events(run.with_inputs(items=edited), cancel=cancel)[-1].run
        through = graph.load_run(stopped.to_json()).resume()
        assert sorted(sent) == direct_calls and through.results == direct.results
        run, items = direct, edited


def test_a_call_that_returned_skipped_is_a_result_and_is_not_made_again_live_or_through_json():
    """A node that returned ``SKIPPED`` was called: ``again`` takes that as it takes any result, and what reads it is skipped on the later run as on the first."""
    calls: list[str] = []

    @node("only-long", title="Only long", description="")
    def only_long(text: str) -> str:
        calls.append(text)
        return text if len(text) > 2 else SKIPPED

    @node("shout", title="Shout", description="")
    def shout(text: str) -> str:
        return text.upper()

    graph = registry_of(_each, only_long, shout).load(GraphRecord(
        nodes=[_EACH, NodeRecord(id="o", node="only-long"), NodeRecord(id="s", node="shout")], edges=[("each.result", "o.text"), ("o.result", "s.text")],
    ))
    first = graph.run(items=["a", "abc", "b"])
    assert first.results["o"] == {"result": [None, "abc", None]} and first.results["s"] == {"result": [None, "ABC", None]}
    assert calls == ["a", "abc", "b"]

    events = _events(first.with_inputs())
    again = events[-1].run
    loaded_again = graph.load_run(first.to_json()).again()

    assert calls == ["a", "abc", "b"]
    assert [(e.units, e.reused_units) for e in events if e.type == "node_complete" and e.loc == ("o",)] == [(3, 3)]
    assert again.results == first.results == loaded_again.results and first.not_called == []
    nothing_long = _events(graph.with_inputs(items=["a", "b"]))
    assert [e.loc for e in nothing_long if e.type == "node_skipped"] == [("o",), ("s",)]  # every unit came to no value, called or not


def test_a_first_run_reports_no_reuse():
    graph, _ = _mails()

    events = _events(graph.with_inputs(items=["ida", "bo"]))

    assert {e.loc[0]: e.reused_units for e in events if e.type == "node_complete"} == {"each": 0, "mail": 0, "confirm": 0}


def test_a_run_with_changed_inputs_not_yet_started_shows_no_results():
    graph, sent = _mails()
    first = graph.run(items=["ida"])
    sent.clear()

    prepared = first.with_inputs(items=["ida", "bo"])

    assert prepared.status == "ready" and sent == []
    assert prepared.results == {"each": {"result": None}, "mail": {"result": None}, "confirm": {"result": None}}  # no Series made yet: nothing so far


def test_a_typed_in_value_counts_as_its_type_writes_it_not_as_it_was_spelled():
    calls: list[float] = []

    @node("half", title="Half", description="")
    def half(x: float) -> float:
        calls.append(x)
        return x / 2

    graph = registry_of(half).load(GraphRecord(nodes=[NodeRecord(id="h", node="half")]))
    first = graph.run(x=2)

    again = first.again(x=2.0)

    assert calls == [2.0] and again.outputs == {"h.result": 1.0}


def test_a_table_column_its_row_type_doesnt_name_reads_back_from_json_as_written():
    import datetime as dt
    import math

    import polars as pl
    from conductor import Table

    @node("mixed", title="Mixed", description="")
    def mixed(text: str) -> Table:
        return pl.DataFrame({"when": [dt.date(2026, 1, 1)], "at": [dt.time(1, 2)], "x": [math.nan], "n": [1]})

    graph = registry_of(mixed).load(GraphRecord(nodes=[NodeRecord(id="m", node="mixed")]))
    frame = graph.load_run(graph.run(text="a").to_json()).results["m"]["result"]

    assert frame.schema == pl.Schema({"when": pl.Date, "at": pl.Time, "x": pl.Float64, "n": pl.Int64})
    assert math.isnan(frame["x"][0])


def _alone_node() -> Any:
    @node("n", title="N", description="")
    def n(text: str = "") -> str:
        return text

    return n


def test_waiting_items_and_a_saved_run_keep_node_then_item_order_whatever_threads_did():
    import random

    @node("slow-ask", title="Slow ask", description="", policy=Policy(concurrency=4))
    def slow_ask(text: str) -> str:
        import time

        time.sleep(random.random() * 0.02)
        raise Ask(f"{text}?")

    @node("split", title="Split", description="")
    def split(text: str) -> Series[str]:
        return text.split(",")

    graph = registry_of(split, slow_ask).load(make_graph(split, slow_ask))
    for _ in range(3):
        run = graph.run(text=",".join("abcdefgh"))
        assert [place(w) for w in run.waiting] == [f"slow-ask@{i}" for i in range(8)]
        assert [question["prompt"] for question in json.loads(run.to_json())["units"]["slow-ask"]["questions"].values()] == [f"{letter}?" for letter in "abcdefgh"]


def test_a_saved_run_that_does_not_fit_is_refused_naming_where_and_never_with_a_raw_error():
    @node("split", title="Split", description="")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @node("echo", title="Echo", description="")
    def echo(text: str) -> str:
        return text

    graph = registry_of(split, echo).load(make_graph(split, echo))
    saved = json.loads(graph.run(text="a,b,c").to_json())
    for tamper, at in (
        (lambda s: s["units"]["echo"]["states"].append(None), "units.echo"),
        (lambda s: s["units"]["echo"]["outputs"]["result"].pop(), "units.echo"),
        (lambda s: s["units"]["echo"]["outputs"].pop("result"), "units.echo.outputs.result"),
        (lambda s: s["units"]["echo"]["outputs"]["result"].__setitem__(0, 5), "units.echo.outputs.result.0"),
    ):
        copy = json.loads(json.dumps(saved))
        tamper(copy)
        with pytest.raises(InvalidCallError) as refused:
            graph.load_run(copy)
        (problem,) = refused.value.errors()
        assert (problem.type, problem.ctx["at"]) == ("state_unreadable", at), problem


# -- identical items, generations, waiting items and forgetting --------------------------------


def _answered(names: list[str], answers: list[bool]) -> Any:
    @node("approve", title="Approve", description="")
    def approve(name: str) -> bool:
        raise Ask(f"Mail {name}?")

    graph = registry_of(_each, approve).load(GraphRecord(nodes=[_EACH, NodeRecord(id="a", node="approve")], edges=[("each.result", "a.name")]))
    run = graph.run(items=names)
    return run.answer({w.id: answer for w, answer in zip(run.waiting, answers, strict=True)}).resume()


def test_identical_items_keep_their_own_answers_kth_to_kth_when_an_item_is_inserted_above():
    """Bo yes, Bo no, Al yes; with Eva above, the first Bo is still yes and the second still no."""
    run = _answered(["bo", "bo", "al"], [True, False, True])

    moved = run.again(items=["eva", "bo", "bo", "al"])

    assert [place(w) for w in moved.waiting] == ["a@0"] and moved.results["a"]["result"] == [None, True, False, True]
    assert run.again(items=["bo", "bo", "al"]).results["a"]["result"] == [True, False, True]


def test_identical_answered_items_keep_their_answers_in_order_when_they_all_move_and_through_json():
    run = _answered(["bo", "al", "bo"], [True, True, False])

    moved = run.graph.load_run(run.to_json()).again(items=["al", "bo", "bo"])

    assert moved.results["a"]["result"] == [True, True, False] and moved.waiting == []


def test_a_third_identical_item_is_asked_and_two_earlier_answers_cover_only_two_items():
    run = _answered(["bo", "bo"], [True, False])

    more = run.again(items=["bo", "bo", "bo"])

    assert [place(w) for w in more.waiting] == ["a@2"] and more.results["a"]["result"] == [True, False, None]


def test_identical_lines_answered_differently_under_documents_keep_their_documents_when_the_documents_swap():
    @node("lines", title="Lines", description="")
    def lines(doc: str) -> Series[str]:
        return ["x", "x"]

    @node("approve", title="Approve", description="")
    def approve(line: str) -> bool:
        raise Ask(f"{line}?")

    graph = registry_of(_each, lines, approve).load(GraphRecord(
        nodes=[_EACH, NodeRecord(id="l", node="lines"), NodeRecord(id="a", node="approve")], edges=[("each.result", "l.doc"), ("l.result", "a.line")],
    ))
    run = graph.run(items=["d1", "d2"])
    answers = dict(zip([w.id for w in run.waiting], [True, False, False, True], strict=True))
    done = run.answer(answers).resume()

    swapped = done.again(items=["d2", "d1"])

    assert done.results["a"]["result"] == [[True, False], [False, True]]
    assert swapped.results["a"]["result"] == [[False, True], [True, False]] and swapped.waiting == []  # d2's lines, then d1's


def test_identical_items_share_an_earlier_result_and_a_third_takes_it_without_a_call():
    calls: list[str] = []

    @node("echo", title="Echo", description="")
    def echo(text: str) -> str:
        calls.append(text)
        return text.upper()

    graph = registry_of(_each, echo).load(GraphRecord(nodes=[_EACH, NodeRecord(id="e", node="echo")], edges=[("each.result", "e.text")]))
    run = graph.run(items=["a", "a"])
    calls.clear()

    assert run.again(items=["a", "a", "a"]).results["e"]["result"] == ["A", "A", "A"] and calls == []


def test_an_answer_survives_an_edit_and_its_revert_while_the_edited_item_still_waits_and_mails_once():
    """A answered; B changes Bo and waits; C reverts to A's input: Bo keeps A's answer and is not mailed twice.

    The node hadn't finished in B, so what it was carried and didn't take is still there.
    """
    sent: list[str] = []

    @node("approve", title="Approve", description="")
    def approve(name: str) -> str:
        raise Ask(f"Mail {name}?")

    @node("mail", title="Mail", description="")
    def mail(to: str) -> str:
        sent.append(to)
        return f"sent {to}"

    graph = registry_of(_each, approve, mail).load(GraphRecord(
        nodes=[_EACH, NodeRecord(id="a", node="approve"), NodeRecord(id="m", node="mail")], edges=[("each.result", "a.name"), ("a.result", "m.to")],
    ))
    first = graph.run(items=["ida", "bo"])
    a = first.answer({w.id: f"{w.inputs['name']}@a.dk" for w in first.waiting}).resume()
    b = a.again(items=["ida", "bob"])
    assert [w.inputs["name"] for w in b.waiting] == ["bob"]
    sent.clear()

    c = graph.load_run(b.to_json()).again(items=["ida", "bo"])

    assert c.status == "complete" and c.results["m"]["result"] == ["sent ida@a.dk", "sent bo@a.dk"] and sent == []


def test_a_second_edit_after_a_failed_leg_still_reuses_what_the_failed_leg_had_not_reached():
    calls: list[str] = []

    @node("echo", title="Echo", description="")
    def echo(text: str) -> str:
        calls.append(text)
        if text == "boom":
            raise ValueError("no")
        return text.upper()

    graph = registry_of(_each, echo).load(GraphRecord(nodes=[_EACH, NodeRecord(id="e", node="echo")], edges=[("each.result", "e.text")]))
    first = graph.run(items=["a", "b", "c"])
    failed = first.again(items=["boom", "b", "c"])
    assert failed.status == "failed"
    calls.clear()

    fixed = failed.again(items=["a", "b", "c"])

    assert fixed.status == "complete" and calls == [] and fixed.results["e"]["result"] == ["A", "B", "C"]


def test_work_a_finished_node_did_not_take_is_dropped_so_an_edit_undone_two_runs_later_calls_it_again():
    calls: list[str] = []

    @node("echo", title="Echo", description="")
    def echo(text: str) -> str:
        calls.append(text)
        return text.upper()

    graph = registry_of(_each, echo).load(GraphRecord(nodes=[_EACH, NodeRecord(id="e", node="echo")], edges=[("each.result", "e.text")]))
    second = graph.run(items=["a"]).again(items=["b"])  # echo finished on b: its unit for a is gone
    calls.clear()

    assert second.again(items=["a", "b"]).results["e"]["result"] == ["A", "B"] and calls == ["a"]


def test_a_waiting_item_goes_on_waiting_at_its_new_place_with_no_new_question():
    asked: list[str] = []

    @node("approve", title="Approve", description="")
    def approve(name: str) -> bool:
        asked.append(name)
        raise Ask(f"Mail {name}?")

    graph = registry_of(_each, approve).load(GraphRecord(nodes=[_EACH, NodeRecord(id="a", node="approve")], edges=[("each.result", "a.name")]))
    run = graph.run(items=["ida", "bo"])
    asked.clear()

    moved = run.again(items=["eva", "ida", "bo"])

    assert asked == ["eva"] and [(place(w), w.inputs["name"]) for w in moved.waiting] == [("a@0", "eva"), ("a@1", "ida"), ("a@2", "bo")]


def test_a_waiting_item_carried_through_a_cancelled_again_waits_at_its_new_place():
    asked: list[str] = []

    @node("approve", title="Approve", description="", policy=Policy(concurrency=2))
    def approve(name: str) -> bool:
        asked.append(name)
        raise Ask(f"Mail {name}?")

    graph = registry_of(_each, approve).load(GraphRecord(nodes=[_EACH, NodeRecord(id="a", node="approve")], edges=[("each.result", "a.name")]))
    run = graph.run(items=["ida", "bo"])
    cancel = asyncio.Event()
    cancel.set()
    stopped = _events(run.with_inputs(items=["eva", "ida", "bo"]), cancel=cancel)[-1].run
    asked.clear()

    resumed = graph.load_run(stopped.to_json()).resume()

    assert asked == ["eva"] and [w.inputs["name"] for w in resumed.waiting] == ["eva", "ida", "bo"]


def test_an_answer_for_a_place_that_now_holds_another_question_is_answer_stale_naming_the_current_one_and_where_its_own_went():
    @node("approve", title="Approve", description="")
    def approve(name: str) -> bool:
        raise Ask(f"Mail {name}?")

    graph = registry_of(_each, approve).load(GraphRecord(nodes=[_EACH, NodeRecord(id="a", node="approve")], edges=[("each.result", "a.name")]))
    first = graph.run(items=["ida", "bo"])
    (_, bo) = first.waiting
    moved = first.again(items=["ida", "al", "bo"])

    with pytest.raises(InvalidCallError) as refused:
        moved.answer(bo.id, True)  # bo's id names a@1, where al waits now
    (problem,) = refused.value.errors()
    current = moved.waiting[1].id
    moved_to = moved.waiting[2].id  # bo's question waits at a@2
    assert (problem.type, problem.loc, problem.ctx) == ("answer_stale", ("a", 1), {"waiting_id": bo.id, "current": current, "moved_to": moved_to})
    assert moved.answer("a@1", True).results["a"]["result"] == [None, True, None]  # a place alone answers what waits there


def test_a_changed_node_asks_again_with_its_new_prompt_and_its_old_unit_is_gone():
    @node("approve", title="Approve", description="")
    def approve(name: str) -> bool:
        raise Ask(f"Mail {name}?")

    @node("approve", title="Approve v2", description="", replaces=approve)
    def approve_v2(name: str) -> bool:
        raise Ask(f"Send to {name}?")

    record = GraphRecord(nodes=[NodeRecord(id="a", node="approve", version=1)])
    old = registry_of(approve_v2).load(record).run(name="ida")
    (waiting,) = old.waiting
    newer = registry_of(approve_v2).upgrade(record, "a")
    graph = registry_of(approve_v2).load(newer)
    run = graph.run(name="ida")

    assert [w.prompt for w in run.waiting] == ["Send to ida?"] and run.waiting[0].id != waiting.id


def test_a_node_redeclared_under_its_id_and_version_with_other_code_is_new_work_after_a_save():
    @node("echo", title="Echo", description="")
    def echo(text: str) -> str:
        return text.upper()

    @node("echo", title="Echo changed", description="")
    def echo_changed(text: str) -> str:
        return text.lower()

    record = GraphRecord(nodes=[NodeRecord(id="e", node="echo")])
    saved = registry_of(echo).load(record).run(text="Ab").to_json()

    again = registry_of(echo_changed).load(record).load_run(saved).again(text="Ab")

    assert again.results["e"]["result"] == "ab"


@pytest.mark.parametrize(("before", "after", "given"), [
    (list[int], list[str], []),
    (Literal["a", "b"], Literal["a"], "a"),
], ids=["list-item", "literal-values"])
def test_a_node_redeclared_under_its_id_and_version_with_the_same_code_and_another_type_is_new_work_after_a_save(before: Any, after: Any, given: Any):
    calls: list[Any] = []

    def made(value: Any) -> str:
        calls.append(value)
        return "made"

    def declared(input_type: Any) -> Any:
        made.__annotations__["value"] = input_type
        return node("made", title="Made", description="")(made)

    record = GraphRecord(nodes=[NodeRecord(id="m", node="made")])
    saved = registry_of(declared(before)).load(record).run(value=given).to_json()

    registry_of(declared(after)).load(record).load_run(saved).again(value=given)

    assert calls == [given, given]


def test_a_node_redeclared_to_take_a_blank_is_called_for_the_blank_item_it_was_not_called_for_before():
    class Row(TypedDict):
        name: str
        email: str | None

    @node("rows", title="Rows", description="")
    def rows(text: str) -> Table[Row]:
        return pl.DataFrame({"name": ["ida", "bo"], "email": ["i@a.dk", None]}, schema={"name": pl.String, "email": pl.String})

    @node("records", title="Records", description="")
    def records(table: Table[Row]) -> Series[Row]:
        return table.to_dicts()

    @node("send", title="Send", description="")
    def send(email: str) -> str:
        return f"to {email}"

    @node("send", title="Send blank", description="")
    def send_blank(email: str | None) -> str:
        return f"to {email}"

    record = GraphRecord(
        nodes=[NodeRecord(id="r", node="rows"), NodeRecord(id="p", node="records"), NodeRecord(id="s", node="send")],
        edges=[("r.result", "p.table"), ("p.result.email", "s.email")],
    )
    saved = registry_of(rows, records, send).load(record).run(text="x").to_json()

    again = registry_of(rows, records, send_blank).load(record).load_run(saved).again()

    assert again.results["s"]["result"] == ["to i@a.dk", "to None"]


def test_an_item_that_comes_back_before_an_identical_earlier_item_takes_its_earlier_work_once_that_item_has():
    calls: list[str] = []

    @node("a", title="A", description="", policy=Policy(concurrency=2))
    def a(text: str) -> Series[str]:
        return ["x"]

    @node("c", title="C", description="", policy=Policy(concurrency=2))
    def c(text: str) -> str:
        if text == "p":
            time.sleep(0.3)  # the second item's inputs are made first
        return "c"

    @node("b", title="B", description="")
    def b(x: str, y: str) -> str:
        calls.append(x + y)
        return x + y

    @node("split", title="Split", description="")
    def split(text: str) -> Series[str]:
        return text.split(",")

    graph = registry_of(split, a, c, b).load(GraphRecord(
        nodes=[NodeRecord(id=n, node=n) for n in ("split", "a", "c", "b")],
        edges=[("split.result", "a.text"), ("split.result", "c.text"), ("a.result", "b.x"), ("c.result", "b.y")],
    ))
    first = graph.run(text="p,q")
    calls.clear()

    again = first.forget("a", "c").resume()

    assert again.status == "complete" and again.results["b"] == {"result": [["xc"], ["xc"]]} and calls == []


def _counted() -> tuple[Any, list[str]]:
    calls: list[str] = []

    @node("fetch", title="Fetch", description="")
    def fetch(text: str) -> str:
        calls.append(f"fetch {text}")
        return text.upper()

    @node("show", title="Show", description="")
    def show(text: str) -> str:
        calls.append(f"show {text}")
        return f"<{text}>"

    graph = registry_of(_each, fetch, show).load(GraphRecord(
        nodes=[_EACH, NodeRecord(id="f", node="fetch"), NodeRecord(id="s", node="show")], edges=[("each.result", "f.text"), ("f.result", "s.text")],
    ))
    return graph, calls


def test_forget_runs_a_node_again_though_nothing_changed_and_a_reader_that_receives_the_same_is_reused():
    graph, calls = _counted()
    run = graph.run(items=["a", "b"])
    calls.clear()

    again = run.forget("f").resume()

    assert calls == ["fetch a", "fetch b"] and again.results == run.results
    assert run.forget().resume().results == run.results and calls == ["fetch a", "fetch b"]


def test_forget_is_not_started_survives_json_and_refuses_a_node_the_graph_lacks():
    graph, calls = _counted()
    run = graph.run(items=["a"])
    forgotten = run.forget("f")
    calls.clear()

    assert forgotten.status == "ready" and forgotten.results["f"] == {"result": None}
    assert graph.load_run(forgotten.to_json()).resume().status == "complete" and calls == ["fetch a"]
    with pytest.raises(InvalidCallError) as refused:
        run.forget("nope")
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("node_not_in_graph", ("nope",))]


def test_a_table_nothing_holds_after_forget_is_not_written():
    @node("split", title="Split", description="")
    def split(text: str) -> Series[str]:
        return text.split(",")

    graph = registry_of(split).load(GraphRecord(nodes=[NodeRecord(id="s", node="split")]))
    run = graph.run(text="a,b")

    written = json.loads(run.forget("s").to_json())

    assert "s" not in written["units"] and "s" not in written.get("earlier", {})


def test_identical_answered_items_keep_their_answers_across_a_cancelled_leg_as_on_a_direct_one():
    run = _answered(["bo", "bo", "al"], [True, False, True])
    direct = run.again(items=["eva", "bo", "bo", "al"])
    cancel = asyncio.Event()
    cancel.set()
    stopped = _events(run.with_inputs(items=["eva", "bo", "bo", "al"]), cancel=cancel)[-1].run

    through = run.graph.load_run(stopped.to_json()).resume()

    assert through.results == direct.results and direct.results["a"] == {"result": [None, True, False, True]}
    assert [w.inputs for w in through.waiting] == [w.inputs for w in direct.waiting] == [{"name": "eva"}]


def test_a_save_of_a_node_redeclared_so_it_runs_per_other_items_does_not_fit_and_is_refused_naming_the_node():
    @node("count", title="Count", description="")
    def count(text: str) -> int:
        return len(text)

    @node("count", title="Count all", description="")
    def count_all(text: Series[str]) -> int:
        return len(text)

    @node("split", title="Split", description="")
    def split(text: str) -> Series[str]:
        return text.split(",")

    record = GraphRecord(nodes=[NodeRecord(id="s", node="split"), NodeRecord(id="c", node="count")], edges=[("s.result", "c.text")])
    saved = registry_of(split, count).load(record).run(text="ab,c").to_json()

    with pytest.raises(InvalidCallError) as refused:
        registry_of(split, count_all).load(record).load_run(saved)

    assert [(p.type, p.ctx["at"]) for p in refused.value.errors()] == [("state_unreadable", "units.c")]


def test_identical_items_take_earlier_answers_in_item_order_whatever_thread_finished_first():
    """Item 1's upstream finishes before item 0's, so item 1 is tried first; it still takes the second earlier answer."""
    import threading

    first_done = threading.Event()

    @node("fetch", title="Fetch", description="", policy=Policy(concurrency=2, reuse_when_inputs_match=False))
    def fetch(name: str) -> str:
        if name == "bo-0":
            assert first_done.wait(timeout=10)  # item 0 comes back after item 1
        else:
            first_done.set()
        return "bo"

    @node("approve", title="Approve", description="")
    def approve(name: str) -> bool:
        raise Ask(f"Mail {name}?")

    graph = registry_of(_each, fetch, approve).load(GraphRecord(
        nodes=[_EACH, NodeRecord(id="f", node="fetch"), NodeRecord(id="a", node="approve")], edges=[("each.result", "f.name"), ("f.result", "a.name")],
    ))
    first = graph.run(items=["bo-0", "bo-1"])
    answered = first.answer({w.id: w.loc[1:] == (0,) for w in first.waiting}).resume()
    assert answered.results["a"]["result"] == [True, False]

    for _ in range(3):
        first_done.clear()
        again = answered.again(items=["bo-0", "bo-1"])
        assert again.results["a"]["result"] == [True, False] and again.waiting == []


def test_again_keeps_every_waiting_question_when_an_earlier_item_waits_on_a_person_above():
    """split → ask_a → ask_b; a@0 still waits, b@1 and b@2 wait: ``again`` with the same inputs keeps all three, calling nothing."""
    calls: list[str] = []

    @node("split", title="Split", description="")
    def split(text: str) -> Series[str]:
        return text.split()

    @node("ask_a", title="Ask a", description="")
    def ask_a(text: str) -> str:
        calls.append(f"a {text}")
        raise Ask(f"A {text}?")

    @node("ask_b", title="Ask b", description="")
    def ask_b(text: str) -> str:
        calls.append(f"b {text}")
        raise Ask(f"B {text}?")

    graph = registry_of(split, ask_a, ask_b).load(GraphRecord(
        nodes=[NodeRecord(id="s", node="split"), NodeRecord(id="a", node="ask_a"), NodeRecord(id="b", node="ask_b")],
        edges=[("s.result", "a.text"), ("a.result", "b.text")],
    ))
    first = graph.run(text="x y z")
    second = first.answer({"a@1": "Y", "a@2": "Z"}).resume()
    waiting = [w.id for w in second.waiting]
    assert [w.split("~")[0] for w in waiting] == ["a@0", "b@1", "b@2"]
    calls.clear()

    again = second.again()

    assert [w.id for w in again.waiting] == waiting and calls == []
    assert again.answer({waiting[1]: "y", waiting[2]: "z"}).results["b"]["result"] == [None, "y", "z"]


def test_an_inner_nodes_wrong_output_fails_where_the_node_is_inside_and_names_the_nodes_own_output():
    @node("a", title="A", description="")
    def a(x: str) -> int:
        return "not an int"  # type: ignore[return-value]

    @node("b", title="B", description="")
    def b(y: str) -> int:
        return 1

    box = node("emb", title="Emb", description="")(make_graph(a, b, edges=[]))
    graph = registry_of(a, b, box).load(make_graph(box))
    run = graph.run(x="q", y="w")

    assert graph.outputs == ["emb.a.result", "emb.b.result"]
    assert (run.status, run.state.cause.type, run.state.cause.loc, run.state.cause.ctx["name"]) == ("failed", "invalid_output", ("emb", "a"), "result")


def test_a_saved_run_stays_the_same_size_over_one_row_edits_at_ten_thousand_rows():
    """What a finished node was carried and didn't take is dropped, so ``again`` doesn't keep every generation."""

    class Line(TypedDict):
        name: str
        text: str

    @node("read_lines", title="Read lines", description="")
    def read_lines(csv: str) -> Table[Line]:
        return pl.read_csv(io.StringIO(csv))

    @node("records", title="Records", description="")
    def records(table: Table[Line]) -> Series[Line]:
        return table.to_dicts()

    @node("enrich", title="Enrich", description="")
    def enrich(name: str, text: str) -> str:
        return f"{name}: {text[:20]}"

    graph = registry_of(read_lines, records, enrich).load(GraphRecord(
        nodes=[NodeRecord(id="read", node="read_lines"), NodeRecord(id="records", node="records"), NodeRecord(id="enrich", node="enrich")],
        edges=[("read.result", "records.table"), ("records.result.name", "enrich.name"), ("records.result.text", "enrich.text")],
    ))
    csv = "name,text\n" + "".join(f"p{i},some text for row {i}\n" for i in range(10_000))
    run = graph.run(csv=csv)
    sizes = []
    for edit in range(5):
        run = graph.load_run(run.to_json()).again(csv=csv.replace("p5,", f"p5v{edit},", 1))
        sizes.append(len(run.to_json()))

    assert run.status == "complete" and max(sizes) - min(sizes) < 100, sizes


def test_an_edited_graph_takes_an_earlier_runs_work_for_the_nodes_the_edit_left_as_they_were():
    """The author changes the last node's prompt: its items run again, the 300 calls above it do not."""
    calls: list[str] = []

    @node("enrich", title="Enrich", description="")
    def enrich(text: str) -> str:
        calls.append(f"enrich {text}")
        return text.upper()

    @node("summarize", title="Summarize", description="")
    def summarize(text: str, prompt: str) -> str:
        calls.append(f"summarize {text}")
        return f"{prompt}: {text}"

    registry = registry_of(_each, enrich, summarize)

    def drawn(prompt: str) -> Any:
        return registry.load(GraphRecord(
            nodes=[_EACH, NodeRecord(id="e", node="enrich"), NodeRecord(id="s", node="summarize", values={"prompt": prompt})],
            edges=[("each.result", "e.text"), ("e.result", "s.text")],
        ))

    texts = [f"t{i}" for i in range(300)]
    first = drawn("Short").run(items=texts)
    calls.clear()
    edited = drawn("Long")
    assert edited.fingerprint != first.graph.fingerprint

    second = edited.with_inputs(items=texts, earlier=first.graph.load_run(first.to_json())).resume()

    assert second.results["s"]["result"][0] == "Long: T0" and calls == [f"summarize T{i}" for i in range(300)]
    calls.clear()
    assert edited.with_inputs(items=texts, earlier=second).resume().results == second.results and calls == []


def test_again_and_with_inputs_take_an_input_by_its_address_as_graph_run_does():
    @node("upper", title="Upper", description="")
    def upper(text: str) -> str:
        return text.upper()

    @node("join", title="Join", description="")
    def join(a: str, b: str) -> str:
        return a + b

    box = node("box", title="Box", description="")(GraphRecord(nodes=[NodeRecord(id="j", node="join")]))
    graph = registry_of(upper, join, box).load(GraphRecord(nodes=[NodeRecord(id="u", node="upper"), NodeRecord(id="e", node="box")]))
    first = graph.run({"u.text": "a", "e.a": "x", "e.b": "y"})

    assert first.again({"u.text": "b"}).outputs == {"u.result": "B", "e.j.result": "xy"}
    assert first.again({"e.a": "z"}).outputs == {"u.result": "A", "e.j.result": "zy"}
    assert first.with_inputs({"e.b": "w"}, text="c").resume().outputs == {"u.result": "C", "e.j.result": "xw"}
    with pytest.raises(InvalidCallError) as refused:
        first.again({"u.text": "b"}, text="c")
    assert [p.type for p in refused.value.errors()] == ["input_given_twice"]


def test_a_unit_made_under_another_limit_or_other_choices_is_never_taken():
    """What a unit's result rests on includes what each field's type refuses: an input whose limit or choices
    changed reads the same value another way, so earlier work on it is called again and the edge value is held to
    the limit the graph has."""
    made: list[int] = []

    @node("give", title="Give", description="")
    def give(n: int) -> int:
        return n

    def graph_taking(hint: object):
        @node("take", title="Take", description="")
        def take(n: hint) -> int:  # type: ignore[valid-type]
            made.append(n)
            return n

        record = GraphRecord(nodes=[NodeRecord(id="g", node="give"), NodeRecord(id="t", node="take")], edges=[("g.result", "t.n")])
        return registry_of(give, take).load(record)

    loose = graph_taking(Annotated[int, Param(ge=0)])
    first = loose.run(n=3)
    assert first.status == "complete" and made == [3]

    same = graph_taking(Annotated[int, Param(ge=0, title="N")]).with_inputs(n=3, earlier=first).resume()
    assert same.status == "complete" and made == [3]  # a title changes nothing that is read

    tight = graph_taking(Annotated[int, Param(ge=5)]).with_inputs(n=3, earlier=first).resume()
    assert tight.status == "failed" and tight.state.cause.type == "invalid_edge_value"

    chosen = graph_taking(Annotated[int, Param(choices=[1, 2])]).with_inputs(n=3, earlier=first).resume()
    assert chosen.status == "failed" and chosen.state.cause.type == "invalid_edge_value"
