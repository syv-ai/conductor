"""How a run fails, and what ``run.again`` reuses from the previous run.

A node's own failure is a problem on the run, never an exception out of it; only an outside service's failure is
retried, and a failure never quotes a value. ``again`` calls a node only for an item whose arguments differ from those
of the node's call in the previous run, so a CSV edited on one row sends one mail.
"""

import asyncio
import json
import logging
import time
from asyncio import Event as AsyncioEvent
from collections.abc import Mapping, Sequence
from io import StringIO
from random import Random
from typing import Annotated, Any, Literal, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Ask,
    ExternalError,
    FunctionDefinition,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeError,
    Param,
    Policy,
    Run,
    Series,
    Table,
    host_type,
    make_graph,
    node,
    subgraph,
)
from pydantic import BaseModel, ValidationError

from tests.test_core.nodes import place, registry_of


def _record(nodes: Mapping[str, FunctionDefinition], edges: Sequence[tuple[str, str]] = (), constants: Mapping[str, Any] | None = None) -> GraphRecord:
    """``make_graph`` of ``nodes`` by id, joined by exactly ``edges``, with ``constants`` by ``"node.input"``."""
    return make_graph(*nodes.items(), edges=list(edges), constants=constants or {})


def _alone(definition: Any, **values: Any) -> Any:
    """A graph of one node, ``n``, with these constants."""
    return Graph.compile(_record({"n": definition}, constants={f"n.{name}": value for name, value in values.items()}), registry_of(definition))


@node("each", title="Each", description="For a test.")
def _each(items: list[str]) -> Series[str]:
    """Returns the graph input list as a series, so a node wired to it is called once per item."""
    return items


def _events(run: Any, **kwargs: Any) -> list[Any]:
    async def collect() -> list[Any]:
        return [event async for event in run.stream(**kwargs)]

    return asyncio.run(collect())


# -- failures ---------------------------------------------------------------------


def test_a_code_fingerprint_is_the_same_on_every_compile_and_names_the_loop_the_node_is_inside():
    """A waiting call's id and a saved run contain the code fingerprint, so compiling the same record again gives
    the same one, and the same node inside the loop over another output gets another."""

    @node("words", title="Words", description="For a test.")
    def words(text: str) -> Series[str]:
        return text.split()

    @node("letters", title="Letters", description="For a test.")
    def letters(text: str) -> Series[str]:
        return list(text)

    @node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    def digest(first: Any) -> str:
        graph = Graph.compile(make_graph(first, upper), registry_of(first, upper))
        return json.loads(graph.run(text="a").to_json())["code"]["upper"]

    assert digest(words) == digest(words) != digest(letters)


def test_a_foreign_exception_runs_once_and_is_execution_failed_and_quotes_nothing(caplog):
    """``{}['missing']`` under ``Policy(retries=3)`` is a bug, not weather: one call, no retry, and the problem says only that the node failed."""
    calls: list[int] = []

    @node("buggy", title="Buggy", description="For a test.", policy=Policy(retries=3, retry_delay=0.01))
    def buggy(text: str = "") -> str:
        calls.append(1)
        return {}["the secret"]  # type: ignore[no-any-return]

    with caplog.at_level(logging.ERROR, logger="conductor"):
        events = _events(_alone(buggy).with_inputs())

    assert len(calls) == 1 and not [e for e in events if e.type == "node_retry"]
    (failed,) = [e for e in events if e.type == "node_failed"]
    assert (failed.cause.type, failed.cause.msg, failed.cause.loc, failed.cause.ctx) == ("execution_failed", "The node raised an exception; its developer finds it in the conductor log.", ("n",), {"error": "KeyError"})
    assert "secret" not in json.dumps([e.model_dump(mode="json") for e in events])
    assert "the secret" in caplog.text  # the text goes to the log, with its traceback
    assert events[-1].run.status == "failed" and events[-1].run.state.cause == failed.cause


@pytest.mark.parametrize("concurrency", [1, 2])
def test_a_node_error_a_wrong_output_and_an_ask_are_never_retried_whatever_retry_on_names(concurrency):
    calls: list[str] = []
    policy = Policy(retries=3, retry_delay=0.01, retry_on=(BaseException,), concurrency=concurrency)

    @node("own", title="Own", description="For a test.", policy=policy)
    def own(text: str = "") -> str:
        calls.append("own")
        raise NodeError("The document could not be read.", code="unreadable")

    @node("wrong", title="Wrong", description="For a test.", policy=policy)
    def wrong(text: str = "") -> int:
        calls.append("wrong")
        return "not a number"  # type: ignore[return-value]

    @node("asks", title="Asks", description="For a test.", policy=policy, asks=True)
    def asks(text: str = "") -> str:
        calls.append("asks")
        raise Ask("Which?")

    causes = [_alone(definition).run().state.cause for definition in (own, wrong)]

    assert [c.type for c in causes] == ["unreadable", "invalid_output"] and _alone(asks).run().status == "waiting"
    assert calls == ["own", "wrong", "asks"]


def test_a_foreign_exception_named_in_retry_on_is_external_and_retries():
    calls: list[int] = []

    @node("flaky", title="Flaky", description="For a test.", policy=Policy(retries=2, retry_delay=0.01, retry_on=(ConnectionError,)))
    def flaky(text: str = "") -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("socket closed")
        return "ok"

    events = _events(_alone(flaky).with_inputs())

    retries = [e for e in events if e.type == "node_retry"]
    assert len(calls) == 3 and [(e.attempt, e.retries, e.loc[0]) for e in retries] == [(1, 2, "n"), (2, 2, "n")]
    assert retries[0].cause.type == "external_failed" and retries[0].cause.msg == "An outside service did not answer."
    assert events[-1].run.outputs == {"result": "ok"}


def test_an_exhausted_external_failure_is_external_failed_without_its_text():
    calls: list[int] = []

    @node("down", title="Down", description="For a test.", policy=Policy(retries=2, retry_delay=0.01, retry_on=(ConnectionError,)))
    def down(text: str = "") -> str:
        calls.append(1)
        raise ConnectionError("socket closed")

    run = _alone(down).run()

    assert len(calls) == 3 and run.status == "failed"
    assert (run.state.cause.type, run.state.cause.msg) == ("external_failed", "An outside service did not answer.")


@node("double", title="Double", description="For a test.")
def _double(n: int) -> int:
    return 2 * n


def _runs_another_graph(given: Mapping[str, Any], policy: Policy) -> FunctionDefinition:
    """A node whose function runs a graph of ``double`` with ``given`` and lets its ``InvalidCallError`` escape."""
    inner = Graph.compile(make_graph(_double), registry_of(_double))

    @node("outer", title="Outer", description="For a test.", policy=policy)
    def outer(text: str = "") -> int:
        return inner.run(given).outputs["result"]  # type: ignore[no-any-return]

    return outer


@pytest.mark.parametrize("concurrency", [1, 2])
@pytest.mark.parametrize("given", [{"n": "many"}, {"n": "many", "colour": "red"}], ids=["one problem", "two problems"])
def test_an_invalid_call_error_the_node_lets_escape_is_its_own_failure_not_a_refusal(given, concurrency):
    """The node's body called ``graph.run`` wrongly: a bug in the node, ``execution_failed``, however many problems the error has."""
    run = _alone(_runs_another_graph(given, Policy(concurrency=concurrency))).run()

    assert (run.status, run.state.cause.type, run.state.cause.loc, run.state.cause.ctx) == ("failed", "execution_failed", ("n",), {"error": "InvalidCallError"})


def test_an_invalid_call_error_the_node_raises_is_retried_when_retry_on_names_value_error():
    """``InvalidCallError`` is a ``ValueError``; from the node's own body it is retried like any exception ``retry_on`` names."""
    calls: list[int] = []
    outer = _runs_another_graph({"n": "many"}, Policy())

    @node("counted", title="Counted", description="For a test.", policy=Policy(retries=2, retry_delay=0.01, retry_on=(ValueError,)))
    def counted(text: str = "") -> int:
        calls.append(1)
        return outer.function(text)  # type: ignore[no-any-return]

    run = _alone(counted).run()

    assert len(calls) == 3 and (run.status, run.state.cause.type) == ("failed", "external_failed")


def test_an_external_failure_the_node_raised_retries_and_keeps_its_message():
    """A node that knows the outside world failed says so with ``ExternalError``; the message it wrote is what people read."""
    calls: list[int] = []

    @node("bank", title="Bank", description="For a test.", policy=Policy(retries=1, retry_delay=0.01))
    def bank(text: str = "") -> str:
        calls.append(1)
        raise ExternalError("Nationalbanken svarede ikke.")

    run = _alone(bank).run()

    assert len(calls) == 2
    assert (run.state.cause.type, run.state.cause.msg) == ("external_failed", "Nationalbanken svarede ikke.")


def test_a_node_error_runs_once_and_keeps_its_message_code_and_ctx():
    calls: list[int] = []

    @node("strict", title="Strict", description="For a test.", policy=Policy(retries=3, retry_delay=0.01))
    def strict(text: str = "") -> str:
        calls.append(1)
        raise NodeError("Dokumentet kunne ikke læses.", code="unreadable", ctx={"pages": 0})

    events = _events(_alone(strict).with_inputs())

    (failed,) = [e for e in events if e.type == "node_failed"]
    assert len(calls) == 1
    assert failed.cause.model_dump(mode="json") == {"type": "unreadable", "loc": ["n"], "msg": "Dokumentet kunne ikke læses.", "ctx": {"pages": 0}}


@pytest.mark.parametrize("ctx", [{"handle": object()}, {"pages": {1, 2}}, {"pages": (1, 2)}])
def test_a_node_error_whose_ctx_has_no_json_form_is_refused_where_the_node_raises_it_so_the_run_still_saves(ctx: dict[str, Any]) -> None:
    @node("read", title="Read", description="For a test.")
    def read(path: str = "") -> str:
        raise NodeError("The document could not be read.", code="unreadable", ctx=ctx)

    with pytest.raises(ValidationError):
        NodeError("The document could not be read.", ctx=ctx)
    with pytest.raises(ValidationError):
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

    @node("parse", title="Parse", description="For a test.", policy=Policy(retries=3, retry_delay=0.01))
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
        with pytest.raises(ValidationError):
            Policy(**wrong)
    assert Policy(retry_delay=2, attempt_timeout=5).attempt_timeout == 5  # a whole number of seconds is a number
    assert "retry_on" not in Policy(retry_on=(ConnectionError,)).model_dump()


def test_a_failed_item_is_retried_alone():
    """A node that runs per item retries the item that failed; the items that completed are not run again."""
    calls: list[str] = []

    @node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @node("flaky-on-b", title="Flaky on b", description="For a test.", policy=Policy(retries=2, retry_delay=0.01))
    def flaky_on_b(text: str) -> str:
        calls.append(text)
        if text == "b" and calls.count("b") == 1:
            raise ExternalError("transient")
        return text.upper()

    graph = Graph.compile(make_graph(split, flaky_on_b, constants={"split.text": "a,b,c"}), registry_of(split, flaky_on_b))

    assert graph.run().results["flaky-on-b"]["result"] == ["A", "B", "C"]
    assert sorted(calls) == ["a", "b", "b", "c"]


def test_a_flaky_node_in_one_branch_retries_while_the_other_branch_completes():
    calls = {"a": 0, "b": 0}

    @node("flaky-a", title="Flaky a", description="For a test.", policy=Policy(retries=2, retry_delay=0.01))
    def flaky_a(text: str) -> str:
        calls["a"] += 1
        if calls["a"] == 1:
            raise ExternalError("first try")
        return f"A:{text}"

    @node("fast-b", title="Fast b", description="For a test.")
    def fast_b(text: str) -> str:
        calls["b"] += 1
        return f"B:{text}"

    @node("join-two", title="Join two", description="For a test.")
    def join_two(a: str, b: str) -> str:
        return f"{a}+{b}"

    record = _record({"n1": flaky_a, "n2": fast_b, "n3": join_two}, [("n1.result", "n3.a"), ("n2.result", "n3.b")], constants={"n1.text": "x", "n2.text": "y"})
    graph = Graph.compile(record, registry_of(flaky_a, fast_b, join_two))

    assert graph.run().outputs == {"result": "A:x+B:y"}
    assert calls == {"a": 2, "b": 1}


def test_a_run_that_failed_resumes_from_its_json_calling_only_what_had_not_finished():
    calls: list[str] = []
    broken = {"on": True}

    @node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @node("mail", title="Mail", description="For a test.")
    def mail(to: str) -> str:
        calls.append(to)
        if to == "b" and broken["on"]:
            raise ValueError("smtp down")
        return f"sent {to}"

    graph = Graph.compile(make_graph(split, mail, constants={"split.text": "a,b,c"}), registry_of(split, mail))
    failed = graph.run()
    broken["on"] = False
    resumed = Run.from_json(failed.to_json(), graph).resume()

    assert failed.status == "failed" and failed.state.cause.loc == ("mail", 1)
    assert resumed.results["mail"]["result"] == ["sent a", "sent b", "sent c"]
    assert calls.count("a") == 1 and calls.count("b") == 2


def test_a_cancelled_run_keeps_the_calls_it_finished_and_resume_calls_none_of_them_again():
    calls: list[str] = []

    @node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @node("count-calls", title="Count calls", description="For a test.")
    def count_calls(text: str) -> str:
        calls.append(text)
        return text

    graph = Graph.compile(make_graph(split, count_calls, constants={"split.text": "a,b,c,d"}), registry_of(split, count_calls))

    async def cancel_after_two() -> Any:
        cancel = AsyncioEvent()
        last = None
        async for event in graph.with_inputs().stream(cancel=cancel):
            if len(calls) >= 2:
                cancel.set()
            last = event
        return last.run  # type: ignore[union-attr]

    stopped = asyncio.run(cancel_after_two())
    finished = Run.from_json(stopped.to_json(), graph).resume()

    assert finished.results["count-calls"]["result"] == ["a", "b", "c", "d"]
    assert sorted(calls) == ["a", "b", "c", "d"]


# -- running again ----------------------------------------------------------------


def _mails() -> tuple[Any, list[str]]:
    """A graph that mails each address of a list, and the calls it made."""
    sent: list[str] = []

    @node("mail", title="Mail", description="For a test.")
    def mail(to: str) -> str:
        sent.append(to)
        return f"sent {to}"

    @node("confirm", title="Confirm", description="For a test.")
    def confirm(receipt: str) -> str:
        sent.append(f"confirm {receipt}")
        return receipt.upper()

    graph = Graph.compile(_record({"each": _each, "mail": mail, "confirm": confirm}, [("each.result", "mail.to"), ("mail.result", "confirm.receipt")]), registry_of(_each, mail, confirm))
    return graph, sent


def test_a_changed_item_is_called_again_at_every_downstream_node_and_nothing_else_is():
    graph, sent = _mails()
    first = graph.run(items=["ida", "bo", "al"])
    sent.clear()

    again = first.again(items=["ida", "bob", "al"])

    assert sent == ["bob", "confirm sent bob"]
    assert again.results["confirm"]["result"] == ["SENT IDA", "SENT BOB", "SENT AL"]


def test_an_item_inserted_first_or_a_duplicate_taken_out_reuses_the_other_items():
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

    completed = {e.loc[0]: e.reused for e in events if e.type == "node_complete"}
    assert sent == ["al", "confirm sent al"] and completed == {"each": 0, "mail": 2, "confirm": 2}
    assert events[-1].run.results["confirm"]["result"] == ["SENT IDA", "SENT BO", "SENT AL"]


def test_a_series_input_reordered_is_called_again_and_one_unchanged_is_reused():
    calls: list[list[str]] = []

    @node("join-texts", title="Join texts", description="For a test.")
    def join_texts(texts: Series[str]) -> str:
        calls.append(list(texts))
        return "+".join(texts)

    graph = Graph.compile(_record({"j": join_texts}), registry_of(join_texts))
    first = graph.run(texts=["a", "b"])

    first.again(texts=["a", "b"])
    first.again(texts=["b", "a"])

    assert calls == [["a", "b"], ["b", "a"]]


def test_a_dict_given_again_in_another_key_order_is_reused():
    calls: list[Any] = []

    @node("take-mapping", title="Take mapping", description="For a test.")
    def take_mapping(value: dict[str, Any] | None) -> int:
        calls.append(value)
        return 1

    graph = Graph.compile(_record({"t": take_mapping}), registry_of(take_mapping))
    first = Run.from_json(graph.run(value={"a": 1, "b": 2}).to_json(), graph)
    first.again(value={"b": 2, "a": 1})

    assert calls == [{"a": 1, "b": 2}]


def test_values_that_compare_equal_but_differ_in_type_are_never_reused_for_each_other():
    calls: list[Any] = []

    @node("echo-any", title="Echo any", description="For a test.")
    def echo_any(value: Any) -> str:
        calls.append(value)
        return repr(value)

    graph = Graph.compile(_record({"e": echo_any}), registry_of(echo_any))
    run = graph.run(value=1)
    for value in (1, "1", True, 1.0, [1], {"a": 1}):
        run = run.again(value=value)

    assert calls == [1, "1", True, 1.0, [1], {"a": 1}]


def test_a_value_with_no_json_form_is_called_again_and_never_fails_the_run():
    calls: list[Any] = []

    @host_type("Opaque", title="Opaque")
    class Opaque:
        """A host's value pydantic carries but has no JSON form for."""

        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
            from pydantic_core import core_schema

            return core_schema.is_instance_schema(cls)

    @node("make", title="Make", description="For a test.")
    def make(text: str) -> Opaque:
        return Opaque()

    @node("use", title="Use", description="For a test.")
    def use(thing: Opaque) -> str:
        calls.append(thing)
        return "used"

    graph = Graph.compile(make_graph(make, use), registry_of(make, use))
    first = graph.run(text="a")
    again = first.again(text="a")

    assert again.status == "complete" and again.outputs == {"result": "used"}
    assert len(calls) == 2


def test_an_answer_stays_with_its_item_when_an_item_is_inserted_first_and_an_edited_item_asks_again():
    @node("approve", title="Approve", description="For a test.", asks=True)
    def approve(name: str) -> bool:
        raise Ask(f"Approve {name}?")

    graph = Graph.compile(_record({"each": _each, "a": approve}, [("each.result", "a.name")]), registry_of(_each, approve))
    run = graph.run(items=["ida", "bo"])
    run = run.answer("a@0", True).answer("a@1", False).resume()

    moved = run.again(items=["eva", "ida", "bo"])
    edited = run.again(items=["ida", "bob"])

    assert [place(w) for w in moved.waiting] == ["a@0"] and moved.results["a"]["result"] == [None, True, False]
    assert [w.inputs["name"] for w in edited.waiting] == ["bob"] and edited.results["a"]["result"] == [True, None]


def test_a_table_output_reads_back_by_its_own_column_types_whatever_graphs_came_before():
    """Each output checks its own columns: a graph compiled and freed before never passes its column types to another."""
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

    @node("mk", title="Mk", description="For a test.", compute_outputs=outs)
    def mk(kind: str) -> dict[str, Any]:
        return {"result": pl.DataFrame({"v": [date(2020, 1, 1)] if kind == "date" else ["2020-01-01"]})}

    registry = registry_of(mk)
    for i in range(60):
        kind = "date" if i % 2 == 0 else "str"
        graph = Graph.compile(GraphRecord.model_validate({"nodes": {"m": {"node": "mk", "inputs": {"kind": {"constant": kind}}}}}), registry)
        value = Run.from_json(graph.run().to_json(), graph).results["m"]["result"]["v"][0]
        assert isinstance(value, date) == (kind == "date"), (i, value)
        del graph
        gc.collect()


# -- what again reuses, item by item -----------------------------------------------------------


@node("docs", title="Docs", description="For a test.")
def _docs(text: str) -> Series[str]:
    return text.split("|") if text else []


class _Line(TypedDict):
    word: str


@node("line-records", title="Line records", description="For a test.")
def _line_records(table: Table[_Line]) -> Series[_Line]:
    return table.to_dicts()


def test_a_document_inserted_first_is_the_only_one_read_and_the_others_lines_move_with_them():
    read: list[str] = []
    upper_calls: list[str] = []

    @node("lines", title="Lines", description="For a test.")
    def lines(text: str) -> Table[_Line]:
        read.append(text)
        return pl.DataFrame({"word": text.split(",")})

    @node("shout", title="Shout", description="For a test.")
    def shout(word: str) -> str:
        upper_calls.append(word)
        return word.upper()

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"d": {"node": "docs", "inputs": {"text": {"graph_input": "text"}}}, "l": {"node": "lines", "inputs": {"text": {"sources": [{"node": "d", "output": "result"}]}}}, "r": {"node": "line-records", "inputs": {"table": {"sources": [{"node": "l", "output": "result"}]}}}, "s": {"node": "shout", "inputs": {"word": {"sources": [{"node": "r", "output": "result", "field": "word"}]}}}}}), registry_of(_docs, lines, _line_records, shout))
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

    @node("read-rows", title="Read rows", description="For a test.")
    def read_rows(text: str) -> Table[Row]:
        return pl.read_csv(StringIO(text))

    @node("threshold", title="Threshold", description="For a test.")
    def threshold(min: int = threshold_default) -> int:
        return min

    @node("older-than", title="Older than", description="For a test.")
    def older_than(rows: pl.DataFrame, min: int) -> Table[Row]:
        filtered.append("called")
        return rows.filter(pl.col("age") > min)

    @node("names", title="Names", description="For a test.")
    def names(rows: pl.DataFrame) -> str:
        counted.append("called")
        return ",".join(rows["name"])

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"r": {"node": "read-rows", "inputs": {"text": {"graph_input": "text"}}}, "t": {"node": "threshold", "inputs": {"min": {"graph_input": "min", "default": threshold_default}}}, "f": {"node": "older-than", "inputs": {"rows": {"sources": [{"node": "r", "output": "result"}]}, "min": {"sources": [{"node": "t", "output": "result"}]}}}, "n": {"node": "names", "inputs": {"rows": {"sources": [{"node": "f", "output": "result"}]}}}}, "outputs": {"result": {"node": "n", "output": "result"}}}), registry_of(read_rows, threshold, older_than, names))
    return graph, filtered, counted


def test_a_row_the_filter_drops_inserted_first_leaves_the_reader_of_the_filtered_table_uncalled():
    graph, filtered, counted = _filtered()
    first = graph.run(text="name,age\nida,30\nbo,40\n")
    filtered.clear()
    counted.clear()

    again = first.again(text="name,age\neva,1\nida,30\nbo,40\n")

    assert filtered == ["called"] and counted == []
    assert again.outputs == {"result": "ida,bo"}


def test_equal_tables_from_two_filter_settings_reuse_the_downstream_reader():
    graph, filtered, counted = _filtered()
    first = graph.run(text="name,age\nida,30\nbo,40\n", min=2)
    counted.clear()

    again = first.again(min=10)

    assert filtered == ["called", "called"] and counted == []
    assert again.outputs == {"result": "ida,bo"}


def test_a_reused_series_and_items_called_afresh_line_up_item_by_item_after_a_document_is_inserted_first():
    @node("words", title="Words", description="For a test.")
    def words(text: str) -> Series[str]:
        return text.split(",")

    @node("stamp", title="Stamp", description="For a test.", policy=Policy(reuse_when_inputs_match=False))
    def stamp(word: str) -> str:
        return f"<{word}>"

    @node("show", title="Show", description="For a test.")
    def show(words: Series[str], stamps: Series[str]) -> str:
        return ";".join(f"{w}={s}" for w, s in zip(words, stamps, strict=True))

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"d": {"node": "docs", "inputs": {"text": {"graph_input": "text"}}}, "w": {"node": "words", "inputs": {"text": {"sources": [{"node": "d", "output": "result"}]}}}, "t": {"node": "stamp", "inputs": {"word": {"sources": [{"node": "w", "output": "result"}]}}}, "s": {"node": "show", "inputs": {"words": {"sources": [{"node": "w", "output": "result"}]}, "stamps": {"sources": [{"node": "t", "output": "result"}]}}}}}), registry_of(_docs, words, stamp, show))
    first = graph.run(text="a,b|c")

    again = first.again(text="x,y,z|a,b|c")

    assert first.results["s"]["result"] == ["a=<a>;b=<b>", "c=<c>"]
    assert again.results["s"]["result"] == ["x=<x>;y=<y>;z=<z>", "a=<a>;b=<b>", "c=<c>"]


@pytest.mark.parametrize("rows", ["a,b", "a,b,c,d", "x,y,z"], ids=["fewer", "more", "same-count"])
def test_again_with_changed_items_upstream_of_a_subgraph_gives_its_inner_constant_node_to_every_item(rows):
    @node("label", title="Label", description="For a test.")
    def label(prefix: str) -> str:
        return prefix + ":"

    @node("pair", title="Pair", description="For a test.")
    def pair(text: str, tag: str) -> str:
        return tag + text

    @node("same", title="Same", description="For a test.")
    def same(text: str) -> str:
        return text

    inner = subgraph("tagged", title="Tagged", description="For a test.")(GraphRecord.model_validate({"nodes": {"same": {"node": "same", "inputs": {"text": {"graph_input": "text"}}}, "label": {"node": "label", "inputs": {"prefix": {"constant": "p"}}}, "pair": {"node": "pair", "inputs": {"text": {"sources": [{"node": "same", "output": "result"}]}, "tag": {"sources": [{"node": "label", "output": "result"}]}}}}, "outputs": {"result": {"node": "pair", "output": "result"}}}))
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"d": {"node": "docs", "inputs": {"text": {"graph_input": "text"}}}, "t": {"node": "tagged", "inputs": {"text": {"sources": [{"node": "d", "output": "result"}]}}}}}), registry_of(_docs, same, label, pair, inner))
    first = graph.run(text="a|b|c")

    again = first.again(text=rows.replace(",", "|"))
    direct = graph.run(text=rows.replace(",", "|"))

    assert again.status == "complete" and again.results == direct.results
    assert again.results["t"]["result"] == [f"p:{r}" for r in rows.split(",")]


def test_an_input_given_again_overrides_the_first_value_and_the_previous_run_is_unchanged():
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

    @node("read-people", title="Read people", description="For a test.")
    def read_people(text: str) -> Table[Person]:
        return pl.read_csv(StringIO(text))

    @node("people", title="People", description="For a test.")
    def people(table: Table[Person]) -> Series[Person]:
        return table.to_dicts()

    @node("body", title="Body text", description="For a test.")
    def body_text(body: str = "hej") -> str:
        return body

    @node("send-body", title="Send body", description="For a test.")
    def send_body(email: str, body: str) -> str:
        sent.append(email)
        return f"{body} {email}"

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"r": {"node": "read-people", "inputs": {"text": {"graph_input": "text"}}}, "p": {"node": "people", "inputs": {"table": {"sources": [{"node": "r", "output": "result"}]}}}, "b": {"node": "body", "inputs": {"body": {"graph_input": "body", "default": "hej"}}}, "m": {"node": "send-body", "inputs": {"email": {"sources": [{"node": "p", "output": "result", "field": "email"}]}, "body": {"sources": [{"node": "b", "output": "result"}]}}}}}), registry_of(read_people, people, body_text, send_body))
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


def test_a_document_inserted_first_in_a_nested_loop_calls_only_its_own_items():
    calls: list[str] = []

    @node("words", title="Words", description="For a test.")
    def words(text: str) -> Series[str]:
        return text.split(",")

    @node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        calls.append(text)
        return text.upper()

    graph = Graph.compile(make_graph(_docs, words, shout), registry_of(_docs, words, shout))
    first = graph.run(text="a,b|c")
    calls.clear()

    again = first.again(text="x,y|a,b|c")

    assert calls == ["x", "y"]
    assert again.results["shout"]["result"] == [["X", "Y"], ["A", "B"], ["C"]]


def _approvals() -> Any:
    @node("approve", title="Approve", description="For a test.", asks=True)
    def approve(name: str) -> bool:
        raise Ask(f"Approve {name}?")

    return Graph.compile(_record({"each": _each, "a": approve}, [("each.result", "a.name")]), registry_of(_each, approve))


def test_each_item_still_waiting_after_again_is_announced_on_the_stream():
    run = _approvals().run(items=["ida", "bo"])

    events = _events(run.with_inputs(items=["eva", "ida", "bo"]))

    assert [place(e.waiting) for e in events if e.type == "node_waiting"] == ["a@0", "a@1", "a@2"]
    assert events[-1].status == "waiting"


def test_an_again_that_fails_before_it_reaches_the_old_items_still_reuses_them_on_resume():
    broken = {"on": True}
    sent: list[str] = []

    @node("mail-or-fail", title="Mail or fail", description="For a test.")
    def mail_or_fail(to: str) -> str:
        sent.append(to)
        if to == "eva" and broken["on"]:
            raise ValueError("smtp down")
        return f"sent {to}"

    graph = Graph.compile(_record({"each": _each, "m": mail_or_fail}, [("each.result", "m.to")]), registry_of(_each, mail_or_fail))
    first = graph.run(items=["ida", "bo"])
    sent.clear()

    failed = first.again(items=["eva", "ida", "bo"])
    assert failed.status == "failed" and sent == ["eva"]
    broken["on"] = False
    resumed = Run.from_json(failed.to_json(), graph).resume()

    assert sent == ["eva", "eva"] and resumed.results["m"]["result"] == ["sent eva", "sent ida", "sent bo"]


def test_an_again_cancelled_before_the_node_runs_then_resumed_sends_only_the_new_mail():
    graph, sent = _mails()
    first = graph.run(items=["ida", "bo"])
    sent.clear()
    cancel = AsyncioEvent()
    cancel.set()

    stopped = _events(first.with_inputs(items=["ida", "bo", "al"]), cancel=cancel)[-1].run
    assert stopped.status == "cancelled"
    resumed = Run.from_json(stopped.to_json(), graph).resume()

    assert sent == ["al", "confirm sent al"] and resumed.results["confirm"]["result"] == ["SENT IDA", "SENT BO", "SENT AL"]


def test_after_an_answered_question_again_with_one_item_added_sends_one_mail():
    sent: list[str] = []

    @node("approve", title="Approve", description="For a test.", asks=True)
    def approve(name: str) -> str:
        raise Ask(f"Mail {name}?")

    @node("mail", title="Mail", description="For a test.")
    def mail(to: str) -> str:
        sent.append(to)
        return f"sent {to}"

    graph = Graph.compile(_record({"each": _each, "a": approve, "m": mail}, [("each.result", "a.name"), ("a.result", "m.to")]), registry_of(_each, approve, mail))
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


def test_again_cancelled_and_resumed_reuses_what_again_run_to_the_end_reuses():
    rng = Random(7)
    graph, sent = _mails()
    names = ["ida", "bo", "al", "eva", "kim"]
    items = ["ida", "bo"]
    run = graph.run(items=items)
    cancel = AsyncioEvent()
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
        through = Run.from_json(stopped.to_json(), graph).resume()
        assert sorted(sent) == direct_calls and through.results == direct.results
        run, items = direct, edited


def test_a_call_that_returned_skipped_is_an_outcome_again_reuses_live_and_after_from_json():
    """A node that returned ``SKIPPED`` was called: ``again`` reuses that outcome like any other, and the node
    downstream of it is skipped again."""
    calls: list[str] = []

    @node("only-long", title="Only long", description="For a test.")
    def only_long(text: str) -> str:
        calls.append(text)
        return text if len(text) > 2 else SKIPPED

    @node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    graph = Graph.compile(_record({"each": _each, "o": only_long, "s": shout}, [("each.result", "o.text"), ("o.result", "s.text")]), registry_of(_each, only_long, shout))
    first = graph.run(items=["a", "abc", "b"])
    assert first.results["o"] == {"result": [None, "abc", None]} and first.results["s"] == {"result": [None, "ABC", None]}
    assert calls == ["a", "abc", "b"]

    events = _events(first.with_inputs())
    again = events[-1].run
    read_back_again = Run.from_json(first.to_json(), graph).again()

    assert calls == ["a", "abc", "b"]
    assert [(e.outcomes, e.reused) for e in events if e.type == "node_complete" and e.loc == ("o",)] == [(3, 3)]
    assert again.results == first.results == read_back_again.results and first.not_called == []
    nothing_long = _events(graph.with_inputs(items=["a", "b"]))
    assert [e.loc for e in nothing_long if e.type == "node_skipped"] == [("o",), ("s",)]  # every call ended with no value, whether the node was called or not


def test_a_first_run_reports_no_reuse():
    graph, _ = _mails()

    events = _events(graph.with_inputs(items=["ida", "bo"]))

    assert {e.loc[0]: e.reused for e in events if e.type == "node_complete"} == {"each": 0, "mail": 0, "confirm": 0}


def test_a_run_with_changed_inputs_not_yet_started_shows_no_results():
    graph, sent = _mails()
    first = graph.run(items=["ida"])
    sent.clear()

    prepared = first.with_inputs(items=["ida", "bo"])

    assert prepared.status == "ready" and sent == []
    assert prepared.results == {"each": {"result": None}, "mail": {"result": None}, "confirm": {"result": None}}  # no series returned yet, so no results


def test_an_input_is_compared_for_reuse_as_its_type_reads_it_so_2_and_2_0_are_one_float():
    calls: list[float] = []

    @node("half", title="Half", description="For a test.")
    def half(x: float) -> float:
        calls.append(x)
        return x / 2

    graph = Graph.compile(_record({"h": half}), registry_of(half))
    first = graph.run(x=2)

    again = first.again(x=2.0)

    assert calls == [2.0] and again.outputs == {"result": 1.0}


def test_a_table_column_its_row_type_doesnt_name_reads_back_from_json_as_written():
    import datetime as dt
    import math

    import polars as pl
    from conductor import Table

    @node("mixed", title="Mixed", description="For a test.")
    def mixed(text: str) -> Table:
        return pl.DataFrame({"when": [dt.date(2026, 1, 1)], "at": [dt.time(1, 2)], "x": [math.nan], "n": [1]})

    graph = Graph.compile(_record({"m": mixed}), registry_of(mixed))
    frame = Run.from_json(graph.run(text="a").to_json(), graph).results["m"]["result"]

    assert frame.schema == pl.Schema({"when": pl.Date, "at": pl.Time, "x": pl.Float64, "n": pl.Int64})
    assert math.isnan(frame["x"][0])


def _alone_node() -> Any:
    @node("n", title="N", description="For a test.")
    def n(text: str = "") -> str:
        return text

    return n


def test_waiting_items_and_a_saved_run_keep_node_then_item_order_whatever_threads_did():
    import random

    @node("slow-ask", title="Slow ask", description="For a test.", policy=Policy(concurrency=4), asks=True)
    def slow_ask(text: str) -> str:
        import time

        time.sleep(random.random() * 0.02)
        raise Ask(f"{text}?")

    @node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split(",")

    graph = Graph.compile(make_graph(split, slow_ask), registry_of(split, slow_ask))
    for _ in range(3):
        run = graph.run(text=",".join("abcdefgh"))
        assert [place(w) for w in run.waiting] == [f"slow-ask@{i}" for i in range(8)]
        assert [question["prompt"] for question in json.loads(run.to_json())["outcomes"]["slow-ask"]["questions"].values()] == [f"{letter}?" for letter in "abcdefgh"]


def test_a_saved_run_that_does_not_fit_is_refused_naming_where_and_never_with_a_raw_error():
    @node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        return text

    graph = Graph.compile(make_graph(split, echo), registry_of(split, echo))
    saved = json.loads(graph.run(text="a,b,c").to_json())
    for tamper, at in (
        (lambda s: s["outcomes"]["echo"]["states"].append(None), "outcomes.echo"),
        (lambda s: s["outcomes"]["echo"]["outputs"]["result"].pop(), "outcomes.echo"),
        (lambda s: s["outcomes"]["echo"]["outputs"].pop("result"), "outcomes.echo.outputs.result"),
        (lambda s: s["outcomes"]["echo"]["outputs"]["result"].__setitem__(0, 5), "outcomes.echo.outputs.result.0"),
    ):
        copy = json.loads(json.dumps(saved))
        tamper(copy)
        with pytest.raises(InvalidCallError) as refused:
            Run.from_json(json.dumps(copy), graph)
        (problem,) = refused.value.errors()
        assert (problem.type, problem.ctx["where"]) == ("saved_run_unreadable", at), problem


# -- identical items, generations, waiting items and forgetting --------------------------------


def _answered(names: list[str], answers: list[bool]) -> Any:
    @node("approve", title="Approve", description="For a test.", asks=True)
    def approve(name: str) -> bool:
        raise Ask(f"Mail {name}?")

    graph = Graph.compile(_record({"each": _each, "a": approve}, [("each.result", "a.name")]), registry_of(_each, approve))
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

    moved = Run.from_json(run.to_json(), run.graph).again(items=["al", "bo", "bo"])

    assert moved.results["a"]["result"] == [True, True, False] and moved.waiting == []


def test_a_third_identical_item_is_asked_and_two_previous_answers_cover_only_two_items():
    run = _answered(["bo", "bo"], [True, False])

    more = run.again(items=["bo", "bo", "bo"])

    assert [place(w) for w in more.waiting] == ["a@2"] and more.results["a"]["result"] == [True, False, None]


def test_identical_lines_answered_differently_under_documents_keep_their_documents_when_the_documents_swap():
    @node("lines", title="Lines", description="For a test.")
    def lines(doc: str) -> Series[str]:
        return ["x", "x"]

    @node("approve", title="Approve", description="For a test.", asks=True)
    def approve(line: str) -> bool:
        raise Ask(f"{line}?")

    graph = Graph.compile(_record({"each": _each, "l": lines, "a": approve}, [("each.result", "l.doc"), ("l.result", "a.line")]), registry_of(_each, lines, approve))
    run = graph.run(items=["d1", "d2"])
    answers = dict(zip([w.id for w in run.waiting], [True, False, False, True], strict=True))
    done = run.answer(answers).resume()

    swapped = done.again(items=["d2", "d1"])

    assert done.results["a"]["result"] == [[True, False], [False, True]]
    assert swapped.results["a"]["result"] == [[False, True], [True, False]] and swapped.waiting == []  # d2's lines, then d1's


def test_identical_items_share_one_outcome_and_a_third_reuses_it_without_a_call():
    calls: list[str] = []

    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        calls.append(text)
        return text.upper()

    graph = Graph.compile(_record({"each": _each, "e": echo}, [("each.result", "e.text")]), registry_of(_each, echo))
    run = graph.run(items=["a", "a"])
    calls.clear()

    assert run.again(items=["a", "a", "a"]).results["e"]["result"] == ["A", "A", "A"] and calls == []


def test_an_answer_survives_an_edit_and_its_revert_while_the_edited_item_still_waits_and_mails_once():
    """A answered; B changes Bo and waits; C reverts to A's input: Bo keeps A's answer and is not mailed twice.

    The node was not complete in B, so the reusable outcome from A that B did not use is still there for C.
    """
    sent: list[str] = []

    @node("approve", title="Approve", description="For a test.", asks=True)
    def approve(name: str) -> str:
        raise Ask(f"Mail {name}?")

    @node("mail", title="Mail", description="For a test.")
    def mail(to: str) -> str:
        sent.append(to)
        return f"sent {to}"

    graph = Graph.compile(_record({"each": _each, "a": approve, "m": mail}, [("each.result", "a.name"), ("a.result", "m.to")]), registry_of(_each, approve, mail))
    first = graph.run(items=["ida", "bo"])
    a = first.answer({w.id: f"{w.inputs['name']}@a.dk" for w in first.waiting}).resume()
    b = a.again(items=["ida", "bob"])
    assert [w.inputs["name"] for w in b.waiting] == ["bob"]
    sent.clear()

    c = Run.from_json(b.to_json(), graph).again(items=["ida", "bo"])

    assert c.status == "complete" and c.results["m"]["result"] == ["sent ida@a.dk", "sent bo@a.dk"] and sent == []


def test_an_edit_after_a_failed_run_still_reuses_the_outcomes_the_failed_run_had_not_reached():
    calls: list[str] = []

    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        calls.append(text)
        if text == "boom":
            raise ValueError("no")
        return text.upper()

    graph = Graph.compile(_record({"each": _each, "e": echo}, [("each.result", "e.text")]), registry_of(_each, echo))
    first = graph.run(items=["a", "b", "c"])
    failed = first.again(items=["boom", "b", "c"])
    assert failed.status == "failed"
    calls.clear()

    fixed = failed.again(items=["a", "b", "c"])

    assert fixed.status == "complete" and calls == [] and fixed.results["e"]["result"] == ["A", "B", "C"]


def test_an_outcome_a_complete_node_did_not_reuse_is_dropped_so_an_edit_undone_two_runs_later_calls_again():
    calls: list[str] = []

    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        calls.append(text)
        return text.upper()

    graph = Graph.compile(_record({"each": _each, "e": echo}, [("each.result", "e.text")]), registry_of(_each, echo))
    second = graph.run(items=["a"]).again(items=["b"])  # echo is complete on b, so its outcome for a is dropped
    calls.clear()

    assert second.again(items=["a", "b"]).results["e"]["result"] == ["A", "B"] and calls == ["a"]


def test_a_waiting_call_moves_with_its_item_and_is_not_asked_again():
    asked: list[str] = []

    @node("approve", title="Approve", description="For a test.", asks=True)
    def approve(name: str) -> bool:
        asked.append(name)
        raise Ask(f"Mail {name}?")

    graph = Graph.compile(_record({"each": _each, "a": approve}, [("each.result", "a.name")]), registry_of(_each, approve))
    run = graph.run(items=["ida", "bo"])
    asked.clear()

    moved = run.again(items=["eva", "ida", "bo"])

    assert asked == ["eva"] and [(place(w), w.inputs["name"]) for w in moved.waiting] == [("a@0", "eva"), ("a@1", "ida"), ("a@2", "bo")]


def test_a_waiting_call_moves_with_its_item_through_a_cancelled_again():
    asked: list[str] = []

    @node("approve", title="Approve", description="For a test.", policy=Policy(concurrency=2), asks=True)
    def approve(name: str) -> bool:
        asked.append(name)
        raise Ask(f"Mail {name}?")

    graph = Graph.compile(_record({"each": _each, "a": approve}, [("each.result", "a.name")]), registry_of(_each, approve))
    run = graph.run(items=["ida", "bo"])
    cancel = AsyncioEvent()
    cancel.set()
    stopped = _events(run.with_inputs(items=["eva", "ida", "bo"]), cancel=cancel)[-1].run
    asked.clear()

    resumed = Run.from_json(stopped.to_json(), graph).resume()

    assert asked == ["eva"] and [w.inputs["name"] for w in resumed.waiting] == ["eva", "ida", "bo"]


def test_an_answer_to_a_call_whose_item_moved_is_answer_stale_naming_the_question_there_and_where_its_own_went():
    @node("approve", title="Approve", description="For a test.", asks=True)
    def approve(name: str) -> bool:
        raise Ask(f"Mail {name}?")

    graph = Graph.compile(_record({"each": _each, "a": approve}, [("each.result", "a.name")]), registry_of(_each, approve))
    first = graph.run(items=["ida", "bo"])
    (_, bo) = first.waiting
    moved = first.again(items=["ida", "al", "bo"])

    with pytest.raises(InvalidCallError) as refused:
        moved.answer(bo.id, True)  # bo's id names a@1, where al's question waits
    (problem,) = refused.value.errors()
    current = moved.waiting[1].id
    moved_to = moved.waiting[2].id  # bo's question waits at a@2
    assert (problem.type, problem.loc, problem.ctx) == ("answer_stale", ("a", 1), {"waiting_id": bo.id, "current": current, "moved_to": moved_to})
    assert moved.answer("a@1", True).results["a"]["result"] == [None, True, None]  # an id without its question's fingerprint answers whatever waits at that call


def test_an_upgraded_node_asks_again_with_its_new_prompt_under_another_waiting_id():
    @node("approve", title="Approve", description="For a test.", asks=True)
    def approve(name: str) -> bool:
        raise Ask(f"Mail {name}?")

    @node("approve", title="Approve v2", description="For a test.", replaces=approve, asks=True)
    def approve_v2(name: str) -> bool:
        raise Ask(f"Send to {name}?")

    record = GraphRecord.model_validate({"nodes": {"a": {"node": "approve", "version": 1, "inputs": {"name": {"graph_input": "name"}}}}})
    old = Graph.compile(record, registry_of(approve_v2)).run(name="ida")
    (waiting,) = old.waiting
    newer = registry_of(approve_v2).upgrade(record, "a")
    graph = Graph.compile(newer, registry_of(approve_v2))
    run = graph.run(name="ida")

    assert [w.prompt for w in run.waiting] == ["Send to ida?"] and run.waiting[0].id != waiting.id


def test_a_node_redeclared_under_its_id_and_version_with_other_code_is_called_again_on_a_saved_run():
    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        return text.upper()

    @node("echo", title="Echo changed", description="For a test.")
    def echo_changed(text: str) -> str:
        return text.lower()

    record = GraphRecord.model_validate({"nodes": {"e": {"node": "echo", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "e", "output": "result"}}})
    saved = Graph.compile(record, registry_of(echo)).run(text="Ab").to_json()

    again = Run.from_json(saved, Graph.compile(record, registry_of(echo_changed))).again(text="Ab")

    assert again.results["e"]["result"] == "ab"


#: One node, ``ask``, written four ways. Each is compiled from its own text, so the functions inside have the same
#: qualified names: those are constants of the node's code, and they would differ between four functions of this file.
_ASK = {
    "without docstrings": '''
def ask(name: str) -> bool:
    def polite(text: str) -> str:
        return text.title()

    class Note:
        pass

    raise Ask(f"Mail {polite(name)}?")
''',
    "with docstrings": '''
def ask(name: str) -> bool:
    """Asks whether to mail the person."""
    def polite(text: str) -> str:
        """Returns the text in title case."""
        return text.title()

    class Note:
        """A note."""

    raise Ask(f"Mail {polite(name)}?")
''',
    "with other docstrings": '''
def ask(name: str) -> bool:
    """Asks a person whether a mail goes to ``name``."""
    def polite(text: str) -> str:
        """``"ida"`` gives ``"Ida"``."""
        return text.title()

    class Note:
        """Another note."""

    raise Ask(f"Mail {polite(name)}?")
''',
    "with another message": '''
def ask(name: str) -> bool:
    """Asks whether to mail the person."""
    def polite(text: str) -> str:
        """Returns the text in title case."""
        return text.title()

    class Note:
        """A note."""

    raise Ask(f"Send to {polite(name)}?")
''',
}


def _waiting_id(source: str) -> str:
    """Returns the id of the call that the node ``ask`` written as ``source`` leaves waiting for ``name="ida"``."""
    namespace: dict[str, Any] = {"Ask": Ask}
    exec(source, namespace)
    function = namespace["ask"]
    record = GraphRecord.model_validate({"nodes": {"a": {"node": "ask", "inputs": {"name": {"graph_input": "name"}}}}})
    (waiting,) = Graph.compile(record, registry_of(node("ask", title="Ask", description="For a test.", asks=True)(function))).run(name="ida").waiting
    return waiting.id


def test_a_docstring_of_a_node_or_of_a_function_or_class_inside_it_is_not_part_of_its_code():
    """Adding or changing a docstring keeps a waiting call's id; another message gives another id."""
    ids = {way: _waiting_id(source) for way, source in _ASK.items()}

    assert ids["without docstrings"] == ids["with docstrings"] == ids["with other docstrings"]
    assert ids["with another message"] != ids["with docstrings"]


def test_a_node_redeclared_with_another_docstring_is_reused_on_a_saved_run_and_with_another_constant_is_not():
    calls: list[str] = []

    @node("echo", title="Echo", description="For a test.")
    def echo(text: str) -> str:
        """Returns the text in capitals."""
        calls.append("echo")
        return text.upper() + "!"

    @node("echo", title="Echo", description="For a test.")
    def echo_documented(text: str) -> str:
        """``"ab"`` gives ``"AB!"``."""
        calls.append("echo")
        return text.upper() + "!"

    @node("echo", title="Echo", description="For a test.")
    def echo_other_mark(text: str) -> str:
        """Returns the text in capitals."""
        calls.append("echo")
        return text.upper() + "?"

    record = GraphRecord.model_validate({"nodes": {"e": {"node": "echo", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "e", "output": "result"}}})
    saved = Graph.compile(record, registry_of(echo)).run(text="ab").to_json()
    Run.from_json(saved, Graph.compile(record, registry_of(echo_documented))).again(text="ab")
    assert calls == ["echo"]

    again = Run.from_json(saved, Graph.compile(record, registry_of(echo_other_mark))).again(text="ab")
    assert calls == ["echo", "echo"] and again.results["e"]["result"] == "AB?"


@pytest.mark.parametrize(("before", "after", "given"), [
    (list[int], list[str], []),
    (Literal["a", "b"], Literal["a"], "a"),
], ids=["list-item", "literal-values"])
def test_a_node_redeclared_with_the_same_code_and_another_input_type_is_called_again_on_a_saved_run(before: Any, after: Any, given: Any):
    calls: list[Any] = []

    def take(value: Any) -> str:
        calls.append(value)
        return "take"

    def declared(input_type: Any) -> Any:
        take.__annotations__["value"] = input_type
        return node("take", title="Take", description="For a test.")(take)

    record = GraphRecord.model_validate({"nodes": {"m": {"node": "take", "inputs": {"value": {"graph_input": "value"}}}}})
    saved = Graph.compile(record, registry_of(declared(before))).run(value=given).to_json()

    Run.from_json(saved, Graph.compile(record, registry_of(declared(after)))).again(value=given)

    assert calls == [given, given]


def test_a_node_redeclared_to_take_none_is_called_for_the_item_it_was_not_called_for():
    class Row(TypedDict):
        name: str
        email: str | None

    @node("rows", title="Rows", description="For a test.")
    def rows(text: str) -> Table[Row]:
        return pl.DataFrame({"name": ["ida", "bo"], "email": ["i@a.dk", None]}, schema={"name": pl.String, "email": pl.String})

    @node("records", title="Records", description="For a test.")
    def records(table: Table[Row]) -> Series[Row]:
        return table.to_dicts()

    @node("send", title="Send", description="For a test.")
    def send(email: str) -> str:
        return f"to {email}"

    @node("send", title="Send or none", description="For a test.")
    def send_or_none(email: str | None) -> str:
        return f"to {email}"

    record = GraphRecord.model_validate({"nodes": {"r": {"node": "rows", "inputs": {"text": {"graph_input": "text"}}}, "p": {"node": "records", "inputs": {"table": {"sources": [{"node": "r", "output": "result"}]}}}, "s": {"node": "send", "inputs": {"email": {"sources": [{"node": "p", "output": "result", "field": "email"}]}}}}})
    saved = Graph.compile(record, registry_of(rows, records, send)).run(text="x").to_json()

    again = Run.from_json(saved, Graph.compile(record, registry_of(rows, records, send_or_none))).again()

    assert again.results["s"]["result"] == ["to i@a.dk", "to None"]


def test_identical_items_finishing_out_of_order_after_forget_both_reuse_the_previous_outcome():
    calls: list[str] = []

    @node("a", title="A", description="For a test.", policy=Policy(concurrency=2))
    def a(text: str) -> Series[str]:
        return ["x"]

    @node("c", title="C", description="For a test.", policy=Policy(concurrency=2))
    def c(text: str) -> str:
        if text == "p":
            time.sleep(0.3)  # so the second item's arguments are ready first
        return "c"

    @node("b", title="B", description="For a test.")
    def b(x: str, y: str) -> str:
        calls.append(x + y)
        return x + y

    @node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split(",")

    graph = Graph.compile(make_graph(
        split, a, c, b, edges=[("split.result", "a.text"), ("split.result", "c.text"), ("a.result", "b.x"), ("c.result", "b.y")],
    ), registry_of(split, a, c, b))
    first = graph.run(text="p,q")
    calls.clear()

    again = first.forget("a", "c").resume()

    assert again.status == "complete" and again.results["b"] == {"result": [["xc"], ["xc"]]} and calls == []


def _counted() -> tuple[Any, list[str]]:
    calls: list[str] = []

    @node("fetch", title="Fetch", description="For a test.")
    def fetch(text: str) -> str:
        calls.append(f"fetch {text}")
        return text.upper()

    @node("show", title="Show", description="For a test.")
    def show(text: str) -> str:
        calls.append(f"show {text}")
        return f"<{text}>"

    graph = Graph.compile(_record({"each": _each, "f": fetch, "s": show}, [("each.result", "f.text"), ("f.result", "s.text")]), registry_of(_each, fetch, show))
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
    assert Run.from_json(forgotten.to_json(), graph).resume().status == "complete" and calls == ["fetch a"]
    with pytest.raises(InvalidCallError) as refused:
        run.forget("nope")
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("node_not_in_graph", ("nope",))]


def test_a_forgotten_nodes_table_is_not_written_in_the_saved_run():
    @node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split(",")

    graph = Graph.compile(_record({"s": split}), registry_of(split))
    run = graph.run(text="a,b")

    written = json.loads(run.forget("s").to_json())

    assert "s" not in written["outcomes"] and "s" not in written.get("reusable", {})


def test_identical_answered_items_keep_their_answers_across_a_cancelled_again_as_across_one_to_the_end():
    run = _answered(["bo", "bo", "al"], [True, False, True])
    direct = run.again(items=["eva", "bo", "bo", "al"])
    cancel = AsyncioEvent()
    cancel.set()
    stopped = _events(run.with_inputs(items=["eva", "bo", "bo", "al"]), cancel=cancel)[-1].run

    through = Run.from_json(stopped.to_json(), run.graph).resume()

    assert through.results == direct.results and direct.results["a"] == {"result": [None, True, False, True]}
    assert [w.inputs for w in through.waiting] == [w.inputs for w in direct.waiting] == [{"name": "eva"}]


def test_a_saved_run_of_a_node_redeclared_to_collect_its_loop_is_refused_naming_the_node():
    @node("count", title="Count", description="For a test.")
    def count(text: str) -> int:
        return len(text)

    @node("count", title="Count all", description="For a test.")
    def count_all(text: Series[str]) -> int:
        return len(text)

    @node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split(",")

    record = GraphRecord.model_validate({"nodes": {"s": {"node": "split", "inputs": {"text": {"graph_input": "text"}}}, "c": {"node": "count", "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}}}}})
    saved = Graph.compile(record, registry_of(split, count)).run(text="ab,c").to_json()

    with pytest.raises(InvalidCallError) as refused:
        Run.from_json(saved, Graph.compile(record, registry_of(split, count_all)))

    assert [(p.type, p.ctx["where"]) for p in refused.value.errors()] == [("saved_run_unreadable", "outcomes.c")]


def test_identical_items_reuse_previous_answers_in_item_order_whatever_thread_finished_first():
    """Item 1's upstream call finishes before item 0's, so item 1 is tried first; it still reuses the second answer."""
    from threading import Event as ThreadEvent

    first_done = ThreadEvent()

    @node("fetch", title="Fetch", description="For a test.", policy=Policy(concurrency=2, reuse_when_inputs_match=False))
    def fetch(name: str) -> str:
        if name == "bo-0":
            assert first_done.wait(timeout=10)  # item 0 comes back after item 1
        else:
            first_done.set()
        return "bo"

    @node("approve", title="Approve", description="For a test.", asks=True)
    def approve(name: str) -> bool:
        raise Ask(f"Mail {name}?")

    graph = Graph.compile(_record({"each": _each, "f": fetch, "a": approve}, [("each.result", "f.name"), ("f.result", "a.name")]), registry_of(_each, fetch, approve))
    first = graph.run(items=["bo-0", "bo-1"])
    answered = first.answer({w.id: w.loc[1:] == (0,) for w in first.waiting}).resume()
    assert answered.results["a"]["result"] == [True, False]

    for _ in range(3):
        first_done.clear()
        again = answered.again(items=["bo-0", "bo-1"])
        assert again.results["a"]["result"] == [True, False] and again.waiting == []


def test_again_with_the_same_inputs_keeps_the_waiting_questions_of_two_nodes_and_calls_nothing():
    """split -> ask_a -> ask_b; a@0 waits, b@1 and b@2 wait: ``again`` with the same inputs keeps all three and calls nothing."""
    calls: list[str] = []

    @node("split", title="Split", description="For a test.")
    def split(text: str) -> Series[str]:
        return text.split()

    @node("ask_a", title="Ask a", description="For a test.", asks=True)
    def ask_a(text: str) -> str:
        calls.append(f"a {text}")
        raise Ask(f"A {text}?")

    @node("ask_b", title="Ask b", description="For a test.", asks=True)
    def ask_b(text: str) -> str:
        calls.append(f"b {text}")
        raise Ask(f"B {text}?")

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "split", "inputs": {"text": {"graph_input": "text"}}}, "a": {"node": "ask_a", "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}}}, "b": {"node": "ask_b", "inputs": {"text": {"sources": [{"node": "a", "output": "result"}]}}}}}), registry_of(split, ask_a, ask_b))
    first = graph.run(text="x y z")
    second = first.answer({"a@1": "Y", "a@2": "Z"}).resume()
    waiting = [w.id for w in second.waiting]
    assert [w.split("~")[0] for w in waiting] == ["a@0", "b@1", "b@2"]
    calls.clear()

    again = second.again()

    assert [w.id for w in again.waiting] == waiting and calls == []
    assert again.answer({waiting[1]: "y", waiting[2]: "z"}).results["b"]["result"] == [None, "y", "z"]


def test_an_inner_nodes_wrong_output_fails_where_the_node_is_inside_and_names_the_nodes_own_output():
    @node("a", title="A", description="For a test.")
    def a(x: str) -> int:
        return "not an int"  # type: ignore[return-value]

    @node("b", title="B", description="For a test.")
    def b(y: str) -> int:
        return 1

    box = subgraph("sub", title="Sub", description="For a test.")(make_graph(a, b, edges=[], outputs={"a": "a.result", "b": "b.result"}))
    graph = Graph.compile(make_graph(box), registry_of(a, b, box))
    run = graph.run(x="q", y="w")

    assert graph.outputs == ["a", "b"]
    assert (run.status, run.state.cause.type, run.state.cause.loc, run.state.cause.ctx["name"]) == ("failed", "invalid_output", ("sub", "a"), "result")


def test_a_saved_run_stays_the_same_size_over_one_row_edits_at_ten_thousand_rows():
    """A complete node drops the reusable outcomes it did not reuse, so the saved run does not keep every previous run's."""

    class Line(TypedDict):
        name: str
        text: str

    @node("read_lines", title="Read lines", description="For a test.")
    def read_lines(csv: str) -> Table[Line]:
        return pl.read_csv(StringIO(csv))

    @node("records", title="Records", description="For a test.")
    def records(table: Table[Line]) -> Series[Line]:
        return table.to_dicts()

    @node("enrich", title="Enrich", description="For a test.")
    def enrich(name: str, text: str) -> str:
        return f"{name}: {text[:20]}"

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"read": {"node": "read_lines", "inputs": {"csv": {"graph_input": "csv"}}}, "records": {"node": "records", "inputs": {"table": {"sources": [{"node": "read", "output": "result"}]}}}, "enrich": {"node": "enrich", "inputs": {"name": {"sources": [{"node": "records", "output": "result", "field": "name"}]}, "text": {"sources": [{"node": "records", "output": "result", "field": "text"}]}}}}}), registry_of(read_lines, records, enrich))
    csv = "name,text\n" + "".join(f"p{i},some text for row {i}\n" for i in range(10_000))
    run = graph.run(csv=csv)
    sizes = []
    for edit in range(5):
        run = Run.from_json(run.to_json(), graph).again(csv=csv.replace("p5,", f"p5v{edit},", 1))
        sizes.append(len(run.to_json()))

    assert run.status == "complete" and max(sizes) - min(sizes) < 100, sizes


def test_an_edited_graph_reuses_the_previous_runs_outcomes_of_the_nodes_the_edit_left_unchanged():
    """The author changes the last node's prompt: its 300 calls run again, the 300 calls upstream of it do not."""
    calls: list[str] = []

    @node("enrich", title="Enrich", description="For a test.")
    def enrich(text: str) -> str:
        calls.append(f"enrich {text}")
        return text.upper()

    @node("summarize", title="Summarize", description="For a test.")
    def summarize(text: str, prompt: str) -> str:
        calls.append(f"summarize {text}")
        return f"{prompt}: {text}"

    registry = registry_of(_each, enrich, summarize)

    def drawn(prompt: str) -> Any:
        return Graph.compile(_record(
            {"each": _each, "e": enrich, "s": summarize}, [("each.result", "e.text"), ("e.result", "s.text")], constants={"s.prompt": prompt},
        ), registry)

    texts = [f"t{i}" for i in range(300)]
    first = drawn("Short").run(items=texts)
    calls.clear()
    edited = drawn("Long")
    assert edited.fingerprint != first.graph.fingerprint

    second = edited.with_inputs(items=texts, previous=Run.from_json(first.to_json(), first.graph)).resume()

    assert second.results["s"]["result"][0] == "Long: T0" and calls == [f"summarize T{i}" for i in range(300)]
    calls.clear()
    assert edited.with_inputs(items=texts, previous=second).resume().results == second.results and calls == []


def test_again_and_with_inputs_take_inputs_by_name_in_a_mapping_as_graph_run_does():
    @node("upper", title="Upper", description="For a test.")
    def upper(text: str) -> str:
        return text.upper()

    @node("join", title="Join", description="For a test.")
    def join(a: str, b: str) -> str:
        return a + b

    box = subgraph("box", title="Box", description="For a test.")(make_graph(join))
    graph = Graph.compile(GraphRecord.model_validate({
        "nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
                  "e": {"node": "box", "inputs": {"a": {"graph_input": "first-half"}, "b": {"graph_input": "second-half"}}}},
        "outputs": {"upper": {"node": "u", "output": "result"}, "joined": {"node": "e", "output": "result"}},
    }), registry_of(upper, join, box))
    first = graph.run({"text": "a", "first-half": "x", "second-half": "y"})

    assert first.again({"text": "b"}).outputs == {"upper": "B", "joined": "xy"}
    assert first.again({"first-half": "z"}).outputs == {"upper": "A", "joined": "zy"}
    assert first.with_inputs({"second-half": "w"}, text="c").resume().outputs == {"upper": "C", "joined": "xw"}


def test_an_outcome_called_under_another_limit_or_other_choices_is_never_reused():
    """Reuse compares input types too: an input whose limit or choices changed may refuse the same value, so the node
    is called again and the value on the edge is checked against the limit of the graph that runs."""
    called: list[int] = []

    @node("give", title="Give", description="For a test.")
    def give(n: int) -> int:
        return n

    def graph_taking(hint: object):
        @node("take", title="Take", description="For a test.")
        def take(n: hint) -> int:  # type: ignore[valid-type]
            called.append(n)
            return n

        record = GraphRecord.model_validate({"nodes": {"g": {"node": "give", "inputs": {"n": {"graph_input": "n"}}}, "t": {"node": "take", "inputs": {"n": {"sources": [{"node": "g", "output": "result"}]}}}}})
        return Graph.compile(record, registry_of(give, take))

    loose = graph_taking(Annotated[int, Param(ge=0)])
    first = loose.run(n=3)
    assert first.status == "complete" and called == [3]

    same = graph_taking(Annotated[int, Param(ge=0, title="N")]).with_inputs(n=3, previous=first).resume()
    assert same.status == "complete" and called == [3]  # a title changes nothing that is read

    tight = graph_taking(Annotated[int, Param(ge=5)]).with_inputs(n=3, previous=first).resume()
    assert tight.status == "failed" and tight.state.cause.type == "invalid_edge_value"

    chosen = graph_taking(Annotated[int, Param(choices=[1, 2])]).with_inputs(n=3, previous=first).resume()
    assert chosen.status == "failed" and chosen.state.cause.type == "invalid_edge_value"
