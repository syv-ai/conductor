"""How a run fails and what ``run.again`` reuses: the behaviours a host's side effects rest on.

A node's own failure is a ``Problem`` on the run, never an exception out of
it; only the outside world's failure is retried, and a failure never quotes a
value. ``again`` calls a node only on a row whose inputs differ from what an
earlier unit of that node received, so a CSV edited on one row sends one mail.
"""

import asyncio
import io
import json
import logging
import random
from typing import Any, TypedDict

import polars as pl
import pytest
from conductor import (
    Ask,
    ExternalFailure,
    FromRun,
    GraphRecord,
    NodeError,
    NodeRecord,
    Policy,
    Series,
    Table,
    make_graph,
    node,
)
from pydantic import BaseModel

from tests.test_core.nodes import registry_of


def _alone(definition: Any, **values: Any) -> Any:
    """A graph of one node, ``n``, with these values typed in."""
    return registry_of(definition).load(GraphRecord(nodes=[NodeRecord(id="n", node=definition.id, values=values)]))


def _events(run: Any, **kwargs: Any) -> list[Any]:
    async def collect() -> list[Any]:
        return [event async for event in run.stream(**kwargs)]

    return asyncio.run(collect())


# -- failures ---------------------------------------------------------------------


def test_a_foreign_exception_runs_once_and_is_execution_failed_and_quotes_nothing(caplog):
    """``{}['missing']`` under ``Policy(retries=3)`` is a bug, not weather: one call, no retry, and the problem says only that the node failed."""
    calls: list[int] = []

    @node("buggy", policy=Policy(retries=3, retry_delay=0.01))
    def buggy(text: str = "") -> str:
        calls.append(1)
        return {}["the secret"]  # type: ignore[no-any-return]

    with caplog.at_level(logging.ERROR, logger="conductor"):
        events = _events(_alone(buggy).with_inputs())

    assert len(calls) == 1 and not [e for e in events if e.type == "node_retry"]
    (failed,) = [e for e in events if e.type == "node_failed"]
    assert (failed.cause.type, failed.cause.msg, failed.cause.loc, failed.cause.ctx) == ("execution_failed", "The node failed.", ("n",), {"error": "KeyError"})
    assert "secret" not in json.dumps([e.model_dump(mode="json") for e in events])
    assert "the secret" in caplog.text  # the text goes to the log, with its traceback
    assert events[-1].run.status == "failed" and events[-1].run.cause == failed.cause


def test_a_foreign_exception_named_in_retry_on_is_external_and_retries():
    calls: list[int] = []

    @node("flaky", policy=Policy(retries=2, retry_delay=0.01, retry_on=(ConnectionError,)))
    def flaky(text: str = "") -> str:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("socket closed")
        return "ok"

    events = _events(_alone(flaky).with_inputs())

    retries = [e for e in events if e.type == "node_retry"]
    assert len(calls) == 3 and [(e.attempt, e.retries, e.node_id) for e in retries] == [(1, 2, "n"), (2, 2, "n")]
    assert retries[0].cause.type == "external_failed" and retries[0].cause.msg == "An outside service did not answer."
    assert events[-1].run.outputs == {"n.result": "ok"}


def test_an_exhausted_external_failure_is_external_failed_without_its_text():
    calls: list[int] = []

    @node("down", policy=Policy(retries=2, retry_delay=0.01, retry_on=(ConnectionError,)))
    def down(text: str = "") -> str:
        calls.append(1)
        raise ConnectionError("socket closed")

    run = _alone(down).run()

    assert len(calls) == 3 and run.status == "failed"
    assert (run.cause.type, run.cause.msg) == ("external_failed", "An outside service did not answer.")


def test_an_external_failure_the_node_raised_retries_and_keeps_its_message():
    """A node that knows the outside world failed says so with ``ExternalFailure``; the message it wrote is what people read."""
    calls: list[int] = []

    @node("bank", policy=Policy(retries=1, retry_delay=0.01))
    def bank(text: str = "") -> str:
        calls.append(1)
        raise ExternalFailure("Nationalbanken svarede ikke.")

    run = _alone(bank).run()

    assert len(calls) == 2
    assert (run.cause.type, run.cause.msg) == ("external_failed", "Nationalbanken svarede ikke.")


def test_a_node_error_runs_once_and_keeps_its_message_code_and_ctx():
    calls: list[int] = []

    @node("strict", policy=Policy(retries=3, retry_delay=0.01))
    def strict(text: str = "") -> str:
        calls.append(1)
        raise NodeError("Dokumentet kunne ikke læses.", code="unreadable", ctx={"pages": 0})

    events = _events(_alone(strict).with_inputs())

    (failed,) = [e for e in events if e.type == "node_failed"]
    assert len(calls) == 1
    assert failed.cause.model_dump(mode="json") == {"type": "unreadable", "loc": ["n"], "msg": "Dokumentet kunne ikke læses.", "ctx": {"pages": 0}}


def test_a_validation_error_the_node_raised_is_never_retried():
    calls: list[int] = []

    class Strict(BaseModel):
        n: int

    @node("parse", policy=Policy(retries=3, retry_delay=0.01))
    def parse(text: str = "") -> int:
        calls.append(1)
        return Strict.model_validate({"n": text}).n

    run = _alone(parse, text="not a number").run()

    assert len(calls) == 1 and (run.cause.type, run.cause.ctx) == ("execution_failed", {"error": "ValidationError"})


def test_a_policy_out_of_bounds_is_refused_and_its_description_leaves_retry_on_out():
    for wrong in ({"concurrency": 0}, {"retries": -1}, {"retry_delay": -1}, {"attempt_timeout": 0}):
        with pytest.raises(ValueError):
            Policy(**wrong)
    assert "retry_on" not in Policy(retry_on=(ConnectionError,)).describe()


def test_a_failed_row_is_retried_alone():
    """A node that runs per row retries the row that failed; the rows that completed are not run again."""
    calls: list[str] = []

    @node("split")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @node("flaky-on-b", policy=Policy(retries=2, retry_delay=0.01))
    def flaky_on_b(text: str) -> str:
        calls.append(text)
        if text == "b" and calls.count("b") == 1:
            raise ExternalFailure("transient")
        return text.upper()

    graph = registry_of(split, flaky_on_b).load(make_graph(split, flaky_on_b, values={"split.text": "a,b,c"}))

    assert graph.run().results["flaky-on-b"]["result"] == ["A", "B", "C"]
    assert sorted(calls) == ["a", "b", "b", "c"]


def test_a_flaky_node_in_one_branch_retries_while_the_other_branch_completes():
    calls = {"a": 0, "b": 0}

    @node("flaky-a", policy=Policy(retries=2, retry_delay=0.01))
    def flaky_a(text: str) -> str:
        calls["a"] += 1
        if calls["a"] == 1:
            raise ExternalFailure("first try")
        return f"A:{text}"

    @node("fast-b")
    def fast_b(text: str) -> str:
        calls["b"] += 1
        return f"B:{text}"

    @node("join-two")
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

    @node("split")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @node("mail")
    def mail(to: str) -> str:
        calls.append(to)
        if to == "b" and broken["on"]:
            raise ValueError("smtp down")
        return f"sent {to}"

    graph = registry_of(split, mail).load(make_graph(split, mail, values={"split.text": "a,b,c"}))
    failed = graph.run()
    broken["on"] = False
    resumed = graph.load_run(failed.to_json()).resume()

    assert failed.status == "failed" and failed.cause.loc == ("mail", 1)
    assert resumed.results["mail"]["result"] == ["sent a", "sent b", "sent c"]
    assert calls.count("a") == 1 and calls.count("b") == 2


def test_a_cancelled_leg_keeps_what_it_finished_and_resuming_calls_none_of_it_again():
    calls: list[str] = []

    @node("split")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @node("count-calls")
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

    @node("mail")
    def mail(to: str) -> str:
        sent.append(to)
        return f"sent {to}"

    @node("confirm")
    def confirm(receipt: str) -> str:
        sent.append(f"confirm {receipt}")
        return receipt.upper()

    graph = registry_of(mail, confirm).load(GraphRecord(
        nodes=[NodeRecord(id="mail", node="mail", many={"to"}), NodeRecord(id="confirm", node="confirm")],
        edges=[("mail.result", "confirm.receipt")],
    ))
    return graph, sent


def test_a_changed_row_is_new_work_all_the_way_down_and_nothing_else_is():
    graph, sent = _mails()
    first = graph.run(to=["ida", "bo", "al"])
    sent.clear()

    again = first.again(to=["ida", "bob", "al"])

    assert sent == ["bob", "confirm sent bob"]
    assert again.results["confirm"]["result"] == ["SENT IDA", "SENT BOB", "SENT AL"]


def test_a_row_inserted_above_or_a_duplicate_taken_out_moves_the_rest_without_calls():
    graph, sent = _mails()
    first = graph.run(to=["ida", "ida", "bo"])
    sent.clear()

    assert first.again(to=["eva", "ida", "bo"]).results["mail"]["result"] == ["sent eva", "sent ida", "sent bo"]
    assert sent == ["eva", "confirm sent eva"]
    sent.clear()
    first.again(to=["ida", "bo"])
    assert sent == []


def test_a_run_with_changed_inputs_streams_and_a_node_entirely_reused_says_how_much():
    graph, sent = _mails()
    first = graph.run(to=["ida", "bo"])
    sent.clear()

    events = _events(first.with_inputs(to=["ida", "bo", "al"]))

    completed = {e.node_id: e.reused_units for e in events if e.type == "node_complete"}
    assert sent == ["al", "confirm sent al"] and completed == {"mail": 2, "confirm": 2}
    assert events[-1].run.results["confirm"]["result"] == ["SENT IDA", "SENT BO", "SENT AL"]


def test_a_series_reordered_is_new_work_and_one_unchanged_is_not():
    calls: list[list[str]] = []

    @node("join-texts")
    def join_texts(texts: Series[str]) -> str:
        calls.append(list(texts))
        return "+".join(texts)

    graph = registry_of(join_texts).load(GraphRecord(nodes=[NodeRecord(id="j", node="join-texts")]))
    first = graph.run(texts=["a", "b"])

    first.again(texts=["a", "b"])
    first.again(texts=["b", "a"])

    assert calls == [["a", "b"], ["b", "a"]]


def test_what_a_value_is_stays_apart_in_what_a_unit_received():
    calls: list[Any] = []

    @node("echo-any")
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

    @node("signed")
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

        @classmethod
        def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
            from pydantic_core import core_schema

            return core_schema.is_instance_schema(cls)

    @node("make")
    def make(text: str) -> Opaque:
        return Opaque()

    @node("use")
    def use(thing: Opaque) -> str:
        calls.append(thing)
        return "used"

    graph = registry_of(make, use).load(make_graph(make, use))
    first = graph.run(text="a")
    again = first.again(text="a")

    assert again.status == "complete" and again.outputs == {"use.result": "used"}
    assert len(calls) == 2


def test_inner_units_of_an_embedded_graph_are_reused_like_any_others():
    sent: list[str] = []

    @node("mail")
    def mail(to: str) -> str:
        sent.append(to)
        return f"sent {to}"

    @node("upper")
    def upper(text: str) -> str:
        return text.upper()

    @node("split")
    def split(text: str) -> Series[str]:
        return text.split(",")

    inner = node("notify")(make_graph(upper, mail))
    graph = registry_of(split, upper, mail, inner).load(GraphRecord(nodes=[NodeRecord(id="s", node="split"), NodeRecord(id="n", node="notify")], edges=[("s.result", "n.text")]))
    first = graph.run(text="ida,bo")
    sent.clear()

    again = first.again(text="ida,bo,al")

    assert sent == ["AL"] and again.results["n"]["result"] == ["sent IDA", "sent BO", "sent AL"]


def test_an_answer_stays_with_its_row_when_a_row_is_inserted_above_and_an_edited_row_asks_again():
    @node("approve")
    def approve(name: str) -> bool:
        raise Ask(f"Approve {name}?")

    graph = registry_of(approve).load(GraphRecord(nodes=[NodeRecord(id="a", node="approve", many={"name"})]))
    run = graph.run(name=["ida", "bo"])
    run = run.answer("a@0", True).answer("a@1", False).resume()

    moved = run.again(name=["eva", "ida", "bo"])
    edited = run.again(name=["ida", "bob"])

    assert [w.id for w in moved.waiting] == ["a@0"] and moved.results["a"]["result"] == [None, True, False]
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

    @node("mk", compute_outputs=outs)
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


# -- what again reuses, row by row -------------------------------------------------------------


@node("docs")
def _docs(text: str) -> Series[str]:
    return text.split("|") if text else []


class _Line(TypedDict):
    word: str


def test_a_document_inserted_at_the_top_reads_only_the_new_document_and_its_lines_are_read_where_they_now_are():
    read: list[str] = []
    upper_calls: list[str] = []

    @node("lines")
    def lines(text: str) -> Table[_Line]:
        read.append(text)
        return pl.DataFrame({"word": text.split(",")})

    @node("shout")
    def shout(word: str) -> str:
        upper_calls.append(word)
        return word.upper()

    graph = registry_of(_docs, lines, shout).load(GraphRecord(
        nodes=[NodeRecord(id="d", node="docs"), NodeRecord(id="l", node="lines"), NodeRecord(id="s", node="shout")],
        edges=[("d.result", "l.text"), ("l.result.word", "s.word")],
    ))
    first = graph.run(text="a,b|c")
    read.clear()
    upper_calls.clear()

    again = first.again(text="x|a,b|c")

    assert read == ["x"] and upper_calls == ["x"]
    assert again.results["s"]["result"] == ["X", "A", "B", "C"]


def _filtered(threshold_default: int = 2) -> tuple[Any, list[str], list[str]]:
    """CSV text, read to a table, filtered by a ``min`` the caller gives, its names read whole; the calls of the filter and the reader."""
    filtered: list[str] = []
    counted: list[str] = []

    class Row(TypedDict):
        name: str
        age: int

    @node("read-rows")
    def read_rows(text: str) -> Table[Row]:
        return pl.read_csv(io.StringIO(text))

    @node("threshold")
    def threshold(min: int = threshold_default) -> int:
        return min

    @node("older-than")
    def older_than(rows: pl.DataFrame, min: int) -> Table[Row]:
        filtered.append("called")
        return rows.filter(pl.col("age") > min)

    @node("names")
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


def test_a_reused_table_and_a_column_called_afresh_line_up_row_by_row_after_a_document_is_inserted_above():
    @node("lines")
    def lines(text: str) -> Table[_Line]:
        return pl.DataFrame({"word": text.split(",")})

    @node("stamp", policy=Policy(reuse_when_inputs_match=False))
    def stamp(word: str) -> str:
        return f"<{word}>"

    @node("show")
    def show(rows: pl.DataFrame) -> str:
        return ";".join(f"{w}={s}" for w, s in zip(rows["word"], rows["result"], strict=True))

    graph = registry_of(_docs, lines, stamp, show).load(GraphRecord(
        nodes=[NodeRecord(id="d", node="docs"), NodeRecord(id="l", node="lines"), NodeRecord(id="t", node="stamp"), NodeRecord(id="s", node="show")],
        edges=[("d.result", "l.text"), ("l.result.word", "t.word"), ("l.result", "s.rows"), ("t.result", "s.rows")],
    ))
    first = graph.run(text="a,b|c")

    again = first.again(text="x,y,z|a,b|c")

    assert first.results["s"]["result"] == ["a=<a>;b=<b>", "c=<c>"]
    assert again.results["s"]["result"] == ["x=<x>;y=<y>;z=<z>", "a=<a>;b=<b>", "c=<c>"]


@pytest.mark.parametrize("rows", ["a,b", "a,b,c,d", "x,y,z"], ids=["fewer", "more", "same-count"])
def test_again_with_changed_rows_above_an_embedded_graph_gives_its_inner_fixed_value_node_on_every_row(rows):
    @node("label")
    def label(prefix: str) -> str:
        return prefix + ":"

    @node("pair")
    def pair(text: str, tag: str) -> str:
        return tag + text

    @node("same")
    def same(text: str) -> str:
        return text

    inner = node("tagged")(GraphRecord(
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
    assert again.results["t"]["result"] == [f"p:{r}" for r in rows.split(",")]


def test_an_input_filled_again_overrides_the_first_value_and_the_earlier_run_is_untouched():
    graph, sent = _mails()
    first = graph.run(to=["ida"])

    prepared = first.with_inputs(to=["bo"])
    again = prepared.resume()

    assert "to" in graph.inputs
    assert first.inputs != prepared.inputs and first.results["mail"]["result"] == ["sent ida"]
    assert again.results["mail"]["result"] == ["sent bo"]


def _csv_mails() -> tuple[Any, list[str]]:
    """A CSV of names and emails, and a ``body`` (default ``hej``) sent to each email; the mails sent."""
    sent: list[str] = []

    class Person(TypedDict):
        name: str
        email: str

    @node("read-people")
    def read_people(text: str) -> Table[Person]:
        return pl.read_csv(io.StringIO(text))

    @node("body")
    def body_text(body: str = "hej") -> str:
        return body

    @node("send-body")
    def send_body(email: str, body: str) -> str:
        sent.append(email)
        return f"{body} {email}"

    graph = registry_of(read_people, body_text, send_body).load(GraphRecord(
        nodes=[NodeRecord(id="r", node="read-people"), NodeRecord(id="b", node="body"), NodeRecord(id="m", node="send-body")],
        edges=[("r.result.email", "m.email"), ("b.result", "m.body")],
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


def test_a_row_inserted_above_rows_split_from_rows_calls_only_the_new_rows():
    calls: list[str] = []

    @node("words")
    def words(text: str) -> Series[str]:
        return text.split(",")

    @node("shout")
    def shout(text: str) -> str:
        calls.append(text)
        return text.upper()

    graph = registry_of(_docs, words, shout).load(make_graph(_docs, words, shout))
    first = graph.run(text="a,b|c")
    calls.clear()

    again = first.again(text="x,y|a,b|c")

    assert calls == ["x", "y"]
    assert again.results["shout"]["result"] == ["X", "Y", "A", "B", "C"]


def _approvals() -> Any:
    @node("approve")
    def approve(name: str) -> bool:
        raise Ask(f"Approve {name}?")

    return registry_of(approve).load(GraphRecord(nodes=[NodeRecord(id="a", node="approve", many={"name"})]))


def test_each_row_still_waiting_after_again_is_announced_on_the_stream():
    run = _approvals().run(name=["ida", "bo"])

    events = _events(run.with_inputs(name=["eva", "ida", "bo"]))

    assert [e.unit.id for e in events if e.type == "unit_waiting"] == ["a@0", "a@1", "a@2"]
    assert events[-1].status == "waiting"


def test_an_again_that_fails_before_it_reaches_the_old_rows_still_reuses_them_on_resume():
    broken = {"on": True}
    sent: list[str] = []

    @node("mail-or-fail")
    def mail_or_fail(to: str) -> str:
        sent.append(to)
        if to == "eva" and broken["on"]:
            raise ValueError("smtp down")
        return f"sent {to}"

    graph = registry_of(mail_or_fail).load(GraphRecord(nodes=[NodeRecord(id="m", node="mail-or-fail", many={"to"})]))
    first = graph.run(to=["ida", "bo"])
    sent.clear()

    failed = first.again(to=["eva", "ida", "bo"])
    assert failed.status == "failed" and sent == ["eva"]
    broken["on"] = False
    resumed = graph.load_run(failed.to_json()).resume()

    assert sent == ["eva", "eva"] and resumed.results["m"]["result"] == ["sent eva", "sent ida", "sent bo"]


def test_an_again_cancelled_before_the_node_runs_then_resumed_sends_only_the_new_mail():
    graph, sent = _mails()
    first = graph.run(to=["ida", "bo"])
    sent.clear()
    cancel = asyncio.Event()
    cancel.set()

    stopped = _events(first.with_inputs(to=["ida", "bo", "al"]), cancel=cancel)[-1].run
    assert stopped.status == "cancelled"
    resumed = graph.load_run(stopped.to_json()).resume()

    assert sent == ["al", "confirm sent al"] and resumed.results["confirm"]["result"] == ["SENT IDA", "SENT BO", "SENT AL"]


def test_after_an_answered_question_again_with_one_row_added_sends_one_mail():
    sent: list[str] = []

    @node("approve")
    def approve(name: str) -> str:
        raise Ask(f"Mail {name}?")

    @node("mail")
    def mail(to: str) -> str:
        sent.append(to)
        return f"sent {to}"

    graph = registry_of(approve, mail).load(GraphRecord(
        nodes=[NodeRecord(id="a", node="approve", many={"name"}), NodeRecord(id="m", node="mail")], edges=[("a.result", "m.to")],
    ))
    run = graph.run(name=["ida", "bo"]).answer("a@0", "ida@a.dk").answer("a@1", "bo@a.dk").resume()
    sent.clear()

    again = run.again(name=["ida", "bo", "al"])
    assert [w.id for w in again.waiting] == ["a@2"]
    done = again.answer("a@2", "al@a.dk").resume()

    assert sent == ["al@a.dk"] and done.results["m"]["result"] == ["sent ida@a.dk", "sent bo@a.dk", "sent al@a.dk"]


def test_a_third_run_reuses_what_the_second_had_itself_reused():
    graph, sent = _mails()
    second = graph.run(to=["ida", "bo"]).again(to=["ida", "bo", "al"])
    sent.clear()

    third = second.again(to=["ida", "bo", "al", "eva"])

    assert sent == ["eva", "confirm sent eva"] and third.results["mail"]["result"][:2] == ["sent ida", "sent bo"]


def test_a_cancelled_leg_between_two_runs_reuses_what_one_direct_leg_does():
    rng = random.Random(7)
    graph, sent = _mails()
    names = ["ida", "bo", "al", "eva", "kim"]
    rows = ["ida", "bo"]
    run = graph.run(to=rows)
    cancel = asyncio.Event()
    cancel.set()
    for _ in range(20):
        edited = list(rows)
        choice = rng.randrange(3)
        if choice == 0:
            edited.insert(rng.randrange(len(edited) + 1), rng.choice(names))
        elif choice == 1 and edited:
            edited.pop(rng.randrange(len(edited)))
        else:
            rng.shuffle(edited)
        sent.clear()
        direct = run.again(to=edited)
        direct_calls = sorted(sent)
        sent.clear()
        stopped = _events(run.with_inputs(to=edited), cancel=cancel)[-1].run
        through = graph.load_run(stopped.to_json()).resume()
        assert sorted(sent) == direct_calls and through.results == direct.results
        run, rows = direct, edited


def test_a_first_run_reports_no_reuse():
    graph, _ = _mails()

    events = _events(graph.with_inputs(to=["ida", "bo"]))

    assert {e.node_id: e.reused_units for e in events if e.type == "node_complete"} == {"mail": 0, "confirm": 0}


def test_a_run_with_changed_inputs_not_yet_started_shows_no_results():
    graph, sent = _mails()
    first = graph.run(to=["ida"])
    sent.clear()

    prepared = first.with_inputs(to=["ida", "bo"])

    assert prepared.status == "ready" and sent == []
    assert prepared.results == {"mail": {"result": [None, None]}, "confirm": {"result": [None, None]}}


def test_a_typed_in_value_counts_as_its_type_writes_it_not_as_it_was_spelled():
    calls: list[float] = []

    @node("half")
    def half(x: float) -> float:
        calls.append(x)
        return x / 2

    graph = registry_of(half).load(GraphRecord(nodes=[NodeRecord(id="h", node="half")]))
    first = graph.run(x=2)

    again = first.again(x=2.0)

    assert calls == [2.0] and again.outputs == {"h.result": 1.0}
