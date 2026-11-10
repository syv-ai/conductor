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
    InvalidCallError,
    NodeError,
    NodeRecord,
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

    assert [place(e.unit) for e in events if e.type == "unit_waiting"] == ["a@0", "a@1", "a@2"]
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
    assert [place(w) for w in again.waiting] == ["a@2"]
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


def test_a_table_column_its_row_type_doesnt_name_reads_back_from_json_as_written():
    import datetime as dt
    import math

    import polars as pl
    from conductor import Table

    @node("mixed")
    def mixed(text: str) -> Table:
        return pl.DataFrame({"when": [dt.date(2026, 1, 1)], "at": [dt.time(1, 2)], "x": [math.nan], "n": [1]})

    graph = registry_of(mixed).load(GraphRecord(nodes=[NodeRecord(id="m", node="mixed")]))
    frame = graph.load_run(graph.run(text="a").to_json()).results["m"]["result"]

    assert frame.schema == pl.Schema({"when": pl.Date, "at": pl.Time, "x": pl.Float64, "n": pl.Int64})
    assert math.isnan(frame["x"][0])


def test_a_graph_and_a_run_each_have_one_way_in():
    from conductor import Graph, Run

    graph = registry_of(_alone_node()).load(GraphRecord(nodes=[NodeRecord(id="n", node="n")]))
    with pytest.raises(TypeError, match="registry.load"):
        Graph(graph.record, {})
    with pytest.raises(TypeError, match="graph.with_inputs"):
        Run(graph, {})


def _alone_node() -> Any:
    @node("n")
    def n(text: str = "") -> str:
        return text

    return n


def test_waiting_rows_and_a_saved_run_keep_node_then_row_order_whatever_threads_did():
    import random

    @node("slow-ask", policy=Policy(concurrency=4))
    def slow_ask(text: str) -> str:
        import time

        time.sleep(random.random() * 0.02)
        raise Ask(f"{text}?")

    @node("split")
    def split(text: str) -> Series[str]:
        return text.split(",")

    graph = registry_of(split, slow_ask).load(make_graph(split, slow_ask))
    for _ in range(3):
        run = graph.run(text=",".join("abcdefgh"))
        assert [place(w) for w in run.waiting] == [f"slow-ask@{i}" for i in range(8)]
        assert json.loads(run.to_json())["units"]["slow-ask"]["keys"] == [[i] for i in range(8)]


def test_a_saved_run_that_does_not_fit_is_refused_naming_where_and_never_with_a_raw_error():
    @node("split")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @node("echo")
    def echo(text: str) -> str:
        return text

    graph = registry_of(split, echo).load(make_graph(split, echo))
    saved = json.loads(graph.run(text="a,b,c").to_json())
    for tamper, at in (
        (lambda s: s["units"]["echo"]["keys"].__setitem__(2, [7]), "echo"),
        (lambda s: s["units"]["echo"]["outputs"]["result"]["values"].pop(), "echo.result"),
        (lambda s: s["units"]["echo"]["outputs"].pop("result"), "echo.result"),
        (lambda s: s["units"]["echo"]["outputs"]["result"]["values"].__setitem__(0, 5), "echo.result"),
    ):
        copy = json.loads(json.dumps(saved))
        tamper(copy)
        with pytest.raises(InvalidCallError) as refused:
            graph.load_run(copy)
        (problem,) = refused.value.errors()
        assert (problem.type, problem.ctx["at"]) == ("state_unreadable", at), problem


# -- identical rows, generations, waiting rows and forgetting ----------------------------------


def _answered(names: list[str], answers: list[bool]) -> Any:
    @node("approve")
    def approve(name: str) -> bool:
        raise Ask(f"Mail {name}?")

    graph = registry_of(approve).load(GraphRecord(nodes=[NodeRecord(id="a", node="approve", many={"name"})]))
    run = graph.run(name=names)
    return run.answer({w.id: answer for w, answer in zip(run.waiting, answers, strict=True)}).resume()


def test_identical_rows_keep_their_own_answers_kth_to_kth_when_a_row_is_inserted_above():
    """Bo yes, Bo no, Al yes; with Eva above, the first Bo is still yes and the second still no."""
    run = _answered(["bo", "bo", "al"], [True, False, True])

    moved = run.again(name=["eva", "bo", "bo", "al"])

    assert [place(w) for w in moved.waiting] == ["a@0"] and moved.results["a"]["result"] == [None, True, False, True]
    assert run.again(name=["bo", "bo", "al"]).results["a"]["result"] == [True, False, True]


def test_identical_answered_rows_keep_their_answers_in_order_when_they_all_move_and_through_json():
    run = _answered(["bo", "al", "bo"], [True, True, False])

    moved = run.graph.load_run(run.to_json()).again(name=["al", "bo", "bo"])

    assert moved.results["a"]["result"] == [True, True, False] and moved.waiting == []


def test_a_third_identical_row_is_asked_and_two_earlier_answers_cover_only_two_rows():
    run = _answered(["bo", "bo"], [True, False])

    more = run.again(name=["bo", "bo", "bo"])

    assert [place(w) for w in more.waiting] == ["a@2"] and more.results["a"]["result"] == [True, False, None]


def test_identical_lines_answered_differently_under_documents_keep_their_documents_when_the_documents_swap():
    @node("lines")
    def lines(doc: str) -> Series[str]:
        return ["x", "x"]

    @node("approve")
    def approve(line: str) -> bool:
        raise Ask(f"{line}?")

    graph = registry_of(lines, approve).load(GraphRecord(
        nodes=[NodeRecord(id="l", node="lines", many={"doc"}), NodeRecord(id="a", node="approve")], edges=[("l.result", "a.line")],
    ))
    run = graph.run(doc=["d1", "d2"])
    answers = dict(zip([w.id for w in run.waiting], [True, False, False, True], strict=True))
    done = run.answer(answers).resume()

    swapped = done.again(doc=["d2", "d1"])

    assert done.results["a"]["result"] == [True, False, False, True]
    assert swapped.results["a"]["result"] == [False, True, True, False] and swapped.waiting == []  # d2's lines, then d1's


def test_identical_rows_never_share_a_result_a_third_gets_a_call_of_its_own():
    calls: list[str] = []

    @node("echo")
    def echo(text: str) -> str:
        calls.append(text)
        return text.upper()

    graph = registry_of(echo).load(GraphRecord(nodes=[NodeRecord(id="e", node="echo", many={"text"})]))
    run = graph.run(text=["a", "a"])
    calls.clear()

    assert run.again(text=["a", "a", "a"]).results["e"]["result"] == ["A", "A", "A"] and calls == ["a"]


def test_again_remembers_every_generation_an_answer_survives_an_edit_and_its_revert_and_mails_once():
    """A answered; B changes Bo and waits; C reverts to A's input: Bo keeps A's answer and is not mailed twice."""
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
    first = graph.run(name=["ida", "bo"])
    a = first.answer({w.id: f"{w.inputs['name']}@a.dk" for w in first.waiting}).resume()
    b = a.again(name=["ida", "bob"])
    assert [w.inputs["name"] for w in b.waiting] == ["bob"]
    sent.clear()

    c = graph.load_run(b.to_json()).again(name=["ida", "bo"])

    assert c.status == "complete" and c.results["m"]["result"] == ["sent ida@a.dk", "sent bo@a.dk"] and sent == []


def test_a_second_edit_after_a_failed_leg_still_reuses_what_the_failed_leg_had_not_reached():
    calls: list[str] = []

    @node("echo")
    def echo(text: str) -> str:
        calls.append(text)
        if text == "boom":
            raise ValueError("no")
        return text.upper()

    graph = registry_of(echo).load(GraphRecord(nodes=[NodeRecord(id="e", node="echo", many={"text"})]))
    first = graph.run(text=["a", "b", "c"])
    failed = first.again(text=["boom", "b", "c"])
    assert failed.status == "failed"
    calls.clear()

    fixed = failed.again(text=["a", "b", "c"])

    assert fixed.status == "complete" and calls == [] and fixed.results["e"]["result"] == ["A", "B", "C"]


def test_earlier_work_of_two_edits_can_sit_at_the_same_row_and_neither_hides_the_other():
    calls: list[str] = []

    @node("echo")
    def echo(text: str) -> str:
        calls.append(text)
        return text.upper()

    graph = registry_of(echo).load(GraphRecord(nodes=[NodeRecord(id="e", node="echo", many={"text"})]))
    second = graph.run(text=["a"]).again(text=["b"])
    calls.clear()

    assert second.again(text=["a", "b"]).results["e"]["result"] == ["A", "B"] and calls == []


def test_a_waiting_row_goes_on_waiting_at_its_new_row_with_no_new_question():
    asked: list[str] = []

    @node("approve")
    def approve(name: str) -> bool:
        asked.append(name)
        raise Ask(f"Mail {name}?")

    graph = registry_of(approve).load(GraphRecord(nodes=[NodeRecord(id="a", node="approve", many={"name"})]))
    run = graph.run(name=["ida", "bo"])
    asked.clear()

    moved = run.again(name=["eva", "ida", "bo"])

    assert asked == ["eva"] and [(place(w), w.inputs["name"]) for w in moved.waiting] == [("a@0", "eva"), ("a@1", "ida"), ("a@2", "bo")]


def test_a_waiting_row_carried_through_a_cancelled_again_waits_at_its_new_row():
    asked: list[str] = []

    @node("approve", policy=Policy(concurrency=2))
    def approve(name: str) -> bool:
        asked.append(name)
        raise Ask(f"Mail {name}?")

    graph = registry_of(approve).load(GraphRecord(nodes=[NodeRecord(id="a", node="approve", many={"name"})]))
    run = graph.run(name=["ida", "bo"])
    cancel = asyncio.Event()
    cancel.set()
    stopped = _events(run.with_inputs(name=["eva", "ida", "bo"]), cancel=cancel)[-1].run
    asked.clear()

    resumed = graph.load_run(stopped.to_json()).resume()

    assert asked == ["eva"] and [w.inputs["name"] for w in resumed.waiting] == ["eva", "ida", "bo"]


def test_an_answer_for_a_place_that_now_holds_another_question_is_answer_stale_naming_the_current_one():
    @node("approve")
    def approve(name: str) -> bool:
        raise Ask(f"Mail {name}?")

    graph = registry_of(approve).load(GraphRecord(nodes=[NodeRecord(id="a", node="approve", many={"name"})]))
    first = graph.run(name=["ida", "bo"])
    (_, bo) = first.waiting
    moved = first.again(name=["ida", "al", "bo"])

    with pytest.raises(InvalidCallError) as refused:
        moved.answer(bo.id, True)  # bo's id names a@1, where al waits now
    (problem,) = refused.value.errors()
    current = moved.waiting[1].id
    assert (problem.type, problem.loc, problem.ctx) == ("answer_stale", ("a", 1), {"waiting_id": bo.id, "current": current})
    assert moved.answer("a@1", True).results["a"]["result"] == [None, True, None]  # a place alone answers what waits there


def test_a_changed_node_asks_again_with_its_new_prompt_and_its_old_unit_is_gone():
    @node("approve")
    def approve(name: str) -> bool:
        raise Ask(f"Mail {name}?")

    @node("approve", replaces=approve)
    def approve_v2(name: str) -> bool:
        raise Ask(f"Send to {name}?")

    record = GraphRecord(nodes=[NodeRecord(id="a", node="approve", version=1, many={"name"})])
    old = registry_of(approve_v2).load(record).run(name=["ida"])
    (waiting,) = old.waiting
    newer = registry_of(approve_v2).upgrade(record, "a")
    graph = registry_of(approve_v2).load(newer)
    run = graph.run(name=["ida"])

    assert [w.prompt for w in run.waiting] == ["Send to ida?"] and run.waiting[0].id != waiting.id


def test_a_node_redeclared_under_its_id_and_version_with_other_code_is_new_work_after_a_save():
    @node("echo")
    def echo(text: str) -> str:
        return text.upper()

    @node("echo")
    def echo_changed(text: str) -> str:
        return text.lower()

    record = GraphRecord(nodes=[NodeRecord(id="e", node="echo", many={"text"})])
    saved = registry_of(echo).load(record).run(text=["Ab"]).to_json()

    again = registry_of(echo_changed).load(record).load_run(saved).again(text=["Ab"])

    assert again.results["e"]["result"] == ["ab"]


def test_a_node_redeclared_to_take_a_blank_is_called_on_the_blank_row_it_skipped_before():
    class Row(TypedDict):
        name: str
        email: str | None

    @node("rows")
    def rows(text: str) -> Table[Row]:
        return pl.DataFrame({"name": ["ida", "bo"], "email": ["i@a.dk", None]}, schema={"name": pl.String, "email": pl.String})

    @node("send")
    def send(email: str) -> str:
        return f"to {email}"

    @node("send")
    def send_blank(email: str | None) -> str:
        return f"to {email}"

    record = GraphRecord(nodes=[NodeRecord(id="r", node="rows"), NodeRecord(id="s", node="send")], edges=[("r.result.email", "s.email")])
    saved = registry_of(rows, send).load(record).run(text="x").to_json()

    again = registry_of(rows, send_blank).load(record).load_run(saved).again()

    assert again.results["s"]["result"] == ["to i@a.dk", "to None"]


def _counted() -> tuple[Any, list[str]]:
    calls: list[str] = []

    @node("fetch")
    def fetch(text: str) -> str:
        calls.append(f"fetch {text}")
        return text.upper()

    @node("show")
    def show(text: str) -> str:
        calls.append(f"show {text}")
        return f"<{text}>"

    graph = registry_of(fetch, show).load(GraphRecord(
        nodes=[NodeRecord(id="f", node="fetch", many={"text"}), NodeRecord(id="s", node="show")], edges=[("f.result", "s.text")],
    ))
    return graph, calls


def test_forget_runs_a_node_again_though_nothing_changed_and_a_reader_that_receives_the_same_is_reused():
    graph, calls = _counted()
    run = graph.run(text=["a", "b"])
    calls.clear()

    again = run.forget("f").resume()

    assert calls == ["fetch a", "fetch b"] and again.results == run.results
    assert run.forget().resume().results == run.results and calls == ["fetch a", "fetch b"]


def test_forget_is_not_started_survives_json_and_refuses_a_node_the_graph_lacks():
    graph, calls = _counted()
    run = graph.run(text=["a"])
    forgotten = run.forget("f")
    calls.clear()

    assert forgotten.status == "ready" and forgotten.results["f"] == {"result": [None]}
    assert graph.load_run(forgotten.to_json()).resume().status == "complete" and calls == ["fetch a"]
    with pytest.raises(InvalidCallError) as refused:
        run.forget("nope")
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("node_not_in_graph", ("nope",))]


def test_forgetting_an_embedded_graph_forgets_its_inner_nodes():
    calls: list[str] = []

    @node("fetch")
    def fetch(text: str) -> str:
        calls.append(text)
        return text.upper()

    inner = node("inner")(GraphRecord(nodes=[NodeRecord(id="f", node="fetch")]))
    graph = registry_of(fetch, inner).load(GraphRecord(nodes=[NodeRecord(id="e", node="inner")]))
    run = graph.run(text="a")
    calls.clear()

    assert run.forget("e").resume().results["e"] == {"result": "A"} and calls == ["a"]


def test_a_table_nothing_holds_after_forget_is_not_written():
    @node("split")
    def split(text: str) -> Series[str]:
        return text.split(",")

    graph = registry_of(split).load(GraphRecord(nodes=[NodeRecord(id="s", node="split")]))
    run = graph.run(text="a,b")

    written = json.loads(run.forget("s").to_json())

    assert written["units"]["s"]["states"] == [] and "s" not in written["earlier"]


def test_identical_answered_rows_keep_their_answers_across_a_cancelled_leg_as_on_a_direct_one():
    run = _answered(["bo", "bo", "al"], [True, False, True])
    direct = run.again(name=["eva", "bo", "bo", "al"])
    cancel = asyncio.Event()
    cancel.set()
    stopped = _events(run.with_inputs(name=["eva", "bo", "bo", "al"]), cancel=cancel)[-1].run

    through = run.graph.load_run(stopped.to_json()).resume()

    assert through.results == direct.results == {"a": {"result": [None, True, False, True]}}
    assert [w.inputs for w in through.waiting] == [w.inputs for w in direct.waiting] == [{"name": "eva"}]


def test_a_save_of_a_node_redeclared_so_it_runs_on_other_rows_does_not_fit_and_is_refused_naming_the_node():
    @node("count")
    def count(text: str) -> int:
        return len(text)

    @node("count")
    def count_all(text: Series[str]) -> int:
        return len(text)

    @node("split")
    def split(text: str) -> Series[str]:
        return text.split(",")

    record = GraphRecord(nodes=[NodeRecord(id="s", node="split"), NodeRecord(id="c", node="count")], edges=[("s.result", "c.text")])
    saved = registry_of(split, count).load(record).run(text="ab,c").to_json()

    with pytest.raises(InvalidCallError) as refused:
        registry_of(split, count_all).load(record).load_run(saved)

    assert [(p.type, p.ctx["at"]) for p in refused.value.errors()] == [("state_unreadable", "c")]
