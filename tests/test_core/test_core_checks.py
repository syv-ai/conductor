"""The core's rules through the public API: registry, ids, reuse, answers, graph records, items, subgraphs, upgrades.

The invoice graph reads a CSV, asks a person to approve each big amount, and
sums the decisions; ``calls`` records which nodes were called, so a test can
say what ran again and what was reused.
"""
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Annotated, TypedDict
from unittest import mock

import pytest
from conductor import (
    SKIPPED,
    Ask,
    Graph,
    GraphRecord,
    InputEntry,
    InvalidCallError,
    NodeDeclarationError,
    NodeRegistry,
    NotCalled,
    Param,
    Policy,
    Run,
    Series,
    Source,
    Table,
    Upgrade,
    make_graph,
    node,
    subgraph,
)

from tests.test_core.nodes import (
    join,
    lower,
    mail,
    place,
    read_csv,
    registry,
    registry_of,
    split,
    tail,
    upper,
)

CSV = "name,email,amount\nIda,ida@a.dk,100\nBo,bo@b.dk,5000\nAl,al@c.dk,7000\n"


class InvoiceLine(TypedDict):
    name: str
    email: str | None
    amount: int


@node("each-invoice", title="Each invoice", description="For a test.")
def each_invoice(table: Table[InvoiceLine]) -> Series[InvoiceLine]:
    return table.to_dicts()


@dataclass(frozen=True)
class Length:
    long: Annotated[str, Param(exclusive_group="length")] = SKIPPED
    short: Annotated[str, Param(exclusive_group="length")] = SKIPPED


@node("route", title="Route", description="For a test.")
def route(text: str) -> Length:
    return Length(long=text) if len(text) > 3 else Length(short=text)


#: The shared nodes and the one that makes an item of each invoice.
invoicing = NodeRegistry()
invoicing.add_subregistry(registry)
invoicing.add(each_invoice)

#: The invoices read from a CSV, then one item per invoice.
INVOICES = {"nodes": {"read": {"node": "read-csv", "inputs": {"text": {"graph_input": "text"}}}, "each": {"node": "each-invoice", "inputs": {"table": {"sources": [{"node": "read", "output": "result"}]}}}}}


def invoices_then(nodes: dict) -> dict:
    """The record of ``INVOICES`` with ``nodes`` added, each keyed by its id."""
    return {"nodes": {**INVOICES["nodes"], **nodes}}


def each_field(field: str) -> dict:
    """The entry of an input wired to one field of the invoice of each item."""
    return {"sources": [{"node": "each", "output": "result", "field": field}]}


def edge_from(node_id: str, output: str = "result") -> dict:
    """The entry of an input wired to the output ``output`` of ``node_id``."""
    return {"sources": [{"node": node_id, "output": output}]}


def calls(fn):
    """What ``fn`` returns, and the calls the nodes noted while it ran."""
    with TemporaryDirectory() as work, mock.patch.dict(os.environ, {"CALLS": str(Path(work) / "calls")}):
        Path(os.environ["CALLS"]).touch()
        out = fn()
        return out, Path(os.environ["CALLS"]).read_text().split("\n")[:-1]


def test_calls_leaves_the_environment_as_it_found_it_when_what_it_runs_fails():
    def failing() -> None:
        raise AssertionError("the test's own failure")

    with pytest.raises(AssertionError):
        calls(failing)
    assert "CALLS" not in os.environ


def test_registry():
    assert "upper" in registry and registry["approve"].category == "invoices" and registry["approve"].tags == ("finance",)
    assert [d.id for d in registry.filter(lambda d: d.category == "text")][:2] == ["upper", "lower"]
    with pytest.raises(NodeDeclarationError):
        registry.add(upper)
    small = registry.filter(lambda d: d.id != "upper")
    assert "upper" not in small and "upper" in registry


def test_node_used_twice():
    record = make_graph(upper, tail, upper)  # a chain; upper repeats
    assert list(record.nodes) == ["upper-1", "tail", "upper-2"]
    graph = Graph.compile(record.model_dump_json(), registry)
    assert graph.run(text="hi").outputs == {"result": "HI."}


def test_per_item_reuse_on_again():
    graph = Graph.compile(GraphRecord.model_validate(invoices_then({"mail": {"node": "mail", "inputs": {"email": each_field("email")}}})), invoicing)
    first, before = calls(lambda: graph.run(text=CSV))
    second, after = calls(lambda: first.again(text=CSV.replace("bo@b.dk", "bo@new.dk") + "Eva,eva@e.dk,1\n"))
    assert len(before) == 3 and after == ["mail bo@new.dk", "mail eva@e.dk"], after
    assert second.results["mail"]["result"][1] == "sent to bo@new.dk"


def test_answer_stays_with_its_item_on_again():
    graph = Graph.compile(GraphRecord.model_validate(invoices_then({
        "approve": {"node": "approve", "inputs": {"name": each_field("name"), "amount": each_field("amount")}},
        "summary": {"node": "summary", "inputs": {"decisions": edge_from("approve")}},
    })), invoicing)
    run, _ = calls(lambda: graph.run(text=CSV))
    for waiting_call in run.waiting:
        run = run.answer(waiting_call, "yes")
    run, _ = calls(run.resume)
    edited = CSV.replace("Ida,ida@a.dk,100", "Ida,ida@a.dk,200").replace("Al,al@c.dk,7000", "Al,al@c.dk,8000")
    again, called = calls(lambda: run.again(text=edited))
    assert called == ["approve Ida", "approve Al"], called  # Bo's answer stays with Bo
    assert again.status == "waiting" and [w.inputs["name"] for w in again.waiting] == ["Al"]
    try:
        again.answer(again.waiting[0], "maybe")
        raise AssertionError
    except InvalidCallError as error:
        assert error.problems[0].type == "invalid_answer"


def test_a_half_finished_graph_record_compiles_with_every_problem_and_refuses_to_run():
    record = invoices_then({
        "approve": {"node": "approve", "inputs": {"name": each_field("name"), "amount": edge_from("ghost")}},
        "mystery": {"node": "gone"},
        "rep": {"node": "repeat", "inputs": {"times": {"constant": -1}, "text": each_field("nope")}},
    })
    graph = Graph.compile(GraphRecord.model_validate(record), invoicing)
    found = sorted((p.type, p.loc) for p in graph.problems)
    assert found == [("edge_names_missing_node", ("approve", "amount")), ("invalid_value", ("rep", "times")), ("unknown_node", ("mystery",)),
                     ("unknown_output", ("rep", "text"))], found
    try:
        graph.run(text=CSV)
        raise AssertionError
    except InvalidCallError as error:
        assert len(error.problems) == 4
    open_first = Graph.compile(GraphRecord.model_validate({"nodes": {"a": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}}, "outputs": {"result": {"node": "a", "output": "result"}}}), registry)
    assert open_first.problems == [] and list(open_first.inputs) == ["text"]
    try:
        open_first.run()
        raise AssertionError
    except InvalidCallError as error:
        assert error.problems[0].type == "missing_input"


def test_every_problem_of_one_node_is_reported_at_once():
    later = Graph.compile(GraphRecord.model_validate({"nodes": {"a": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "b": {"node": "repeat", "inputs": {"times": {"sources": [{"node": "a", "output": "result"}]}}}}, "outputs": {"result": {"node": "b", "output": "result"}}}), registry)
    assert [(p.type, p.loc) for p in later.problems] == [("input_has_no_value", ("b", "text")), ("type_mismatch", ("b", "times"))]
    unbound = Graph.compile(GraphRecord.model_validate({"nodes": {"a": {"node": "upper"}, "b": {"node": "join", "inputs": {"texts": {"sources": [{"node": "a", "output": "result"}]}}}, "c": {"node": "repeat", "inputs": {"times": {"sources": [{"node": "b", "output": "result"}]}}}}}), registry)
    assert "type_mismatch" in [p.type for p in unbound.problems]


def test_nested_loops_are_collected_one_level_at_a_time():
    # documents -> lines -> per-line upper -> per-document join; the lines read whole are a list per document too.
    record = {"nodes": {"docs": {"node": "split", "inputs": {"sep": {"constant": ";"}, "text": {"graph_input": "text"}}}, "lines": {"node": "split", "inputs": {"text": {"sources": [{"node": "docs", "output": "result"}]}}}, "up": {"node": "upper", "inputs": {"text": {"sources": [{"node": "lines", "output": "result"}]}}}, "per_doc": {"node": "join", "inputs": {"texts": {"sources": [{"node": "up", "output": "result"}]}}}, "all": {"node": "join", "inputs": {"texts": {"sources": [{"node": "lines", "output": "result"}]}}}}, "outputs": {"all_result": {"node": "all", "output": "result"}, "per_doc_result": {"node": "per_doc", "output": "result"}}}
    graph = Graph.compile(GraphRecord.model_validate(record), registry)
    assert {k: (v.runs.mode, v.runs.series) for k, v in graph.describe().nodes.items()} == {
        "docs": ("once", None), "lines": ("per_item", "docs.result"), "up": ("per_item", "lines.result"),
        "per_doc": ("per_item", "docs.result"), "all": ("per_item", "docs.result"),
    }
    run = graph.run(text="a,b;c;")
    assert run.results["per_doc"]["result"] == ["A+B", "C", ""], run.results["per_doc"]
    assert run.results["all"]["result"] == ["a+b", "c", ""]  # a node collecting lines is inside the loop over docs, as lines is


def test_items_of_two_series_at_one_node_are_refused():
    record = {"nodes": {"a": {"node": "split", "inputs": {"text": {"graph_input": "a_text"}, "sep": {"graph_input": "a_sep"}}}, "b": {"node": "split", "inputs": {"text": {"graph_input": "b_text"}, "sep": {"graph_input": "b_sep"}}}, "r": {"node": "repeat", "inputs": {"text": {"sources": [{"node": "a", "output": "result"}]}, "times": {"sources": [{"node": "b", "output": "result"}]}}}}, "outputs": {"result": {"node": "r", "output": "result"}}}
    graph = Graph.compile(GraphRecord.model_validate(record), registry)
    assert [p.type for p in graph.problems] == ["type_mismatch"]
    record["nodes"]["r"] = {"node": "pair", "inputs": {"a": edge_from("a"), "b": edge_from("b")}}

    @node("pair", title="Pair", description="For a test.")
    def pair(a: str, b: str) -> str:
        return a + b

    graph = Graph.compile(GraphRecord.model_validate(record), registry_of(split, pair))
    assert [(p.type, p.ctx["series"]) for p in graph.problems] == [("series_not_nested", ["a.result", "b.result"])], graph.problems
    assert "neither is inside the loop over the other" in graph.problems[0].msg


def test_branches_are_collected_in_item_order_and_a_none_email_is_not_mailed():
    @node("pick", title="Pick", description="For a test.")
    def pick(long: str | None = None, short: str | None = None) -> str:
        return long or short

    reg = registry_of(split, route, upper, join, pick, mail)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "split", "inputs": {"text": {"graph_input": "text"}, "sep": {"graph_input": "sep", "default": ","}}}, "r": {"node": "route", "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}}}, "L": {"node": "upper", "inputs": {"text": {"sources": [{"node": "r", "output": "long"}]}}}, "J": {"node": "join", "inputs": {"texts": {"sources": [{"node": "L", "output": "result"}]}}}, "both": {"node": "join", "inputs": {"texts": {"sources": [{"node": "L", "output": "result"}, {"node": "r", "output": "short"}]}}}}, "outputs": {"J_result": {"node": "J", "output": "result"}, "both_result": {"node": "both", "output": "result"}}}), reg)
    run = graph.run(text="abcd,ab,xyzzy")
    assert run.results["L"]["result"] == ["ABCD", None, "XYZZY"]
    assert run.results["J"]["result"] == "ABCD+XYZZY"
    assert run.results["both"]["result"] == "ABCD+ab+XYZZY"  # the two branches of one node are collected in item order
    with_none = Graph.compile(GraphRecord.model_validate(invoices_then({"mail": {"node": "mail", "inputs": {"email": each_field("email")}}})), registry_of(read_csv, each_invoice, mail))
    run = with_none.run(text="name,email,amount\nIda,ida@a.dk,1\nBo,,2\n")
    assert run.status == "complete" and run.results["mail"]["result"] == ["sent to ida@a.dk", None]
    assert run.not_called == [NotCalled(loc=("mail", 1), input="email")]


def test_one_value_is_collected_as_a_series_of_one_and_a_list_given_for_one_value_is_refused():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"up": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "j": {"node": "join", "inputs": {"texts": {"sources": [{"node": "up", "output": "result"}]}}}}, "outputs": {"result": {"node": "j", "output": "result"}}}), registry)
    assert graph.run(text="a").outputs == {"result": "A"}
    with pytest.raises(InvalidCallError) as refused:
        graph.run(text=["a", "b"])
    assert [(p.type, p.loc) for p in refused.value.problems] == [("invalid_input", ("inputs", "text"))]


def test_max_items():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "split", "inputs": {"text": {"graph_input": "text"}, "sep": {"graph_input": "sep", "default": ","}}}, "m": {"node": "mail", "inputs": {"email": {"sources": [{"node": "s", "output": "result"}]}}}}, "outputs": {"result": {"node": "m", "output": "result"}}}), registry)
    (run, called) = calls(lambda: graph.run(text=",".join(f"x{i}" for i in range(50)), max_items=10))
    assert run.status == "stopped" and run.cause.type == "item_limit" and called == []
    assert run.cause.ctx == {"series": "s.result", "items": 50, "max_items": 10}
    saved = Run.from_json(run.to_json(), graph)
    (done, called) = calls(lambda: saved.with_max_items(100).resume())
    assert done.status == "complete" and len(called) == 50


def test_an_upgraded_subgraph_keeps_the_name_of_its_output():
    """The second version declares the same graph output, so what read ``e.result`` still does."""
    enrich_v1 = make_graph(upper, tail)
    enrich_v2 = {"nodes": {"lower": {"node": "lower", "inputs": {"text": {"graph_input": "text"}}}, "dot": {"node": "tail", "inputs": {"text": edge_from("lower")}}, "tail": {"node": "tail", "inputs": {"text": edge_from("dot")}}}, "outputs": {"result": {"node": "tail", "output": "result"}}}
    v1 = subgraph("enrich", description="For a test.", title="Enrich")(enrich_v1)
    v2 = subgraph("enrich", title="Enrich", description="For a test.", replaces=v1)(GraphRecord.model_validate(enrich_v2))
    reg = registry_of(upper, lower, tail, v2)
    outer = GraphRecord.model_validate({"nodes": {"e": {"node": "enrich", "version": 1, "inputs": {"text": {"graph_input": "text"}}}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "e", "output": "result"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}})
    assert Graph.compile(outer, reg).run(text="Hi").outputs == {"result": "HI."}
    newer = reg.upgrade(outer, "e")
    assert newer.nodes["e"].version == 2 and Graph.compile(newer, reg).run(text="Hi").outputs == {"result": "HI.."}
    graph = Graph.compile(outer, reg)
    assert set(graph._nodes) == {"e", "u"}


def test_an_upgrade_moves_an_edge_reading_a_subgraphs_output_to_the_name_the_newer_version_gives_it():
    v1 = subgraph("enrich", description="For a test.", title="Enrich")(make_graph(upper, tail))
    v2 = subgraph("enrich", title="Enrich", description="For a test.", replaces=v1, upgrade=Upgrade(renamed_outputs={"result": "dotted"}))(make_graph(lower, tail, tail, outputs={"dotted": "tail-2.result"}))
    reg = registry_of(upper, lower, tail, v2)
    outer = GraphRecord.model_validate({"nodes": {"e": {"node": "enrich", "version": 1, "inputs": {"text": {"graph_input": "text"}}}, "u": {"node": "upper", "inputs": {"text": {"sources": [{"node": "e", "output": "result"}]}}}}, "outputs": {"result": {"node": "u", "output": "result"}}})
    newer = reg.upgrade(outer, "e")
    assert newer.nodes["u"].inputs["text"].sources == (Source("e", "dotted"),)
    assert Graph.compile(newer, reg).run(text="Hi").outputs == {"result": "HI.."}


def test_upgrade_renames():
    @node("greet", title="Greet v1", description="For a test.")
    def greet_v1(name: str) -> str:
        return f"hello {name}"

    @node("greet", title="Greet", description="For a test.", replaces=greet_v1, upgrade=Upgrade(renamed_inputs={"name": "who"}))
    def greet(who: str, punct: str = "!") -> str:
        return f"hello {who}{punct}"

    reg = registry_of(greet, upper)
    stored = GraphRecord.model_validate({"nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}}, "g": {"node": "greet", "inputs": {"name": {"sources": [{"node": "u", "output": "result"}]}}}}, "outputs": {"result": {"node": "g", "output": "result"}}})
    assert Graph.compile(stored, reg).run(text="ida").outputs == {"result": "hello IDA"}
    moved = reg.upgrade(stored, "g")
    assert moved.nodes["g"].inputs == {"who": InputEntry(sources=(Source("u", "result"),))}
    assert Graph.compile(moved, reg).run(text="ida").outputs == {"result": "hello IDA!"}


def test_ask_forms_and_several():
    @dataclass(frozen=True)
    class Review:
        approved: bool
        note: str

    @node("review", title="Review", description="For a test.", asks=True)
    def review(name: str) -> Review:
        raise Ask(f"Review {name}", default={"approved": True, "note": ""})

    @node("tags", title="Tags", description="For a test.", asks=True)
    def tags(name: str) -> Series[str]:
        raise Ask(f"Tags for {name}", choices=["red", "green", "blue"])

    @node("pick", title="Pick", description="For a test.", asks=True)
    def pick(name: str) -> bool:
        raise Ask(f"Keep {name}?")

    reg = registry_of(split, review, tags, pick, join)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "split", "inputs": {"text": {"graph_input": "text"}, "sep": {"graph_input": "sep", "default": ","}}}, "r": {"node": "review", "inputs": {"name": {"sources": [{"node": "s", "output": "result"}]}}}, "t": {"node": "tags", "inputs": {"name": {"sources": [{"node": "s", "output": "result"}]}}}, "p": {"node": "pick", "inputs": {"name": {"sources": [{"node": "s", "output": "result"}]}}}, "j": {"node": "join", "inputs": {"texts": {"sources": [{"node": "t", "output": "result"}]}}}}, "outputs": {"p_result": {"node": "p", "output": "result"}, "approved": {"node": "r", "output": "approved"}, "note": {"node": "r", "output": "note"}, "j_result": {"node": "j", "output": "result"}}}), reg)
    run = graph.run(text="Ida,Bo")
    by = {place(w): w for w in run.waiting}
    approved, note = by["r@0"].questions
    assert (approved.name, approved.json_schema, approved.default, note.default) == ("approved", {"type": "boolean"}, True, "")
    assert by["p@1"].questions[0].json_schema == {"type": "boolean"} and by["t@0"].questions[0].is_series
    run = Run.from_json(run.to_json(), graph)
    run = run.answer("r@0", {"approved": False, "note": "late"}).answer("r@1", {"approved": True, "note": ""})
    run = run.answer("t@0", ["red", "blue"]).answer("t@1", []).answer("p@0", True).answer("p@1", False).resume()
    assert run.status == "complete", run.status
    assert run.results["r"]["approved"] == [False, True] and run.results["j"]["result"] == ["red+blue", ""]


def test_pause_on_a_nested_item():
    @node("check", title="Check", description="For a test.", asks=True)
    def check(text: str) -> str:
        if text == "b":
            raise Ask("b?")
        return text

    reg = registry_of(split, check, join)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"docs": {"node": "split", "inputs": {"sep": {"constant": ";"}, "text": {"graph_input": "text"}}}, "lines": {"node": "split", "inputs": {"text": {"sources": [{"node": "docs", "output": "result"}]}}}, "c": {"node": "check", "inputs": {"text": {"sources": [{"node": "lines", "output": "result"}]}}}, "j": {"node": "join", "inputs": {"texts": {"sources": [{"node": "c", "output": "result"}]}}}}, "outputs": {"result": {"node": "j", "output": "result"}}}), reg)
    run = graph.run(text="a,b;c")
    assert [place(w) for w in run.waiting] == ["c@0@1"] and run.results["j"]["result"] == [None, "c"]
    run = Run.from_json(run.to_json(), graph).answer("c@0@1", "B").resume()
    assert run.results["j"]["result"] == ["a+B", "c"]




def test_concurrency():
    import threading
    import time as t

    seen = set()

    @node("slow", title="Slow", description="For a test.", policy=Policy(concurrency=8))
    def slow(text: str) -> str:
        seen.add(threading.get_ident())
        t.sleep(0.05)
        return text

    graph = Graph.compile(GraphRecord.model_validate({"nodes": {"s": {"node": "split", "inputs": {"text": {"graph_input": "text"}, "sep": {"graph_input": "sep", "default": ","}}}, "w": {"node": "slow", "inputs": {"text": {"sources": [{"node": "s", "output": "result"}]}}}}, "outputs": {"result": {"node": "w", "output": "result"}}}), registry_of(split, slow))
    start = t.monotonic()
    run = graph.run(text=",".join("abcdefgh"))
    assert t.monotonic() - start < 0.2 and len(seen) > 1 and run.results["w"]["result"] == list("abcdefgh")


def test_again_waiting_read_back_and_answered_alike_sends_no_mail_again():
    @node("gate", title="Gate", description="For a test.", asks=True)
    def gate(name: str, email: str) -> str:
        if name.startswith("B"):
            raise Ask(f"Address for {name}?")
        return email

    reg = registry_of(read_csv, each_invoice, gate, mail)
    graph = Graph.compile(GraphRecord.model_validate(invoices_then({
        "g": {"node": "gate", "inputs": {"name": each_field("name"), "email": each_field("email")}},
        "m": {"node": "mail", "inputs": {"email": edge_from("g")}},
    })), reg)
    run, _ = calls(lambda: graph.run(text=CSV))
    run, _ = calls(run.answer("g@1", "bo@b.dk").resume)
    paused, called = calls(lambda: run.again(text=CSV.replace("Bo,", "Bob,")))
    assert [place(w) for w in paused.waiting] == ["g@1"] and called == []
    resumed, called = calls(lambda: Run.from_json(paused.to_json(), graph).answer("g@1", "bo@b.dk").resume())
    assert resumed.status == "complete" and called == [], called  # the mail to Bo's address went out once
