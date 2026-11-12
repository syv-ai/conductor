"""The core's rules through the public door: registry, ids, reuse, answers, drawings, items, embedding, upgrades.

The invoice graph reads a CSV, asks a person to approve each big amount, and
sums the decisions; ``calls`` records which nodes were called, so a test can
say what ran again and what was reused.
"""
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, TypedDict
from unittest import mock

import pytest
from conductor import (
    SKIPPED,
    Ask,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeDeclarationError,
    NodeRegistry,
    NotCalled,
    Param,
    Policy,
    Run,
    Series,
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
INVOICES = {"nodes": [{"id": "read", "node": "read-csv"}, {"id": "each", "node": "each-invoice"}], "edges": [["read.result", "each.table"]]}


def invoices_then(*nodes: dict, edges: list[list[str]]) -> dict:
    """The record of ``INVOICES`` followed by ``nodes``, wired by ``edges``."""
    return {"nodes": [*INVOICES["nodes"], *nodes], "edges": [*INVOICES["edges"], *edges]}


def calls(fn):
    """What ``fn`` returns, and the calls the nodes noted while it ran."""
    with tempfile.TemporaryDirectory() as work, mock.patch.dict(os.environ, {"CALLS": str(Path(work) / "calls")}):
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
    assert [n.id for n in record.nodes] == ["upper-1", "tail", "upper-2"]
    graph = Graph.compile(record.model_dump_json(), registry)
    assert graph.run(text="hi").outputs == {"upper-2.result": "HI."}


def test_per_item_reuse_on_again():
    graph = Graph.compile(GraphRecord.model_validate(invoices_then({"id": "mail", "node": "mail"}, edges=[["each.result.email", "mail.email"]])), invoicing)
    first, before = calls(lambda: graph.run(text=CSV))
    second, after = calls(lambda: first.again(text=CSV.replace("bo@b.dk", "bo@new.dk") + "Eva,eva@e.dk,1\n"))
    assert len(before) == 3 and after == ["mail bo@new.dk", "mail eva@e.dk"], after
    assert second.results["mail"]["result"][1] == "sent to bo@new.dk"


def test_answer_stays_with_its_item_on_again():
    graph = Graph.compile(GraphRecord.model_validate(invoices_then(
        {"id": "approve", "node": "approve"}, {"id": "summary", "node": "summary"},
        edges=[["each.result.name", "approve.name"], ["each.result.amount", "approve.amount"], ["approve.result", "summary.decisions"]],
    )), invoicing)
    run, _ = calls(lambda: graph.run(text=CSV))
    for unit in run.waiting:
        run = run.answer(unit, "yes")
    run, _ = calls(run.resume)
    edited = CSV.replace("Ida,ida@a.dk,100", "Ida,ida@a.dk,200").replace("Al,al@c.dk,7000", "Al,al@c.dk,8000")
    again, called = calls(lambda: run.again(text=edited))
    assert called == ["approve Ida", "approve Al"], called  # Bo's answer stays with Bo
    assert again.status == "waiting" and [w.inputs["name"] for w in again.waiting] == ["Al"]
    try:
        again.answer(again.waiting[0], "maybe")
        raise AssertionError
    except InvalidCallError as error:
        assert error.errors()[0].type == "invalid_answer"


def test_half_finished_drawing():
    drawing = invoices_then(
        {"id": "approve", "node": "approve"}, {"id": "mystery", "node": "gone"}, {"id": "rep", "node": "repeat", "constants": {"times": -1}},
        edges=[["each.result.name", "approve.name"], ["each.result.nope", "rep.text"], ["ghost.result", "approve.amount"]],
    )
    graph = Graph.compile(GraphRecord.model_validate(drawing), invoicing)
    found = sorted((p.type, p.loc) for p in graph.problems)
    assert found == [("edge_names_missing_node", ("approve", "amount")), ("invalid_value", ("rep", "times")), ("unknown_node", ("mystery",)),
                     ("unknown_output", ("rep", "text"))], found
    try:
        graph.run(text=CSV)
        raise AssertionError
    except InvalidCallError as error:
        assert len(error.errors()) == 4
    open_first = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "a", "node": "upper"}]}), registry)
    assert open_first.problems == [] and list(open_first.inputs) == ["text"]
    try:
        open_first.run()
        raise AssertionError
    except InvalidCallError as error:
        assert error.errors()[0].type == "missing_input"


def test_every_fault_of_one_node_is_said_at_once():
    later = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "repeat"}], "edges": [["a.result", "b.times"]]}), registry)
    assert [(p.type, p.loc) for p in later.problems] == [("unbound_on_later_node", ("b", "text")), ("type_mismatch", ("b", "times"))]
    unbound = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "join"}, {"id": "c", "node": "repeat"}],
                             "edges": [["a.result", "b.texts"], ["b.result", "c.times"]]}), registry)
    assert "type_mismatch" in [p.type for p in unbound.problems]


def test_nested_series_gather_one_level_at_a_time():
    # documents -> lines -> per-line upper -> per-document join; the lines read whole are a list per document too.
    record = {"nodes": [{"id": "docs", "node": "split", "constants": {"sep": ";"}}, {"id": "lines", "node": "split"},
                        {"id": "up", "node": "upper"}, {"id": "per_doc", "node": "join"}, {"id": "all", "node": "join"}],
              "edges": [["docs.result", "lines.text"], ["lines.result", "up.text"], ["up.result", "per_doc.texts"], ["lines.result", "all.texts"]]}
    graph = Graph.compile(GraphRecord.model_validate(record), registry)
    assert {k: (v.mode, v.series) for k, v in graph.runs_on.items()} == {
        "docs": ("once", None), "lines": ("per_item", "docs.result"), "up": ("per_item", "lines.result"),
        "per_doc": ("per_item", "docs.result"), "all": ("per_item", "docs.result"),
    }
    run = graph.run(text="a,b;c;")
    assert run.results["per_doc"]["result"] == ["A+B", "C", ""], run.results["per_doc"]
    assert run.results["all"]["result"] == ["a+b", "c", ""]  # a Series input sits where its Series was made


def test_items_of_two_series_at_one_node_are_refused():
    record = {"nodes": [{"id": "a", "node": "split"}, {"id": "b", "node": "split"}, {"id": "r", "node": "repeat"}],
              "edges": [["a.result", "r.text"], ["b.result", "r.times"]]}
    graph = Graph.compile(GraphRecord.model_validate(record), registry)
    assert [p.type for p in graph.problems] == ["type_mismatch"]
    record["nodes"][2] = {"id": "r", "node": "pair"}

    @node("pair", title="Pair", description="For a test.")
    def pair(a: str, b: str) -> str:
        return a + b

    graph = Graph.compile(GraphRecord.model_validate({"nodes": record["nodes"], "edges": [["a.result", "r.a"], ["b.result", "r.b"]]}), registry_of(split, pair))
    assert [(p.type, p.ctx["series"]) for p in graph.problems] == [("series_not_nested", ["a.result", "b.result"])], graph.problems
    assert "wire 'a.result' into 'b'" in graph.problems[0].msg


def test_branches_and_blanks():
    @node("pick", title="Pick", description="For a test.")
    def pick(long: str | None = None, short: str | None = None) -> str:
        return long or short

    reg = registry_of(split, route, upper, join, pick, mail)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "s", "node": "split"}, {"id": "r", "node": "route"}, {"id": "L", "node": "upper"}, {"id": "J", "node": "join"},
                                {"id": "both", "node": "join"}],
                      "edges": [["s.result", "r.text"], ["r.long", "L.text"], ["L.result", "J.texts"], ["L.result", "both.texts"], ["r.short", "both.texts"]]}), reg)
    run = graph.run(text="abcd,ab,xyzzy")
    assert run.results["L"]["result"] == ["ABCD", None, "XYZZY"]
    assert run.results["J"]["result"] == "ABCD+XYZZY"
    assert run.results["both"]["result"] == "ABCD+ab+XYZZY"  # opposite branches of one decision gather in item order
    blank = Graph.compile(GraphRecord.model_validate(invoices_then({"id": "mail", "node": "mail"}, edges=[["each.result.email", "mail.email"]])), registry_of(read_csv, each_invoice, mail))
    run = blank.run(text="name,email,amount\nIda,ida@a.dk,1\nBo,,2\n")
    assert run.status == "complete" and run.results["mail"]["result"] == ["sent to ida@a.dk", None]
    assert run.not_called == [NotCalled(loc=("mail", 1), input="email")]


def test_one_value_gathers_as_a_series_of_one_and_a_list_given_for_one_value_is_refused():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "up", "node": "upper"}, {"id": "j", "node": "join"}], "edges": [["up.result", "j.texts"]]}), registry)
    assert graph.run(text="a").outputs == {"j.result": "A"}
    with pytest.raises(InvalidCallError) as refused:
        graph.run(text=["a", "b"])
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("invalid_input", ("inputs", "text"))]


def test_max_items():
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "s", "node": "split"}, {"id": "m", "node": "mail"}], "edges": [["s.result", "m.email"]]}), registry)
    (run, called) = calls(lambda: graph.run(text=",".join(f"x{i}" for i in range(50)), max_items=10))
    assert run.status == "stopped" and run.state.cause.type == "item_ceiling" and called == []
    assert run.state.cause.ctx == {"series": "s.result", "items": 50, "max_items": 10}
    saved = Run.from_json(run.to_json(), graph)
    (done, called) = calls(lambda: saved.with_max_items(100).resume())
    assert done.status == "complete" and len(called) == 50


def test_embedded_graph_and_versions():
    """The second version keeps the id of its last node, so what read ``e.tail.result`` still does."""
    enrich_v1 = make_graph(upper, tail)
    enrich_v2 = {"nodes": [{"id": "lower", "node": "lower"}, {"id": "dot", "node": "tail"}, {"id": "tail", "node": "tail"}], "edges": [["lower.result", "dot.text"], ["dot.result", "tail.text"]]}
    v1 = subgraph("enrich", description="For a test.", title="Enrich")(enrich_v1)
    v2 = subgraph("enrich", title="Enrich", description="For a test.", replaces=v1)(GraphRecord.model_validate(enrich_v2))
    reg = registry_of(upper, lower, tail, v2)
    outer = GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich", "version": 1}, {"id": "u", "node": "upper"}], "edges": [["e.tail.result", "u.text"]]})
    assert Graph.compile(outer, reg).run(text="Hi").outputs == {"u.result": "HI."}
    newer = reg.upgrade(outer, "e")
    assert newer.nodes[0].version == 2 and Graph.compile(newer, reg).run(text="Hi").outputs == {"u.result": "HI.."}
    graph = Graph.compile(outer, reg)
    assert set(graph._nodes) == {"e", "u"}


def test_an_upgrade_moves_an_edge_reading_an_embeddings_output_to_the_address_the_newer_version_gives_it():
    v1 = subgraph("enrich", description="For a test.", title="Enrich")(make_graph(upper, tail))
    v2 = subgraph("enrich", title="Enrich", description="For a test.", replaces=v1, upgrade=Upgrade(renamed_outputs={"tail.result": "tail-2.result"}))(make_graph(lower, tail, tail))
    reg = registry_of(upper, lower, tail, v2)
    outer = GraphRecord.model_validate({"nodes": [{"id": "e", "node": "enrich", "version": 1}, {"id": "u", "node": "upper"}], "edges": [["e.tail.result", "u.text"]]})
    newer = reg.upgrade(outer, "e")
    assert newer.edges == (("e.tail-2.result", "u.text"),) and Graph.compile(newer, reg).run(text="Hi").outputs == {"u.result": "HI.."}


def test_upgrade_renames():
    @node("greet", title="Greet v1", description="For a test.")
    def greet_v1(name: str) -> str:
        return f"hello {name}"

    @node("greet", title="Greet", description="For a test.", replaces=greet_v1, upgrade=Upgrade(renamed_inputs={"name": "who"}))
    def greet(who: str, punct: str = "!") -> str:
        return f"hello {who}{punct}"

    reg = registry_of(greet, upper)
    stored = GraphRecord.model_validate({"nodes": [{"id": "u", "node": "upper"}, {"id": "g", "node": "greet", "constants": {}}], "edges": [["u.result", "g.name"]]})
    assert Graph.compile(stored, reg).run(text="ida").outputs == {"g.result": "hello IDA"}
    moved = reg.upgrade(stored, "g")
    assert moved.edges == (("u.result", "g.who"),) and Graph.compile(moved, reg).run(text="ida").outputs == {"g.result": "hello IDA!"}


def test_ask_forms_and_several():
    @dataclass(frozen=True)
    class Review:
        approved: bool
        note: str

    @node("review", title="Review", description="For a test.")
    def review(name: str) -> Review:
        raise Ask(f"Review {name}", default={"approved": True, "note": ""})

    @node("tags", title="Tags", description="For a test.")
    def tags(name: str) -> Series[str]:
        raise Ask(f"Tags for {name}", choices=["red", "green", "blue"])

    @node("pick", title="Pick", description="For a test.")
    def pick(name: str) -> bool:
        raise Ask(f"Keep {name}?")

    reg = registry_of(split, review, tags, pick, join)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "s", "node": "split"}, {"id": "r", "node": "review"}, {"id": "t", "node": "tags"}, {"id": "p", "node": "pick"}, {"id": "j", "node": "join"}],
                      "edges": [["s.result", "r.name"], ["s.result", "t.name"], ["s.result", "p.name"], ["t.result", "j.texts"]]}), reg)
    run = graph.run(text="Ida,Bo")
    by = {place(w): w for w in run.waiting}
    approved, note = by["r@0"].questions
    assert (approved.name, approved.json_schema, approved.default, note.default) == ("approved", {"type": "boolean"}, True, "")
    assert by["p@1"].questions[0].json_schema == {"type": "boolean"} and by["t@0"].questions[0].many
    run = Run.from_json(run.to_json(), graph)
    run = run.answer("r@0", {"approved": False, "note": "late"}).answer("r@1", {"approved": True, "note": ""})
    run = run.answer("t@0", ["red", "blue"]).answer("t@1", []).answer("p@0", True).answer("p@1", False).resume()
    assert run.status == "complete", run.status
    assert run.results["r"]["approved"] == [False, True] and run.results["j"]["result"] == ["red+blue", ""]


def test_pause_on_a_nested_item():
    @node("check", title="Check", description="For a test.")
    def check(text: str) -> str:
        if text == "b":
            raise Ask("b?")
        return text

    reg = registry_of(split, check, join)
    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "docs", "node": "split", "constants": {"sep": ";"}}, {"id": "lines", "node": "split"}, {"id": "c", "node": "check"},
                                {"id": "j", "node": "join"}], "edges": [["docs.result", "lines.text"], ["lines.result", "c.text"], ["c.result", "j.texts"]]}), reg)
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

    graph = Graph.compile(GraphRecord.model_validate({"nodes": [{"id": "s", "node": "split"}, {"id": "w", "node": "slow"}], "edges": [["s.result", "w.text"]]}), registry_of(split, slow))
    start = t.monotonic()
    run = graph.run(text=",".join("abcdefgh"))
    assert t.monotonic() - start < 0.2 and len(seen) > 1 and run.results["w"]["result"] == list("abcdefgh")


def test_again_paused_saved_and_resumed_keeps_earlier_items():
    @node("gate", title="Gate", description="For a test.")
    def gate(name: str, email: str) -> str:
        if name.startswith("B"):
            raise Ask(f"Address for {name}?")
        return email

    reg = registry_of(read_csv, each_invoice, gate, mail)
    graph = Graph.compile(GraphRecord.model_validate(invoices_then(
        {"id": "g", "node": "gate"}, {"id": "m", "node": "mail"},
        edges=[["each.result.name", "g.name"], ["each.result.email", "g.email"], ["g.result", "m.email"]],
    )), reg)
    run, _ = calls(lambda: graph.run(text=CSV))
    run, _ = calls(run.answer("g@1", "bo@b.dk").resume)
    paused, called = calls(lambda: run.again(text=CSV.replace("Bo,", "Bob,")))
    assert [place(w) for w in paused.waiting] == ["g@1"] and called == []
    resumed, called = calls(lambda: Run.from_json(paused.to_json(), graph).answer("g@1", "bo@b.dk").resume())
    assert resumed.status == "complete" and called == [], called  # the mail to Bo's address went out once
