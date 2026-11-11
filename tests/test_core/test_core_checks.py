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
    InvalidCallError,
    NodeDeclarationError,
    NodeRegistry,
    NotCalled,
    Param,
    Policy,
    Series,
    Table,
    Upgrade,
    make_graph,
    node,
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


@node("each-invoice", title="Each invoice", description="")
def each_invoice(table: Table[InvoiceLine]) -> Series[InvoiceLine]:
    return table.to_dicts()


@dataclass(frozen=True)
class Length:
    long: Annotated[str, Param(exclusive_group="length")] = SKIPPED
    short: Annotated[str, Param(exclusive_group="length")] = SKIPPED


@node("route", title="Route", description="")
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
    graph = registry.load(record.model_dump_json())
    assert graph.run(text="hi").outputs == {"upper-2.result": "HI."}


def test_per_item_reuse_on_again():
    graph = invoicing.load(invoices_then({"id": "mail", "node": "mail"}, edges=[["each.result.email", "mail.email"]]))
    first, before = calls(lambda: graph.run(text=CSV))
    second, after = calls(lambda: first.again(text=CSV.replace("bo@b.dk", "bo@new.dk") + "Eva,eva@e.dk,1\n"))
    assert len(before) == 3 and after == ["mail bo@new.dk", "mail eva@e.dk"], after
    assert second.results["mail"]["result"][1] == "sent to bo@new.dk"


def test_answer_stays_with_its_item_on_again():
    graph = invoicing.load(invoices_then(
        {"id": "approve", "node": "approve"}, {"id": "summary", "node": "summary"},
        edges=[["each.result.name", "approve.name"], ["each.result.amount", "approve.amount"], ["approve.result", "summary.decisions"]],
    ))
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
        {"id": "approve", "node": "approve"}, {"id": "mystery", "node": "gone"}, {"id": "rep", "node": "repeat", "values": {"times": -1}},
        edges=[["each.result.name", "approve.name"], ["each.result.nope", "rep.text"], ["ghost.result", "approve.amount"]],
    )
    graph = invoicing.load(drawing)
    found = sorted((p.type, p.loc) for p in graph.problems)
    assert found == [("edge_names_missing_node", ("approve", "amount")), ("invalid_value", ("rep", "times")), ("unknown_node", ("mystery",)),
                     ("unknown_output", ("rep", "text"))], found
    try:
        graph.run(text=CSV)
        raise AssertionError
    except InvalidCallError as error:
        assert len(error.errors()) == 4
    open_first = registry.load({"nodes": [{"id": "a", "node": "upper"}]})
    assert open_first.problems == [] and list(open_first.inputs) == ["text"]
    try:
        open_first.run()
        raise AssertionError
    except InvalidCallError as error:
        assert error.errors()[0].type == "missing_input"


@pytest.mark.xfail(reason="engine fault: loading stops at a node's first input it can't plan, so a second fault on the node goes unsaid", strict=True)
def test_every_fault_of_one_node_is_said_at_once():
    later = registry.load({"nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "repeat"}], "edges": [["a.result", "b.times"]]})
    assert [(p.type, p.loc) for p in later.problems] == [("unbound_on_later_node", ("b", "text")), ("type_mismatch", ("b", "times"))]
    unbound = registry.load({"nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "join"}, {"id": "c", "node": "repeat"}],
                             "edges": [["a.result", "b.texts"], ["b.result", "c.times"]]})
    assert "type_mismatch" in [p.type for p in unbound.problems]


def test_nested_series_gather_one_level_at_a_time():
    # documents -> lines -> per-line upper -> per-document join; the lines read whole are a list per document too.
    record = {"nodes": [{"id": "docs", "node": "split", "values": {"sep": ";"}}, {"id": "lines", "node": "split"},
                        {"id": "up", "node": "upper"}, {"id": "per_doc", "node": "join"}, {"id": "all", "node": "join"}],
              "edges": [["docs.result", "lines.text"], ["lines.result", "up.text"], ["up.result", "per_doc.texts"], ["lines.result", "all.texts"]]}
    graph = registry.load(record)
    assert {k: (v.mode, v.series) for k, v in graph.plan.items()} == {
        "docs": ("once", None), "lines": ("per_item", "docs.result"), "up": ("per_item", "lines.result"),
        "per_doc": ("per_item", "docs.result"), "all": ("per_item", "docs.result"),
    }
    run = graph.run(text="a,b;c;")
    assert run.results["per_doc"]["result"] == ["A+B", "C", ""], run.results["per_doc"]
    assert run.results["all"]["result"] == ["a+b", "c", ""]  # a Series input sits where its Series was made


def test_items_of_two_series_at_one_node_are_refused():
    record = {"nodes": [{"id": "a", "node": "split"}, {"id": "b", "node": "split"}, {"id": "r", "node": "repeat"}],
              "edges": [["a.result", "r.text"], ["b.result", "r.times"]]}
    graph = registry.load(record)
    assert [p.type for p in graph.problems] == ["type_mismatch"]
    record["nodes"][2] = {"id": "r", "node": "pair"}

    @node("pair", title="Pair", description="")
    def pair(a: str, b: str) -> str:
        return a + b

    graph = registry_of(split, pair).load({"nodes": record["nodes"], "edges": [["a.result", "r.a"], ["b.result", "r.b"]]})
    assert [(p.type, p.ctx["series"]) for p in graph.problems] == [("series_not_nested", ["a.result", "b.result"])], graph.problems
    assert "gather one of them first" in graph.problems[0].msg


def test_branches_and_blanks():
    @node("pick", title="Pick", description="")
    def pick(long: str | None = None, short: str | None = None) -> str:
        return long or short

    reg = registry_of(split, route, upper, join, pick, mail)
    graph = reg.load({"nodes": [{"id": "s", "node": "split"}, {"id": "r", "node": "route"}, {"id": "L", "node": "upper"}, {"id": "J", "node": "join"},
                                {"id": "both", "node": "join"}],
                      "edges": [["s.result", "r.text"], ["r.long", "L.text"], ["L.result", "J.texts"], ["L.result", "both.texts"], ["r.short", "both.texts"]]})
    run = graph.run(text="abcd,ab,xyzzy")
    assert run.results["L"]["result"] == ["ABCD", None, "XYZZY"]
    assert run.results["J"]["result"] == "ABCD+XYZZY"
    assert run.results["both"]["result"] == "ABCD+ab+XYZZY"  # opposite branches of one decision gather in item order
    blank = registry_of(read_csv, each_invoice, mail).load(invoices_then({"id": "mail", "node": "mail"}, edges=[["each.result.email", "mail.email"]]))
    run = blank.run(text="name,email,amount\nIda,ida@a.dk,1\nBo,,2\n")
    assert run.status == "complete" and run.results["mail"]["result"] == ["sent to ida@a.dk", None]
    assert run.not_called == [NotCalled(node_id="mail", item=(1,), input="email")]


def test_one_value_gathers_as_a_series_of_one_and_a_list_given_for_one_value_is_refused():
    graph = registry.load({"nodes": [{"id": "up", "node": "upper"}, {"id": "j", "node": "join"}], "edges": [["up.result", "j.texts"]]})
    assert graph.run(text="a").outputs == {"j.result": "A"}
    with pytest.raises(InvalidCallError) as refused:
        graph.run(text=["a", "b"])
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("invalid_input", ("inputs", "text"))]


def test_max_items():
    graph = registry.load({"nodes": [{"id": "s", "node": "split"}, {"id": "m", "node": "mail"}], "edges": [["s.result", "m.email"]]})
    (run, called) = calls(lambda: graph.run(text=",".join(f"x{i}" for i in range(50)), max_items=10))
    assert run.status == "stopped" and run.state.cause.type == "item_ceiling" and called == []
    assert run.state.cause.ctx == {"series": "s.result", "items": 50, "max_items": 10}
    saved = graph.load_run(run.to_json())
    (done, called) = calls(lambda: saved.resume(max_items=100))
    assert done.status == "complete" and len(called) == 50


@pytest.mark.skip(reason="embedding: nested run not built")
def test_embedded_graph_and_versions():
    enrich_v1 = make_graph(upper, tail)
    enrich_v2 = make_graph(lower, tail, tail)
    v1 = node("enrich", description="", title="Enrich")(enrich_v1)
    v2 = node("enrich", title="Enrich", description="", replaces=v1)(enrich_v2)
    reg = registry_of(upper, lower, tail, v2)
    outer = {"nodes": [{"id": "e", "node": "enrich", "version": 1}, {"id": "u", "node": "upper"}], "edges": [["e.result", "u.text"]]}
    assert reg.load(outer).run(text="Hi").outputs == {"u.result": "HI."}
    newer = reg.upgrade(outer, "e")
    assert newer.nodes[0].version == 2 and reg.load(newer).run(text="Hi").outputs == {"u.result": "HI.."}
    graph = reg.load(outer)
    assert set(graph._nodes) == {"e", "u"}


def test_upgrade_renames():
    @node("greet", title="Greet v1", description="")
    def greet_v1(name: str) -> str:
        return f"hello {name}"

    @node("greet", title="Greet", description="", replaces=greet_v1, upgrade=Upgrade(renamed_inputs={"name": "who"}))
    def greet(who: str, punct: str = "!") -> str:
        return f"hello {who}{punct}"

    reg = registry_of(greet, upper)
    stored = {"nodes": [{"id": "u", "node": "upper"}, {"id": "g", "node": "greet", "values": {}}], "edges": [["u.result", "g.name"]]}
    assert reg.load(stored).run(text="ida").outputs == {"g.result": "hello IDA"}
    moved = reg.upgrade(stored, "g")
    assert moved.edges == (("u.result", "g.who"),) and reg.load(moved).run(text="ida").outputs == {"g.result": "hello IDA!"}


@pytest.mark.skip(reason="embedding: nested run not built")
def test_embedded_graph_runs_per_item():
    enrich = node("enrich", title="Enrich", description="")(make_graph(upper, tail))
    reg = registry_of(split, upper, tail, join, enrich)
    graph = reg.load({"nodes": [{"id": "s", "node": "split"}, {"id": "e", "node": "enrich"}, {"id": "j", "node": "join"}],
                      "edges": [["s.result", "e.text"], ["e.result", "j.texts"]]})
    assert (graph.plan["e"].mode, graph.plan["e"].series) == ("per_item", "s.result") and list(graph.plan) == ["s", "e", "j"]
    assert graph.run(text="a,b").outputs == {"j.result": "A.+B."}


def test_ask_forms_and_several():
    @dataclass(frozen=True)
    class Review:
        approved: bool
        note: str

    @node("review", title="Review", description="")
    def review(name: str) -> Review:
        raise Ask(f"Review {name}", default={"approved": True, "note": ""})

    @node("tags", title="Tags", description="")
    def tags(name: str) -> Series[str]:
        raise Ask(f"Tags for {name}", choices=["red", "green", "blue"])

    @node("pick", title="Pick", description="")
    def pick(name: str) -> bool:
        raise Ask(f"Keep {name}?")

    reg = registry_of(split, review, tags, pick, join)
    graph = reg.load({"nodes": [{"id": "s", "node": "split"}, {"id": "r", "node": "review"}, {"id": "t", "node": "tags"}, {"id": "p", "node": "pick"}, {"id": "j", "node": "join"}],
                      "edges": [["s.result", "r.name"], ["s.result", "t.name"], ["s.result", "p.name"], ["t.result", "j.texts"]]})
    run = graph.run(text="Ida,Bo")
    by = {place(w): w for w in run.waiting}
    approved, note = by["r@0"].questions
    assert (approved.name, approved.json_schema, approved.default, note.default) == ("approved", {"type": "boolean"}, True, "")
    assert by["p@1"].questions[0].json_schema == {"type": "boolean"} and by["t@0"].questions[0].shape == "many"
    run = graph.load_run(run.to_json())
    run = run.answer("r@0", {"approved": False, "note": "late"}).answer("r@1", {"approved": True, "note": ""})
    run = run.answer("t@0", ["red", "blue"]).answer("t@1", []).answer("p@0", True).answer("p@1", False).resume()
    assert run.status == "complete", run.status
    assert run.results["r"]["approved"] == [False, True] and run.results["j"]["result"] == ["red+blue", ""]


def test_pause_on_a_nested_item():
    @node("check", title="Check", description="")
    def check(text: str) -> str:
        if text == "b":
            raise Ask("b?")
        return text

    reg = registry_of(split, check, join)
    graph = reg.load({"nodes": [{"id": "docs", "node": "split", "values": {"sep": ";"}}, {"id": "lines", "node": "split"}, {"id": "c", "node": "check"},
                                {"id": "j", "node": "join"}], "edges": [["docs.result", "lines.text"], ["lines.result", "c.text"], ["c.result", "j.texts"]]})
    run = graph.run(text="a,b;c")
    assert [place(w) for w in run.waiting] == ["c@0@1"] and run.results["j"]["result"] == [None, "c"]
    run = graph.load_run(run.to_json()).answer("c@0@1", "B").resume()
    assert run.results["j"]["result"] == ["a+B", "c"]




def test_concurrency():
    import threading
    import time as t

    seen = set()

    @node("slow", title="Slow", description="", policy=Policy(concurrency=8))
    def slow(text: str) -> str:
        seen.add(threading.get_ident())
        t.sleep(0.05)
        return text

    graph = registry_of(split, slow).load({"nodes": [{"id": "s", "node": "split"}, {"id": "w", "node": "slow"}], "edges": [["s.result", "w.text"]]})
    start = t.monotonic()
    run = graph.run(text=",".join("abcdefgh"))
    assert t.monotonic() - start < 0.2 and len(seen) > 1 and run.results["w"]["result"] == list("abcdefgh")


def test_again_paused_saved_and_resumed_keeps_earlier_items():
    @node("gate", title="Gate", description="")
    def gate(name: str, email: str) -> str:
        if name.startswith("B"):
            raise Ask(f"Address for {name}?")
        return email

    reg = registry_of(read_csv, each_invoice, gate, mail)
    graph = reg.load(invoices_then(
        {"id": "g", "node": "gate"}, {"id": "m", "node": "mail"},
        edges=[["each.result.name", "g.name"], ["each.result.email", "g.email"], ["g.result", "m.email"]],
    ))
    run, _ = calls(lambda: graph.run(text=CSV))
    run, _ = calls(run.answer("g@1", "bo@b.dk").resume)
    paused, called = calls(lambda: run.again(text=CSV.replace("Bo,", "Bob,")))
    assert [place(w) for w in paused.waiting] == ["g@1"] and called == []
    resumed, called = calls(lambda: graph.load_run(paused.to_json()).answer("g@1", "bo@b.dk").resume())
    assert resumed.status == "complete" and called == [], called  # the mail to Bo's address went out once
