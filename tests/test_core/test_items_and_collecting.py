"""Loops, ``None``, outer values, collecting, branches, tables and subgraphs, through the public API.

With ``split -> upper -> join``: ``split`` returns ``["a", "b"]``; ``upper``, whose input takes one ``str``, is inside
the loop over ``split.result`` and is called once per item; ``join``, whose input takes a ``Series[str]``, collects the
loop and is called once with ``["A", "B"]``. A node outside the loop is called once, and its output (an outer value) is
the same on every call inside the loop. A table is one value; a node that splits it returns a ``Series`` of its
records. A node given ``None`` where its input does not take ``None`` is not called; a branch not chosen skips it.
"""
import asyncio
import datetime
import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Event, Timer
from types import MappingProxyType
from typing import Annotated, Any, Literal, Optional, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Address,
    Ask,
    Graph,
    GraphRecord,
    InputEntry,
    InvalidCallError,
    NodeRecord,
    NotCalled,
    Param,
    Policy,
    Run,
    RunsOn,
    Series,
    Table,
    host_type,
    make_graph,
    node,
    subgraph,
)

from tests.test_core.nodes import registry_of

# -- the nodes ---------------------------------------------------------------------------------

CALLED: list[str] = []


class Contact(TypedDict):
    name: str
    email: str | None


class Invoice(TypedDict):
    vendor: str
    amount: float


@node("contacts", title="Contacts", description="``Ida,ida@x;Bo,;Nina,nina@x``: a contact per ``;``, the email ``None`` where nothing follows the comma.")
def contacts(text: str) -> Table[Contact]:
    """``Ida,ida@x;Bo,;Nina,nina@x``: a contact per ``;``, the email ``None`` where nothing follows the comma."""
    rows = [part.split(",") for part in text.split(";") if part]
    return pl.DataFrame({"name": [r[0] for r in rows], "email": [r[1] or None for r in rows]}, schema={"name": pl.String, "email": pl.String})


@node("records", title="Records", description="A table of contacts, split into its records.")
def records(table: Table[Contact]) -> Series[Contact]:
    return table.to_dicts()


@node("split", title="Split", description="For a test.")
def split(text: str, sep: str = ",") -> Series[str]:
    return [part for part in text.split(sep)] if text else []


@node("upper", title="Upper", description="For a test.")
def upper(text: str) -> str:
    CALLED.append(f"upper {text}")
    return text.upper()


@node("join", title="Join", description="For a test.")
def join(texts: Series[str]) -> str:
    CALLED.append(f"join {texts}")
    return "+".join(texts)


@node("count", title="Count", description="For a test.")
def count(values: Series[Any]) -> int:
    CALLED.append(f"count {len(values)}")
    return len(values)


@node("pair", title="Pair", description="For a test.")
def pair(a: str, b: str) -> str:
    return f"{a}:{b}"


@node("send", title="Send", description="For a test.")
def send(email: str) -> str:
    CALLED.append(f"send {email}")
    return f"sent to {email}"


@node("send-or-say", title="Send or say", description="For a test.")
def send_or_say(email: str | None) -> str:
    return f"sent to {email}" if email else "no address"


@node("show", title="Show", description="For a test.")
def show(table: Table) -> Table:
    return table


@dataclass(frozen=True)
class YesNo:
    yes: Annotated[str, Param(exclusive_group="answer")]
    no: Annotated[str, Param(exclusive_group="answer")]


@node("decide", title="Decide", description="``yes`` unless the text has ``no`` in it.")
def decide(text: str) -> YesNo:
    """``yes`` unless the text has ``no`` in it."""
    return YesNo(SKIPPED, text) if "no" in text else YesNo(text, SKIPPED)


@node("docs", title="Docs", description="Documents, one per ``|``.")
def docs(text: str) -> Series[str]:
    """Documents, one per ``|``."""
    return text.split("|")


@node("invoices", title="Invoices", description="``acme:10,beta:5``: an invoice per ``,``; ``skip`` returns ``SKIPPED``.")
def invoices(text: str) -> Table[Invoice]:
    """``acme:10,beta:5``: an invoice per ``,``; ``skip`` returns ``SKIPPED``."""
    if text == "skip":
        return SKIPPED
    rows = [part.split(":") for part in text.split(",") if part]
    return pl.DataFrame({"vendor": [r[0] for r in rows], "amount": [float(r[1]) for r in rows]}, schema={"vendor": pl.String, "amount": pl.Float64})


@node("invoice-lines", title="Invoice lines", description="A table of invoices, split into its records.")
def invoice_lines(table: Table[Invoice]) -> Series[Invoice]:
    return table.to_dicts()


@node("check", title="Check", description="For a test.")
def check(vendor: str) -> str:
    return SKIPPED if vendor == "unknown" else f"{vendor} ok"


@node("total", title="Total", description="For a test.")
def total(amounts: Series[float]) -> float:
    return sum(amounts)


REGISTRY = registry_of(contacts, records, split, upper, join, count, pair, send, send_or_say, show, decide, docs, invoices, invoice_lines, check, total)

CSV = "Ida,ida@x;Bo,;Nina,nina@x"
CONTACTS = ({"id": "c", "node": "contacts"}, {"id": "r", "node": "records"})
SPLIT_CONTACTS = ("c.result", "r.table")


def record_of(*nodes: dict[str, Any], edges: tuple[tuple[str, str], ...] = (), registry: Any = REGISTRY,
              inputs: Mapping[str, str | list[str]] = MappingProxyType({}), outputs: Mapping[str, str] = MappingProxyType({})) -> GraphRecord:
    """``make_graph`` of ``nodes`` (``{"id", "node", "constants"?}``, each node's definition from ``registry``),
    ``edges`` (``(output, input)``) and the graph's ``inputs`` and ``outputs`` where a test names them."""
    return make_graph(
        *((n["id"], registry[n["node"]]) for n in nodes),
        edges=list(edges),
        constants={f"{n['id']}.{name}": value for n in nodes for name, value in n.get("constants", {}).items()},
        inputs=inputs,
        outputs=outputs,
    )


def graph(*nodes: dict[str, Any], edges: tuple[tuple[str, str], ...] = (), registry: Any = REGISTRY, **ports: Any) -> Any:
    """The compiled graph of ``record_of(nodes, edges, inputs, outputs)``, problems or not."""
    return Graph.compile(record_of(*nodes, edges=edges, registry=registry, **ports), registry)


def ran(*nodes: dict[str, Any], edges: tuple[tuple[str, str], ...] = (), registry: Any = REGISTRY,
        ports: dict[str, Any] | None = None, **given: Any) -> Any:
    """The run of the graph, given ``given``, which must compile with no problems and complete. ``ports`` are the
    ``inputs`` and ``outputs`` of ``record_of``."""
    compiled = graph(*nodes, edges=edges, registry=registry, **(ports or {}))
    assert compiled.problems == [], compiled.problems
    run = compiled.run(**given)
    assert run.status == "complete", run.cause
    return run


def problems(*nodes: dict[str, Any], edges: tuple[tuple[str, str], ...] = (), registry: Any = REGISTRY, **ports: Any) -> list[str]:
    return [p.type for p in graph(*nodes, edges=edges, registry=registry, **ports).problems]


def through_json(run: Any) -> Any:
    return Run.from_json(run.to_json(), run.graph)


def streamed(run: Any) -> list[Any]:
    async def collect() -> list[Any]:
        return [event async for event in run.stream()]

    return asyncio.run(collect())


# -- None --------------------------------------------------------------------------------------


def test_a_node_with_no_outputs_that_returns_skipped_is_complete_not_skipped():
    @node("log", title="Log", description="For a test.")
    def log(text: str) -> None:
        return SKIPPED

    said = streamed(graph({"id": "l", "node": "log"}, registry=registry_of(log)).with_inputs(text="x"))
    assert [(e.type, e.outcomes) for e in said if e.type in ("node_complete", "node_skipped")] == [("node_complete", 1)]


def test_an_optional_series_input_given_a_none_constant_is_called_as_if_left_out():
    @node("maybe", title="Maybe", description="For a test.")
    def maybe(xs: Series[str] | None = None) -> str:
        return str(xs)

    reg = registry_of(maybe)
    set_to_none = ran({"id": "m", "node": "maybe", "constants": {"xs": None}}, registry=reg)
    left_out = ran({"id": "m", "node": "maybe"}, registry=reg)
    assert set_to_none.outputs == left_out.outputs == {"result": "None"}


class Amount(TypedDict):
    a: int


def test_an_optional_table_input_gets_none_from_the_caller_from_a_return_and_from_a_node_not_called():
    @node("count-rows", title="Count rows", description="For a test.")
    def count_rows(t: Table[Amount] | None = None) -> int:
        return -1 if t is None else t.height

    @node("maybe-table", title="Maybe table", description="For a test.")
    def maybe_table(x: int) -> Table[Amount] | None:
        return None if x < 0 else pl.DataFrame({"a": [x]})

    @node("always-none", title="Always none", description="For a test.")
    def always_none(y: int) -> int | None:
        return None

    from_the_caller = graph({"id": "c", "node": "count-rows"}, registry=registry_of(count_rows), inputs={"t": "c.t"})
    alone = graph({"id": "c", "node": "count-rows"}, registry=registry_of(count_rows))
    assert from_the_caller.run(t=None).outputs == alone.run().outputs == {"result": -1}

    from_a_node = graph({"id": "m", "node": "maybe-table"}, {"id": "c", "node": "count-rows"}, edges=(("m.result", "c.t"),), registry=registry_of(maybe_table, count_rows))
    assert from_a_node.run(x=-1).outputs == {"result": -1} and from_a_node.run(x=3).outputs == {"result": 1}
    assert Run.from_json(from_a_node.run(x=-1).to_json(), from_a_node).results["m"] == {"result": None}

    never = graph({"id": "b", "node": "always-none"}, {"id": "m", "node": "maybe-table"}, {"id": "c", "node": "count-rows"},
                  edges=(("b.result", "m.x"), ("m.result", "c.t")), registry=registry_of(always_none, maybe_table, count_rows))
    assert never.run(y=1).outputs == {"result": -1}


def test_an_optional_series_input_gets_none_from_a_return_and_from_a_node_not_called():
    @node("count-items", title="Count items", description="For a test.")
    def count_items(xs: Series[int] | None = None) -> int:
        return -1 if xs is None else len(xs)

    @node("maybe-items", title="Maybe items", description="For a test.")
    def maybe_items(x: int) -> Series[int] | None:
        return None if x < 0 else [x]

    @node("always-none", title="Always none", description="For a test.")
    def always_none(y: int) -> int | None:
        return None

    from_a_node = graph({"id": "m", "node": "maybe-items"}, {"id": "c", "node": "count-items"}, edges=(("m.result", "c.xs"),), registry=registry_of(maybe_items, count_items))
    assert from_a_node.run(x=-1).outputs == {"result": -1} and from_a_node.run(x=3).outputs == {"result": 1}

    never = graph({"id": "b", "node": "always-none"}, {"id": "m", "node": "maybe-items"}, {"id": "c", "node": "count-items"},
                  edges=(("b.result", "m.x"), ("m.result", "c.xs")), registry=registry_of(always_none, maybe_items, count_items))
    assert never.run(y=1).outputs == {"result": -1}


@pytest.mark.parametrize(("annotation", "takes_none"), [
    (str, False),
    (str | None, True),
    (Optional[str], True),  # noqa: UP045  Optional[str] is str | None
    (Annotated[str | None, Param(title="X")], True),
    (Annotated[str, Param(title="X")] | None, True),
    (Any, True),
    (Series[str], False),
    (Series[str | None], True),
    (Series[Any], True),
])
def test_an_input_takes_none_when_its_type_hint_allows_none(annotation, takes_none):
    def seen(x):
        return str(x)

    seen.__annotations__ = {"x": annotation, "return": str}
    reader = node("seen", title="Seen", description="For a test.")(seen)
    run = ran(*CONTACTS, {"id": "s", "node": "seen"}, edges=(SPLIT_CONTACTS, ("r.result.email", "s.x")),
              registry=registry_of(contacts, records, reader), text=CSV)
    got = run.results["s"]["result"]
    if isinstance(got, list):  # one value per contact
        assert got[1] == ("None" if takes_none else None), got
    else:  # one value collecting every contact
        assert got == ("['ida@x', None, 'nina@x']" if takes_none else "['ida@x', 'nina@x']"), got


def test_a_node_whose_input_takes_none_is_called_with_none():
    run = ran(*CONTACTS, {"id": "s", "node": "send-or-say"}, edges=(SPLIT_CONTACTS, ("r.result.email", "s.email")), text=CSV)
    assert run.results["s"]["result"] == ["sent to ida@x", "no address", "sent to nina@x"]
    assert run.not_called == []


def test_a_default_is_for_an_unwired_input_never_in_place_of_a_none_from_an_edge():
    @node("defaulted", title="Defaulted", description="For a test.")
    def defaulted(email: str = "fallback") -> str:
        return email

    run = ran(*CONTACTS, {"id": "d", "node": "defaulted"}, edges=(SPLIT_CONTACTS, ("r.result.email", "d.email")),
              registry=registry_of(contacts, records, defaulted), text=CSV)
    assert run.results["d"]["result"] == ["ida@x", None, "nina@x"]
    assert run.not_called == [NotCalled(loc=("d", 1), input="email")]


def test_a_node_not_called_for_none_is_complete_and_never_skipped():
    @node("nothing", title="Nothing", description="For a test.")
    def nothing() -> str | None:
        return None

    compiled = graph({"id": "n", "node": "nothing"}, {"id": "s", "node": "send"}, edges=(("n.result", "s.email"),), registry=registry_of(nothing, send))

    said = streamed(compiled.with_inputs())
    assert not [e for e in said if e.type == "node_skipped"]
    (complete,) = [e for e in said if e.type == "node_complete" and e.loc[0] == "s"]
    assert (complete.outcomes, complete.not_called) == (1, 1)
    assert said[-1].status == "complete" and said[-1].run.not_called == [NotCalled(loc=("s",), input="email")]


@pytest.mark.parametrize("none_into", ["a", "b"])
def test_a_skipped_input_wins_over_a_none_input_whichever_parameter_comes_first(none_into):
    """One input skipped and the other ``None``: the node is skipped, not "not called", whichever parameter is first."""
    @node("nothing", title="Nothing", description="For a test.")
    def nothing(text: str) -> str | None:
        return None

    branch_into = "b" if none_into == "a" else "a"
    run = ran({"id": "d", "node": "decide"}, {"id": "n", "node": "nothing"}, {"id": "p", "node": "pair"}, {"id": "u", "node": "upper"},
              edges=(("d.no", f"p.{branch_into}"), ("n.result", f"p.{none_into}"), ("p.result", "u.text"), ("d.yes", "u.text")),
              registry=registry_of(decide, nothing, pair, upper), ports={"inputs": {"decided": "d.text", "nothing": "n.text"}}, decided="ok", nothing="z")
    assert run.results["u"]["result"] == "OK"
    assert run.not_called == []


@pytest.mark.parametrize("collecting_first", [False, True])
@pytest.mark.parametrize(("one_reads", "many_reads", "outcome"), [
    ("skipped", "none", "skipped"), ("none", "skipped", "skipped"), ("none", "none", "not called"),
])
def test_an_input_read_from_one_edge_and_a_collecting_input_keep_the_order_a_skip_and_a_none_are_seen_in(collecting_first, one_reads, many_reads, outcome):
    """``mixed`` takes ``one`` through one edge and ``many`` collected over the loop over ``docs.result``, in both
    orders of its signature. ``many`` is ``None`` or skipped because the node in the loop, ``pair``, reads ``None`` or
    a branch not chosen from outside the loop. A skip wins over a ``None`` whichever input has it; two ``None``
    arguments name the first in the signature."""
    @node("nothing", title="Nothing", description="For a test.")
    def nothing(text: str) -> str | None:
        return None

    if collecting_first:
        @node("mixed", title="Mixed", description="For a test.")
        def mixed(many: Series[str], one: str) -> str:
            return one + "".join(many)
    else:
        @node("mixed", title="Mixed", description="For a test.")
        def mixed(one: str, many: Series[str]) -> str:
            return one + "".join(many)

    sources = {"skipped": ("decide", "no"), "none": ("nothing", "result")}
    (one_node, one_output), (outside_node, outside_output) = sources[one_reads], sources[many_reads]
    run = ran({"id": "o", "node": one_node}, {"id": "x", "node": outside_node}, {"id": "s", "node": "docs"}, {"id": "p", "node": "pair"},
              {"id": "m", "node": "mixed"},
              edges=((f"o.{one_output}", "m.one"), ("s.result", "p.a"), (f"x.{outside_output}", "p.b"), ("p.result", "m.many")),
              registry=registry_of(decide, nothing, docs, pair, mixed), ports={"inputs": {"text": ["o.text", "x.text"], "docs": "s.text"}}, text="ok", docs="a|b")
    first = "many" if collecting_first else "one"
    states = json.loads(run.to_json())["outcomes"]["m"]["states"]  # a call not made for a None is saved as no state
    not_called = [n for n in run.not_called if n.loc == ("m",)]
    assert (states, not_called) == ((["skipped"], []) if outcome == "skipped" else ([None], [NotCalled(loc=("m",), input=first)]))


@pytest.mark.parametrize("waiting_into", ["a", "b"])
def test_a_skipped_input_skips_the_node_at_once_while_its_other_input_waits_for_a_person(waiting_into):
    """One input skipped and the other waiting for a person: the node is skipped in the same ``resume()`` call,
    whichever parameter is first, so the nodes downstream are called while the question waits."""
    @node("asking", title="Asking", description="For a test.", asks=True)
    def asking(text: str) -> str:
        raise Ask(f"What about {text}?")

    branch_into = "b" if waiting_into == "a" else "a"
    compiled = graph({"id": "d", "node": "decide"}, {"id": "q", "node": "asking"}, {"id": "p", "node": "pair"}, {"id": "u", "node": "upper"},
                   edges=(("d.no", f"p.{branch_into}"), ("q.result", f"p.{waiting_into}"), ("p.result", "u.text"), ("d.yes", "u.text")),
                   registry=registry_of(decide, asking, pair, upper), inputs={"decided": "d.text", "asked": "q.text"})
    run = compiled.run(decided="ok", asked="z")
    assert run.status == "waiting" and run.results["u"]["result"] == "OK"
    assert json.loads(run.to_json())["outcomes"]["p"]["states"] == ["skipped"]  # an outcome in this resume() call, before the answer
    done = run.answer(run.waiting[0], "yes").resume()
    assert done.status == "complete" and done.results["p"]["result"] is None and done.results["u"]["result"] == "OK"


def test_both_branches_of_a_node_not_called_are_skipped():
    @node("branched", title="Branched", description="For a test.")
    def branched(email: str) -> YesNo:
        return YesNo(email, SKIPPED)

    run = ran(*CONTACTS, {"id": "b", "node": "branched"}, {"id": "u", "node": "upper"},
              edges=(SPLIT_CONTACTS, ("r.result.email", "b.email"), ("b.yes", "u.text")), registry=registry_of(contacts, records, branched, upper), text=CSV)
    assert run.results["b"] == {"yes": ["ida@x", None, "nina@x"], "no": [None, None, None]}
    assert run.results["u"]["result"] == ["IDA@X", None, "NINA@X"]
    assert [(n.loc[0], n.loc[1:]) for n in run.not_called] == [("b", (1,))]  # b was not called there, so no branch was chosen and u is skipped


def test_a_branching_node_not_called_for_an_item_skips_both_branches_even_an_input_taking_none():
    """``d`` is not called for Bo's ``None`` email, so no branch is chosen for Bo: ``y`` and ``n`` are skipped there,
    although their input takes ``None``."""
    run = ran(*CONTACTS, {"id": "d", "node": "decide"}, {"id": "y", "node": "send-or-say"}, {"id": "n", "node": "send-or-say"}, {"id": "j", "node": "join"},
              edges=(SPLIT_CONTACTS, ("r.result.email", "d.text"), ("d.yes", "y.email"), ("d.no", "n.email"), ("y.result", "j.texts"), ("n.result", "j.texts")), text=CSV)
    assert run.results["d"] == {"yes": ["ida@x", None, "nina@x"], "no": [None, None, None]}
    assert run.results["y"]["result"] == ["sent to ida@x", None, "sent to nina@x"] and run.results["n"]["result"] == [None, None, None]
    assert run.results["j"]["result"] == "sent to ida@x+sent to nina@x"
    assert run.not_called == [NotCalled(loc=("d", 1), input="text")]
    again = through_json(run).again()
    assert (again.results["y"], again.results["n"], again.not_called) == (run.results["y"], run.results["n"], run.not_called)


def test_a_branching_node_outside_loops_not_called_skips_the_nodes_on_its_branches():
    @node("nothing", title="Nothing", description="For a test.")
    def nothing() -> str | None:
        return None

    said = streamed(graph({"id": "n", "node": "nothing"}, {"id": "d", "node": "decide"}, {"id": "y", "node": "send-or-say"},
                          edges=(("n.result", "d.text"), ("d.yes", "y.email")), registry=registry_of(nothing, decide, send_or_say)).with_inputs())
    run = said[-1].run
    assert run.status == "complete" and run.results["y"]["result"] is None and run.not_called == [NotCalled(loc=("d",), input="text")]
    assert [e.loc for e in said if e.type == "node_skipped"] == [("y",)]


def test_a_node_collecting_calls_that_read_a_table_from_outside_the_loop_is_called():
    """Whether collecting is skipped is decided by the outcome of the node that returned the outer value, never by
    comparing values: a table has no plain ``==``."""
    @node("with-height", title="With height", description="For a test.")
    def with_height(text: str, table: Table) -> str:
        return f"{text}{table.height}"

    run = ran({"id": "c", "node": "contacts"}, {"id": "s", "node": "split"}, {"id": "h", "node": "with-height"}, {"id": "j", "node": "join"},
              edges=(("s.result", "h.text"), ("c.result", "h.table"), ("h.result", "j.texts")), registry=registry_of(contacts, split, with_height, join),
              ports={"inputs": {"contacts": "c.text", "split": "s.text"}}, contacts=CSV, split="a,b")
    assert run.status == "complete" and run.results["j"]["result"] == "a3+b3"


@node("maybe-template", title="Maybe template", description="``nothing`` returns ``None``, ``skip`` returns ``SKIPPED``.")
def maybe_template(kind: str) -> str | None:
    return SKIPPED if kind == "skip" else None if kind == "nothing" else kind


@node("count-texts", title="Count texts", description="For a test.")
def count_texts(values: Series[str]) -> int:
    return len(values)


@node("count-or-none", title="Count or none", description="For a test.")
def count_or_none(values: Series[str] | None) -> str:
    return "none" if values is None else str(len(values))


@node("pair-or-alone", title="Pair or alone", description="For a test.")
def pair_or_alone(a: str, b: str | None) -> str:
    return f"{a}+{b}"


def _pairs_with_an_outer_value(kind: str, reader: str = "pair", counter: str = "count-texts") -> Any:
    """The run of ``split -> pair -> count``, where ``pair.b`` is an outer value from ``maybe-template`` (``kind``:
    ``"nothing"`` returns ``None``, ``"skip"`` returns ``SKIPPED``, any other text is the value)."""
    return ran({"id": "s", "node": "split"}, {"id": "t", "node": "maybe-template"}, {"id": "p", "node": reader}, {"id": "n", "node": counter},
               edges=(("s.result", "p.a"), ("t.result", "p.b"), ("p.result", "n.values")),
               registry=registry_of(split, maybe_template, pair, pair_or_alone, count_texts, count_or_none), text="a,b", kind=kind)


def test_a_none_from_outside_the_loop_leaves_the_collecting_node_not_called_as_a_skip_skips_it():
    """No call of ``pair`` was tried, so a count of 0 would be wrong: the ``None`` came from outside the loop."""
    nothing = _pairs_with_an_outer_value("nothing")
    assert nothing.status == "complete" and nothing.results["n"]["result"] is None
    assert nothing.not_called == [NotCalled(loc=("p", 0), input="b"), NotCalled(loc=("p", 1), input="b"), NotCalled(loc=("n",), input="values")]
    assert through_json(nothing).again().not_called == nothing.not_called
    skipped = _pairs_with_an_outer_value("skip")
    assert skipped.results["n"]["result"] is None and skipped.not_called == []
    assert _pairs_with_an_outer_value("x").results["n"]["result"] == 2


def test_a_collecting_input_that_takes_none_gets_the_none_from_outside_the_loop():
    assert _pairs_with_an_outer_value("nothing", counter="count-or-none").results["n"]["result"] == "none"


def test_a_none_from_outside_that_an_input_takes_gives_the_collecting_node_every_item():
    """``pair-or-alone`` takes ``None``, so it is called for every item and ``count`` collects both."""
    run = _pairs_with_an_outer_value("nothing", reader="pair-or-alone")
    assert run.results["n"]["result"] == 2 and run.not_called == []


def test_a_none_per_item_is_left_out_and_none_for_every_item_collects_an_empty_series():
    run = ran(*CONTACTS, {"id": "u", "node": "upper"}, {"id": "n", "node": "count-texts"},
              edges=(SPLIT_CONTACTS, ("r.result.email", "u.text"), ("u.result", "n.values")), registry=registry_of(contacts, records, upper, count_texts), text="Bo,;Cy,")
    assert run.results["n"]["result"] == 0 and [n.loc for n in run.not_called] == [("u", 0), ("u", 1)]


def test_a_series_not_returned_for_an_item_given_none_is_not_empty_so_its_count_is_not_called():
    run = ran(*CONTACTS, {"id": "s", "node": "split"}, {"id": "n", "node": "count"},
              edges=(SPLIT_CONTACTS, ("r.result.email", "s.text"), ("s.result", "n.values")), text=CSV)
    assert run.results["n"]["result"] == [1, None, 1]
    assert NotCalled(loc=("n", 1), input="values") in run.not_called


def test_a_table_not_returned_for_an_item_given_none_is_not_split_so_its_count_is_not_called():
    run = ran(*CONTACTS, {"id": "i", "node": "invoices"}, {"id": "l", "node": "invoice-lines"}, {"id": "n", "node": "count"},
              edges=(SPLIT_CONTACTS, ("r.result.email", "i.text"), ("i.result", "l.table"), ("l.result.vendor", "n.values")), text="Ida,a:1;Bo,")
    assert run.results["n"]["result"] == [1, None]
    assert run.results["i"]["result"][1] is None  # invoices was not called for Bo


def test_collecting_leaves_none_out():
    run = ran(*CONTACTS, {"id": "j", "node": "join"}, edges=(SPLIT_CONTACTS, ("r.result.email", "j.texts")), text=CSV)
    assert run.results["j"]["result"] == "ida@x+nina@x"


def test_a_collecting_input_that_takes_none_keeps_each_none():
    @node("keep", title="Keep", description="For a test.")
    def keep(values: Series[str | None]) -> int:
        return values.count(None)

    run = ran(*CONTACTS, {"id": "k", "node": "keep"}, edges=(SPLIT_CONTACTS, ("r.result.email", "k.values")),
              registry=registry_of(contacts, records, keep), text=CSV)
    assert run.results["k"]["result"] == 1


@node("zipped", title="Zipped", description="For a test.")
def zipped(a: Series[str], b: Series[str]) -> str:
    return "|".join(x + y for x, y in zip(a, b, strict=True))


def test_two_inputs_collecting_one_series_leave_out_the_same_none_item():
    run = ran(*CONTACTS, {"id": "z", "node": "zipped"}, edges=(SPLIT_CONTACTS, ("r.result.name", "z.a"), ("r.result.email", "z.b")),
              registry=registry_of(contacts, records, zipped), text=CSV)
    assert run.results["z"]["result"] == "Idaida@x|Ninanina@x"


def test_a_node_downstream_of_one_not_called_is_not_called_either():
    CALLED.clear()
    run = ran(*CONTACTS, {"id": "u", "node": "upper"}, {"id": "v", "node": "upper"},
              edges=(SPLIT_CONTACTS, ("r.result.email", "u.text"), ("u.result", "v.text")), text=CSV)
    assert run.results["v"]["result"] == ["IDA@X", None, "NINA@X"]
    assert len(CALLED) == 4
    assert [(n.loc[0], n.loc[1:]) for n in run.not_called] == [("u", (1,)), ("v", (1,))]


def test_a_none_item_is_left_out_of_every_collecting_input_so_they_stay_paired():
    @node("none-for-underscore", title="None for underscore", description="For a test.")
    def none_for_underscore(text: str) -> Series[str | None]:
        return [None if x == "_" else x for x in text.split(",")]

    @node("ask", title="Ask", description="For a test.")
    def ask(text: str) -> str | None:
        return text + "?"

    @node("paired", title="Paired", description="For a test.")
    def paired(a: Series[str], b: Series[str | None]) -> str:
        return str(list(zip(a, b, strict=True)))

    run = ran({"id": "a", "node": "none-for-underscore"}, {"id": "b", "node": "ask"}, {"id": "z", "node": "paired"},
              edges=(("a.result", "b.text"), ("a.result", "z.a"), ("b.result", "z.b")), registry=registry_of(none_for_underscore, ask, paired), text="1,_,3")
    assert run.results["z"]["result"] == str([("1", "1?"), ("3", "3?")])


def test_collecting_a_series_not_returned_for_a_none_item_is_not_called_not_an_empty_series():
    """A document that is ``None`` is not an empty document: its count is not called, and ``not_called`` says which
    input."""

    @node("docs", title="Docs", description="For a test.")
    def docs(text: str) -> Series[str | None]:
        return [None if x == "-" else x for x in text.split(";")]

    @node("lines", title="Lines", description="For a test.")
    def lines(text: str) -> Series[str]:
        return [x for x in text.split(",") if x]

    @node("count", title="Count", description="For a test.")
    def count(items: Series[str]) -> int:
        return len(items)

    @node("width", title="Width", description="For a test.")
    def width(table: Table) -> int:
        return table.width

    @node("as_table", title="As table", description="For a test.")
    def as_table(text: str) -> Table:
        return pl.DataFrame({"part": text.split(",")})

    @node("total", title="Total", description="For a test.")
    def total(ns: Series[int]) -> int:
        return sum(ns)

    reg = registry_of(docs, lines, count, width, as_table, total)
    compiled = Graph.compile(GraphRecord(
        nodes={
            "d": NodeRecord(node="docs", inputs={"text": InputEntry(graph_input="text")}),
            "l": NodeRecord(node="lines", inputs={"text": InputEntry(sources=[Address("d", "result")])}),
            "c": NodeRecord(node="count", inputs={"items": InputEntry(sources=[Address("l", "result")])}),
            "t": NodeRecord(node="total", inputs={"ns": InputEntry(sources=[Address("c", "result")])}),
            "a": NodeRecord(node="as_table", inputs={"text": InputEntry(sources=[Address("d", "result")])}),
            "w": NodeRecord(node="width", inputs={"table": InputEntry(sources=[Address("a", "result")])}),
        },
    ), reg)
    run = compiled.run(text="a,b;-;c;")

    assert run.results["c"]["result"] == [2, None, 1, 0]  # the last document is empty: 0 lines, called
    assert run.results["w"]["result"] == [1, None, 1, 1]
    assert run.results["t"]["result"] == 3
    assert {(n.loc[0], n.loc[1:], n.input) for n in run.not_called} >= {("c", (1,), "items"), ("w", (1,), "table")}


# -- loops, outer values and collecting --------------------------------------------------------

THREE_LEVELS = (
    {"id": "docs", "node": "split", "constants": {"sep": "|"}},
    {"id": "paras", "node": "split", "constants": {"sep": ";"}},
    {"id": "lines", "node": "split"},
)
THREE_EDGES = (("docs.result", "paras.text"), ("paras.result", "lines.text"))


@node("labelled", title="Labelled", description="For a test.")
def labelled(label: str, texts: Series[str]) -> str:
    return f"{label}={','.join(texts)}"


@node("two-series", title="Two series", description="For a test.")
def two_series(a: Series[str], b: Series[str]) -> str:
    return f"{','.join(a)}+{','.join(b)}"


@node("joined-beside", title="Joined beside", description="For a test.")
def joined_beside(items: Series[str], at: str) -> str:
    return f"{at}:{'+'.join(items)}"


DEEP = registry_of(split, labelled, two_series, joined_beside, show)


def test_each_series_output_of_a_node_starts_its_own_loop_of_its_own_length():
    @dataclass(frozen=True)
    class Both:
        a: Series[str]
        b: Series[str]

    @node("uneven", title="Uneven", description="For a test.")
    def uneven(text: str) -> Both:
        return Both(a=["x"], b=("y", "z"))  # any sequence will do

    run = ran({"id": "u", "node": "uneven"}, {"id": "ua", "node": "upper"}, {"id": "ub", "node": "upper"},
              edges=(("u.a", "ua.text"), ("u.b", "ub.text")), registry=registry_of(uneven, upper),
              ports={"outputs": {"a": "ua.result", "b": "ub.result"}}, text="t")
    assert {k: v.msg for k, v in run.graph.runs_on.items()} == {"u": "once", "ua": "per item of u.a", "ub": "per item of u.b"}
    assert (run.results["u"]["a"], run.results["u"]["b"]) == (["x"], ["y", "z"])
    assert (run.results["ua"]["result"], run.results["ub"]["result"]) == (["X"], ["Y", "Z"])


def test_a_node_receives_a_plain_list_and_may_return_any_sequence():
    seen = []

    @node("tuple-split", title="Tuple split", description="For a test.")
    def tuple_split(text: str) -> Series[str]:
        return tuple(text.split(","))

    @node("collect", title="Collect", description="For a test.")
    def collect(parts: Series[str]) -> str:
        seen.append(type(parts))
        return "+".join(parts)

    run = ran({"id": "s", "node": "tuple-split"}, {"id": "c", "node": "collect"}, edges=(("s.result", "c.parts"),),
              registry=registry_of(tuple_split, collect), text="a,b")
    assert run.results == {"s": {"result": ["a", "b"]}, "c": {"result": "a+b"}}
    assert seen == [list]


def test_a_polars_series_returned_for_a_series_output_is_its_values():
    @node("lengths", title="Lengths", description="For a test.")
    def lengths(text: str) -> Series[int]:
        return pl.Series([len(word) for word in text.split()])

    assert Graph.compile(make_graph(lengths), registry_of(lengths)).run(text="a bb ccc").outputs == {"result": [1, 2, 3]}


def test_an_outer_value_is_the_same_on_each_call_inside_the_loop():
    run = ran({"id": "docs", "node": "split"}, {"id": "prefix", "node": "upper"}, {"id": "p", "node": "pair"},
              edges=(("prefix.result", "p.a"), ("docs.result", "p.b")),
              ports={"inputs": {"docs": "docs.text", "prefix": "prefix.text"}}, docs="a,b", prefix="p")
    assert run.graph.runs_on["p"].msg == "per item of docs.result"
    assert run.results["p"]["result"] == ["P:a", "P:b"]


def test_a_document_is_the_outer_value_of_each_call_for_one_of_its_lines():
    run = ran({"id": "docs", "node": "split", "constants": {"sep": "|"}}, {"id": "lines", "node": "split", "constants": {"sep": "/"}}, {"id": "p", "node": "pair"},
              edges=(("docs.result", "lines.text"), ("lines.result", "p.a"), ("docs.result", "p.b")), text="a/b|c")
    assert run.graph.runs_on["p"].msg == "per item of lines.result"
    assert run.results["p"]["result"] == [["a:a/b", "b:a/b"], ["c:c"]]


def test_one_value_into_a_series_input_is_a_series_of_one():
    run = ran({"id": "a", "node": "upper"}, {"id": "j", "node": "join"}, edges=(("a.result", "j.texts"),), text="a")
    assert run.graph.runs_on["j"].msg == "once" and run.results["j"]["result"] == "A"


def test_a_series_input_given_a_constant_list_or_its_default_gets_it_whole_in_one_call():
    @node("join-or-nothing", title="Join or nothing", description="For a test.")
    def join_or_nothing(texts: Series[str] = ()) -> str:
        return "+".join(texts) or "nothing"

    reg = registry_of(join, join_or_nothing)
    run = ran({"id": "typed", "node": "join", "constants": {"texts": ["a", "b"]}}, {"id": "absent", "node": "join-or-nothing"}, registry=reg,
              ports={"outputs": {"typed": "typed.result", "absent": "absent.result"}})
    assert {k: v.msg for k, v in run.graph.runs_on.items()} == {"typed": "once", "absent": "once"}
    assert run.results == {"typed": {"result": "a+b"}, "absent": {"result": "nothing"}}


def test_an_unconnected_series_input_takes_its_default():
    @node("join-or-nothing", title="Join or nothing", description="For a test.")
    def join_or_nothing(texts: Series[str] = ()) -> str:
        return "+".join(texts)

    assert ran({"id": "j", "node": "join-or-nothing"}, registry=registry_of(join_or_nothing)).outputs == {"result": ""}


def test_a_closed_list_input_is_one_value_with_its_list_and_with_its_default():
    @node("tags", title="Tags", description="For a test.")
    def tags(tags: list[str] = ["x"]) -> str:  # noqa: B006
        return "|".join(tags)

    reg = registry_of(tags)
    assert ran({"id": "t", "node": "tags", "constants": {"tags": ["a", "b", "c"]}}, registry=reg).outputs == {"result": "a|b|c"}
    assert ran({"id": "t", "node": "tags"}, registry=reg).outputs == {"result": "x"}


def test_collecting_ends_one_loop_so_a_node_reading_a_document_and_its_lines_runs_per_paragraph():
    @node("deep", title="Deep", description="For a test.")
    def deep(doc: str, lines: Series[str]) -> str:
        return f"{doc}:{len(lines)}"

    run = ran(*THREE_LEVELS, {"id": "d", "node": "deep"}, edges=(*THREE_EDGES, ("docs.result", "d.doc"), ("lines.result", "d.lines")),
              registry=registry_of(split, deep), text="a,b;c|d")
    assert run.graph.runs_on["d"].msg == "per item of paras.result"
    assert run.results["d"]["result"] == [["a,b;c:2", "a,b;c:1"], ["d:1"]]


def test_a_node_per_paragraph_collects_only_that_paragraphs_lines():
    run = ran(*THREE_LEVELS, {"id": "n", "node": "labelled"}, edges=(*THREE_EDGES, ("paras.result", "n.label"), ("lines.result", "n.texts")),
              registry=DEEP, text="a,b;c|d")
    assert run.results["n"]["result"] == [["a,b=a,b", "c=c"], ["d=d"]]


def test_two_inputs_collecting_at_different_depths_run_the_node_in_the_deeper_loop():
    """``a`` collects a document's paragraphs, ``b`` a paragraph's lines: the node is called per paragraph."""
    run = ran(*THREE_LEVELS, {"id": "n", "node": "two-series"}, edges=(*THREE_EDGES, ("paras.result", "n.a"), ("lines.result", "n.b")),
              registry=DEEP, text="a,b;c|d")
    assert run.graph.runs_on["n"].msg == "per item of paras.result"
    assert run.results["n"]["result"] == [["a,b,c+a,b", "a,b,c+c"], ["d+d"]]


def test_a_document_with_no_lines_is_called_with_an_empty_series():
    run = ran({"id": "docs", "node": "split", "constants": {"sep": "|"}}, {"id": "lines", "node": "split"}, {"id": "n", "node": "labelled"},
              edges=(("docs.result", "lines.text"), ("docs.result", "n.label"), ("lines.result", "n.texts")), registry=DEEP, text="a||b")
    assert run.results["n"]["result"] == ["a=a", "=", "b=b"]


def test_a_series_read_whole_beside_its_own_items_is_the_whole_series_on_each_call():
    run = ran({"id": "s", "node": "split"}, {"id": "j", "node": "joined-beside"}, edges=(("s.result", "j.items"), ("s.result", "j.at")),
              registry=DEEP, text="a,b")
    assert run.graph.runs_on["j"].msg == "per item of s.result"
    assert run.results["j"]["result"] == ["a:a+b", "b:a+b"]


def test_collecting_another_series_gives_an_outer_value_beside_the_loop_the_node_is_in():
    run = ran({"id": "s", "node": "split"}, {"id": "o", "node": "split"}, {"id": "j", "node": "joined-beside"},
              edges=(("o.result", "j.items"), ("s.result", "j.at")), registry=DEEP,
              ports={"inputs": {"at": "s.text", "items": "o.text"}}, at="a,b", items="x,y")
    assert run.results["j"]["result"] == ["a:x+y", "b:x+y"]


def test_a_documents_keywords_collected_are_beside_each_of_its_invoices_also_after_json():
    """Every keyword of a document beside each of its invoices: the keywords are collected per document."""

    @dataclass(frozen=True)
    class Parts:
        invoices: Series[str]
        keywords: Series[str]

    @node("parts", title="Parts", description="For a test.")
    def parts(text: str) -> Parts:
        head, _, tail = text.partition(":")
        return Parts(head.split(","), tail.split(","))

    @node("beside", title="Beside", description="For a test.")
    def beside(invoice: str, keywords: Series[str]) -> str:
        return f"{invoice}[{'/'.join(keywords)}]"

    reg = registry_of(split, parts, beside)
    nodes = ({"id": "docs", "node": "split", "constants": {"sep": "|"}}, {"id": "p", "node": "parts"}, {"id": "b", "node": "beside"})
    edges = (("docs.result", "p.text"), ("p.invoices", "b.invoice"), ("p.keywords", "b.keywords"))
    run = ran(*nodes, edges=edges, registry=reg, text="i1,i2:k1,k2|i3:k3")

    assert run.results["b"]["result"] == [["i1[k1/k2]", "i2[k1/k2]"], ["i3[k3]"]]
    assert through_json(run).results == run.results
    assert run.graph.runs_on["b"].msg == "per item of p.invoices"


def test_two_series_outputs_of_one_node_and_two_series_under_one_document_never_pair_up():
    @dataclass(frozen=True)
    class Two:
        a: Series[str]
        b: Series[str]

    @node("two", title="Two", description="For a test.")
    def two(text: str) -> Two:
        return Two([text], [text])

    reg = registry_of(two, split, pair)
    assert problems({"id": "t", "node": "two"}, {"id": "p", "node": "pair"}, edges=(("t.a", "p.a"), ("t.b", "p.b")), registry=reg) == ["series_not_nested"]
    assert problems({"id": "s", "node": "split"}, {"id": "x", "node": "split"}, {"id": "y", "node": "split"}, {"id": "p", "node": "pair"},
                    edges=(("s.result", "x.text"), ("s.result", "y.text"), ("x.result", "p.a"), ("y.result", "p.b")), registry=reg) == ["series_not_nested"]


def test_each_input_in_describe_says_how_it_receives_on_each_call_of_its_node():
    compiled = graph({"id": "s", "node": "split"}, {"id": "u", "node": "upper"}, {"id": "j", "node": "join"}, {"id": "p", "node": "pair", "constants": {"b": "x"}},
                   {"id": "k", "node": "pair"},
                   edges=(("s.result", "u.text"), ("u.result", "j.texts"), ("u.result", "p.a"), ("j.result", "k.a"), ("u.result", "k.b")),
                   outputs={"p": "p.result", "k": "k.result"})
    nodes = compiled.describe().nodes

    def source_of(node_id: str) -> dict[str, str]:
        return {i.field.name: i.source.msg for i in nodes[node_id].inputs}

    assert nodes["k"].inputs[0].source.model_dump() == {"mode": "outer_value", "series": None, "msg": "one value, the same for every item"}
    assert nodes["j"].runs == RunsOn(mode="once", msg="once")
    assert compiled.runs_on["u"] == RunsOn(mode="per_item", series="s.result", msg="per item of s.result")
    assert source_of("u") == {"text": "per item of s.result"}
    assert source_of("j") == {"texts": "every item of s.result"}
    assert source_of("p") == {"a": "per item of s.result", "b": "a constant"}
    assert source_of("k") == {"a": "one value, the same for every item", "b": "per item of s.result"}
    assert source_of("s") == {"text": "the graph input", "sep": "a constant"}  # sep has only its default


# -- no items, and skips upstream of a loop ----------------------------------------------------


def test_a_node_in_a_loop_over_an_empty_series_never_starts_and_collecting_gets_an_empty_series():
    compiled = graph({"id": "s", "node": "split"}, {"id": "u", "node": "upper"}, {"id": "j", "node": "join"},
                   edges=(("s.result", "u.text"), ("u.result", "j.texts")))

    said = streamed(compiled.with_inputs(text=""))
    assert [e.type for e in said if getattr(e, "loc", None) == ("u",)] == ["node_complete"]  # never started
    run = said[-1].run
    assert run.status == "complete" and run.results["u"]["result"] == [] and run.results["j"]["result"] == ""
    assert through_json(run).results["u"]["result"] == []


def test_a_branch_not_chosen_skips_the_node_downstream():
    run = ran({"id": "g", "node": "decide"}, {"id": "y", "node": "upper"}, {"id": "n", "node": "upper"},
              edges=(("g.yes", "y.text"), ("g.no", "n.text")), ports={"outputs": {"yes": "y.result", "no": "n.result"}}, text="hey")
    assert run.results["y"]["result"] == "HEY" and run.results["n"]["result"] is None


def test_a_failing_node_on_the_branch_not_chosen_is_never_called():
    @node("always-fail", title="Always fail", description="For a test.")
    def always_fail(text: str) -> str:
        raise ValueError("never")

    run = ran({"id": "g", "node": "decide"}, {"id": "y", "node": "upper"}, {"id": "n", "node": "always-fail"},
              edges=(("g.yes", "y.text"), ("g.no", "n.text")), registry=registry_of(decide, upper, always_fail),
              ports={"outputs": {"yes": "y.result", "no": "n.result"}}, text="hey")
    assert run.results["y"]["result"] == "HEY"


def test_a_series_never_returned_leaves_nodes_downstream_without_results_also_after_json():
    for nodes, edges in (
        (({"id": "s", "node": "split"},), (("g.yes", "s.text"), ("s.result", "u.text"))),
        (CONTACTS, (("g.yes", "c.text"), SPLIT_CONTACTS, ("r.result.name", "u.text"))),
    ):
        run = ran({"id": "g", "node": "decide"}, *nodes, {"id": "u", "node": "upper"}, edges=edges, text="no")
        assert run.results["u"]["result"] is None and through_json(run).results["u"]["result"] is None


def test_a_skip_upstream_of_a_loop_skips_collecting_and_no_items_collect_an_empty_series():
    skipped = ran({"id": "b", "node": "decide"}, {"id": "a", "node": "split"}, {"id": "j", "node": "join"},
                  edges=(("b.no", "a.text"), ("a.result", "j.texts")), text="ab")
    assert skipped.results["j"]["result"] is None
    empty = ran({"id": "a", "node": "split"}, {"id": "j", "node": "join"}, edges=(("a.result", "j.texts"),), text="")
    assert empty.results["j"]["result"] == ""


def test_a_skipped_series_node_has_no_items_so_everything_downstream_is_skipped():
    run = ran({"id": "g", "node": "decide"}, {"id": "docs", "node": "split"}, {"id": "u", "node": "upper"}, {"id": "j", "node": "join"},
              edges=(("g.no", "docs.text"), ("docs.result", "u.text"), ("u.result", "j.texts")), text="a,b")
    assert run.results["docs"]["result"] is None and run.results["u"]["result"] is None and run.results["j"]["result"] is None


def test_a_branch_not_chosen_outside_the_loop_skips_collecting_but_per_item_collects_an_empty_series():
    CALLED.clear()
    outside = ran({"id": "b", "node": "decide"}, {"id": "j", "node": "join"}, edges=(("b.no", "j.texts"),), text="yes")
    assert outside.results["j"]["result"] is None and CALLED == []

    per_item = ran({"id": "s", "node": "split"}, {"id": "b", "node": "decide"}, {"id": "j", "node": "join"}, {"id": "p", "node": "pair"},
                   edges=(("s.result", "b.text"), ("b.no", "j.texts"), ("s.result", "p.a"), ("j.result", "p.b")), text="p,q")
    assert per_item.results["j"]["result"] == "" and CALLED == ["join []"]  # called once, over two rejected items
    assert per_item.results["p"]["result"] == ["p:", "q:"]


def test_a_skip_for_one_document_reaches_the_calls_for_its_lines_and_the_other_document_runs():
    run = ran({"id": "d", "node": "split", "constants": {"sep": "|"}}, {"id": "g", "node": "decide"}, {"id": "l", "node": "split"},
              {"id": "t", "node": "pair"}, {"id": "u", "node": "upper"},
              edges=(("d.result", "g.text"), ("d.result", "l.text"), ("l.result", "t.a"), ("g.yes", "t.b"), ("t.result", "u.text")), text="a,b|c,no")
    assert run.graph.runs_on["u"].msg == "per item of l.result"
    assert run.results["u"]["result"] == [["A:A,B", "B:A,B"], None]


def test_a_skip_for_one_document_skips_collecting_its_lines_also_after_json():
    run = ran({"id": "d", "node": "split", "constants": {"sep": "|"}}, {"id": "g", "node": "decide"}, {"id": "l", "node": "split"},
              {"id": "p", "node": "pair"}, {"id": "j", "node": "join"},
              edges=(("d.result", "g.text"), ("d.result", "l.text"), ("l.result", "p.a"), ("g.yes", "p.b"), ("p.result", "j.texts")), text="x|y,no")
    assert run.results["j"]["result"] == ["x:x", None]
    assert through_json(run).results["j"]["result"] == ["x:x", None]


def test_a_skip_upstream_of_an_empty_series_skips_the_collecting_node():
    run = ran({"id": "d", "node": "split", "constants": {"sep": "|"}}, {"id": "g", "node": "decide"}, {"id": "l", "node": "split"},
              {"id": "t", "node": "pair"}, {"id": "m", "node": "split"}, {"id": "x", "node": "pair"}, {"id": "j", "node": "join"},
              edges=(("d.result", "g.text"), ("d.result", "l.text"), ("l.result", "t.a"), ("g.yes", "t.b"), ("l.result", "m.text"),
                     ("t.result", "x.a"), ("m.result", "x.b"), ("x.result", "j.texts")), text="p,no")
    assert run.graph.runs_on["j"].msg == "per item of l.result"
    assert run.results["j"]["result"] == [None]


def test_a_document_on_the_branch_not_chosen_has_no_lines_and_collecting_them_is_skipped():
    run = ran({"id": "docs", "node": "split", "constants": {"sep": "|"}}, {"id": "g", "node": "decide"}, {"id": "lines", "node": "split"},
              {"id": "up", "node": "upper"}, {"id": "j", "node": "labelled"},
              edges=(("docs.result", "g.text"), ("g.yes", "lines.text"), ("lines.result", "up.text"), ("docs.result", "j.label"), ("up.result", "j.texts")),
              registry=registry_of(split, decide, upper, labelled), text="a,b|no")
    assert run.results["lines"]["result"] == [["a", "b"], None]
    assert run.results["j"]["result"] == ["a,b=A,B", None]


def test_collecting_waits_for_the_branch_upstream_of_its_loop_before_it_is_called_with_nothing():
    release = Event()

    @node("delayed", title="Delayed", description="For a test.", policy=Policy(concurrency=2))
    def delayed(text: str) -> YesNo:
        release.wait(5)
        return YesNo(SKIPPED, text) if "no" in text else YesNo(text, SKIPPED)

    compiled = graph({"id": "docs", "node": "split", "constants": {"sep": "|"}}, {"id": "g", "node": "delayed"}, {"id": "l", "node": "split"},
                   {"id": "j", "node": "join"}, edges=(("docs.result", "g.text"), ("g.yes", "l.text"), ("l.result", "j.texts")),
                   registry=registry_of(split, delayed, join))
    CALLED.clear()
    timer = Timer(0.2, release.set)
    timer.start()
    started = time.monotonic()
    run = compiled.run(text="a|slow-no")
    assert time.monotonic() - started >= 0.2 and run.status == "complete", run.cause
    assert run.results["j"]["result"] == ["a", None]
    assert "join []" not in CALLED


def test_skipped_items_of_a_returned_series_are_none_downstream_also_after_json():
    @node("sparse", title="Sparse", description="For a test.")
    def sparse() -> Series[str]:
        return ["a", SKIPPED, "c"]

    run = ran({"id": "s", "node": "sparse"}, {"id": "u", "node": "upper"}, edges=(("s.result", "u.text"),), registry=registry_of(sparse, upper))
    assert run.results["u"]["result"] == ["A", None, "C"]
    assert through_json(run).results == run.results


def test_a_branch_on_a_whole_series_skips_the_nodes_on_the_branch_not_chosen():
    @dataclass(frozen=True)
    class Decided:
        full: Annotated[Series[str], Param(exclusive_group="d")]
        empty: Annotated[Series[str], Param(exclusive_group="d")]

    @node("decide-list", title="Decide list", description="For a test.")
    def decide_list(texts: Series[str]) -> Decided:
        return Decided(full=list(texts), empty=SKIPPED) if texts else Decided(full=SKIPPED, empty=[])

    CALLED.clear()
    run = ran({"id": "docs", "node": "split"}, {"id": "d", "node": "decide-list"}, {"id": "full", "node": "join"},
              {"id": "up", "node": "upper"}, {"id": "after", "node": "join"},
              edges=(("docs.result", "d.texts"), ("d.full", "full.texts"), ("d.empty", "up.text"), ("up.result", "after.texts")),
              registry=registry_of(split, decide_list, join, upper), ports={"outputs": {"full": "full.result", "after": "after.result"}}, text="a,b")
    assert run.results["full"]["result"] == "a+b"
    assert run.results["up"]["result"] is None and run.results["after"]["result"] is None
    assert CALLED == ["join ['a', 'b']"]


# -- a table is one value ----------------------------------------------------------------------

DOCS = {"id": "docs", "node": "docs"}
READ = {"id": "read", "node": "invoices"}
LINES = {"id": "lines", "node": "invoice-lines"}
READ_EDGES = (("docs.result", "read.text"), ("read.result", "lines.table"))


def test_a_table_wired_into_a_table_input_reaches_the_node_as_that_table():
    seen = []

    @node("relay", title="Relay", description="For a test.")
    def relay(table: Table) -> Table:
        seen.append(table)
        return table

    run = ran({"id": "c", "node": "contacts"}, {"id": "a", "node": "relay"}, {"id": "b", "node": "relay"},
              edges=(("c.result", "a.table"), ("a.result", "b.table")), registry=registry_of(contacts, relay), text=CSV)
    assert seen[0].equals(seen[1]) and seen[0].equals(run.results["c"]["result"])
    assert {k: v.msg for k, v in run.graph.runs_on.items()} == {"c": "once", "a": "once", "b": "once"}


def test_a_node_filtering_every_row_away_returns_a_table_of_no_rows():
    @node("none-left", title="None left", description="For a test.")
    def none_left(table: Table) -> Table:
        return table.filter(pl.lit(False))

    run = ran({"id": "c", "node": "contacts"}, {"id": "f", "node": "none-left"}, edges=(("c.result", "f.table"),), registry=registry_of(contacts, none_left), text=CSV)
    assert run.results["f"]["result"].height == 0


def test_a_column_of_a_table_is_no_address_until_the_table_is_split():
    compiled = graph({"id": "c", "node": "contacts"}, {"id": "u", "node": "upper"}, edges=(("c.result.email", "u.text"),))
    assert [(p.type, p.loc) for p in compiled.problems] == [("unknown_output", ("u", "text"))]
    assert problems(*CONTACTS, {"id": "u", "node": "upper"}, edges=(SPLIT_CONTACTS, ("r.result.email", "u.text"))) == []


def test_a_filtered_table_is_split_and_each_records_output_is_collected_beside_the_records():
    @node("with-email", title="With email", description="For a test.")
    def with_email(table: Table[Contact]) -> Table[Contact]:
        return table.filter(pl.col("email").is_not_null())

    @node("beside", title="Beside", description="For a test.")
    def beside(table: Table[Contact], sent: Series[str]) -> Table:
        return table.with_columns(pl.Series("sent", sent))

    run = ran({"id": "c", "node": "contacts"}, {"id": "f", "node": "with-email"}, {"id": "r", "node": "records"}, {"id": "s", "node": "send"},
              {"id": "x", "node": "beside"},
              edges=(("c.result", "f.table"), ("f.result", "r.table"), ("r.result.email", "s.email"), ("r.result", "x.table"), ("s.result", "x.sent")),
              registry=registry_of(contacts, with_email, records, send, beside), text=CSV)
    assert run.results["x"]["result"].to_dicts() == [
        {"name": "Ida", "email": "ida@x", "sent": "sent to ida@x"}, {"name": "Nina", "email": "nina@x", "sent": "sent to nina@x"},
    ]


def test_a_sorted_table_is_split_in_its_sorted_order_and_a_none_email_is_not_sent():
    @node("sort", title="Sort", description="For a test.")
    def sort(table: Table[Contact]) -> Table[Contact]:
        return table.sort("name", descending=True)

    run = ran({"id": "c", "node": "contacts"}, {"id": "o", "node": "sort"}, {"id": "r", "node": "records"}, {"id": "s", "node": "send"},
              edges=(("c.result", "o.table"), ("o.result", "r.table"), ("r.result.email", "s.email")),
              registry=registry_of(contacts, sort, records, send), text=CSV)
    assert run.results["s"]["result"] == ["sent to nina@x", "sent to ida@x", None]


def test_each_invoice_of_each_document_is_checked_and_a_skipped_check_is_none_for_that_invoice():
    run = ran(DOCS, READ, LINES, {"id": "check", "node": "check"}, {"id": "total", "node": "total"},
              edges=(*READ_EDGES, ("lines.result.vendor", "check.vendor"), ("lines.result.amount", "total.amounts")),
              ports={"outputs": {"checks": "check.result", "totals": "total.result"}}, text="acme:10,unknown:5|beta:7")
    assert {k: v.msg for k, v in run.graph.runs_on.items()} == {
        "docs": "once", "read": "per item of docs.result", "lines": "per item of docs.result", "check": "per item of lines.result",
        "total": "per item of docs.result",
    }
    assert run.results["check"]["result"] == [["acme ok", None], ["beta ok"]]
    assert run.results["total"]["result"] == [15.0, 7.0]


def test_a_document_with_no_invoices_totals_nothing():
    run = ran(DOCS, READ, LINES, {"id": "total", "node": "total"}, edges=(*READ_EDGES, ("lines.result.amount", "total.amounts")), text="acme:10|")
    assert run.results["total"]["result"] == [10.0, 0.0]


def test_a_document_skipped_upstream_of_its_table_has_no_total_and_the_others_do():
    run = ran(DOCS, {"id": "gate", "node": "decide"}, READ, LINES, {"id": "total", "node": "total"},
              edges=(("docs.result", "gate.text"), ("gate.yes", "read.text"), ("read.result", "lines.table"), ("lines.result.amount", "total.amounts")),
              text="acme:10|no|beta:7")
    assert run.results["total"]["result"] == [10.0, None, 7.0]


def test_a_table_input_collects_each_documents_records_into_one_table_per_document():
    @node("checked", title="Checked", description="For a test.")
    def checked(table: Table[Invoice], checks: Series[str | None]) -> Table:
        return table.with_columns(pl.Series("check", checks, dtype=pl.String))

    run = ran(DOCS, READ, LINES, {"id": "check", "node": "check"}, {"id": "x", "node": "checked"},
              edges=(*READ_EDGES, ("lines.result.vendor", "check.vendor"), ("lines.result", "x.table"), ("check.result", "x.checks")),
              registry=registry_of(docs, invoices, invoice_lines, check, checked), text="acme:10,unknown:5||gamma:7")
    assert run.graph.runs_on["x"].msg == "per item of docs.result"
    assert [sheet.to_dicts() for sheet in run.results["x"]["result"]] == [
        [{"vendor": "acme", "amount": 10.0, "check": "acme ok"}, {"vendor": "unknown", "amount": 5.0, "check": None}],
        [],
        [{"vendor": "gamma", "amount": 7.0, "check": "gamma ok"}],
    ]
    assert run.results["x"]["result"][1].columns == ["vendor", "amount", "check"]  # no invoices: no rows, every column


def test_a_table_returned_per_document_is_described_as_a_table_with_its_columns():
    compiled = graph(DOCS, READ, LINES, {"id": "check", "node": "check"}, edges=(*READ_EDGES, ("lines.result.vendor", "check.vendor")))
    assert compiled.problems == []
    (result,) = compiled.describe().nodes["read"].outputs
    assert not result.is_series and result.type.startswith("Table") and {c["properties"]["name"]["const"]: c["properties"]["values"]["items"] for c in result.json_schema["properties"]["columns"]["prefixItems"]}["amount"] == {"type": "number"}


def test_a_bare_table_offers_no_columns():
    (result,) = graph({"id": "s", "node": "show"}).describe().nodes["s"].outputs
    assert (result.is_series, result.type) == (False, "Table") and "prefixItems" not in result.json_schema["properties"]["columns"]


def test_a_constant_list_of_records_is_one_table_not_a_loop():
    run = ran({"id": "x", "node": "show", "constants": {"table": [{"a": "1"}, {"a": "2"}]}})
    assert run.graph.runs_on["x"].msg == "once"
    assert isinstance(run.results["x"]["result"], pl.DataFrame) and run.results["x"]["result"].to_dicts() == [{"a": "1"}, {"a": "2"}]


class Dated(TypedDict):
    name: str
    when: datetime.date


@host_type("Comma", title="Comma")
class Comma(float):
    """A number written with a decimal comma, as a host's own type reads it."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.no_info_before_validator_function(lambda v: float(v.replace(",", ".")) if isinstance(v, str) else v, core_schema.float_schema())


class Priced(TypedDict):
    price: Comma


@node("describe-dated", title="Describe dated", description="For a test.")
def describe_dated(rows: Table[Dated]) -> str:
    return f"{rows.columns}:{rows['when'].to_list()}"


@node("total-price", title="Total price", description="For a test.")
def total_price(rows: Table[Priced]) -> float:
    return float(rows["price"].sum())


DATED = registry_of(describe_dated, total_price, show, upper)


def test_a_table_the_caller_gives_is_read_by_its_column_types_and_read_back_alike_from_json():
    compiled = graph({"id": "d", "node": "describe-dated"}, registry=DATED)

    run = compiled.run(rows=[{"name": "a", "when": "2024-01-02", "extra": 1}])

    assert run.results["d"]["result"] == "['name', 'when']:[datetime.date(2024, 1, 2)]"
    assert through_json(run).results == run.results
    assert compiled.run(rows=pl.DataFrame({"name": ["b"], "when": [datetime.date(2024, 5, 6)]})).status == "complete"


def test_a_caller_s_table_missing_a_column_is_refused_naming_it_and_an_empty_list_is_a_table_of_no_rows():
    compiled = graph({"id": "d", "node": "describe-dated"}, registry=DATED)

    with pytest.raises(InvalidCallError) as refused:
        compiled.run(rows=[{"name": "a"}])
    (problem,) = refused.value.problems
    assert (problem.type, problem.loc, problem.ctx["column"]) == ("invalid_input", ("inputs", "rows"), "when")
    assert compiled.run(rows=[]).results["d"]["result"] == "['name', 'when']:[]"


def test_a_constant_table_reads_each_cell_by_its_columns_own_type():
    run = ran({"id": "t", "node": "total-price", "constants": {"rows": [{"price": "1,5"}, {"price": "2"}]}}, registry=DATED)

    assert run.results["t"]["result"] == 3.5


def test_a_constant_cell_its_column_type_cannot_read_is_a_compile_problem_naming_the_column():
    compiled = graph({"id": "d", "node": "describe-dated", "constants": {"rows": [{"name": "a", "when": "junk"}]}}, registry=DATED)

    (problem,) = compiled.problems
    assert (problem.type, problem.loc, problem.ctx["column"]) == ("invalid_value", ("d", "rows"), "when")
    assert ran({"id": "d", "node": "describe-dated", "constants": {"rows": [{"name": "a", "when": "2024-01-02"}]}}, registry=DATED).status == "complete"


def test_a_constant_on_a_table_of_any_columns_arrives_with_its_own_columns():
    run = ran({"id": "s", "node": "show", "constants": {"table": [{"email": "a@b.dk", "n": 1}]}}, registry=DATED)

    assert run.results["s"]["result"].columns == ["email", "n"]


# -- branches ----------------------------------------------------------------------------------


@node("parity", title="Parity", description="For a test.")
def parity(n: str) -> YesNo:
    return YesNo(n, SKIPPED) if int(n) % 2 == 0 else YesNo(SKIPPED, n)


def test_collecting_inputs_downstream_of_opposite_branches_per_item_are_a_problem():
    assert problems({"id": "n", "node": "split"}, {"id": "r", "node": "parity"}, {"id": "x", "node": "upper"}, {"id": "y", "node": "upper"},
                    {"id": "c", "node": "two-series"},
                    edges=(("n.result", "r.n"), ("r.yes", "x.text"), ("r.no", "y.text"), ("x.result", "c.a"), ("y.result", "c.b")),
                    registry=registry_of(split, parity, upper, two_series)) == ["lists_from_different_loops"]


def test_a_decision_per_document_read_by_nodes_per_line_still_keeps_two_branches_apart():
    assert problems({"id": "s", "node": "split", "constants": {"sep": "|"}}, {"id": "d", "node": "parity"}, {"id": "sp", "node": "split"},
                    {"id": "ua", "node": "pair"}, {"id": "ub", "node": "pair"}, {"id": "z", "node": "two-series"},
                    edges=(("s.result", "d.n"), ("s.result", "sp.text"), ("sp.result", "ua.a"), ("d.yes", "ua.b"), ("sp.result", "ub.a"), ("d.no", "ub.b"),
                           ("ua.result", "z.a"), ("ub.result", "z.b")),
                    registry=registry_of(split, parity, pair, two_series)) == ["lists_from_different_loops"]


def test_an_item_skipped_on_one_series_input_is_left_out_of_every_series_input_of_the_node():
    run = ran({"id": "n", "node": "split"}, {"id": "r", "node": "parity"}, {"id": "x", "node": "upper"}, {"id": "z", "node": "zipped"},
              edges=(("n.result", "r.n"), ("r.yes", "x.text"), ("n.result", "z.a"), ("x.result", "z.b")), registry=registry_of(split, parity, upper, zipped),
              text="1,2,3,4")
    assert run.results["z"]["result"] == "22|44"


def test_two_edges_into_one_input_that_can_both_have_a_value_are_a_problem_never_first_wins():
    """``upper(a, b)`` would silently lose ``b``; two edges per item of one Series would cover each item twice."""
    assert problems({"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "u", "node": "upper"},
                    edges=(("a.result", "u.text"), ("b.result", "u.text")), inputs={"text": ["a.text", "b.text"]}) == ["several_edges_into_one_input"]
    compiled = graph({"id": "s", "node": "split"}, {"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "m", "node": "upper"},
                   edges=(("s.result", "a.text"), ("s.result", "b.text"), ("a.result", "m.text"), ("b.result", "m.text")))
    (problem,) = compiled.problems
    assert (problem.type, problem.loc, problem.ctx) == ("several_edges_into_one_input", ("m", "text"), {"node_id": "m", "field": "text", "sources": ["a.result", "b.result"]})


def test_two_branches_wired_into_one_input_merge_back_per_item_and_the_chain_stays_per_item():
    run = ran({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "u", "node": "upper"}, {"id": "after", "node": "upper"},
              edges=(("s.result", "d.text"), ("d.yes", "u.text"), ("d.no", "u.text"), ("u.result", "after.text")), text="ok,no,fine")

    assert run.results["u"]["result"] == ["OK", "NO", "FINE"]
    assert run.graph.runs_on["after"].msg == "per item of s.result"


def test_nodes_downstream_of_two_branches_merge_back_too():
    """Nodes downstream of opposite branches never both have a value, as the branches never do."""
    run = ran({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "y", "node": "upper"}, {"id": "n", "node": "upper"}, {"id": "u", "node": "upper"},
              edges=(("s.result", "d.text"), ("d.yes", "y.text"), ("d.no", "n.text"), ("y.result", "u.text"), ("n.result", "u.text")), text="ok,no")

    assert run.results["u"]["result"] == ["OK", "NO"]


def test_a_node_reading_both_branches_through_one_input_lies_on_neither_so_it_pairs_only_with_items_that_have_both():
    run = ran({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "p", "node": "pair"},
              edges=(("s.result", "d.text"), ("d.yes", "p.a"), ("d.no", "p.a"), ("d.yes", "p.b")), text="ok,no,fine")

    assert run.results["p"]["result"] == ["ok:ok", None, "fine:fine"]


def test_a_series_input_fed_by_both_branches_shares_items_with_one_fed_by_either():
    assert problems({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "z", "node": "zipped"},
                    edges=(("s.result", "d.text"), ("d.yes", "z.a"), ("d.no", "z.a"), ("d.yes", "z.b")),
                    registry=registry_of(split, decide, zipped)) == []


def test_opposite_branches_through_one_series_input_are_one_series_in_item_order():
    """``join(d.no, d.yes)`` reads the items in their order, not one branch's items after the other's."""
    run = ran({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "j", "node": "join"},
              edges=(("s.result", "d.text"), ("d.no", "j.texts"), ("d.yes", "j.texts")), text="v10,no11,v12")

    assert run.results["j"]["result"] == "v10+no11+v12"


def test_alternatives_per_item_of_different_series_into_one_input_are_a_problem_never_a_value_lost():
    """``d.yes`` split into words and ``d.no`` as it is: a document that went ``no`` has no words to put its value on."""
    @node("words-of", title="Words of", description="For a test.")
    def words_of(text: str) -> Series[str]:
        return text.split("+")

    assert problems({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "w", "node": "words-of"}, {"id": "m", "node": "upper"},
                    edges=(("s.result", "d.text"), ("d.yes", "w.text"), ("w.result", "m.text"), ("d.no", "m.text")),
                    registry=registry_of(split, decide, words_of, upper)) == ["alternatives_from_different_loops"]


def test_a_node_returning_values_on_two_branches_fails_with_invalid_output():
    @node("both", title="Both", description="For a test.")
    def both(text: str) -> YesNo:
        return YesNo(text, text)

    run = graph({"id": "b", "node": "both"}, registry=registry_of(both)).run(text="a")

    assert (run.status, run.state.cause.type, run.state.cause.loc, run.state.cause.ctx["name"]) == ("failed", "invalid_output", ("b",), "no")


class Line(TypedDict):
    text: str


@dataclass(frozen=True)
class BigOrSmall:
    big: Annotated[Table[Line], Param(exclusive_group="size")]
    small: Annotated[Table[Line], Param(exclusive_group="size")]


@node("sort-out", title="Sort out", description="A table of the text's words, as ``big`` when there are more than two, else ``small``.")
def sort_out(text: str) -> BigOrSmall:
    """A table of the text's words, as ``big`` when there are more than two, else ``small``."""
    words = pl.DataFrame({"text": text.split()})
    return BigOrSmall(words, SKIPPED) if words.height > 2 else BigOrSmall(SKIPPED, words)


@node("words", title="Words", description="For a test.")
def words(table: Table[Line]) -> str:
    return "-".join(table["text"])


@node("lines-of", title="Lines of", description="For a test.")
def lines_of(text: str) -> Table[Line]:
    return pl.DataFrame({"text": [text.upper()]})


ALTERNATIVES = registry_of(split, sort_out, words, lines_of, upper)


def test_alternative_tables_into_one_table_input_are_the_branch_that_ran():
    nodes = ({"id": "o", "node": "sort-out"}, {"id": "w", "node": "words"})
    edges = (("o.big", "w.table"), ("o.small", "w.table"))

    assert ran(*nodes, edges=edges, registry=ALTERNATIVES, text="a b c").outputs == {"result": "a-b-c"}
    assert ran(*nodes, edges=edges, registry=ALTERNATIVES, text="a b").outputs == {"result": "a-b"}


def test_alternative_tables_per_item_are_the_branch_each_item_chose_also_after_json():
    run = ran({"id": "s", "node": "split"}, {"id": "o", "node": "sort-out"}, {"id": "w", "node": "words"},
              edges=(("s.result", "o.text"), ("o.big", "w.table"), ("o.small", "w.table")), registry=ALTERNATIVES, text="a b c,d e")

    assert run.results["w"]["result"] == ["a-b-c", "d-e"]
    assert through_json(run).results["w"] == run.results["w"]


def test_tables_downstream_of_two_branches_into_one_table_input_are_the_branch_that_ran():
    nodes = ({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "y", "node": "lines-of"}, {"id": "n", "node": "lines-of"},
             {"id": "w", "node": "words"})
    edges = (("s.result", "d.text"), ("d.yes", "y.text"), ("d.no", "n.text"), ("y.result", "w.table"), ("n.result", "w.table"))
    reg = registry_of(split, decide, lines_of, words)

    assert ran(*nodes, edges=edges, registry=reg, text="ok,no").results["w"]["result"] == ["OK", "NO"]


def test_a_table_not_returned_on_the_branch_that_ran_is_none_as_from_a_single_source():
    @node("nothing", title="Nothing", description="For a test.")
    def nothing(text: str) -> str | None:
        return None

    @node("lines-with", title="Lines with", description="For a test.")
    def lines_with(text: str, extra: str) -> Table[Line]:
        return pl.DataFrame({"text": [text]})

    reg = registry_of(decide, nothing, lines_with, lines_of, words)
    nodes = ({"id": "d", "node": "decide"}, {"id": "n", "node": "nothing"}, {"id": "y", "node": "lines-with"}, {"id": "w", "node": "words"})
    edges = (("d.yes", "y.text"), ("n.result", "y.extra"), ("y.result", "w.table"))
    given = {"ports": {"inputs": {"decided": "d.text", "nothing": "n.text"}}, "decided": "ok", "nothing": "z"}
    one_source = ran(*nodes, edges=edges, registry=reg, **given)
    alternatives = ran(*nodes, {"id": "o", "node": "lines-of"}, edges=(*edges, ("d.no", "o.text"), ("o.result", "w.table")), registry=reg, **given)

    for run in (one_source, alternatives):
        assert run.results["w"] == {"result": None}
        assert NotCalled(loc=("w",), input="table") in run.not_called


def test_two_tables_that_can_both_have_a_value_into_one_table_input_are_a_problem():
    assert problems({"id": "a", "node": "lines-of"}, {"id": "b", "node": "lines-of"}, {"id": "w", "node": "words"},
                    edges=(("a.result", "w.table"), ("b.result", "w.table")), registry=ALTERNATIVES, inputs={"text": ["a.text", "b.text"]}) == ["several_edges_into_one_input"]


# -- a list node per document ------------------------------------------------------------------


@node("questions", title="Questions", description="The constant questions, as a Series; wired to ``for_each``, it is called once per item.")
def questions(items: list[str], for_each: Any = None) -> Series[str]:
    return items


@node("answer", title="Answer", description="For a test.", asks=True)
def answer(doc: str, question: str) -> bool:
    raise Ask(f"{doc}: {question}")


ASKING = registry_of(split, decide, questions, answer)
QUESTIONS = {"id": "l", "node": "questions", "constants": {"items": ["a", "b", "c"]}}


def test_questions_listed_per_document_are_asked_of_each_document_and_each_answer_goes_to_its_question():
    """Three questions, two documents: six questions asked, and each answer goes to the call that asked it."""
    compiled = graph({"id": "d", "node": "split"}, QUESTIONS, {"id": "q", "node": "answer"},
                   edges=(("d.result", "l.for_each"), ("d.result", "q.doc"), ("l.result", "q.question")), registry=ASKING)
    assert compiled.problems == []
    run = compiled.run(text="x,y")

    assert compiled.runs_on["q"].msg == "per item of l.result"
    assert [w.prompt for w in run.waiting] == ["x: a", "x: b", "x: c", "y: a", "y: b", "y: c"]
    assert [w.loc[1:] for w in run.waiting] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    saved = Run.from_json(run.to_json(), compiled)
    done = saved.answer({w.id: w.loc[1:] == (1, 2) for w in saved.waiting}).resume()
    assert done.results["q"]["result"] == [[False, False, False], [False, False, True]]


def test_questions_listed_once_beside_documents_are_two_series_that_never_pair_up():
    assert problems({"id": "d", "node": "split"}, QUESTIONS, {"id": "q", "node": "answer"},
                    edges=(("d.result", "q.doc"), ("l.result", "q.question")), registry=ASKING) == ["series_not_nested"]


def test_no_documents_ask_no_questions_and_a_skipped_document_asks_none_of_its_own():
    nothing = ran({"id": "d", "node": "split"}, QUESTIONS, {"id": "q", "node": "answer"},
                  edges=(("d.result", "l.for_each"), ("d.result", "q.doc"), ("l.result", "q.question")), registry=ASKING, text="")
    assert nothing.results["q"]["result"] == []

    skipped = graph({"id": "d", "node": "split"}, {"id": "y", "node": "decide"}, QUESTIONS, {"id": "q", "node": "answer"},
                    edges=(("d.result", "y.text"), ("y.yes", "l.for_each"), ("y.yes", "q.doc"), ("l.result", "q.question")), registry=ASKING).run(text="ok,no")
    assert [w.prompt for w in skipped.waiting] == ["ok: a", "ok: b", "ok: c"]


# -- the output of a node called inside two loops ------------------------------------------------


@node("documents", title="Documents", description="For a test.")
def _documents(text: str) -> Series[str | None]:
    return [document or None for document in text.split("|")]


@node("lines", title="Lines", description="For a test.")
def _lines(document: str) -> Series[str]:
    return document.split(",") if document != "-" else []


@node("shout", title="Shout", description="For a test.")
def _shout(line: str) -> str:
    return line.upper()


@node("words-of", title="Words of", description="For a test.")
def _words_of(line: str) -> Series[str]:
    return line.split()


_PER_LINE = make_graph(_documents, _lines, _shout, edges=[("documents.result", "lines.document"), ("lines.result", "shout.line")])


def test_an_output_inside_two_loops_is_a_list_per_outer_item_and_none_where_no_inner_series_was_returned():
    """Three documents split into lines, the second empty: ``None`` for it, where a document with no lines shows ``[]``."""
    registry = registry_of(_documents, _lines, _shout, _words_of)
    flat = Graph.compile(_PER_LINE, registry).run(text="a,b||c")

    assert flat.results["shout"] == {"result": [["A", "B"], None, ["C"]]}
    assert flat.outputs == {"result": [["A", "B"], None, ["C"]]} and json.loads(flat.outputs_to_json()) == {"result": [["A", "B"], None, ["C"]]}
    assert Graph.compile(_PER_LINE, registry).run(text="a,b|-|c").results["shout"] == {"result": [["A", "B"], [], ["C"]]}
    assert Graph.compile(_PER_LINE, registry).with_inputs(text="a,b||c").results["shout"] == {"result": None}
    assert through_json(flat).results == flat.results

    per_line = subgraph("per-line", title="Per line", description="For a test.")(make_graph(_lines, _shout, edges=[("lines.result", "shout.line")]))
    outer_graph = Graph.compile(make_graph(_documents, per_line, edges=[("documents.result", "per-line.document")]), registry_of(_documents, _lines, _shout, per_line))
    outer_run = outer_graph.run(text="a,b||c")
    assert outer_run.results["per-line"] == {"result": [["A", "B"], None, ["C"]]}
    assert json.loads(outer_run.outputs_to_json()) == {"result": [["A", "B"], None, ["C"]]}

    per_word = make_graph(_documents, _lines, _words_of, _shout, edges=[("documents.result", "lines.document"), ("lines.result", "words-of.line"), ("words-of.result", "shout.line")])
    assert Graph.compile(per_word, registry).run(text="a b,c||d").results["shout"] == {"result": [[["A", "B"], ["C"]], None, [["D"]]]}


# -- the item limit ----------------------------------------------------------------------------


def test_a_series_longer_than_the_item_limit_stops_the_run_before_any_call_in_its_loop():
    CALLED.clear()
    compiled = graph({"id": "s", "node": "split"}, {"id": "u", "node": "upper"}, edges=(("s.result", "u.text"),))

    stopped = compiled.run(text="a,b,c", max_items=2)

    assert (stopped.status, stopped.state.cause.type, stopped.state.cause.loc) == ("stopped", "item_limit", ("s",))
    assert stopped.state.cause.ctx == {"series": "s.result", "items": 3, "max_items": 2} and CALLED == []
    assert stopped.with_max_items(3).resume().results["u"]["result"] == ["A", "B", "C"]


def test_with_max_items_is_the_same_run_with_a_higher_item_limit_and_calls_nothing():
    CALLED.clear()
    compiled = graph({"id": "s", "node": "split"}, {"id": "u", "node": "upper"}, edges=(("s.result", "u.text"),))
    stopped = compiled.run(text="a,b,c", max_items=2)

    raised = stopped.with_max_items(3)

    assert (raised.max_items, stopped.max_items) == (3, 2) and CALLED == []
    assert raised.state == stopped.state and raised.results == stopped.results and raised.inputs == stopped.inputs
    assert json.loads(raised.to_json()) == {**json.loads(stopped.to_json()), "max_items": 3}
    assert Run.from_json(raised.to_json(), compiled).resume().status == "complete"
    with pytest.raises(InvalidCallError) as refused:
        stopped.with_max_items(0)
    assert [(p.type, p.loc) for p in refused.value.problems] == [("invalid_max_items", ("max_items",))]
    for method in (stopped.resume, stopped.stream):  # the item limit is the run's, never one resume() call's
        with pytest.raises(TypeError):
            method(max_items=3)


@pytest.mark.parametrize("max_items", [0, -1, True, False, 2.5, "5"])
def test_max_items_is_a_whole_number_of_at_least_one_and_a_bool_is_not_one(max_items: Any) -> None:
    compiled = graph({"id": "s", "node": "split"}, {"id": "u", "node": "upper"}, edges=(("s.result", "u.text"),))
    calls = [
        lambda: compiled.with_inputs(text="a", max_items=max_items),
        lambda: compiled.run(text="a", max_items=max_items),
        lambda: compiled.with_inputs(text="a").with_max_items(max_items),
        lambda: compiled.with_inputs(text="a").with_inputs(max_items=max_items),
        lambda: compiled.with_inputs(text="a").again(max_items=max_items),
    ]
    for call in calls:
        with pytest.raises(InvalidCallError) as refused:
            call()
        assert [(p.type, p.loc, p.ctx) for p in refused.value.problems] == [("invalid_max_items", ("max_items",), {"max_items": max_items})]
    assert refused.value.problems[0].msg == f"max_items must be a whole number of at least 1; it was {max_items}."


def test_the_item_limit_is_on_each_series_never_on_all_their_items_together():
    compiled = graph({"id": "d", "node": "split", "constants": {"sep": "|"}}, {"id": "l", "node": "split"}, {"id": "u", "node": "upper"},
                   edges=(("d.result", "l.text"), ("l.result", "u.text")))

    assert compiled.run(text="a,b|c,d|e,f", max_items=3).status == "complete"  # six lines in all, never more than three in one Series
    stopped = compiled.run(text="a,b|c,d,e,f|g", max_items=3)
    assert (stopped.status, stopped.state.cause.ctx) == ("stopped", {"series": "l.result", "items": 4, "max_items": 3})


@pytest.mark.timeout(10)
def test_the_item_limit_stops_a_run_whose_calls_wait_for_threads_and_resume_calls_each_item_once():
    calls: list[str] = []

    @node("explode", title="Explode", description="For a test.", policy=Policy(concurrency=2))
    def explode(letter: str) -> Series[str]:
        calls.append(letter)
        return [letter + str(i) for i in range(20 if letter == "c" else 2)]

    compiled = graph({"id": "s", "node": "split"}, {"id": "e", "node": "explode"}, {"id": "u", "node": "upper"},
                   edges=(("s.result", "e.letter"), ("e.result", "u.text")), registry=registry_of(split, explode, upper))

    stopped = compiled.run(text=",".join("abcdefghij"), max_items=10)

    assert (stopped.status, stopped.state.cause.type) == ("stopped", "item_limit") and len(calls) < 10
    assert stopped.with_max_items(100).resume().status == "complete" and sorted(calls) == list("abcdefghij")


# -- a subgraph is one node -------------------------------------------------------------

def as_subgraph(node_id: str, *nodes: dict[str, Any], edges: tuple[tuple[str, str], ...] = (), registry: Any = REGISTRY, **ports: Any) -> Any:
    """A subgraph definition ``node_id`` of ``record_of(nodes, edges, inputs, outputs)``."""
    title = node_id.replace("-", " ").replace("_", " ").strip().capitalize()
    return subgraph(node_id, title=title, description="For a test.")(record_of(*nodes, edges=edges, registry=registry, **ports))


def test_two_series_into_one_subgraph_node_are_one_problem_with_the_outer_graphs_addresses():
    inner = as_subgraph("pairs", {"id": "p", "node": "pair"})
    compiled = graph({"id": "d1", "node": "split"}, {"id": "d2", "node": "split"}, {"id": "sub", "node": "pairs"},
                   edges=(("d1.result", "sub.a"), ("d2.result", "sub.b")), registry=registry_of(split, pair, inner),
                   inputs={"text": ["d1.text", "d2.text"], "sep": ["d1.sep", "d2.sep"]})
    (problem,) = compiled.problems
    assert (problem.type, problem.loc, problem.ctx) == ("series_not_nested", ("sub",), {"node_id": "sub", "series": ["d1.result", "d2.result"], "inputs": ["a", "b"]})
    assert "sub/" not in problem.msg


def test_two_series_out_of_one_subgraph_are_a_problem_where_they_meet():
    @dataclass(frozen=True)
    class Left:
        a: Series[str]

    @dataclass(frozen=True)
    class Right:
        b: Series[str]

    @node("left", title="Left", description="For a test.")
    def left() -> Left:
        return Left(["x", "y"])

    @node("right", title="Right", description="For a test.")
    def right() -> Right:
        return Right(["p", "q"])

    inner = as_subgraph("two-lists", {"id": "l", "node": "left"}, {"id": "r", "node": "right"}, registry=registry_of(left, right))
    assert problems({"id": "sub", "node": "two-lists"}, {"id": "after", "node": "pair"}, edges=(("sub.a", "after.a"), ("sub.b", "after.b")),
                    registry=registry_of(left, right, pair, inner)) == ["series_not_nested"]


def test_a_loop_over_a_series_returned_inside_a_subgraph_is_named_by_its_inner_address():
    inner = as_subgraph("splitup", {"id": "split", "node": "split"}, {"id": "upper", "node": "upper"}, edges=(("split.result", "upper.text"),))
    compiled = graph({"id": "e", "node": "splitup"}, {"id": "split", "node": "split"}, {"id": "u2", "node": "upper"},
                   edges=(("e.result", "u2.text"),), registry=registry_of(split, upper, inner),
                   inputs={"text": ["e.text", "split.text"], "sep": ["e.sep", "split.sep"]}, outputs={"split": "split.result", "upper": "u2.result"})

    assert {k: v.msg for k, v in compiled.runs_on.items()} == {"e": "once", "split": "once", "u2": "per item of e/split.result"}
    assert compiled.describe().nodes["u2"].inputs[0].source.series == "e/split.result"


def test_the_exclusive_groups_of_a_subgraph_name_the_outputs_it_offers():
    inner = as_subgraph("decider", {"id": "d", "node": "decide"})
    compiled = graph({"id": "e", "node": "decider"}, registry=registry_of(decide, inner))

    assert compiled.exclusive_groups == {"e": {"d.answer": ["no", "yes"]}}


TWO_DECISIONS = {
    "inputs": {"first": "d1.text", "second": "d2.text"},
    "outputs": {"first-yes": "d1.yes", "first-no": "d1.no", "second-yes": "d2.yes", "second-no": "d2.no"},
}


def test_two_branching_nodes_inside_one_subgraph_are_two_exclusive_groups():
    inner = as_subgraph("two", {"id": "d1", "node": "decide"}, {"id": "d2", "node": "decide"}, **TWO_DECISIONS)
    compiled = graph({"id": "e", "node": "two"}, registry=registry_of(decide, inner))

    assert compiled.exclusive_groups == {"e": {"d1.answer": ["first-no", "first-yes"], "d2.answer": ["second-no", "second-yes"]}}
    assert [f.exclusive_group for f in compiled.describe().nodes["e"].outputs] == ["d1.answer", "d1.answer", "d2.answer", "d2.answer"]


def test_two_branching_nodes_inside_one_subgraph_each_choose_their_own_branch():
    inner = as_subgraph("two", {"id": "d1", "node": "decide"}, {"id": "d2", "node": "decide"}, **TWO_DECISIONS)
    compiled = graph({"id": "e", "node": "two"}, registry=registry_of(decide, inner))

    assert compiled.run(first="yes", second="no").outputs == {"first-yes": "yes", "first-no": None, "second-yes": None, "second-no": "no"}


def test_a_subgraph_node_wired_from_upstream_needs_every_other_input_wired_too():
    inner = as_subgraph("pairer", {"id": "p", "node": "pair"})
    record = GraphRecord.model_validate({
        "nodes": {"u": {"node": "upper", "inputs": {"text": {"graph_input": "text"}}},
                  "e": {"node": "pairer", "inputs": {"a": {"sources": [{"node": "u", "output": "result"}]}}}},
        "outputs": {"p": {"node": "e", "output": "result"}},
    })
    compiled = Graph.compile(record, registry_of(upper, pair, inner))

    assert [(p.type, p.loc) for p in compiled.problems] == [("input_has_no_value", ("e", "b"))]


def _nested() -> Any:
    """The graph of one node, ``outer``, a subgraph of ``two`` (a subgraph of two nodes) and one node more; each
    record names its inputs and outputs after the inner names it reads."""
    two = as_subgraph("two", {"id": "x", "node": "upper"}, {"id": "y", "node": "upper"},
                      inputs={"first": "x.text", "second": "y.text"}, outputs={"first": "x.result", "second": "y.result"})
    outer = as_subgraph("outer", {"id": "f", "node": "two"}, {"id": "z", "node": "upper"}, registry=registry_of(upper, two),
                        inputs={"a": "f.first", "b": "f.second", "c": "z.text"}, outputs={"a": "f.first", "b": "f.second", "c": "z.result"})
    return graph({"id": "o", "node": "outer"}, registry=registry_of(upper, two, outer))


def test_a_nested_subgraph_names_its_inputs_and_outputs_by_the_names_each_graph_record_declares():
    compiled = _nested()

    assert compiled.problems == [] and sorted(compiled.inputs) == ["a", "b", "c"]
    assert [o.name for o in compiled.describe().nodes["o"].outputs] == ["a", "b", "c"]


def test_a_nested_subgraph_runs_with_its_inputs_and_outputs_named_by_each_graph_record():
    assert _nested().run(a="a", b="b", c="c").results["o"] == {"a": "A", "b": "B", "c": "C"}


def test_a_series_returned_inside_a_subgraph_in_a_loop_is_returned_per_item_wired_or_constant():
    wired_inside = as_subgraph("splitter", {"id": "lines", "node": "split", "constants": {"sep": "/"}}, {"id": "join", "node": "join"}, edges=(("lines.result", "join.texts"),))
    run = ran({"id": "docs", "node": "split"}, {"id": "sub", "node": "splitter"}, edges=(("docs.result", "sub.text"),),
              registry=registry_of(split, join, wired_inside), text="a/b,c")
    assert run.graph.runs_on["sub"].msg == "per item of docs.result"
    assert run.results["sub"]["result"] == ["a+b", "c"]

    constant_inside = as_subgraph("splitter-constant", {"id": "entered", "node": "upper"}, {"id": "lines", "node": "split", "constants": {"text": "x/y", "sep": "/"}},
                     {"id": "join", "node": "join"}, {"id": "p", "node": "pair"},
                     edges=(("lines.result", "join.texts"), ("entered.result", "p.a"), ("join.result", "p.b")))
    run = ran({"id": "docs", "node": "split"}, {"id": "sub", "node": "splitter-constant"}, edges=(("docs.result", "sub.text"),),
              registry=registry_of(split, join, upper, pair, constant_inside), text="a,b")
    assert run.results["sub"]["result"] == ["A:x+y", "B:x+y"]


def test_an_inner_output_in_a_loop_inside_is_a_series_for_each_item_of_the_outer_loop():
    """Inside, ``p`` reads the subgraph's input and each item of ``lines``: the subgraph gives a list per outer item."""
    inner = as_subgraph("letters", {"id": "entered", "node": "upper"}, {"id": "lines", "node": "split", "constants": {"text": "x/y", "sep": "/"}},
                     {"id": "p", "node": "pair"}, edges=(("entered.result", "p.a"), ("lines.result", "p.b")))
    run = ran({"id": "docs", "node": "split"}, {"id": "sub", "node": "letters"}, edges=(("docs.result", "sub.text"),),
              registry=registry_of(split, upper, pair, inner), text="a,b")
    assert run.results["sub"]["result"] == [["A:x", "A:y"], ["B:x", "B:y"]]


def test_inputs_from_a_document_and_from_its_lines_call_the_subgraph_per_line():
    @node("shout", title="Shout", description="For a test.")
    def shout(word: str) -> str:
        return word.upper()

    inner = as_subgraph("pair-up", {"id": "ua", "node": "shout"}, {"id": "ub", "node": "upper"}, {"id": "p", "node": "pair"},
                        edges=(("ua.result", "p.a"), ("ub.result", "p.b")), registry=registry_of(shout, upper, pair))
    run = ran({"id": "docs", "node": "split", "constants": {"sep": "|"}}, {"id": "lines", "node": "split"}, {"id": "sub", "node": "pair-up"},
              edges=(("docs.result", "lines.text"), ("docs.result", "sub.word"), ("lines.result", "sub.text")),
              registry=registry_of(split, upper, pair, shout, inner), text="a,b|c")
    assert run.graph.runs_on["sub"].msg == "per item of lines.result"
    assert run.results["sub"]["result"] == [["A,B:A", "A,B:B"], ["C:C"]]


def test_a_series_into_a_subgraphs_series_input_is_collected_and_the_subgraph_runs_once():
    inner = as_subgraph("joiner", {"id": "join", "node": "join"})
    run = ran({"id": "docs", "node": "split"}, {"id": "sub", "node": "joiner"}, edges=(("docs.result", "sub.texts"),),
              registry=registry_of(split, join, inner), text="a,b")
    assert run.graph.runs_on["sub"].msg == "once" and run.results["sub"]["result"] == "a+b"


def test_a_subgraph_reading_a_series_whole_and_per_item_gets_the_whole_series_on_each_call():
    inner = as_subgraph("mixed", {"id": "u", "node": "upper"}, {"id": "j", "node": "join"}, outputs={"u": "u.result", "j": "j.result"})
    run = ran({"id": "docs", "node": "split"}, {"id": "sub", "node": "mixed"}, edges=(("docs.result", "sub.text"), ("docs.result", "sub.texts")),
              registry=registry_of(split, upper, join, inner), text="a,b")
    assert set(run.graph.runs_on) == {"docs", "sub"} and run.graph.runs_on["sub"].msg == "per item of docs.result"
    assert run.results["sub"] == {"u": ["A", "B"], "j": ["a+b", "a+b"]}


def test_a_subgraph_in_a_loop_gives_what_its_nodes_give_in_the_outer_graph_itself():
    nodes = ({"id": "i", "node": "split", "constants": {"sep": ";"}}, {"id": "c", "node": "split", "constants": {"sep": "/"}}, {"id": "p", "node": "pair"})
    edges = (("i.result", "c.text"), ("c.result", "p.a"), ("i.result", "p.b"))
    inner = as_subgraph("deep", *nodes, edges=edges)
    reg = registry_of(split, pair, inner)
    subgraph_run = ran({"id": "s", "node": "split"}, {"id": "e", "node": "deep"}, edges=(("s.result", "e.text"),), registry=reg, text="a/b;c,d")
    flat_run = ran({"id": "s", "node": "split"}, *nodes, edges=(("s.result", "i.text"), *edges), registry=reg, text="a/b;c,d")
    assert subgraph_run.results["e"]["result"] == flat_run.results["p"]["result"]
    assert subgraph_run.graph.runs_on["e"].msg == "per item of s.result"  # the loop it is wired into; its own loops are inside it


def test_an_inner_node_collecting_the_subgraphs_input_gets_a_series_of_one_per_outer_item():
    inner = as_subgraph("inner", {"id": "entering", "node": "upper"}, {"id": "collect", "node": "join"}, edges=(("entering.result", "collect.texts"),))
    run = ran({"id": "docs", "node": "split"}, {"id": "sub", "node": "inner"}, {"id": "after", "node": "join"},
              edges=(("docs.result", "sub.text"), ("sub.result", "after.texts")), registry=registry_of(split, upper, join, inner), text="a,b,c")
    assert run.graph.runs_on["sub"].msg == "per item of docs.result"
    assert run.results["sub"]["result"] == ["A", "B", "C"]  # one item collected per call, never all three in one call
    assert run.results["after"]["result"] == "A+B+C"


def test_a_branch_not_chosen_inside_a_subgraph_skips_its_inner_collecting_node_per_outer_item():
    inner = as_subgraph("skips", {"id": "b", "node": "decide"}, {"id": "j", "node": "join"}, edges=(("b.no", "j.texts"),))
    run = ran({"id": "s", "node": "split"}, {"id": "e", "node": "skips"}, edges=(("s.result", "e.text"),), registry=registry_of(split, decide, join, inner), text="p,q")
    assert run.results["e"]["result"] == [None, None]


def test_a_subgraph_with_an_output_per_inner_item_and_one_per_call_runs_per_item_wired_into_it():
    @node("names", title="Names", description="For a test.")
    def names(text: str) -> Series[str]:
        return text.split()

    @node("case", title="Case", description="For a test.")
    def case(value: str) -> str:
        return value

    @node("greet", title="Greet", description="For a test.")
    def greet(name: str, case: str) -> str:
        return f"{name} {case}"

    @node("shout", title="Shout", description="For a test.")
    def shout(text: str) -> str:
        return text.upper()

    @node("count", title="Count", description="For a test.")
    def count(texts: Series[str]) -> int:
        return len(texts)

    inner = make_graph(
        names, case, greet, count, shout,
        edges=[("names.result", "greet.name"), ("case.result", "greet.case"), ("greet.result", "count.texts"), ("greet.result", "shout.text")],
        constants={"names.text": "Ida Bo"},
        outputs={"count": "count.result", "shout": "shout.result"},
    )
    registry = registry_of(split, names, case, greet, count, shout)
    registry.add(subgraph("inner", title="Inner", description="For a test.")(inner))
    graph = Graph.compile(make_graph(split, registry["inner"], edges=[("split.result", "inner.value")], constants={"split.text": "A-1,B-2"}), registry)
    assert graph.runs_on["inner"].series == "split.result"
    assert graph.run().results["inner"] == {"count": [2, 2], "shout": [["IDA A-1", "BO A-1"], ["IDA B-2", "BO B-2"]]}


@node("each", title="Each", description="For a test.")
def each(texts: Series[str]) -> Series[str]:
    return texts


def test_an_inner_node_collecting_an_inner_series_and_the_subgraph_input_gets_the_same_in_either_node_order():
    """``c`` collects the lines split inside and the texts wired into the subgraph; the order the graph record lists
    the inner nodes in changes nothing about what each input gets."""
    inner_nodes = {
        "lines": {"id": "lines", "node": "split", "constants": {"text": "x/y", "sep": "/"}},
        "ent": {"id": "ent", "node": "each"},
        "c": {"id": "c", "node": "two-series"},
    }
    found = []
    for order in (("lines", "ent", "c"), ("c", "ent", "lines")):
        inner = as_subgraph("inner", *(inner_nodes[n] for n in order), edges=(("lines.result", "c.a"), ("ent.result", "c.b")), registry=registry_of(split, each, two_series))
        compiled = graph({"id": "docs", "node": "split"}, {"id": "sub", "node": "inner"}, edges=(("docs.result", "sub.texts"),),
                         registry=registry_of(split, each, two_series, inner))
        assert compiled.problems == []
        found.append(compiled.run(text="a,b").outputs)
    assert found[0] == found[1] == {"result": "x,y+a,b"}


def test_a_subgraph_node_listed_before_its_upstream_node_compiles_and_runs_the_same():
    """The node upstream of a subgraph node may come after it in the graph record."""
    inner = as_subgraph(
        "letters", {"id": "entered", "node": "upper"}, {"id": "lines", "node": "split", "constants": {"text": "x/y", "sep": "/"}},
        {"id": "p", "node": "pair"}, edges=(("entered.result", "p.a"), ("lines.result", "p.b")),
    )
    registry = registry_of(split, upper, pair, inner)
    found = []
    docs = {"node": "split", "inputs": {"text": {"graph_input": "text"}}}
    sub = {"node": "letters", "inputs": {"text": {"sources": [{"node": "docs", "output": "result"}]}}}
    for outer in ({"docs": docs, "sub": sub}, {"sub": sub, "docs": docs}):
        compiled = Graph.compile(GraphRecord.model_validate({"nodes": outer, "outputs": {"result": {"node": "sub", "output": "result"}}}), registry)
        assert compiled.problems == []
        found.append(compiled.run(text="a,b").outputs)
    assert found[0] == found[1]


# -- three graphs an author writes --------------------------------------------------------------


class Named(TypedDict):
    name: str


def test_a_contact_with_no_email_is_not_mailed_and_keeps_its_row_in_the_collected_table_with_none():
    """Contacts as a table, split to records, a mail per email, and the records collected with what each mail gave."""
    mailed: list[str] = []

    @node("mail", title="Mail", description="Sends one mail.")
    def mail(email: str) -> bool:
        mailed.append(email)
        return True

    @node("collect", title="Collect", description="The records as a table, with a column saying whether each was mailed.")
    def collect(records: Table[Named], sent: Series[bool | None]) -> Table:
        return records.with_columns(pl.Series("sent", sent, dtype=pl.Boolean))

    compiled = graph(*CONTACTS, {"id": "m", "node": "mail"}, {"id": "k", "node": "collect"},
                   edges=(SPLIT_CONTACTS, ("r.result.email", "m.email"), ("r.result", "k.records"), ("m.result", "k.sent")),
                   registry=registry_of(contacts, records, mail, collect))
    assert compiled.problems == []
    run = compiled.run(text=CSV)

    assert run.status == "complete" and mailed == ["ida@x", "nina@x"]
    assert run.not_called == [NotCalled(loc=("m", 1), input="email")]
    sheet = run.outputs["result"]
    assert sheet.select("name", "sent").to_dicts() == [{"name": "Ida", "sent": True}, {"name": "Bo", "sent": None}, {"name": "Nina", "sent": True}]
    assert through_json(run).outputs["result"].equals(sheet)


def test_a_document_with_no_lines_counts_zero_and_its_count_is_called():
    counted: list[int] = []

    @node("documents", title="Documents", description="The documents of a folder, one per ``|``.")
    def documents(folder: str) -> Series[str]:
        return folder.split("|")

    @node("lines", title="Lines", description="The lines of a document, one per ``,``.")
    def lines(document: str) -> Series[str]:
        return [line for line in document.split(",") if line]

    @node("count-lines", title="Count lines", description="How many lines a document has.")
    def count_lines(items: Series[str]) -> int:
        counted.append(len(items))
        return len(items)

    @node("sum", title="Sum", description="For a test.")
    def sum_(counts: Series[int]) -> int:
        return sum(counts)

    run = ran({"id": "d", "node": "documents"}, {"id": "l", "node": "lines"}, {"id": "c", "node": "count-lines"}, {"id": "t", "node": "sum"},
              edges=(("d.result", "l.document"), ("l.result", "c.items"), ("c.result", "t.counts")),
              registry=registry_of(documents, lines, count_lines, sum_), folder="x,y||z")

    assert run.results["c"]["result"] == [2, 0, 1] and counted == [2, 0, 1]
    assert run.outputs == {"result": 3} and run.not_called == []


def test_an_approval_per_line_of_each_document_pauses_saves_resumes_and_a_new_document_asks_only_its_own_lines():
    asked: list[str] = []

    @node("documents", title="Documents", description="The documents of a folder, one per ``|``.")
    def documents(folder: str) -> Series[str]:
        return folder.split("|")

    @node("lines", title="Lines", description="The lines of a document, one per ``,``.")
    def lines(document: str) -> Series[str]:
        return document.split(",")

    @node("approve", title="Approve", description="Asks a person about one line.", asks=True)
    def approve(line: str) -> Literal["yes", "no"]:
        asked.append(line)
        raise Ask(f"approve {line}?")

    @node("approved", title="Approved", description="How many lines of a document were approved.")
    def approved(answers: Series[str]) -> int:
        return answers.count("yes")

    compiled = graph({"id": "d", "node": "documents"}, {"id": "l", "node": "lines"}, {"id": "a", "node": "approve"}, {"id": "n", "node": "approved"},
                   edges=(("d.result", "l.document"), ("l.result", "a.line"), ("a.result", "n.answers")),
                   registry=registry_of(documents, lines, approve, approved))
    assert compiled.problems == []

    paused = compiled.run(folder="a1,a2|b1")
    assert paused.status == "waiting"
    assert [(w.loc[0], w.loc[1:], w.prompt) for w in paused.waiting] == [
        ("a", (0, 0), "approve a1?"), ("a", (0, 1), "approve a2?"), ("a", (1, 0), "approve b1?"),
    ]

    saved = Run.from_json(paused.to_json(), compiled)
    assert [w.id for w in saved.waiting] == [w.id for w in paused.waiting]
    done = saved.answer({w.id: "no" if w.prompt == "approve a2?" else "yes" for w in saved.waiting}).resume()
    assert done.status == "complete" and done.results["a"]["result"] == [["yes", "no"], ["yes"]]
    assert done.outputs == {"result": [1, 1]}

    asked.clear()
    again = done.again(folder="a1,a2|new1,new2|b1")
    assert again.status == "waiting" and asked == ["new1", "new2"]
    assert [(w.loc[1:], w.prompt) for w in again.waiting] == [((1, 0), "approve new1?"), ((1, 1), "approve new2?")]
    finished = again.answer({w.id: "yes" for w in again.waiting}).resume()
    assert finished.results["a"]["result"] == [["yes", "no"], ["yes", "yes"], ["yes"]]
    assert finished.outputs == {"result": [1, 2, 1]}


# -- a collecting node while an outer value is not known ----------------------------------------


def test_a_collecting_node_over_no_items_waits_for_a_value_its_loop_reads_from_outside():
    """``p`` is called once per item of ``s`` and also reads ``m``, outside the loop. Until ``m`` has a value
    nobody knows whether any call of ``p`` could be from_a_node, so ``n``, which collects ``p``, is not called, even when
    ``s`` has no items: ``m`` may still turn out a branch not chosen, which skips ``n``."""

    @node("mark", title="Mark", description="Asks a person for the mark.", asks=True)
    def mark() -> str:
        raise Ask("which mark?")

    CALLED.clear()
    compiled = graph({"id": "s", "node": "split"}, {"id": "m", "node": "mark"}, {"id": "p", "node": "pair"}, {"id": "n", "node": "count"},
                   edges=(("s.result", "p.a"), ("m.result", "p.b"), ("p.result", "n.values")), registry=registry_of(split, mark, pair, count))
    assert compiled.problems == []
    paused = compiled.run(text="")
    assert paused.status == "waiting" and CALLED == []
    done = paused.answer(paused.waiting[0], "!").resume()
    assert done.status == "complete" and done.results["n"]["result"] == 0 and CALLED == ["count 0"]
