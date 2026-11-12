"""Items and gathers through the public door: blanks, outer values, gathers, branches, tables and subgraphs.

A node at the graph's own scope runs once. A ``Series`` output read by an input of one value runs the reader once
per item, and everything that reads the reader with it; a value made outside those items is there on each call. A
``Series`` or ``Table`` input gathers what its source made, one level up. A table is one value: a node that splits
it returns a ``Series`` of its records. A blank keeps a node from being called for an item unless the input says
``| None``; a branch not taken skips it.
"""
import asyncio
import datetime
import json
import threading
import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Optional, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Ask,
    Graph,
    GraphRecord,
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
    stored_graph,
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


@node("contacts", title="Contacts", description="``Ida,ida@x;Bo,;Nina,nina@x``: a contact per ``;``, a blank email where nothing follows the comma.")
def contacts(text: str) -> Table[Contact]:
    """``Ida,ida@x;Bo,;Nina,nina@x``: a contact per ``;``, a blank email where nothing follows the comma."""
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


@node("invoices", title="Invoices", description="``acme:10,beta:5``: an invoice per ``,``; ``skip`` is a branch not taken.")
def invoices(text: str) -> Table[Invoice]:
    """``acme:10,beta:5``: an invoice per ``,``; ``skip`` is a branch not taken."""
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


def graph(*nodes: dict[str, Any], edges: tuple[tuple[str, str], ...] = (), registry: Any = REGISTRY) -> Any:
    """``nodes`` as ``{"id", "node", ...}`` and edges as ``(from, to)``, loaded; a graph with problems too."""
    return Graph.compile(GraphRecord.model_validate({"nodes": [{"node": n["id"], **n} for n in nodes], "edges": [list(e) for e in edges]}), registry)


def ran(*nodes: dict[str, Any], edges: tuple[tuple[str, str], ...] = (), registry: Any = REGISTRY, **inputs: Any) -> Any:
    """The run of a graph that must load clean and complete."""
    loaded = graph(*nodes, edges=edges, registry=registry)
    assert loaded.problems == [], loaded.problems
    run = loaded.run(**inputs)
    assert run.status == "complete", run.cause
    return run


def problems(*nodes: dict[str, Any], edges: tuple[tuple[str, str], ...] = (), registry: Any = REGISTRY) -> list[str]:
    return [p.type for p in graph(*nodes, edges=edges, registry=registry).problems]


def through_json(run: Any) -> Any:
    return Run.from_json(run.to_json(), run.graph)


def streamed(run: Any) -> list[Any]:
    async def collect() -> list[Any]:
        return [event async for event in run.stream()]

    return asyncio.run(collect())


# -- blanks ------------------------------------------------------------------------------------


def test_a_node_with_no_output_that_returns_skipped_was_called_so_it_completes():
    @node("log", title="Log", description="For a test.")
    def log(text: str) -> None:
        return SKIPPED

    said = streamed(graph({"id": "l", "node": "log"}, registry=registry_of(log)).with_inputs(text="x"))
    assert [(e.type, e.outcomes) for e in said if e.type in ("node_complete", "node_skipped")] == [("node_complete", 1)]


def test_a_series_that_may_be_nothing_reads_an_explicit_none_like_leaving_it_out():
    @node("maybe", title="Maybe", description="For a test.")
    def maybe(xs: Series[str] | None = None) -> str:
        return str(xs)

    reg = registry_of(maybe)
    set_to_none = ran({"id": "m", "node": "maybe", "values": {"xs": None}}, registry=reg)
    left_out = ran({"id": "m", "node": "maybe"}, registry=reg)
    assert set_to_none.outputs == left_out.outputs == {"m.result": "None"}


class Amount(TypedDict):
    a: int


def test_a_table_that_may_be_nothing_takes_a_callers_none_a_return_of_none_and_a_table_never_made():
    @node("count-rows", title="Count rows", description="For a test.")
    def count_rows(t: Table[Amount] | None = None) -> int:
        return -1 if t is None else t.height

    @node("maybe-table", title="Maybe table", description="For a test.")
    def maybe_table(x: int) -> Table[Amount] | None:
        return None if x < 0 else pl.DataFrame({"a": [x]})

    @node("may-be-blank", title="May be blank", description="For a test.")
    def may_be_blank(y: int) -> int | None:
        return None

    alone = graph({"id": "c", "node": "count-rows"}, registry=registry_of(count_rows))
    assert alone.run(t=None).outputs == alone.run().outputs == {"c.result": -1}

    made = graph({"id": "m", "node": "maybe-table"}, {"id": "c", "node": "count-rows"}, edges=(("m.result", "c.t"),), registry=registry_of(maybe_table, count_rows))
    assert made.run(x=-1).outputs == {"c.result": -1} and made.run(x=3).outputs == {"c.result": 1}
    assert Run.from_json(made.run(x=-1).to_json(), made).results["m"] == {"result": None}

    never = graph({"id": "b", "node": "may-be-blank"}, {"id": "m", "node": "maybe-table"}, {"id": "c", "node": "count-rows"},
                  edges=(("b.result", "m.x"), ("m.result", "c.t")), registry=registry_of(may_be_blank, maybe_table, count_rows))
    assert never.run(y=1).outputs == {"c.result": -1}


def test_a_series_that_may_be_nothing_takes_a_return_of_none_and_a_series_never_made_as_a_table_does():
    @node("count-items", title="Count items", description="For a test.")
    def count_items(xs: Series[int] | None = None) -> int:
        return -1 if xs is None else len(xs)

    @node("maybe-items", title="Maybe items", description="For a test.")
    def maybe_items(x: int) -> Series[int] | None:
        return None if x < 0 else [x]

    @node("may-be-blank", title="May be blank", description="For a test.")
    def may_be_blank(y: int) -> int | None:
        return None

    made = graph({"id": "m", "node": "maybe-items"}, {"id": "c", "node": "count-items"}, edges=(("m.result", "c.xs"),), registry=registry_of(maybe_items, count_items))
    assert made.run(x=-1).outputs == {"c.result": -1} and made.run(x=3).outputs == {"c.result": 1}

    never = graph({"id": "b", "node": "may-be-blank"}, {"id": "m", "node": "maybe-items"}, {"id": "c", "node": "count-items"},
                  edges=(("b.result", "m.x"), ("m.result", "c.xs")), registry=registry_of(may_be_blank, maybe_items, count_items))
    assert never.run(y=1).outputs == {"c.result": -1}


@pytest.mark.parametrize(("annotation", "takes_blank"), [
    (str, False),
    (str | None, True),
    (Optional[str], True),  # noqa: UP045  the old spelling says the same
    (Annotated[str | None, Param(title="X")], True),
    (Annotated[str, Param(title="X")] | None, True),
    (Any, True),
    (Series[str], False),
    (Series[str | None], True),
    (Series[Any], True),
])
def test_an_input_takes_a_blank_when_its_hint_says_so(annotation, takes_blank):
    def seen(x):
        return str(x)

    seen.__annotations__ = {"x": annotation, "return": str}
    reader = node("seen", title="Seen", description="For a test.")(seen)
    run = ran(*CONTACTS, {"id": "s", "node": "seen"}, edges=(SPLIT_CONTACTS, ("r.result.email", "s.x")),
              registry=registry_of(contacts, records, reader), text=CSV)
    got = run.results["s"]["result"]
    if isinstance(got, list):  # one value per contact
        assert got[1] == ("None" if takes_blank else None), got
    else:  # a gather over every contact
        assert got == ("['ida@x', None, 'nina@x']" if takes_blank else "['ida@x', 'nina@x']"), got


def test_an_input_saying_none_is_called_with_the_blank():
    run = ran(*CONTACTS, {"id": "s", "node": "send-or-say"}, edges=(SPLIT_CONTACTS, ("r.result.email", "s.email")), text=CSV)
    assert run.results["s"]["result"] == ["sent to ida@x", "no address", "sent to nina@x"]
    assert run.not_called == []


def test_a_default_is_for_an_input_nothing_is_wired_to_not_for_a_blank():
    @node("defaulted", title="Defaulted", description="For a test.")
    def defaulted(email: str = "fallback") -> str:
        return email

    run = ran(*CONTACTS, {"id": "d", "node": "defaulted"}, edges=(SPLIT_CONTACTS, ("r.result.email", "d.email")),
              registry=registry_of(contacts, records, defaulted), text=CSV)
    assert run.results["d"]["result"] == ["ida@x", None, "nina@x"]
    assert run.not_called == [NotCalled(loc=("d", 1), input="email")]


def test_a_node_all_blank_completes_it_is_never_skipped():
    @node("nothing", title="Nothing", description="For a test.")
    def nothing() -> str | None:
        return None

    loaded = graph({"id": "n", "node": "nothing"}, {"id": "s", "node": "send"}, edges=(("n.result", "s.email"),), registry=registry_of(nothing, send))

    said = streamed(loaded.with_inputs())
    assert not [e for e in said if e.type == "node_skipped"]
    (complete,) = [e for e in said if e.type == "node_complete" and e.loc[0] == "s"]
    assert (complete.outcomes, complete.not_called) == (1, 1)
    assert said[-1].status == "complete" and said[-1].run.not_called == [NotCalled(loc=("s",), input="email")]


@pytest.mark.parametrize("blank_into", ["a", "b"])
def test_a_node_with_a_skipped_input_is_skipped_whichever_of_its_inputs_is_blank(blank_into):
    """Whether a node is skipped or uncalled for a blank doesn't hang on the order of its parameters: a skip wins."""
    @node("nothing", title="Nothing", description="For a test.")
    def nothing(text: str) -> str | None:
        return None

    branch_into = "b" if blank_into == "a" else "a"
    run = ran({"id": "d", "node": "decide"}, {"id": "n", "node": "nothing"}, {"id": "p", "node": "pair"}, {"id": "u", "node": "upper"},
              edges=(("d.no", f"p.{branch_into}"), ("n.result", f"p.{blank_into}"), ("p.result", "u.text"), ("d.yes", "u.text")),
              registry=registry_of(decide, nothing, pair, upper), **{"d.text": "ok", "n.text": "z"})
    assert run.results["u"]["result"] == "OK"
    assert run.not_called == []


@pytest.mark.parametrize("waiting_into", ["a", "b"])
def test_a_node_with_a_skipped_input_is_skipped_at_once_whichever_of_its_inputs_waits_for_a_person(waiting_into):
    """A skip wins over an input that isn't there yet too, whatever the order of the parameters: what reads the
    node goes on while the question waits."""
    @node("asking", title="Asking", description="For a test.")
    def asking(text: str) -> str:
        raise Ask(f"What about {text}?")

    branch_into = "b" if waiting_into == "a" else "a"
    loaded = graph({"id": "d", "node": "decide"}, {"id": "q", "node": "asking"}, {"id": "p", "node": "pair"}, {"id": "u", "node": "upper"},
                   edges=(("d.no", f"p.{branch_into}"), ("q.result", f"p.{waiting_into}"), ("p.result", "u.text"), ("d.yes", "u.text")),
                   registry=registry_of(decide, asking, pair, upper))
    run = loaded.run({"d.text": "ok", "q.text": "z"})
    assert run.status == "waiting" and run.results["u"]["result"] == "OK"
    assert json.loads(run.to_json())["outcomes"]["p"]["states"] == ["skipped"]  # settled on this leg, before the answer
    done = run.answer(run.waiting[0], "yes").resume()
    assert done.status == "complete" and done.results["p"]["result"] is None and done.results["u"]["result"] == "OK"


def test_choice_outputs_are_skipped_when_not_called():
    @node("branched", title="Branched", description="For a test.")
    def branched(email: str) -> YesNo:
        return YesNo(email, SKIPPED)

    run = ran(*CONTACTS, {"id": "b", "node": "branched"}, {"id": "u", "node": "upper"},
              edges=(SPLIT_CONTACTS, ("r.result.email", "b.email"), ("b.yes", "u.text")), registry=registry_of(contacts, records, branched, upper), text=CSV)
    assert run.results["b"] == {"yes": ["ida@x", None, "nina@x"], "no": [None, None, None]}
    assert run.results["u"]["result"] == ["IDA@X", None, "NINA@X"]
    assert [(n.loc[0], n.loc[1:]) for n in run.not_called] == [("b", (1,))]  # no branch was chosen there, so its reader is skipped


def test_a_decision_not_called_for_a_blank_item_runs_neither_branch_even_one_that_takes_a_blank():
    """No branch was chosen for the item, so a reader on each branch is skipped there, whatever its input takes."""
    run = ran(*CONTACTS, {"id": "d", "node": "decide"}, {"id": "y", "node": "send-or-say"}, {"id": "n", "node": "send-or-say"}, {"id": "j", "node": "join"},
              edges=(SPLIT_CONTACTS, ("r.result.email", "d.text"), ("d.yes", "y.email"), ("d.no", "n.email"), ("y.result", "j.texts"), ("n.result", "j.texts")), text=CSV)
    assert run.results["d"] == {"yes": ["ida@x", None, "nina@x"], "no": [None, None, None]}
    assert run.results["y"]["result"] == ["sent to ida@x", None, "sent to nina@x"] and run.results["n"]["result"] == [None, None, None]
    assert run.results["j"]["result"] == "sent to ida@x+sent to nina@x"
    assert run.not_called == [NotCalled(loc=("d", 1), input="text")]
    again = through_json(run).again()
    assert (again.results["y"], again.results["n"], again.not_called) == (run.results["y"], run.results["n"], run.not_called)


def test_a_decision_made_once_and_not_called_skips_the_readers_of_its_branches():
    @node("nothing", title="Nothing", description="For a test.")
    def nothing() -> str | None:
        return None

    said = streamed(graph({"id": "n", "node": "nothing"}, {"id": "d", "node": "decide"}, {"id": "y", "node": "send-or-say"},
                          edges=(("n.result", "d.text"), ("d.yes", "y.email")), registry=registry_of(nothing, decide, send_or_say)).with_inputs())
    run = said[-1].run
    assert run.status == "complete" and run.results["y"]["result"] is None and run.not_called == [NotCalled(loc=("d",), input="text")]
    assert [e.loc for e in said if e.type == "node_skipped"] == [("y",)]


def test_a_gather_below_items_that_read_a_table_made_once_is_called_as_below_any_outer_value():
    """Whether a gathering edge is skipped is asked of what holds the outer value, never by comparing the value: a table has no plain ``==``."""
    @node("with-height", title="With height", description="For a test.")
    def with_height(text: str, table: Table) -> str:
        return f"{text}{table.height}"

    run = ran({"id": "c", "node": "contacts"}, {"id": "s", "node": "split"}, {"id": "h", "node": "with-height"}, {"id": "j", "node": "join"},
              edges=(("s.result", "h.text"), ("c.result", "h.table"), ("h.result", "j.texts")), registry=registry_of(contacts, split, with_height, join),
              **{"c.text": CSV, "s.text": "a,b"})
    assert run.status == "complete" and run.results["j"]["result"] == "a3+b3"


@node("maybe-template", title="Maybe template", description="``blank`` gives a blank, ``skip`` a branch not taken.")
def maybe_template(kind: str) -> str | None:
    return SKIPPED if kind == "skip" else None if kind == "blank" else kind


@node("count-texts", title="Count texts", description="For a test.")
def count_texts(values: Series[str]) -> int:
    return len(values)


@node("count-or-none", title="Count or none", description="For a test.")
def count_or_none(values: Series[str] | None) -> str:
    return "none" if values is None else str(len(values))


@node("pair-or-alone", title="Pair or alone", description="For a test.")
def pair_or_alone(a: str, b: str | None) -> str:
    return f"{a}+{b}"


def _below_an_outer_value(kind: str, reader: str = "pair", counter: str = "count-texts") -> Any:
    """Items each paired with one value made outside them (``kind`` says what it is), and a count of the pairs."""
    return ran({"id": "s", "node": "split"}, {"id": "t", "node": "maybe-template"}, {"id": "p", "node": reader}, {"id": "n", "node": counter},
               edges=(("s.result", "p.a"), ("t.result", "p.b"), ("p.result", "n.values")),
               registry=registry_of(split, maybe_template, pair, pair_or_alone, count_texts, count_or_none), **{"s.text": "a,b", "t.kind": kind})


def test_a_blank_made_outside_the_items_leaves_a_gather_below_them_uncalled_as_a_skip_there_skips_it():
    """Nothing was tried for any item, so "0" would be a wrong answer: the blank was decided outside the items."""
    blank = _below_an_outer_value("blank")
    assert blank.status == "complete" and blank.results["n"]["result"] is None
    assert blank.not_called == [NotCalled(loc=("p", 0), input="b"), NotCalled(loc=("p", 1), input="b"), NotCalled(loc=("n",), input="values")]
    assert through_json(blank).again().not_called == blank.not_called
    skipped = _below_an_outer_value("skip")
    assert skipped.results["n"]["result"] is None and skipped.not_called == []
    assert _below_an_outer_value("x").results["n"]["result"] == 2


def test_a_gather_that_may_be_nothing_gets_the_blank_made_outside_the_items():
    assert _below_an_outer_value("blank", counter="count-or-none").results["n"]["result"] == "none"


def test_an_outer_blank_an_input_takes_is_no_blank_for_the_gather_below():
    """The node between is called with the blank for every item, so there are items to gather."""
    run = _below_an_outer_value("blank", reader="pair-or-alone")
    assert run.results["n"]["result"] == 2 and run.not_called == []


def test_a_blank_decided_per_item_is_still_left_out_and_every_item_blank_calls_the_gather_with_an_empty_series():
    run = ran(*CONTACTS, {"id": "u", "node": "upper"}, {"id": "n", "node": "count-texts"},
              edges=(SPLIT_CONTACTS, ("r.result.email", "u.text"), ("u.result", "n.values")), registry=registry_of(contacts, records, upper, count_texts), text="Bo,;Cy,")
    assert run.results["n"]["result"] == 0 and [n.loc for n in run.not_called] == [("u", 0), ("u", 1)]


def test_a_series_not_made_for_a_blank_item_is_not_an_empty_one_so_a_count_of_it_is_not_called_either():
    run = ran(*CONTACTS, {"id": "s", "node": "split"}, {"id": "n", "node": "count"},
              edges=(SPLIT_CONTACTS, ("r.result.email", "s.text"), ("s.result", "n.values")), text=CSV)
    assert run.results["n"]["result"] == [1, None, 1]
    assert NotCalled(loc=("n", 1), input="values") in run.not_called


def test_a_table_not_made_for_a_blank_item_is_not_split_so_a_count_of_its_records_is_not_called_either():
    run = ran(*CONTACTS, {"id": "i", "node": "invoices"}, {"id": "l", "node": "invoice-lines"}, {"id": "n", "node": "count"},
              edges=(SPLIT_CONTACTS, ("r.result.email", "i.text"), ("i.result", "l.table"), ("l.result.vendor", "n.values")), text="Ida,a:1;Bo,")
    assert run.results["n"]["result"] == [1, None]
    assert run.results["i"]["result"][1] is None  # nothing was made for Bo


def test_a_gather_leaves_blanks_out():
    run = ran(*CONTACTS, {"id": "j", "node": "join"}, edges=(SPLIT_CONTACTS, ("r.result.email", "j.texts")), text=CSV)
    assert run.results["j"]["result"] == "ida@x+nina@x"


def test_a_gather_saying_none_keeps_them():
    @node("keep", title="Keep", description="For a test.")
    def keep(values: Series[str | None]) -> int:
        return values.count(None)

    run = ran(*CONTACTS, {"id": "k", "node": "keep"}, edges=(SPLIT_CONTACTS, ("r.result.email", "k.values")),
              registry=registry_of(contacts, records, keep), text=CSV)
    assert run.results["k"]["result"] == 1


@node("zipped", title="Zipped", description="For a test.")
def zipped(a: Series[str], b: Series[str]) -> str:
    return "|".join(x + y for x, y in zip(a, b, strict=True))


def test_two_gathers_of_one_series_lose_a_blank_item_together():
    run = ran(*CONTACTS, {"id": "z", "node": "zipped"}, edges=(SPLIT_CONTACTS, ("r.result.name", "z.a"), ("r.result.email", "z.b")),
              registry=registry_of(contacts, records, zipped), text=CSV)
    assert run.results["z"]["result"] == "Idaida@x|Ninanina@x"


def test_a_reader_of_a_blank_chain_is_not_called_either():
    CALLED.clear()
    run = ran(*CONTACTS, {"id": "u", "node": "upper"}, {"id": "v", "node": "upper"},
              edges=(SPLIT_CONTACTS, ("r.result.email", "u.text"), ("u.result", "v.text")), text=CSV)
    assert run.results["v"]["result"] == ["IDA@X", None, "NINA@X"]
    assert len(CALLED) == 4
    assert [(n.loc[0], n.loc[1:]) for n in run.not_called] == [("u", (1,)), ("v", (1,))]


def test_a_blank_item_is_left_out_of_every_series_input_together_so_a_zip_stays_paired():
    @node("blank-some", title="Blank some", description="For a test.")
    def blank_some(text: str) -> Series[str | None]:
        return [None if x == "_" else x for x in text.split(",")]

    @node("ask", title="Ask", description="For a test.")
    def ask(text: str) -> str | None:
        return text + "?"

    @node("paired", title="Paired", description="For a test.")
    def paired(a: Series[str], b: Series[str | None]) -> str:
        return str(list(zip(a, b, strict=True)))

    run = ran({"id": "a", "node": "blank-some"}, {"id": "b", "node": "ask"}, {"id": "z", "node": "paired"},
              edges=(("a.result", "b.text"), ("a.result", "z.a"), ("b.result", "z.b")), registry=registry_of(blank_some, ask, paired), text="1,_,3")
    assert run.results["z"]["result"] == str([("1", "1?"), ("3", "3?")])


def test_a_gather_over_a_series_a_blank_kept_from_being_made_is_not_called_and_is_not_an_empty_series():
    """A document that couldn't be read is not an empty document: its count is not made, and ``not_called`` says which input."""

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

    graph = Graph.compile(GraphRecord(
        nodes=[NodeRecord(id=n, node=d) for n, d in [("d", "docs"), ("l", "lines"), ("c", "count"), ("t", "total"), ("a", "as_table"), ("w", "width")]],
        edges=[("d.result", "l.text"), ("l.result", "c.items"), ("c.result", "t.ns"), ("d.result", "a.text"), ("a.result", "w.table")],
    ), registry_of(docs, lines, count, width, as_table, total))
    run = graph.run(text="a,b;-;c;")

    assert run.results["c"]["result"] == [2, None, 1, 0]  # the last document is empty: 0 lines, made
    assert run.results["w"]["result"] == [1, None, 1, 1]
    assert run.results["t"]["result"] == 3
    assert {(n.loc[0], n.loc[1:], n.input) for n in run.not_called} >= {("c", (1,), "items"), ("w", (1,), "table")}


# -- items, outer values and gathers -----------------------------------------------------------

THREE_LEVELS = (
    {"id": "docs", "node": "split", "values": {"sep": "|"}},
    {"id": "paras", "node": "split", "values": {"sep": ";"}},
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


def test_each_series_output_of_a_node_makes_items_of_its_own_and_of_its_own_length():
    @dataclass(frozen=True)
    class Both:
        a: Series[str]
        b: Series[str]

    @node("uneven", title="Uneven", description="For a test.")
    def uneven(text: str) -> Both:
        return Both(a=["x"], b=("y", "z"))  # any sequence will do

    run = ran({"id": "u", "node": "uneven"}, {"id": "ua", "node": "upper"}, {"id": "ub", "node": "upper"},
              edges=(("u.a", "ua.text"), ("u.b", "ub.text")), registry=registry_of(uneven, upper), text="t")
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

    assert Graph.compile(make_graph(lengths), registry_of(lengths)).run(text="a bb ccc").outputs == {"lengths.result": [1, 2, 3]}


def test_a_value_made_once_is_there_on_each_call_of_a_node_that_runs_per_item():
    run = ran({"id": "docs", "node": "split"}, {"id": "prefix", "node": "upper"}, {"id": "p", "node": "pair"},
              edges=(("prefix.result", "p.a"), ("docs.result", "p.b")), **{"docs.text": "a,b", "prefix.text": "p"})
    assert run.graph.runs_on["p"].msg == "per item of docs.result"
    assert run.results["p"]["result"] == ["P:a", "P:b"]


def test_a_document_is_there_on_each_call_for_a_line_of_it():
    run = ran({"id": "docs", "node": "split", "values": {"sep": "|"}}, {"id": "lines", "node": "split", "values": {"sep": "/"}}, {"id": "p", "node": "pair"},
              edges=(("docs.result", "lines.text"), ("lines.result", "p.a"), ("docs.result", "p.b")), text="a/b|c")
    assert run.graph.runs_on["p"].msg == "per item of lines.result"
    assert run.results["p"]["result"] == [["a:a/b", "b:a/b"], ["c:c"]]


def test_one_value_into_a_series_input_is_a_series_of_one():
    run = ran({"id": "a", "node": "upper"}, {"id": "j", "node": "join"}, edges=(("a.result", "j.texts"),), text="a")
    assert run.graph.runs_on["j"].msg == "once" and run.results["j"]["result"] == "A"


def test_a_series_input_with_a_typed_in_list_or_its_default_gets_it_whole_once():
    @node("join-or-nothing", title="Join or nothing", description="For a test.")
    def join_or_nothing(texts: Series[str] = ()) -> str:
        return "+".join(texts) or "nothing"

    reg = registry_of(join, join_or_nothing)
    run = ran({"id": "typed", "node": "join", "values": {"texts": ["a", "b"]}}, {"id": "absent", "node": "join-or-nothing"}, registry=reg)
    assert {k: v.msg for k, v in run.graph.runs_on.items()} == {"typed": "once", "absent": "once"}
    assert run.results == {"typed": {"result": "a+b"}, "absent": {"result": "nothing"}}


def test_an_unconnected_series_input_takes_its_default():
    @node("join-or-nothing", title="Join or nothing", description="For a test.")
    def join_or_nothing(texts: Series[str] = ()) -> str:
        return "+".join(texts)

    assert ran({"id": "j", "node": "join-or-nothing"}, registry=registry_of(join_or_nothing)).outputs == {"j.result": ""}


def test_a_closed_list_input_is_one_value_with_its_list_and_with_its_default():
    @node("tags", title="Tags", description="For a test.")
    def tags(tags: list[str] = ["x"]) -> str:  # noqa: B006
        return "|".join(tags)

    reg = registry_of(tags)
    assert ran({"id": "t", "node": "tags", "values": {"tags": ["a", "b", "c"]}}, registry=reg).outputs == {"t.result": "a|b|c"}
    assert ran({"id": "t", "node": "tags"}, registry=reg).outputs == {"t.result": "x"}


def test_a_gather_goes_up_one_level_so_a_node_reading_a_document_and_a_series_of_lines_runs_per_paragraph():
    @node("deep", title="Deep", description="For a test.")
    def deep(doc: str, lines: Series[str]) -> str:
        return f"{doc}:{len(lines)}"

    run = ran(*THREE_LEVELS, {"id": "d", "node": "deep"}, edges=(*THREE_EDGES, ("docs.result", "d.doc"), ("lines.result", "d.lines")),
              registry=registry_of(split, deep), text="a,b;c|d")
    assert run.graph.runs_on["d"].msg == "per item of paras.result"
    assert run.results["d"]["result"] == [["a,b;c:2", "a,b;c:1"], ["d:1"]]


def test_a_value_on_a_paragraph_keeps_a_series_of_lines_to_that_paragraph():
    run = ran(*THREE_LEVELS, {"id": "n", "node": "labelled"}, edges=(*THREE_EDGES, ("paras.result", "n.label"), ("lines.result", "n.texts")),
              registry=DEEP, text="a,b;c|d")
    assert run.results["n"]["result"] == [["a,b=a,b", "c=c"], ["d=d"]]


def test_two_gathers_one_level_apart_run_the_node_where_the_deeper_one_lands_with_the_other_as_an_outer_value():
    """``a`` gathers a document's paragraphs, ``b`` a paragraph's lines: the node runs per paragraph."""
    run = ran(*THREE_LEVELS, {"id": "n", "node": "two-series"}, edges=(*THREE_EDGES, ("paras.result", "n.a"), ("lines.result", "n.b")),
              registry=DEEP, text="a,b;c|d")
    assert run.graph.runs_on["n"].msg == "per item of paras.result"
    assert run.results["n"]["result"] == [["a,b,c+a,b", "a,b,c+c"], ["d+d"]]


def test_a_document_with_no_lines_is_called_with_an_empty_series():
    run = ran({"id": "docs", "node": "split", "values": {"sep": "|"}}, {"id": "lines", "node": "split"}, {"id": "n", "node": "labelled"},
              edges=(("docs.result", "lines.text"), ("docs.result", "n.label"), ("lines.result", "n.texts")), registry=DEEP, text="a||b")
    assert run.results["n"]["result"] == ["a=a", "=", "b=b"]


def test_a_series_read_whole_beside_its_own_items_is_the_whole_series_on_each_call():
    run = ran({"id": "s", "node": "split"}, {"id": "j", "node": "joined-beside"}, edges=(("s.result", "j.items"), ("s.result", "j.at")),
              registry=DEEP, text="a,b")
    assert run.graph.runs_on["j"].msg == "per item of s.result"
    assert run.results["j"]["result"] == ["a:a+b", "b:a+b"]


def test_a_gather_of_another_series_is_an_outer_value_beside_the_items_a_node_runs_per():
    run = ran({"id": "s", "node": "split"}, {"id": "o", "node": "split"}, {"id": "j", "node": "joined-beside"},
              edges=(("o.result", "j.items"), ("s.result", "j.at")), registry=DEEP, **{"s.text": "a,b", "o.text": "x,y"})
    assert run.results["j"]["result"] == ["a:x+y", "b:x+y"]


def test_a_series_read_whole_lands_beside_each_item_of_another_series_of_the_same_document_through_json():
    """Every keyword of a document beside each of its invoices: the keywords are gathered where the document is."""

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
    nodes = ({"id": "docs", "node": "split", "values": {"sep": "|"}}, {"id": "p", "node": "parts"}, {"id": "b", "node": "beside"})
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
    loaded = graph({"id": "s", "node": "split"}, {"id": "u", "node": "upper"}, {"id": "j", "node": "join"}, {"id": "p", "node": "pair", "values": {"b": "x"}},
                   {"id": "k", "node": "pair"},
                   edges=(("s.result", "u.text"), ("u.result", "j.texts"), ("u.result", "p.a"), ("j.result", "k.a"), ("u.result", "k.b")))
    nodes = loaded.describe().nodes

    def fed(node_id: str) -> dict[str, str]:
        return {i.field.name: i.feed.msg for i in nodes[node_id].inputs}

    assert nodes["k"].inputs[0].feed.model_dump() == {"mode": "outer_value", "series": None, "msg": "the value of the graph"}
    assert nodes["j"].runs == RunsOn(mode="once", msg="once")
    assert loaded.runs_on["u"] == RunsOn(mode="per_item", series="s.result", msg="per item of s.result")
    assert fed("u") == {"text": "per item of s.result"}
    assert fed("j") == {"texts": "every item of s.result"}
    assert fed("p") == {"a": "per item of s.result", "b": "one value"}
    assert fed("k") == {"a": "the value of the graph", "b": "per item of s.result"}
    assert fed("s") == {"text": "one value", "sep": "one value"}


# -- no items, and skips above them ------------------------------------------------------------


def test_a_node_over_an_empty_series_never_starts_and_the_gather_is_called_with_an_empty_series():
    loaded = graph({"id": "s", "node": "split"}, {"id": "u", "node": "upper"}, {"id": "j", "node": "join"},
                   edges=(("s.result", "u.text"), ("u.result", "j.texts")))

    said = streamed(loaded.with_inputs(text=""))
    assert [e.type for e in said if getattr(e, "loc", None) == ("u",)] == ["node_complete"]  # never started
    run = said[-1].run
    assert run.status == "complete" and run.results["u"]["result"] == [] and run.results["j"]["result"] == ""
    assert through_json(run).results["u"]["result"] == []


def test_a_skipped_value_skips_what_hangs_off_it():
    run = ran({"id": "g", "node": "decide"}, {"id": "y", "node": "upper"}, {"id": "n", "node": "upper"},
              edges=(("g.yes", "y.text"), ("g.no", "n.text")), text="hey")
    assert run.results["y"]["result"] == "HEY" and run.results["n"]["result"] is None


def test_a_failing_node_on_the_branch_not_taken_never_runs():
    @node("always-fail", title="Always fail", description="For a test.")
    def always_fail(text: str) -> str:
        raise ValueError("never")

    run = ran({"id": "g", "node": "decide"}, {"id": "y", "node": "upper"}, {"id": "n", "node": "always-fail"},
              edges=(("g.yes", "y.text"), ("g.no", "n.text")), registry=registry_of(decide, upper, always_fail), text="hey")
    assert run.results["y"]["result"] == "HEY"


def test_a_series_never_made_leaves_its_readers_without_results_before_and_after_json():
    for nodes, edges in (
        (({"id": "s", "node": "split"},), (("g.yes", "s.text"), ("s.result", "u.text"))),
        (CONTACTS, (("g.yes", "c.text"), SPLIT_CONTACTS, ("r.result.name", "u.text"))),
    ):
        run = ran({"id": "g", "node": "decide"}, *nodes, {"id": "u", "node": "upper"}, edges=edges, text="no")
        assert run.results["u"]["result"] is None and through_json(run).results["u"]["result"] is None


def test_a_skip_above_the_items_skips_the_gather_and_no_items_give_it_an_empty_series():
    skipped = ran({"id": "b", "node": "decide"}, {"id": "a", "node": "split"}, {"id": "j", "node": "join"},
                  edges=(("b.no", "a.text"), ("a.result", "j.texts")), text="ab")
    assert skipped.results["j"]["result"] is None
    empty = ran({"id": "a", "node": "split"}, {"id": "j", "node": "join"}, edges=(("a.result", "j.texts"),), text="")
    assert empty.results["j"]["result"] == ""


def test_a_skipped_node_makes_no_items_so_everything_below_is_skipped():
    run = ran({"id": "g", "node": "decide"}, {"id": "docs", "node": "split"}, {"id": "u", "node": "upper"}, {"id": "j", "node": "join"},
              edges=(("g.no", "docs.text"), ("docs.result", "u.text"), ("u.result", "j.texts")), text="a,b")
    assert run.results["docs"]["result"] is None and run.results["u"]["result"] is None and run.results["j"]["result"] is None


def test_a_branch_not_taken_outside_the_items_skips_the_gather_and_every_item_rejected_calls_it_with_none():
    CALLED.clear()
    outside = ran({"id": "b", "node": "decide"}, {"id": "j", "node": "join"}, edges=(("b.no", "j.texts"),), text="yes")
    assert outside.results["j"]["result"] is None and CALLED == []

    per_item = ran({"id": "s", "node": "split"}, {"id": "b", "node": "decide"}, {"id": "j", "node": "join"}, {"id": "p", "node": "pair"},
                   edges=(("s.result", "b.text"), ("b.no", "j.texts"), ("s.result", "p.a"), ("j.result", "p.b")), text="p,q")
    assert per_item.results["j"]["result"] == "" and CALLED == ["join []"]  # called once, over two rejected items
    assert per_item.results["p"]["result"] == ["p:", "q:"]


def test_a_skip_on_a_document_reaches_a_reader_on_its_lines_and_the_other_document_runs():
    run = ran({"id": "d", "node": "split", "values": {"sep": "|"}}, {"id": "g", "node": "decide"}, {"id": "l", "node": "split"},
              {"id": "t", "node": "pair"}, {"id": "u", "node": "upper"},
              edges=(("d.result", "g.text"), ("d.result", "l.text"), ("l.result", "t.a"), ("g.yes", "t.b"), ("t.result", "u.text")), text="a,b|c,no")
    assert run.graph.runs_on["u"].msg == "per item of l.result"
    assert run.results["u"]["result"] == [["A:A,B", "B:A,B"], None]


def test_a_skip_on_a_document_skips_the_gather_over_its_lines_and_survives_json():
    run = ran({"id": "d", "node": "split", "values": {"sep": "|"}}, {"id": "g", "node": "decide"}, {"id": "l", "node": "split"},
              {"id": "p", "node": "pair"}, {"id": "j", "node": "join"},
              edges=(("d.result", "g.text"), ("d.result", "l.text"), ("l.result", "p.a"), ("g.yes", "p.b"), ("p.result", "j.texts")), text="x|y,no")
    assert run.results["j"]["result"] == ["x:x", None]
    assert through_json(run).results["j"]["result"] == ["x:x", None]


def test_a_skip_above_an_empty_series_hides_what_the_gather_returned():
    run = ran({"id": "d", "node": "split", "values": {"sep": "|"}}, {"id": "g", "node": "decide"}, {"id": "l", "node": "split"},
              {"id": "t", "node": "pair"}, {"id": "m", "node": "split"}, {"id": "x", "node": "pair"}, {"id": "j", "node": "join"},
              edges=(("d.result", "g.text"), ("d.result", "l.text"), ("l.result", "t.a"), ("g.yes", "t.b"), ("l.result", "m.text"),
                     ("t.result", "x.a"), ("m.result", "x.b"), ("x.result", "j.texts")), text="p,no")
    assert run.graph.runs_on["j"].msg == "per item of l.result"
    assert run.results["j"]["result"] == [None]


def test_a_document_whose_branch_is_not_taken_makes_no_lines_and_the_gather_over_them_is_skipped():
    run = ran({"id": "docs", "node": "split", "values": {"sep": "|"}}, {"id": "g", "node": "decide"}, {"id": "lines", "node": "split"},
              {"id": "up", "node": "upper"}, {"id": "j", "node": "labelled"},
              edges=(("docs.result", "g.text"), ("g.yes", "lines.text"), ("lines.result", "up.text"), ("docs.result", "j.label"), ("up.result", "j.texts")),
              registry=registry_of(split, decide, upper, labelled), text="a,b|no")
    assert run.results["lines"]["result"] == [["a", "b"], None]
    assert run.results["j"]["result"] == ["a,b=A,B", None]


def test_a_gather_waits_for_the_decision_above_its_items_before_it_runs_on_nothing():
    release = threading.Event()

    @node("held", title="Held", description="For a test.", policy=Policy(concurrency=2))
    def held(text: str) -> YesNo:
        release.wait(5)
        return YesNo(SKIPPED, text) if "no" in text else YesNo(text, SKIPPED)

    loaded = graph({"id": "docs", "node": "split", "values": {"sep": "|"}}, {"id": "g", "node": "held"}, {"id": "l", "node": "split"},
                   {"id": "j", "node": "join"}, edges=(("docs.result", "g.text"), ("g.yes", "l.text"), ("l.result", "j.texts")),
                   registry=registry_of(split, held, join))
    CALLED.clear()
    timer = threading.Timer(0.2, release.set)
    timer.start()
    started = time.monotonic()
    run = loaded.run(text="a|slow-no")
    assert time.monotonic() - started >= 0.2 and run.status == "complete", run.cause
    assert run.results["j"]["result"] == ["a", None]
    assert "join []" not in CALLED


def test_skipped_inside_a_returned_series_leaves_gaps_for_those_items_through_json():
    @node("sparse", title="Sparse", description="For a test.")
    def sparse() -> Series[str]:
        return ["a", SKIPPED, "c"]

    run = ran({"id": "s", "node": "sparse"}, {"id": "u", "node": "upper"}, edges=(("s.result", "u.text"),), registry=registry_of(sparse, upper))
    assert run.results["u"]["result"] == ["A", None, "C"]
    assert through_json(run).results == run.results


def test_a_decision_over_a_whole_series_skips_what_hangs_off_the_branch_not_taken():
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
              registry=registry_of(split, decide_list, join, upper), text="a,b")
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

    @node("hold", title="Hold", description="For a test.")
    def hold(table: Table) -> Table:
        seen.append(table)
        return table

    run = ran({"id": "c", "node": "contacts"}, {"id": "a", "node": "hold"}, {"id": "b", "node": "hold"},
              edges=(("c.result", "a.table"), ("a.result", "b.table")), registry=registry_of(contacts, hold), text=CSV)
    assert seen[0].equals(seen[1]) and seen[0].equals(run.results["c"]["result"])
    assert {k: v.msg for k, v in run.graph.runs_on.items()} == {"c": "once", "a": "once", "b": "once"}


def test_a_node_filtering_every_row_away_returns_a_table_of_no_rows():
    @node("none-left", title="None left", description="For a test.")
    def none_left(table: Table) -> Table:
        return table.filter(pl.lit(False))

    run = ran({"id": "c", "node": "contacts"}, {"id": "f", "node": "none-left"}, edges=(("c.result", "f.table"),), registry=registry_of(contacts, none_left), text=CSV)
    assert run.results["f"]["result"].height == 0


def test_a_column_of_a_table_is_no_address_until_the_table_is_split():
    loaded = graph({"id": "c", "node": "contacts"}, {"id": "u", "node": "upper"}, edges=(("c.result.email", "u.text"),))
    assert [(p.type, p.loc) for p in loaded.problems] == [("unknown_output", ("u", "text"))]
    assert problems(*CONTACTS, {"id": "u", "node": "upper"}, edges=(SPLIT_CONTACTS, ("r.result.email", "u.text"))) == []


def test_a_filtered_table_is_split_and_what_a_node_made_per_record_is_gathered_beside_the_records():
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


def test_a_sorted_table_is_split_in_its_new_order_and_a_blank_is_not_called():
    @node("sort", title="Sort", description="For a test.")
    def sort(table: Table[Contact]) -> Table[Contact]:
        return table.sort("name", descending=True)

    run = ran({"id": "c", "node": "contacts"}, {"id": "o", "node": "sort"}, {"id": "r", "node": "records"}, {"id": "s", "node": "send"},
              edges=(("c.result", "o.table"), ("o.result", "r.table"), ("r.result.email", "s.email")),
              registry=registry_of(contacts, sort, records, send), text=CSV)
    assert run.results["s"]["result"] == ["sent to nina@x", "sent to ida@x", None]


def test_invoices_per_document_get_their_check_and_a_skipped_check_is_a_gap_for_its_own_invoice():
    run = ran(DOCS, READ, LINES, {"id": "check", "node": "check"}, {"id": "total", "node": "total"},
              edges=(*READ_EDGES, ("lines.result.vendor", "check.vendor"), ("lines.result.amount", "total.amounts")), text="acme:10,unknown:5|beta:7")
    assert {k: v.msg for k, v in run.graph.runs_on.items()} == {
        "docs": "once", "read": "per item of docs.result", "lines": "per item of docs.result", "check": "per item of lines.result",
        "total": "per item of docs.result",
    }
    assert run.results["check"]["result"] == [["acme ok", None], ["beta ok"]]
    assert run.results["total"]["result"] == [15.0, 7.0]


def test_a_document_with_no_invoices_totals_nothing():
    run = ran(DOCS, READ, LINES, {"id": "total", "node": "total"}, edges=(*READ_EDGES, ("lines.result.amount", "total.amounts")), text="acme:10|")
    assert run.results["total"]["result"] == [10.0, 0.0]


def test_a_document_skipped_above_its_table_is_not_totalled_and_the_others_still_total():
    run = ran(DOCS, {"id": "gate", "node": "decide"}, READ, LINES, {"id": "total", "node": "total"},
              edges=(("docs.result", "gate.text"), ("gate.yes", "read.text"), ("read.result", "lines.table"), ("lines.result.amount", "total.amounts")),
              text="acme:10|no|beta:7")
    assert run.results["total"]["result"] == [10.0, None, 7.0]


def test_a_table_input_gathers_the_records_under_each_document_into_a_table_per_document():
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


def test_a_table_made_per_document_is_described_as_a_table_with_its_columns():
    loaded = graph(DOCS, READ, LINES, {"id": "check", "node": "check"}, edges=(*READ_EDGES, ("lines.result.vendor", "check.vendor")))
    assert loaded.problems == []
    (result,) = loaded.describe().nodes["read"].outputs
    assert not result.many and result.type.startswith("Table") and {c["properties"]["name"]["const"]: c["properties"]["values"]["items"] for c in result.json_schema["properties"]["columns"]["prefixItems"]}["amount"] == {"type": "number"}


def test_a_bare_table_offers_no_columns():
    (result,) = graph({"id": "s", "node": "show"}).describe().nodes["s"].outputs
    assert (result.many, result.type) == (False, "Table") and "prefixItems" not in result.json_schema["properties"]["columns"]


def test_a_typed_in_list_of_records_is_one_table_not_many():
    run = ran({"id": "x", "node": "show", "values": {"table": [{"a": "1"}, {"a": "2"}]}})
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


def test_a_table_a_caller_gives_is_read_to_its_columns_by_their_types_and_survives_a_save():
    loaded = graph({"id": "d", "node": "describe-dated"}, registry=DATED)

    run = loaded.run(rows=[{"name": "a", "when": "2024-01-02", "extra": 1}])

    assert run.results["d"]["result"] == "['name', 'when']:[datetime.date(2024, 1, 2)]"
    assert through_json(run).results == run.results
    assert loaded.run(rows=pl.DataFrame({"name": ["b"], "when": [datetime.date(2024, 5, 6)]})).status == "complete"


def test_a_caller_s_table_missing_a_column_is_refused_naming_it_and_an_empty_list_is_a_table_of_no_rows():
    loaded = graph({"id": "d", "node": "describe-dated"}, registry=DATED)

    with pytest.raises(InvalidCallError) as refused:
        loaded.run(rows=[{"name": "a"}])
    (problem,) = refused.value.errors()
    assert (problem.type, problem.loc, problem.ctx["column"]) == ("invalid_input", ("inputs", "rows"), "when")
    assert loaded.run(rows=[]).results["d"]["result"] == "['name', 'when']:[]"


def test_a_typed_in_table_reads_each_cell_through_its_column_s_own_type():
    run = ran({"id": "t", "node": "total-price", "values": {"rows": [{"price": "1,5"}, {"price": "2"}]}}, registry=DATED)

    assert run.results["t"]["result"] == 3.5


def test_a_typed_in_cell_its_type_cannot_read_is_refused_at_load_naming_the_column():
    loaded = graph({"id": "d", "node": "describe-dated", "values": {"rows": [{"name": "a", "when": "junk"}]}}, registry=DATED)

    (problem,) = loaded.problems
    assert (problem.type, problem.loc, problem.ctx["column"]) == ("invalid_value", ("d", "rows"), "when")
    assert ran({"id": "d", "node": "describe-dated", "values": {"rows": [{"name": "a", "when": "2024-01-02"}]}}, registry=DATED).status == "complete"


def test_a_typed_in_frame_on_a_table_of_any_columns_arrives_with_the_columns_it_holds():
    run = ran({"id": "s", "node": "show", "values": {"table": [{"email": "a@b.dk", "n": 1}]}}, registry=DATED)

    assert run.results["s"]["result"].columns == ["email", "n"]


# -- branches ----------------------------------------------------------------------------------


@node("parity", title="Parity", description="For a test.")
def parity(n: str) -> YesNo:
    return YesNo(n, SKIPPED) if int(n) % 2 == 0 else YesNo(SKIPPED, n)


def test_series_inputs_reading_what_hangs_off_opposite_branches_per_item_are_refused():
    assert problems({"id": "n", "node": "split"}, {"id": "r", "node": "parity"}, {"id": "x", "node": "upper"}, {"id": "y", "node": "upper"},
                    {"id": "c", "node": "two-series"},
                    edges=(("n.result", "r.n"), ("r.yes", "x.text"), ("r.no", "y.text"), ("x.result", "c.a"), ("y.result", "c.b")),
                    registry=registry_of(split, parity, upper, two_series)) == ["series_inputs_never_share_an_item"]


def test_a_decision_per_document_read_by_nodes_per_line_still_keeps_two_branches_apart():
    assert problems({"id": "s", "node": "split", "values": {"sep": "|"}}, {"id": "d", "node": "parity"}, {"id": "sp", "node": "split"},
                    {"id": "ua", "node": "pair"}, {"id": "ub", "node": "pair"}, {"id": "z", "node": "two-series"},
                    edges=(("s.result", "d.n"), ("s.result", "sp.text"), ("sp.result", "ua.a"), ("d.yes", "ua.b"), ("sp.result", "ub.a"), ("d.no", "ub.b"),
                           ("ua.result", "z.a"), ("ub.result", "z.b")),
                    registry=registry_of(split, parity, pair, two_series)) == ["series_inputs_never_share_an_item"]


def test_an_item_skipped_on_one_series_input_is_left_out_of_every_series_input_of_the_node():
    run = ran({"id": "n", "node": "split"}, {"id": "r", "node": "parity"}, {"id": "x", "node": "upper"}, {"id": "z", "node": "zipped"},
              edges=(("n.result", "r.n"), ("r.yes", "x.text"), ("n.result", "z.a"), ("x.result", "z.b")), registry=registry_of(split, parity, upper, zipped),
              text="1,2,3,4")
    assert run.results["z"]["result"] == "22|44"


def test_two_edges_into_one_input_that_can_both_hold_a_value_are_a_problem_never_first_wins():
    """``upper(a, b)`` would silently lose ``b``; two edges per item of one Series would cover each item twice."""
    assert problems({"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "u", "node": "upper"},
                    edges=(("a.result", "u.text"), ("b.result", "u.text"))) == ["several_edges_into_one_input"]
    loaded = graph({"id": "s", "node": "split"}, {"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "m", "node": "upper"},
                   edges=(("s.result", "a.text"), ("s.result", "b.text"), ("a.result", "m.text"), ("b.result", "m.text")))
    (problem,) = loaded.problems
    assert (problem.type, problem.loc, problem.ctx) == ("several_edges_into_one_input", ("m", "text"), {"node_id": "m", "field": "text", "sources": ["a.result", "b.result"]})


def test_two_branches_wired_into_one_input_merge_back_per_item_and_the_chain_stays_per_item():
    run = ran({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "u", "node": "upper"}, {"id": "after", "node": "upper"},
              edges=(("s.result", "d.text"), ("d.yes", "u.text"), ("d.no", "u.text"), ("u.result", "after.text")), text="ok,no,fine")

    assert run.results["u"]["result"] == ["OK", "NO", "FINE"]
    assert run.graph.runs_on["after"].msg == "per item of s.result"


def test_what_hangs_off_two_branches_merges_back_too():
    """What hangs off opposite branches is as exclusive as the branches themselves."""
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
                    registry=registry_of(split, decide, words_of, upper)) == ["alternatives_in_different_series"]


def test_a_node_filling_two_outputs_of_one_exclusive_group_fails_as_invalid_output():
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

    assert ran(*nodes, edges=edges, registry=ALTERNATIVES, text="a b c").outputs == {"w.result": "a-b-c"}
    assert ran(*nodes, edges=edges, registry=ALTERNATIVES, text="a b").outputs == {"w.result": "a-b"}


def test_alternative_tables_per_item_are_the_branch_each_item_took_through_json():
    run = ran({"id": "s", "node": "split"}, {"id": "o", "node": "sort-out"}, {"id": "w", "node": "words"},
              edges=(("s.result", "o.text"), ("o.big", "w.table"), ("o.small", "w.table")), registry=ALTERNATIVES, text="a b c,d e")

    assert run.results["w"]["result"] == ["a-b-c", "d-e"]
    assert through_json(run).results["w"] == run.results["w"]


def test_tables_hanging_off_two_branches_into_one_table_input_are_the_branch_that_ran():
    nodes = ({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "y", "node": "lines-of"}, {"id": "n", "node": "lines-of"},
             {"id": "w", "node": "words"})
    edges = (("s.result", "d.text"), ("d.yes", "y.text"), ("d.no", "n.text"), ("y.result", "w.table"), ("n.result", "w.table"))
    reg = registry_of(split, decide, lines_of, words)

    assert ran(*nodes, edges=edges, registry=reg, text="ok,no").results["w"]["result"] == ["OK", "NO"]


def test_a_table_never_made_on_the_branch_that_ran_is_a_blank_as_it_is_from_one_maker():
    @node("nothing", title="Nothing", description="For a test.")
    def nothing(text: str) -> str | None:
        return None

    @node("lines-with", title="Lines with", description="For a test.")
    def lines_with(text: str, extra: str) -> Table[Line]:
        return pl.DataFrame({"text": [text]})

    reg = registry_of(decide, nothing, lines_with, lines_of, words)
    nodes = ({"id": "d", "node": "decide"}, {"id": "n", "node": "nothing"}, {"id": "y", "node": "lines-with"}, {"id": "w", "node": "words"})
    edges = (("d.yes", "y.text"), ("n.result", "y.extra"), ("y.result", "w.table"))
    one_maker = ran(*nodes, edges=edges, registry=reg, **{"d.text": "ok", "n.text": "z"})
    alternatives = ran(*nodes, {"id": "o", "node": "lines-of"}, edges=(*edges, ("d.no", "o.text"), ("o.result", "w.table")), registry=reg,
                       **{"d.text": "ok", "n.text": "z"})

    for run in (one_maker, alternatives):
        assert run.results["w"] == {"result": None}
        assert NotCalled(loc=("w",), input="table") in run.not_called


def test_whole_tables_that_can_both_hold_a_value_into_one_table_input_are_a_problem_as_into_any_input():
    assert problems({"id": "a", "node": "lines-of"}, {"id": "b", "node": "lines-of"}, {"id": "w", "node": "words"},
                    edges=(("a.result", "w.table"), ("b.result", "w.table")), registry=ALTERNATIVES) == ["several_edges_into_one_input"]


# -- a list node per document ------------------------------------------------------------------


@node("questions", title="Questions", description="The questions typed in, as a Series; ``for_each`` makes it once per item wired there.")
def questions(items: list[str], for_each: Any = None) -> Series[str]:
    return items


@node("answer", title="Answer", description="For a test.")
def answer(doc: str, question: str) -> bool:
    raise Ask(f"{doc}: {question}")


ASKING = registry_of(split, decide, questions, answer)
QUESTIONS = {"id": "l", "node": "questions", "values": {"items": ["a", "b", "c"]}}


def test_questions_listed_per_document_are_asked_of_each_document_and_each_answer_lands_on_its_own_question():
    """Three questions, two documents, six questions asked, each answered at its own place."""
    loaded = graph({"id": "d", "node": "split"}, QUESTIONS, {"id": "q", "node": "answer"},
                   edges=(("d.result", "l.for_each"), ("d.result", "q.doc"), ("l.result", "q.question")), registry=ASKING)
    assert loaded.problems == []
    run = loaded.run(text="x,y")

    assert loaded.runs_on["q"].msg == "per item of l.result"
    assert [w.prompt for w in run.waiting] == ["x: a", "x: b", "x: c", "y: a", "y: b", "y: c"]
    assert [w.loc[1:] for w in run.waiting] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    saved = Run.from_json(run.to_json(), loaded)
    done = saved.answer({w.id: w.loc[1:] == (1, 2) for w in saved.waiting}).resume()
    assert done.results["q"]["result"] == [[False, False, False], [False, False, True]]


def test_questions_listed_once_beside_documents_are_two_series_that_never_pair_up():
    assert problems({"id": "d", "node": "split"}, QUESTIONS, {"id": "q", "node": "answer"},
                    edges=(("d.result", "q.doc"), ("l.result", "q.question")), registry=ASKING) == ["series_not_nested"]


def test_no_documents_ask_no_questions_and_a_document_skipped_above_asks_none_of_its_own():
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


def test_the_output_of_a_node_called_inside_two_loops_is_a_list_per_outer_item_and_none_where_the_inner_list_was_never_made():
    """Three documents split into lines, the second empty: ``None`` for it, where a document with no lines shows ``[]``."""
    registry = registry_of(_documents, _lines, _shout, _words_of)
    flat = Graph.compile(_PER_LINE, registry).run(text="a,b||c")

    assert flat.results["shout"] == {"result": [["A", "B"], None, ["C"]]}
    assert flat.outputs == {"shout.result": [["A", "B"], None, ["C"]]} and json.loads(flat.outputs_json()) == {"shout.result": [["A", "B"], None, ["C"]]}
    assert Graph.compile(_PER_LINE, registry).run(text="a,b|-|c").results["shout"] == {"result": [["A", "B"], [], ["C"]]}
    assert Graph.compile(_PER_LINE, registry).with_inputs(text="a,b||c").results["shout"] == {"result": None}
    assert through_json(flat).results == flat.results

    per_line = stored_graph("per-line", title="Per line", description="For a test.")(make_graph(_lines, _shout, edges=[("lines.result", "shout.line")]))
    embedding = Graph.compile(make_graph(_documents, per_line, edges=[("documents.result", "per-line.document")]), registry_of(_documents, _lines, _shout, per_line))
    embedded = embedding.run(text="a,b||c")
    assert embedded.results["per-line"] == {"shout.result": [["A", "B"], None, ["C"]]}
    assert json.loads(embedded.outputs_json()) == {"per-line.shout.result": [["A", "B"], None, ["C"]]}

    per_word = make_graph(_documents, _lines, _words_of, _shout, edges=[("documents.result", "lines.document"), ("lines.result", "words-of.line"), ("words-of.result", "shout.line")])
    assert Graph.compile(per_word, registry).run(text="a b,c||d").results["shout"] == {"result": [[["A", "B"], ["C"]], None, [["D"]]]}


# -- the ceiling on a Series' items ------------------------------------------------------------


def test_a_series_longer_than_the_ceiling_stops_the_run_before_any_call_per_item():
    CALLED.clear()
    loaded = graph({"id": "s", "node": "split"}, {"id": "u", "node": "upper"}, edges=(("s.result", "u.text"),))

    stopped = loaded.run(text="a,b,c", max_items=2)

    assert (stopped.status, stopped.state.cause.type, stopped.state.cause.loc) == ("stopped", "item_ceiling", ("s",))
    assert stopped.state.cause.ctx == {"series": "s.result", "items": 3, "max_items": 2} and CALLED == []
    assert stopped.with_max_items(3).resume().results["u"]["result"] == ["A", "B", "C"]


def test_with_max_items_is_the_same_run_with_its_ceiling_raised_and_starts_nothing():
    CALLED.clear()
    loaded = graph({"id": "s", "node": "split"}, {"id": "u", "node": "upper"}, edges=(("s.result", "u.text"),))
    stopped = loaded.run(text="a,b,c", max_items=2)

    raised = stopped.with_max_items(3)

    assert (raised.max_items, stopped.max_items) == (3, 2) and CALLED == []
    assert raised.state == stopped.state and raised.results == stopped.results and raised.inputs == stopped.inputs
    assert json.loads(raised.to_json()) == {**json.loads(stopped.to_json()), "max_items": 3}
    assert Run.from_json(raised.to_json(), loaded).resume().status == "complete"
    with pytest.raises(InvalidCallError) as refused:
        stopped.with_max_items(0)
    assert [(p.type, p.loc) for p in refused.value.errors()] == [("invalid_max_items", ("max_items",))]
    for leg in (stopped.resume, stopped.stream):  # a ceiling is the run's, never a leg's
        with pytest.raises(TypeError):
            leg(max_items=3)


def test_the_ceiling_is_on_each_series_made_never_on_the_items_of_all_of_them_together():
    loaded = graph({"id": "d", "node": "split", "values": {"sep": "|"}}, {"id": "l", "node": "split"}, {"id": "u", "node": "upper"},
                   edges=(("d.result", "l.text"), ("l.result", "u.text")))

    assert loaded.run(text="a,b|c,d|e,f", max_items=3).status == "complete"  # six lines in all, never more than three in one Series
    stopped = loaded.run(text="a,b|c,d,e,f|g", max_items=3)
    assert (stopped.status, stopped.state.cause.ctx) == ("stopped", {"series": "l.result", "items": 4, "max_items": 3})


@pytest.mark.timeout(10)
def test_the_ceiling_stops_a_leg_whose_items_wait_for_a_thread_and_resuming_calls_each_item_once():
    calls: list[str] = []

    @node("explode", title="Explode", description="For a test.", policy=Policy(concurrency=2))
    def explode(letter: str) -> Series[str]:
        calls.append(letter)
        return [letter + str(i) for i in range(20 if letter == "c" else 2)]

    loaded = graph({"id": "s", "node": "split"}, {"id": "e", "node": "explode"}, {"id": "u", "node": "upper"},
                   edges=(("s.result", "e.letter"), ("e.result", "u.text")), registry=registry_of(split, explode, upper))

    stopped = loaded.run(text=",".join("abcdefghij"), max_items=10)

    assert (stopped.status, stopped.state.cause.type) == ("stopped", "item_ceiling") and len(calls) < 10
    assert stopped.with_max_items(100).resume().status == "complete" and sorted(calls) == list("abcdefghij")


# -- a subgraph is one node -------------------------------------------------------------

def embedded(node_id: str, *nodes: dict[str, Any], edges: tuple[tuple[str, str], ...] = ()) -> Any:
    """A graph of ``nodes`` as one node ``node_id``: its inputs are the open inputs of its first nodes."""
    return stored_graph(node_id, title=node_id.replace("-", " ").replace("_", " ").strip().capitalize(), description="For a test.")(GraphRecord.model_validate({"nodes": [{"node": n["id"], **n} for n in nodes], "edges": [list(e) for e in edges]}))


def test_two_series_into_one_embedding_are_refused_once_with_the_authors_addresses():
    inner = embedded("pairs", {"id": "p", "node": "pair"})
    loaded = graph({"id": "d1", "node": "split"}, {"id": "d2", "node": "split"}, {"id": "emb", "node": "pairs"},
                   edges=(("d1.result", "emb.a"), ("d2.result", "emb.b")), registry=registry_of(split, pair, inner))
    (problem,) = loaded.problems
    assert (problem.type, problem.loc, problem.ctx) == ("series_not_nested", ("emb",), {"node_id": "emb", "series": ["d1.result", "d2.result"], "inputs": ["a", "b"], "wire_from": "d1.result", "wire_into": "d2"})
    assert "emb/" not in problem.msg


def test_two_series_out_of_one_embedding_are_refused_where_they_meet():
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

    inner = embedded("two-lists", {"id": "l", "node": "left"}, {"id": "r", "node": "right"})
    assert problems({"id": "emb", "node": "two-lists"}, {"id": "after", "node": "pair"}, edges=(("emb.l.a", "after.a"), ("emb.r.b", "after.b")),
                    registry=registry_of(left, right, pair, inner)) == ["series_not_nested"]


def test_a_loop_over_items_made_inside_a_stored_graph_is_named_by_the_list_inside_it():
    inner = embedded("splitup", {"id": "split", "node": "split"}, {"id": "upper", "node": "upper"}, edges=(("split.result", "upper.text"),))
    loaded = graph({"id": "e", "node": "splitup"}, {"id": "split", "node": "split"}, {"id": "u2", "node": "upper"},
                   edges=(("e.upper.result", "u2.text"),), registry=registry_of(split, upper, inner))

    assert {k: v.msg for k, v in loaded.runs_on.items()} == {"e": "once", "split": "once", "u2": "per item of e.split.result"}
    assert loaded.describe().nodes["u2"].inputs[0].feed.series == "e.split.result"


def test_exclusive_groups_of_an_embedding_name_the_outputs_it_offers():
    inner = embedded("decider", {"id": "d", "node": "decide"})
    loaded = graph({"id": "e", "node": "decider"}, registry=registry_of(decide, inner))

    assert loaded.exclusive_groups == {"e": {"d.answer": ["d.yes", "d.no"]}}


def test_two_decisions_inside_one_embedding_are_two_exclusive_groups():
    inner = embedded("two", {"id": "d1", "node": "decide"}, {"id": "d2", "node": "decide"})
    loaded = graph({"id": "e", "node": "two"}, registry=registry_of(decide, inner))

    assert loaded.exclusive_groups == {"e": {"d1.answer": ["d1.yes", "d1.no"], "d2.answer": ["d2.yes", "d2.no"]}}
    assert [f.exclusive_group for f in loaded.describe().nodes["e"].outputs] == ["d1.answer", "d1.answer", "d2.answer", "d2.answer"]


def test_two_decisions_inside_one_embedding_each_take_their_own_branch():
    inner = embedded("two", {"id": "d1", "node": "decide"}, {"id": "d2", "node": "decide"})
    loaded = graph({"id": "e", "node": "two"}, registry=registry_of(decide, inner))

    assert loaded.run({"d1.text": "yes", "d2.text": "no"}).outputs == {"e.d1.yes": "yes", "e.d1.no": None, "e.d2.yes": None, "e.d2.no": "no"}


def test_an_embedding_fed_from_outside_is_a_later_node_so_its_other_inputs_must_be_fed_too():
    inner = embedded("pairer", {"id": "p", "node": "pair"})
    loaded = graph({"id": "u", "node": "upper"}, {"id": "e", "node": "pairer"}, edges=(("u.result", "e.a"),), registry=registry_of(upper, pair, inner))

    assert [(p.type, p.loc) for p in loaded.problems] == [("unbound_on_later_node", ("e", "b"))]


def test_a_nested_embedding_names_its_fields_as_each_record_says_them():
    two = embedded("two", {"id": "x", "node": "upper"}, {"id": "y", "node": "upper"})
    outer = embedded("outer", {"id": "f", "node": "two"}, {"id": "z", "node": "upper"})
    loaded = graph({"id": "o", "node": "outer"}, registry=registry_of(upper, two, outer))

    assert loaded.problems == [] and list(loaded.inputs) == ["x.text", "y.text", "text"]
    assert all(i.address.startswith("o.") for i in loaded.inputs.values())
    assert [o.name for o in loaded.describe().nodes["o"].outputs] == ["f.x.result", "f.y.result", "z.result"]


def test_a_nested_embedding_runs_with_its_fields_named_as_each_record_says_them():
    two = embedded("two", {"id": "x", "node": "upper"}, {"id": "y", "node": "upper"})
    outer = embedded("outer", {"id": "f", "node": "two"}, {"id": "z", "node": "upper"})
    loaded = graph({"id": "o", "node": "outer"}, registry=registry_of(upper, two, outer))

    assert loaded.run({"x.text": "a", "y.text": "b", "text": "c"}).results["o"] == {"f.x.result": "A", "f.y.result": "B", "z.result": "C"}


def test_a_series_made_inside_an_embedding_that_runs_per_item_is_made_for_each_item_fed_or_not():
    fed = embedded("splitter", {"id": "lines", "node": "split", "values": {"sep": "/"}}, {"id": "join", "node": "join"}, edges=(("lines.result", "join.texts"),))
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "splitter"}, edges=(("docs.result", "emb.text"),),
              registry=registry_of(split, join, fed), text="a/b,c")
    assert run.graph.runs_on["emb"].msg == "per item of docs.result"
    assert run.results["emb"]["join.result"] == ["a+b", "c"]

    unfed = embedded("splitter-unfed", {"id": "entered", "node": "upper"}, {"id": "lines", "node": "split", "values": {"text": "x/y", "sep": "/"}},
                     {"id": "join", "node": "join"}, {"id": "p", "node": "pair"},
                     edges=(("lines.result", "join.texts"), ("entered.result", "p.a"), ("join.result", "p.b")))
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "splitter-unfed"}, edges=(("docs.result", "emb.text"),),
              registry=registry_of(split, join, upper, pair, unfed), text="a,b")
    assert run.results["emb"]["p.result"] == ["A:x+y", "B:x+y"]


def test_an_inner_output_made_per_item_inside_is_a_series_for_each_item_the_embedding_runs_per():
    """Inside, ``p`` reads the entering value and each item of ``lines``: the embedding gives a Series per outer item."""
    inner = embedded("letters", {"id": "entered", "node": "upper"}, {"id": "lines", "node": "split", "values": {"text": "x/y", "sep": "/"}},
                     {"id": "p", "node": "pair"}, edges=(("entered.result", "p.a"), ("lines.result", "p.b")))
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "letters"}, edges=(("docs.result", "emb.text"),),
              registry=registry_of(split, upper, pair, inner), text="a,b")
    assert run.results["emb"]["p.result"] == [["A:x", "A:y"], ["B:x", "B:y"]]


def test_inputs_from_a_document_and_from_its_lines_run_the_embedding_per_line():
    @node("shout", title="Shout", description="For a test.")
    def shout(word: str) -> str:
        return word.upper()

    inner = embedded("pair-up", {"id": "ua", "node": "shout"}, {"id": "ub", "node": "upper"}, {"id": "p", "node": "pair"},
                     edges=(("ua.result", "p.a"), ("ub.result", "p.b")))
    run = ran({"id": "docs", "node": "split", "values": {"sep": "|"}}, {"id": "lines", "node": "split"}, {"id": "emb", "node": "pair-up"},
              edges=(("docs.result", "lines.text"), ("docs.result", "emb.word"), ("lines.result", "emb.text")),
              registry=registry_of(split, upper, pair, shout, inner), text="a,b|c")
    assert run.graph.runs_on["emb"].msg == "per item of lines.result"
    assert run.results["emb"]["p.result"] == [["A,B:A", "A,B:B"], ["C:C"]]


def test_a_series_into_an_embedded_series_input_is_gathered_and_the_embedding_runs_once():
    inner = embedded("joiner", {"id": "join", "node": "join"})
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "joiner"}, edges=(("docs.result", "emb.texts"),),
              registry=registry_of(split, join, inner), text="a,b")
    assert run.graph.runs_on["emb"].msg == "once" and run.results["emb"]["join.result"] == "a+b"


def test_an_embedding_reading_a_series_whole_and_per_item_gets_the_whole_series_on_each_call():
    inner = embedded("mixed", {"id": "u", "node": "upper"}, {"id": "j", "node": "join"})
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "mixed"}, edges=(("docs.result", "emb.text"), ("docs.result", "emb.texts")),
              registry=registry_of(split, upper, join, inner), text="a,b")
    assert set(run.graph.runs_on) == {"docs", "emb"} and run.graph.runs_on["emb"].msg == "per item of docs.result"
    assert run.results["emb"] == {"u.result": ["A", "B"], "j.result": ["a+b", "a+b"]}


def test_a_graph_embedded_under_an_item_gives_what_the_same_nodes_drawn_flat_give_for_that_item():
    nodes = ({"id": "i", "node": "split", "values": {"sep": ";"}}, {"id": "c", "node": "split", "values": {"sep": "/"}}, {"id": "p", "node": "pair"})
    edges = (("i.result", "c.text"), ("c.result", "p.a"), ("i.result", "p.b"))
    inner = embedded("deep", *nodes, edges=edges)
    reg = registry_of(split, pair, inner)
    embedded_run = ran({"id": "s", "node": "split"}, {"id": "e", "node": "deep"}, edges=(("s.result", "e.text"),), registry=reg, text="a/b;c,d")
    flat_run = ran({"id": "s", "node": "split"}, *nodes, edges=(("s.result", "i.text"), *edges), registry=reg, text="a/b;c,d")
    assert embedded_run.results["e"]["p.result"] == flat_run.results["p"]["result"]
    assert embedded_run.graph.runs_on["e"].msg == "per item of s.result"  # the items it is fed; its own Series are inside it


def test_an_inner_gather_of_the_entering_value_is_a_series_of_one_for_each_item_the_embedding_runs_per():
    inner = embedded("inner", {"id": "holder", "node": "upper"}, {"id": "gather", "node": "join"}, edges=(("holder.result", "gather.texts"),))
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "inner"}, {"id": "after", "node": "join"},
              edges=(("docs.result", "emb.text"), ("emb.gather.result", "after.texts")), registry=registry_of(split, upper, join, inner), text="a,b,c")
    assert run.graph.runs_on["emb"].msg == "per item of docs.result"
    assert run.results["emb"]["gather.result"] == ["A", "B", "C"]  # a gather of one per item, never all three once
    assert run.results["after"]["result"] == "A+B+C"


def test_a_branch_not_taken_inside_an_embedding_skips_its_inner_gather_for_each_item_the_embedding_runs_per():
    inner = embedded("skips", {"id": "b", "node": "decide"}, {"id": "j", "node": "join"}, edges=(("b.no", "j.texts"),))
    run = ran({"id": "s", "node": "split"}, {"id": "e", "node": "skips"}, edges=(("s.result", "e.text"),), registry=registry_of(split, decide, join, inner), text="p,q")
    assert run.results["e"]["j.result"] == [None, None]


def test_an_embedding_with_one_output_per_inner_item_and_one_per_run_runs_per_item_it_is_fed():
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
        values={"names.text": "Ida Bo"},
    )
    registry = registry_of(split, names, case, greet, count, shout)
    registry.add(stored_graph("inner", title="Inner", description="For a test.")(inner))
    graph = Graph.compile(make_graph(split, registry["inner"], edges=[("split.result", "inner.value")], values={"split.text": "A-1,B-2"}), registry)
    assert graph.runs_on["inner"].series == "split.result"
    assert graph.run().results["inner"] == {"count.result": [2, 2], "shout.result": [["IDA A-1", "BO A-1"], ["IDA B-2", "BO B-2"]]}


@node("each", title="Each", description="For a test.")
def each(texts: Series[str]) -> Series[str]:
    return texts


def test_an_embedded_node_gathering_a_series_made_inside_and_one_entering_gathers_the_same_in_either_drawing_order():
    """``c`` gathers the lines made inside and the texts the embedding is fed; which of them the record lists first
    changes nothing about what each gather receives."""
    drawn = {
        "lines": {"id": "lines", "node": "split", "values": {"text": "x/y", "sep": "/"}},
        "ent": {"id": "ent", "node": "each"},
        "c": {"id": "c", "node": "two-series"},
    }
    found = []
    for order in (("lines", "ent", "c"), ("c", "ent", "lines")):
        inner = embedded("inner", *(drawn[n] for n in order), edges=(("lines.result", "c.a"), ("ent.result", "c.b")))
        loaded = Graph.compile(
            GraphRecord.model_validate({"nodes": [{"id": "docs", "node": "split"}, {"id": "emb", "node": "inner"}], "edges": [["docs.result", "emb.texts"]]}), registry_of(split, each, two_series, inner),
        )
        assert loaded.problems == []
        found.append(loaded.run(text="a,b").outputs)
    assert found[0] == found[1] == {"emb.c.result": "x,y+a,b"}


def test_an_embedding_fed_by_a_series_drawn_after_it_loads_and_runs_as_one_drawn_before_it():
    """The node feeding an embedding may come after it in the record."""
    inner = embedded(
        "letters", {"id": "entered", "node": "upper"}, {"id": "lines", "node": "split", "values": {"text": "x/y", "sep": "/"}},
        {"id": "p", "node": "pair"}, edges=(("entered.result", "p.a"), ("lines.result", "p.b")),
    )
    registry = registry_of(split, upper, pair, inner)
    found = []
    for outer in ([{"id": "docs", "node": "split"}, {"id": "emb", "node": "letters"}], [{"id": "emb", "node": "letters"}, {"id": "docs", "node": "split"}]):
        loaded = Graph.compile(GraphRecord.model_validate({"nodes": outer, "edges": [["docs.result", "emb.text"]]}), registry)
        assert loaded.problems == []
        found.append(loaded.run(text="a,b").outputs)
    assert found[0] == found[1]


# -- three graphs an author draws ---------------------------------------------------------------


class Named(TypedDict):
    name: str


def test_a_contact_with_a_blank_email_is_not_mailed_and_keeps_its_place_in_the_sheet_with_a_blank():
    """Contacts as a table, split to records, a mail per email, and the records collected with what each mail gave."""
    mailed: list[str] = []

    @node("mail", title="Mail", description="Sends one mail.")
    def mail(email: str) -> bool:
        mailed.append(email)
        return True

    @node("collect", title="Collect", description="The records as a table, with a column saying whether each was mailed.")
    def collect(records: Table[Named], sent: Series[bool | None]) -> Table:
        return records.with_columns(pl.Series("sent", sent, dtype=pl.Boolean))

    loaded = graph(*CONTACTS, {"id": "m", "node": "mail"}, {"id": "k", "node": "collect"},
                   edges=(SPLIT_CONTACTS, ("r.result.email", "m.email"), ("r.result", "k.records"), ("m.result", "k.sent")),
                   registry=registry_of(contacts, records, mail, collect))
    assert loaded.problems == []
    run = loaded.run(text=CSV)

    assert run.status == "complete" and mailed == ["ida@x", "nina@x"]
    assert run.not_called == [NotCalled(loc=("m", 1), input="email")]
    sheet = run.outputs["k.result"]
    assert sheet.select("name", "sent").to_dicts() == [{"name": "Ida", "sent": True}, {"name": "Bo", "sent": None}, {"name": "Nina", "sent": True}]
    assert through_json(run).outputs["k.result"].equals(sheet)


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
    assert run.outputs == {"t.result": 3} and run.not_called == []


def test_an_approval_per_line_of_each_document_pauses_saves_resumes_and_a_new_document_asks_only_its_own_lines():
    asked: list[str] = []

    @node("documents", title="Documents", description="The documents of a folder, one per ``|``.")
    def documents(folder: str) -> Series[str]:
        return folder.split("|")

    @node("lines", title="Lines", description="The lines of a document, one per ``,``.")
    def lines(document: str) -> Series[str]:
        return document.split(",")

    @node("approve", title="Approve", description="Asks a person about one line.")
    def approve(line: str) -> Literal["yes", "no"]:
        asked.append(line)
        raise Ask(f"approve {line}?")

    @node("approved", title="Approved", description="How many lines of a document were approved.")
    def approved(answers: Series[str]) -> int:
        return answers.count("yes")

    loaded = graph({"id": "d", "node": "documents"}, {"id": "l", "node": "lines"}, {"id": "a", "node": "approve"}, {"id": "n", "node": "approved"},
                   edges=(("d.result", "l.document"), ("l.result", "a.line"), ("a.result", "n.answers")),
                   registry=registry_of(documents, lines, approve, approved))
    assert loaded.problems == []

    paused = loaded.run(folder="a1,a2|b1")
    assert paused.status == "waiting"
    assert [(w.loc[0], w.loc[1:], w.prompt) for w in paused.waiting] == [
        ("a", (0, 0), "approve a1?"), ("a", (0, 1), "approve a2?"), ("a", (1, 0), "approve b1?"),
    ]

    saved = Run.from_json(paused.to_json(), loaded)
    assert [w.id for w in saved.waiting] == [w.id for w in paused.waiting]
    done = saved.answer({w.id: "no" if w.prompt == "approve a2?" else "yes" for w in saved.waiting}).resume()
    assert done.status == "complete" and done.results["a"]["result"] == [["yes", "no"], ["yes"]]
    assert done.outputs == {"n.result": [1, 1]}

    asked.clear()
    again = done.again(folder="a1,a2|new1,new2|b1")
    assert again.status == "waiting" and asked == ["new1", "new2"]
    assert [(w.loc[1:], w.prompt) for w in again.waiting] == [((1, 0), "approve new1?"), ((1, 1), "approve new2?")]
    finished = again.answer({w.id: "yes" for w in again.waiting}).resume()
    assert finished.results["a"]["result"] == [["yes", "no"], ["yes", "yes"], ["yes"]]
    assert finished.outputs == {"n.result": [1, 2, 1]}
