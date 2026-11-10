"""Rows and tables through the public door: blanks, copy-down, gathers, branches, tables and embedded graphs.

Every value sits on a row of a table. A node fed a column runs once per row of
it; a value from a row above is copied down; a ``Series`` or ``Table`` input
gathers the rows under the node's row, or under the row it shares with a
sibling table. One-value sources must lie on one path of the row tree. A blank cell keeps a node from being called on its row unless the input
says ``| None``; a branch not taken skips the row.
"""
import datetime
import json
from dataclasses import dataclass
from typing import Annotated, Any, Optional, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Ask,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    Param,
    Series,
    Table,
    node,
)

from tests.test_core.nodes import registry_of

# -- the nodes ---------------------------------------------------------------------------------

CALLED: list[str] = []


class Contact(TypedDict):
    name: str
    email: str


class Invoice(TypedDict):
    vendor: str
    amount: float


@node("contacts")
def contacts(text: str) -> Table[Contact]:
    """``Ida,ida@x;Bo,;Nina,nina@x``: a contact per ``;``, a blank email where nothing follows the comma."""
    rows = [part.split(",") for part in text.split(";") if part]
    return pl.DataFrame({"name": [r[0] for r in rows], "email": [r[1] or None for r in rows]}, schema={"name": pl.String, "email": pl.String})


@node("split")
def split(text: str, sep: str = ",") -> Series[str]:
    return [part for part in text.split(sep)] if text else []


@node("upper")
def upper(text: str) -> str:
    CALLED.append(f"upper {text}")
    return text.upper()


@node("join")
def join(texts: Series[str]) -> str:
    CALLED.append(f"join {texts}")
    return "+".join(texts)


@node("count")
def count(values: Series[Any]) -> int:
    return len(values)


@node("pair")
def pair(a: str, b: str) -> str:
    return f"{a}:{b}"


@node("send")
def send(email: str) -> str:
    CALLED.append(f"send {email}")
    return f"sent to {email}"


@node("send-or-say")
def send_or_say(email: str | None) -> str:
    return f"sent to {email}" if email else "no address"


@node("show")
def show(table: Table) -> Table:
    return table


@dataclass(frozen=True)
class YesNo:
    yes: Annotated[str, Param(exclusive_group="answer")]
    no: Annotated[str, Param(exclusive_group="answer")]


@node("decide")
def decide(text: str) -> YesNo:
    """``yes`` unless the text has ``no`` in it."""
    return YesNo(SKIPPED, text) if "no" in text else YesNo(text, SKIPPED)


@node("docs")
def docs(text: str) -> Series[str]:
    """Documents, one per ``|``."""
    return text.split("|")


@node("invoices")
def invoices(text: str) -> Table[Invoice]:
    """``acme:10,beta:5``: an invoice per ``,``; ``skip`` is a branch not taken."""
    if text == "skip":
        return SKIPPED
    rows = [part.split(":") for part in text.split(",") if part]
    return pl.DataFrame({"vendor": [r[0] for r in rows], "amount": [float(r[1]) for r in rows]}, schema={"vendor": pl.String, "amount": pl.Float64})


@node("check")
def check(vendor: str) -> str:
    return SKIPPED if vendor == "unknown" else f"{vendor} ok"


@node("total")
def total(amounts: Series[float]) -> float:
    return sum(amounts)


REGISTRY = registry_of(contacts, split, upper, join, count, pair, send, send_or_say, show, decide, docs, invoices, check, total)

CSV = "Ida,ida@x;Bo,;Nina,nina@x"


def graph(*nodes: dict[str, Any], edges: tuple[tuple[str, str], ...] = (), registry: Any = REGISTRY) -> Any:
    """``nodes`` as ``{"id", "node", ...}`` and edges as ``(from, to)``, loaded; a graph with problems too."""
    return registry.load({"nodes": [{"node": n["id"], **n} for n in nodes], "edges": [list(e) for e in edges]})


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
    return run.graph.load_run(run.to_json())


# -- blanks ------------------------------------------------------------------------------------


def test_a_series_that_may_be_nothing_reads_an_explicit_none_like_leaving_it_out():
    @node("maybe")
    def maybe(xs: Series[str] | None = None) -> str:
        return str(xs)

    reg = registry_of(maybe)
    set_to_none = ran({"id": "m", "node": "maybe", "values": {"xs": None}}, registry=reg)
    left_out = ran({"id": "m", "node": "maybe"}, registry=reg)
    assert set_to_none.outputs == left_out.outputs == {"m.result": "None"}


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
    reader = node("seen")(seen)
    run = ran({"id": "c", "node": "contacts"}, {"id": "r", "node": "seen"}, edges=(("c.result.email", "r.x"),),
              registry=registry_of(contacts, reader), text=CSV)
    got = run.results["r"]["result"]
    if isinstance(got, list):  # one value per contact
        assert got[1] == ("None" if takes_blank else None), got
    else:  # a gather over every contact
        assert got == ("['ida@x', None, 'nina@x']" if takes_blank else "['ida@x', 'nina@x']"), got


def test_an_input_saying_none_is_called_with_the_blank():
    run = ran({"id": "c", "node": "contacts"}, {"id": "s", "node": "send-or-say"}, edges=(("c.result.email", "s.email"),), text=CSV)
    assert run.results["s"]["result"] == ["sent to ida@x", "no address", "sent to nina@x"]
    assert run.not_called == []


def test_a_default_is_for_an_input_nothing_is_wired_to_not_for_a_blank():
    @node("defaulted")
    def defaulted(email: str = "fallback") -> str:
        return email

    run = ran({"id": "c", "node": "contacts"}, {"id": "d", "node": "defaulted"}, edges=(("c.result.email", "d.email"),),
              registry=registry_of(contacts, defaulted), text=CSV)
    assert run.results["d"]["result"] == ["ida@x", None, "nina@x"]
    assert run.not_called == [{"node_id": "d", "row": (1,), "input": "email"}]


def test_a_node_all_blank_completes_it_is_never_skipped():
    @node("nothing")
    def nothing() -> str | None:
        return None

    import asyncio

    loaded = graph({"id": "n", "node": "nothing"}, {"id": "s", "node": "send"}, edges=(("n.result", "s.email"),), registry=registry_of(nothing, send))

    async def events():
        return [e async for e in loaded.with_inputs().stream()]

    said = asyncio.run(events())
    assert not [e for e in said if e.type == "node_skipped"]
    (complete,) = [e for e in said if e.type == "node_complete" and e.node_id == "s"]
    assert (complete.rows, complete.not_called) == (1, 1)
    assert said[-1].status == "complete" and said[-1].run.not_called == [{"node_id": "s", "row": (), "input": "email"}]


def test_choice_outputs_are_skipped_when_not_called():
    @node("branched")
    def branched(email: str) -> YesNo:
        return YesNo(email, SKIPPED)

    run = ran({"id": "c", "node": "contacts"}, {"id": "b", "node": "branched"}, {"id": "u", "node": "upper"},
              edges=(("c.result.email", "b.email"), ("b.yes", "u.text")), registry=registry_of(contacts, branched, upper), text=CSV)
    assert run.results["b"] == {"yes": ["ida@x", None, "nina@x"], "no": [None, None, None]}
    assert run.results["u"]["result"] == ["IDA@X", None, "NINA@X"]
    assert [(n["node_id"], n["row"]) for n in run.not_called] == [("b", (1,)), ("u", (1,))]


def test_a_row_making_output_not_called_makes_no_rows_so_a_count_of_them_is_not_called_either():
    run = ran({"id": "c", "node": "contacts"}, {"id": "s", "node": "split"}, {"id": "n", "node": "count"},
              edges=(("c.result.email", "s.text"), ("s.result", "n.values")), text=CSV)
    assert run.results["n"]["result"] == [1, None, 1]
    assert {"node_id": "n", "row": (1,), "input": "values"} in run.not_called


def test_a_table_output_not_called_makes_no_rows_so_a_count_of_them_is_not_called_either():
    run = ran({"id": "c", "node": "contacts"}, {"id": "i", "node": "invoices"}, {"id": "n", "node": "count"},
              edges=(("c.result.email", "i.text"), ("i.result.vendor", "n.values")), text="Ida,a:1;Bo,")
    assert run.results["n"]["result"] == [1, None]
    assert run.results["i"]["result"][1] is None  # nothing was made on Bo's row


def test_a_series_leaves_blanks_out():
    run = ran({"id": "c", "node": "contacts"}, {"id": "j", "node": "join"}, edges=(("c.result.email", "j.texts"),), text=CSV)
    assert run.results["j"]["result"] == "ida@x+nina@x"


def test_a_series_saying_none_keeps_them():
    @node("keep")
    def keep(values: Series[str | None]) -> int:
        return values.count(None)

    run = ran({"id": "c", "node": "contacts"}, {"id": "k", "node": "keep"}, edges=(("c.result.email", "k.values"),),
              registry=registry_of(contacts, keep), text=CSV)
    assert run.results["k"]["result"] == 1


@node("zipped")
def zipped(a: Series[str], b: Series[str]) -> str:
    return "|".join(x + y for x, y in zip(a, b, strict=True))


def test_series_on_one_index_lose_a_blank_row_together():
    run = ran({"id": "c", "node": "contacts"}, {"id": "z", "node": "zipped"}, edges=(("c.result.name", "z.a"), ("c.result.email", "z.b")),
              registry=registry_of(contacts, zipped), text=CSV)
    assert run.results["z"]["result"] == "Idaida@x|Ninanina@x"


def test_a_reader_of_a_blank_chain_is_not_called_either():
    CALLED.clear()
    run = ran({"id": "c", "node": "contacts"}, {"id": "u", "node": "upper"}, {"id": "v", "node": "upper"},
              edges=(("c.result.email", "u.text"), ("u.result", "v.text")), text=CSV)
    assert run.results["v"]["result"] == ["IDA@X", None, "NINA@X"]
    assert len(CALLED) == 4
    assert [(n["node_id"], n["row"]) for n in run.not_called] == [("u", (1,)), ("v", (1,))]


def test_a_blank_row_leaves_every_series_on_its_index_so_a_zip_stays_paired():
    @node("blank-some")
    def blank_some(text: str) -> Series[str | None]:
        return [None if x == "_" else x for x in text.split(",")]

    @node("ask")
    def ask(text: str) -> str | None:
        return text + "?"

    @node("paired")
    def paired(a: Series[str], b: Series[str | None]) -> str:
        return str(list(zip(a, b, strict=True)))

    run = ran({"id": "a", "node": "blank-some"}, {"id": "b", "node": "ask"}, {"id": "z", "node": "paired"},
              edges=(("a.result", "b.text"), ("a.result", "z.a"), ("b.result", "z.b")), registry=registry_of(blank_some, ask, paired), text="1,_,3")
    assert run.results["z"]["result"] == str([("1", "1?"), ("3", "3?")])


# -- copy-down and gathers ---------------------------------------------------------------------

THREE_LEVELS = (
    {"id": "docs", "node": "split", "values": {"sep": "|"}},
    {"id": "paras", "node": "split", "values": {"sep": ";"}},
    {"id": "lines", "node": "split"},
)
THREE_EDGES = (("docs.result", "paras.text"), ("paras.result", "lines.text"))


@node("labelled")
def labelled(label: str, texts: Series[str]) -> str:
    return f"{label}={','.join(texts)}"


@node("labelled-table")
def labelled_table(label: str, table: Table) -> str:
    return f"{label}={','.join(table['result'].to_list())}"


@node("two-columns")
def two_columns(a: Series[str], b: Series[str]) -> str:
    return f"{','.join(a)}+{','.join(b)}"


DEEP = registry_of(split, labelled, labelled_table, two_columns, show)


def test_each_series_output_makes_rows_on_an_index_of_its_own_and_of_its_own_length():
    @dataclass(frozen=True)
    class Both:
        a: Series[str]
        b: Series[str]

    @node("uneven")
    def uneven(text: str) -> Both:
        return Both(a=["x"], b=("y", "z"))  # any sequence will do

    run = ran({"id": "u", "node": "uneven"}, {"id": "ua", "node": "upper"}, {"id": "ub", "node": "upper"},
              edges=(("u.a", "ua.text"), ("u.b", "ub.text")), registry=registry_of(uneven, upper), text="t")
    assert run.graph.plan == {"u": "once", "ua": "per row of u.a", "ub": "per row of u.b"}
    assert (run.results["u"]["a"], run.results["u"]["b"]) == (["x"], ["y", "z"])
    assert (run.results["ua"]["result"], run.results["ub"]["result"]) == (["X"], ["Y", "Z"])


def test_a_node_receives_a_plain_list_and_may_return_any_sequence():
    seen = []

    @node("tuple-split")
    def tuple_split(text: str) -> Series[str]:
        return tuple(text.split(","))

    @node("collect")
    def collect(parts: Series[str]) -> str:
        seen.append(type(parts))
        return "+".join(parts)

    run = ran({"id": "s", "node": "tuple-split"}, {"id": "c", "node": "collect"}, edges=(("s.result", "c.parts"),),
              registry=registry_of(tuple_split, collect), text="a,b")
    assert run.results == {"s": {"result": ["a", "b"]}, "c": {"result": "a+b"}}
    assert seen == [list]


def test_a_scalar_beside_a_series_is_copied_down_every_row():
    run = ran({"id": "docs", "node": "split"}, {"id": "prefix", "node": "upper"}, {"id": "p", "node": "pair"},
              edges=(("prefix.result", "p.a"), ("docs.result", "p.b")), **{"docs.text": "a,b", "prefix.text": "p"})
    assert run.graph.plan["p"] == "per row of docs.result"
    assert run.results["p"]["result"] == ["P:a", "P:b"]


def test_a_parent_value_is_copied_down_onto_its_child_rows():
    run = ran({"id": "docs", "node": "split", "values": {"sep": "|"}}, {"id": "lines", "node": "split", "values": {"sep": "/"}}, {"id": "p", "node": "pair"},
              edges=(("docs.result", "lines.text"), ("lines.result", "p.a"), ("docs.result", "p.b")), text="a/b|c")
    assert run.graph.plan["p"] == "per row of lines.result"
    assert run.results["p"]["result"] == ["a:a/b", "b:a/b", "c:c"]


def test_one_once_value_into_a_series_input_is_a_list_of_one():
    run = ran({"id": "a", "node": "upper"}, {"id": "j", "node": "join"}, edges=(("a.result", "j.texts"),), text="a")
    assert run.graph.plan["j"] == "once" and run.results["j"]["result"] == "A"


def test_a_series_input_with_a_typed_in_list_or_its_default_gets_it_whole_once():
    @node("join-or-nothing")
    def join_or_nothing(texts: Series[str] = ()) -> str:
        return "+".join(texts) or "nothing"

    reg = registry_of(join, join_or_nothing)
    run = ran({"id": "typed", "node": "join", "values": {"texts": ["a", "b"]}}, {"id": "absent", "node": "join-or-nothing"}, registry=reg)
    assert run.graph.plan == {"typed": "once", "absent": "once"}
    assert run.results == {"typed": {"result": "a+b"}, "absent": {"result": "nothing"}}


def test_a_closed_list_input_is_one_value_with_its_list_and_with_its_default():
    @node("tags")
    def tags(tags: list[str] = ["x"]) -> str:  # noqa: B006
        return "|".join(tags)

    reg = registry_of(tags)
    assert ran({"id": "t", "node": "tags", "values": {"tags": ["a", "b", "c"]}}, registry=reg).outputs == {"t.result": "a|b|c"}
    assert ran({"id": "t", "node": "tags"}, registry=reg).outputs == {"t.result": "x"}


def test_a_one_value_input_on_a_node_and_a_deep_column_run_the_node_per_row_of_the_node_input():
    @node("deep")
    def deep(doc: str, lines: Series[str]) -> str:
        return f"{doc}:{len(lines)}"

    run = ran(*THREE_LEVELS, {"id": "d", "node": "deep"}, edges=(*THREE_EDGES, ("docs.result", "d.doc"), ("lines.result", "d.lines")),
              registry=registry_of(split, deep), text="a,b;c|d")
    assert run.graph.plan["d"] == "per row of docs.result"
    assert run.results["d"]["result"] == ["a,b;c:3", "d:1"]


def test_a_value_on_a_document_with_a_series_of_lines_gets_every_line_of_the_document_flat():
    run = ran(*THREE_LEVELS, {"id": "n", "node": "labelled"}, edges=(*THREE_EDGES, ("docs.result", "n.label"), ("lines.result", "n.texts")),
              registry=DEEP, text="a,b;c|d")
    assert run.graph.plan["n"] == "per row of docs.result"
    assert run.results["n"]["result"] == ["a,b;c=a,b,c", "d=d"]


def test_a_table_of_lines_with_a_value_on_a_document_gets_every_line_of_the_document():
    run = ran(*THREE_LEVELS, {"id": "n", "node": "labelled-table"}, edges=(*THREE_EDGES, ("docs.result", "n.label"), ("lines.result", "n.table")),
              registry=DEEP, text="a,b;c|d")
    assert run.results["n"]["result"] == ["a,b;c=a,b,c", "d=d"]


def test_a_value_on_a_paragraph_keeps_a_series_of_lines_to_that_paragraph():
    run = ran(*THREE_LEVELS, {"id": "n", "node": "labelled"}, edges=(*THREE_EDGES, ("paras.result", "n.label"), ("lines.result", "n.texts")),
              registry=DEEP, text="a,b;c|d")
    assert run.results["n"]["result"] == ["a,b=a,b", "c=c", "d=d"]


def test_a_document_with_no_paragraphs_runs_with_an_empty_series():
    run = ran(*THREE_LEVELS, {"id": "n", "node": "labelled"}, edges=(*THREE_EDGES, ("docs.result", "n.label"), ("lines.result", "n.texts")),
              registry=DEEP, text="a||b")
    assert run.results["n"]["result"] == ["a=a", "=", "b=b"]


def test_a_gather_on_a_node_put_on_a_row_reads_only_the_rows_under_it_from_several_sources():
    @node("joined-at-row")
    def joined_at_row(items: Series[str], at: str) -> str:
        return f"{at}:{'+'.join(items)}"

    run = ran({"id": "s", "node": "split"}, {"id": "a", "node": "split", "values": {"sep": ";"}}, {"id": "o", "node": "upper"},
              {"id": "j", "node": "joined-at-row"},
              edges=(("s.result", "a.text"), ("o.result", "j.items"), ("a.result", "j.items"), ("s.result", "j.at")),
              registry=registry_of(split, upper, joined_at_row), **{"s.text": "a;b,c", "o.text": "o"})
    assert run.results["j"]["result"] == ["a;b:a+b+O", "c:c+O"]  # several edges gather in node order


def test_a_gather_of_a_source_on_the_nodes_own_row_is_a_list_of_one():
    @node("joined-at-row")
    def joined_at_row(items: Series[str], at: str) -> str:
        return f"{at}:{'+'.join(items)}"

    run = ran({"id": "s", "node": "split"}, {"id": "j", "node": "joined-at-row"}, edges=(("s.result", "j.items"), ("s.result", "j.at")),
              registry=registry_of(split, joined_at_row), text="a,b")
    assert run.results["j"]["result"] == ["a:a", "b:b"]


def test_a_gather_reading_rows_that_are_not_under_the_nodes_row_is_refused():
    @node("joined-at-row")
    def joined_at_row(items: Series[str], at: str) -> str:
        return at

    reg = registry_of(split, joined_at_row)
    assert problems({"id": "s", "node": "split"}, {"id": "o", "node": "split"}, {"id": "j", "node": "joined-at-row"},
                    edges=(("o.result", "j.items"), ("s.result", "j.at")), registry=reg) == ["rows_not_related"]


def test_two_series_of_sibling_tables_each_gather_under_the_row_they_share():
    """``n`` runs per paragraph: ``a`` gathers that paragraph's lines, ``b`` every line of the other table under the same document."""
    run = ran(*THREE_LEVELS, {"id": "again", "node": "split", "values": {"sep": ";"}}, {"id": "lines2", "node": "split"},
              {"id": "n", "node": "two-columns"},
              edges=(*THREE_EDGES, ("docs.result", "again.text"), ("again.result", "lines2.text"), ("lines.result", "n.a"), ("lines2.result", "n.b")),
              registry=DEEP, text="a;b,c|d")

    assert run.results["n"]["result"] == ["a+a,b,c", "b,c+a,b,c", "d+d"]


def test_a_reduction_of_a_sibling_table_lands_beside_each_row_of_the_other_through_json():
    """Every keyword of a document beside each of its invoices: the gather starts at the document, the rows they share."""

    @dataclass(frozen=True)
    class Parts:
        invoices: Series[str]
        keywords: Series[str]

    @node("parts")
    def parts(text: str) -> Parts:
        head, _, tail = text.partition(":")
        return Parts(head.split(","), tail.split(","))

    @node("beside")
    def beside(invoice: str, keywords: Series[str]) -> str:
        return f"{invoice}[{'/'.join(keywords)}]"

    reg = registry_of(split, parts, beside)
    nodes = ({"id": "docs", "node": "split", "values": {"sep": "|"}}, {"id": "p", "node": "parts"}, {"id": "b", "node": "beside"})
    edges = (("docs.result", "p.text"), ("p.invoices", "b.invoice"), ("p.keywords", "b.keywords"))
    run = ran(*nodes, edges=edges, registry=reg, text="i1,i2:k1,k2|i3:k3")

    assert run.results["b"]["result"] == ["i1[k1/k2]", "i2[k1/k2]", "i3[k3]"]
    assert through_json(run).results == run.results
    assert graph(*nodes, edges=edges, registry=reg).plan["b"] == "per row of p.invoices"


# -- empty and skipped rows --------------------------------------------------------------------


def test_an_unconnected_series_input_takes_its_default():
    @node("join-or-nothing")
    def join_or_nothing(texts: Series[str] = ()) -> str:
        return "+".join(texts)

    assert ran({"id": "j", "node": "join-or-nothing"}, registry=registry_of(join_or_nothing)).outputs == {"j.result": ""}


def test_a_node_over_an_empty_list_never_starts_and_the_reduction_runs_on_an_empty_list():
    import asyncio

    loaded = graph({"id": "s", "node": "split"}, {"id": "u", "node": "upper"}, {"id": "j", "node": "join"},
                   edges=(("s.result", "u.text"), ("u.result", "j.texts")))

    async def events():
        return [e async for e in loaded.with_inputs(text="").stream()]

    said = asyncio.run(events())
    assert [e.type for e in said if getattr(e, "node_id", None) == "u"] == ["node_complete"]  # never started
    run = said[-1].run
    assert run.status == "complete" and run.results["u"]["result"] == [] and run.results["j"]["result"] == ""
    assert through_json(run).results["u"]["result"] == []


def test_a_skipped_once_value_skips_what_hangs_off_it():
    run = ran({"id": "g", "node": "decide"}, {"id": "y", "node": "upper"}, {"id": "n", "node": "upper"},
              edges=(("g.yes", "y.text"), ("g.no", "n.text")), text="hey")
    assert run.results["y"]["result"] == "HEY" and run.results["n"]["result"] is None


def test_a_failing_node_on_the_branch_not_taken_never_runs():
    @node("always-fail")
    def always_fail(text: str) -> str:
        raise ValueError("never")

    run = ran({"id": "g", "node": "decide"}, {"id": "y", "node": "upper"}, {"id": "n", "node": "always-fail"},
              edges=(("g.yes", "y.text"), ("g.no", "n.text")), registry=registry_of(decide, upper, always_fail), text="hey")
    assert run.results["y"]["result"] == "HEY"


def test_a_skip_at_the_top_leaves_the_readers_of_a_table_or_a_series_without_results_before_and_after_json():
    for reader in ("c.result.name", "s.result"):
        run = ran({"id": "g", "node": "decide"}, {"id": "c", "node": "contacts"}, {"id": "s", "node": "split"}, {"id": "r", "node": "upper"},
                  edges=(("g.yes", "c.text"), ("g.yes", "s.text"), (reader, "r.text")), text="no")
        assert run.results["r"]["result"] == [] == through_json(run).results["r"]["result"]


def test_a_skipped_source_gives_a_series_an_empty_list_alone_and_per_row():
    alone = ran({"id": "b", "node": "decide"}, {"id": "j", "node": "join"}, edges=(("b.no", "j.texts"),), text="yes")
    assert alone.results["j"]["result"] == ""
    per_row = ran({"id": "s", "node": "split"}, {"id": "b", "node": "decide"}, {"id": "j", "node": "join"}, {"id": "p", "node": "pair"},
                  edges=(("s.result", "b.text"), ("b.no", "j.texts"), ("s.result", "p.a"), ("j.result", "p.b")), text="p,q")
    assert per_row.results["j"]["result"] == ""  # the gather runs once, over two skipped rows


def test_a_skip_on_a_document_reaches_a_reader_on_its_lines_and_the_other_document_runs():
    run = ran({"id": "d", "node": "split", "values": {"sep": "|"}}, {"id": "g", "node": "decide"}, {"id": "l", "node": "split"},
              {"id": "t", "node": "pair"}, {"id": "u", "node": "upper"},
              edges=(("d.result", "g.text"), ("d.result", "l.text"), ("l.result", "t.a"), ("g.yes", "t.b"), ("t.result", "u.text")), text="a,b|c,no")
    assert run.graph.plan["u"] == "per row of l.result"
    assert run.results["u"]["result"] == ["A:A,B", "B:A,B", None, None]


# -- tables ------------------------------------------------------------------------------------

DOCS = {"id": "docs", "node": "docs"}
READ = {"id": "read", "node": "invoices"}
READ_EDGE = ("docs.result", "read.text")


def sheet(run: Any, node_id: str = "x") -> Any:
    return run.results[node_id]["result"]


def test_a_table_wired_alone_into_a_table_input_reaches_the_node_as_that_table():
    seen = []

    @node("hold")
    def hold(table: Table) -> Table:
        seen.append(table)
        return table

    run = ran({"id": "c", "node": "contacts"}, {"id": "a", "node": "hold"}, {"id": "b", "node": "hold"},
              edges=(("c.result", "a.table"), ("a.result", "b.table")), registry=registry_of(contacts, hold), text=CSV)
    assert seen[0].equals(seen[1]) and seen[0].equals(run.results["c"]["result"])


def test_one_column_is_a_table_of_one_column_named_after_its_output():
    run = ran({"id": "c", "node": "contacts"}, {"id": "s", "node": "send-or-say"}, {"id": "x", "node": "show"},
              edges=(("c.result.email", "s.email"), ("s.result", "x.table")), text=CSV)
    assert sheet(run).to_dicts() == [{"result": "sent to ida@x"}, {"result": "no address"}, {"result": "sent to nina@x"}]


def test_a_table_and_a_column_of_it_wired_beside_it_are_one_table_with_no_column_twice():
    run = ran({"id": "c", "node": "contacts"}, {"id": "x", "node": "show"}, edges=(("c.result.name", "x.table"), ("c.result", "x.table")), text=CSV)
    assert sorted(sheet(run).columns) == ["email", "name"] and sheet(run).height == 3


def test_a_node_filtering_every_row_away_returns_a_table_of_no_rows():
    @node("none-left")
    def none_left(table: Table) -> Table:
        return table.filter(pl.lit(False))

    run = ran({"id": "c", "node": "contacts"}, {"id": "f", "node": "none-left"}, edges=(("c.result", "f.table"),), registry=registry_of(contacts, none_left), text=CSV)
    assert run.results["f"]["result"].height == 0


def test_a_filter_is_followed_by_a_per_row_node_on_its_columns_and_both_combine():
    @node("with-email")
    def with_email(table: Table[Contact]) -> Table[Contact]:
        return table.filter(pl.col("email").is_not_null())

    run = ran({"id": "c", "node": "contacts"}, {"id": "f", "node": "with-email"}, {"id": "s", "node": "send"}, {"id": "x", "node": "show"},
              edges=(("c.result", "f.table"), ("f.result.email", "s.email"), ("f.result", "x.table"), ("s.result", "x.table")),
              registry=registry_of(contacts, with_email, send, show), text=CSV)
    assert sheet(run).to_dicts() == [{"name": "Ida", "email": "ida@x", "result": "sent to ida@x"}, {"name": "Nina", "email": "nina@x", "result": "sent to nina@x"}]


def test_after_a_sort_and_a_column_choice_a_per_row_node_runs_in_the_new_order_and_a_blank_is_not_called():
    @node("sort")
    def sort(table: Table) -> Table:
        return table.sort("name", descending=True)

    @node("choose")
    def choose(table: Table, keep: str) -> Table:
        return table.select(keep.split(","))

    run = ran({"id": "c", "node": "contacts"}, {"id": "o", "node": "sort"}, {"id": "p", "node": "choose", "values": {"keep": "email"}},
              {"id": "s", "node": "send"}, edges=(("c.result", "o.table"), ("o.result", "p.table"), ("p.result.email", "s.email")),
              registry=registry_of(contacts, sort, choose, send), text=CSV)
    assert run.results["s"]["result"] == ["sent to nina@x", "sent to ida@x", None]


def test_a_per_row_node_on_a_column_whose_input_says_none_is_called_with_the_blank_and_its_column_lands_beside_the_table():
    run = ran({"id": "c", "node": "contacts"}, {"id": "s", "node": "send-or-say"}, {"id": "x", "node": "show"},
              edges=(("c.result.email", "s.email"), ("c.result", "x.table"), ("s.result", "x.table")), text=CSV)
    assert sheet(run)["result"].to_list() == ["sent to ida@x", "no address", "sent to nina@x"]
    assert sheet(run)["name"].to_list() == ["Ida", "Bo", "Nina"]


def test_invoices_per_document_get_their_check_and_a_skipped_check_is_a_gap_on_its_own_row():
    run = ran(DOCS, READ, {"id": "check", "node": "check"}, {"id": "total", "node": "total"},
              edges=(READ_EDGE, ("read.result.vendor", "check.vendor"), ("read.result.amount", "total.amounts")), text="acme:10,unknown:5|beta:7")
    assert run.graph.plan == {"docs": "once", "read": "per row of docs.result", "check": "per row of read.result", "total": "per row of docs.result"}
    assert run.results["check"]["result"] == ["acme ok", None, "beta ok"]
    assert run.results["total"]["result"] == [15.0, 7.0]


def test_a_document_with_no_invoices_totals_nothing():
    run = ran(DOCS, READ, {"id": "total", "node": "total"}, edges=(READ_EDGE, ("read.result.amount", "total.amounts")), text="acme:10|")
    assert run.results["total"]["result"] == [10.0, 0.0]


def test_a_parent_value_is_copied_down_onto_each_row_of_the_table_under_it():
    run = ran(DOCS, READ, {"id": "x", "node": "show"}, edges=(READ_EDGE, ("read.result.vendor", "x.table"), ("docs.result", "x.table")),
              text="acme:10,beta:5|gamma:7")
    assert run.graph.plan["x"] == "per row of docs.result"  # a column into a table input gathers the rows under the row above it
    assert [s.select("vendor", "result").to_dicts() for s in sheet(run)] == [
        [{"vendor": "acme", "result": "acme:10,beta:5"}, {"vendor": "beta", "result": "acme:10,beta:5"}], [{"vendor": "gamma", "result": "gamma:7"}],
    ]


def test_a_table_input_on_a_node_running_per_document_gets_that_documents_rows():
    run = ran(DOCS, READ, {"id": "check", "node": "check"}, {"id": "x", "node": "show"},
              edges=(READ_EDGE, ("read.result.vendor", "check.vendor"), ("read.result", "x.table"), ("check.result", "x.table")), text="acme:10,beta:5|gamma:7")
    assert run.graph.plan["x"] == "per row of docs.result"
    assert [s.to_dicts() for s in sheet(run)] == [
        [{"vendor": "acme", "amount": 10.0, "result": "acme ok"}, {"vendor": "beta", "amount": 5.0, "result": "beta ok"}],
        [{"vendor": "gamma", "amount": 7.0, "result": "gamma ok"}],
    ]


def test_a_parent_column_beside_a_table_made_per_row_of_it_is_copied_down_per_parent():
    run = ran(DOCS, READ, {"id": "j", "node": "upper"}, {"id": "x", "node": "show"},
              edges=(READ_EDGE, ("docs.result", "j.text"), ("j.result", "x.table"), ("read.result", "x.table")), text="acme:10,beta:5|gamma:7")
    assert run.graph.plan["x"] == "per row of docs.result"
    assert [s.to_dicts() for s in sheet(run)] == [
        [{"result": "ACME:10,BETA:5", "vendor": "acme", "amount": 10.0}, {"result": "ACME:10,BETA:5", "vendor": "beta", "amount": 5.0}],
        [{"result": "GAMMA:7", "vendor": "gamma", "amount": 7.0}],
    ]


def test_wiring_the_parent_as_well_adds_no_row():
    alone = ran(DOCS, READ, {"id": "x", "node": "show"}, edges=(READ_EDGE, ("read.result.vendor", "x.table")), text="acme:10,beta:5||gamma:7")
    beside = ran(DOCS, READ, {"id": "x", "node": "show"}, edges=(READ_EDGE, ("docs.result", "x.table"), ("read.result.vendor", "x.table")),
                 text="acme:10,beta:5||gamma:7")
    vendors = [[s["vendor"].to_list() for s in sheet(run)] for run in (alone, beside)]
    assert vendors[0] == vendors[1] == [["acme", "beta"], [], ["gamma"]]


def test_a_merge_three_levels_deep_copies_every_level_down_and_a_paragraph_with_no_lines_has_no_row():
    run = ran(*THREE_LEVELS, {"id": "x", "node": "show"},
              edges=(*THREE_EDGES, ("docs.result", "x.table"), ("paras.result", "x.table"), ("lines.result", "x.table")), registry=DEEP, text="a,b;;c|d")
    assert run.graph.plan["x"] == "per row of paras.result"
    assert {tuple(s.columns) for s in sheet(run)} == {("result", "result_paras", "result_lines")}
    assert [s.rows() for s in sheet(run)] == [[("a,b;;c", "a,b", "a"), ("a,b;;c", "a,b", "b")], [], [("a,b;;c", "c", "c")], [("d", "d", "d")]]


def test_parents_with_nothing_under_them_give_a_table_of_no_rows_with_every_column():
    run = ran(DOCS, READ, {"id": "x", "node": "show"},
              edges=(READ_EDGE, ("docs.result", "x.table"), ("read.result.vendor", "x.table"), ("read.result.amount", "x.table")), text="|")
    assert [(s.height, sorted(s.columns)) for s in sheet(run)] == [(0, ["amount", "result", "vendor"])] * 2


def test_two_unrelated_tables_into_one_table_input_are_refused_and_told_to_join_on_a_key():
    loaded = graph({"id": "c", "node": "contacts"}, {"id": "o", "node": "contacts"}, {"id": "x", "node": "show"},
                   edges=(("c.result", "x.table"), ("o.result", "x.table")))
    assert [p.type for p in loaded.problems] == ["rows_not_related"]
    assert "join them on a key" in loaded.problems[0].msg


def test_two_tables_made_per_document_share_no_rows_and_the_author_is_told():
    assert problems(DOCS, READ, {"id": "again", "node": "invoices"}, {"id": "x", "node": "show"},
                    edges=(READ_EDGE, ("docs.result", "again.text"), ("read.result.vendor", "x.table"), ("again.result.amount", "x.table"))) == ["rows_not_related"]


def test_a_table_a_node_made_from_another_table_is_unrelated_to_a_third():
    assert problems({"id": "c", "node": "contacts"}, {"id": "o", "node": "contacts"}, {"id": "keep", "node": "show"}, {"id": "x", "node": "show"},
                    edges=(("o.result", "keep.table"), ("c.result", "x.table"), ("keep.result", "x.table"))) == ["rows_not_related"]


def test_a_table_made_per_document_offers_its_columns_on_rows_under_each_document():
    loaded = graph(DOCS, READ, {"id": "check", "node": "check"}, edges=(READ_EDGE, ("read.result.vendor", "check.vendor")))
    assert loaded.problems == []
    assert loaded.plan == {"docs": "once", "read": "per row of docs.result", "check": "per row of read.result"}
    (result,) = loaded.describe()["nodes"]["read"]["outputs"]
    assert result["shape"] == "table" and result["columns"]["amount"] == {"type": "number"}


def test_a_bare_table_offers_no_columns():
    (result,) = graph({"id": "s", "node": "show"}).describe()["nodes"]["s"]["outputs"]
    assert (result["shape"], result["columns"]) == ("table", {})


# -- branches ----------------------------------------------------------------------------------


@node("parity")
def parity(n: str) -> YesNo:
    return YesNo(n, SKIPPED) if int(n) % 2 == 0 else YesNo(SKIPPED, n)


def test_series_inputs_reading_what_hangs_off_opposite_per_row_branches_are_refused():
    assert problems({"id": "n", "node": "split"}, {"id": "r", "node": "parity"}, {"id": "x", "node": "upper"}, {"id": "y", "node": "upper"},
                    {"id": "c", "node": "two-columns"},
                    edges=(("n.result", "r.n"), ("r.yes", "x.text"), ("r.no", "y.text"), ("x.result", "c.a"), ("y.result", "c.b")),
                    registry=registry_of(split, parity, upper, two_columns)) == ["series_inputs_never_share_a_row"]


def test_a_per_row_decision_copied_down_from_the_row_above_still_keeps_two_branches_apart():
    assert problems({"id": "s", "node": "split", "values": {"sep": "|"}}, {"id": "d", "node": "parity"}, {"id": "sp", "node": "split"},
                    {"id": "ua", "node": "pair"}, {"id": "ub", "node": "pair"}, {"id": "z", "node": "two-columns"},
                    edges=(("s.result", "d.n"), ("s.result", "sp.text"), ("sp.result", "ua.a"), ("d.yes", "ua.b"), ("sp.result", "ub.a"), ("d.no", "ub.b"),
                           ("ua.result", "z.a"), ("ub.result", "z.b")),
                    registry=registry_of(split, parity, pair, two_columns)) == ["series_inputs_never_share_a_row"]


def test_a_row_skipped_on_one_series_leaves_every_series_on_that_table():
    run = ran({"id": "n", "node": "split"}, {"id": "r", "node": "parity"}, {"id": "x", "node": "upper"}, {"id": "z", "node": "zipped"},
              edges=(("n.result", "r.n"), ("r.yes", "x.text"), ("n.result", "z.a"), ("x.result", "z.b")), registry=registry_of(split, parity, upper, zipped),
              text="1,2,3,4")
    assert run.results["z"]["result"] == "22|44"


# -- embedded graphs ---------------------------------------------------------------------------


def embedded(node_id: str, *nodes: dict[str, Any], edges: tuple[tuple[str, str], ...] = ()) -> Any:
    """A graph of ``nodes`` as one node ``node_id``: its inputs are the open inputs of its first nodes."""
    return node(node_id)(GraphRecord.model_validate({"nodes": [{"node": n["id"], **n} for n in nodes], "edges": [list(e) for e in edges]}))


def test_a_table_made_inside_per_outer_row_is_a_child_of_that_row_fed_or_not():
    fed = embedded("splitter", {"id": "lines", "node": "split", "values": {"sep": "/"}}, {"id": "join", "node": "join"}, edges=(("lines.result", "join.texts"),))
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "splitter"}, edges=(("docs.result", "emb.text"),),
              registry=registry_of(split, join, fed), text="a/b,c")
    assert run.graph.plan["emb"] == "per row of docs.result"
    assert run.results["emb"]["result"] == ["a+b", "c"]

    unfed = embedded("splitter-unfed", {"id": "entered", "node": "upper"}, {"id": "lines", "node": "split", "values": {"text": "x/y", "sep": "/"}},
                     {"id": "join", "node": "join"}, {"id": "p", "node": "pair"},
                     edges=(("lines.result", "join.texts"), ("entered.result", "p.a"), ("join.result", "p.b")))
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "splitter-unfed"}, edges=(("docs.result", "emb.text"),),
              registry=registry_of(split, join, upper, pair, unfed), text="a,b")
    assert run.results["emb"]["result"] == ["A:x+y", "B:x+y"]


def test_an_inner_table_no_entering_edge_reaches_read_beside_an_entering_value_is_made_under_each_outer_row():
    """Inside, ``p`` reads the entering value and each row of ``lines``: run alone that is one graph, so it is one per outer row too."""
    inner = embedded("letters", {"id": "entered", "node": "upper"}, {"id": "lines", "node": "split", "values": {"text": "x/y", "sep": "/"}},
                     {"id": "p", "node": "pair"}, edges=(("entered.result", "p.a"), ("lines.result", "p.b")))
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "letters"}, edges=(("docs.result", "emb.text"),),
              registry=registry_of(split, upper, pair, inner), text="a,b")
    assert run.results["emb"]["result"] == ["A:x", "A:y", "B:x", "B:y"]


def test_rows_not_related_inside_an_embedding_names_no_inner_id():
    """``x`` gathers the whole outer column and makes its own rows; ``p`` reads one of them beside each outer row."""

    @node("each")
    def each(texts: Series[str]) -> Series[str]:
        return texts

    inner = embedded("mixing", {"id": "entered", "node": "upper"}, {"id": "x", "node": "each"}, {"id": "p", "node": "pair"},
                     edges=(("entered.result", "p.a"), ("x.result", "p.b")))
    (problem,) = graph({"id": "docs", "node": "split"}, {"id": "emb", "node": "mixing"}, edges=(("docs.result", "emb.text"), ("docs.result", "emb.texts")),
                       registry=registry_of(split, upper, pair, each, inner)).problems
    assert problem.type == "rows_not_related" and problem.loc == ("emb",) and "/" not in problem.msg


def test_inputs_from_a_table_and_its_child_table_run_the_inner_nodes_on_the_deeper_rows():
    @node("shout")
    def shout(word: str) -> str:
        return word.upper()

    inner = embedded("pair-up", {"id": "ua", "node": "shout"}, {"id": "ub", "node": "upper"}, {"id": "p", "node": "pair"},
                     edges=(("ua.result", "p.a"), ("ub.result", "p.b")))
    run = ran({"id": "docs", "node": "split", "values": {"sep": "|"}}, {"id": "lines", "node": "split"}, {"id": "emb", "node": "pair-up"},
              edges=(("docs.result", "lines.text"), ("docs.result", "emb.word"), ("lines.result", "emb.text")),
              registry=registry_of(split, upper, pair, shout, inner), text="a,b|c")
    assert run.graph.plan["emb"] == "per row of lines.result"
    assert run.results["emb"]["result"] == ["A,B:A", "A,B:B", "C:C"]


def test_a_list_into_an_embedded_series_input_is_one_value_and_the_graph_runs_once():
    inner = embedded("joiner", {"id": "join", "node": "join"})
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "joiner"}, edges=(("docs.result", "emb.texts"),),
              registry=registry_of(split, join, inner), text="a,b")
    assert run.graph.plan["emb"] == "once" and run.results["emb"]["result"] == "a+b"


def test_two_unrelated_tables_into_one_embedding_are_refused_once_with_the_authors_addresses():
    inner = embedded("pairs", {"id": "p", "node": "pair"})
    loaded = graph({"id": "d1", "node": "split"}, {"id": "d2", "node": "split"}, {"id": "emb", "node": "pairs"},
                   edges=(("d1.result", "emb.a"), ("d2.result", "emb.b")), registry=registry_of(split, pair, inner))
    (problem,) = loaded.problems
    assert problem.type == "rows_not_related"
    assert "emb/" not in problem.msg and "emb/" not in str(problem.ctx), problem


def test_two_unrelated_tables_out_of_one_embedding_are_refused_where_they_meet():
    @dataclass(frozen=True)
    class Left:
        a: Series[str]

    @dataclass(frozen=True)
    class Right:
        b: Series[str]

    @node("left")
    def left() -> Left:
        return Left(["x", "y"])

    @node("right")
    def right() -> Right:
        return Right(["p", "q"])

    inner = embedded("two-lists", {"id": "l", "node": "left"}, {"id": "r", "node": "right"})
    assert problems({"id": "emb", "node": "two-lists"}, {"id": "after", "node": "pair"}, edges=(("emb.a", "after.a"), ("emb.b", "after.b")),
                    registry=registry_of(left, right, pair, inner)) == ["rows_not_related"]


def test_a_table_input_from_rows_the_node_does_not_sit_on_is_refused_flat_and_embedded():
    @node("merged-at-row")
    def merged_at_row(table: Table, at: str) -> str:
        return at

    tables = ({"id": "x", "node": "contacts"}, {"id": "y", "node": "split"})
    flat = problems(*tables, {"id": "m", "node": "merged-at-row"}, edges=(("x.result", "m.table"), ("y.result", "m.at")),
                    registry=registry_of(contacts, split, merged_at_row))
    inner = embedded("g", {"id": "m", "node": "merged-at-row"})
    inside = problems(*tables, {"id": "e", "node": "g"}, edges=(("x.result", "e.table"), ("y.result", "e.at")),
                      registry=registry_of(contacts, split, merged_at_row, inner))
    assert flat == ["rows_not_related"] and inside == ["rows_not_related"]


def test_a_gather_inside_a_per_row_graph_reads_only_its_own_rows_from_several_sources():
    @node("shout")
    def shout(word: str) -> str:
        return word.upper()

    inner = embedded("gathers", {"id": "a", "node": "split", "values": {"sep": ";"}}, {"id": "o", "node": "shout"}, {"id": "j", "node": "join"},
                     edges=(("a.result", "j.texts"), ("o.result", "j.texts")))
    run = ran({"id": "s", "node": "split"}, {"id": "e", "node": "gathers"}, edges=(("s.result", "e.text"), ("s.result", "e.word")),
              registry=registry_of(split, shout, join, inner), text="a;b,c")
    assert run.graph.plan["e"] == "per row of s.result"
    assert run.results["e"]["result"] == ["a+b+A;B", "c+C"]


def test_a_series_input_of_an_embedding_fed_a_whole_column_sees_it_whole():
    inner = embedded("mixed", {"id": "u", "node": "upper"}, {"id": "j", "node": "join"})
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "mixed"}, edges=(("docs.result", "emb.text"), ("docs.result", "emb.texts")),
              registry=registry_of(split, upper, join, inner), text="a,b")
    assert set(run.graph.plan) == {"docs", "emb"}
    assert run.results["emb"] == {"u.result": ["A", "B"], "j.result": "a+b"}


def test_a_graph_embedded_under_a_row_runs_as_the_same_nodes_drawn_flat():
    nodes = ({"id": "i", "node": "split", "values": {"sep": ";"}}, {"id": "c", "node": "split", "values": {"sep": "/"}}, {"id": "p", "node": "pair"})
    edges = (("i.result", "c.text"), ("c.result", "p.a"), ("i.result", "p.b"))
    inner = embedded("deep", *nodes, edges=edges)
    reg = registry_of(split, pair, inner)
    embedded_run = ran({"id": "s", "node": "split"}, {"id": "e", "node": "deep"}, edges=(("s.result", "e.text"),), registry=reg, text="a/b;c,d")
    flat_run = ran({"id": "s", "node": "split"}, *nodes, edges=(("s.result", "i.text"), *edges), registry=reg, text="a/b;c,d")
    assert embedded_run.results["e"] == flat_run.results["p"]
    assert embedded_run.graph.plan["e"] == flat_run.graph.plan["p"]


def test_a_table_inputs_columns_are_named_the_same_embedded_as_alone():
    @node("columns")
    def columns(table: Table) -> str:
        return ",".join(table.columns)

    merge = ({"id": "c", "node": "contacts", "values": {"text": CSV}}, {"id": "s", "node": "send-or-say"}, {"id": "t", "node": "columns"})
    edges = (("c.result.email", "s.email"), ("c.result.name", "t.table"), ("s.result", "t.table"))
    reg = registry_of(split, contacts, send_or_say, columns, embedded("sheet", *merge, edges=edges))
    alone = ran(*merge, edges=edges, registry=reg)
    inside = ran({"id": "e", "node": "sheet"}, registry=reg)
    assert alone.results["t"]["result"] == inside.results["e"]["result"] == "name,result"


def test_a_table_input_inside_an_embedded_graph_combines_its_inner_sources():
    merge = ({"id": "c", "node": "contacts", "values": {"text": CSV}}, {"id": "s", "node": "send-or-say"}, {"id": "x", "node": "show"})
    edges = (("c.result.email", "s.email"), ("c.result", "x.table"), ("s.result", "x.table"))
    run = ran({"id": "e", "node": "sheet"}, registry=registry_of(contacts, send_or_say, show, embedded("sheet", *merge, edges=edges)))
    assert run.results["e"]["result"]["result"].to_list() == ["sent to ida@x", "no address", "sent to nina@x"]


def test_inside_an_embedding_a_rebuilt_table_and_a_column_made_per_row_of_it_combine():
    @node("rebuild")
    def rebuild(table: Table) -> Table[Contact]:
        return table.filter(pl.col("email").is_not_null())

    inner = embedded("rb", {"id": "k", "node": "rebuild"}, {"id": "s", "node": "send"}, {"id": "m", "node": "show"},
                     edges=(("k.result.email", "s.email"), ("k.result", "m.table"), ("s.result", "m.table")))
    run = ran({"id": "c", "node": "contacts"}, {"id": "sub", "node": "rb"}, edges=(("c.result", "sub.table"),),
              registry=registry_of(contacts, rebuild, send, show, inner), text=CSV)
    assert run.results["sub"]["result"].to_dicts() == [
        {"name": "Ida", "email": "ida@x", "result": "sent to ida@x"}, {"name": "Nina", "email": "nina@x", "result": "sent to nina@x"},
    ]


# -- skips above a group -----------------------------------------------------------------------


def test_a_skip_above_the_rows_skips_the_reduction_and_absent_rows_give_an_empty_list():
    skipped = ran({"id": "b", "node": "decide"}, {"id": "a", "node": "split"}, {"id": "j", "node": "join"},
                  edges=(("b.no", "a.text"), ("a.result", "j.texts")), text="ab")
    assert skipped.results["j"]["result"] is None
    empty = ran({"id": "a", "node": "split"}, {"id": "j", "node": "join"}, edges=(("a.result", "j.texts"),), text="")
    assert empty.results["j"]["result"] == ""


def test_a_skipped_node_makes_no_rows_so_everything_below_is_skipped():
    run = ran({"id": "g", "node": "decide"}, {"id": "docs", "node": "split"}, {"id": "u", "node": "upper"}, {"id": "j", "node": "join"},
              edges=(("g.no", "docs.text"), ("docs.result", "u.text"), ("u.result", "j.texts")), text="a,b")
    assert run.results["docs"]["result"] is None and run.results["u"]["result"] == [] and run.results["j"]["result"] is None


def test_a_document_whose_gate_is_shut_gets_no_rows_under_it_and_its_reduction_is_skipped():
    run = ran({"id": "docs", "node": "split", "values": {"sep": "|"}}, {"id": "g", "node": "decide"}, {"id": "paras", "node": "split", "values": {"sep": ";"}},
              {"id": "lines", "node": "split"}, {"id": "up", "node": "upper"}, {"id": "j", "node": "labelled"},
              edges=(("docs.result", "g.text"), ("g.yes", "paras.text"), ("paras.result", "lines.text"), ("lines.result", "up.text"),
                     ("docs.result", "j.label"), ("up.result", "j.texts")),
              registry=registry_of(split, decide, upper, labelled), text="a,b;c|no")
    assert run.results["lines"]["result"] == [["a", "b"], ["c"]]
    assert run.results["j"]["result"] == ["a,b;c=A,B,C", None]


def test_a_reduction_waits_for_the_unit_above_its_group_before_it_runs_on_nothing():
    import threading
    import time

    from conductor import Policy

    release = threading.Event()

    @node("held", policy=Policy(concurrency=2))
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


def test_a_document_skipped_above_its_table_is_not_totalled_and_the_others_still_total():
    run = ran(DOCS, {"id": "gate", "node": "decide"}, READ, {"id": "total", "node": "total"},
              edges=(("docs.result", "gate.text"), ("gate.yes", "read.text"), ("read.result.amount", "total.amounts")), text="acme:10|no|beta:7")
    assert run.results["total"]["result"] == [10.0, None, 7.0]


def test_skipped_inside_a_returned_list_leaves_gaps_on_those_rows_through_json():
    @node("sparse")
    def sparse() -> Series[str]:
        return ["a", SKIPPED, "c"]

    run = ran({"id": "s", "node": "sparse"}, {"id": "u", "node": "upper"}, edges=(("s.result", "u.text"),), registry=registry_of(sparse, upper))
    assert run.results["u"]["result"] == ["A", None, "C"]
    assert through_json(run).results == run.results


def test_two_series_at_different_depths_are_both_read_per_row_of_the_shallowest_level_above_them():
    run = ran(*THREE_LEVELS, {"id": "n", "node": "two-columns"}, edges=(*THREE_EDGES, ("paras.result", "n.a"), ("lines.result", "n.b")),
              registry=DEEP, text="a,b;c|d")
    assert run.graph.plan["n"] == "per row of docs.result"
    assert run.results["n"]["result"] == ["a,b,c+a,b,c", "d+d"]


def test_a_child_table_a_branch_skipped_gives_no_rows_and_the_sheet_keeps_its_columns():
    run = ran(DOCS, READ, {"id": "x", "node": "show"}, edges=(READ_EDGE, ("docs.result", "x.table"), ("read.result", "x.table")), text="acme:10|skip")
    assert [(s.height, sorted(s.columns)) for s in sheet(run)] == [(1, ["amount", "result", "vendor"]), (0, ["amount", "result", "vendor"])]


# -- a table input's columns -------------------------------------------------------------------


class Names(TypedDict):
    name: str


class Scored(TypedDict):
    name: str
    score: float


@node("names-in")
def names_in(table: Table[Names]) -> str:
    return ",".join(table.columns)


@node("scored-in")
def scored_in(table: Table[Scored]) -> Table[Scored]:
    return table


TABLES = registry_of(contacts, send, names_in, scored_in, show, upper)


def test_a_table_input_takes_a_table_with_at_least_its_columns_and_gets_exactly_them():
    run = ran({"id": "c", "node": "contacts"}, {"id": "n", "node": "names-in"}, edges=(("c.result", "n.table"),), registry=TABLES, text=CSV)
    assert run.results["n"]["result"] == "name"  # the email column is dropped
    assert problems({"id": "c", "node": "contacts"}, {"id": "s", "node": "scored-in"}, edges=(("c.result", "s.table"),), registry=TABLES) != []


def test_a_declared_table_input_missing_a_column_is_refused_when_checked():
    loaded = graph({"id": "c", "node": "contacts"}, {"id": "s", "node": "send"}, {"id": "t", "node": "scored-in"},
                   edges=(("c.result.email", "s.email"), ("c.result.name", "t.table"), ("s.result", "t.table")), registry=TABLES)
    assert [p.loc for p in loaded.problems] == [("t", "table")] and "score" in str(loaded.problems[0].ctx), loaded.problems


def test_a_typed_in_list_of_records_is_one_table_not_many():
    run = ran({"id": "x", "node": "show", "values": {"table": [{"a": "1"}, {"a": "2"}]}}, registry=TABLES)
    assert run.graph.plan["x"] == "once"
    assert isinstance(sheet(run), pl.DataFrame) and sheet(run).to_dicts() == [{"a": "1"}, {"a": "2"}]


# -- embedded graphs fold per outer row --------------------------------------------------------


def test_an_inner_reduction_over_the_entering_rows_runs_once_per_outer_row():
    inner = embedded("inner", {"id": "holder", "node": "upper"}, {"id": "gather", "node": "join"}, edges=(("holder.result", "gather.texts"),))
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "inner"}, {"id": "after", "node": "join"},
              edges=(("docs.result", "emb.text"), ("emb.result", "after.texts")), registry=registry_of(split, upper, join, inner), text="a,b,c")
    assert run.graph.plan["emb"] == "per row of docs.result"
    assert run.results["emb"]["result"] == ["A", "B", "C"]  # a gather of one per row, never all three once
    assert run.results["after"]["result"] == "A+B+C"


def test_a_skipped_source_inside_a_per_row_graph_gives_its_series_an_empty_list_on_each_row():
    inner = embedded("skips", {"id": "b", "node": "decide"}, {"id": "j", "node": "join"}, edges=(("b.no", "j.texts"),))
    run = ran({"id": "s", "node": "split"}, {"id": "e", "node": "skips"}, edges=(("s.result", "e.text"),), registry=registry_of(split, decide, join, inner), text="p,q")
    assert run.results["e"]["result"] == ["", ""]


def test_a_decision_over_a_whole_column_gates_what_hangs_off_the_branch_not_taken():
    @dataclass(frozen=True)
    class Decided:
        full: Annotated[Series[str], Param(exclusive_group="d")]
        empty: Annotated[Series[str], Param(exclusive_group="d")]

    @node("decide-list")
    def decide_list(texts: Series[str]) -> Decided:
        return Decided(full=list(texts), empty=SKIPPED) if texts else Decided(full=SKIPPED, empty=[])

    CALLED.clear()
    run = ran({"id": "docs", "node": "split"}, {"id": "d", "node": "decide-list"}, {"id": "full", "node": "join"},
              {"id": "up", "node": "upper"}, {"id": "after", "node": "join"},
              edges=(("docs.result", "d.texts"), ("d.full", "full.texts"), ("d.empty", "up.text"), ("up.result", "after.texts")),
              registry=registry_of(split, decide_list, join, upper), text="a,b")
    assert run.results["full"]["result"] == "a+b"
    assert run.results["up"]["result"] == [] and run.results["after"]["result"] is None
    assert CALLED == ["join ['a', 'b']"]


def test_a_skip_on_a_document_skips_the_reduction_over_its_lines_and_survives_json():
    run = ran({"id": "d", "node": "split", "values": {"sep": "|"}}, {"id": "g", "node": "decide"}, {"id": "l", "node": "split"},
              {"id": "p", "node": "pair"}, {"id": "j", "node": "join"},
              edges=(("d.result", "g.text"), ("d.result", "l.text"), ("l.result", "p.a"), ("g.yes", "p.b"), ("p.result", "j.texts")), text="x|y,no")
    assert run.results["j"]["result"] == ["x:x", None]
    assert through_json(run).results["j"]["result"] == ["x:x", None]


def test_a_skip_above_an_empty_group_hides_what_the_reduction_returned():
    run = ran({"id": "d", "node": "split", "values": {"sep": "|"}}, {"id": "g", "node": "decide"}, {"id": "l", "node": "split"},
              {"id": "t", "node": "pair"}, {"id": "m", "node": "split"}, {"id": "x", "node": "pair"}, {"id": "j", "node": "join"},
              edges=(("d.result", "g.text"), ("d.result", "l.text"), ("l.result", "t.a"), ("g.yes", "t.b"), ("l.result", "m.text"),
                     ("t.result", "x.a"), ("m.result", "x.b"), ("x.result", "j.texts")), text="p,no")
    assert run.graph.plan["j"] == "per row of l.result"
    assert run.results["j"]["result"] == [None, None]


# -- several edges into one input --------------------------------------------------------------


def test_two_edges_into_one_plain_input_that_can_both_hold_a_value_are_a_problem_never_first_wins():
    """``upper(a, b)`` would silently lose ``b``; two per-row edges on one table into one value would cover each row twice."""
    assert problems({"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "u", "node": "upper"},
                    edges=(("a.result", "u.text"), ("b.result", "u.text"))) == ["several_edges_into_one_value"]
    loaded = graph({"id": "s", "node": "split"}, {"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "m", "node": "upper"},
                   edges=(("s.result", "a.text"), ("s.result", "b.text"), ("a.result", "m.text"), ("b.result", "m.text")))
    (problem,) = loaded.problems
    assert (problem.type, problem.loc, problem.ctx) == ("several_edges_into_one_value", ("m", "text"), {"sources": "a.result, b.result"})


def test_two_branches_wired_into_one_plain_input_merge_back_onto_the_rows_and_the_chain_stays_on_them():
    run = ran({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "u", "node": "upper"}, {"id": "after", "node": "upper"},
              edges=(("s.result", "d.text"), ("d.yes", "u.text"), ("d.no", "u.text"), ("u.result", "after.text")), text="ok,no,fine")

    assert run.results["u"]["result"] == ["OK", "NO", "FINE"]
    assert run.graph.plan["after"] == "per row of s.result"


def test_tables_below_two_branches_merge_back_too():
    """What hangs off opposite branches is as exclusive as the branches themselves."""
    run = ran({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "y", "node": "upper"}, {"id": "n", "node": "upper"}, {"id": "u", "node": "upper"},
              edges=(("s.result", "d.text"), ("d.yes", "y.text"), ("d.no", "n.text"), ("y.result", "u.text"), ("n.result", "u.text")), text="ok,no")

    assert run.results["u"]["result"] == ["OK", "NO"]


def test_a_node_reading_both_branches_through_one_input_lies_on_neither_so_it_pairs_only_with_rows_that_have_both():
    run = ran({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "p", "node": "pair"},
              edges=(("s.result", "d.text"), ("d.yes", "p.a"), ("d.no", "p.a"), ("d.yes", "p.b")), text="ok,no,fine")

    assert run.results["p"]["result"] == ["ok:ok", None, "fine:fine"]


def test_a_series_fed_by_both_branches_shares_rows_with_one_fed_by_either():
    assert problems({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "z", "node": "zipped"},
                    edges=(("s.result", "d.text"), ("d.yes", "z.a"), ("d.no", "z.a"), ("d.yes", "z.b")),
                    registry=registry_of(split, decide, zipped)) == []


def test_opposite_branches_through_one_series_input_are_one_column_in_row_order():
    """``join(d.no, d.yes)`` reads the rows as the table holds them, not one source's rows after the other's."""
    run = ran({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "j", "node": "join"},
              edges=(("s.result", "d.text"), ("d.no", "j.texts"), ("d.yes", "j.texts")), text="v10,no11,v12")

    assert run.results["j"]["result"] == "v10+no11+v12"


def test_two_columns_of_one_table_into_a_series_input_read_that_table_row_by_row():
    run = ran({"id": "s", "node": "split"}, {"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "j", "node": "join"},
              edges=(("s.result", "a.text"), ("s.result", "b.text"), ("a.result", "j.texts"), ("b.result", "j.texts")), text="x,y")

    assert run.results["j"]["result"] == "X+X+Y+Y"


def test_columns_of_two_tables_of_their_own_into_one_series_input_follow_one_another():
    run = ran({"id": "a", "node": "split"}, {"id": "b", "node": "split"}, {"id": "j", "node": "join"},
              edges=(("a.result", "j.texts"), ("b.result", "j.texts")), **{"a.text": "x,y", "b.text": "z"})

    assert run.results["j"]["result"] == "x+y+z"


class Line(TypedDict):
    text: str


@dataclass(frozen=True)
class BigOrSmall:
    big: Annotated[Table[Line], Param(exclusive_group="size")]
    small: Annotated[Table[Line], Param(exclusive_group="size")]


@node("sort-out")
def sort_out(text: str) -> BigOrSmall:
    """A table of the text's words, as ``big`` when there are more than two, else ``small``."""
    words = pl.DataFrame({"text": text.split()})
    return BigOrSmall(words, SKIPPED) if words.height > 2 else BigOrSmall(SKIPPED, words)


@node("words")
def words(table: Table[Line]) -> str:
    return "-".join(table["text"])


@node("lines-of")
def lines_of(text: str) -> Table[Line]:
    return pl.DataFrame({"text": [text.upper()]})


ALTERNATIVES = registry_of(split, sort_out, words, lines_of, upper)


def test_alternative_tables_into_one_table_input_are_the_branch_that_ran():
    nodes = ({"id": "o", "node": "sort-out"}, {"id": "w", "node": "words"})
    edges = (("o.big", "w.table"), ("o.small", "w.table"))

    assert ran(*nodes, edges=edges, registry=ALTERNATIVES, text="a b c").outputs == {"w.result": "a-b-c"}
    assert ran(*nodes, edges=edges, registry=ALTERNATIVES, text="a b").outputs == {"w.result": "a-b"}


def test_alternatives_per_row_take_the_branch_each_row_took_through_json():
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


def test_whole_tables_that_can_both_hold_rows_into_one_table_input_are_still_rows_of_different_tables():
    assert problems({"id": "a", "node": "lines-of"}, {"id": "b", "node": "lines-of"}, {"id": "w", "node": "words"},
                    edges=(("a.result", "w.table"), ("b.result", "w.table")), registry=ALTERNATIVES) == ["rows_not_related"]


# -- lists typed in ----------------------------------------------------------------------------


@node("ask-each")
def ask_each(doc: str, question: str) -> str:
    return f"{doc}?{question}"


@node("answer")
def answer(doc: str, question: str) -> bool:
    raise Ask(f"{doc}: {question}")


TYPED = registry_of(split, upper, join, ask_each, answer, decide, pair)
QUESTIONS = {"id": "q", "node": "ask-each", "values": {"question": ["a", "b", "c"]}, "many": ["question"]}


def test_a_list_typed_into_a_node_that_runs_once_is_a_table_of_its_own_and_the_node_runs_per_item():
    run = ran({"id": "u", "node": "upper", "values": {"text": ["x", "y"]}, "many": ["text"]}, {"id": "j", "node": "join"},
              edges=(("u.result", "j.texts"),), registry=TYPED)

    assert run.graph.plan == {"u": "per row of u.text", "j": "once"}
    assert run.results == {"u": {"result": ["X", "Y"]}, "j": {"result": "X+Y"}}
    assert list(run.graph.inputs) == []


def test_flow_8a_questions_typed_into_a_node_per_document_are_a_table_under_each_document_and_answers_sit_beside_them():
    """«Spørg» runs per document: three questions typed in, two documents, six questions, each answered on its own row."""
    loaded = graph({"id": "d", "node": "split"}, {"id": "q", "node": "answer", "values": {"question": ["a", "b", "c"]}, "many": ["question"]},
                   edges=(("d.result", "q.doc"),), registry=TYPED)
    run = loaded.run(text="x,y")

    assert loaded.plan["q"] == "per row of q.question"
    assert [w.prompt for w in run.waiting] == ["x: a", "x: b", "x: c", "y: a", "y: b", "y: c"]
    assert [w.row for w in run.waiting] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    saved = loaded.load_run(run.to_json())
    done = saved.answer({w.id: w.row == (1, 2) for w in saved.waiting}).resume()
    assert done.results["q"]["result"] == [False, False, False, False, False, True]


def test_a_series_reads_a_typed_in_list_column_under_each_document_and_a_per_row_node_reads_it_per_row():
    run = ran({"id": "d", "node": "split"}, QUESTIONS, {"id": "j", "node": "join"}, {"id": "u", "node": "upper"},
              edges=(("d.result", "q.doc"), ("q.result", "j.texts"), ("q.result", "u.text")), registry=TYPED, text="x,y")

    assert run.results["j"]["result"] == ["x?a+x?b+x?c", "y?a+y?b+y?c"]
    assert run.results["u"]["result"] == ["X?A", "X?B", "X?C", "Y?A", "Y?B", "Y?C"]
    assert through_json(run).results == run.results


def test_an_empty_typed_in_list_is_a_table_of_no_rows_and_a_gather_of_it_is_empty():
    run = ran({"id": "u", "node": "upper", "values": {"text": []}, "many": ["text"]}, {"id": "j", "node": "join"},
              edges=(("u.result", "j.texts"),), registry=TYPED)

    assert run.results == {"u": {"result": []}, "j": {"result": ""}}


def test_two_lists_typed_into_one_node_are_refused_whether_it_runs_once_or_per_row():
    both = {"id": "p", "node": "pair", "values": {"a": ["1"], "b": ["2"]}, "many": ["a", "b"]}
    per_row = {"id": "q", "node": "ask-each", "values": {"question": ["a"], "doc": ["d"]}, "many": ["question", "doc"]}

    assert problems(both, registry=TYPED) == ["two_typed_in_lists"]
    assert problems(per_row, registry=TYPED) == ["two_typed_in_lists"]
    inner = embedded("two-lists", both)
    assert problems({"id": "e", "node": "two-lists"}, registry=registry_of(pair, inner)) == ["embedded_graph_broken"]


def test_no_documents_ask_no_questions_and_a_document_skipped_above_asks_none_of_its_own():
    nothing = ran({"id": "d", "node": "split"}, QUESTIONS, edges=(("d.result", "q.doc"),), registry=TYPED, text="")
    assert nothing.results["q"]["result"] == []

    skipped = graph({"id": "d", "node": "split"}, {"id": "y", "node": "decide"}, {**QUESTIONS, "node": "answer"},
                    edges=(("d.result", "y.text"), ("y.yes", "q.doc")), registry=TYPED).run(text="ok,no")
    assert [w.prompt for w in skipped.waiting] == ["ok: a", "ok: b", "ok: c"]


def test_a_list_typed_in_inside_an_embedded_graph_per_outer_row_is_a_child_table_under_each_outer_row_through_json():
    inner = embedded("asker", {"id": "q", "node": "ask-each", "values": {"question": ["a", "b"]}, "many": ["question"]}, {"id": "j", "node": "join"},
                     edges=(("q.result", "j.texts"),))
    run = ran({"id": "d", "node": "split"}, {"id": "e", "node": "asker"}, edges=(("d.result", "e.doc"),),
              registry=registry_of(split, join, ask_each, inner), text="x,y")

    assert run.results["e"]["result"] == ["x?a+x?b", "y?a+y?b"]
    assert through_json(run).results == run.results


def test_a_skip_above_an_embedded_typed_list_leaves_it_no_rows_and_its_nodes_complete_through_json():
    inner = embedded("asker", {"id": "q", "node": "ask-each", "values": {"question": ["a", "b"]}, "many": ["question"]}, {"id": "j", "node": "join"},
                     edges=(("q.result", "j.texts"),))
    run = ran({"id": "d", "node": "decide"}, {"id": "e", "node": "asker"}, edges=(("d.yes", "e.doc"),),
              registry=registry_of(decide, join, ask_each, inner), text="no")

    assert run.results["e"]["result"] is None and through_json(run).status == "complete"


def test_an_empty_table_above_a_typed_list_and_two_more_levels_ends_cleanly():
    run = ran({"id": "d", "node": "split"}, QUESTIONS, {"id": "s", "node": "split"}, {"id": "u", "node": "upper"},
              edges=(("d.result", "q.doc"), ("q.result", "s.text"), ("s.result", "u.text")), registry=TYPED, text="")

    assert run.results["u"]["result"] == []


def test_a_typed_in_value_is_received_once_and_a_typed_in_list_once_per_item():
    calls: list[str] = []

    @node("note")
    def note(text: str, tag: str) -> str:
        calls.append(f"{tag}:{text}")
        return text

    ran({"id": "n", "node": "note", "values": {"text": ["a", "b"], "tag": "t"}, "many": ["text"]}, registry=registry_of(note))
    assert calls == ["t:a", "t:b"]


def test_two_series_outputs_of_one_node_and_two_children_of_one_parent_do_not_line_up_in_a_per_row_node():
    @dataclass(frozen=True)
    class Two:
        a: Series[str]
        b: Series[str]

    @node("two")
    def two(text: str) -> Two:
        return Two([text], [text])

    reg = registry_of(two, split, pair)
    assert problems({"id": "t", "node": "two"}, {"id": "p", "node": "pair"}, edges=(("t.a", "p.a"), ("t.b", "p.b")), registry=reg) == ["rows_not_related"]
    assert problems({"id": "s", "node": "split"}, {"id": "x", "node": "split"}, {"id": "y", "node": "split"}, {"id": "p", "node": "pair"},
                    edges=(("s.result", "x.text"), ("s.result", "y.text"), ("x.result", "p.a"), ("y.result", "p.b")), registry=reg) == ["rows_not_related"]


# -- an embedded graph is one node from outside ------------------------------------------------


def _streamed(run: Any) -> list[Any]:
    import asyncio

    async def collect() -> list[Any]:
        return [event async for event in run.stream()]

    return asyncio.run(collect())


@node("ask-ok")
def ask_ok(text: str) -> bool:
    raise Ask(f"OK {text}?")


@node("fails")
def fails(text: str) -> str:
    raise ValueError("inner trouble")


def test_events_waiting_ids_and_failures_name_an_embedding_never_its_inner_nodes():
    inner = embedded("checker", {"id": "u", "node": "upper"}, {"id": "a", "node": "ask-ok"}, edges=(("u.result", "a.text"),))
    loaded = graph({"id": "s", "node": "split"}, {"id": "e", "node": "checker"}, edges=(("s.result", "e.text"),),
                   registry=registry_of(split, upper, ask_ok, inner))

    events = _streamed(loaded.with_inputs(text="x,y"))
    run = events[-1].run

    named = {e.node_id for e in events if hasattr(e, "node_id")} | {e.unit.node_id for e in events if e.type == "unit_waiting"}
    assert named == {"s", "e"} and [e.node_id for e in events if e.type == "node_start"] == ["s", "e"]
    assert [(w.node_id, w.row) for w in run.waiting] == [("e", (0,)), ("e", (1,))] and all(w.id.startswith("e@") for w in run.waiting)
    done = run.answer({w.id: True for w in run.waiting}).with_inputs()
    finished = _streamed(done)
    (complete,) = [e for e in finished if e.type == "node_complete" and e.node_id == "e"]
    assert complete.rows == 2 and finished[-1].run.results["e"] == {"result": [True, True]}
    assert "/" not in json.dumps([e.model_dump(mode="json") for e in finished])

    broken = embedded("breaker", {"id": "f", "node": "fails"})
    failed = graph({"id": "b", "node": "breaker"}, registry=registry_of(fails, broken)).run(text="x")
    assert (failed.cause.type, failed.cause.loc) == ("execution_failed", ("b",))


def test_decisions_of_an_embedding_name_the_outputs_it_offers():
    inner = embedded("decider", {"id": "d", "node": "decide"})
    loaded = graph({"id": "e", "node": "decider"}, registry=registry_of(decide, inner))

    assert loaded.decisions == {"e": {"answer": ["yes", "no"]}}


def test_an_embedding_fed_partly_from_outside_still_offers_its_other_inputs():
    inner = embedded("pairer", {"id": "p", "node": "pair"})
    loaded = graph({"id": "u", "node": "upper"}, {"id": "e", "node": "pairer"}, edges=(("u.result", "e.a"),), registry=registry_of(upper, pair, inner))

    assert loaded.problems == [] and list(loaded.inputs) == ["text", "b"]
    assert loaded.run(text="x", b="y").results["e"] == {"result": "X:y"}


def test_a_nested_embedding_names_its_fields_as_each_record_says_them_never_by_a_spliced_id():
    two = embedded("two", {"id": "x", "node": "upper"}, {"id": "y", "node": "upper"})
    outer = embedded("outer", {"id": "f", "node": "two"}, {"id": "z", "node": "upper"})
    loaded = graph({"id": "o", "node": "outer"}, registry=registry_of(upper, two, outer))

    assert list(loaded.inputs) == ["x.text", "y.text", "text"]
    assert all("/" not in i.address for i in loaded.inputs.values())
    assert loaded.run({"x.text": "a", "y.text": "b", "text": "c"}).results["o"] == {"x.result": "A", "y.result": "B", "result": "C"}


def test_many_on_an_embedded_node_takes_a_list_as_its_inner_input_would():
    inner = embedded("shouter", {"id": "u", "node": "upper"})
    loaded = graph({"id": "e", "node": "shouter", "many": ["text"]}, registry=registry_of(upper, inner))

    assert loaded.inputs["text"].many and loaded.run(text=["a", "b"]).results["e"] == {"result": ["A", "B"]}
    assert problems({"id": "e", "node": "shouter", "many": ["nope"]}, registry=registry_of(upper, inner)) == ["unknown_input"]


# -- tables typed in or given ------------------------------------------------------------------


class Dated(TypedDict):
    name: str
    when: datetime.date


class Comma(float):
    """A number written with a decimal comma, as a host's own type reads it."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        from pydantic_core import core_schema

        return core_schema.no_info_before_validator_function(lambda v: float(v.replace(",", ".")) if isinstance(v, str) else v, core_schema.float_schema())


class Priced(TypedDict):
    price: Comma


@node("describe-dated")
def describe_dated(rows: Table[Dated]) -> str:
    return f"{rows.columns}:{rows['when'].to_list()}"


@node("total-price")
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


def test_a_typed_in_frame_on_a_table_of_any_columns_arrives_with_the_columns_it_holds_and_an_edge_from_one_loads():
    run = ran({"id": "s", "node": "show", "values": {"table": [{"email": "a@b.dk", "n": 1}]}}, {"id": "u", "node": "upper"},
              edges=(("s.result.email", "u.text"),), registry=DATED)

    assert run.results["s"]["result"].columns == ["email", "n"] and run.results["u"]["result"] == ["A@B.DK"]


def test_each_input_in_describe_says_how_it_receives_on_each_row_its_node_runs_on():
    loaded = graph({"id": "s", "node": "split"}, {"id": "u", "node": "upper"}, {"id": "j", "node": "join"}, {"id": "p", "node": "pair", "values": {"b": "x"}},
                   {"id": "k", "node": "pair"},
                   edges=(("s.result", "u.text"), ("u.result", "j.texts"), ("u.result", "p.a"), ("j.result", "k.a"), ("u.result", "k.b")))
    nodes = loaded.describe()["nodes"]

    def receives(node_id: str) -> dict[str, str]:
        return {i["name"]: i["receives"] for i in nodes[node_id]["inputs"]}

    assert receives("u") == {"text": "per row of s.result"}
    assert receives("j") == {"texts": "whole column of s.result"}
    assert receives("p") == {"a": "per row of s.result", "b": "one"}
    assert receives("k") == {"a": "copied down from the graph's own row", "b": "per row of s.result"}
    assert receives("s") == {"text": "one", "sep": "one"}


def test_alternatives_on_rows_of_different_tables_into_one_value_are_a_problem_never_a_value_lost():
    """``d.yes`` split into words and ``d.no`` as it is: a document that went ``no`` has no word rows to put its value on."""
    @node("words-of")
    def words_of(text: str) -> Series[str]:
        return text.split("+")

    assert problems({"id": "s", "node": "split"}, {"id": "d", "node": "decide"}, {"id": "w", "node": "words-of"}, {"id": "m", "node": "upper"},
                    edges=(("s.result", "d.text"), ("d.yes", "w.text"), ("w.result", "m.text"), ("d.no", "m.text")),
                    registry=registry_of(split, decide, words_of, upper)) == ["alternatives_on_different_rows"]


def test_a_node_filling_two_outputs_of_one_exclusive_group_fails_as_invalid_output():
    @node("both")
    def both(text: str) -> YesNo:
        return YesNo(text, text)

    run = graph({"id": "b", "node": "both"}, registry=registry_of(both)).run(text="a")

    assert (run.status, run.cause.type, run.cause.loc, run.cause.ctx["name"]) == ("failed", "invalid_output", ("b",), "no")


def test_the_row_ceiling_counts_a_typed_list_s_rows_under_every_row_it_goes_under_before_any_call():
    calls: list[str] = []

    @node("ask-each-counted")
    def ask_each_counted(doc: str, question: str) -> str:
        calls.append(question)
        return question

    loaded = graph({"id": "d", "node": "split"}, {"id": "q", "node": "ask-each-counted", "values": {"question": ["a", "b", "c"]}, "many": ["question"]},
                   edges=(("d.result", "q.doc"),), registry=registry_of(split, ask_each_counted))

    stopped = loaded.run(text="x,y,z", max_rows=4)

    assert (stopped.status, stopped.cause.ctx["table"], stopped.cause.ctx["rows"]) == ("stopped", "q.question", 9) and calls == []
    assert stopped.resume(max_rows=9).status == "complete" and len(calls) == 9


def test_a_sibling_gather_says_it_gathers_under_the_row_the_two_share():
    @dataclass(frozen=True)
    class Parts:
        invoices: Series[str]
        keywords: Series[str]

    @node("parts")
    def parts(text: str) -> Parts:
        return Parts([text], [text])

    @node("beside")
    def beside(invoice: str, keywords: Series[str]) -> str:
        return invoice

    loaded = graph({"id": "s", "node": "split"}, {"id": "p", "node": "parts"}, {"id": "b", "node": "beside"},
                   edges=(("s.result", "p.text"), ("p.invoices", "b.invoice"), ("p.keywords", "b.keywords")), registry=registry_of(split, parts, beside))

    assert {i["name"]: i["receives"] for i in loaded.describe()["nodes"]["b"]["inputs"]} == {
        "invoice": "per row of p.invoices", "keywords": "whole column of p.keywords under each row of s.result",
    }


def test_a_gather_over_rows_a_blank_kept_from_being_made_is_not_called_and_is_not_zero_rows():
    """A document that couldn't be read is not an empty document: its count is not made, and ``not_called`` says which input."""

    @node("docs")
    def docs(text: str) -> Series[str | None]:
        return [None if x == "-" else x for x in text.split(";")]

    @node("lines")
    def lines(text: str) -> Series[str]:
        return [x for x in text.split(",") if x]

    @node("count")
    def count(items: Series[str]) -> int:
        return len(items)

    @node("width")
    def width(table: Table) -> int:
        return table.width

    @node("as_table")
    def as_table(text: str) -> Table:
        return pl.DataFrame({"part": text.split(",")})

    @node("total")
    def total(ns: Series[int]) -> int:
        return sum(ns)

    graph = registry_of(docs, lines, count, width, as_table, total).load(GraphRecord(
        nodes=[NodeRecord(id=n, node=d) for n, d in [("d", "docs"), ("l", "lines"), ("c", "count"), ("t", "total"), ("a", "as_table"), ("w", "width")]],
        edges=[("d.result", "l.text"), ("l.result", "c.items"), ("c.result", "t.ns"), ("d.result", "a.text"), ("a.result", "w.table")],
    ))
    run = graph.run(text="a,b;-;c;")

    assert run.results["c"]["result"] == [2, None, 1, 0]  # the last document is empty: 0 lines, made
    assert run.results["w"]["result"] == [1, None, 1, 1]
    assert run.results["t"]["result"] == 3
    assert {(n["node_id"], n["row"], n["input"]) for n in run.not_called} >= {("c", (1,), "items"), ("w", (1,), "table")}
