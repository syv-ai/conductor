"""Rows and tables through the public door: blanks, copy-down, gathers, branches, tables and embedded graphs.

Every value sits on a row of a table. A node fed a column runs once per row of
it; a value from a row above is copied down; a ``Series`` or ``Table`` input
gathers the rows under the node's row. Sources must lie on one path of the row
tree. A blank cell keeps a node from being called on its row unless the input
says ``| None``; a branch not taken skips the row.
"""
from dataclasses import dataclass
from typing import Annotated, Any, Optional, TypedDict

import polars as pl
import pytest
from conductor import SKIPPED, GraphRecord, Param, Series, Table, node

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
    assert complete.not_called == [{"row": [], "input": "email"}]
    assert said[-1].status == "complete"


def test_choice_outputs_are_skipped_when_not_called():
    @node("branched")
    def branched(email: str) -> YesNo:
        return YesNo(email, SKIPPED)

    run = ran({"id": "c", "node": "contacts"}, {"id": "b", "node": "branched"}, {"id": "u", "node": "upper"},
              edges=(("c.result.email", "b.email"), ("b.yes", "u.text")), registry=registry_of(contacts, branched, upper), text=CSV)
    assert run.results["b"] == {"yes": ["ida@x", None, "nina@x"], "no": [None, None, None]}
    assert run.results["u"]["result"] == ["IDA@X", None, "NINA@X"]
    assert [(n["node_id"], n["row"]) for n in run.not_called] == [("b", (1,)), ("u", (1,))]


def test_a_row_making_output_has_no_rows_when_not_called_so_a_count_gives_zero():
    run = ran({"id": "c", "node": "contacts"}, {"id": "s", "node": "split"}, {"id": "n", "node": "count"},
              edges=(("c.result.email", "s.text"), ("s.result", "n.values")), text=CSV)
    assert run.results["n"]["result"] == [1, 0, 1]


def test_a_table_output_has_no_rows_when_not_called():
    run = ran({"id": "c", "node": "contacts"}, {"id": "i", "node": "invoices"}, {"id": "n", "node": "count"},
              edges=(("c.result.email", "i.text"), ("i.result.vendor", "n.values")), text="Ida,a:1;Bo,")
    assert run.results["n"]["result"] == [1, 0]
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


def test_two_series_under_unrelated_rows_are_refused():
    assert problems(*THREE_LEVELS, {"id": "again", "node": "split", "values": {"sep": ";"}}, {"id": "lines2", "node": "split"},
                    {"id": "n", "node": "two-columns"},
                    edges=(*THREE_EDGES, ("docs.result", "again.text"), ("again.result", "lines2.text"), ("lines.result", "n.a"), ("lines2.result", "n.b")),
                    registry=DEEP) == ["rows_not_related"]


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
    assert run.graph.plan["emb/lines"] == run.graph.plan["emb/join"] == "per row of docs.result"
    assert run.results["emb/join"]["result"] == ["a+b", "c"]

    unfed = embedded("splitter-unfed", {"id": "entered", "node": "upper"}, {"id": "lines", "node": "split", "values": {"text": "x/y", "sep": "/"}},
                     {"id": "join", "node": "join"}, {"id": "p", "node": "pair"},
                     edges=(("lines.result", "join.texts"), ("entered.result", "p.a"), ("join.result", "p.b")))
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "splitter-unfed"}, edges=(("docs.result", "emb.text"),),
              registry=registry_of(split, join, upper, pair, unfed), text="a,b")
    assert run.results["emb/p"]["result"] == ["A:x+y", "B:x+y"]


def test_inputs_from_a_table_and_its_child_table_run_the_inner_nodes_on_the_deeper_rows():
    @node("shout")
    def shout(word: str) -> str:
        return word.upper()

    inner = embedded("pair-up", {"id": "ua", "node": "shout"}, {"id": "ub", "node": "upper"}, {"id": "p", "node": "pair"},
                     edges=(("ua.result", "p.a"), ("ub.result", "p.b")))
    run = ran({"id": "docs", "node": "split", "values": {"sep": "|"}}, {"id": "lines", "node": "split"}, {"id": "emb", "node": "pair-up"},
              edges=(("docs.result", "lines.text"), ("docs.result", "emb.word"), ("lines.result", "emb.text")),
              registry=registry_of(split, upper, pair, shout, inner), text="a,b|c")
    assert run.graph.plan["emb/ua"] == "per row of docs.result" and run.graph.plan["emb/p"] == "per row of lines.result"
    assert run.results["emb/p"]["result"] == ["A,B:A", "A,B:B", "C:C"]


def test_a_list_into_an_embedded_series_input_is_one_value_and_the_graph_runs_once():
    inner = embedded("joiner", {"id": "join", "node": "join"})
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "joiner"}, edges=(("docs.result", "emb.texts"),),
              registry=registry_of(split, join, inner), text="a,b")
    assert run.graph.plan["emb/join"] == "once" and run.results["emb/join"]["result"] == "a+b"


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
    assert run.graph.plan["e/j"] == "per row of s.result"
    assert run.results["e/j"]["result"] == ["a+b+A;B", "c+C"]


def test_a_series_input_of_an_embedding_fed_a_whole_column_sees_it_whole():
    inner = embedded("mixed", {"id": "u", "node": "upper"}, {"id": "j", "node": "join"})
    run = ran({"id": "docs", "node": "split"}, {"id": "emb", "node": "mixed"}, edges=(("docs.result", "emb.text"), ("docs.result", "emb.texts")),
              registry=registry_of(split, upper, join, inner), text="a,b")
    assert run.graph.plan == {"docs": "once", "emb/u": "per row of docs.result", "emb/j": "once"}
    assert run.results["emb/j"]["result"] == "a+b"


def test_a_graph_embedded_under_a_row_runs_as_the_same_nodes_drawn_flat():
    nodes = ({"id": "i", "node": "split", "values": {"sep": ";"}}, {"id": "c", "node": "split", "values": {"sep": "/"}}, {"id": "p", "node": "pair"})
    edges = (("i.result", "c.text"), ("c.result", "p.a"), ("i.result", "p.b"))
    inner = embedded("deep", *nodes, edges=edges)
    reg = registry_of(split, pair, inner)
    embedded_run = ran({"id": "s", "node": "split"}, {"id": "e", "node": "deep"}, edges=(("s.result", "e.text"),), registry=reg, text="a/b;c,d")
    flat_run = ran({"id": "s", "node": "split"}, *nodes, edges=(("s.result", "i.text"), *edges), registry=reg, text="a/b;c,d")
    for name in ("i", "c", "p"):
        assert embedded_run.results[f"e/{name}"] == flat_run.results[name]
        assert embedded_run.graph.plan[f"e/{name}"].replace("e/", "") == flat_run.graph.plan[name]


def test_a_table_inputs_columns_are_named_the_same_embedded_as_alone():
    @node("columns")
    def columns(table: Table) -> str:
        return ",".join(table.columns)

    merge = ({"id": "c", "node": "contacts", "values": {"text": CSV}}, {"id": "s", "node": "send-or-say"}, {"id": "t", "node": "columns"})
    edges = (("c.result.email", "s.email"), ("c.result.name", "t.table"), ("s.result", "t.table"))
    reg = registry_of(split, contacts, send_or_say, columns, embedded("sheet", *merge, edges=edges))
    alone = ran(*merge, edges=edges, registry=reg)
    inside = ran({"id": "e", "node": "sheet"}, registry=reg)
    assert alone.results["t"]["result"] == inside.results["e/t"]["result"] == "name,result"


def test_a_table_input_inside_an_embedded_graph_combines_its_inner_sources():
    merge = ({"id": "c", "node": "contacts", "values": {"text": CSV}}, {"id": "s", "node": "send-or-say"}, {"id": "x", "node": "show"})
    edges = (("c.result.email", "s.email"), ("c.result", "x.table"), ("s.result", "x.table"))
    run = ran({"id": "e", "node": "sheet"}, registry=registry_of(contacts, send_or_say, show, embedded("sheet", *merge, edges=edges)))
    assert run.results["e/x"]["result"]["result"].to_list() == ["sent to ida@x", "no address", "sent to nina@x"]


def test_inside_an_embedding_a_rebuilt_table_and_a_column_made_per_row_of_it_combine():
    @node("rebuild")
    def rebuild(table: Table) -> Table[Contact]:
        return table.filter(pl.col("email").is_not_null())

    inner = embedded("rb", {"id": "k", "node": "rebuild"}, {"id": "s", "node": "send"}, {"id": "m", "node": "show"},
                     edges=(("k.result.email", "s.email"), ("k.result", "m.table"), ("s.result", "m.table")))
    run = ran({"id": "c", "node": "contacts"}, {"id": "sub", "node": "rb"}, edges=(("c.result", "sub.table"),),
              registry=registry_of(contacts, rebuild, send, show, inner), text=CSV)
    assert run.results["sub/m"]["result"].to_dicts() == [
        {"name": "Ida", "email": "ida@x", "result": "sent to ida@x"}, {"name": "Nina", "email": "nina@x", "result": "sent to nina@x"},
    ]
