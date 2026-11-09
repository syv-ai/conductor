"""What a node inside a graph run per row receives, and what follows from it.

The rule the tests hold compile to: whatever is received under a row is cut at
that row, as far as the row and the source share one. A graph embedded under a
row behaves as the same nodes drawn where it stands.
"""

import json
from dataclasses import dataclass
from typing import Annotated, Any, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    AsItArrives,
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    InvalidCallError,
    NodeRegistry,
    Param,
    Result,
    RunState,
    Series,
    Static,
    Table,
    node,
)
from conductor.widgets import Textarea

from tests.test_core.embedded import embedded_graph_node


class Txt(DType, str):
    id = "embedded-receiving-text"
    display_title = "Text"


class Person(TypedDict):
    name: Txt


Text = Annotated[Txt, Param(title="Text", widget=Textarea())]
Out = Annotated[Txt, Result(title="Result")]
Many = Annotated[Series[Txt], Result(title="Many")]
Texts = Annotated[Series[Txt], Param(title="Texts")]
People = Annotated[Table[Person], Result(title="People")]


@node("src", title="s", description="d", category="t")
def src(text: Text = Txt("")) -> Out:
    return text


@node("upper", title="s", description="d", category="t")
def upper(text: Text = Txt("")) -> Out:
    return Txt(text.upper())


@node("split", title="s", description="d", category="t")
def split(text: Text = Txt("")) -> Many:
    """One value per letter; ``x`` gives no rows."""
    return [Txt(f"{text}.{letter}") for letter in text if letter != "x"]


@node("join", title="s", description="d", category="t")
def join(items: Texts) -> Out:
    return Txt("+".join(items))


@node("joined_at_row", title="s", description="d", category="t")
def joined_at_row(items: Texts, at: Text = Txt("")) -> Out:
    return Txt(f"{at}:" + "+".join(items))


@node("pair", title="s", description="d", category="t")
def pair(a: Texts, b: Texts) -> Out:
    return Txt(f"{list(a)}|{list(b)}")


@node("two", title="s", description="d", category="t")
def two(a: Text, b: Text) -> Out:
    return Txt(f"{a}+{b}")


@node("whatever_arrives", title="s", description="d", category="t")
def whatever_arrives(**thing: AsItArrives) -> Out:
    return Txt(",".join(thing["thing"]))


@node("whatever_arrives_at_row", title="s", description="d", category="t")
def whatever_arrives_at_row(at: Text, **thing: AsItArrives) -> Out:
    return Txt(f"{at}:" + ",".join(thing["thing"]))


@dataclass(frozen=True)
class YesNo:
    yes: Annotated[Txt, Result(title="Y", exclusive_group="c")]
    no: Annotated[Txt, Result(title="N", exclusive_group="c")]


@node("always_yes", title="s", description="d", category="t")
def always_yes(text: Text = Txt("")) -> YesNo:
    return YesNo(yes=text, no=SKIPPED)


@node("people", title="s", description="d", category="t")
def people(text: Text = Txt("")) -> People:
    return pl.DataFrame({"name": [f"{text}-person"]})


@node("merged_columns", title="s", description="d", category="t")
def merged_columns(table: Annotated[Table, Param(title="Table")]) -> Out:
    return Txt(",".join(column for column in table.columns if not column.startswith("__")))


@node("merged_columns_at_row", title="s", description="d", category="t")
def merged_columns_at_row(table: Annotated[Table, Param(title="Table")], at: Text) -> Out:
    return Txt(f"{at}:" + ",".join(column for column in table.columns if not column.startswith("__")))


REGISTRY = NodeRegistry([merged_columns_at_row, src, upper, split, join, joined_at_row, pair, two, whatever_arrives, whatever_arrives_at_row, always_yes, people, merged_columns])


def n(node_id: str, node_type: str, **bindings: Any) -> GraphNode:
    return GraphNode(id=node_id, type=node_type, version=1, bindings=bindings)


def embedded_per_row(inner: list[GraphNode], values: list[str], registry: NodeRegistry = REGISTRY, **bindings: Any) -> CompiledGraph:
    """``inner`` as a graph node run once per row of a list of ``values``."""
    definition = embedded_graph_node("g", tuple(inner), registry)
    outer = [n("s", "src", text=Static(values)), n("e", "g", **(bindings or {"text": From("s.result")}))]
    return CompiledGraph.from_graph(Graph(nodes=outer), registry.extended_with([definition]))


def by_row(values: dict[str, list[Any]]) -> dict[tuple[int, ...], Any]:
    return {tuple(row): value for row, value in zip(values["rows"], values["values"], strict=True)}


def receive_of(compiled: CompiledGraph, node_id: str, input_name: str):
    return compiled.run_plan.nodes[node_id].inputs[input_name]


# -- the rows a node under a row reads ---------------------------------------


def test_run_plan_says_how_each_gathered_source_is_read():
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        n("s", "src", text=Static(["ab", "c"])),
        n("a", "split", text=From("s.result")),
        n("o", "split", text=Static("zy")),
        n("j", "joined_at_row", items=From("a.result", "s.result", "o.result"), at=From("s.result")),
    ]), REGISTRY)
    gather = receive_of(compiled, "j", "items")
    assert [(read.receive_mode, read.rows_of, read.depth) for read in gather.source_reads] == [
        ("all_rows_under_row", "a.result", 1), ("all_rows_under_row", "s.text", 1), ("entire_value", None, 0),
    ]
    assert str(gather) == "collected into one list: all rows of a.result under my row (grouped under s.text) + the value at my row (s.result) + all of o.result"


def test_a_merge_cannot_hold_a_per_row_table_on_rows_the_node_does_not_sit_on():
    """A table arriving per row puts its node on that row, so a merge only ever reads a table its row sits on; unrelated rows are refused, in a graph as well as flat."""
    tables_and_rows = [
        n("x", "split", text=Static("ab")), n("p", "people", text=From("x.result")),
        n("y", "split", text=Static("cd")), n("o", "split", text=Static("ef")),
    ]
    flat = CompiledGraph.from_graph(Graph(nodes=[*tables_and_rows, n("m", "merged_columns_at_row", table=From("p.result", "o.result"), at=From("y.result"))]), REGISTRY)
    definition = embedded_graph_node("g", (n("u", "upper"), n("m", "merged_columns")), REGISTRY)
    embedded = CompiledGraph.from_graph(
        Graph(nodes=[*tables_and_rows, n("e", "g", text=From("y.result"), table=From("p.result", "o.result"))]), REGISTRY.extended_with([definition]),
    )

    assert [problem.type for problem in flat.problems] == ["misaligned"]
    assert [problem.type for problem in embedded.problems] == ["misaligned"]


# -- embedded under a row is the same nodes drawn flat ------------------------


# -- a skipped source ----------------------------------------------------------


# -- a refused graph inside a graph ---------------------------------------------


def test_a_nested_graph_refused_gives_problems_and_never_raises():
    inner = embedded_graph_node("in", (n("i", "two"),), REGISTRY)
    with_inner = REGISTRY.extended_with([inner])
    middle = embedded_graph_node("mid", (n("b0", "in"), n("after", "upper", text=From("b0.result"))), with_inner)
    compiled = CompiledGraph.from_graph(Graph(nodes=[n("o", "mid", a=Static(["1", "2"]), b=Static(["3", "4"]))]), with_inner.extended_with([middle]))
    assert not compiled.is_runnable
    assert "lists_on_a_per_row_node" in {problem.type for problem in compiled.problems if problem.fatal}


# -- resume ----------------------------------------------------------------------


def _static_only_inside(values: list[str]) -> CompiledGraph:
    inner = [n("a", "upper"), n("b", "upper", text=Static("fixed")), n("c", "two", a=From("a.result"), b=From("b.result"))]
    return embedded_per_row(inner, values, **{"a.text": From("s.result")})


# -- a merge's renamed column ----------------------------------------------------


def _merge_inside() -> list[GraphNode]:
    return [n("m", "people"), n("i1", "upper", text=From("m.result.name")), n("t", "merged_columns", table=From("m.result", "i1.result"))]


# -- a state changed by hand -----------------------------------------------------


def _finished_table_state() -> tuple[CompiledGraph, dict[str, Any]]:
    compiled = CompiledGraph.from_graph(Graph(nodes=[
        n("m", "split", text=Static("ab")), n("t", "people", text=From("m.result")), n("r", "upper", text=From("t.result.name")),
    ]), REGISTRY)
    return compiled, json.loads(compiled.run_sync().model_dump_json())


def _refusal_of(compiled: CompiledGraph, data: dict[str, Any]) -> set[str]:
    with pytest.raises(InvalidCallError) as refused:
        compiled.run_sync(RunState.model_validate(data))
    return {problem.type for problem in refused.value.errors()}


