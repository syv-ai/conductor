"""How a run holds a field's values: compile says it once, and a state read back holds the same."""

from typing import Annotated

import polars as pl
from conductor import (
    SKIPPED,
    CompiledGraph,
    From,
    Graph,
    GraphNode,
    NodeRegistry,
    Result,
    Series,
    Static,
    node,
    table_with_columns,
)
from conductor._internal.execution.values_of_run import (
    CellsInAList,
    ColumnOfTable,
    ListOnNode,
    NotHeld,
    OneValue,
    TableAtRow,
)

from test_core.test_table import Txt

Contacts = table_with_columns({"name": Txt, "email": Txt})


@node("contacts", title="Contacts", description="d", category="test")
def contacts(text: Txt = Txt("")) -> Annotated[Contacts, Result(title="Rows")]:
    if text == "skip":
        return SKIPPED
    return pl.DataFrame({"name": text.split(","), "email": ["e"] * len(text.split(","))})


@node("names", title="Names", description="d", category="test")
def names(text: Txt = Txt("")) -> Annotated[Series[Txt], Result(title="Names")]:
    if text == "skip":
        return SKIPPED
    return [Txt(part) for part in text.split(",")]


@node("shout", title="Shout", description="d", category="test")
def shout(text: Txt = Txt("")) -> Annotated[Txt, Result(title="Shouted")]:
    return Txt(text.upper())


REGISTRY = NodeRegistry([contacts, names, shout])


def _compiled(*graph_nodes: GraphNode) -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=list(graph_nodes)), REGISTRY)


def _held_as(compiled: CompiledGraph, address: str) -> object:
    return compiled.field(address)._held_as


def test_compile_says_once_how_each_field_is_held():
    compiled = _compiled(
        GraphNode(id="t", type="contacts", version=1, bindings={"text": Static("a,b")}),
        GraphNode(id="row", type="shout", version=1, bindings={"text": From("t.result.name")}),
        GraphNode(id="s", type="names", version=1, bindings={"text": Static("a,b")}),
        GraphNode(id="one", type="shout", version=1, bindings={"text": Static("a")}),
        GraphNode(id="list", type="shout", version=1, bindings={"text": Static(["a", "b"])}),
    )

    assert isinstance(_held_as(compiled, "t.result"), TableAtRow)
    assert _held_as(compiled, "t.result.name") == ColumnOfTable(table="t.result", column="name")
    assert isinstance(_held_as(compiled, "s.result"), CellsInAList)
    assert isinstance(_held_as(compiled, "one.result"), OneValue)
    assert isinstance(_held_as(compiled, "row.result"), OneValue)
    assert isinstance(_held_as(compiled, "row.text"), NotHeld)
    held = _held_as(compiled, "list.text")
    assert isinstance(held, ListOnNode)
    assert held.cells == ["a", "b"]
    assert held.rows_made is not None and held.rows_made.id == "list.text"


