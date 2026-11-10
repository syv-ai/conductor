"""How a run holds a field's values: compile says it once, and a state read back holds the same."""

from typing import Annotated

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    CompiledGraph,
    InvalidCallError,
    Result,
    RunState,
    Series,
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
from conductor._internal.graph.binding import From, Static
from helpers import Placed, placed, record, registry_of

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


REGISTRY = registry_of(contacts, names, shout)


def _compiled(*graph_nodes: Placed) -> CompiledGraph:
    return CompiledGraph.from_graph(record(*list(graph_nodes)), REGISTRY)


def _held_as(compiled: CompiledGraph, address: str) -> object:
    return compiled.field(address)._held_as


def test_compile_says_once_how_each_field_is_held():
    compiled = _compiled(
        placed("t", "contacts", text=Static("a,b")),
        placed("row", "shout", text=From("t.result.name")),
        placed("s", "names", text=Static("a,b")),
        placed("one", "shout", text=Static("a")),
        placed("list", "shout", text=Static(["a", "b"])),
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


@pytest.mark.parametrize("maker", ["contacts", "names"])
def test_a_skip_at_the_top_leaves_a_node_on_its_rows_absent_from_a_state_read_back(maker):
    """Whether the skip is read live or off a state's columns, what a reader of those rows gets is the same."""
    reader_edge = "m.result.name" if maker == "contacts" else "m.result"
    compiled = _compiled(
        placed("m", maker, text=Static("skip")),
        placed("r", "shout", text=From(reader_edge)),
    )
    state = compiled.run_sync()

    assert compiled.results(state) == {}
    assert compiled.results(RunState.model_validate_json(state.model_dump_json())) == {}


def test_a_null_stored_where_a_table_makes_rows_is_refused_when_read_for_a_result():
    compiled = _compiled(placed("t", "contacts", text=Static("a,b")))
    data = compiled.run_sync().model_dump(mode="json")
    data["nodes"]["t"]["outputs"]["result"]["values"][0] = None
    corrupted = RunState.model_validate(data)

    with pytest.raises(InvalidCallError) as raised:
        compiled.results(corrupted)

    assert {problem.type for problem in raised.value.errors()} == {"state_unreadable"}
