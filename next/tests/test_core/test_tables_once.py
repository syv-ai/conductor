"""A dumped ``RunState`` holds each table once: under ``tables``, by digest, with a reference in every place that held it."""

import json
from typing import Any

import polars as pl
from conductor import (
    AskPerson,
    CompiledGraph,
    From,
    Graph,
    GraphNode,
    NodeRegistry,
    RunState,
    Series,
    Static,
    Table,
    node,
)
from conductor._internal.execution.run_state import OutputColumn, WorkOfNode


def _frame(rows: int) -> pl.DataFrame:
    return pl.DataFrame({"name": [f"name-{i}" for i in range(rows)], "score": [i * 1.5 for i in range(rows)]})


SOURCE: dict[str, pl.DataFrame] = {}


@node("load", category="t")
def load(key: str) -> Table:
    return SOURCE[key]


@node("keep", category="t")
def keep(table: Table) -> Table:
    return table


@node("names", category="t")
def names(count: int) -> Series[str]:
    return [f"n{i}" for i in range(count)]


@node("approve", category="t")
def approve(name: str, table: Table) -> str | AskPerson:
    return AskPerson(prompt="ok?")


@node("ask_with_default", category="t")
def ask_with_default(key: str) -> Table | AskPerson:
    return AskPerson(defaults={"result": SOURCE[key]})


REGISTRY = NodeRegistry([load, keep, names, approve, ask_with_default])


def _graph(*nodes: GraphNode) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(Graph(nodes=list(nodes)), REGISTRY)
    assert compiled.is_runnable, compiled.problems
    return compiled


def _node(node_id: str, node_type: str, **bindings: Any) -> GraphNode:
    return GraphNode(id=node_id, type=node_type, version=1, bindings=bindings)


def _chain_and_waiting_units(rows: int, units: int) -> tuple[CompiledGraph, RunState]:
    SOURCE["t"] = _frame(rows)
    compiled = _graph(
        _node("s", "load", key=Static("t")),
        *[_node(f"p{i}", "keep", table=From("s.result" if i == 0 else f"p{i - 1}.result")) for i in range(5)],
        _node("g", "names", count=Static(units)),
        _node("q", "approve", name=From("g.result"), table=From("p4.result")),
    )
    return compiled, compiled.run_sync()


def test_two_frames_with_equal_contents_share_an_entry_and_a_different_one_does_not() -> None:
    state = _state_holding([_frame(4), _frame(4), _frame(5)])
    assert len(json.loads(state.model_dump_json())["tables"]) == 2


def _state_holding(value: Any) -> RunState:
    return RunState(status="complete", nodes={"v": WorkOfNode(outputs={"result": OutputColumn(rows=[()], values=[value])})})


def test_a_table_is_found_inside_a_list_or_a_record_in_a_slot_typed_any() -> None:
    frame = _frame(2)
    state = _state_holding({"nested": ["label", 1, frame, {"deeper": frame}]})
    document = state.model_dump_json()
    parsed = json.loads(document)
    assert len(parsed["tables"]) == 1 and '"columns"' not in json.dumps(parsed["nodes"])
    (value,) = RunState.model_validate_json(document).nodes["v"].outputs["result"].values
    assert value["nested"][2] is value["nested"][3]["deeper"] and value["nested"][2].equals(frame)


def test_a_value_shaped_like_a_reference_to_no_table_stays_a_value() -> None:
    state = _state_holding({"table": "not-a-digest-of-anything"})
    loaded = RunState.model_validate_json(state.model_dump_json())
    assert loaded.nodes["v"].outputs["result"].values == [{"table": "not-a-digest-of-anything"}]


def test_the_published_schema_has_the_tables_of_a_document() -> None:
    assert "tables" in RunState.model_json_schema()["properties"]
    assert "tables" in RunState.model_json_schema(mode="serialization")["properties"]
