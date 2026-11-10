"""A dumped ``RunState`` holds each table once: under ``tables``, by digest, with a reference in every place that held it."""

import json
from typing import Any

import polars as pl
import pytest
from conductor import AskPerson, CompiledGraph, RunState, Series, Table, node
from conductor._internal.execution.run_state import OutputColumn, WorkOfNode
from conductor._internal.graph.binding import From, Static
from conductor.events import RunEnd, UnitWaiting
from helpers import Placed, answer, placed, record, registry_of
from pydantic import ValidationError


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


REGISTRY = registry_of(load, keep, names, approve, ask_with_default)


def _graph(*nodes: Placed) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(record(*list(nodes)), REGISTRY)
    assert compiled.is_runnable, compiled.problems
    return compiled


def _node(node_id: str, node_type: str, **bindings: Any) -> Placed:
    return placed(node_id, node_type, **bindings)


def _chain_and_waiting_units(rows: int, units: int) -> tuple[CompiledGraph, RunState]:
    SOURCE["t"] = _frame(rows)
    compiled = _graph(
        _node("s", "load", key=Static("t")),
        *[_node(f"p{i}", "keep", table=From("s.result" if i == 0 else f"p{i - 1}.result")) for i in range(5)],
        _node("g", "names", count=Static(units)),
        _node("q", "approve", name=From("g.result"), table=From("p4.result")),
    )
    return compiled, compiled.run_sync()


def test_a_chain_and_many_waiting_units_cost_one_table_and_read_back_equal() -> None:
    compiled, state = _chain_and_waiting_units(50_000, 500)
    assert state.status == "waiting" and len(state.waiting) == 500
    document = state.model_dump_json()
    (table,) = json.loads(document)["tables"].values()
    assert len(document) < 1.25 * len(json.dumps(table, separators=(",", ":")))  # the one table, and the rows' input hashes
    loaded = RunState.model_validate_json(document)
    for node_id, outputs in compiled.results(state).items():
        for name, value in outputs.items():
            read = compiled.results(loaded)[node_id][name]
            assert read.equals(value) if isinstance(value, pl.DataFrame) else read == value
    frames = {id(column.values[0]) for node_id, work in loaded.nodes.items() for name, column in work.outputs.items() if node_id != "g" and (node_id, name) != ("q", "result")}
    frames |= {id(unit.inputs["table"]) for unit in loaded.waiting}
    assert len(frames) == 1
    assert isinstance(loaded.nodes["s"].outputs["result"].values[0], pl.DataFrame)


def test_a_document_has_one_entry_per_table_and_references_everywhere() -> None:
    _, state = _chain_and_waiting_units(3, 2)
    document = json.loads(state.model_dump_json())
    assert len(document["tables"]) == 1
    (digest,) = document["tables"]
    assert document["tables"][digest]["data"]["name"] == ["name-0", "name-1", "name-2"]
    assert document["nodes"]["p0"]["outputs"]["result"]["values"] == [{"table": digest}]
    assert [unit["inputs"]["table"] for unit in document["nodes"]["q"]["waiting"]] == [{"table": digest}, {"table": digest}]


def test_a_plain_dump_stands_alone_and_a_python_dump_holds_frames() -> None:
    _, state = _chain_and_waiting_units(3, 1)
    assert RunState.model_validate_json(state.model_dump_json()).waiting[0].inputs["table"].height == 3
    assert RunState.model_validate(state.model_dump(mode="json")).waiting[0].inputs["table"].height == 3
    assert state.model_dump()["tables"] == {} and state.tables == {}
    assert isinstance(state.model_dump()["nodes"]["s"]["outputs"]["result"]["values"][0], pl.DataFrame)


def test_two_frames_with_equal_contents_share_an_entry_and_a_different_one_does_not() -> None:
    state = _state_holding([_frame(4), _frame(4), _frame(5)])
    assert len(json.loads(state.model_dump_json())["tables"]) == 2


def test_a_table_on_a_table_input_reaches_the_node_as_the_frame_the_state_holds() -> None:
    _, state = _chain_and_waiting_units(3, 2)
    source = state.nodes["s"].outputs["result"].values[0]
    assert state.nodes["p4"].outputs["result"].values[0] is source and state.waiting[0].inputs["table"] is source


def test_tables_of_different_nodes_read_back_each_as_itself() -> None:
    SOURCE["a"], SOURCE["b"] = _frame(4), _frame(5)
    compiled = _graph(_node("x", "load", key=Static("a")), _node("y", "load", key=Static("b")))
    state = compiled.run_sync()
    loaded = RunState.model_validate_json(state.model_dump_json())
    assert compiled.results(loaded)["x"]["result"].height == 4 and compiled.results(loaded)["y"]["result"].height == 5


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


def test_a_table_default_is_a_reference_and_answers_like_any_default() -> None:
    SOURCE["a"] = _frame(1)
    compiled = _graph(_node("ask", "ask_with_default", key=Static("a")))
    state = compiled.run_sync()
    assert isinstance(state.waiting[0].questions[0].default, pl.DataFrame)
    assert "DataFrame" in repr(state.waiting[0].questions[0]) or "shape" in repr(state.waiting[0].questions[0])
    document = json.loads(state.model_dump_json())
    assert set(document["nodes"]["ask"]["waiting"][0]["questions"][0]["default"]) == {"table"}
    loaded = RunState.model_validate_json(state.model_dump_json())
    assert compiled.results(answer(compiled, loaded, {"ask": {}}))["ask"]["result"].height == 1


def test_a_frame_is_hashed_once_however_many_places_hold_it(monkeypatch: pytest.MonkeyPatch) -> None:
    from conductor._internal import table

    SOURCE["t"] = _frame(5)
    compiled = _graph(_node("s", "load", key=Static("t")), _node("g", "names", count=Static(50)), _node("q", "approve", name=From("g.result"), table=From("s.result")))
    state = compiled.run_sync()  # the run's key hashes what it holds, once
    hashed: list[int] = []
    real = table.hashlib.sha256
    monkeypatch.setattr(table.hashlib, "sha256", lambda *data: hashed.append(1) or real(*data))
    state.model_dump_json()
    state.model_dump_json()
    assert len(hashed) == 1  # the first dump hashes its row ids; what it holds it keeps from the run's key, and the second reads what the frame kept


def test_a_loaded_state_dumps_the_same_digests_with_no_hashing(monkeypatch: pytest.MonkeyPatch) -> None:
    from conductor._internal import table

    _, state = _chain_and_waiting_units(3, 2)
    document = state.model_dump_json()
    loaded = RunState.model_validate_json(document)
    monkeypatch.setattr(table.hashlib, "sha256", lambda data: pytest.fail("a loaded frame keeps its digest"))
    assert json.loads(loaded.model_dump_json())["tables"].keys() == json.loads(document)["tables"].keys()


def test_a_value_shaped_like_a_reference_to_no_table_stays_a_value() -> None:
    state = _state_holding({"table": "not-a-digest-of-anything"})
    loaded = RunState.model_validate_json(state.model_dump_json())
    assert loaded.nodes["v"].outputs["result"].values == [{"table": "not-a-digest-of-anything"}]


def test_a_malformed_table_refuses_the_load() -> None:
    SOURCE["a"] = _frame(1)
    state = _graph(_node("s", "load", key=Static("a"))).run_sync()
    document = json.loads(state.model_dump_json())
    (digest,) = document["tables"]
    document["tables"][digest] = {"columns": "nope", "data": {}, "rows": None}
    with pytest.raises(ValidationError):
        RunState.model_validate(document)


def test_a_table_nothing_holds_any_more_is_not_written() -> None:
    SOURCE["a"], SOURCE["b"] = _frame(1), _frame(2)
    state = _graph(_node("x", "load", key=Static("a")), _node("y", "load", key=Static("b"))).run_sync()
    assert len(json.loads(state.without_nodes("x").model_dump_json())["tables"]) == 1


def test_an_event_stands_alone_and_a_state_has_its_own_tables() -> None:
    compiled, state = _chain_and_waiting_units(2, 1)
    (unit,) = state.waiting
    alone = UnitWaiting(unit=unit).model_dump(mode="json")
    assert alone["unit"]["inputs"]["table"]["data"]["name"] == ["name-0", "name-1"]
    assert RunEnd(state=state).model_dump(mode="json") == {"type": "run_end", "status": "waiting"}
    kept = state.model_dump(mode="json")
    assert len(kept["tables"]) == 1
    assert set(kept["nodes"]["q"]["waiting"][0]["inputs"]["table"]) == {"table"}


def test_the_published_schema_has_the_tables_of_a_document() -> None:
    assert "tables" in RunState.model_json_schema()["properties"]
    assert "tables" in RunState.model_json_schema(mode="serialization")["properties"]
