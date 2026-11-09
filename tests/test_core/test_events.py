"""The events are records.

Every event a stream yields is a frozen model, one of a union
discriminated on ``type``: a caller can ``match`` on the class. ``RunEnd``
carries the run's state; the values in it are read through the compiled
graph, ``compiled.results(state)``. The JSON a host
sends is a plain dict, and the schema a host generates its
client from names the same fields, with ``type`` required on every event.
"""

import asyncio
import json
from typing import Annotated

import pytest
from conductor import AskPerson, GraphNode, NodeRegistry, Param, node
from conductor.dtype import DType
from conductor.execution.events import (
    Event,
    NodeComplete,
    NodeProgress,
    NodeStart,
    RunEnd,
)
from conductor.execution.run_state import RunState
from conductor.graph.binding import Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph
from conductor.metadata import Result
from conductor.series import Index, Series
from conductor.widgets import Textarea
from pydantic import TypeAdapter, ValidationError


class Txt(DType, str):
    id = "events-test-text"
    title = "Text"


@node("split", title="Split", description="d", category="test")
def split_node(text: Annotated[Txt, Param(title="In", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Parts")]:
    return [Txt(part) for part in text.split(",")]


@node("ask", title="Ask", description="d", category="test")
def ask(text: Annotated[Txt, Param(title="In", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Answer")] | AskPerson:
    return AskPerson()


def _compiled(text: str) -> CompiledGraph:
    registry = NodeRegistry([split_node])
    return CompiledGraph.from_graph(
        Graph(nodes=(GraphNode(id="s", type="split", version=1, bindings={"text": Static(text)}),)), registry
    )


#: Every event's fields and required fields, by its ``type``.
SHAPES = {
    "node_complete": (["node_id", "result", "rows_not_called", "type"], ["node_id", "result", "rows_not_called", "type"]),
    "node_failed": (["cause", "node_id", "type"], ["cause", "node_id", "type"]),
    "node_progress": (["done", "node_id", "total", "type"], ["done", "node_id", "total", "type"]),
    "node_retry": (
        ["attempt", "delay", "error", "node_id", "retries", "row", "type"],
        ["attempt", "delay", "error", "node_id", "retries", "row", "type"],
    ),
    "node_skipped": (["node_id", "type"], ["node_id", "type"]),
    "node_start": (["node_id", "type"], ["node_id", "type"]),
    "run_checkpoint": (["state", "type"], ["state", "type"]),
    "run_end": (["state", "type"], ["state", "type"]),
    "run_start": (["changed_nodes", "reused_nodes", "type", "waiting"], ["changed_nodes", "reused_nodes", "type", "waiting"]),
    "unit_waiting": (["type", "unit"], ["type", "unit"]),
}


def test_every_event_names_its_fields_and_requires_its_type_in_the_schema():
    """A generated client sees every field, and ``type`` is never optional."""
    definitions = TypeAdapter(Event).json_schema(mode="serialization")["$defs"]
    shapes = {
        schema["properties"]["type"]["const"]: (sorted(schema["properties"]), sorted(schema.get("required", [])))
        for name, schema in definitions.items()
        if "type" in schema.get("properties", {}) and name != "Problem"
    }

    assert shapes == SHAPES


def test_the_last_event_is_matched_by_class():
    compiled = _compiled("a,b")

    async def last():
        async with compiled.stream() as events:
            return [event async for event in events][-1]

    match asyncio.run(last()):
        case RunEnd(state=state):
            assert list(compiled.results(state)["s"]["result"]) == ["a", "b"]
        case other:
            pytest.fail(f"the leg ended on {other}")


def test_the_values_are_read_from_the_state_live_or_stored():
    """An ending carries the run's state; a stored dump reads back to the same values."""
    compiled = _compiled("a,b")
    state = compiled.run_sync()

    stored = RunState.model_validate(json.loads(state.model_dump_json()))

    assert compiled.results(stored) == compiled.results(state)
    assert list(compiled.results(stored)["s"]["result"]) == ["a", "b"]


def test_an_event_is_frozen_and_refuses_a_key_it_does_not_have():
    event = NodeStart(type="node_start", node_id="a")

    with pytest.raises(ValidationError):
        event.node_id = "b"
    with pytest.raises(ValidationError):
        NodeStart(type="node_start", node_id="a", row=None)


def test_the_union_reads_an_event_back_by_its_type():
    progress = TypeAdapter(Event).validate_python({"type": "node_progress", "node_id": "a", "done": 1, "total": None})

    assert progress == NodeProgress(type="node_progress", node_id="a", done=1, total=None)


def test_a_frame_is_the_json_the_dict_was():
    """The provider's frame of a model is the dict's frame; a series keeps its index and rows."""
    from conductor_providers.fastapi.sse import sse_frame

    event = NodeComplete(
        type="node_complete", node_id="s", result={"result": Series[Txt](Index("s"), [Txt("a")], rows=[(0,)])}
    )

    assert json.loads(sse_frame(event).removeprefix("data: ")) == {
        "type": "node_complete",
        "rows_not_called": {},
        "node_id": "s",
        "result": {"result": {"index": {"id": "s", "parent": None}, "rows": [[0]], "values": ["a"]}},
    }


def test_a_pending_unit_is_a_record_too():
    registry = NodeRegistry([ask])
    compiled = CompiledGraph.from_graph(
        Graph(nodes=(GraphNode(id="q", type="ask", version=1, bindings={"text": Static("x")}),)), registry
    )

    ending = compiled.run_sync()

    assert ending.status == "waiting"
    (unit,) = ending.waiting
    assert (unit.node_id, unit.row) == ("q", ())
    assert unit.questions[0].name == "result"
