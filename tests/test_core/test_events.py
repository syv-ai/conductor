"""The events are records.

Every event ``execute`` yields is a frozen model, one of a union
discriminated on ``type``: a caller reads ``ending.state`` and can
``match`` on the class. An ending says why the leg stopped and carries the
run's state; the values in it are read through the compiled
graph, ``state.results(compiled)``. The JSON a host
sends is the dict it always was, and the schema a host generates its
client from names the same fields, with ``type`` required on every event.
"""

import json
from typing import Annotated, get_args

import pytest
from conductor import Asks, GraphNode, NodeRegistry, Param, run_sync
from conductor.dtype import DType
from conductor.execution.events import (
    Ending,
    EndingEvent,
    ExecutionEvent,
    GraphCompleteEvent,
    GraphPendingEvent,
    NodeCompleteEvent,
    NodeProgressEvent,
    NodeStartEvent,
)
from conductor.execution.state import RunState
from conductor.graph.binding import Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph
from conductor.metadata import Result
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.series import Index, Series
from conductor.widgets import Textarea
from pydantic import TypeAdapter, ValidationError


class Txt(DType, str):
    id = "events-test-text"
    title = "Text"


class Split(NodeDefinition):
    id = "split"
    title = "Split"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="In", widget=Textarea())] = Txt("")) -> Annotated[Series[Txt], Result(title="Parts")]:
        return [Txt(part) for part in text.split(",")]


class Ask(NodeDefinition):
    id = "ask"
    title = "Ask"
    description = "d"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="In", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="Answer")] | Asks:
        return Asks()


def _compiled(text: str) -> CompiledGraph:
    registry = NodeRegistry()
    registry.register(Split)
    return CompiledGraph.from_graph(
        Graph(nodes=(GraphNode(id="s", type="split", version=1, bindings={"text": Static(text)}),)), registry
    )


#: Every event's fields and required fields, by its ``type``.
SHAPES = {
    "graph_cancelled": (["state", "type"], ["state", "type"]),
    "graph_complete": (["state", "type"], ["state", "type"]),
    "graph_error": (["cause", "error", "node_id", "state", "type"], ["cause", "error", "node_id", "state", "type"]),
    "graph_pending": (["pending", "state", "type"], ["pending", "state", "type"]),
    "graph_timeout": (
        ["elapsed_seconds", "state", "timeout_seconds", "type"],
        ["elapsed_seconds", "state", "timeout_seconds", "type"],
    ),
    "node_complete": (["cached", "node_id", "result", "type"], ["node_id", "result", "type"]),
    "node_error": (["cause", "error", "node_id", "type"], ["cause", "error", "node_id", "type"]),
    "node_progress": (["done", "node_id", "total", "type"], ["done", "node_id", "total", "type"]),
    "node_retry": (
        ["attempt", "delay", "error", "node_id", "retries", "row", "type"],
        ["attempt", "delay", "error", "node_id", "retries", "row", "type"],
    ),
    "node_skipped": (["node_id", "type"], ["node_id", "type"]),
    "node_start": (["node_id", "type"], ["node_id", "type"]),
}


def test_every_event_names_its_fields_and_requires_its_type_in_the_schema():
    """A generated client sees every field, and ``type`` is never optional."""
    definitions = TypeAdapter(ExecutionEvent).json_schema(mode="serialization")["$defs"]
    shapes = {
        schema["properties"]["type"]["const"]: (sorted(schema["properties"]), sorted(schema.get("required", [])))
        for schema in definitions.values()
        if "type" in schema.get("properties", {})
    }

    assert shapes == SHAPES


def test_an_ending_is_read_by_attribute_and_matched_by_class():
    compiled = _compiled("a,b")
    ending = run_sync(compiled)

    match ending:
        case GraphCompleteEvent(state=state):
            assert list(state.results(compiled)["s"]["result"]) == ["a", "b"]
        case _:
            pytest.fail(f"the leg ended {ending.type}")


def test_the_values_are_read_from_the_state_live_or_stored():
    """An ending carries the run's state; a stored dump reads back to the same values."""
    compiled = _compiled("a,b")
    state = run_sync(compiled).state

    stored = RunState.model_validate(json.loads(state.model_dump_json()))

    assert stored.results(compiled) == state.results(compiled)
    assert list(stored.results(compiled)["s"]["result"]) == ["a", "b"]


def test_an_event_is_frozen_and_refuses_a_key_it_does_not_have():
    event = NodeStartEvent(type="node_start", node_id="a")

    with pytest.raises(ValidationError):
        event.node_id = "b"
    with pytest.raises(ValidationError):
        NodeStartEvent(type="node_start", node_id="a", row=None)


def test_the_union_reads_an_event_back_by_its_type():
    progress = TypeAdapter(ExecutionEvent).validate_python({"type": "node_progress", "node_id": "a", "done": 1, "total": None})

    assert progress == NodeProgressEvent(type="node_progress", node_id="a", done=1, total=None)


def test_a_frame_is_the_json_the_dict_was():
    """The provider's frame of a model is the dict's frame; a series keeps its index and rows."""
    from conductor_providers.fastapi.sse import sse_frame

    event = NodeCompleteEvent(
        type="node_complete", node_id="s", result={"result": Series[Txt](Index("s"), [Txt("a")], rows=[(0,)])}
    )

    assert json.loads(sse_frame(event).removeprefix("data: ")) == {
        "type": "node_complete",
        "node_id": "s",
        "result": {"result": {"index": {"id": "s", "parent": None}, "rows": [[0]], "values": ["a"]}},
        "cached": False,
    }


def test_a_pending_unit_is_a_record_too():
    registry = NodeRegistry()
    registry.register(Ask)
    compiled = CompiledGraph.from_graph(
        Graph(nodes=(GraphNode(id="q", type="ask", version=1, bindings={"text": Static("x")}),)), registry
    )

    ending = run_sync(compiled)

    assert isinstance(ending, GraphPendingEvent)
    (unit,) = ending.pending
    assert (unit.node_id, unit.row) == ("q", None)
    assert unit.questions[0].name == Ref("q", "result")


def test_a_leg_ends_on_an_ending_and_run_returns_it():
    """``EndingEvent`` is the events a leg stops on — every ``graph_*`` event
    in the union and nothing else — each an ``Ending`` carrying ``state``, and
    ``run`` returns one, so its ``state`` is read without narrowing first."""
    endings = {get_args(event.model_fields["type"].annotation)[0] for event in get_args(EndingEvent)}
    assert endings == {name for name in SHAPES if name.startswith("graph_")}
    assert all(issubclass(event, Ending) for event in get_args(EndingEvent))
    assert isinstance(run_sync(_compiled("a,b")), Ending)
    assert not isinstance(NodeStartEvent(type="node_start", node_id="s"), Ending)
