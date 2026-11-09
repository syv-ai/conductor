"""The events are records.

Every event a stream yields is a frozen model, one of a union
discriminated on ``type``: a caller can ``match`` on the class. ``RunEnd``
carries the run's state; the values in it are read through the compiled
graph, ``compiled.results(state)``. An event writes itself as JSON, each
value by its own type, and the schema a host generates its
client from names the same fields, with ``type`` required on every event.
"""

import asyncio
import json
from typing import Annotated

import polars as pl
import pytest
from conductor import (
    AskPerson,
    CompiledGraph,
    DType,
    Graph,
    GraphNode,
    NodeRegistry,
    Param,
    Problem,
    Ref,
    Result,
    RunState,
    Series,
    Static,
    node,
)
from conductor._internal.execution.run_state import OutputColumn, Question, WaitingUnit
from conductor.events import (
    Event,
    NodeComplete,
    NodeFailed,
    NodeProgress,
    NodeRetry,
    NodeSkipped,
    NodeStart,
    RunCheckpoint,
    RunEnd,
    RunStart,
    UnitWaiting,
)
from conductor.widgets import Textarea
from pydantic import TypeAdapter, ValidationError


class Txt(DType, str):
    id = "events-test-text"
    display_title = "Text"


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
    "node_complete": (["node_id", "result", "reused_units", "rows_not_called", "type"], ["node_id", "result", "reused_units", "rows_not_called", "type"]),
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
            assert compiled.results(state)["s"]["result"]["values"] == ["a", "b"]
        case other:
            pytest.fail(f"the leg ended on {other}")


def test_the_values_are_read_from_the_state_live_or_stored():
    """An ending carries the run's state; a stored dump reads back to the same values."""
    compiled = _compiled("a,b")
    state = compiled.run_sync()

    stored = RunState.model_validate(json.loads(state.model_dump_json()))

    assert compiled.results(stored) == compiled.results(state)
    assert compiled.results(stored)["s"]["result"]["values"] == ["a", "b"]


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
        type="node_complete", node_id="s", result={"result": {"rows": [(0,)], "values": [Txt("a")]}}
    )

    assert json.loads(sse_frame(event).removeprefix("data: ")) == {
        "type": "node_complete",
        "rows_not_called": {},
        "reused_units": 0,
        "node_id": "s",
        "result": {"result": {"rows": [[0]], "values": ["a"]}},
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


def _strict(document: str | bytes) -> object:
    def refuse(token: str) -> None:
        raise AssertionError(f"{token} is not JSON")

    return json.loads(document, parse_constant=refuse)


def _holding_many_values_a_frame_and_a_nan() -> dict[str, object]:
    return {
        "many": {"rows": [(0,), (1,)], "values": [float("nan"), 1.5]},
        "frame": pl.DataFrame({"a": [float("nan"), 1.0], "b": ["x", None]}),
        "nan": float("nan"),
        "deep": {"list": [float("inf"), {"more": float("-inf")}]},
    }


def _every_event() -> list[Event]:
    values = _holding_many_values_a_frame_and_a_nan()
    question = Question(name="q", title="Q", dtype=None, json_schema={}, default=[values["nan"], values["frame"]])
    unit = WaitingUnit(id="u", node_id="n", row=(), questions=(question,), inputs=values)
    state = RunState(status="waiting", waiting=(unit,), output_columns={Ref("n.a"): OutputColumn(rows=[()], values=[values["nan"]]), Ref("n.b"): OutputColumn(rows=[()], values=[values["frame"]]), Ref("n.c"): OutputColumn(rows=[()], values=[values["many"]["values"]])})
    return [
        RunStart(waiting=(unit,)),
        NodeStart(node_id="n"),
        NodeProgress(node_id="n", done=1, total=None),
        NodeComplete(node_id="n", result=values),
        NodeSkipped(node_id="n"),
        NodeFailed(node_id="n", cause=Problem(type="boom", msg="Boom.", loc=("n",), ctx={"seen": float("nan")})),
        NodeRetry(node_id="n", row=(), attempt=1, retries=2, error="e", delay=0.5),
        UnitWaiting(unit=unit),
        RunCheckpoint(state=state.model_copy(update={"status": "running"})),
        RunEnd(state=state),
    ]


@pytest.mark.parametrize("event", _every_event(), ids=lambda event: event.type)
def test_every_event_writes_itself_as_json_and_a_strict_parser_reads_it(event):
    document = _strict(event.model_dump_json())
    assert document == _strict(TypeAdapter(Event).dump_json(event))
    assert document == _strict(json.dumps(event.model_dump(mode="json"), allow_nan=False))


def test_a_frame_of_values_is_written_whole_with_one_rule_for_a_float_that_is_not_finite():
    event = NodeComplete(node_id="n", result=_holding_many_values_a_frame_and_a_nan())

    written = _strict(event.model_dump_json())["result"]

    assert written["nan"] == "NaN"
    assert written["many"] == {"rows": [[0], [1]], "values": ["NaN", 1.5]}
    assert written["frame"]["data"] == {"a": ["NaN", 1.0], "b": ["x", None]}
    assert written["deep"] == {"list": ["Infinity", {"more": "-Infinity"}]}
