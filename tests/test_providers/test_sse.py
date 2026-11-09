"""An event is what pydantic writes: every value by its own type, a float that isn't finite as text."""

from __future__ import annotations

import json

import pytest
from conductor import Problem, RunState
from conductor._internal.execution.run_state import OutputColumn, Question, WaitingUnit
from conductor.events import Event, NodeComplete, NodeFailed, RunEnd, UnitWaiting
from conductor_nodes.types import Json
from helpers import stream_of


def _payload(event: Event) -> dict:
    return json.loads(event.model_dump_json())


def test_many_values_keep_their_rows_and_an_ending_dumps_its_state():
    lines = {"rows": [(0, 0), (1, 0)], "values": ["a", "b"]}
    state = RunState(status="complete", output_columns={"split.result": OutputColumn(rows=[(0,)], values=["a"])}, node_fingerprints={"split": "f" * 64})

    payload = _payload(NodeComplete(type="node_complete", node_id="split", result={"result": lines}))
    ending = _payload(RunEnd(state=state))

    assert payload["result"]["result"] == {
        "rows": [[0, 0], [1, 0]],
        "values": ["a", "b"],
    }
    assert ending["state"] == state.model_dump(mode="json")


def test_a_pending_units_questions_and_a_failures_cause_are_records():
    question = Question(name="result", title="Svar", dtype={"id": "text"}, json_schema={"type": "string"}, widget={"kind": "textarea"}, default="forslag", optional=True)

    pending = _payload(UnitWaiting(unit=WaitingUnit(id="ask@1", node_id="ask", row=(1,), questions=(question,))))
    failed = _payload(NodeFailed(type="node_failed", node_id="n", cause=Problem(type="boom", msg="Boom.", loc=("n", 2))))

    (asked,) = pending["unit"]["questions"]
    assert (asked["name"], asked["default"], asked["widget"]["kind"]) == ("result", "forslag", "textarea")
    assert pending["unit"]["row"] == [1]
    assert failed["cause"] == {"type": "boom", "msg": "Boom.", "loc": ["n", 2], "ctx": {}, "fatal": True}


def test_a_typed_value_dumps_through_its_own_type_and_one_with_no_form_raises():
    parsed = {"rows": [(0,), (1,)], "values": [Json({"a": 1}), Json([2])]}

    assert _payload(NodeComplete(type="node_complete", node_id="parse", result={"result": parsed}))["result"]["result"]["values"] == [{"a": 1}, [2]]
    assert _payload(NodeComplete(type="node_complete", node_id="n", result={"n": float("nan")}))["result"] == {"n": "NaN"}
    with pytest.raises(Exception, match="object"):
        NodeComplete(type="node_complete", node_id="n", result={"x": object()}).model_dump_json()


def test_a_branch_nodes_completion_streams_with_only_the_branch_taken():
    import asyncio

    import conductor_nodes
    from conductor import CompiledGraph, Graph, GraphNode, Static

    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="d", type="logic-if-empty", version=1, bindings={"text": Static("x")})]),
        conductor_nodes.registry(),
    )

    async def events() -> list[Event]:
        return [event async for event in stream_of(compiled)]

    (completed,) = [_payload(event) for event in asyncio.run(events()) if event.type == "node_complete"]

    assert completed["result"] == {"not_empty": "x"}
