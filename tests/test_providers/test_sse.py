"""An event is what pydantic writes: every value by its own type, a float that isn't finite as text."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import (
    SKIPPED,
    CompiledGraph,
    Graph,
    GraphNode,
    Problem,
    Result,
    RunState,
    Static,
    node,
)
from conductor._internal.execution.run_state import OutputColumn, Question, WaitingUnit, WorkOfNode
from conductor.events import Event, NodeComplete, NodeFailed, RunEnd, UnitWaiting
from helpers import Json, registry_of, stream_of


def _payload(event: Event) -> dict:
    return json.loads(event.model_dump_json())


def test_many_values_keep_their_rows_and_an_ending_says_only_its_status():
    lines = {"rows": [(0, 0), (1, 0)], "values": ["a", "b"]}
    state = RunState(status="complete", nodes={"split": WorkOfNode(fingerprint="f" * 64, outputs={"result": OutputColumn(rows=[(0,)], values=["a"])})})

    payload = _payload(NodeComplete(type="node_complete", node_id="split", outputs={"result": lines}))
    ending = _payload(RunEnd(state=state))

    assert payload["outputs"]["result"] == {
        "rows": [[0, 0], [1, 0]],
        "values": ["a", "b"],
    }
    assert ending == {"type": "run_end", "status": "complete"}


def test_a_pending_units_questions_and_a_failures_cause_are_records():
    question = Question(name="result", title="Svar", dtype={"id": "text"}, json_schema={"type": "string"}, widget={"type": "textarea"}, default="forslag", required=False)

    pending = _payload(UnitWaiting(unit=WaitingUnit(id="ask@1", node_id="ask", row=(1,), questions=(question,))))
    failed = _payload(NodeFailed(type="node_failed", node_id="n", cause=Problem(type="boom", msg="Boom.", loc=("n", 2))))

    (asked,) = pending["unit"]["questions"]
    assert (asked["name"], asked["default"], asked["widget"]["type"]) == ("result", "forslag", "textarea")
    assert pending["unit"]["row"] == [1]
    assert failed["cause"] == {"type": "boom", "msg": "Boom.", "loc": ["n", 2], "ctx": {}, "fatal": True}


def test_a_typed_value_dumps_through_its_own_type_and_one_with_no_form_raises():
    parsed = {"rows": [(0,), (1,)], "values": [Json({"a": 1}), Json([2])]}

    assert _payload(NodeComplete(type="node_complete", node_id="parse", outputs={"result": parsed}))["outputs"]["result"]["values"] == [{"a": 1}, [2]]
    assert _payload(NodeComplete(type="node_complete", node_id="n", outputs={"n": float("nan")}))["outputs"] == {"n": "NaN"}
    with pytest.raises(Exception, match="object"):
        NodeComplete(type="node_complete", node_id="n", outputs={"x": object()}).model_dump_json()


@dataclass(frozen=True)
class Emptiness:
    not_empty: Annotated[str, Result(exclusive_group="emptiness")]
    empty: Annotated[str, Result(exclusive_group="emptiness")]


@node("if-empty")
def if_empty(text: str) -> Emptiness:
    return Emptiness(not_empty=text, empty=SKIPPED) if text.strip() else Emptiness(not_empty=SKIPPED, empty=text)


def test_a_branch_nodes_completion_streams_with_only_the_branch_taken():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="d", type="if-empty", version=1, bindings={"text": Static("x")})]),
        registry_of(if_empty),
    )

    async def events() -> list[Event]:
        return [event async for event in stream_of(compiled)]

    (completed,) = [_payload(event) for event in asyncio.run(events()) if event.type == "node_complete"]

    assert completed["outputs"] == {"not_empty": "x"}
