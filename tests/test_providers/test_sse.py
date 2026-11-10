"""An event is what pydantic writes: every value by its own type, a float that isn't finite as text."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from conductor import SKIPPED, Ask, GraphRecord, NodeRecord, Param, Problem, node
from conductor.events import NodeComplete, NodeFailed, RunEnd, UnitWaiting
from conductor.widgets import Textarea

from tests.test_core.nodes import registry_of


def _payload(event: Any) -> dict:
    return json.loads(event.model_dump_json())


def _events(graph: GraphRecord, *nodes: Any) -> list[Any]:
    run = registry_of(*nodes).load(graph).with_inputs()

    async def collect() -> list[Any]:
        return [event async for event in run.stream()]

    return asyncio.run(collect())


@node("split")
def split(text: str) -> list[str]:
    return text.split(",")


def test_an_ending_says_only_its_status():
    (ending,) = [e for e in _events(GraphRecord(nodes=[NodeRecord(id="s", node="split", values={"text": "a,b"})]), split) if isinstance(e, RunEnd)]

    assert _payload(ending) == {"type": "run_end", "status": "complete"}
    assert ending.run.outputs == {"s.result": ["a", "b"]}


def test_a_waiting_units_question_and_a_failures_cause_are_records():
    @node("ask")
    def ask(text: str) -> Annotated[str, Param(title="Svar")]:
        raise Ask("Hvad?", default="forslag", widget=Textarea())

    (waiting,) = [e for e in _events(GraphRecord(nodes=[NodeRecord(id="ask", node="ask", values={"text": "x"})]), ask) if isinstance(e, UnitWaiting)]
    failed = _payload(NodeFailed(node_id="n", cause=Problem(type="boom", msg="Boom.", loc=("n", 2))))

    unit = _payload(waiting)["unit"]
    assert (unit["id"], unit["prompt"], unit["default"], unit["widget"]["type"], unit["answer_schema"]["type"]) == ("ask", "Hvad?", "forslag", "textarea", "string")
    assert unit["inputs"] == {"text": "x"}
    assert failed["cause"] == {"type": "boom", "msg": "Boom.", "loc": ["n", 2], "ctx": {}}


def test_a_float_that_isnt_finite_is_text_and_a_value_with_no_form_raises():
    assert _payload(NodeComplete(node_id="n", outputs={"n": float("nan")}))["outputs"] == {"n": "NaN"}
    with pytest.raises(Exception, match="object"):
        NodeComplete(node_id="n", outputs={"x": object()}).model_dump_json()


@dataclass(frozen=True)
class Emptiness:
    not_empty: Annotated[str, Param(exclusive_group="emptiness")]
    empty: Annotated[str, Param(exclusive_group="emptiness")]


@node("if-empty")
def if_empty(text: str) -> Emptiness:
    return Emptiness(not_empty=text, empty=SKIPPED) if text.strip() else Emptiness(not_empty=SKIPPED, empty=text)


def test_a_branch_nodes_completion_streams_with_the_branch_not_taken_blank():
    events = _events(GraphRecord(nodes=[NodeRecord(id="d", node="if-empty", values={"text": "x"})]), if_empty)

    (completed,) = [_payload(event) for event in events if event.type == "node_complete"]

    assert completed["outputs"] == {"not_empty": "x", "empty": None}
