"""An event is what pydantic writes: every value by its own type, a float that isn't finite as text."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Annotated, Any

from conductor import SKIPPED, Ask, GraphRecord, NodeRecord, Param, Problem, Waiting, node
from conductor.events import NodeComplete, NodeFailed, NodeWaiting, RunEnd
from conductor.widgets import Textarea

from tests.test_core.nodes import registry_of


def _payload(event: Any) -> dict:
    return json.loads(event.model_dump_json())


def _events(graph: GraphRecord, *nodes: Any) -> list[Any]:
    run = registry_of(*nodes).load(graph).with_inputs()

    async def collect() -> list[Any]:
        return [event async for event in run.stream()]

    return asyncio.run(collect())


@node("split", title="Split", description="For a test.")
def split(text: str) -> list[str]:
    return text.split(",")


def test_an_ending_says_its_status_and_cause_and_leaves_the_run_out():
    (ending,) = [e for e in _events(GraphRecord(nodes=[NodeRecord(id="s", node="split", values={"text": "a,b"})]), split) if isinstance(e, RunEnd)]

    assert _payload(ending) == {"type": "run_end", "status": "complete", "cause": None}
    assert ending.run.outputs == {"s.result": ["a", "b"]}


def test_a_waiting_units_question_and_a_failures_cause_are_records():
    @node("ask", title="Ask", description="For a test.")
    def ask(text: str) -> Annotated[str, Param(title="Svar", widget=Textarea())]:
        raise Ask("Hvad?", default="forslag")

    (waiting,) = [e for e in _events(GraphRecord(nodes=[NodeRecord(id="ask", node="ask", values={"text": "x"})]), ask) if isinstance(e, NodeWaiting)]
    failed = _payload(NodeFailed(loc=("n",), cause=Problem(type="boom", msg="Boom.", loc=("n", 2))))

    unit = _payload(waiting)["waiting"]
    (question,) = unit["questions"]
    assert (unit["id"].partition("~")[0], unit["prompt"], question["title"], question["default"], question["widget"]["type"], question["schema"]["type"]) == ("ask", "Hvad?", "Svar", "forslag", "textarea", "string")
    assert unit["inputs"] == {"text": "x"}
    assert failed["cause"] == {"type": "boom", "msg": "Boom.", "loc": ["n", 2], "ctx": {}}


def test_a_completion_is_counts_and_a_float_that_isnt_finite_in_a_question_is_text():
    assert _payload(NodeComplete(loc=("n",), units=3, not_called=1)) == {"type": "node_complete", "loc": ["n"], "units": 3, "not_called": 1, "reused_units": 0}
    waiting = NodeWaiting(waiting=Waiting(id="n", loc=("n",), prompt="?", questions=[], inputs={"x": float("nan")}))
    assert _payload(waiting)["waiting"]["inputs"] == {"x": "NaN"}


@dataclass(frozen=True)
class Emptiness:
    not_empty: Annotated[str, Param(exclusive_group="emptiness")]
    empty: Annotated[str, Param(exclusive_group="emptiness")]


@node("if-empty", title="If empty", description="For a test.")
def if_empty(text: str) -> Emptiness:
    return Emptiness(not_empty=text, empty=SKIPPED) if text.strip() else Emptiness(not_empty=SKIPPED, empty=text)


def test_a_branch_nodes_completion_streams_with_the_branch_not_taken_blank():
    events = _events(GraphRecord(nodes=[NodeRecord(id="d", node="if-empty", values={"text": "x"})]), if_empty)

    (completed,) = [_payload(event) for event in events if event.type == "node_complete"]

    assert completed["units"] == 1 and events[-1].run.results["d"] == {"not_empty": "x", "empty": None}
