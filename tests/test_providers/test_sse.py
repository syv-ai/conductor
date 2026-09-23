"""A frame is an event as JSON-ready data: records through pydantic, a typed value through the codec by its own type."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from conductor.errors import ErrorCause  # noqa: E402
from conductor.execution.events import (  # noqa: E402
    ExecutionEvent,
    GraphCompleteEvent,
    GraphPendingEvent,
    NodeCompleteEvent,
    NodeErrorEvent,
    PendingUnit,
)
from conductor.execution.record import RunRecord  # noqa: E402
from conductor.metadata import Input  # noqa: E402
from conductor.series import Index, Series  # noqa: E402
from conductor.widgets import Textarea  # noqa: E402
from conductor_nodes.types import Json, Text  # noqa: E402
from conductor_providers.fastapi.sse import sse_frame  # noqa: E402


def _payload(event: ExecutionEvent) -> dict:
    frame = sse_frame(event)
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    return json.loads(frame[len("data: "):])


def test_an_ending_dumps_its_results_and_its_record():
    lines = Series(Index("lines", parent=Index("docs")), [Text("a"), Text("b")], rows=((0, 0), (1, 0)))
    record = RunRecord(cells=[{"ref": ["split", "result"], "row": [0], "value": "a"}], node_fingerprints={"split": "f" * 64})

    payload = _payload(GraphCompleteEvent(type="graph_complete", results={"split": {"result": lines}}, record=record))

    assert payload["results"]["split"]["result"] == {
        "index": {"id": "lines", "parent": {"id": "docs", "parent": None}},
        "rows": [[0, 0], [1, 0]],
        "values": ["a", "b"],
    }
    assert payload["record"] == record.model_dump(mode="json")


def test_a_pending_units_questions_and_a_failures_cause_are_records():
    question = Input(name="ask.result", dtype=Text, title="Svar", widget=Textarea(), default=Text("forslag"), optional=True)

    pending = _payload(GraphPendingEvent(
        type="graph_pending",
        pending=[PendingUnit(node_id="ask", row=(1,), prompt=None, questions=(question,))],
        results={},
        record=RunRecord(),
    ))
    failed = _payload(NodeErrorEvent(type="node_error", node_id="n", error="boom", cause=ErrorCause(code="boom", message="Boom.", row=(2,))))

    (asked,) = pending["pending"][0]["questions"]
    assert (asked["name"], asked["default"], asked["widget"]["kind"]) == ("ask.result", "forslag", "textarea")
    assert pending["pending"][0]["row"] == [1]
    assert failed["cause"] == {"code": "boom", "message": "Boom.", "details": {}, "row": [2]}


def test_a_typed_value_dumps_through_its_own_type_and_one_with_no_form_raises():
    parsed = Series(Index("parts"), [Json({"a": 1}), Json([2])])

    assert _payload(GraphCompleteEvent(type="graph_complete", results={"parse": {"result": parsed}}, record=RunRecord()))["results"]["parse"]["result"]["values"] == [{"a": 1}, [2]]
    assert _payload(NodeCompleteEvent(type="node_complete", node_id="n", result={"n": float("nan")}))["result"] == {"n": None}
    with pytest.raises(Exception, match="object"):
        sse_frame(NodeCompleteEvent(type="node_complete", node_id="n", result={"x": object()}))
