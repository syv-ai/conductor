"""A frame's values take one JSON form: records through pydantic, the rest through their type."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("fastapi")

from conductor.errors import ErrorCause  # noqa: E402
from conductor.metadata import Input  # noqa: E402
from conductor.series import Index, Series  # noqa: E402
from conductor.widgets import Textarea  # noqa: E402
from conductor_nodes.types import Json, Text  # noqa: E402
from conductor_providers.fastapi.sse import sse_frame  # noqa: E402


def _payload(event: dict) -> dict:
    frame = sse_frame(event)
    assert frame.startswith("data: ") and frame.endswith("\n\n")
    return json.loads(frame[len("data: "):])


def test_a_series_in_a_result_keeps_its_index_and_rows():
    lines = Series(Index("lines", parent=Index("docs")), [Text("a"), Text("b")], rows=((0, 0), (1, 0)))

    payload = _payload({"type": "graph_complete", "results": {"split": {"result": lines}}, "cells": {}})

    assert payload["results"]["split"]["result"] == {
        "index": {"id": "lines", "parent": {"id": "docs", "parent": None}},
        "rows": [[0, 0], [1, 0]],
        "values": ["a", "b"],
    }


def test_a_pending_units_questions_and_a_failures_cause_are_records():
    question = Input(name="ask.result", dtype=Text, title="Svar", widget=Textarea(title="Svar"), default=Text("forslag"), optional=True)

    pending = _payload({"type": "graph_pending", "pending": [{"node_id": "ask", "row": None, "prompt": None, "questions": (question,)}], "results": {}, "cells": {}})
    failed = _payload({"type": "node_error", "node_id": "n", "error": "boom", "cause": ErrorCause(code="boom", message="Boom.", row=(2,))})

    (asked,) = pending["pending"][0]["questions"]
    assert (asked["name"], asked["default"], asked["widget"]["kind"]) == ("ask.result", "forslag", "textarea")
    assert failed["cause"] == {"code": "boom", "message": "Boom.", "details": {}, "row": [2]}


def test_a_type_dumps_through_its_own_schema_and_a_value_with_none_raises():
    assert _payload({"type": "node_complete", "result": {"parsed": Json({"a": [1]})}})["result"] == {"parsed": {"a": [1]}}

    with pytest.raises(Exception, match="serialize"):
        sse_frame({"type": "node_complete", "result": {"x": object()}})
