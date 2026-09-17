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

    with pytest.raises(TypeError, match="no JSON form"):
        sse_frame({"type": "node_complete", "result": {"x": object()}})


def test_what_a_value_holds_dumps_through_its_own_type_too():
    """An iterating node that returns ``Json`` gives a series of them; the
    series dumps through its schema and each ``Json`` through its own."""
    parsed = Series(Index("parts"), [Json({"a": 1}), Json([2])])

    payload = _payload({"type": "graph_complete", "results": {"parse": {"result": parsed}}, "cells": {}})

    assert payload["results"]["parse"]["result"]["values"] == [{"a": 1}, [2]]
    assert _payload({"type": "node_complete", "result": {"x": Json(Json({"k": Text("t")}))}})["result"] == {"x": {"k": "t"}}
    with pytest.raises(TypeError, match="no JSON form"):
        sse_frame({"type": "node_complete", "result": {"x": Series(Index("parts"), [object()])}})


def test_a_float_that_is_not_a_number_is_null_everywhere():
    nan = float("nan")

    payload = _payload({"type": "node_complete", "result": {"n": nan, "s": Series(Index("parts"), [nan])}})

    assert payload["result"] == {"n": None, "s": {"index": {"id": "parts", "parent": None}, "rows": [[0]], "values": [None]}}
