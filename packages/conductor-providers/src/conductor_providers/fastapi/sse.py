"""Server-Sent Events framing for conductor's execution stream, and an event as JSON-ready data."""

from __future__ import annotations

import json
from typing import Any

from conductor.codec import to_wire
from pydantic_core import to_jsonable_python


def as_data(event: Any) -> Any:
    """An event as JSON-ready data, the one place a frame is serialised.

    A record (a ``RunState``, an ``ErrorCause``, an ``Input`` among a
    pending unit's questions) dumps through pydantic; a typed value pydantic
    has no rule for by itself — a ``Series`` in a result, a type a host
    declared — dumps through the codec by its own type, so a series keeps
    its index and rows and whatever it holds dumps the same way. A float
    that is not a number is ``null``. A value with no JSON form raises.
    """
    return to_jsonable_python(event, inf_nan_mode="null", fallback=lambda value: to_wire(value, type(value)))


def sse_frame(event: Any) -> str:
    """One ``ExecutionEvent`` as an SSE ``data:`` frame, ending with ``\n\n`` per the SSE spec."""
    return f"data: {json.dumps(as_data(event))}\n\n"
