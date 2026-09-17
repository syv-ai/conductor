"""Server-Sent Events framing for conductor's execution stream, and the one JSON form a frame's values take."""

from __future__ import annotations

import json
from typing import Any

from pydantic import TypeAdapter
from pydantic_core import to_jsonable_python


def _jsonable(value: Any) -> Any:
    """``value`` as JSON-ready data, the one place a frame or a result is serialised.

    A record (an ``ErrorCause``, an ``Input`` among a pending unit's
    questions) dumps through pydantic. A value pydantic has no rule for by
    itself — a ``Series`` in a result, a type a host declared — dumps
    through its own type's schema, so a series keeps its index and rows.
    A value with no JSON form raises rather than turning into its repr.
    """
    return to_jsonable_python(value, fallback=lambda unknown: TypeAdapter(type(unknown)).dump_python(unknown, mode="json"))


def sse_frame(event: Any) -> str:
    """Serialize a single conductor ``ExecutionEvent`` as an SSE ``data:`` frame.

    The frame ends with ``\\n\\n`` per the SSE spec.
    """
    return f"data: {json.dumps(_jsonable(event))}\n\n"
