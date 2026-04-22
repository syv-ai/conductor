"""Server-Sent Events framing helpers for conductor's execution stream."""

from __future__ import annotations

import json
from typing import Any


def sse_frame(event: Any) -> str:
    """Serialize a single conductor ``ExecutionEvent`` as an SSE ``data:`` frame.

    The frame ends with ``\\n\\n`` per the SSE spec; the JSON payload uses
    ``default=str`` so that dataclasses, datetimes, and sentinel objects
    don't explode the stream on unusual node results.
    """
    return f"data: {json.dumps(event, default=str)}\n\n"
