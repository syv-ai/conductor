"""Server-Sent Events framing for conductor's execution stream."""

from __future__ import annotations

from conductor.events import Event


def sse_frame(event: Event) -> str:
    """One ``Event`` as an SSE ``data:`` frame, ending with ``\n\n`` per the SSE spec."""
    return f"data: {event.model_dump_json()}\n\n"
