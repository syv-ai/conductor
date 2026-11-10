"""What a run says as it goes.

``run.stream()`` yields these records one by one, ending with ``RunEnd``. A
host that streams a run (a UI, an SSE endpoint) imports them from here. Each
is a frozen pydantic model, one of the union ``Event`` told apart by
``type``, and writes itself as JSON with ``model_dump(mode="json")``.
"""

from conductor._internal.leg import (
    Event,
    NodeComplete,
    NodeFailed,
    NodeProgress,
    NodeRetry,
    NodeSkipped,
    NodeStart,
    RunCheckpoint,
    RunEnd,
    RunStart,
    UnitWaiting,
)

__all__ = [
    "Event",
    "NodeComplete",
    "NodeFailed",
    "NodeProgress",
    "NodeRetry",
    "NodeSkipped",
    "NodeStart",
    "RunCheckpoint",
    "RunEnd",
    "RunStart",
    "UnitWaiting",
]
