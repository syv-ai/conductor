"""The events a run reports while ``resume()`` calls its nodes.

``async for event in run.stream()`` gets them one at a time; the last is ``RunEnd``. For the README graph,
``lines -> shout -> join``, the events start ``RunStart()``, ``NodeStart(loc=("lines",))`` and end
``NodeComplete(loc=("join",), outcomes=1)``, ``RunEnd(status="complete")``. A host that shows a run as it goes,
such as a web page fed by server-sent events, imports them from here. Each is a frozen pydantic model with a
``type`` (``"node_complete"``), written as JSON with ``model_dump(mode="json")``; ``Event`` is their union, which
reads any of them back.
"""

from conductor._internal.run.events import (
    Event,
    NodeComplete,
    NodeFailed,
    NodeProgress,
    NodeRetry,
    NodeSkipped,
    NodeStart,
    NodeWaiting,
    RunCheckpoint,
    RunEnd,
    RunStart,
)

# The order in which the reference (``python -m conductor.about``) lists them: the order in which a run reports them.
__all__ = [
    "Event",
    "RunStart",
    "NodeStart",
    "NodeProgress",
    "NodeComplete",
    "NodeSkipped",
    "NodeFailed",
    "NodeRetry",
    "NodeWaiting",
    "RunCheckpoint",
    "RunEnd",
]
