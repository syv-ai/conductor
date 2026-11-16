"""The events a run reports while it calls its nodes.

A host that shows a run as it happens, such as a web page fed by server-sent events, reads them one at a time::

    async for event in run.stream():
        ...

The first event is ``RunStart`` and the last is ``RunEnd``. Take the README graph, ``lines -> shout -> join``. Its
events start with ``RunStart()`` and ``NodeStart(loc=("lines",))``. They end with
``NodeComplete(loc=("join",), count=CallCount(finished=1, ...))`` and ``RunEnd(status="complete")``.

Each event is a frozen pydantic model with a ``type``, such as ``"node_complete"``. Write one as JSON with
``model_dump(mode="json")``. ``Event`` is the union of them all, so it reads any of them back.
"""

from conductor._internal.run import (
    Event,
    NodeComplete,
    NodeFailed,
    NodeProgress,
    NodeRetry,
    NodeStart,
    NodeWaiting,
    RunCheckpoint,
    RunEnd,
    RunStart,
)

# The reference (``python -m conductor.about``) lists the events in this order, which is the order a run reports them in.
__all__ = [
    "Event",
    "RunStart",
    "NodeStart",
    "NodeProgress",
    "NodeComplete",
    "NodeFailed",
    "NodeRetry",
    "NodeWaiting",
    "RunCheckpoint",
    "RunEnd",
]
