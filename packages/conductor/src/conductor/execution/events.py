"""Execution event types and thread-safe EventSink."""

import threading
from collections import deque
from typing import Any, Literal, TypedDict


class NodeStartEvent(TypedDict, total=False):
    type: Literal["node_start"]
    node_id: str
    idempotency_key: str


class NodeCompleteEvent(TypedDict, total=False):
    type: Literal["node_complete"]
    node_id: str
    result: Any
    cached: bool


class NodeSkippedEvent(TypedDict):
    type: Literal["node_skipped"]
    node_id: str


class NodeErrorEvent(TypedDict, total=False):
    type: Literal["node_error"]
    node_id: str
    error: str
    is_validation: bool
    is_timeout: bool


class NodeProgressEvent(TypedDict):
    type: Literal["node_progress"]
    node_id: str
    current: int
    total: int


class FlowCompleteEvent(TypedDict):
    type: Literal["flow_complete"]
    results: dict[str, Any]


class FlowErrorEvent(TypedDict):
    type: Literal["flow_error"]
    error: str
    is_validation: bool


class FlowCancelledEvent(TypedDict):
    type: Literal["flow_cancelled"]
    completed_nodes: list[str]


class FlowTimeoutEvent(TypedDict):
    type: Literal["flow_timeout"]
    completed_nodes: list[str]
    elapsed_seconds: float
    timeout_seconds: int


class FlowPausedEvent(TypedDict, total=False):
    type: Literal["flow_paused"]
    node_id: str
    prompt: str
    schema: dict | None
    checkpoint: dict


class NodeRetryEvent(TypedDict):
    type: Literal["node_retry"]
    node_id: str
    attempt: int
    max_retries: int
    error: str
    delay: float


class CompensationStartEvent(TypedDict):
    type: Literal["compensation_start"]
    node_id: str
    compensation_node_id: str


class CompensationCompleteEvent(TypedDict):
    type: Literal["compensation_complete"]
    node_id: str
    compensation_node_id: str
    result: Any


class CompensationFailedEvent(TypedDict):
    type: Literal["compensation_failed"]
    node_id: str
    compensation_node_id: str
    error: str


class SignalWaitingEvent(TypedDict, total=False):
    type: Literal["signal_waiting"]
    node_id: str
    signal_name: str
    correlation: str | None
    timeout_seconds: float | None
    checkpoint: dict


ExecutionEvent = (
    NodeStartEvent
    | NodeCompleteEvent
    | NodeSkippedEvent
    | NodeErrorEvent
    | NodeProgressEvent
    | FlowCompleteEvent
    | FlowErrorEvent
    | FlowCancelledEvent
    | FlowTimeoutEvent
    | FlowPausedEvent
    | NodeRetryEvent
    | CompensationStartEvent
    | CompensationCompleteEvent
    | CompensationFailedEvent
    | SignalWaitingEvent
)


class EventSink:
    """Thread-safe event queue. Compound nodes push; engine drains."""

    def __init__(self) -> None:
        self._queue: deque[ExecutionEvent] = deque()
        self._lock = threading.Lock()

    def push(self, event: ExecutionEvent) -> None:
        with self._lock:
            self._queue.append(event)

    def pop(self) -> ExecutionEvent | None:
        with self._lock:
            return self._queue.popleft() if self._queue else None
