"""FlowRunState — mutable state for a single flow execution run."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from flowengine.execution.events import EventSink, ExecutionEvent
from flowengine.execution.store import FlowStore
from flowengine.graph.compiler import CompiledGraph
from flowengine.execution.resolver import InputResolver
from flowengine.types import NodeResult


@dataclass
class FlowRunState:
    """Mutable state for a single flow execution run."""

    compiled: CompiledGraph
    resolver: InputResolver
    results: dict[str, NodeResult] = field(default_factory=dict)
    store: FlowStore = field(default_factory=FlowStore)
    _event_sink: EventSink = field(default_factory=EventSink)
    _cancelled: threading.Event = field(default_factory=threading.Event)
    _started_at: float = field(default_factory=time.monotonic)
    _timeout_seconds: int = 300
    context: dict[str, Any] = field(default_factory=dict)

    def emit(self, event: ExecutionEvent) -> None:
        self._event_sink.push(event)

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def is_timed_out(self) -> bool:
        return (time.monotonic() - self._started_at) >= self._timeout_seconds

    def remaining_seconds(self) -> float | None:
        elapsed = time.monotonic() - self._started_at
        remaining = self._timeout_seconds - elapsed
        return max(0.0, remaining)
