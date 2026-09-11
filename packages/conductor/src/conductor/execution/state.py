"""FlowRunState — mutable state for a single flow execution run."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from conductor.execution.resolver import InputResolver
from conductor.execution.store import FlowStore
from conductor.graph.compiled import CompiledGraph
from conductor.types import NodeResult


@dataclass
class FlowRunState:
    """Mutable state for a single flow execution run."""

    compiled: CompiledGraph
    resolver: InputResolver
    results: dict[str, NodeResult] = field(default_factory=dict)
    store: FlowStore = field(default_factory=FlowStore)
    _cancelled: threading.Event = field(default_factory=threading.Event)
    _started_at: float = field(default_factory=time.monotonic)
    _timeout_seconds: int = 300
    context: dict[str, Any] = field(default_factory=dict)
    # Nodes that have completed, in order.
    completed_order: list[str] = field(default_factory=list)
    # Signal registry — node_id -> signal name + correlation expression +
    # timeout deadline. Populated when the flow pauses on a signal node.
    pending_signals: dict[str, dict[str, Any]] = field(default_factory=dict)

    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()

    def is_timed_out(self) -> bool:
        return (time.monotonic() - self._started_at) >= self._timeout_seconds

    def remaining_seconds(self) -> float | None:
        elapsed = time.monotonic() - self._started_at
        remaining = self._timeout_seconds - elapsed
        return max(0.0, remaining)
