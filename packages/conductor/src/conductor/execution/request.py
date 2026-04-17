"""NodeExecRequest — the universal DTO passed to every node."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from conductor.execution.state import FlowRunState


@dataclass(frozen=True)
class NodeExecRequest:
    """Passed to every node execution — simple, compound, or extension."""

    node_id: str
    node_type: str
    inputs: dict[str, Any]
    data: dict[str, Any]
    state: FlowRunState
