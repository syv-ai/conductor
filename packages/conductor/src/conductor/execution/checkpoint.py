"""Flow checkpointing for human-in-the-loop pause/resume."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class FlowCheckpoint:
    """Serializable snapshot of execution state at the point of pause.

    This is JSON-serializable so it can be stored in a database, sent to
    a frontend, or passed between processes. On resume, the caller provides
    this checkpoint plus the compiled graph to continue execution.

    Fields:
        completed_node_ids: Nodes that finished before the pause.
        waiting_node_id: The node that raised HumanInputRequired.
        waiting_node_type: Registry type of the waiting node.
        results: All completed node results (dict[node_id, NodeResult]).
        store_data: FlowStore contents at pause time.
        context: Host-app context dict.
        prompt: Human-readable description of what input is needed.
        input_schema: Optional schema describing expected response shape.
        execution_index: Position in execution_order where we paused.
    """

    completed_node_ids: list[str]
    waiting_node_id: str
    waiting_node_type: str
    results: dict[str, Any]
    store_data: dict[str, Any]
    context: dict[str, Any]
    prompt: str
    input_schema: dict[str, Any] | None
    execution_index: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowCheckpoint:
        """Restore from a plain dict."""
        return cls(**data)
