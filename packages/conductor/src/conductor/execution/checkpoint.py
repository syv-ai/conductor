"""Flow checkpointing for human-in-the-loop pause/resume."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
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
        skipped_edges: Edge IDs marked skipped by decision nodes. Restored
            on resume so downstream dependents see the same branches.
        signal_name: If the pause is for an external signal, its name.
        correlation: Optional CEL expression for signal correlation.
        signal_timeout_seconds: Optional timeout for the signal wait.
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
    skipped_edges: list[str] = field(default_factory=list)
    signal_name: str | None = None
    correlation: str | None = None
    signal_timeout_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (JSON-safe)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlowCheckpoint:
        """Restore from a plain dict."""
        # Tolerate older checkpoints that don't have the new fields.
        known = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)

    def matches_signal(
        self,
        signal_name: str,
        payload: dict[str, Any] | None = None,
    ) -> bool:
        """Return True if an incoming signal should wake this checkpoint.

        Hosts use this to route external events. The match:

        * name must equal ``self.signal_name``.
        * if ``self.correlation`` is set, it's evaluated as a CEL expression
          against the candidate payload (exposed as ``signal`` and as the
          root ``$``). A truthy result wins; any expression error is treated
          as "no match" so a buggy correlator never drags the wrong flow
          into motion.
        """
        if self.signal_name is None or self.signal_name != signal_name:
            return False
        if not self.correlation:
            return True
        from conductor.expr import ExpressionError, parse

        ctx = {"signal": payload or {}, "$": payload or {}}
        if payload:
            ctx.update(payload)
        try:
            return bool(parse(self.correlation).evaluate(ctx))
        except ExpressionError:
            return False
