"""Built-in decision node.

A decision node evaluates CEL ``when`` expressions on its outgoing edges
and selectively marks branches as SKIPPED. Unlike an ``if-else`` node
whose logic lives inside Python, the decision node makes branching
visible on the diagram as data on the edges — the same pattern as BPMN
gateways, Step Functions ``Choice``, and Temporal's continue-as.

Register via :func:`register`; the runtime magic is in
``conductor/execution/engine.py`` which detects ``is_decision=True`` nodes
and post-processes their guards.

The node's output is the ``value`` input passed through unchanged, so
downstream consumers can still read the data that drove the decision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from conductor.types import NodeCategory
from conductor.widgets import Output, Text

if TYPE_CHECKING:
    from conductor import NodeRegistry


def register(registry: "NodeRegistry") -> None:
    """Register the canonical ``decision`` node."""

    @registry.node(
        "decision",
        version=1,
        name="Decision",
        description=(
            "Branching gateway. Evaluates `when` CEL expressions on outgoing "
            "edges — exactly one branch is taken, the others are marked "
            "SKIPPED."
        ),
        category=NodeCategory.DECISION,
        is_decision=True,
    )
    def decision(
        value: Annotated[Any, Text(label="Value")] = None,
    ) -> Annotated[Any, Output(label="Pass-through")]:
        """Pass the input through so downstream readers still see the value."""
        return value
