"""Views derived from the stored bindings: dependencies, and a flow's interface.

Nothing here is persisted. A dependency map answers "which nodes must
finish before this one starts" from the one place wiring lives; a flow's
interface answers "what does this flow take and return" from its nodes.

There is deliberately no edge view: an edge is something a canvas draws,
and a canvas derives its own from ``bindings``.
"""

from __future__ import annotations

from collections.abc import Iterable

from conductor.graph.binding import Sources
from conductor.graph.model import GraphNode


def dependencies_of(nodes: Iterable[GraphNode]) -> dict[str, frozenset[str]]:
    """Which nodes each node waits for, by id.

    A set: a node that feeds two inputs of the same target is one
    dependency. Operand order matters only within a ``Sources`` and is
    kept there.
    """
    return {
        node.id: frozenset(
            ref.node_id
            for binding in node.bindings.values()
            if isinstance(binding, Sources)
            for ref in binding.refs
        )
        for node in nodes
    }
