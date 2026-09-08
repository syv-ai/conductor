"""The edges facts of a graph: the dependency map, read once, and an order over it.

``dependencies_of`` is the one graph-wide reading of the edges; everything
that needs a whole-graph fact — the order, which nodes nothing consumes —
reads the map rather than walking the bindings again.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping

from conductor.graph.binding import Edges
from conductor.graph.model import GraphNode


def dependencies_of(nodes: Iterable[GraphNode]) -> dict[str, frozenset[str]]:
    """Which nodes each node waits for, by id.

    A set: a node that feeds two inputs of the same target is one
    dependency. Operand order matters only within a ``Edges`` and is
    kept there. An empty set means an input node — nothing is connected in.
    """
    return {
        node.id: frozenset(
            ref.node_id
            for binding in node.bindings.values()
            if isinstance(binding, Edges)
            for ref in binding.refs
        )
        for node in nodes
    }


def order_of(
    dependencies: Mapping[str, frozenset[str]],
) -> tuple[tuple[str, ...], frozenset[str]]:
    """A topological order of ``dependencies`` (Kahn's algorithm).

    Returns ``(order, cyclic)``: the ids in an order where every dependency
    precedes its dependent, and the ids that could not be placed because
    they lie on a cycle. A cycle is a state an editor can be in, so it is
    returned rather than raised; compile turns each id into a ``Problem``.
    A dependency on an id the map does not hold is ignored here — compile
    reports it on the field that holds the dangling reference.
    """
    ids = set(dependencies)
    pending = {i: len(deps & ids) for i, deps in dependencies.items()}
    dependents: dict[str, list[str]] = {i: [] for i in ids}
    for i, deps in dependencies.items():
        for dep in deps & ids:
            dependents[dep].append(i)

    ready = deque(i for i, n in pending.items() if n == 0)
    order: list[str] = []
    while ready:
        i = ready.popleft()
        order.append(i)
        for dependent in dependents[i]:
            pending[dependent] -= 1
            if pending[dependent] == 0:
                ready.append(dependent)

    return tuple(order), frozenset(ids - set(order))
