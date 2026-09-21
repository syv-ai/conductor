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
    """A topological order of ``dependencies``, and the ids on a cycle.

    Returns ``(order, cyclic)``. ``cyclic`` is exactly the nodes that lie on
    a cycle — a strongly connected component of more than one node, or a
    node that depends on itself — and no other: a node downstream of a
    cycle is not part of it. ``order`` holds every other id, in an order
    where each dependency that is not cyclic precedes its dependent; an
    edge from a cyclic node is left out of the ordering, so a node fed by
    a cycle still has a place and an editor can draw it. A cycle is a state
    an editor can be in, so it is returned rather than raised; compile
    turns each cyclic id into a ``Problem``. A dependency on an id the map
    does not hold is ignored here — compile reports it on the field that
    holds the dangling reference.
    """
    ids = set(dependencies)
    cyclic = _cyclic(dependencies, ids)
    placeable = ids - cyclic
    counted = {i: deps & placeable for i, deps in dependencies.items() if i in placeable}
    pending = {i: len(deps) for i, deps in counted.items()}
    dependents: dict[str, list[str]] = {i: [] for i in placeable}
    for i, deps in counted.items():
        for dep in deps:
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
    return tuple(order), frozenset(cyclic)


def _cyclic(dependencies: Mapping[str, frozenset[str]], ids: set[str]) -> set[str]:
    """The ids on a cycle: every member of a strongly connected component
    with more than one node, and every node that depends on itself.
    Tarjan's algorithm, iterative so a long chain cannot exhaust the stack."""
    index: dict[str, int] = {}
    low: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    cyclic: set[str] = set()
    counter = 0

    for root in dependencies:
        if root in index:
            continue
        work: list[tuple[str, list[str]]] = [(root, [dep for dep in dependencies[root] if dep in ids])]
        index[root] = low[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, remaining = work[-1]
            if remaining:
                dep = remaining.pop()
                if dep not in index:
                    index[dep] = low[dep] = counter
                    counter += 1
                    stack.append(dep)
                    on_stack.add(dep)
                    work.append((dep, [d for d in dependencies[dep] if d in ids]))
                elif dep in on_stack:
                    low[node] = min(low[node], index[dep])
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                low[parent] = min(low[parent], low[node])
            if low[node] == index[node]:
                component: list[str] = []
                while True:
                    member = stack.pop()
                    on_stack.discard(member)
                    component.append(member)
                    if member == node:
                        break
                if len(component) > 1 or node in dependencies[node]:
                    cyclic.update(component)
    return cyclic
