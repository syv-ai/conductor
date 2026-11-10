"""Which nodes each node of a graph reads, and an order over them.

``dependencies_of`` turns the edges into a map from each node to the nodes
it reads, and the order and which nodes nothing consumes are read off that
map. The map keeps node ids only. A pass that needs the fields an edge
joins — the edge walk, the conditions, embedding a compiled graph, a drawn
edge — or the sources of one node — the fatal cause a node inherits —
walks the bindings itself.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping

from conductor._internal.graph.binding import From
from conductor._internal.graph.graph_records import LoadedNode


def dependencies_of(nodes: Iterable[LoadedNode]) -> dict[str, frozenset[str]]:
    """Which nodes each node waits for, by id.

    A set: a node that feeds two inputs of the same target is one
    dependency. Operand order matters only within a ``From`` and is
    kept there. An empty set means an input node — nothing is connected in.
    """
    return {
        node.id: frozenset(
            ref.node_id
            for binding in node.bindings.values()
            if isinstance(binding, From)
            for ref in binding.refs
        )
        for node in nodes
    }


def topological_order(
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
    dependencies_outside_cycles = {node_id: dependencies_of_node & placeable for node_id, dependencies_of_node in dependencies.items() if node_id in placeable}
    unmet_dependency_count = {node_id: len(dependencies_of_node) for node_id, dependencies_of_node in dependencies_outside_cycles.items()}
    dependents: dict[str, list[str]] = {node_id: [] for node_id in placeable}
    for node_id, dependencies_of_node in dependencies_outside_cycles.items():
        for dep in dependencies_of_node:
            dependents[dep].append(node_id)

    ready = deque(node_id for node_id, count in unmet_dependency_count.items() if count == 0)
    order: list[str] = []
    while ready:
        node_id = ready.popleft()
        order.append(node_id)
        for dependent in dependents[node_id]:
            unmet_dependency_count[dependent] -= 1
            if unmet_dependency_count[dependent] == 0:
                ready.append(dependent)
    return tuple(order), frozenset(cyclic)


def _cyclic(dependencies: Mapping[str, frozenset[str]], ids: set[str]) -> set[str]:
    """The ids on a cycle: every member of a strongly connected component
    with more than one node, and every node that depends on itself.
    Tarjan's algorithm, iterative so a long chain cannot exhaust the stack."""
    visit_number: dict[str, int] = {}
    lowest_reachable: dict[str, int] = {}
    on_stack: set[str] = set()
    stack: list[str] = []
    cyclic: set[str] = set()
    counter = 0

    for root in dependencies:
        if root in visit_number:
            continue
        call_stack: list[tuple[str, list[str]]] = [(root, [dep for dep in dependencies[root] if dep in ids])]
        visit_number[root] = lowest_reachable[root] = counter
        counter += 1
        stack.append(root)
        on_stack.add(root)
        while call_stack:
            node, remaining = call_stack[-1]
            if remaining:
                dep = remaining.pop()
                if dep not in visit_number:
                    visit_number[dep] = lowest_reachable[dep] = counter
                    counter += 1
                    stack.append(dep)
                    on_stack.add(dep)
                    call_stack.append((dep, [d for d in dependencies[dep] if d in ids]))
                elif dep in on_stack:
                    lowest_reachable[node] = min(lowest_reachable[node], visit_number[dep])
                continue
            call_stack.pop()
            if call_stack:
                parent = call_stack[-1][0]
                lowest_reachable[parent] = min(lowest_reachable[parent], lowest_reachable[node])
            if lowest_reachable[node] == visit_number[node]:
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
