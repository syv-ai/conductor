"""The edge facts of a graph, derived once from the bindings.

``dependencies_of`` is the dependency map — which nodes each node waits
for — and the one graph-wide reading of the edges; everything that needs
a whole-graph fact (the execution order, which nodes nothing consumes)
reads that map rather than walking the bindings again. ``edge_maps`` is
the per-input view the old engine's resolver and skip check still key on.
"""

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping

from conductor.errors import CycleDetectionError
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


def topological_sort(dependencies: Mapping[str, frozenset[str]]) -> list[str]:
    """Node ids in an order where every dependency precedes its dependent.

    Kahn's algorithm over the dependency map. Raises
    ``CycleDetectionError`` if the graph contains cycles. A dependency on
    an id the map does not hold is ignored here; compile reports it on
    the binding that names it.
    """
    ids = set(dependencies)
    in_degree = {i: len(deps & ids) for i, deps in dependencies.items()}
    dependents: dict[str, list[str]] = defaultdict(list)
    for i, deps in dependencies.items():
        for dep in deps & ids:
            dependents[dep].append(i)

    queue = deque(i for i, degree in in_degree.items() if degree == 0)
    result: list[str] = []
    while queue:
        i = queue.popleft()
        result.append(i)
        for dependent in dependents.get(i, []):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(ids):
        raise CycleDetectionError(f"Cycle detected involving nodes: {ids - set(result)}")

    return result


def edge_maps(
    nodes: Iterable[GraphNode],
) -> tuple[dict[tuple[str, str], list[tuple[str, str, str]]], dict[str, list[tuple[str, str, str, str]]]]:
    """The two per-input views of the edges the engine reads, from one pass.

    ``edge_map`` is ``(target_id, target_handle) -> [(source_id,
    source_handle, edge_id), ...]`` and ``incoming_map`` its inversion,
    ``target_id -> [(target_handle, source_id, source_handle, edge_id),
    ...]``, in ref order. The edge id is ``"source.handle->target.handle"``,
    derived and stored nowhere; the resolver's skip bookkeeping keys on it.
    """
    edge_map: dict[tuple[str, str], list[tuple[str, str, str]]] = defaultdict(list)
    incoming: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    for node in nodes:
        for handle, binding in node.bindings.items():
            if not isinstance(binding, Edges):
                continue
            for ref in binding.refs:
                edge_id = f"{ref.node_id}.{ref.field}->{node.id}.{handle}"
                edge_map[(node.id, handle)].append((ref.node_id, ref.field, edge_id))
                incoming[node.id].append((handle, ref.node_id, ref.field, edge_id))
    return dict(edge_map), dict(incoming)
