"""Topological sort and cycle detection."""

from collections import defaultdict, deque

from conductor.errors import CycleDetectionError
from conductor.graph.model import GraphEdge, GraphNode


def topological_sort(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    *,
    extra_dependencies: list[tuple[str, str]] | None = None,
) -> list[str]:
    """Topologically sort nodes based on edge (and optional extra) dependencies.

    Uses Kahn's algorithm. Returns node IDs in execution order.
    Raises CycleDetectionError if the graph contains cycles.

    ``extra_dependencies`` is a list of ``(source, target)`` pairs representing
    dependencies that are not explicit edges — shared reference consume
    bindings are the primary use. They participate in in-degree counting and
    cycle detection identically to drawn edges.
    """
    node_ids = {n.id for n in nodes}

    seen_pairs: set[tuple[str, str]] = set()
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    dependents: dict[str, list[str]] = defaultdict(list)

    def _add(source: str, target: str) -> None:
        if source not in node_ids or target not in node_ids:
            return
        pair = (source, target)
        if pair in seen_pairs:
            return
        seen_pairs.add(pair)
        in_degree[target] += 1
        dependents[source].append(target)

    for edge in edges:
        _add(edge.source, edge.target)

    for source, target in extra_dependencies or []:
        _add(source, target)

    queue = deque(nid for nid, deg in in_degree.items() if deg == 0)
    result: list[str] = []

    while queue:
        nid = queue.popleft()
        result.append(nid)
        for dep in dependents.get(nid, []):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    if len(result) != len(node_ids):
        cycle_nodes = node_ids - set(result)
        raise CycleDetectionError(f"Cycle detected involving nodes: {cycle_nodes}")

    return result


def build_edge_map(
    edges: list[GraphEdge],
) -> dict[tuple[str, str], list[tuple[str, str]]]:
    """Build (target_id, target_handle) -> [(source_id, source_handle), ...] map."""
    edge_map: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    for edge in edges:
        key = (edge.target, edge.target_handle or "")
        value = (edge.source, edge.source_handle or "result")
        edge_map[key].append(value)
    return dict(edge_map)
