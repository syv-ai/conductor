"""Topological sort and cycle detection."""

from collections import defaultdict, deque

from conductor.errors import CycleDetectionError
from conductor.graph.model import GraphEdge, GraphNode


def topological_sort(nodes: list[GraphNode], edges: list[GraphEdge]) -> list[str]:
    """Topologically sort nodes based on edge dependencies.

    Uses Kahn's algorithm. Returns node IDs in execution order.
    Raises CycleDetectionError if the graph contains cycles.
    """
    node_ids = {n.id for n in nodes}

    # Build in-degree map and adjacency
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    dependents: dict[str, list[str]] = defaultdict(list)

    for edge in edges:
        if edge.target in node_ids and edge.source in node_ids:
            in_degree[edge.target] += 1
            dependents[edge.source].append(edge.target)

    # Deduplicate edges per (source, target) pair for in-degree counting
    # We need to recount because multiple edges between same pair shouldn't
    # increase in-degree multiple times for topological sort purposes
    seen_pairs: set[tuple[str, str]] = set()
    in_degree = {nid: 0 for nid in node_ids}
    dependents = defaultdict(list)

    for edge in edges:
        pair = (edge.source, edge.target)
        if pair not in seen_pairs and edge.source in node_ids and edge.target in node_ids:
            seen_pairs.add(pair)
            in_degree[edge.target] += 1
            dependents[edge.source].append(edge.target)

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
