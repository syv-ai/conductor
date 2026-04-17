"""Region discovery for compound nodes."""

from collections import defaultdict, deque

from conductor.compound.protocol import CompoundNodeType, Region
from conductor.errors import CompilationError
from conductor.graph.model import GraphEdge, GraphNode


def discover_regions(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    compound_types: list[CompoundNodeType],
) -> list[tuple[CompoundNodeType, Region]]:
    """Discover all compound node regions in the graph."""
    results: list[tuple[CompoundNodeType, Region]] = []
    for ct in compound_types:
        regions = ct.discover(nodes, edges)
        for region in regions:
            results.append((ct, region))
    return results


def discover_for_each_regions(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
) -> list[Region]:
    """Discover for-each loop regions by BFS from start to end nodes."""
    forward: dict[str, list[str]] = defaultdict(list)
    for edge in edges:
        forward[edge.source].append(edge.target)

    node_type_map = {n.id: n.type for n in nodes}
    start_ids = [n.id for n in nodes if n.type.startswith("for-each-start")]
    end_ids = {n.id for n in nodes if n.type.startswith("for-each-end")}
    matched_ends: set[str] = set()

    regions: list[Region] = []
    for start_id in start_ids:
        visited: set[str] = set()
        queue = deque(forward.get(start_id, []))
        found_end: str | None = None
        body: set[str] = set()

        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)

            if current in end_ids:
                found_end = current
                continue

            body.add(current)
            for neighbor in forward.get(current, []):
                queue.append(neighbor)

        if found_end is None:
            raise CompilationError(
                f"For-each start node '{start_id}' has no matching end node."
            )

        matched_ends.add(found_end)
        regions.append(Region(
            start_id=start_id,
            end_id=found_end,
            body_ids=frozenset(body),
        ))

    orphan_ends = end_ids - matched_ends
    if orphan_ends:
        raise CompilationError(
            f"For-each end node(s) {orphan_ends} have no matching start node."
        )

    return regions
