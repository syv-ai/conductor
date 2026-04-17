"""Simple topological layout for ReactFlow positions.

ReactFlow requires every node to have a position. Conductor doesn't track
positions. When a downstream caller converts a conductor graph to
ReactFlow JSON without positions, this module assigns coordinates so the
result renders as a readable left-to-right DAG.

The layout is deliberately naive — it's meant as a fallback, not as a
visual design. Hosts that want pretty layouts should run their own
engine (elk, dagre) on the output.
"""

from __future__ import annotations

from collections import defaultdict

from conductor.graph.model import GraphEdge, GraphNode


def topological_positions(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    *,
    x_gap: int = 260,
    y_gap: int = 120,
) -> dict[str, dict[str, int]]:
    """Assign each node a ``{x, y}`` position.

    The algorithm:
      1. Compute each node's longest-path depth from any root.
      2. Group nodes by depth — depth maps to an x-column.
      3. Within a column, spread nodes vertically in declaration order.

    Returns a dict keyed by node id.
    """
    depth: dict[str, int] = {n.id: 0 for n in nodes}
    forward: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {n.id: 0 for n in nodes}

    for e in edges:
        if e.source in depth and e.target in depth:
            forward[e.source].append(e.target)
            in_degree[e.target] += 1

    # Kahn's — process roots first so depth[target] = max(depth[sources]) + 1
    queue = [n.id for n in nodes if in_degree[n.id] == 0]
    while queue:
        nid = queue.pop(0)
        for neighbor in forward.get(nid, []):
            if depth[neighbor] < depth[nid] + 1:
                depth[neighbor] = depth[nid] + 1
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Bucket by depth and spread vertically within each bucket
    by_depth: dict[int, list[str]] = defaultdict(list)
    # Preserve declaration order when placing in columns
    for n in nodes:
        by_depth[depth[n.id]].append(n.id)

    positions: dict[str, dict[str, int]] = {}
    for d, ids in by_depth.items():
        for row, nid in enumerate(ids):
            positions[nid] = {"x": d * x_gap, "y": row * y_gap}
    return positions
