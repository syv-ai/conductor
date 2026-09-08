"""A ``Graph`` to ReactFlow JSON and back.

Each ReactFlow node carries the node record whole under ``data``
(``TypeAdapter(GraphNode)`` is the schema), a ``position`` the canvas
needs, and the canvas's own ``type``. The edges are derived from the
bindings, one per ref, for the canvas to draw; reading back ignores them,
since the bindings in ``data`` already say where every value comes from.
"""

from __future__ import annotations

from typing import Any

from conductor.graph.binding import Edges
from conductor.graph.model import Graph, GraphNode
from pydantic import TypeAdapter

from conductor_providers.react.layout import topological_positions

_NODE = TypeAdapter(GraphNode)


def graph_to_react(graph: Graph) -> dict[str, Any]:
    """Serialize ``graph`` to a ReactFlow-compatible dict.

    A node whose ``display`` holds a ``position`` keeps it; the rest
    are laid out left to right by ``topological_positions``.
    """
    auto = topological_positions(graph)
    rf_nodes = [
        {
            "id": node.id,
            "type": node.type,
            "position": node.display.get("position", auto[node.id]),
            "data": _NODE.dump_python(node, mode="json", exclude={"display"}),
        }
        for node in graph.nodes
    ]
    rf_edges = [
        {
            "id": f"{ref.node_id}.{ref.field}->{node.id}.{handle}",
            "source": ref.node_id,
            "target": node.id,
            "sourceHandle": ref.field,
            "targetHandle": handle,
        }
        for node in graph.nodes
        for handle, binding in node.bindings.items()
        if isinstance(binding, Edges)
        for ref in binding.refs
    ]
    return {"nodes": rf_nodes, "edges": rf_edges}


def react_to_graph(wire: dict[str, Any]) -> Graph:
    """Parse a ReactFlow dict back into a ``Graph``.

    Each node's ``data`` is the node record; the canvas's ``position``
    lands in the node's ``display``. Keys the canvas added beside
    those are ignored, so a host can decorate the wire without breaking
    the round trip.
    """
    return Graph(nodes=[
        _NODE.validate_python({**raw["data"], "display": {"position": raw["position"]}})
        for raw in wire["nodes"]
    ])
