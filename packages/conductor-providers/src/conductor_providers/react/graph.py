"""A ``Flow`` to ReactFlow JSON and back.

Each ReactFlow node carries the placement record whole under ``data``
(``TypeAdapter(GraphNode)`` is the schema), a ``position`` the canvas
needs, and the canvas's own ``type``. The cables are derived from the
bindings, one per ref, for the canvas to draw; reading back ignores them,
since the bindings in ``data`` already say where every value comes from.
"""

from __future__ import annotations

from typing import Any

from conductor.graph.binding import Sources
from conductor.graph.model import Flow, GraphNode
from pydantic import TypeAdapter

from conductor_providers.react.layout import topological_positions

_NODE = TypeAdapter(GraphNode)


def graph_to_react(flow: Flow) -> dict[str, Any]:
    """Serialize ``flow`` to a ReactFlow-compatible dict.

    A placement whose ``display`` holds a ``position`` keeps it; the rest
    are laid out left to right by ``topological_positions``.
    """
    auto = topological_positions(flow)
    rf_nodes = [
        {
            "id": node.id,
            "type": node.type,
            "position": node.display.get("position", auto[node.id]),
            "data": _NODE.dump_python(node, mode="json", exclude={"display"}),
        }
        for node in flow.nodes
    ]
    rf_edges = [
        {
            "id": f"{ref.node_id}.{ref.field}->{node.id}.{handle}",
            "source": ref.node_id,
            "target": node.id,
            "sourceHandle": ref.field,
            "targetHandle": handle,
        }
        for node in flow.nodes
        for handle, binding in node.bindings.items()
        if isinstance(binding, Sources)
        for ref in binding.refs
    ]
    return {"nodes": rf_nodes, "edges": rf_edges}


def react_to_graph(wire: dict[str, Any]) -> Flow:
    """Parse a ReactFlow dict back into a ``Flow``.

    Each node's ``data`` is the placement record; the canvas's ``position``
    lands in the placement's ``display``. Keys the canvas added beside
    those are ignored, so a host can decorate the wire without breaking
    the round trip.
    """
    return Flow(nodes=[
        _NODE.validate_python({**raw["data"], "display": {"position": raw["position"]}})
        for raw in wire["nodes"]
    ])
