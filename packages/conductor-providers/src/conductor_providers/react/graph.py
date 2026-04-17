"""Bidirectional conversion between conductor graphs and ReactFlow JSON."""

from __future__ import annotations

from typing import Any

from conductor.graph.model import GraphEdge, GraphNode

from conductor_providers.react.layout import topological_positions


def graph_to_react(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    *,
    positions: dict[str, dict[str, int]] | None = None,
) -> dict[str, Any]:
    """Serialize a conductor graph to a ReactFlow-compatible dict.

    ``positions`` lets a host pass existing coordinates (e.g. loaded from
    a previously-saved layout). When it's None or missing entries, missing
    positions are filled in via ``topological_positions``.
    """
    pos = dict(positions) if positions else {}
    # Fill in missing positions only — respect any the caller supplied.
    if any(n.id not in pos for n in nodes):
        auto = topological_positions(nodes, edges)
        for nid, xy in auto.items():
            pos.setdefault(nid, xy)

    rf_nodes: list[dict[str, Any]] = []
    for n in nodes:
        data: dict[str, Any] = {}
        if n.data is not None:
            data["data"] = n.data
        if n.produces:
            data["produces"] = dict(n.produces)
        if n.consumes:
            # Tuples become lists in JSON; consumers of this output must
            # call react_to_graph to get the tuples back.
            data["consumes"] = {k: list(v) for k, v in n.consumes.items()}

        rf_nodes.append({
            "id": n.id,
            "type": n.type,
            "position": pos.get(n.id, {"x": 0, "y": 0}),
            "data": data,
        })

    rf_edges: list[dict[str, Any]] = []
    for e in edges:
        rf_edges.append({
            "id": e.id,
            "source": e.source,
            "target": e.target,
            "sourceHandle": e.source_handle,
            "targetHandle": e.target_handle,
        })

    return {"nodes": rf_nodes, "edges": rf_edges}


def react_to_graph(
    flow: dict[str, Any],
) -> tuple[list[GraphNode], list[GraphEdge]]:
    """Parse a ReactFlow dict back into conductor GraphNode / GraphEdge objects.

    Accepts ``consumes`` values as either tuples or lists (JSON loses the
    tuple distinction) and normalizes them back to tuples for conductor.
    Unknown top-level keys are ignored so hosts can decorate the wire
    format without breaking the round-trip.
    """
    nodes_out: list[GraphNode] = []
    for raw in flow.get("nodes", []):
        payload = raw.get("data") or {}
        produces = payload.get("produces")
        consumes_raw = payload.get("consumes")
        consumes: dict[str, tuple[str, str]] | None = None
        if consumes_raw:
            consumes = {
                handle: (ref[0], ref[1]) for handle, ref in consumes_raw.items()
            }
        static_data = payload.get("data")

        nodes_out.append(
            GraphNode(
                id=raw["id"],
                type=raw["type"],
                data=static_data,
                produces=dict(produces) if produces else None,
                consumes=consumes,
            )
        )

    edges_out: list[GraphEdge] = []
    for raw in flow.get("edges", []):
        edges_out.append(
            GraphEdge(
                id=raw["id"],
                source=raw["source"],
                target=raw["target"],
                source_handle=raw.get("sourceHandle"),
                target_handle=raw.get("targetHandle"),
            )
        )

    return nodes_out, edges_out
