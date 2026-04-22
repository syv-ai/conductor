"""Canonical pydantic payloads for conductor's HTTP surface.

These are the long-lived contract between any conductor-backed server
and the clients that talk to it. Bumping them is a breaking change for
every host using ``conductor_router``.

The shapes deliberately mirror ``conductor.graph.model.GraphNode`` /
``GraphEdge`` but use snake_case field names — that's the canonical
wire format. Framework-specific adapters (ReactFlow's camelCase) live
in ``conductor_providers.react``.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from conductor.graph.model import GraphEdge, GraphNode


class NodeInput(BaseModel):
    """A single node in an ``ExecuteRequest``. Maps 1:1 to ``GraphNode``."""

    model_config = ConfigDict(extra="ignore")

    id: str
    type: str
    data: dict[str, Any] | None = None
    produces: dict[str, str] | None = None
    # JSON has no tuples — inbound consumes use ``[producer_id, output_handle]``
    # lists; we normalize to tuples when converting to ``GraphNode``.
    consumes: dict[str, tuple[str, str]] | None = None


class EdgeInput(BaseModel):
    """A single edge in an ``ExecuteRequest``. Maps 1:1 to ``GraphEdge``."""

    model_config = ConfigDict(extra="ignore")

    id: str
    source: str
    target: str
    source_handle: str | None = None
    target_handle: str | None = None


class ExecuteRequest(BaseModel):
    """The POST body shared by ``/execute``, ``/execute-stream``, and ``/compile``."""

    model_config = ConfigDict(extra="ignore")

    nodes: list[NodeInput]
    edges: list[EdgeInput]

    def to_graph(self) -> tuple[list[GraphNode], list[GraphEdge]]:
        """Convert to conductor's internal ``GraphNode`` / ``GraphEdge`` types.

        Single source of truth for the payload → engine conversion — every
        host that used to hand-roll a ``_build_graph`` helper now delegates
        to this method.
        """
        nodes = [
            GraphNode(
                id=n.id,
                type=n.type,
                data=n.data,
                produces=n.produces or None,
                consumes=(
                    {k: (v[0], v[1]) for k, v in n.consumes.items()}
                    if n.consumes
                    else None
                ),
            )
            for n in self.nodes
        ]
        edges = [
            GraphEdge(
                id=e.id,
                source=e.source,
                target=e.target,
                source_handle=e.source_handle,
                target_handle=e.target_handle,
            )
            for e in self.edges
        ]
        return nodes, edges
