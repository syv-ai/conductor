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

from conductor.graph.model import GraphEdge, GraphNode
from pydantic import BaseModel, ConfigDict


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
    # Process-standard additions
    compensation: str | None = None
    on_error: str | None = None
    # Host-defined display hints — see GraphNode.node_label / output_labels.
    node_label: str | None = None
    output_labels: dict[str, str] | None = None


class EdgeInput(BaseModel):
    """A single edge in an ``ExecuteRequest``. Maps 1:1 to ``GraphEdge``."""

    model_config = ConfigDict(extra="ignore")

    id: str
    source: str
    target: str
    source_handle: str | None = None
    target_handle: str | None = None
    # Decision-node guards
    when: str | None = None
    priority: int = 0


class ExecuteRequest(BaseModel):
    """The POST body shared by ``/execute``, ``/execute-stream``, and ``/compile``."""

    model_config = ConfigDict(extra="ignore")

    nodes: list[NodeInput]
    edges: list[EdgeInput]
    # Optional precomputed node results, keyed by node id. Nodes listed here
    # are seeded as already-completed (the engine emits ``node_complete`` with
    # ``cached=True`` and skips running them), so a host can reuse outputs from
    # a previous run instead of recomputing the whole graph. Unset = run all.
    cache: dict[str, Any] | None = None

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
                compensation=n.compensation,
                on_error=n.on_error,
                node_label=n.node_label,
                output_labels=n.output_labels,
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
                when=e.when,
                priority=e.priority,
            )
            for e in self.edges
        ]
        return nodes, edges
