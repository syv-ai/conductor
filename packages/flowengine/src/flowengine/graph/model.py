"""Graph wire format — GraphNode and GraphEdge."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    """A node in the graph (matches ReactFlow's data model)."""

    id: str
    type: str
    data: dict[str, Any] | None


@dataclass(frozen=True)
class GraphEdge:
    """An edge connecting two nodes."""

    id: str
    source: str
    target: str
    source_handle: str | None
    target_handle: str | None
