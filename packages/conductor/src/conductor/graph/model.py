"""Graph wire format — GraphNode and GraphEdge."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GraphNode:
    """A node in the graph (matches ReactFlow's data model).

    Attributes:
        produces: Optional map of output_handle → display label. Presence of a
            handle in this dict marks the output as a shared reference that
            other nodes may consume. The label is UI-only; references are
            bound by identity (node_id, output_handle).
        consumes: Optional map of input_handle → (producer_node_id, output_handle).
            Declares that this input should be filled by the producer's shared
            output instead of (or in the absence of) a drawn edge.
    """

    id: str
    type: str
    data: dict[str, Any] | None
    produces: dict[str, str] | None = None
    consumes: dict[str, tuple[str, str]] | None = None


@dataclass(frozen=True)
class GraphEdge:
    """An edge connecting two nodes."""

    id: str
    source: str
    target: str
    source_handle: str | None
    target_handle: str | None
