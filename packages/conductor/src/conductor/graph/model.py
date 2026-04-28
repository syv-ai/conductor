"""Graph wire format — GraphNode and GraphEdge."""

from dataclasses import dataclass, field
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
        compensation: Optional id of another node that should run to "undo"
            this node's work if the flow fails after it completes. See
            docs/compensation.md for the saga semantics.
        on_error: Per-instance error policy override —
            ``"fail"`` (default: halt the flow), ``"continue"`` (treat the
            failure as success with a ``null`` result), or ``"compensate"``
            (trigger the compensation cascade). ``None`` means "use the flow
            default" (``"fail"``).
    """

    id: str
    type: str
    data: dict[str, Any] | None
    produces: dict[str, str] | None = None
    consumes: dict[str, tuple[str, str]] | None = None
    compensation: str | None = None
    on_error: str | None = None
    # Optional, host-defined display hints. Pure UX — the engine consults
    # them only when building human-readable labels (e.g. ConnectionList
    # aggregator keys); execution is unaffected when absent.
    node_label: str | None = None
    output_labels: dict[str, str] | None = None


@dataclass(frozen=True)
class GraphEdge:
    """An edge connecting two nodes.

    Attributes:
        when: Optional CEL expression evaluated at runtime to decide whether
            the edge is "taken". Only meaningful on outgoing edges from
            ``decision`` nodes. ``None`` means "else / fallback". Decision
            nodes must declare exactly one else edge and at least one
            guarded edge.
        priority: Explicit integer ordering for guard evaluation. Higher
            priority is evaluated first; ties fall back to edge-declaration
            order. Default ``0``.
    """

    id: str
    source: str
    target: str
    source_handle: str | None
    target_handle: str | None
    when: str | None = None
    priority: int = 0


@dataclass(frozen=True)
class FlowDependency:
    """A declared top-level dependency of a flow.

    Flows declare every external system they touch up-front so hosts can
    do dependency-aware things (credential injection, rate limiting, audit
    reports) without executing the flow.

    Attributes:
        id: Stable id used by node ``uses`` lists and dependency registries.
        kind: One of ``"api"``, ``"db"``, ``"queue"``, ``"subprocess"``,
            ``"notification"``, or any project-defined string.
        config: Arbitrary host-specific config (endpoint, auth method, etc.)
            — opaque to the engine.
    """

    id: str
    kind: str
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FlowTrigger:
    """A declared way to start a flow.

    The engine never fires triggers — it just records them so hosts can
    inspect them and wire external machinery (cron, webhook router, queue
    consumer).

    Attributes:
        id: Stable id referenced by logs and observability.
        kind: ``"manual"``, ``"schedule"``, ``"event"``, or ``"webhook"``.
        config: Kind-specific config (e.g. ``{"cron": "0 9 * * 1",
            "timezone": "UTC"}``). Opaque to the engine.
        input_map: Optional CEL expression that transforms the trigger's raw
            payload into the flow's inputs.
    """

    id: str
    kind: str
    config: dict[str, Any] = field(default_factory=dict)
    input_map: str | None = None


@dataclass(frozen=True)
class Flow:
    """A complete process definition — nodes, edges, and top-level metadata.

    This is the "flow file" in memory. The compiler can take either this
    rich shape or the raw ``nodes`` / ``edges`` lists.

    Attributes:
        nodes: The graph's nodes.
        edges: The graph's edges (may carry ``when`` guards on decision
            outputs).
        id: Optional stable id used for subprocess references.
        version: Optional version used for subprocess pinning. Defaults to
            ``1`` when a flow is referenced.
        name: Human-readable name.
        description: Free-form description.
        dependencies: Top-level dependency manifest (SOC2, rate limiting).
        triggers: Declarative triggers that can start this flow.
        on_error_default: Flow-level default for node ``on_error`` policy.
    """

    nodes: list[GraphNode]
    edges: list[GraphEdge]
    id: str | None = None
    version: int = 1
    name: str | None = None
    description: str | None = None
    dependencies: tuple[FlowDependency, ...] = ()
    triggers: tuple[FlowTrigger, ...] = ()
    on_error_default: str = "fail"
