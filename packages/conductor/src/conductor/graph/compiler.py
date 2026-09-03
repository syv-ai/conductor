"""Graph compilation — validate and produce an immutable execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from conductor.errors import CompilationError
from conductor.graph.dynamic_inputs import resolve_node_inputs
from conductor.graph.dynamic_outputs import _resolve_in_order
from conductor.graph.model import Flow, GraphEdge, GraphNode
from conductor.graph.shared_refs import validate_and_build_consume_map
from conductor.graph.topology import build_edge_map, build_incoming_map, topological_sort
from conductor.metadata import Input, Output

if TYPE_CHECKING:
    from conductor.registry import NodeRegistry

__all__ = [
    "compile",
    "CompiledGraph",
    "ExtensionResolver",
]


class ExtensionResolver(Protocol):
    """Implemented by host applications for custom node types."""

    def is_known_type(self, node_type: str) -> bool: ...
    def create_executor(self, node_type: str) -> Any: ...



@dataclass(frozen=True)
class CompiledGraph:
    """Immutable, validated, ready-to-execute graph."""

    execution_order: tuple[str, ...]
    edge_map: dict[tuple[str, str], list[tuple[str, str, str]]]
    node_map: dict[str, GraphNode]
    registry: Any  # NodeRegistry
    extension_resolver: ExtensionResolver | None = None
    compound_nodes: dict[str, Any] = field(default_factory=dict)
    managed_ids: frozenset[str] = field(default_factory=frozenset)
    # (target_id, target_handle) -> (producer_id, output_handle)
    consume_map: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)
    # managed_node_id -> its region's start_id (for scheduling redirection)
    managed_to_region_start: dict[str, str] = field(default_factory=dict)
    edges: tuple[GraphEdge, ...] = ()
    # Flow-level metadata
    flow: Flow | None = None
    # Nodes that are only ever dispatched during compensation (they're the
    # target of a node's ``compensation=`` field). Excluded from normal
    # scheduling.
    compensation_node_ids: frozenset[str] = field(default_factory=frozenset)
    # target_id -> [(target_handle, source_id, source_handle, edge_id), ...]
    # Inverted edge view — faster than scanning edge_map per node.
    incoming_map: dict[str, list[tuple[str, str, str, str]]] = field(default_factory=dict)
    # Resolved outputs per node id — populated for every node in
    # ``execution_order``. For nodes without a ``compute_outputs`` hook
    # this is a copy of ``NodeDefinition.outputs``; for hook-driven nodes
    # it carries the dynamically derived shape. Extension nodes have an
    # empty tuple. Type-checking, shared-ref validation, and compound
    # runtimes consult this in preference to the static schema.
    node_outputs: dict[str, tuple[Output, ...]] = field(default_factory=dict)
    # Resolved inputs per node id — populated for every node. For nodes
    # without a ``compute_inputs`` hook this is a copy of
    # ``NodeDefinition.inputs``; for hook-driven nodes it carries the
    # dynamically derived roster. Extension nodes have an empty tuple.
    # Type-checking, consume validation, the input resolver and
    # validation-error labelling consult this in preference to the static
    # schema.
    node_inputs: dict[str, tuple[Input, ...]] = field(default_factory=dict)


def compile(
    nodes: list[GraphNode] | None = None,
    edges: list[GraphEdge] | None = None,
    registry: "NodeRegistry" = None,
    *,
    compound_types: list[Any] | None = None,
    extension_resolver: ExtensionResolver | None = None,
    flow: Flow | None = None,
    subprocess_registry: Any = None,
) -> CompiledGraph:
    """Validate and compile a graph into an immutable execution plan.

    Accepts either a ``Flow`` via ``flow=`` or the traditional ``nodes``
    + ``edges`` args. ``subprocess_registry`` is forwarded to subprocess
    nodes so they can look up their target flow by id.
    """
    if flow is not None:
        nodes = flow.nodes
        edges = flow.edges
    if nodes is None or edges is None:
        raise TypeError("compile() needs either `flow=` or both `nodes=` and `edges=`")
    if registry is None:
        raise TypeError("compile() needs a `registry`")

    node_map = {n.id: n for n in nodes}

    # 1. Validate node types
    for node in nodes:
        known = registry.contains(node.type)
        if not known and extension_resolver:
            known = extension_resolver.is_known_type(node.type)
        if not known:
            raise CompilationError(f"Unknown node type: '{node.type}'")

    # 2. Validate edges reference existing nodes
    for edge in edges:
        if edge.source not in node_map:
            raise CompilationError(
                f"Edge '{edge.id}' references non-existent source node: '{edge.source}'"
            )
        if edge.target not in node_map:
            raise CompilationError(
                f"Edge '{edge.id}' references non-existent target node: '{edge.target}'"
            )

    # 3. Validate compensation references
    _validate_compensation(nodes, node_map)

    # 5. Discover compound regions first, so shared-ref validation can know
    #    which nodes are inside them (producers inside regions are rejected).
    compound_nodes: dict[str, Any] = {}
    managed_ids: set[str] = set()

    if compound_types:
        from conductor.graph.regions import discover_regions

        for ct, region in discover_regions(nodes, edges, compound_types):
            compound_nodes[region.start_id] = ct  # placeholder; rebuilt below
            managed_ids.update(region.body_ids)
            # Only mark end as managed when it's a separate node — a single
            # node subprocess region uses start_id == end_id and must remain
            # schedulable.
            if region.end_id != region.start_id:
                managed_ids.add(region.end_id)

    # 5b. Resolve dynamic inputs. Order-free — an input roster depends on
    #     the node's own ``data`` alone — but it must precede step 6 and
    #     step 10, both of which validate handles against declared inputs.
    node_inputs = {
        node.id: resolve_node_inputs(node=node, node_def=registry.get(node.type))
        for node in nodes
    }

    # 6. Validate shared references (produce/consume), build consume map
    consume_map = validate_and_build_consume_map(
        nodes, edges, node_map, frozenset(managed_ids), registry,
        node_inputs=node_inputs,
    )

    # 7. Topological sort — edges + consume dependencies participate equally
    extra_deps = [
        (producer_id, target_id)
        for (target_id, _), (producer_id, _) in consume_map.items()
    ]
    order = topological_sort(nodes, edges, extra_dependencies=extra_deps)

    # 8. Build edge maps — forward (for resolver) and inverted (for fast
    #    per-node incoming lookup).
    edge_map = build_edge_map(edges)
    incoming_map = build_incoming_map(edges)

    # 8b. Resolve dynamic outputs in topological order. Each node sees its
    #     producers' already-resolved shapes (which may themselves be hook-
    #     driven). Nodes without a hook get a verbatim copy of their static
    #     ``NodeDefinition.outputs``. Extension nodes resolve to ``()``.
    #     The walk is the shared ``_resolve_in_order`` engine — the public
    #     ``resolve_graph_outputs`` is its other caller, so the two can
    #     never diverge. ``order`` here honours consume dependencies too
    #     (a superset constraint; resolution reads drawn edges only).
    node_outputs = _resolve_in_order(order=order, node_map=node_map, lookup=registry)

    # 8c. Re-validate producer handles against resolved outputs — a
    #     ``compute_outputs`` hook may legitimately introduce a handle that
    #     a node then publishes via ``produces``. The first pass above
    #     ran without resolved outputs; we keep the structural errors it
    #     surfaced (managed-region rejection, structural integrity) and now
    #     supplement with handle-existence checks against the post-hook map.
    validate_and_build_consume_map(
        nodes, edges, node_map, frozenset(managed_ids), registry,
        node_outputs=node_outputs,
        node_inputs=node_inputs,
    )

    # 9. Now that we have the topological order, rebuild the compound node
    #    executors with the proper order (matching pre-refactor behavior) and
    #    build the managed-node → region-start lookup used for scheduling.
    compound_nodes = {}
    managed_to_region_start: dict[str, str] = {}
    if compound_types:
        from conductor.graph.regions import discover_regions

        for ct, region in discover_regions(nodes, edges, compound_types):
            executor = ct.factory(region, tuple(order))
            # Allow the factory to set the subprocess registry if supported.
            if subprocess_registry is not None and hasattr(executor, "set_subprocess_registry"):
                executor.set_subprocess_registry(subprocess_registry)
            compound_nodes[region.start_id] = executor
            for body_id in region.body_ids:
                managed_to_region_start[body_id] = region.start_id
            if region.end_id != region.start_id:
                managed_to_region_start[region.end_id] = region.start_id

    # Compensation nodes should never run as regular nodes — the engine
    # only dispatches them via ``_run_compensation``.
    compensation_node_ids = frozenset(
        n.compensation for n in nodes if n.compensation is not None
    )

    return CompiledGraph(
        execution_order=tuple(order),
        edge_map=edge_map,
        node_map=node_map,
        registry=registry,
        extension_resolver=extension_resolver,
        compound_nodes=compound_nodes,
        managed_ids=frozenset(managed_ids),
        consume_map=consume_map,
        managed_to_region_start=managed_to_region_start,
        edges=tuple(edges),
        flow=flow,
        compensation_node_ids=compensation_node_ids,
        incoming_map=incoming_map,
        node_outputs=node_outputs,
        node_inputs=node_inputs,
    )



def _validate_compensation(
    nodes: list[GraphNode],
    node_map: dict[str, GraphNode],
) -> None:
    """Every ``compensation=`` reference must point at an existing node."""
    for node in nodes:
        if node.compensation is None:
            continue
        if node.compensation not in node_map:
            raise CompilationError(
                f"Node '{node.id}' declares compensation='{node.compensation}' "
                f"but no such node exists in the flow."
            )
        if node.compensation == node.id:
            raise CompilationError(
                f"Node '{node.id}' cannot be its own compensation."
            )
        if node.on_error and node.on_error not in ("fail", "continue", "compensate"):
            raise CompilationError(
                f"Node '{node.id}' has invalid on_error='{node.on_error}'. "
                f"Valid values: fail, continue, compensate."
            )

    # Validate on_error on nodes without compensation as well
    for node in nodes:
        if node.on_error and node.on_error not in ("fail", "continue", "compensate"):
            raise CompilationError(
                f"Node '{node.id}' has invalid on_error='{node.on_error}'. "
                f"Valid values: fail, continue, compensate."
            )
