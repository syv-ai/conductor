"""Graph compilation — validate and produce an immutable execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from conductor.errors import CompilationError
from conductor.graph.model import GraphEdge, GraphNode
from conductor.graph.shared_refs import validate_and_build_consume_map
from conductor.graph.topology import build_edge_map, topological_sort
from conductor.graph.type_check import TypeWarning, check_consume_types, check_edge_types

if TYPE_CHECKING:
    from conductor.registry import NodeRegistry


class ExtensionResolver(Protocol):
    """Implemented by host applications for custom node types."""

    def is_known_type(self, node_type: str) -> bool: ...
    def create_executor(self, node_type: str) -> Any: ...


@dataclass(frozen=True)
class CompiledGraph:
    """Immutable, validated, ready-to-execute graph."""

    execution_order: tuple[str, ...]
    edge_map: dict[tuple[str, str], list[tuple[str, str]]]
    node_map: dict[str, GraphNode]
    registry: Any  # NodeRegistry
    extension_resolver: ExtensionResolver | None = None
    compound_nodes: dict[str, Any] = field(default_factory=dict)
    managed_ids: frozenset[str] = field(default_factory=frozenset)
    type_warnings: tuple[TypeWarning, ...] = field(default_factory=tuple)
    # (target_id, target_handle) -> (producer_id, output_handle)
    consume_map: dict[tuple[str, str], tuple[str, str]] = field(default_factory=dict)
    # managed_node_id -> its region's start_id (for scheduling redirection)
    managed_to_region_start: dict[str, str] = field(default_factory=dict)


def compile(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    registry: NodeRegistry,
    *,
    compound_types: list[Any] | None = None,
    extension_resolver: ExtensionResolver | None = None,
    strict_types: bool = False,
) -> CompiledGraph:
    """Validate and compile a graph into an immutable execution plan.

    Args:
        strict_types: If True, type mismatches raise CompilationError.
                      If False (default), they're returned as warnings
                      on CompiledGraph.type_warnings.
    """
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

    # 3. Discover compound regions first, so shared-ref validation can know
    #    which nodes are inside them (producers inside regions are rejected).
    compound_nodes: dict[str, Any] = {}
    managed_ids: set[str] = set()

    if compound_types:
        from conductor.graph.regions import discover_regions

        for ct, region in discover_regions(nodes, edges, compound_types):
            # We don't have the topological order yet — pass an empty tuple;
            # the factory re-derives ordering from body_ids as needed.
            compound_nodes[region.start_id] = ct  # placeholder; rebuilt below
            managed_ids.update(region.body_ids)
            managed_ids.add(region.end_id)

    # 4. Validate shared references (produce/consume), build consume map
    consume_map, shared_warnings = validate_and_build_consume_map(
        nodes, edges, node_map, frozenset(managed_ids), registry,
    )

    # 5. Topological sort — edges + consume dependencies participate equally
    extra_deps = [
        (producer_id, target_id)
        for (target_id, _), (producer_id, _) in consume_map.items()
    ]
    order = topological_sort(nodes, edges, extra_dependencies=extra_deps)

    # 6. Build edge map
    edge_map = build_edge_map(edges)

    # 7. Now that we have the topological order, rebuild the compound node
    #    executors with the proper order (matching pre-refactor behavior) and
    #    build the managed-node → region-start lookup used for scheduling.
    compound_nodes = {}
    managed_to_region_start: dict[str, str] = {}
    if compound_types:
        from conductor.graph.regions import discover_regions

        for ct, region in discover_regions(nodes, edges, compound_types):
            executor = ct.factory(region, tuple(order))
            compound_nodes[region.start_id] = executor
            for body_id in region.body_ids:
                managed_to_region_start[body_id] = region.start_id
            managed_to_region_start[region.end_id] = region.start_id

    # 8. Type-check edges and consume bindings
    edge_warnings = check_edge_types(edges, node_map, registry)
    consume_warnings = check_consume_types(consume_map, node_map, registry)
    type_warnings = [*edge_warnings, *consume_warnings, *shared_warnings]

    # Strict mode promotes only real mismatches (not informational warnings
    # like label collisions) to an error.
    strict_fatal = [w for w in type_warnings if w.code == "type-mismatch"]
    if strict_types and strict_fatal:
        messages = [w.message for w in strict_fatal]
        raise CompilationError(
            f"Type errors in {len(strict_fatal)} connection(s):\n"
            + "\n".join(f"  - {m}" for m in messages)
        )

    return CompiledGraph(
        execution_order=tuple(order),
        edge_map=edge_map,
        node_map=node_map,
        registry=registry,
        extension_resolver=extension_resolver,
        compound_nodes=compound_nodes,
        managed_ids=frozenset(managed_ids),
        type_warnings=tuple(type_warnings),
        consume_map=consume_map,
        managed_to_region_start=managed_to_region_start,
    )
