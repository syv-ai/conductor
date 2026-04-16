"""Graph compilation — validate and produce an immutable execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from flowengine.errors import CompilationError
from flowengine.graph.model import GraphEdge, GraphNode
from flowengine.graph.topology import build_edge_map, topological_sort
from flowengine.graph.type_check import TypeWarning, check_edge_types

if TYPE_CHECKING:
    from flowengine.registry import NodeRegistry


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

    # 3. Topological sort (raises CycleDetectionError on cycles)
    order = topological_sort(nodes, edges)

    # 4. Build edge map
    edge_map = build_edge_map(edges)

    # 5. Compound node discovery
    compound_nodes: dict[str, Any] = {}
    managed_ids: set[str] = set()

    if compound_types:
        from flowengine.graph.regions import discover_regions

        for ct, region in discover_regions(nodes, edges, compound_types):
            executor = ct.factory(region, tuple(order))
            compound_nodes[region.start_id] = executor
            # Body and end nodes are managed by the compound node
            managed_ids.update(region.body_ids)
            managed_ids.add(region.end_id)

    # 6. Type-check edges
    type_warnings = check_edge_types(edges, node_map, registry)

    if strict_types and type_warnings:
        messages = [w.message for w in type_warnings]
        raise CompilationError(
            f"Type errors in {len(type_warnings)} edge(s):\n" + "\n".join(f"  - {m}" for m in messages)
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
    )
