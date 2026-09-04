"""Graph compilation — validate and produce an immutable execution plan."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from conductor.errors import CompilationError
from conductor.graph.dynamic_inputs import resolve_node_inputs
from conductor.graph.dynamic_outputs import _resolve_in_order
from conductor.graph.model import GraphEdge, GraphNode
from conductor.graph.topology import build_edge_map, build_incoming_map, topological_sort
from conductor.metadata import Input, Output

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
    edge_map: dict[tuple[str, str], list[tuple[str, str, str]]]
    node_map: dict[str, GraphNode]
    registry: Any  # NodeRegistry
    extension_resolver: ExtensionResolver | None = None
    edges: tuple[GraphEdge, ...] = ()
    # target_id -> [(target_handle, source_id, source_handle, edge_id), ...]
    # Inverted edge view — faster than scanning edge_map per node.
    incoming_map: dict[str, list[tuple[str, str, str, str]]] = field(default_factory=dict)
    # Resolved outputs per node id — populated for every node in
    # ``execution_order``. For nodes without a ``compute_outputs`` hook
    # this is a copy of ``NodeDefinition.outputs``; for hook-driven nodes
    # it carries the dynamically derived shape. Extension nodes have an
    # empty tuple.
    node_outputs: dict[str, tuple[Output, ...]] = field(default_factory=dict)
    # Resolved inputs per node id — populated for every node. For nodes
    # without a ``compute_inputs`` hook this is a copy of
    # ``NodeDefinition.inputs``; for hook-driven nodes it carries the
    # dynamically derived roster. Extension nodes have an empty tuple.
    # The input resolver and validation-error labelling consult this in
    # preference to the static schema.
    node_inputs: dict[str, tuple[Input, ...]] = field(default_factory=dict)


def compile(
    nodes: list[GraphNode],
    edges: list[GraphEdge],
    registry: "NodeRegistry",
    *,
    extension_resolver: ExtensionResolver | None = None,
) -> CompiledGraph:
    """Validate and compile a graph into an immutable execution plan."""
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

    # 3. Resolve dynamic inputs. Order-free — an input roster depends on
    #    the node's own typed values alone.
    node_inputs = {
        node.id: resolve_node_inputs(node=node, node_def=registry.get(node.type))
        for node in nodes
    }

    # 4. Topological sort
    order = topological_sort(nodes, edges)

    # 5. Build edge maps — forward (for resolver) and inverted (for fast
    #    per-node incoming lookup).
    edge_map = build_edge_map(edges)
    incoming_map = build_incoming_map(edges)

    # 6. Resolve dynamic outputs in topological order. Each node sees its
    #    producers' already-resolved shapes (which may themselves be hook-
    #    driven). Nodes without a hook get a verbatim copy of their static
    #    ``NodeDefinition.outputs``. Extension nodes resolve to ``()``.
    node_outputs = _resolve_in_order(order=order, node_map=node_map, lookup=registry)

    return CompiledGraph(
        execution_order=tuple(order),
        edge_map=edge_map,
        node_map=node_map,
        registry=registry,
        extension_resolver=extension_resolver,
        edges=tuple(edges),
        incoming_map=incoming_map,
        node_outputs=node_outputs,
        node_inputs=node_inputs,
    )
