"""Compile-time resolution of a placement's outputs through compute_outputs.

Walks each node in topological order and asks a fresh instance of its
definition for the outputs this placement has, given the pinned version's
declared outputs and the values the author typed. What arrives on each
wired input is not recorded yet, so the hook is handed an empty
arriving; the compiler records arrivals when it derives bindings. The
result is stored on CompiledGraph.node_outputs; downstream stages
(shared refs, compound runtimes) consult that map in preference to the
declaration. Nothing is checked here: a hook that returns the wrong shape
is a node bug and raises where it is found.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Protocol

from conductor.errors import CompilationError
from conductor.graph.topology import topological_sort
from conductor.metadata import Output

if TYPE_CHECKING:
    from conductor.graph.model import GraphEdge, GraphNode
    from conductor.node import NodeDefinition


class _DefinitionGet(Protocol):
    """The one lookup shape the resolution walk needs: get by type.

    Structurally satisfied by NodeRegistry, RegistryView and the
    mapping adapter below — the walk neither knows nor cares which.
    """

    def get(self, node_type: str) -> "type[NodeDefinition] | None": ...


def resolve_node_outputs(
    node: "GraphNode",
    node_def: "type[NodeDefinition] | None",
) -> tuple[Output, ...]:
    """The outputs this placement actually exposes.

    Extension nodes (no definition) resolve to an empty tuple.
    """
    if node_def is None:
        return ()
    declared = node_def.versions[node.version].interface.outputs
    return tuple(node_def().compute_outputs(declared, node.data or {}, {}))


class _MappingLookup:
    """_DefinitionGet over a host-supplied definitions mapping."""

    def __init__(self, definitions: Mapping[str, "type[NodeDefinition] | None"]) -> None:
        self._definitions = definitions

    def get(self, node_type: str) -> "type[NodeDefinition] | None":
        return self._definitions.get(node_type)


def _resolve_in_order(
    order: list[str],
    node_map: "dict[str, GraphNode]",
    lookup: _DefinitionGet,
) -> dict[str, tuple[Output, ...]]:
    """The ONE resolution walk — both compile() and
    :func:`resolve_graph_outputs` are callers.

    order must be topological over the drawn edges, so that when the
    compiler records arrivals every producer is resolved before its
    consumer.
    """
    resolved: dict[str, tuple[Output, ...]] = {}
    for node_id in order:
        node = node_map[node_id]
        resolved[node_id] = resolve_node_outputs(node, lookup.get(node.type))
    return resolved


def resolve_graph_outputs(
    nodes: "list[GraphNode]",
    edges: "list[GraphEdge]",
    definitions: Mapping[str, "type[NodeDefinition] | None"],
) -> dict[str, tuple[Output, ...]]:
    """Resolve every node's effective outputs in topological order.

    The ahead-of-compile entry to the same walk compile() runs.
    compile() answers *executability*; this answers the weaker "what
    fields does each node expose right now" — well-defined on graphs that
    are not yet executable (a draft mid-edit), which is exactly when hosts
    need it.

    definitions is **required** and keyed by node *type*: the host
    resolves each type however it wants. A None value means "known but
    definition-less" (extension semantics — resolves to ()); a
    *missing key* is a host bug and raises. Every edge endpoint must
    reference an existing node, and the graph must be acyclic.
    """
    node_map = {n.id: n for n in nodes}

    for node in nodes:
        if node.type not in definitions:
            raise CompilationError(f"Unknown node type: '{node.type}'")

    for edge in edges:
        if edge.source not in node_map:
            raise CompilationError(
                f"Edge '{edge.id}' references non-existent source node: "
                f"'{edge.source}'"
            )
        if edge.target not in node_map:
            raise CompilationError(
                f"Edge '{edge.id}' references non-existent target node: "
                f"'{edge.target}'"
            )

    return _resolve_in_order(
        order=topological_sort(nodes, edges),
        node_map=node_map,
        lookup=_MappingLookup(definitions),
    )
