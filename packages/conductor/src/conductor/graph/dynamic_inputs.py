"""Compile-time resolution of a placement's inputs through compute_inputs.

Every placement is asked once, on a fresh instance of its definition,
with the pinned version's declared inputs and the values the author
typed. A node with no override answers with the declaration. Nothing is
checked here: a hook that returns the wrong shape is a node bug and
raises where it is found.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from conductor.metadata import Input

if TYPE_CHECKING:
    from conductor.graph.model import GraphNode
    from conductor.node import NodeDefinition


def resolve_node_inputs(
    *,
    node: GraphNode,
    node_def: type[NodeDefinition] | None,
) -> tuple[Input, ...]:
    """The input roster this placement actually exposes.

    Extension nodes (no definition) resolve to an empty tuple, matching
    resolve_node_outputs.
    """
    if node_def is None:
        return ()
    declared = node_def.versions[node.version].interface.inputs
    return tuple(node_def().compute_inputs(declared, node.data or {}))


def resolve_graph_inputs(
    nodes: list[GraphNode],
    definitions: dict[str, type[NodeDefinition] | None],
) -> dict[str, tuple[Input, ...]]:
    """Resolve every node's input roster.

    The public counterpart to :func:`conductor.resolve_graph_outputs`, for
    hosts that need the resolved shape without compiling.
    """
    return {
        node.id: resolve_node_inputs(node=node, node_def=definitions.get(node.type))
        for node in nodes
    }
