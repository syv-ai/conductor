"""``upgrade_node``: one node of a graph record moved to a newer version of its definition.

Compile never upgrades, so a graph pinned at an old version runs that
version; a person asks, and a host calls this. The record it returns is not
compiled here: a host compiles it to show the author what the new version
makes of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from conductor._internal.errors import InvalidCallError, make_problem
from conductor._internal.graph.graph_records import GraphRecord
from conductor._internal.ref import Ref
from conductor._internal.registry import NodeRegistry


def upgrade_node(graph: GraphRecord, node_id: str, registry: NodeRegistry, *, to: int | None = None) -> GraphRecord:
    """``graph`` with one node moved from the version it was saved at to ``to``, the newest by default.

    Runs each version's ``Upgrade`` in order and sets the version. Each step
    first moves what the record holds on the inputs it renames (values,
    defaults and the edges into them), then rewrites what is typed into the
    node, values and defaults as one map: a key that was a default stays
    one, and every other key the rewrite returns is a value, fixed as the
    author's. Every edge out of a renamed output, or out of a column of it,
    reads the new name. A node already at ``to`` returns ``graph`` itself. A version
    whose body is a graph has no steps, so the node only has its version set
    and compile reports any input the new version does not have. A node the
    graph does not have, a node definition the registry does not, and a
    version it can't move to are each ``InvalidCallError``.
    """
    node = next((graph_node for graph_node in graph.nodes if graph_node.id == node_id), None)
    if node is None:
        raise InvalidCallError([make_problem("node_not_in_graph", "node", node_id, node_id=node_id, nodes=", ".join(graph_node.id for graph_node in graph.nodes))])
    if node.node not in registry:
        raise InvalidCallError([make_problem("unknown_node_type", "node", node_id, node_type=node.node, choices=", ".join(sorted(definition.id for definition in registry)) or "none")])
    definition = registry[node.node]
    target = definition.current_version if to is None else to
    if target == node.version:
        return graph
    if target < node.version or target not in definition.versions or node.version not in definition.versions:
        raise InvalidCallError([make_problem(
            "cannot_move_to_version", "node", node_id,
            node_type=node.node, version=node.version, target=target, versions=", ".join(str(number) for number in sorted(definition.versions)),
        )])
    steps = [
        step
        for number in range(node.version + 1, target + 1)
        if number in definition.versions and (step := definition.versions[number].upgrade_from_previous) is not None
    ]
    values, defaults, edges = dict(node.values), dict(node.defaults), list(graph.edges)
    for step in steps:
        values, defaults = _with_keys_renamed(values, step.renamed_inputs), _with_keys_renamed(defaults, step.renamed_inputs)
        if step.rewrite is not None:
            rewritten = step.rewrite({**defaults, **values})
            defaults = {name: value for name, value in rewritten.items() if name in defaults and name not in values}
            values = {name: value for name, value in rewritten.items() if name not in defaults}
        edges = [
            (_renamed_on(source, node_id, step.renamed_outputs), _renamed_on(target_, node_id, step.renamed_inputs))
            for source, target_ in edges
        ]
    moved = node.model_copy(update={"version": target, "values": values, "defaults": defaults})
    return graph.model_copy(update={
        "nodes": tuple(moved if graph_node is node else graph_node for graph_node in graph.nodes),
        "edges": tuple(edges),
    })


def _with_keys_renamed(by_name: Mapping[str, Any], renames: Mapping[str, str]) -> dict[str, Any]:
    """``by_name`` with each key a step renames under its new name."""
    return {renames.get(name, name): value for name, value in by_name.items()}


def _renamed_on(address: str, node_id: str, renames: Mapping[str, str]) -> str:
    """``address`` with its field renamed when it is on ``node_id`` and the step renames that field; a column (``result.email``) moves with its field."""
    ref = Ref(address)
    field, dot, column = ref.field.partition(".")
    return Ref(node_id, renames[field] + dot + column) if ref.node_id == node_id and field in renames else ref
