"""``upgrade_node``: one node of a graph moved to a newer version of its definition.

Compile never upgrades, so a graph pinned at an old version runs that
version; a person asks, and a host calls this. The graph it returns is not
compiled here: a host compiles it to show the author what the new version
makes of it.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from conductor.errors import InvalidCallError, problem
from conductor.graph.binding import From, Static, static_values
from conductor.graph.graph_records import Graph, GraphNode
from conductor.node_decorator import NodeVersion, Upgrade
from conductor.ref import Ref
from conductor.registry import NodeRegistry


def upgrade_node(graph: Graph, node_id: str, registry: NodeRegistry, *, to: int | None = None) -> Graph:
    """``graph`` with one node moved from the version it was saved at to ``to``, the newest by default.

    Runs each version's ``Upgrade`` in order and sets the version. Each
    step first moves the inputs it renames (binding, lock and content),
    then rewrites the bindings, then moves the content of the outputs it
    renames; every edge elsewhere in the graph that read a renamed output
    reads the new name. A node already at ``to`` returns ``graph`` itself.
    A version whose body is a graph has no steps, so the node only has its
    version set and compile reports any binding the new version does not
    have. A node the graph does not have, a type the registry does not, and
    a version it can't move to are each ``InvalidCallError``.
    """
    node = next((n for n in graph.nodes if n.id == node_id), None)
    if node is None:
        raise InvalidCallError([problem("node_not_in_graph", "node", node_id, node=node_id, nodes=", ".join(n.id for n in graph.nodes))])
    if node.type not in registry:
        raise InvalidCallError([problem("unknown_node_type", "node", node_id, node_type=node.type, choices=", ".join(sorted(registry)) or "none")])
    definition = registry[node.type]
    target = definition.current if to is None else to
    if target == node.version:
        return graph
    if target < node.version or target not in definition.versions or node.version not in definition.versions:
        raise InvalidCallError([problem(
            "cannot_move_to_version", "node", node_id,
            node_type=node.type, version=node.version, target=target, versions=", ".join(str(n) for n in sorted(definition.versions)),
        )])
    steps = [
        step
        for number in range(node.version, target)
        if isinstance(version := definition.versions.get(number), NodeVersion) and (step := version.upgrade_to_next) is not None
    ]
    bindings, locked, fields = dict(node.bindings), node.locked, dict(node.fields)
    for step in steps:
        bindings, locked, fields = _renamed(bindings, step.renamed_inputs), _renamed_all(locked, step.renamed_inputs), _renamed(fields, step.renamed_inputs)
        if step.rewrite is not None:
            values = step.rewrite({**bindings, **static_values(bindings)})
            bindings = {name: value if isinstance(value, From) else Static(value) for name, value in values.items()}
        fields = _renamed(fields, step.renamed_outputs)
    moved = node.model_copy(update={"version": target, "bindings": bindings, "locked": locked, "fields": fields})
    return graph.model_copy(update={
        "nodes": tuple(moved if n is node else _reading_renamed(n, node_id, steps) for n in graph.nodes),
    })


def _renamed(by_name: Mapping[str, Any], renames: Mapping[str, str]) -> dict[str, Any]:
    """``by_name`` with each key a step renames under its new name."""
    return {renames.get(name, name): value for name, value in by_name.items()}


def _renamed_all(names: tuple[str, ...], renames: Mapping[str, str]) -> tuple[str, ...]:
    """``names`` with each one a step renames replaced by its new name."""
    return tuple(renames.get(name, name) for name in names)


def _reading_renamed(node: GraphNode, source: str, steps: list[Upgrade]) -> GraphNode:
    """``node`` with every edge it has from ``source`` pointing at the output's name after every step.

    Each ref is taken through the steps one by one, so a later step that
    reuses a name an earlier one renamed away does not catch it."""

    def through(field: str) -> str:
        for step in steps:
            field = step.renamed_outputs.get(field, field)
        return field

    bindings = {
        name: From(*(Ref(source, through(ref.field)) if ref.node_id == source else ref for ref in binding.refs)) if isinstance(binding, From) else binding
        for name, binding in node.bindings.items()
    }
    return node if bindings == dict(node.bindings) else node.model_copy(update={"bindings": bindings})
