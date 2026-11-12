"""Moving one node of a graph record to a newer version of its definition: ``registry.upgrade(record, "split")``.

``split`` at version 1 takes ``sep``; version 2 takes ``separator`` and says ``Upgrade(renamed_inputs={"sep":
"separator"})``. Upgrading a record whose ``split`` is at version 1 moves its constant, default and edges from ``sep``
to ``separator`` and sets the version to 2.
"""

from __future__ import annotations

from collections.abc import Mapping

from conductor._internal.base import split_address
from conductor._internal.gate import RefusedError, read_record, record_values
from conductor._internal.node import NodeDefinition, declared_inputs
from conductor._internal.problem import InvalidCallError
from conductor._internal.problem import catalogue as problem
from conductor._internal.record import GraphRecord


def _with_keys_renamed[V](by_name: Mapping[str, V], renames: Mapping[str, str]) -> dict[str, V]:
    """``by_name`` with each key a step renames under its new name."""
    return {renames.get(name, name): value for name, value in by_name.items()}


def _renamed_on(address: str, node_id: str, renames: Mapping[str, str]) -> str:
    """``address`` with its field renamed when it is on ``node_id`` and the step renames it.

    A field of a record named after it (``result.email``) moves with it. A subgraph's output is named by an
    inner address (``tail.result``), so the longest renamed name the address goes on with is the one it names.
    """
    owner, rest = split_address(address)
    if owner != node_id:
        return address
    for name in sorted(renames, key=len, reverse=True):
        if rest == name or rest.startswith(name + "."):
            return f"{owner}.{renames[name]}{rest[len(name):]}"
    return address


def upgraded(nodes: Mapping[str, NodeDefinition], record: GraphRecord | str | bytes, node_id: str, *, to: int | None) -> GraphRecord:
    """``record`` with the node ``node_id`` moved to version ``to`` of its definition in ``nodes``, the newest when
    ``to`` is ``None``: what ``registry.upgrade`` returns, and its docstring says how."""
    try:
        graph = read_record(record)
    except RefusedError as refused:
        raise InvalidCallError(refused.problems) from None
    node_record = next((n for n in graph.nodes if n.id == node_id), None)
    if node_record is None:
        raise InvalidCallError([problem.node_not_in_graph(node_id, nodes=[n.id for n in graph.nodes])])
    if node_record.node not in nodes:
        raise InvalidCallError([problem.unknown_node(node_id, node=node_record.node, choices=sorted(nodes))])
    newest = nodes[node_record.node]
    target = newest.version if to is None else to
    if target == node_record.version:
        return graph
    if target < node_record.version or target not in newest.versions or node_record.version not in newest.versions:
        raise InvalidCallError([problem.cannot_move_to_version(
            node_id,
            node=node_record.node, version=node_record.version, target=target, versions=sorted(newest.versions),
        )])
    version_chain = newest.version_of(target).version_chain
    steps = [(newer, older) for newer, older in zip(version_chain, version_chain[1:], strict=False) if newer.version > node_record.version]
    constants, defaults, edges = dict(node_record.constants), dict(node_record.defaults), list(graph.edges)
    for step, older in reversed(steps):
        upgrade = step.upgrade
        constants, defaults = _with_keys_renamed(constants, upgrade.renamed_inputs), _with_keys_renamed(defaults, upgrade.renamed_inputs)
        if upgrade.rewrite is not None:
            try:
                rewritten = record_values(node_id, upgrade.rewrite({**defaults, **constants}))
            except RefusedError as refused:
                raise InvalidCallError(refused.problems) from None
            defaults = {name: value for name, value in rewritten.items() if name in defaults and name not in constants}
            constants = {name: value for name, value in rewritten.items() if name not in defaults}
        gone = {f"{node_id}.{name}" for name in declared_inputs(older).keys() - declared_inputs(step).keys() - upgrade.renamed_inputs.keys()}
        edges = [
            (_renamed_on(source, node_id, upgrade.renamed_outputs), _renamed_on(target_, node_id, upgrade.renamed_inputs))
            for source, target_ in edges
            if target_ not in gone
        ]
    moved = node_record.model_copy(update={"version": target, "constants": constants, "defaults": defaults})
    return GraphRecord(nodes=[moved if n is node_record else n for n in graph.nodes], edges=edges)
