"""Moving one node of a graph record to a newer version of its definition: ``registry.upgrade(record, "split")``.

``split`` at version 1 takes ``sep``; version 2 takes ``separator`` and says ``Upgrade(renamed_inputs={"sep":
"separator"})``. Upgrading a record whose ``split`` is at version 1 moves what ``sep`` gets (its constant, its graph
input or the edges into it) to ``separator`` and sets the version to 2.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from conductor._internal.gate import RefusedError, read_record, record_values
from conductor._internal.node import NodeDefinition, declared_inputs
from conductor._internal.problem import InvalidCallError
from conductor._internal.problem import catalogue as problem
from conductor._internal.record import GraphRecord


def upgraded(nodes: Mapping[str, NodeDefinition], record: GraphRecord | str | bytes, node_id: str, *, to: int | None) -> GraphRecord:
    """``record`` with the node ``node_id`` moved to version ``to`` of its definition in ``nodes``, the newest when
    ``to`` is ``None``: what ``registry.upgrade`` returns, and its docstring says how.

    The record is changed as the plain data of its JSON and read again by the record door, so what a step writes is
    held to the record's rules: a rewrite that gives a value to an input that edges lead into gives that input two
    entries, and the door refuses it.
    """
    try:
        graph = read_record(record)
    except RefusedError as refused:
        raise InvalidCallError(refused.problems) from None
    node_record = graph.nodes.get(node_id)
    if node_record is None:
        raise InvalidCallError([problem.node_not_in_graph(node_id, nodes=sorted(graph.nodes))])
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
    data: dict[str, Any] = graph.model_dump(mode="json")
    nodes_data: dict[str, dict[str, Any]] = data["nodes"]
    for step, older in reversed(steps):
        upgrade = step.upgrade
        gone = declared_inputs(older).keys() - declared_inputs(step).keys() - upgrade.renamed_inputs.keys()
        inputs = {
            upgrade.renamed_inputs.get(name, name): entry for name, entry in nodes_data[node_id].get("inputs", {}).items()
            if not (name in gone and "sources" in entry)
        }
        if upgrade.rewrite is not None:
            values = {name: entry.get("constant", entry.get("default")) for name, entry in inputs.items() if "constant" in entry or "default" in entry}
            try:
                rewritten = record_values(node_id, upgrade.rewrite(values))
            except RefusedError as refused:
                raise InvalidCallError(refused.problems) from None
            inputs = {name: {key: value for key, value in entry.items() if key not in ("constant", "default")} for name, entry in inputs.items() if "constant" not in entry}
            for name, value in rewritten.items():
                entry = inputs.get(name, {})
                inputs[name] = {**entry, "default" if "graph_input" in entry else "constant": value}
        nodes_data[node_id] = {**nodes_data[node_id], "inputs": inputs}
        for source in _sources_in(data):
            if source["node"] == node_id:
                source["output"] = upgrade.renamed_outputs.get(source["output"], source["output"])
    nodes_data[node_id]["version"] = target
    try:
        return read_record(data)
    except RefusedError as refused:
        raise InvalidCallError(refused.problems) from None


def _sources_in(data: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every output a graph record's plain data names: each edge's source, and each graph output."""
    edges = [source for node in data["nodes"].values() for entry in node.get("inputs", {}).values() for source in entry.get("sources", ())]
    return [*edges, *data["outputs"].values()]
