"""Moving one node of a graph record to a newer version of its definition: ``registry.upgrade(record, "split")``.

``split`` at version 1 takes ``sep``; version 2 takes ``separator`` and says ``Upgrade(renamed_inputs={"sep":
"separator"})``. Upgrading a record whose ``split`` is at version 1 moves what ``sep`` gets (its constant, its graph
input or the edges into it) to ``separator`` and sets the version to 2.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import MISSING

from conductor._internal.base import Address
from conductor._internal.gate import RefusedError, read_record, record_values
from conductor._internal.node import NodeDefinition
from conductor._internal.problem import InvalidCallError, call_problems, compile_problems
from conductor._internal.record import GraphRecord, InputEntry, NodeRecord


def upgraded(nodes: Mapping[str, NodeDefinition], record: GraphRecord | str | bytes, node_id: str, *, to: int | None) -> GraphRecord:
    """``record`` with the node ``node_id`` moved to version ``to`` of its definition in ``nodes``, the newest when
    ``to`` is ``None``: what ``registry.upgrade`` returns, and its docstring says how.

    Each step's record is read again by the record door, so what a step writes is held to the record's rules: a
    rewrite that gives a value to an input that edges lead into gives that input a constant beside its sources, and
    the door refuses it, naming the step.
    """
    try:
        graph = read_record(record)
    except RefusedError as refused:
        raise InvalidCallError(refused.problems) from None
    node_record = graph.nodes.get(node_id)
    if node_record is None:
        raise InvalidCallError([call_problems.node_not_in_graph(node_id, nodes=sorted(graph.nodes))])
    if node_record.node not in nodes:
        raise InvalidCallError([compile_problems.unknown_node(node_id, node=node_record.node, choices=sorted(nodes))])
    newest = nodes[node_record.node]
    target = newest.version if to is None else to
    if target == node_record.version:
        return graph
    if target < node_record.version or target not in newest.versions or node_record.version not in newest.versions:
        raise InvalidCallError([call_problems.cannot_move_to_version(
            node_id,
            node=node_record.node, version=node_record.version, target=target, versions=sorted(newest.versions),
        )])
    version_chain = newest.version_of(target).version_chain
    steps = [(newer, older) for newer, older in zip(version_chain, version_chain[1:], strict=False) if newer.version > node_record.version]
    for step, older in reversed(steps):
        upgrade = step.upgrade
        gone = older.declared_inputs.keys() - step.declared_inputs.keys() - upgrade.renamed_inputs.keys()
        inputs = {
            upgrade.renamed_inputs.get(name, name): entry for name, entry in graph.nodes[node_id].inputs.items()
            if not (name in gone and entry.sources is not MISSING)
        }
        if upgrade.rewrite is not None:
            inputs = _rewritten(node_id, inputs, upgrade.rewrite)
        records = {
            other: NodeRecord.model_construct(
                node=node.node, version=step.version if other == node_id else node.version, display=node.display,
                inputs={name: _renamed(entry, node_id, upgrade.renamed_outputs) for name, entry in (inputs if other == node_id else node.inputs).items()},
            )
            for other, node in graph.nodes.items()
        }
        outputs = {name: _renamed_source(address, node_id, upgrade.renamed_outputs) for name, address in graph.outputs.items()}
        try:  # unchecked until here: the door checks what the step wrote
            graph = read_record(GraphRecord.model_construct(nodes=records, outputs=outputs).model_dump(mode="json"), written_by=f"the Upgrade of '{node_record.node}' to version {step.version}")
        except RefusedError as refused:
            raise InvalidCallError(refused.problems) from None
    return graph


def _rewritten(node_id: str, inputs: Mapping[str, InputEntry], rewrite: Callable[[dict[str, Any]], Mapping[str, Any]]) -> dict[str, InputEntry]:
    """``inputs`` after an ``Upgrade``'s ``rewrite``: it is given the node's constants and its graph inputs' defaults as
    one map, and what it returns is written as JSON. A graph input's default stays one; every other value returned is a
    constant. A constant it does not return goes, and so does a default.

    Raises:
        InvalidCallError: A value it returns has no JSON form (``invalid_value``).
    """
    values = {name: entry.default if entry.constant is MISSING else entry.constant for name, entry in inputs.items() if entry.constant is not MISSING or entry.default is not MISSING}
    try:
        written = record_values(node_id, rewrite(values))
    except RefusedError as refused:
        raise InvalidCallError(refused.problems) from None
    kept = {
        name: InputEntry.model_construct(graph_input=entry.graph_input, sources=entry.sources)
        for name, entry in inputs.items() if entry.constant is MISSING
    }
    for name, value in written.items():
        entry = kept.get(name)
        if entry is not None and entry.graph_input is not MISSING:
            kept[name] = InputEntry.model_construct(graph_input=entry.graph_input, default=value)
        else:  # beside sources, an entry of two kinds the door refuses
            kept[name] = InputEntry.model_construct(constant=value, sources=MISSING if entry is None else entry.sources)
    return kept


def _renamed(entry: InputEntry, node_id: str, renames: Mapping[str, str]) -> InputEntry:
    """``entry`` with each of its sources that is an output of ``node_id`` the step renames read under the new name."""
    if entry.sources is MISSING:
        return entry
    return InputEntry.model_construct(constant=entry.constant, sources=tuple(_renamed_source(source, node_id, renames) for source in entry.sources))


def _renamed_source(source: Address, node_id: str, renames: Mapping[str, str]) -> Address:
    """``source`` read under the new name of its output, when it is an output of ``node_id`` that a step renames."""
    return Address(source.node, renames[source.output], source.field) if source.node == node_id and source.output in renames else source
