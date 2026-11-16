"""Moving one node of a graph record to a newer version of its node, for ``registry.upgrade``.

Say ``split`` at version 1 takes an input ``sep``. Version 2 calls it ``separator``, and says so with
``Upgrade(renamed_inputs={"sep": "separator"})``. Upgrading a record whose ``split`` is at version 1 moves whatever
``sep`` got, a constant, a graph input or edges, over to ``separator``, and sets the version to 2.

An input the newer version neither has nor renames is dropped. Its ``rewrite`` still sees its constant or default
first, so it can carry the value over to another input.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import MISSING

from conductor._internal.base import Source
from conductor._internal.node import NodeDefinition, gone_inputs
from conductor._internal.problem import InvalidCallError, call_problems, compile_problems
from conductor._internal.record import (
    GraphRecord,
    InputEntry,
    NodeRecord,
    RecordValidator,
    WrittenValuesValidator,
)


def upgraded(
    nodes: Mapping[str, NodeDefinition],
    record: GraphRecord | str | bytes,
    node_id: str,
    *,
    to: int | None,
) -> GraphRecord:
    """Returns ``record`` with the node ``node_id`` moved to version ``to``, or to the newest if ``to`` is ``None``.

    This is ``registry.upgrade``, whose docstring says what each step does. After each step, the new record is
    checked again with ``RecordValidator``, so a step can't write a record that breaks the rules. Say a rewrite
    gives a value to an input that edges lead into. That input would have a constant and sources at once, so the
    record is refused, and the problem names the step.

    Args:
        nodes: The registry's nodes, by id.
        record: The graph record to upgrade.
        node_id: The id of the node in the record to move.
        to: The version to move it to, or ``None`` for the newest.

    Raises:
        InvalidCallError: See ``registry.upgrade``.
    """
    graph = RecordValidator()(record)
    node_record = graph.nodes.get(node_id)
    if node_record is None:
        raise InvalidCallError(
            [call_problems.node_not_in_graph(node_id, nodes=sorted(graph.nodes))]
        )
    if node_record.node not in nodes:
        raise InvalidCallError(
            [compile_problems.unknown_node(node_id, node=node_record.node, choices=sorted(nodes))]
        )
    newest = nodes[node_record.node]
    target = newest.version if to is None else to
    if target == node_record.version:
        return graph
    if (
        target < node_record.version
        or target not in newest.versions
        or node_record.version not in newest.versions
    ):
        raise InvalidCallError(
            [
                call_problems.cannot_move_to_version(
                    node_id,
                    node=node_record.node,
                    version=node_record.version,
                    target=target,
                    versions=sorted(newest.versions),
                )
            ]
        )
    newest_first = list(newest.versions[target].versions.values())
    steps = [
        (newer, older)
        for newer, older in zip(newest_first, newest_first[1:], strict=False)
        if newer.version > node_record.version
    ]
    for step, older in reversed(steps):
        upgrade = step.upgrade
        gone = gone_inputs(older, step)
        renamed = {
            upgrade.renamed_inputs.get(name, name): entry
            for name, entry in graph.nodes[node_id].inputs.items()
        }
        inputs = {name: entry for name, entry in renamed.items() if name not in gone}
        if upgrade.rewrite is not None:
            inputs = _rewritten(node_id, renamed, inputs, upgrade.rewrite)
        records = {
            other: NodeRecord.model_construct(
                node=node.node,
                version=step.version if other == node_id else node.version,
                display=node.display,
                inputs={
                    name: _renamed(entry, node_id, upgrade.renamed_outputs)
                    for name, entry in (inputs if other == node_id else node.inputs).items()
                },
            )
            for other, node in graph.nodes.items()
        }
        outputs = {
            name: _renamed_source(source, node_id, upgrade.renamed_outputs)
            for name, source in graph.outputs.items()
        }
        # The records above are built without checks. The record validator checks what the step wrote.
        graph = RecordValidator(
            written_by=f"the Upgrade of '{node_record.node}' to version {step.version}"
        )(GraphRecord.model_construct(nodes=records, outputs=outputs).model_dump(mode="json"))
    return graph


def _rewritten(
    node_id: str,
    renamed: Mapping[str, InputEntry],
    inputs: Mapping[str, InputEntry],
    rewrite: Callable[[dict[str, Any]], Mapping[str, Any]],
) -> dict[str, InputEntry]:
    """Runs an ``Upgrade``'s ``rewrite`` and returns the node's inputs after it.

    The rewrite gets the constants and graph-input defaults as one dict. That dict still has the inputs the step
    drops, so the rewrite can move their values elsewhere. What it returns is converted to JSON. A key that was a
    graph input's default stays a default. Every other key becomes a constant. A constant or default the rewrite
    leaves out is dropped.

    Args:
        node_id: The node's id. A problem names it.
        renamed: The node's inputs after renaming, with the dropped ones still in.
        inputs: The node's inputs after renaming, without the dropped ones.
        rewrite: The ``Upgrade``'s ``rewrite``.

    Raises:
        InvalidCallError: ``invalid_value`` if a value the rewrite returns can't be written as JSON.
    """
    values = {
        name: entry.default if entry.constant is MISSING else entry.constant
        for name, entry in renamed.items()
        if entry.constant is not MISSING or entry.default is not MISSING
    }
    written = WrittenValuesValidator(node_id)(rewrite(values))
    kept = {
        name: InputEntry.model_construct(graph_input=entry.graph_input, sources=entry.sources)
        for name, entry in inputs.items()
        if entry.constant is MISSING
    }
    for name, value in written.items():
        entry = kept.get(name)
        if entry is not None and entry.graph_input is not MISSING:
            kept[name] = InputEntry.model_construct(graph_input=entry.graph_input, default=value)
        else:  # if the entry has sources, this makes it two kinds at once, which the record validator refuses
            kept[name] = InputEntry.model_construct(
                constant=value, sources=MISSING if entry is None else entry.sources
            )
    return kept


def _renamed(entry: InputEntry, node_id: str, renames: Mapping[str, str]) -> InputEntry:
    """Returns ``entry`` with each source from a renamed output of ``node_id`` pointed at the new name."""
    if entry.sources is MISSING:
        return entry
    return InputEntry.model_construct(
        constant=entry.constant,
        sources=tuple(_renamed_source(source, node_id, renames) for source in entry.sources),
    )


def _renamed_source(source: Source, node_id: str, renames: Mapping[str, str]) -> Source:
    """Returns ``source`` pointed at the new name of its output, if it is an output of ``node_id`` that is renamed."""
    return (
        Source(source.node, renames[source.output], source.field)
        if source.node == node_id and source.output in renames
        else source
    )
