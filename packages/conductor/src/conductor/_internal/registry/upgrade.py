"""Moving one node of a graph record to a newer version of its node, for ``registry.upgrade``.

Say ``split`` at version 1 takes an input ``sep``. Version 2 calls it ``separator``, and says so with
``Upgrade(renamed_inputs={"sep": "separator"})``. Upgrading a record whose ``split`` is at version 1 moves whatever
``sep`` got, a constant, a graph input or edges, over to ``separator``, and sets the version to 2.

An input the newer version neither has nor renames is dropped. Its ``rewrite`` still sees its constant first, so it
can carry the value over to another input. A key the rewrite returns is kept, so a rewrite that returns the dropped
input's key keeps it in the record, as a constant.

A graph input belongs to the whole graph, so an upgrade of one node never changes its default. A graph input that
only the dropped inputs took is dropped with them.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from functools import partial
from typing import Any

from conductor._internal.base import Source
from conductor._internal.node import NodeDefinition, RewriteValidator, gone_inputs
from conductor._internal.problem import (
    InvalidCallError,
    author_code_problem,
    call_problems,
    compile_problems,
)
from conductor._internal.record import (
    ConstantEntry,
    GraphInputEntry,
    GraphRecord,
    InputEntry,
    RecordValidator,
    SourcesEntry,
)

#: An input's entry as an upgrade step writes it: an entry it keeps, or the JSON of one it changes, which the record
#: validator then checks.
type _Written = InputEntry | dict[str, Any]


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
        inputs_of: dict[str, dict[str, _Written]] = {
            other: {
                name: _renamed(entry, node_id, upgrade.renamed_outputs)
                for name, entry in node.inputs.items()
            }
            for other, node in graph.nodes.items()
        }
        inputs_of[node_id] = {
            name: _renamed(entry, node_id, upgrade.renamed_outputs)
            for name, entry in renamed.items()
            if name not in gone
        }
        if upgrade.rewrite is not None:
            inputs_of[node_id] = _rewritten(
                node_id, upgrade.rewrite, older, step, renamed, inputs_of[node_id]
            )
        dropped = _graph_inputs_named(
            node.inputs for node in graph.nodes.values()
        ) - _graph_inputs_named(inputs_of.values())
        graph = RecordValidator(
            written_by=f"the Upgrade of '{node_record.node}' to version {step.version}"
        )(
            {
                "inputs": {
                    name: declared for name, declared in graph.inputs.items() if name not in dropped
                },
                "nodes": {
                    other: node.model_dump(mode="json")
                    | {
                        "version": step.version if other == node_id else node.version,
                        "inputs": inputs_of[other],
                    }
                    for other, node in graph.nodes.items()
                },
                "outputs": {
                    name: _renamed_source(source, node_id, upgrade.renamed_outputs)
                    for name, source in graph.outputs.items()
                },
            }
        )
    return graph


def _rewritten(
    node_id: str,
    rewrite: Callable[[dict[str, Any]], Mapping[str, Any]],
    older: NodeDefinition,
    step: NodeDefinition,
    renamed: Mapping[str, InputEntry],
    inputs: Mapping[str, _Written],
) -> dict[str, _Written]:
    """Runs the ``rewrite`` of an ``Upgrade`` and returns the node's inputs after it.

    The rewrite gets the node's constants as one dict. That dict still has the inputs the step drops, so the
    rewrite can move their values elsewhere. ``RewriteValidator`` checks what it returns, and writes it as JSON.
    Every key it returns becomes a constant, and a constant it leaves out is dropped. An input that takes a graph
    input or edges keeps them, and a key the rewrite returns for one makes an entry of two kinds, which the record
    validator refuses.

    The rewrite is the node author's code. A ``NodeError`` it raises keeps its code and message. Any other exception
    goes to the ``conductor`` logger, and the problem has only its class name.

    Args:
        node_id: The node's id. A problem names it.
        rewrite: The ``Upgrade``'s ``rewrite``.
        older: The version the step moves from.
        step: The version the step moves to, whose ``Upgrade`` has the rewrite.
        renamed: The node's inputs after renaming, with the ones the step drops still in.
        inputs: The node's inputs after renaming, without the ones the step drops.

    Raises:
        InvalidCallError: ``cannot_move_to_version_rewrite_failed`` or ``cannot_move_to_version_raised``, if the rewrite
            raises. See ``RewriteValidator`` for what it may return.
    """
    values = {
        name: entry.constant for name, entry in renamed.items() if isinstance(entry, ConstantEntry)
    }
    try:
        returned = rewrite(values)
    except Exception as error:  # noqa: BLE001  the rewrite is the node author's code, and may raise anything
        raise InvalidCallError(
            [
                author_code_problem(
                    error,
                    "the rewrite of the Upgrade of node %r, from version %d to %d, raised",
                    node_id,
                    older.version,
                    step.version,
                    raised=partial(
                        call_problems.cannot_move_to_version_raised,
                        node_id,
                        node=step.id,
                        version=older.version,
                        target=step.version,
                    ),
                    failed=partial(
                        call_problems.cannot_move_to_version_rewrite_failed,
                        node_id,
                        node=step.id,
                        version=older.version,
                        target=step.version,
                    ),
                )
            ]
        ) from None
    written = RewriteValidator(node_id, step.id, older.version, step.version)(returned)
    kept: dict[str, _Written] = {
        name: entry for name, entry in inputs.items() if not isinstance(entry, ConstantEntry)
    }
    return kept | {
        name: _json(kept[name]) | {"constant": value} if name in kept else {"constant": value}
        for name, value in written.items()
    }


def _renamed(entry: InputEntry, node_id: str, renames: Mapping[str, str]) -> _Written:
    """Returns ``entry`` with each source from a renamed output of ``node_id`` pointed at the new name."""
    if not isinstance(entry, SourcesEntry) or not any(
        source.node == node_id and source.output in renames for source in entry.sources
    ):
        return entry
    return {"sources": [_renamed_source(source, node_id, renames) for source in entry.sources]}


def _renamed_source(source: Source, node_id: str, renames: Mapping[str, str]) -> Source:
    """Returns ``source`` pointed at the new name of its output, if it is an output of ``node_id`` that is renamed."""
    return (
        Source(source.node, renames[source.output], source.field)
        if source.node == node_id and source.output in renames
        else source
    )


def _graph_inputs_named(inputs_of: Iterable[Mapping[str, _Written]]) -> set[str]:
    """Returns the names of the graph inputs that the entries of ``inputs_of``, each node's inputs, take."""
    return {
        entry.graph_input
        for inputs in inputs_of
        for entry in inputs.values()
        if isinstance(entry, GraphInputEntry)
    }


def _json(entry: _Written) -> dict[str, Any]:
    """Returns ``entry`` as the JSON a record writes it as."""
    return entry if isinstance(entry, dict) else entry.model_dump(mode="json")
