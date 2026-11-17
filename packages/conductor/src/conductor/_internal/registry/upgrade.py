"""Moving one node of a graph record to a newer version of its node, for ``registry.upgrade``.

Say ``split`` at version 1 takes an input ``sep``. Version 2 calls it ``separator``, and says so with
``Upgrade(renamed_inputs={"sep": "separator"})``. Upgrading a record whose ``split`` is at version 1 moves whatever
``sep`` got, a constant, a graph input or edges, over to ``separator``, and sets the version to 2.

An input the newer version neither has nor renames is dropped. Its ``rewrite`` still sees its constant or default
first, so it can carry the value over to another input. A key the rewrite returns is kept, so a rewrite that returns
the dropped input's key keeps it in the record, as a constant.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

from pydantic import MISSING

from conductor._internal.base import Source
from conductor._internal.node import NodeDefinition, RewriteValidator, gone_inputs
from conductor._internal.problem import InvalidCallError, NodeError, call_problems, compile_problems
from conductor._internal.record import (
    GraphRecord,
    InputEntry,
    NodeRecord,
    RecordValidator,
)

_logger = logging.getLogger("conductor")


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
            inputs = _rewritten(node_id, upgrade.rewrite, older, step, renamed, inputs)
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
    rewrite: Callable[[dict[str, Any]], Mapping[str, Any]],
    older: NodeDefinition,
    step: NodeDefinition,
    renamed: Mapping[str, InputEntry],
    inputs: Mapping[str, InputEntry],
) -> dict[str, InputEntry]:
    """Runs the ``rewrite`` of an ``Upgrade`` and returns the node's inputs after it.

    The rewrite gets the node's constants and graph-input defaults as one dict. That dict still has the inputs the
    step drops, so the rewrite can move their values elsewhere. ``RewriteValidator`` checks what it returns, and
    writes it as JSON. A key that was a graph input's default stays a default. Every other key becomes a constant. A
    constant or default the rewrite leaves out is dropped.

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
            raises or returns no dict of input names. See ``RewriteValidator`` for the rest.
    """
    values = {
        name: entry.default if entry.constant is MISSING else entry.constant
        for name, entry in renamed.items()
        if entry.constant is not MISSING or entry.default is not MISSING
    }
    try:
        returned = rewrite(values)
    except NodeError as error:
        (problem,) = error.problems
        raise InvalidCallError(
            [
                call_problems.cannot_move_to_version_rewrite_failed(
                    node_id,
                    node=step.id,
                    version=older.version,
                    target=step.version,
                    reason=problem.msg,
                    error=problem.type,
                    error_ctx=dict(problem.ctx) or None,
                )
            ]
        ) from None
    except Exception as raised:  # noqa: BLE001  the rewrite is the node author's code, and may raise anything
        _logger.error(
            "the rewrite of the Upgrade of node %r, from version %d to %d, raised",
            node_id,
            older.version,
            step.version,
            exc_info=raised,
        )
        raise InvalidCallError(
            [
                call_problems.cannot_move_to_version_raised(
                    node_id,
                    node=step.id,
                    version=older.version,
                    target=step.version,
                    error=type(raised).__name__,
                )
            ]
        ) from None
    written = RewriteValidator(node_id, step.id, older.version, step.version)(returned)
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
