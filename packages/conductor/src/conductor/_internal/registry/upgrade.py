"""Moving one node of a graph record to a newer version of its node, for ``registry.upgrade``.

Say ``split`` at version 1 takes an input ``sep``. Version 2 calls it ``separator``, and says so with
``Upgrade(renamed_inputs={"sep": "separator"})``. Upgrading a record whose ``split`` is at version 1 moves whatever
``sep`` got, a constant, a graph input or edges, over to ``separator``, and sets the version to 2.

An input the newer version neither has nor renames is dropped. Its ``rewrite`` still sees its constant first, so it
can carry the value over to another input. A key the rewrite returns is kept, so a rewrite that returns the dropped
input's key keeps it in the record, as a constant.

A graph input belongs to the whole graph, so an upgrade of one node never changes its default, and a rewrite never
sees it. A graph input that only the dropped inputs took is dropped with them, and leaves what a caller gives.

Every step edits the record's JSON, as ``make_graph`` writes a record. ``RecordValidator`` checks it after each step,
and the graph inputs that step dropped are read from the record it returns. The next step starts from that record
written as JSON again, with those graph inputs gone, and the JSON is read back once more at the end.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

from conductor._internal.node import NodeDefinition, RewriteValidator, gone_inputs
from conductor._internal.problem import (
    InvalidCallError,
    author_code_problem,
    call_problems,
    compile_problems,
)
from conductor._internal.record import GraphRecord, RecordValidator


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
    gives a value to an input that edges lead into, or that takes a graph input. That input would have two kinds of
    entry at once, so the record is refused, and the problem names the step.

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
    written = graph.model_dump(mode="json")
    taken = graph.graph_inputs_taken
    for step, older in reversed(steps):
        upgrade = step.upgrade
        node = written["nodes"][node_id]
        renamed = {
            upgrade.renamed_inputs.get(name, name): entry
            for name, entry in node.get("inputs", {}).items()
        }
        gone = gone_inputs(older, step)
        inputs = {name: entry for name, entry in renamed.items() if name not in gone}
        if upgrade.rewrite is not None:
            inputs = _rewritten(node_id, upgrade.rewrite, older, step, renamed, inputs)
        node["inputs"], node["version"] = inputs, step.version
        for other in written["nodes"].values():
            for entry in other.get("inputs", {}).values():
                if "sources" in entry:
                    entry["sources"] = [
                        _renamed_source(source, node_id, upgrade.renamed_outputs)
                        for source in entry["sources"]
                    ]
        written["outputs"] = {
            name: _renamed_source(source, node_id, upgrade.renamed_outputs)
            for name, source in written["outputs"].items()
        }
        checked = RecordValidator(
            written_by=f"the Upgrade of '{node_record.node}' to version {step.version}"
        )(written)
        # a graph input that only the inputs this step dropped took goes with them
        dropped = taken - checked.graph_inputs_taken
        written = checked.model_dump(mode="json")
        written["inputs"] = {
            name: declared for name, declared in written["inputs"].items() if name not in dropped
        }
        taken = checked.graph_inputs_taken
    return RecordValidator()(written)


def _rewritten(
    node_id: str,
    rewrite: Callable[[dict[str, Any]], Mapping[str, Any]],
    older: NodeDefinition,
    step: NodeDefinition,
    renamed: Mapping[str, dict[str, Any]],
    inputs: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Runs the ``rewrite`` of an ``Upgrade`` and returns the node's inputs after it, as JSON.

    The rewrite gets the node's constants as one dict. That dict still has the inputs the step drops, so the
    rewrite can move their values elsewhere. An input that takes a graph input or edges isn't in it, so for a node
    whose inputs are all graph inputs it gets ``{}``. ``RewriteValidator`` checks what it returns, and writes it as
    JSON. Every key it returns becomes a constant, and a constant it leaves out is dropped. An input that takes a
    graph input or edges keeps them, and a key the rewrite returns for one gives its entry two kinds, which the
    record validator refuses.

    The rewrite is the node author's code. A ``NodeError`` it raises keeps its code and message. Any other exception
    goes to the ``conductor`` logger, and the problem has only its class name.

    Args:
        node_id: The node's id. A problem names it.
        rewrite: The ``Upgrade``'s ``rewrite``.
        older: The version the step moves from.
        step: The version the step moves to, whose ``Upgrade`` has the rewrite.
        renamed: The node's entries after renaming, with the ones the step drops still in.
        inputs: The node's entries after renaming, without the ones the step drops.

    Raises:
        InvalidCallError: ``cannot_move_to_version_rewrite_failed`` or ``cannot_move_to_version_raised``, if the rewrite
            raises. See ``RewriteValidator`` for what it may return.
    """
    values = {name: entry["constant"] for name, entry in renamed.items() if "constant" in entry}
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
    kept = {name: entry for name, entry in inputs.items() if "constant" not in entry}
    return kept | {
        name: {**kept.get(name, {}), "constant": value} for name, value in written.items()
    }


def _renamed_source(
    source: dict[str, Any], node_id: str, renames: Mapping[str, str]
) -> dict[str, Any]:
    """Returns ``source``, as JSON, pointed at the new name of its output if it is a renamed output of ``node_id``."""
    if source["node"] == node_id and source["output"] in renames:
        return {**source, "output": renames[source["output"]]}
    return source
