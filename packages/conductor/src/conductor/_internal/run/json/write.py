"""Writes a run as JSON (``run_to_json``), and a complete run's graph outputs as JSON (``outputs_to_json``).

``saved_run`` turns the run into an instance of its graph's codec, the subclass of ``SavedRun`` made for the
graph. That instance then writes itself as text.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, assert_never

import pydantic_core
from pydantic_core import PydanticSerializationError

from conductor._internal.base import SKIPPED, ItemIndexes, NodePath, place_text
from conductor._internal.field import Field
from conductor._internal.problem import InvalidCallError, call_problems
from conductor._internal.types import Type

from ..blueprint import BlueprintFunctionNode
from ..outcome import Done, NoneInput, Outcome, Skipped, Unanswered
from ..progress import all_item_indexes
from ..reuse import ReusableOutcomes
from .codec import saved_type
from .type import (
    SAVED_DONE,
    SAVED_SKIPPED,
    SavedOutcomes,
    SavedOutcomeState,
    SavedReusable,
    SavedRun,
)

if TYPE_CHECKING:
    from ..type import Run


def run_to_json(run: Run) -> str:
    r"""Returns the whole run as JSON text.

    For the finished README run on ``"a\nb\nc"``, the text starts
    ``{"graph":"<16 hex characters>","ended_early":null,"max_items":null,"inputs":{"text":"a\nb\nc"},`` and goes on
    with every node's outcomes. ``SavedRun`` shows all of it.

    Raises:
        InvalidCallError: ``run_unwritable``, naming the input or output, if a value can't be written as JSON.
    """
    saved = saved_run(run)
    try:
        return saved.model_dump_json(exclude_defaults=True)
    except PydanticSerializationError:
        where = first_unwritable_value(saved, run._blueprint.function_nodes)
        raise InvalidCallError(
            [
                call_problems.run_unwritable(
                    "saved_run", where, where=where, reason="a value has no JSON form"
                )
            ]
        ) from None


def outputs_to_json(run: Run) -> str:
    """Returns the graph outputs of a complete run (``run.outputs``) as JSON text, by name.

    For the README run, it returns ``{"result":"A!+B!+C!"}``. Each value is written by its output's type, the same
    way a saved run writes it. A table is written as its columns, ``inf`` and ``nan`` as text, and a host's own type
    by its own schema.

    Raises:
        InvalidCallError: ``run_not_complete`` if the run isn't complete. ``run_unwritable``, naming the output, if a
            value can't be written as JSON.
    """
    written = {}
    for text, value in run.outputs.items():
        edge = run.graph._edge_reading(text)
        function = run._blueprint.function_nodes[edge.node]
        hint: Any = saved_type(function.compiled_node.outputs[edge.output]).hint
        # one list, or None in its place, for each loop the function node is inside
        for _ in function.loops:
            hint = list[hint] | None
        try:
            written[text] = Type(hint).to_json(value)
        except PydanticSerializationError:
            raise InvalidCallError(
                [
                    call_problems.run_unwritable(
                        "outputs", text, where=text, reason="a value has no JSON form"
                    )
                ]
            ) from None
    return pydantic_core.to_json(written).decode()


def saved_run(run: Run) -> SavedRun:
    """Returns ``run`` as an instance of its graph's codec, ready to write itself as JSON.

    For the README run, ``saved_run(run).outcomes["shout"]`` has ``states=["done", "done", "done"]`` and
    ``outputs={"result": ["A!", "B!", "C!"]}``. Every part is an instance of the codec's own class for it, so this is
    the same object the codec would read back from the JSON.

    Only function nodes are saved, since only they are called. A node inside a subgraph is saved like any other,
    under its node path as text (``"loud/shout"``). A node with no outcome and nothing to reuse has no key at all.
    """
    codec = run.graph._codec
    function_nodes = run._blueprint.function_nodes
    names = {path: place_text(path) for path in function_nodes}
    outcomes = {
        names[path]: saved_outcomes(
            codec.outcomes_classes[names[path]],
            node.compiled_node.outputs,
            _outcomes_by_call_number(run, node),
        )
        for path, node in function_nodes.items()
        if run._outcomes[path]
    }
    reusable = {}
    for path, node in function_nodes.items():
        if path in run._reusable.reused_freely or path in run._reusable.reused_in_item_order:
            reusable[names[path]] = saved_reusable(
                codec.reusable_classes[names[path]],
                codec.outcomes_classes[names[path]],
                node.compiled_node.outputs,
                run._reusable,
                path,
            )
    code = {
        names[path]: run._reusable.code_changed.get(path) or node.code_fingerprint
        for path, node in function_nodes.items()
        if names[path] in outcomes or names[path] in reusable
    }
    return codec.model_construct(
        graph=run.graph.fingerprint,
        ended_early=None
        if run._last_resume == "running"
        else run._last_resume,  # a resume still going is never saved
        max_items=run.max_items,
        inputs=dict(run.inputs),
        outcomes=outcomes,
        reusable=reusable,
        code=code,
        complete=[names[path] for path in function_nodes if path in run._complete_nodes],
    )


def _outcomes_by_call_number(run: Run, node: BlueprintFunctionNode) -> list[Outcome | None]:
    """Returns the outcomes of ``node`` as a list with one entry per call, in the order of the calls' item indexes.

    An entry's place is its call's call number, and ``None`` is a call that has no outcome yet. For ``shout`` after
    two of its three calls, it is ``[Done(...), Done(...), None]``.
    """
    outcomes = run._outcomes[node.node_path]
    return [
        outcomes.get(item_indexes)
        for item_indexes in all_item_indexes(run._outcomes, node.loops, ())[0]
    ]


def saved_outcomes(
    saved_class: type[SavedOutcomes], outputs: Mapping[str, Field], outcomes: list[Outcome | None]
) -> SavedOutcomes:
    """Returns the outcomes of one node the way a saved run writes them: a list per output, with an entry per call.

    ``[Done({"result": "A!"}), NoneInput({"result": None}, "line"), Done({"result": "C!"})]`` becomes
    ``saved_class(states=["done", None, "done"], outputs={"result": ["A!", None, "C!"]}, not_called_because={1: "line"})``.

    Args:
        saved_class: The codec's class for the node's outcomes.
        outputs: The node's outputs, by name.
        outcomes: The outcomes in call-number order, with ``None`` for a call with no outcome.
    """
    states: list[SavedOutcomeState | None] = []
    values_by_output: dict[str, list[Any]] = {name: [] for name in outputs}
    questions: dict[int, Unanswered] = {}
    not_called_because: dict[int, str] = {}
    for call_number, outcome in enumerate(outcomes):
        match outcome:
            case Done():
                states.append(SAVED_DONE)
                for name, values in values_by_output.items():
                    values.append(outcome.outputs[name])
                continue
            case Skipped():
                states.append(SAVED_SKIPPED)
            case Unanswered():
                states.append(None)
                questions[call_number] = outcome
            case NoneInput():
                states.append(None)
                not_called_because[call_number] = outcome.input
            case None:
                states.append(None)
            case _:
                assert_never(outcome)
        for values in values_by_output.values():  # only a done call has its outputs saved
            values.append(None)
    skipped: dict[str, list[int]] = {}
    skipped_items: dict[str, dict[int, list[int]]] = {}
    for name, output in outputs.items():
        values = values_by_output[name]
        skipped_calls = [
            call_number for call_number, value in enumerate(values) if value is SKIPPED
        ]
        if skipped_calls:
            skipped[name] = skipped_calls
            for call_number in skipped_calls:
                values[call_number] = None
        if output.is_series:
            for call_number, series in enumerate(values):
                if series is not None:
                    values[call_number], skipped_indexes = series.for_saving()
                    if skipped_indexes:
                        skipped_items.setdefault(name, {})[call_number] = skipped_indexes
    return saved_class.model_construct(
        states=states,
        outputs=values_by_output,
        questions=questions,
        not_called_because=not_called_because,
        skipped=skipped,
        skipped_items=skipped_items,
    )


def saved_reusable(
    saved_class: type[SavedReusable],
    outcomes_class: type[SavedOutcomes],
    outputs: Mapping[str, Field],
    reusable: ReusableOutcomes,
    path: NodePath,
) -> SavedReusable:
    """Returns the outcomes one node may reuse from the previous run, the way a saved run writes them.

    ``SavedReusable`` shows the JSON. Each outcome is written once, grouped by its call's fingerprint. The first
    written with a fingerprint is the one any call with it reuses, and the outcomes of a node that may ask are
    written in the order its calls take them.

    Args:
        saved_class: The codec's class for the node's reusable outcomes.
        outcomes_class: The codec's class for the node's outcomes.
        outputs: The node's outputs, by name.
        reusable: The run's reusable outcomes.
        path: The node's path.
    """
    outcomes: list[Outcome | None] = []
    fingerprints: list[str] = []
    item_indexes: dict[int, ItemIndexes] = {}
    for fingerprint, of_fingerprint in reusable.reused_freely.get(path, {}).items():
        for indexes, done in of_fingerprint.items():
            item_indexes[len(outcomes)] = indexes
            outcomes.append(done)
            fingerprints.append(fingerprint)
    for fingerprint, left in reusable.reused_in_item_order.get(path, {}).items():
        outcomes.extend(left)
        fingerprints.extend([fingerprint] * len(left))
    return saved_class.model_construct(
        outcomes=saved_outcomes(outcomes_class, outputs, outcomes),
        fingerprints=fingerprints,
        item_indexes=item_indexes,
    )


def first_unwritable_value(saved: SavedRun, function_nodes: Iterable[NodePath]) -> str:
    """Returns where a run that couldn't be written has a value that can't be written as JSON.

    The result looks like ``"text"`` for a graph input, or ``"loud/shout.result"`` for an output. It writes each
    graph input and each output's list on its own through the graph's codec (``include=`` picks the one value), and
    returns the first that fails. It is only called after pydantic refused the whole run, so a run that can be
    written pays nothing for it.

    Args:
        saved: The run as ``saved_run`` returned it.
        function_nodes: The path of every function node of the run's graph, in the order they are saved.

    Returns:
        The name of the graph input, or the output as ``place_text`` writes it. ``"the run"`` when each value
        writes fine on its own.
    """
    alone: list[tuple[str, Any]] = [(name, {"inputs": {name}}) for name in saved.inputs]
    for path in function_nodes:
        if (outcomes := saved.outcomes.get(name := place_text(path))) is not None:
            alone.extend(
                (place_text(path, output), {"outcomes": {name: {"outputs": {output}}}})
                for output in outcomes.outputs
            )
    for path in function_nodes:
        if (reusable := saved.reusable.get(name := place_text(path))) is not None:
            alone.extend(
                (
                    place_text(path, output),
                    {"reusable": {name: {"outcomes": {"outputs": {output}}}}},
                )
                for output in reusable.outcomes.outputs
            )
    for where, include in alone:
        try:
            saved.model_dump_json(include=include)
        except PydanticSerializationError:
            return where
    return "the run"
