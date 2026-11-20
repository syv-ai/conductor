"""Writes a run as JSON (``run_to_json``), a complete run's graph outputs as JSON (``outputs_to_json``), and the
outputs a person's answer gave one call (``answer_to_json``).

``saved_run`` turns the run into an instance of its graph's codec, the subclass of ``SavedRun`` made for the
graph. That instance then writes itself as text. This package is the run's own JSON form, so it reads every field
of the ``Run``, the private ones too, as ``read.py`` builds a ``Run`` from them.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, assert_never

import pydantic_core
from pydantic_core import PydanticSerializationError

from conductor._internal.base import SKIPPED, ItemIndexes, NodePath, place_text
from conductor._internal.port import Port
from conductor._internal.problem import InvalidCallError, call_problems
from conductor._internal.types import Type

from ..blueprint import BlueprintFunctionNode
from ..outcome import Done, NoneInput, Outcome, Skipped, Unanswered
from ..outcomes import RunOutcomes
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
        where = first_unwritable_value(saved, run.graph._blueprint.function_nodes)
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
        function = run.graph._blueprint.function_nodes[edge.node]
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
    codec, outcomes, reusable = run.graph._codec, run._outcomes, run._reusable
    function_nodes = outcomes.blueprint.function_nodes
    names = {path: place_text(path) for path in function_nodes}
    saved = {
        names[path]: saved_outcomes(
            codec.outcomes_classes[names[path]],
            node.compiled_node.outputs,
            _outcomes_by_call_number(outcomes, node),
        )
        for path, node in function_nodes.items()
        if outcomes.of(path)
    }
    saved_reusables = {}
    for path, node in function_nodes.items():
        if entries := reusable.entries(path):
            saved_reusables[names[path]] = saved_reusable(
                codec.reusable_classes[names[path]], node.compiled_node.outputs, entries
            )
    return codec.model_construct(
        graph=run.graph.fingerprint,
        # a resume still going is never saved
        ended_early=None if run._last_resume == "running" else run._last_resume,
        max_items=run.max_items,
        inputs=dict(run.inputs),
        outcomes=saved,
        reusable=saved_reusables,
        complete=[names[path] for path in function_nodes if path in run._complete_nodes],
    )


def answer_to_json(
    codec: type[SavedRun], node: BlueprintFunctionNode, answered: Done
) -> dict[str, Any]:
    """Returns the outputs a person's answer gave one call, by output name, as JSON data.

    Each value is written as a saved run writes the call's outputs, by the node's list for the output in the graph's
    codec. A branch the answer left out, ``SKIPPED``, is left out here too. ``Done({"result": True})`` gives
    ``{"result": True}``.

    Args:
        codec: The graph's codec.
        node: The node that asked.
        answered: The call's outcome.
    """
    outputs = node.compiled_node.outputs
    saved = saved_outcomes(codec.outcomes_classes[place_text(node.node_path)], outputs, [answered])
    written: dict[str, list[Any]] = saved.model_dump(mode="json", include={"outputs"})["outputs"]
    return {name: values[0] for name, values in written.items() if name not in saved.skipped}


def _outcomes_by_call_number(
    outcomes: RunOutcomes, node: BlueprintFunctionNode
) -> list[Outcome | None]:
    """Returns the outcomes of ``node`` as a list with one entry per call, in the order of the calls' item indexes.

    An entry's place is its call's call number, and ``None`` is a call that has no outcome yet. For ``shout`` after
    two of its three calls, it is ``[Done(...), Done(...), None]``.
    """
    of_node = outcomes.of(node.node_path)
    return [of_node.get(item_indexes) for item_indexes in outcomes.item_indexes(node.loops)[0]]


def saved_outcomes[S: SavedOutcomes](
    saved_class: type[S], outputs: Mapping[str, Port], outcomes: list[Outcome | None]
) -> S:
    """Returns the outcomes of one node the way a saved run writes them: a list per output, with an entry per call.

    ``[Done({"result": "A!"}), NoneInput({"result": None}, "line"), Done({"result": "C!"})]`` becomes
    ``saved_class(states=["done", None, "done"], outputs={"result": ["A!", None, "C!"]}, fingerprints=[fp_a, None,
    fp_c], none_inputs={1: "line"})``.

    Args:
        saved_class: The codec's class for the node's outcomes.
        outputs: The node's outputs, by name.
        outcomes: The outcomes in call-number order, with ``None`` for a call with no outcome.
    """
    states: list[SavedOutcomeState | None] = []
    fingerprints: list[str | None] = []
    values_by_output: dict[str, list[Any]] = {name: [] for name in outputs}
    questions: dict[int, Unanswered] = {}
    none_inputs: dict[int, str] = {}
    for call_number, outcome in enumerate(outcomes):
        match outcome:
            case Done():
                states.append(SAVED_DONE)
                fingerprints.append(outcome.fingerprint)
                if outcome.asked is not None:  # an answer keeps what the person was shown
                    questions[call_number] = outcome.asked
                for name, values in values_by_output.items():
                    values.append(outcome.outputs[name])
                continue
            case Skipped():
                states.append(SAVED_SKIPPED)
                fingerprints.append(None)
            case Unanswered():
                states.append(None)
                fingerprints.append(outcome.fingerprint)
                questions[call_number] = outcome
            case NoneInput():
                states.append(None)
                fingerprints.append(None)
                none_inputs[call_number] = outcome.input
            case None:
                states.append(None)
                fingerprints.append(None)
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
        fingerprints=fingerprints,
        questions=questions,
        none_inputs=none_inputs,
        skipped=skipped,
        skipped_items=skipped_items,
    )


def saved_reusable(
    saved_class: type[SavedReusable],
    outputs: Mapping[str, Port],
    entries: list[tuple[ItemIndexes, Done | Unanswered]],
) -> SavedReusable:
    """Returns the outcomes one node may reuse from the previous run, the way a saved run writes them.

    ``SavedReusable`` shows the JSON. The outcomes are written in the order ``ReusableOutcomes.entries`` gives
    them, each with the item indexes its call had.

    Args:
        saved_class: The codec's class for the node's reusable outcomes.
        outputs: The node's outputs, by name.
        entries: The outcomes, each with its call's item indexes.
    """
    outcomes: list[Outcome | None] = [outcome for _, outcome in entries]
    return saved_outcomes(saved_class, outputs, outcomes).model_copy(
        update={"item_indexes": [item_indexes for item_indexes, _ in entries]}
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
                    {"reusable": {name: {"outputs": {output}}}},
                )
                for output in reusable.outputs
            )
    for where, include in alone:
        try:
            saved.model_dump_json(include=include)
        except PydanticSerializationError:
            return where
    return "the run"
