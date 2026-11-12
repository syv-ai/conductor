"""Writing a save: the units and the earlier work of a node as columns, and where a run that could not be written holds the value with no JSON form."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import pydantic_core

from conductor._internal.ask import Question
from conductor._internal.field import Field
from conductor._internal.markers import SKIPPED
from conductor._internal.outcomes import Answered, Done, NotCalled, Outcome, Skipped, Waiting
from conductor._internal.run.json.type import (
    SAVED_DONE,
    SAVED_SKIPPED,
    SameAs,
    SavedOutcomes,
    SavedOutcomeState,
    SavedReusable,
    SavedRun,
)
from conductor._internal.run.reuse import ReusableOutcomes
from conductor._internal.value_type import Type

#: The state a save writes for a unit that holds outputs or is a branch not taken; any other has none.
_STATE_WORD: Mapping[type, SavedOutcomeState] = {Done: SAVED_DONE, Answered: SAVED_DONE, Skipped: SAVED_SKIPPED}

#: The kinds of unit that hold outputs.
_HOLDS_OUTPUTS = frozenset({Done, Answered})


def saved_outcomes(outputs: Mapping[str, Field], outcomes: list[Outcome | None]) -> SavedOutcomes:
    """Gives units as the columns a save holds of them.

    Args:
        outputs: The outputs of the units' node, by name.
        outcomes: The units, in the order of their slots; ``None`` for an instance that hasn't run.
    """
    kinds = set(map(type, outcomes))
    columns: dict[str, list[Any]] = {}
    skipped: dict[str, list[int]] = {}
    skipped_items: dict[str, dict[int, list[int]]] = {}
    questions: dict[int, Question] | None = None
    not_called_because: dict[int, str] | None = None
    if kinds & _HOLDS_OUTPUTS:
        every = kinds <= _HOLDS_OUTPUTS  # every unit holds outputs, as those of a node that finished do
        for name, output in outputs.items():
            column = [outcome.outputs[name] for outcome in cast("list[Done | Answered]", outcomes)] if every else [outcome.outputs[name] if type(outcome) is Done or type(outcome) is Answered else None for outcome in outcomes]
            columns[name] = column
            gone = [call_number for call_number, value in enumerate(column) if value is SKIPPED]
            if gone:
                skipped[name] = gone
                for call_number in gone:
                    column[call_number] = None
            if output.many:
                for call_number, items in enumerate(column):
                    if items is not None:
                        column[call_number], gaps = items.for_saving()
                        if gaps:
                            skipped_items.setdefault(name, {})[call_number] = gaps
    else:  # no unit holds outputs yet
        columns = {name: [None] * len(outcomes) for name in outputs}
    if Answered in kinds or Waiting in kinds:
        questions = {call_number: outcome.question for call_number, outcome in enumerate(outcomes) if type(outcome) is Answered or type(outcome) is Waiting}
    if NotCalled in kinds:
        not_called_because = {call_number: outcome.input for call_number, outcome in enumerate(outcomes) if type(outcome) is NotCalled}
    return SavedOutcomes(
        states=list(map(_STATE_WORD.get, map(type, outcomes))), outputs=columns, questions=questions, not_called_because=not_called_because,
        skipped=skipped or None, skipped_items=skipped_items or None,
    )


def tables_written_once(outputs: Mapping[str, Mapping[str, Field]], shareable: Iterable[tuple[str, str]], outcomes: Mapping[str, SavedOutcomes]) -> list[SameAs] | None:
    """Finds the outputs in ``outcomes`` that have the very same table object as an output before them, replaces each
    with ``None`` in its column, and returns where each one's table is written.

    Example: ``read.result`` is a frame and ``split.result`` is ``Records`` of that same frame. The column of
    ``split.result`` becomes ``[None]`` and the result is
    ``[SameAs(output=("split", "result", 0), same_as=("read", "result", 0))]``.

    Args:
        outputs: The outputs of each node, by the node's name in a save.
        shareable: The ``(node, output)`` pairs whose type can have such a table, in the order a save writes them.
        units: The columns of each node, as ``saved_outcomes`` made them.

    Returns:
        The outputs left out, or ``None`` when there are none.
    """
    first: dict[int, tuple[Any, tuple[str, str, int]]] = {}  # by id() of a table alive in ``outcomes`` during this call
    same: list[SameAs] = []
    for node, output in shareable:
        of_node = outcomes.get(node)
        if of_node is None:
            continue
        written_once, column = outputs[node][output].whole.written_once, of_node.outputs[output]
        for call_number, stored in enumerate(column):
            table = None if stored is None else written_once(stored)
            if table is None:
                continue
            before = first.setdefault(id(table), (table, (node, output, call_number)))
            if before[1] != (node, output, call_number) and before[0] is table:
                column[call_number] = None
                same.append(SameAs(output=(node, output, call_number), same_as=before[1]))
    return same or None


def saved_reusable(outputs: Mapping[str, Field], previous: ReusableOutcomes) -> SavedReusable[SavedOutcomes]:
    """Gives what a node may take from an earlier run as a save holds it, each unit once: the computed units by
    the instance they stood in, then the ones only known by what they received, then the answers and the
    questions still waiting.

    Args:
        outputs: The node's outputs, by name.
        previous: The earlier work.
    """
    outcomes: list[Outcome | None] = [done for _, done in previous.by_item_indexes.values()]
    fingerprints = [fingerprint for fingerprint, _ in previous.by_item_indexes.values()]
    item_indexes = dict(enumerate(previous.by_item_indexes))
    saved_fingerprints = set(fingerprints)  # each of these is one of the units above: the first that received it
    for fingerprint, done in previous.by_fingerprint.items():
        if fingerprint not in saved_fingerprints:
            outcomes.append(done)
            fingerprints.append(fingerprint)
    outer_item_fingerprints: dict[int, str] = {}
    for (fingerprint, outer_items_fingerprint), answers in previous.answers.items():
        for outcome in answers:
            if outer_items_fingerprint:
                outer_item_fingerprints[len(outcomes)] = outer_items_fingerprint
            outcomes.append(outcome)
            fingerprints.append(fingerprint)
    return SavedReusable(outcomes=saved_outcomes(outputs, outcomes), fingerprints=fingerprints, item_indexes=item_indexes or None, outer_item_fingerprints=outer_item_fingerprints or None)


def first_unwritable_value(inputs: Mapping[str, Type], outputs: Mapping[str, Mapping[str, Field]], saved: SavedRun[SavedOutcomes, SavedReusable[SavedOutcomes]]) -> str:
    """Says where a run that pydantic refused to write holds a value with no JSON form: the first input, or
    output of a node, that isn't written alone either. It is looked for only once the whole save was refused.

    Args:
        inputs: The type of each input of the run's graph, by address.
        outputs: The outputs of each function node of the graph, by the node's name in a save.
        saved: The run as ``Run._saved`` gives it.

    Returns:
        The input's address, or the node as a save names it and the output (``e/ask.result``).
    """
    columns = {address: (inputs[address], value) for address, value in saved.inputs.items()}
    by_node = [*saved.outcomes.items(), *((name, previous.outcomes) for name, previous in saved.reusable.items())]
    for name, outcomes in by_node:
        columns.update({f"{name}.{output}": (outputs[name][output].saved_list_type, column) for output, column in outcomes.outputs.items()})
    for at, (written_by, value) in columns.items():
        try:
            written_by.adapter.serializer.to_json(value)
        except pydantic_core.PydanticSerializationError:
            return at
    return "the run"
