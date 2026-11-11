"""Writing a save: the units and the earlier work of a node as columns, and where a run that could not be written holds the value with no JSON form."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, cast

import pydantic_core

from conductor._internal.ask import Question
from conductor._internal.earlier_work import EarlierWork
from conductor._internal.field import Field
from conductor._internal.outcomes import Answered, Asking, Done, NotCalled, Outcome, Skipped
from conductor._internal.saved_run.type import (
    SAVED_DONE,
    SAVED_SKIPPED,
    SameAs,
    SavedEarlier,
    SavedRun,
    SavedUnits,
    UnitState,
)
from conductor._internal.sentinels import SKIPPED
from conductor._internal.value_type import Type

#: The state a save writes for a unit that holds outputs or is a branch not taken; any other has none.
_STATE_WORD: Mapping[type, UnitState] = {Done: SAVED_DONE, Answered: SAVED_DONE, Skipped: SAVED_SKIPPED}

#: The kinds of unit that hold outputs.
_HOLDS_OUTPUTS = frozenset({Done, Answered})


def unit_columns(outputs: Mapping[str, Field], slots: list[Outcome | None]) -> SavedUnits:
    """Gives units as the columns a save holds of them.

    Args:
        outputs: The outputs of the units' node, by name.
        slots: The units, in the order of their slots; ``None`` for an instance that hasn't run.
    """
    kinds = set(map(type, slots))
    columns: dict[str, list[Any]] = {}
    skipped: dict[str, list[int]] = {}
    skipped_items: dict[str, dict[int, list[int]]] = {}
    questions: dict[int, Question] | None = None
    blank_inputs: dict[int, str] | None = None
    if kinds & _HOLDS_OUTPUTS:
        every = kinds <= _HOLDS_OUTPUTS  # every unit holds outputs, as those of a node that finished do
        for name, output in outputs.items():
            column = [outcome.out[name] for outcome in cast("list[Done | Answered]", slots)] if every else [outcome.out[name] if type(outcome) is Done or type(outcome) is Answered else None for outcome in slots]
            columns[name] = column
            gone = [slot for slot, value in enumerate(column) if value is SKIPPED]
            if gone:
                skipped[name] = gone
                for slot in gone:
                    column[slot] = None
            if output.many:
                for slot, items in enumerate(column):
                    if items is not None:
                        column[slot], gaps = items.for_saving()
                        if gaps:
                            skipped_items.setdefault(name, {})[slot] = gaps
    else:  # no unit holds outputs yet
        columns = {name: [None] * len(slots) for name in outputs}
    if Answered in kinds or Asking in kinds:
        questions = {slot: outcome.question for slot, outcome in enumerate(slots) if type(outcome) is Answered or type(outcome) is Asking}
    if NotCalled in kinds:
        blank_inputs = {slot: outcome.input for slot, outcome in enumerate(slots) if type(outcome) is NotCalled}
    return SavedUnits(
        states=list(map(_STATE_WORD.get, map(type, slots))), outputs=columns, questions=questions, blank_inputs=blank_inputs,
        skipped=skipped or None, skipped_items=skipped_items or None,
    )


def tables_written_once(outputs: Mapping[str, Mapping[str, Field]], shareable: Iterable[tuple[str, str]], units: Mapping[str, SavedUnits]) -> list[SameAs] | None:
    """Finds the outputs in ``units`` that have the very same table object as an output before them, replaces each
    with ``None`` in its column, and returns where each one's table is written.

    Example: ``read.result`` is a frame and ``split.result`` is ``Records`` of that same frame. The column of
    ``split.result`` becomes ``[None]`` and the result is
    ``[SameAs(output=("split", "result", 0), same_as=("read", "result", 0))]``.

    Args:
        outputs: The outputs of each node, by the node's name in a save.
        shareable: The ``(node, output)`` pairs whose type can have such a table, in the order a save writes them.
        units: The columns of each node, as ``unit_columns`` made them.

    Returns:
        The outputs left out, or ``None`` when there are none.
    """
    first: dict[int, tuple[Any, tuple[str, str, int]]] = {}  # by id() of a table alive in ``units`` during this call
    same: list[SameAs] = []
    for node, output in shareable:
        of_node = units.get(node)
        if of_node is None:
            continue
        written_once, column = outputs[node][output].whole.written_once, of_node.outputs[output]
        for slot, stored in enumerate(column):
            table = None if stored is None else written_once(stored)
            if table is None:
                continue
            before = first.setdefault(id(table), (table, (node, output, slot)))
            if before[1] != (node, output, slot) and before[0] is table:
                column[slot] = None
                same.append(SameAs(output=(node, output, slot), same_as=before[1]))
    return same or None


def earlier_columns(outputs: Mapping[str, Field], earlier: EarlierWork) -> SavedEarlier[SavedUnits]:
    """Gives what a node may take from an earlier run as a save holds it, each unit once: the computed units by
    the instance they stood in, then the ones only known by what they received, then the answers and the
    questions still waiting.

    Args:
        outputs: The node's outputs, by name.
        earlier: The earlier work.
    """
    slots: list[Outcome | None] = [done for _, done in earlier.at.values()]
    fingerprints = [fingerprint for fingerprint, _ in earlier.at.values()]
    places = dict(enumerate(earlier.at))
    stood = set(fingerprints)  # each of these is one of the units above: the first that received it
    for fingerprint, done in earlier.computed.items():
        if fingerprint not in stood:
            slots.append(done)
            fingerprints.append(fingerprint)
    lineages: dict[int, str] = {}
    for (fingerprint, lineage), asked in earlier.asked.items():
        for outcome in asked:
            if lineage:
                lineages[len(slots)] = lineage
            slots.append(outcome)
            fingerprints.append(fingerprint)
    return SavedEarlier(units=unit_columns(outputs, slots), fingerprints=fingerprints, places=places or None, lineages=lineages or None)


def unwritable_at(inputs: Mapping[str, Type], outputs: Mapping[str, Mapping[str, Field]], saved: SavedRun[SavedUnits, SavedEarlier[SavedUnits]]) -> str:
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
    by_node = [*saved.units.items(), *((name, earlier.units) for name, earlier in saved.earlier.items())]
    for name, units in by_node:
        columns.update({f"{name}.{output}": (outputs[name][output].column, column) for output, column in units.outputs.items()})
    for at, (written_by, value) in columns.items():
        try:
            written_by.adapter.serializer.to_json(value)
        except pydantic_core.PydanticSerializationError:
            return at
    return "the run"
