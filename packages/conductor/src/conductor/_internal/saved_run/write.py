"""Writing a save: the units and the earlier work of a node as columns, and where a run that could not be written holds the value with no JSON form."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pydantic_core

from conductor._internal.ask import Question
from conductor._internal.earlier_work import EarlierWork
from conductor._internal.field import Field
from conductor._internal.saved_run.type import (
    SAVED_DONE,
    SAVED_SKIPPED,
    SavedEarlier,
    SavedRun,
    SavedUnits,
    UnitState,
)
from conductor._internal.sentinels import SKIPPED
from conductor._internal.units import Answered, Asking, Blank, Done, Skipped, Unit
from conductor._internal.value_type import Type

#: The state a save writes for a unit that holds outputs or is a branch not taken; any other has none.
_STATE_WORD: Mapping[type, UnitState] = {Done: SAVED_DONE, Answered: SAVED_DONE, Skipped: SAVED_SKIPPED}

#: The kinds of unit that hold outputs.
_HOLDS_OUTPUTS = frozenset({Done, Answered})


def unit_columns(outputs: Mapping[str, Field], slots: list[Unit | None]) -> SavedUnits:
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
            column = [unit.out[name] for unit in cast("list[Done | Answered]", slots)] if every else [unit.out[name] if type(unit) is Done or type(unit) is Answered else None for unit in slots]
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
        questions = {slot: unit.question for slot, unit in enumerate(slots) if type(unit) is Answered or type(unit) is Asking}
    if Blank in kinds:
        blank_inputs = {slot: unit.input for slot, unit in enumerate(slots) if type(unit) is Blank}
    return SavedUnits(
        states=list(map(_STATE_WORD.get, map(type, slots))), outputs=columns, questions=questions, blank_inputs=blank_inputs,
        skipped=skipped or None, skipped_items=skipped_items or None,
    )


def earlier_columns(outputs: Mapping[str, Field], earlier: EarlierWork) -> SavedEarlier[SavedUnits]:
    """Gives what a node may take from an earlier run as a save holds it, each unit once: the computed units by
    the instance they stood in, then the ones only known by what they received, then the answers and the
    questions still waiting.

    Args:
        outputs: The node's outputs, by name.
        earlier: The earlier work.
    """
    slots: list[Unit | None] = [done for _, done in earlier.at.values()]
    fingerprints = [fingerprint for fingerprint, _ in earlier.at.values()]
    places = dict(enumerate(earlier.at))
    stood = set(fingerprints)  # each of these is one of the units above: the first that received it
    for fingerprint, done in earlier.computed.items():
        if fingerprint not in stood:
            slots.append(done)
            fingerprints.append(fingerprint)
    lineages: dict[int, str] = {}
    for (fingerprint, lineage), asked in earlier.asked.items():
        for unit in asked:
            if lineage:
                lineages[len(slots)] = lineage
            slots.append(unit)
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
