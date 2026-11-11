"""Reading a save: the units and the earlier work of a node, from their columns."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic_core import PydanticCustomError

from conductor._internal.earlier_work import EarlierWork
from conductor._internal.field import Field
from conductor._internal.saved_run.type import (
    SAVED_DONE,
    SAVED_SKIPPED,
    SavedEarlier,
    SavedRun,
    SavedUnits,
)
from conductor._internal.sentinels import SKIPPED
from conductor._internal.units import Answered, Asking, Blank, Done, Skipped, Unit
from conductor._internal.value_type import FieldValueError


def units_of(not_taken: dict[str, Any], not_called: dict[str, Any], saved: SavedUnits) -> list[Unit | None]:
    """Makes the units a node's saved columns hold, in the order of their slots; ``None`` for an instance that
    hasn't run.

    Args:
        not_taken: What a skipped unit of the node holds for its outputs.
        not_called: What a unit of the node that wasn't called for a blank holds for its outputs.
        saved: The columns.

    Raises:
        PydanticCustomError: An output doesn't hold one slot for each state (``slots_differ``); or a skip, a
            question or a blank input is placed on a slot that can't hold one (``misplaced``).
    """
    states, outputs = saved.states, saved.outputs
    if any(len(column) != len(states) for column in outputs.values()):
        raise PydanticCustomError("slots_differ", "an output doesn't hold one slot for each state")
    try:
        for name, slots in (saved.skipped or {}).items():
            for slot in slots:
                outputs[name][slot] = SKIPPED
        for name, by_slot in (saved.skipped_items or {}).items():
            for slot, places in by_slot.items():
                for place in places:
                    outputs[name][slot][place] = SKIPPED
        if len(outputs) == 1:  # most nodes have one output
            ((name, column),) = outputs.items()
            outs = [{name: value} for value in column]
        else:
            names = tuple(outputs)
            outs = [dict(zip(names, values, strict=True)) for values in zip(*outputs.values(), strict=True)] if outputs else [{} for _ in states]
        done, skipped = SAVED_DONE, SAVED_SKIPPED
        not_taken_unit = Skipped(not_taken)
        units: list[Unit | None] = [Done(out) if state == done else not_taken_unit if state == skipped else None for state, out in zip(states, outs, strict=True)]
        for slot, question in (saved.questions or {}).items():
            answered = units[slot]
            if states[slot] is None:
                units[slot] = Asking(question)
            elif type(answered) is Done:
                units[slot] = Answered(answered.out, question)
            else:
                raise LookupError(slot)
        for slot, blank in (saved.blank_inputs or {}).items():
            if units[slot] is not None:
                raise LookupError(slot)
            units[slot] = Blank(not_called, blank)
    except (LookupError, TypeError, AttributeError):  # a slot the columns don't have, or one that holds another kind of unit
        raise PydanticCustomError("misplaced", "a skip, a question or a blank input is placed on a slot that can't hold one") from None
    return units


def with_shared_tables(outputs: Mapping[str, Mapping[str, Field]], saved: SavedRun[list[Unit | None], EarlierWork]) -> SavedRun[list[Unit | None], EarlierWork]:
    """Gives each output a save wrote as ``null`` under ``same_as`` the table of the output it names, validated by
    its own type, so both outputs have one table object again.

    Args:
        outputs: The outputs of each node, by the node's name in a save.
        saved: The save as read, each node's units in slot order.

    Raises:
        PydanticCustomError: An entry names an output or slot the save does not have, one that has no table
            written in full, one whose own column is not ``null`` there, or a table its own type refuses
            (``same_as_unreadable``).
    """
    for entry in saved.same_as or ():
        (node, output, slot), (from_node, from_output, from_slot) = entry.output, entry.same_as
        try:
            unit, source = saved.units[node][slot], saved.units[from_node][from_slot]
            if not isinstance(unit, (Done, Answered)) or not isinstance(source, (Done, Answered)) or unit.out[output] is not None:
                raise LookupError(slot)
            table = outputs[from_node][from_output].whole.written_once(source.out[from_output])
            if table is None:
                raise LookupError(from_slot)
            unit.out[output] = outputs[node][output].whole.read_returned(table)
        except (LookupError, AttributeError, FieldValueError):
            raise PydanticCustomError(
                "same_as_unreadable", "same_as: '{output}' can't be read as the same table as '{same_as}', which must be an output written in full before it",
                {"output": f"{node}.{output}[{slot}]", "same_as": f"{from_node}.{from_output}[{from_slot}]"},
            ) from None
    return saved


def earlier_of(saved: SavedEarlier[list[Unit | None]]) -> EarlierWork:
    """Makes the earlier work a node's save holds.

    Raises:
        PydanticCustomError: It doesn't hold one fingerprint for each unit (``slots_differ``), or it carries a
            unit that was never called (``not_carried``).
    """
    if len(saved.fingerprints) != len(saved.units):
        raise PydanticCustomError("slots_differ", "it doesn't hold one fingerprint for each unit")
    places, lineages = saved.places or {}, saved.lineages or {}
    earlier = EarlierWork()
    for slot, (fingerprint, unit) in enumerate(zip(saved.fingerprints, saved.units, strict=True)):
        if type(unit) is Done:
            earlier.computed.setdefault(fingerprint, unit)
            if slot in places:
                earlier.at[places[slot]] = (fingerprint, unit)
        elif type(unit) is Answered or type(unit) is Asking:
            earlier.asked.setdefault((fingerprint, lineages.get(slot, "")), []).append(unit)
        else:
            raise PydanticCustomError("not_carried", "it carries a unit that was never called")
    return earlier
