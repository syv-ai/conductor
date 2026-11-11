"""Reading a save: the units and the earlier work of a node, from their columns."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic_core import PydanticCustomError

from conductor._internal.earlier_work import EarlierWork
from conductor._internal.field import Field
from conductor._internal.outcomes import Answered, Asking, Done, NotCalled, Outcome, Skipped
from conductor._internal.saved_run.type import (
    SAVED_DONE,
    SAVED_SKIPPED,
    SavedEarlier,
    SavedRun,
    SavedUnits,
)
from conductor._internal.sentinels import SKIPPED
from conductor._internal.value_type import FieldValueError


def units_of(not_taken: dict[str, Any], not_called: dict[str, Any], saved: SavedUnits) -> list[Outcome | None]:
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
        skipped_outcome = Skipped(not_taken)
        units: list[Outcome | None] = [Done(out) if state == done else skipped_outcome if state == skipped else None for state, out in zip(states, outs, strict=True)]
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
            units[slot] = NotCalled(not_called, blank)
    except (LookupError, TypeError, AttributeError):  # a slot the columns don't have, or one that holds another kind of unit
        raise PydanticCustomError("misplaced", "a skip, a question or a blank input is placed on a slot that can't hold one") from None
    return units


def with_shared_tables(outputs: Mapping[str, Mapping[str, Field]], saved: SavedRun[list[Outcome | None], EarlierWork]) -> SavedRun[list[Outcome | None], EarlierWork]:
    """Fills in the outputs a save wrote as ``null`` and listed under ``tables_written_once``.

    Example: the save lists ``{"output": ["split", "result", 0], "same_as": ["people", "result", 0]}``. The table
    read for ``people.result`` is validated by the type of ``split.result`` and stored there, so both outputs have
    the same table object again.

    Args:
        outputs: The outputs of each node, by the node's name in a save.
        saved: The save as read, each node's units in slot order.

    Raises:
        PydanticCustomError: An entry names an output or slot the save does not have, one that has no table
            written in full, one whose own column is not ``null`` there, or a table its own type refuses
            (``same_as_unreadable``).
    """
    for entry in saved.tables_written_once or ():
        (node, output, slot), (from_node, from_output, from_slot) = entry.output, entry.same_as
        try:
            outcome, source = saved.units[node][slot], saved.units[from_node][from_slot]
            if not isinstance(outcome, (Done, Answered)) or not isinstance(source, (Done, Answered)) or outcome.out[output] is not None:
                raise LookupError(slot)
            table = outputs[from_node][from_output].whole.written_once(source.out[from_output])
            if table is None:
                raise LookupError(from_slot)
            outcome.out[output] = outputs[node][output].whole.read_returned(table)
        except (LookupError, AttributeError, FieldValueError) as wrong:
            reason = wrong.reason if isinstance(wrong, FieldValueError) else "that is not an output written in full before it"
            raise PydanticCustomError(
                "same_as_unreadable", "tables_written_once: '{output}' can't be read as the table at '{same_as}': {reason}",
                {"output": f"{node}.{output}[{slot}]", "same_as": f"{from_node}.{from_output}[{from_slot}]", "reason": reason},
            ) from None
    return saved


def earlier_of(saved: SavedEarlier[list[Outcome | None]]) -> EarlierWork:
    """Makes the earlier work a node's save holds.

    Raises:
        PydanticCustomError: It doesn't hold one fingerprint for each unit (``slots_differ``), or it carries a
            unit that was never called (``not_carried``).
    """
    if len(saved.fingerprints) != len(saved.units):
        raise PydanticCustomError("slots_differ", "it doesn't hold one fingerprint for each unit")
    places, lineages = saved.places or {}, saved.lineages or {}
    earlier = EarlierWork()
    for slot, (fingerprint, outcome) in enumerate(zip(saved.fingerprints, saved.units, strict=True)):
        if type(outcome) is Done:
            earlier.computed.setdefault(fingerprint, outcome)
            if slot in places:
                earlier.at[places[slot]] = (fingerprint, outcome)
        elif type(outcome) is Answered or type(outcome) is Asking:
            earlier.asked.setdefault((fingerprint, lineages.get(slot, "")), []).append(outcome)
        else:
            raise PydanticCustomError("not_carried", "it carries a unit that was never called")
    return earlier
