"""Reading a save: the units and the carried work of a node, from their columns."""

from __future__ import annotations

from pydantic_core import PydanticCustomError

from conductor._internal.earlier_work import Earlier
from conductor._internal.fields import SKIPPED
from conductor._internal.saved_run.type import _DONE, _SKIPPED, SavedEarlier, SavedUnits
from conductor._internal.units import SKIPPED_UNIT, Answered, Asking, Blank, Done, Unit


def _units_of(saved: SavedUnits) -> list[Unit | None]:
    """Makes the units a node's saved columns hold, in the order of their slots; ``None`` for an instance that
    hasn't run.

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
        done, skipped = _DONE, _SKIPPED
        units: list[Unit | None] = [Done(out) if state == done else SKIPPED_UNIT if state == skipped else None for state, out in zip(states, outs, strict=True)]
        for slot, question in (saved.questions or {}).items():
            units[slot] = Asking(question) if states[slot] is None else Answered(units[slot].out, question)  # type: ignore[union-attr]
        for slot, blank in (saved.blank_inputs or {}).items():
            if units[slot] is not None:
                raise LookupError(slot)
            units[slot] = Blank(blank)
    except (LookupError, TypeError, AttributeError):  # a slot the columns don't have, or one that holds another kind of unit
        raise PydanticCustomError("misplaced", "a skip, a question or a blank input is placed on a slot that can't hold one") from None
    return units


def _earlier_of(saved: SavedEarlier[list[Unit | None]]) -> Earlier:
    """Makes the carried work a node's save holds.

    Raises:
        PydanticCustomError: It doesn't hold one fingerprint for each unit (``slots_differ``), or it carries a
            unit that was never called (``not_carried``).
    """
    if len(saved.fingerprints) != len(saved.units):
        raise PydanticCustomError("slots_differ", "it doesn't hold one fingerprint for each unit")
    places, lineages = saved.places or {}, saved.lineages or {}
    earlier = Earlier()
    for slot, (fingerprint, unit) in enumerate(zip(saved.fingerprints, saved.units, strict=True)):
        kind = type(unit)
        if kind is Done:
            earlier.computed.setdefault(fingerprint, unit)  # type: ignore[arg-type]
            if slot in places:
                earlier.at[places[slot]] = (fingerprint, unit)  # type: ignore[assignment]
        elif kind is Answered or kind is Asking:
            earlier.asked.setdefault((fingerprint, lineages.get(slot, "")), []).append(unit)  # type: ignore[arg-type]
        else:
            raise PydanticCustomError("not_carried", "it carries a unit that was never called")
    return earlier
