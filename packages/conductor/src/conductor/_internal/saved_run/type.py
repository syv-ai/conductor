"""A run as a save holds it: the records a save is declared by, whose keys are the save's keys."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, get_args

from pydantic import ConfigDict, NonNegativeInt, PositiveInt
from pydantic import Field as PydanticField

from conductor._internal.ask import Question
from conductor._internal.run_status import EndedEarly, Going

#: Every record of a save refuses a key it doesn't declare, and writes a float that isn't finite as its text.
_SAVED = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

#: What a save says of a unit that holds outputs, and of one that is a branch not taken; a unit with neither
#: word holds no outputs.
UnitState = Literal["done", "skipped"]
_DONE, _SKIPPED = get_args(UnitState)


#: The slots of one output whose value is a branch not taken.
SkippedSlots = list[NonNegativeInt]

#: By slot, the items of a ``Series`` output there that are a branch not taken.
SkippedItems = dict[NonNegativeInt, list[NonNegativeInt]]


@dataclass(slots=True, kw_only=True)
class SavedUnits:
    """Units of one node as a save holds them: columns with a slot for each unit. A node's own units have a slot
    for each instance of the node, in instance order, so an instance is a place in them and is never written.

    The record is what is written and what is read: a key of a save is a field here and nowhere else.

    Attributes:
        states: What each unit came to; ``None`` for a unit that holds no outputs.
        outputs: By output, its value in each slot, read and written by the output's own type (the one part
            composed for each graph); ``None`` where the unit holds none.
        questions: By slot, the question a unit asked; ``None``, as for each field after it, when there is none,
            which leaves the key out of the save. On a slot that isn't done the unit waits for a person, so a
            waiting unit can't be written without its question.
        blank_inputs: By slot, the input that got a blank, for a unit that wasn't called for it.
        skipped: By output, the slots whose value is a branch not taken (``SKIPPED``); ``None`` in the column.
            A graph's form takes it for the node's own outputs only.
        skipped_items: By output and slot, the items of a ``Series`` there that are a branch not taken; ``None``
            in the column. A graph's form takes it for the node's ``Series`` outputs only, so an output of one
            value can't be saved with any.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _SAVED
    states: list[UnitState | None]
    outputs: dict[str, list[Any]]
    questions: dict[NonNegativeInt, Question] | None = None
    blank_inputs: dict[NonNegativeInt, str] | None = None
    skipped: dict[str, SkippedSlots] | None = None
    skipped_items: dict[str, SkippedItems] | None = None


@dataclass(slots=True, kw_only=True)
class SavedEarlier[UnitsT]:
    """The work of one node carried from an earlier run, as a save holds it: each carried unit once.

    Attributes:
        units: The carried units, a slot for each: a ``SavedUnits`` when written, the units it holds when read.
        fingerprints: For each slot, the fingerprint of what the unit received.
        places: By slot, the instance a computed unit stood in in the earlier run; ``None``, as for ``lineages``,
            when there is none, which leaves the key out of the save.
        lineages: By slot, for an answer or a question still waiting, the fingerprint of the items above it.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _SAVED
    units: UnitsT
    fingerprints: list[str]
    places: dict[NonNegativeInt, tuple[NonNegativeInt, ...]] | None = None
    lineages: dict[NonNegativeInt, str] | None = None


@dataclass(slots=True, kw_only=True)
class SavedRun[UnitsT, EarlierT]:
    """A run as a save holds it. A node is named by its id after the ids of the embedded graphs' nodes it is
    inside, joined by ``/`` (``e/ask``).

    Attributes:
        graph: The fingerprint of the run's graph, so a save of another graph is told apart before anything in
            it is read.
        state: How the run stands, with its cause when a leg ended early.
        max_items: The item ceiling the run goes on under.
        inputs: The caller's inputs by address, each read and written by its field's type.
        units: By node, the units of each function node that has any: a ``SavedUnits`` when written, the units
            it holds when read. An embedded graph's node has none of its own.
        earlier: By node, the work carried from an earlier run and not yet taken: a ``SavedEarlier`` when
            written, an ``Earlier`` when read.
        code: By node, the ``reuse_digest`` of the code that made its units and its carried work, so work made
            by other code is never taken.
        complete: The nodes every unit of which was finished when the last leg ended.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _SAVED
    graph: str
    state: Annotated[Going | EndedEarly, PydanticField(discriminator="status")]
    max_items: PositiveInt | None
    inputs: dict[str, Any]
    units: dict[str, UnitsT]
    earlier: dict[str, EarlierT]
    code: dict[str, str]
    complete: list[str]
