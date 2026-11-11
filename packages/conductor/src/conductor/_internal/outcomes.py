"""What one node came to for one item: the five kinds of unit, and the two ways a unit is not ready.

A unit is one node in one instance of its scope: the node's one call, or its
call for one item. Every unit that is settled holds its outputs by name, so
what reads an output reads ``unit.out[output]`` whatever the unit's kind:

- ``Done`` and ``Answered`` hold what the node returned or a person answered.
- ``Skipped`` (a branch above wasn't taken) holds ``SKIPPED`` for each output.
- ``NotCalled`` (an input that takes no blank got one) holds a blank for each
  output, and ``SKIPPED`` for an output of an exclusive group, as no branch
  was chosen.

``Asking`` waits for a person and holds no outputs: it is not settled, and
nor is an item that holds no unit yet. Which items a node has is read from
the units too: one for each item of each ``Series`` it runs per item of
(``item_paths``).
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, final

from conductor._internal.address import ItemPath
from conductor._internal.ask import Question
from conductor._internal.field import Field
from conductor._internal.sentinels import SKIPPED

if TYPE_CHECKING:
    from conductor._internal.checked_graph.tree import FunctionAtPath, Maps



class NotReady(enum.Enum):
    """Why a unit can't be made yet.

    Attributes:
        NOT_READY: A unit it reads holds no outputs yet: it hasn't run, or it waits for a person.
        NOT_ITS_TURN: A person's earlier answer is there for it to take, and a unit of its node before it,
            which may take that answer first, hasn't been tried.
    """

    NOT_READY = "not_ready"
    NOT_ITS_TURN = "not_its_turn"


NOT_READY: Final = NotReady.NOT_READY
NOT_ITS_TURN: Final = NotReady.NOT_ITS_TURN


@final
@dataclass(frozen=True, slots=True)
class Done:
    """A unit whose node was called and returned.

    Attributes:
        out: Its outputs by name: a value, ``None`` for a blank, ``SKIPPED`` for a branch not taken (every one
            of them, when the node returned ``SKIPPED``), and for a ``Series`` a ``StoredSeries`` (the
            list the node returned, or the table it returned for a ``Series`` of records).
    """

    out: dict[str, Any]


@final
@dataclass(frozen=True, slots=True)
class Answered:
    """A unit a person answered: its outputs are the answer.

    Attributes:
        out: Its outputs by name, as the answer was read.
        question: What the person was asked.
    """

    out: dict[str, Any]
    question: Question


@final
@dataclass(frozen=True, slots=True)
class Skipped:
    """A unit whose node wasn't called because a branch above it wasn't taken: an input came only from branches
    not taken. What reads it is skipped too.

    It differs from ``NotCalled`` only in what a run says of it: ``not_called`` lists a blank unit and not a skipped
    one, a node whose every unit is skipped says ``node_skipped``, and a save writes the two apart. A reader of
    its outputs reads ``out`` and never asks which it is.

    Attributes:
        out: Its outputs by name, each ``SKIPPED``.
    """

    out: dict[str, Any]


@final
@dataclass(frozen=True, slots=True)
class NotCalled:
    """A unit whose node wasn't called because an input that takes no blank got one.

    Attributes:
        out: Its outputs by name: each a blank, except those of an exclusive group, which are ``SKIPPED``, as
            no branch was chosen.
        input: The input that got nothing.
    """

    out: dict[str, Any]
    input: str


@final
@dataclass(frozen=True, slots=True)
class Asking:
    """A unit waiting for a person: its node raised ``Ask``. It holds no outputs, so what reads it waits.

    Attributes:
        question: What the person is asked.
    """

    question: Question


#: A unit whose outputs are known: its node was called, a person answered, or it wasn't called and what its
#: outputs read as follows from why.
Settled = Done | Answered | Skipped | NotCalled

#: What one node came to in one instance of its scope. A run and its copies share their units, so none is changed.
Outcome = Settled | Asking

def outputs_when_skipped(outputs: Mapping[str, Field]) -> dict[str, Any]:
    """What a ``Skipped`` unit of a node with these outputs holds: ``SKIPPED`` for each."""
    return dict.fromkeys(outputs, SKIPPED)


def outputs_when_not_called(outputs: Mapping[str, Field]) -> dict[str, Any]:
    """What a ``NotCalled`` unit of a node with these outputs holds: a blank for each, and ``SKIPPED`` for an output
    of an exclusive group, as no branch was chosen."""
    return {name: SKIPPED if output.param.exclusive_group is not None else None for name, output in outputs.items()}


#: Each function's units by item path, by the function's name (``e/ask``).
Units = dict[str, dict[ItemPath, Outcome]]


def item_paths(units: Units, maps: Maps, under: ItemPath, *, counted: bool = False) -> tuple[list[ItemPath], bool]:
    """The item paths of a node that runs in ``maps`` under the item path ``under`` of a node that runs in the
    first of them, and whether all of them exist yet: a Series not made yet, or one a person is asked for, has
    none so far.

    With ``counted``, the innermost map's items are only counted: ``[(n,)]`` stands for ``n`` of them, unbuilt.
    """
    start, deepest = len(under), len(maps)
    if start >= deepest:
        return [under[:deepest]], True
    paths, complete = [under], True
    for depth in range(start, deepest):
        series, output = maps[depth]
        held = units[series.name]
        below: list[ItemPath] = []
        for parent in paths:
            unit = held.get(parent)
            if unit is None or type(unit) is Asking:
                complete = False
                continue
            count = length(unit.out[output])
            if counted and depth == deepest - 1:
                below = [(below[0][0] + count,)] if below else [(count,)]
            else:
                below.extend((*parent, i) for i in range(count))
        paths = below
    return paths, complete


def has_every_outcome(units: Units, node: FunctionAtPath) -> bool:
    """Whether every instance of ``node`` holds a unit, one waiting for a person included, and every Series
    it runs per item of is made, so no instance is still to come."""
    held = units[node.name]
    if not node.maps:
        return bool(held)
    counted, known = item_paths(units, node.maps, (), counted=True)
    return known and len(held) == (counted[0][0] if counted else 0)


def length(value: Any) -> int:
    """How many items a ``Series`` output holds; none when it is blank or a branch not taken."""
    return 0 if value is SKIPPED or value is None else len(value)
