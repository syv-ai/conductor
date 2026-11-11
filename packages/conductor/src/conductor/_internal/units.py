"""What one node came to for one item: the five kinds of unit, and the two ways a unit is not ready.

A unit is one node in one instance of its scope: the node's one call, or its
call for one item. Every unit that is settled holds its outputs by name, so
what reads an output reads ``unit.out[output]`` whatever the unit's kind:

- ``Done`` and ``Answered`` hold what the node returned or a person answered.
- ``Skipped`` (a branch above wasn't taken) holds ``SKIPPED`` for each output.
- ``Blank`` (an input that takes no blank got one) holds a blank for each
  output, and ``SKIPPED`` for an output of an exclusive group, as no branch
  was chosen.

``Asking`` waits for a person and holds no outputs: it is not settled, and
nor is an instance that holds no unit yet.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Any, Final, final

from conductor._internal.ask import Question


class NotReady(enum.Enum):
    """Why a unit can't be made yet.

    Attributes:
        NOT_SETTLED: A unit it reads holds no outputs yet: it hasn't run, or it waits for a person.
        NOT_ITS_TURN: A person's earlier answer is there for it to take, and a unit of its node before it,
            which may take that answer first, hasn't been tried.
    """

    NOT_SETTLED = "not_settled"
    NOT_ITS_TURN = "not_its_turn"


NOT_SETTLED: Final = NotReady.NOT_SETTLED
NOT_ITS_TURN: Final = NotReady.NOT_ITS_TURN


@final
@dataclass(frozen=True, slots=True)
class Done:
    """A unit whose node was called and returned.

    Attributes:
        out: Its outputs by name: a value, ``None`` for a blank, ``SKIPPED`` for a branch not taken (every one
            of them, when the node returned ``SKIPPED``), and for a ``Series`` a list, or ``Records`` when the
            node returned a table for a ``Series`` of records.
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

    It differs from ``Blank`` only in what a run says of it: ``not_called`` lists a blank unit and not a skipped
    one, a node whose every unit is skipped says ``node_skipped``, and a save writes the two apart. A reader of
    its outputs reads ``out`` and never asks which it is.

    Attributes:
        out: Its outputs by name, each ``SKIPPED``.
    """

    out: dict[str, Any]


@final
@dataclass(frozen=True, slots=True)
class Blank:
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
Settled = Done | Answered | Skipped | Blank

#: What one node came to in one instance of its scope. A run and its copies share their units, so none is changed.
Unit = Settled | Asking
