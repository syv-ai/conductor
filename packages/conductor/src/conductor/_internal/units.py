"""A unit: what one node came to in one instance of its scope.

A unit is ``Done`` (its outputs kept), ``Answered`` (done by a person), ``Skipped`` (a branch not taken),
``Blank`` (an input got nothing, so the node wasn't called) or ``Asking`` (it waits for a person).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from conductor._internal.ask import Question


@dataclass(frozen=True, slots=True)
class Done:
    """A unit whose node was called and returned.

    Attributes:
        out: Its outputs by name: a value, ``None`` for a blank, ``SKIPPED`` for a branch not taken (every one
            of them, when the node returned ``SKIPPED``), and for a ``Series`` a list, or ``Records`` when the
            node returned a table for a ``Series`` of records.
    """

    out: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Answered(Done):
    """A unit a person answered: its outputs are the answer.

    Attributes:
        question: What the person was asked.
    """

    question: Question


@dataclass(frozen=True, slots=True)
class Skipped:
    """A unit whose node wasn't called because a branch above it wasn't taken: an input came only from branches
    not taken. What reads it is skipped too."""


@dataclass(frozen=True, slots=True)
class Blank:
    """A unit whose node wasn't called because an input that takes no blank got one. Its outputs are blank,
    except those of an exclusive group, which are a branch not taken: no branch was chosen.

    Attributes:
        input: The input that got nothing.
    """

    input: str


@dataclass(frozen=True, slots=True)
class Asking:
    """A unit waiting for a person: its node raised ``Ask``.

    Attributes:
        question: What the person is asked.
    """

    question: Question


#: What one node came to in one instance of its scope. A run and its copies share their units, so none is changed.
Unit = Done | Answered | Skipped | Blank | Asking

#: The one skipped unit: it holds nothing, so every node not called for a branch not taken shares it.
SKIPPED_UNIT = Skipped()
