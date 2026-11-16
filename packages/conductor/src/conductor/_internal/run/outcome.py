"""What happened to one call of a node: its outcome.

A run calls each node once, or once per item when the node is inside a loop. Take the README's graph::

    lines(text: str) -> Series[str]           # called once; returns ["a", "b", "c"]
    shout(line: str, mark: str = "!") -> str  # wired to lines.result: called once per line
    join(texts: Series[str]) -> str           # wired to shout.result: called once, with ["A!", "B!", "C!"]

``graph.run(text="a\\nb\\nc")`` calls ``lines`` once, ``shout`` three times and ``join`` once. Each call is known
by its item indexes, one index per loop the node is inside. The calls of ``shout`` are ``(0,)``, ``(1,)`` and
``(2,)``. ``lines`` and ``join`` are inside no loop, so their one call is ``()``.

The outcome of a call is what the run stores about it. A run stores its outcomes by node path, then by item
indexes::

    {("lines",): {():   Done({"result": ["a", "b", "c"]})},
     ("shout",): {(0,): Done({"result": "A!"}), (1,): Done({"result": "B!"}), (2,): Done({"result": "C!"})},
     ("join",):  {():   Done({"result": "A!+B!+C!"})}}

A call has at most one outcome, and an outcome never changes once it is stored. There are four kinds, one class
each:

- ``Done``: the call has its outputs. Either the function returned them, or it asked a person and the person
  answered.
- ``Unanswered``: the function raised ``Ask`` to ask a person, and no answer has come yet.
- ``Skipped``: the function wasn't called, because the node is downstream of a branch that wasn't chosen.
- ``NoneInput``: the function wasn't called, because an argument was ``None`` and its parameter doesn't take
  ``None``.

A call with no outcome yet hasn't been made, or it failed. A failed call stores nothing, so ``resume()`` calls
the function again.

A node is complete when every call it will have has an outcome and none of them is ``Unanswered``. A run is
complete when every node is.

``Done``, ``Skipped`` and ``NoneInput`` have ``outputs``. ``Unanswered`` has none yet. Each class says which with
``has_outputs``, so code that wants an output checks ``outcome.has_outputs`` and then reads
``outcome.outputs[name]``, whatever kind of outcome it has.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Literal, final

from pydantic import JsonValue, SerializeAsAny

from conductor._internal.types import Choice


class NotReady(Enum):
    """Why a call can't be made yet.

    The run returns this in place of a call's arguments, and tries the call again later.

    Attributes:
        NOT_READY: A value the call needs doesn't exist yet. The node it comes from hasn't returned, or is waiting
            for a person. Before ``lines`` has returned, every call of ``shout`` is ``NOT_READY``.
        NOT_ITS_TURN: An earlier call of the same node has to take its reused outcome first. This only happens
            for a node that may ask a person, since the answers it got are handed out in call order. Say a
            previous run of ``lines -> approve`` had the line ``"longer"`` twice, and the person answered ``True``
            to the first and ``False`` to the second. A new run on the same text gives ``True`` to ``(0,)`` and
            ``False`` to ``(1,)``. If ``(1,)`` is ready first, it is ``NOT_ITS_TURN`` until ``(0,)`` has taken its
            answer.
    """

    NOT_READY = "not_ready"
    NOT_ITS_TURN = "not_its_turn"


@final
@dataclass(frozen=True, slots=True)
class Done:
    """The call has its outputs, such as ``Done({"result": "B!"})`` for the call ``(1,)`` of ``shout``.

    Either the function returned the outputs, or it raised ``Ask`` and the outputs are the person's answer. The
    outcome looks the same either way. Only reuse tells them apart, by whether the node may ask (``reuse.py``).

    Attributes:
        outputs: The call's outputs by name. An output may be ``None``, or ``SKIPPED`` when it is a branch that
            wasn't chosen. A ``Series`` output is stored as a ``StoredSeries``, which reads like the list the
            function returned.
        has_outputs: ``True``.
    """

    outputs: dict[str, Any]
    has_outputs: ClassVar[Literal[True]] = True


@final
@dataclass(frozen=True, slots=True)
class Skipped:
    """The function wasn't called, because the node is downstream of a branch that wasn't chosen.

    A branch is one of several outputs that share a ``Param(exclusive_group=...)``. On each call, the node gives a
    value to exactly one of them and ``SKIPPED`` to the rest. Say ``route(line)`` has the branches ``short`` and
    ``long``, and ``shout`` is wired to ``route.long``. For the line ``"a"``, ``route`` picks ``short``, so that call
    of ``shout`` is ``Skipped({"result": SKIPPED})``. So is the call for that line of every node wired to
    ``shout.result``.

    ``Skipped`` and ``NoneInput`` have the same kind of ``outputs``. They are two classes because a host is told
    about them differently: ``run.not_called`` lists the ``NoneInput`` calls, and no list shows the skipped ones.

    Attributes:
        outputs: The call's outputs by name, each ``SKIPPED``.
        has_outputs: ``True``.
    """

    outputs: dict[str, Any]
    has_outputs: ClassVar[Literal[True]] = True


@final
@dataclass(frozen=True, slots=True)
class NoneInput:
    """The function wasn't called, because an argument was ``None`` and its parameter doesn't take ``None``.

    Say ``strip(line: str) -> str | None`` sits between ``lines`` and ``shout``, and returns ``None`` for the line
    ``" "``. ``shout(line: str)`` doesn't take ``None``, so its call for that line is
    ``NoneInput({"result": None}, input="line")``.

    Attributes:
        outputs: The call's outputs by name, each ``None``. An output that is a branch is ``SKIPPED`` instead,
            since no branch was chosen.
        input: The name of the input that got ``None``.
        has_outputs: ``True``.
    """

    outputs: dict[str, Any]
    input: str
    has_outputs: ClassVar[Literal[True]] = True


@final
@dataclass(frozen=True, slots=True)
class Unanswered:
    """The function raised ``Ask`` to ask a person, and no answer has come yet.

    It stores what the ``Ask`` said, after ``AskValidator`` has checked it, and it is saved with the run. Say
    ``assign(ticket: str) -> str`` raises ``Ask("Who?", choices=["ida", "bo"])``. Its outcome is
    ``Unanswered("Who?", {"result": (Choice("ida"), Choice("bo"))})``.

    The arguments the call got aren't stored. When a host asks for ``run.waiting``, they are worked out again from
    the run's outcomes. There are no outputs yet, so a call that needs one of them isn't ready.

    Attributes:
        prompt: The text the person reads.
        choices: The choices the ``Ask`` offered, by output name, each id in its JSON form. An output the ``Ask``
            set no choices for isn't named.
        defaults: The answer the ``Ask`` proposed, by output name, in its JSON form. An output it proposed nothing
            for isn't named.
        has_outputs: ``False``.
    """

    prompt: str
    choices: Mapping[str, tuple[SerializeAsAny[Choice], ...]] = dataclasses.field(
        default_factory=dict
    )
    defaults: Mapping[str, JsonValue] = dataclasses.field(default_factory=dict)
    has_outputs: ClassVar[Literal[False]] = False


#: What happened to one call: one of the four kinds of outcome.
Outcome = Done | Skipped | NoneInput | Unanswered
