"""What happened to one node for one call: the outcomes, and the words for them.

The example graph, the README's first::

    lines(text: str) -> Series[str]           # called once; returns ["a", "b", "c"]
    shout(line: str, mark: str = "!") -> str  # wired to lines.result: called once per line
    join(texts: Series[str]) -> str           # wired to shout.result: called once, with ["A!", "B!", "C!"]

``graph.run(text="a\\nb\\nc")`` calls ``lines`` once, ``shout`` three times and ``join`` once. Each call has
item indexes: one index per loop the node is inside. The three calls of ``shout`` have ``(0,)``, ``(1,)`` and
``(2,)``; ``lines`` and ``join`` are inside no loop, so their one call has ``()``. An **outcome** is what a run
keeps of one call. A run keeps its outcomes in a dict by node path and then by item indexes::

    {("lines",): {():   Done({"result": ["a", "b", "c"]})},
     ("shout",): {(0,): Done({"result": "A!"}), (1,): Done({"result": "B!"}), (2,): Done({"result": "C!"})},
     ("join",):  {():   Done({"result": "A!+B!+C!"})}}

A call has at most one outcome, and an outcome is never changed once it is written. There are four kinds, a
class each. The examples add one small node to the graph above.

- **done** (``Done``): the call has its outputs: the function returned them, or it asked a person and the
  person answered. Every outcome above is done.
- **unanswered** (``Unanswered``): the function asked a person a question, by raising ``Ask``, and no answer has come
  yet. A node ``approve(line: str) -> bool``, wired to ``lines.result``, raises ``Ask("Approve longer?")`` for
  the line ``"longer"``. Its call for that line is ``Unanswered("Approve longer?")``. When the person answers
  ``False``, the answer is the call's outputs, and the call is ``Done({"result": False})``.
- **skipped** (``Skipped``): the function was not called, because the node is downstream of a branch that was
  not chosen. A **branch** is one of several outputs, declared with the same ``Param(exclusive_group=...)``, of
  which a node gives a value to exactly one per call and ``SKIPPED`` to the others. A node ``route(line)`` with
  the branches ``short`` and ``long`` gives ``long`` a value for ``"longer"`` and ``short`` one for ``"a"``. With
  ``shout`` wired to ``route.long``, the call of ``shout`` for ``"a"`` is ``Skipped({"result": SKIPPED})``.
- **none input** (``NoneInput``): the function was not called, because an argument was ``None`` and the
  parameter does not accept ``None``. A node ``strip(line: str) -> str | None`` between ``lines`` and ``shout``
  returns ``None`` for the line ``" "``. ``shout``'s ``line: str`` does not accept ``None``, so that call of
  ``shout`` is ``NoneInput({"result": None}, input="line")``.

Two more words have no outcome of their own:

- **not ready** (``NotReady``): a call cannot be made yet, because a value it needs does not exist yet. Before
  ``lines`` has returned, ``shout`` is not ready. It is tried again later. (``NotReady`` has one more reason,
  for reusing in item order the outcomes of a node that may ask; its docstring has it.)
- **failed**: the function raised an exception other than ``Ask``. The run stops and keeps no outcome for the
  call, so ``resume()`` calls the function again.

A node is **complete** when every call it will have has an outcome and none of them is unanswered; a run is
complete when every node is.

Done, skipped and not called have ``outputs``: the call's outputs by name. Unanswered has none yet. Every class says
which with ``has_outputs``, so code that needs an output asks ``outcome.has_outputs`` and then reads
``outcome.outputs[name]``, whichever of the four classes it has.
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
    """Why a call cannot be made yet. Returned in place of a call's arguments; the call is tried again later.

    Attributes:
        NOT_READY: A value the call needs does not exist yet: the node it comes from has not returned, or is
            waiting for a person. Before ``lines`` has returned, every call of ``shout`` is ``NOT_READY``.
        NOT_ITS_TURN: The call is of a node that may ask a person, it would reuse an outcome from a previous run,
            and an earlier call of the same node has to reuse its outcome first. Answers to the same question are
            reused in the order of the calls. Say a previous run of ``lines -> approve`` had the lines ``"longer"`` and ``"longer"``, and a
            person answered ``True`` for the first and ``False`` for the second. Both calls ask the same question,
            since they have the same arguments, so a new run on the same text gives ``True`` to ``(0,)`` and ``False`` to ``(1,)``. If ``(1,)`` is
            ready first, it is ``NOT_ITS_TURN`` until ``(0,)`` has taken its answer.
    """

    NOT_READY = "not_ready"
    NOT_ITS_TURN = "not_its_turn"



@final
@dataclass(frozen=True, slots=True)
class Done:
    """The call has its outputs: ``Done({"result": "B!"})`` for the call ``(1,)`` of ``shout``.

    The function returned them, or it asked a person, by raising ``Ask``, and the outputs are the person's answer.
    A run tells the two apart only when it reuses outcomes, by whether the node may ask (``run/reuse.py``).

    Attributes:
        outputs: The call's outputs by name. An output is the value the function returned or the person answered
            for it, which may be ``None``, or ``SKIPPED`` for an output on a branch that was not chosen. The value of a ``Series``
            output is a ``StoredSeries``, the form in which a run keeps a series: it reads as the returned list,
            and ``len()`` gives its number of items.
        has_outputs: ``True``.
    """

    outputs: dict[str, Any]
    has_outputs: ClassVar[Literal[True]] = True


@final
@dataclass(frozen=True, slots=True)
class Skipped:
    """The function was not called, because the node is downstream of a branch that was not chosen.

    With ``shout`` wired to ``route.long``, the call of ``shout`` for the line ``"a"`` is
    ``Skipped({"result": SKIPPED})``, and so is the call for that line of any node in the loop that is wired to
    ``shout.result``.

    ``Skipped`` and ``NoneInput`` have the same ``outputs``. They are two classes because a run reports them
    differently: ``run.not_called`` lists the calls that were not called, and no list has the skipped ones.

    Attributes:
        outputs: The call's outputs by name. Each is ``SKIPPED``.
        has_outputs: ``True``.
    """

    outputs: dict[str, Any]
    has_outputs: ClassVar[Literal[True]] = True


@final
@dataclass(frozen=True, slots=True)
class NoneInput:
    """The function was not called, because an argument was ``None`` and the parameter does not accept ``None``.

    With ``strip`` returning ``None`` for the line ``" "``, the call of ``shout`` for that line is
    ``NoneInput({"result": None}, input="line")``.

    Attributes:
        outputs: The call's outputs by name. Each is ``None``, except an output that is a branch: no branch was
            chosen, so it is ``SKIPPED``.
        input: The name of the input that was ``None``.
        has_outputs: ``True``.
    """

    outputs: dict[str, Any]
    input: str
    has_outputs: ClassVar[Literal[True]] = True


@final
@dataclass(frozen=True, slots=True)
class Unanswered:
    """The function asked a person a question, by raising ``Ask``, and no answer has come yet.

    It keeps what the ``Ask`` said, as the ``Ask`` door read it, and is saved with the run. ``Ask("Who?",
    choices=["ida", "bo"])`` raised by ``assign(ticket: str) -> str`` is ``Unanswered("Who?", {"result":
    (Choice("ida"), Choice("bo"))})``. What the call received is not kept: a host is shown it read again from the
    outcomes the call reads (``run.waiting``). There are no outputs yet, so a call that needs one of them is not
    ready.

    Attributes:
        prompt: What the person reads.
        choices: The choices the ``Ask`` offered, by output name, each id read by the output's type in its JSON form
            (``read_choices``); an output it set none on is not named.
        defaults: The answer the ``Ask`` proposed, by output name, in its JSON form; an output it proposed none for is
            not named.
        has_outputs: ``False``.
    """

    prompt: str
    choices: Mapping[str, tuple[SerializeAsAny[Choice], ...]] = dataclasses.field(default_factory=dict)
    defaults: Mapping[str, JsonValue] = dataclasses.field(default_factory=dict)
    has_outputs: ClassVar[Literal[False]] = False


#: What happened to one call: one of the four kinds of outcome.
Outcome = Done | Skipped | NoneInput | Unanswered
