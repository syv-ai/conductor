"""What happened to one node for one call: the outcomes, and the words for them.

The example graph, the README's first::

    lines(text: str) -> Series[str]           # called once; returns ["a", "b", "c"]
    shout(line: str, mark: str = "!") -> str  # wired to lines.result: called once per line
    join(texts: Series[str]) -> str           # wired to shout.result: called once, with ["A!", "B!", "C!"]

``graph.run(text="a\\nb\\nc")`` calls ``lines`` once, ``shout`` three times and ``join`` once. Each call has
item indexes: one index per loop the node is inside. The three calls of ``shout`` have ``(0,)``, ``(1,)`` and
``(2,)``; ``lines`` and ``join`` are inside no loop, so their one call has ``()``. An **outcome** is what a run
keeps of one call. A run keeps its outcomes in a dict by node name and then by item indexes::

    {"lines": {():   Done({"result": ["a", "b", "c"]})},
     "shout": {(0,): Done({"result": "A!"}), (1,): Done({"result": "B!"}), (2,): Done({"result": "C!"})},
     "join":  {():   Done({"result": "A!+B!+C!"})}}

A call has at most one outcome, and an outcome is never changed once it is written. There are five kinds, a
class each. The examples add one small node to the graph above.

- **done** (``Done``): the function was called and returned. Every outcome above is done.
- **waiting** (``Waiting``): the function asked a person a question, by raising ``Ask``, and no answer has come
  yet. A node ``approve(line: str) -> bool``, wired to ``lines.result``, raises ``Ask("Approve longer?")`` for
  the line ``"longer"``. Its call for that line is ``Waiting(question=...)``.
- **answered** (``Answered``): the function asked a person, and the person answered. The answer is the call's
  outputs: when the person answers ``False``, the call of ``approve`` is ``Answered({"result": False}, ...)``.
- **skipped** (``Skipped``): the function was not called, because the node is downstream of a branch that was
  not chosen. A **branch** is one of several outputs of which a node returns exactly one per call; the others
  get the value ``SKIPPED``. A node ``route(line)`` with the outputs ``short`` and ``long`` returns ``long`` for
  ``"longer"`` and ``short`` for ``"a"``. With ``shout`` wired to ``route.long``, the call of ``shout`` for
  ``"a"`` is ``Skipped({"result": SKIPPED})``.
- **not called** (``NotCalled``): the function was not called, because an argument was ``None`` and the
  parameter does not accept ``None``. A node ``strip(line: str) -> str | None`` between ``lines`` and ``shout``
  returns ``None`` for the line ``" "``. ``shout``'s ``line: str`` does not accept ``None``, so that call of
  ``shout`` is ``NotCalled({"result": None}, input="line")``.

Two more words have no outcome of their own:

- **not ready** (``NotReady``): a call cannot be made yet, because a value it needs does not exist yet. Before
  ``lines`` has returned, ``shout`` is not ready. It is tried again later.
- **failed**: the function raised an exception other than ``Ask``. The run stops and keeps no outcome for the
  call, so ``resume()`` calls the function again.

A node is **complete** when every call it will have has an outcome; a run is complete when every node is.

Done, answered, skipped and not called have ``outputs``: the call's outputs by name. Waiting has none yet. Every
class says which with ``has_outputs``, so code that needs an output asks ``outcome.has_outputs`` and then reads
``outcome.outputs[name]``, whichever of the five classes it has.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Literal, final

from conductor._internal.address import ItemIndexes
from conductor._internal.ask import Question
from conductor._internal.field import Field
from conductor._internal.markers import SKIPPED

if TYPE_CHECKING:
    from conductor._internal.compiled.flat_nodes import FlatFunctionNode



class NotReady(Enum):
    """Why a call cannot be made yet. Returned in place of a call's arguments; the call is tried again later.

    Attributes:
        NOT_READY: A value the call needs does not exist yet: the node it comes from has not returned, or is
            waiting for a person. Before ``lines`` has returned, every call of ``shout`` is ``NOT_READY``.
        NOT_ITS_TURN: The call would reuse a person's answer from a previous run, and an earlier call of the same
            node has to reuse its answer first. Answers to the same question are reused in the order of the
            calls. Say a previous run of ``lines -> approve`` had the lines ``"longer"`` and ``"longer"``, and a
            person answered ``True`` for the first and ``False`` for the second. Both calls ask the same question,
            so a new run on the same text gives ``True`` to ``(0,)`` and ``False`` to ``(1,)``. If ``(1,)`` is
            ready first, it is ``NOT_ITS_TURN`` until ``(0,)`` has taken its answer.
    """

    NOT_READY = "not_ready"
    NOT_ITS_TURN = "not_its_turn"



@final
@dataclass(frozen=True, slots=True)
class Done:
    """The function was called and returned: ``Done({"result": "B!"})`` for the call ``(1,)`` of ``shout``.

    Attributes:
        outputs: The call's outputs by name. An output is the value the function returned for it, which may be
            ``None``, or ``SKIPPED`` for an output on a branch that was not chosen. The value of a ``Series``
            output is a ``StoredSeries``: the returned list, wrapped.
        has_outputs: ``True``.
    """

    outputs: dict[str, Any]
    has_outputs: ClassVar[Literal[True]] = True


@final
@dataclass(frozen=True, slots=True)
class Answered:
    """The function asked a person a question, and the person answered: ``Answered({"result": False}, question)``.

    Attributes:
        outputs: The call's outputs by name: the person's answer.
        question: What the person was asked.
        has_outputs: ``True``.
    """

    outputs: dict[str, Any]
    question: Question
    has_outputs: ClassVar[Literal[True]] = True


@final
@dataclass(frozen=True, slots=True)
class Skipped:
    """The function was not called, because the node is downstream of a branch that was not chosen.

    With ``shout`` wired to ``route.long``, the call of ``shout`` for the line ``"a"`` is
    ``Skipped({"result": SKIPPED})``, and so is the call of any node wired to ``shout.result`` for that line.

    ``Skipped`` and ``NotCalled`` have the same ``outputs``. They are two classes because a run reports them
    differently: ``run.not_called`` lists the calls that were not called, and no list has the skipped ones.

    Attributes:
        outputs: The call's outputs by name. Each is ``SKIPPED``.
        has_outputs: ``True``.
    """

    outputs: dict[str, Any]
    has_outputs: ClassVar[Literal[True]] = True


@final
@dataclass(frozen=True, slots=True)
class NotCalled:
    """The function was not called, because an argument was ``None`` and the parameter does not accept ``None``.

    With ``strip`` returning ``None`` for the line ``" "``, the call of ``shout`` for that line is
    ``NotCalled({"result": None}, input="line")``.

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
class Waiting:
    """The function asked a person a question, by raising ``Ask``, and no answer has come yet.

    There are no outputs yet, so a call that needs one of them is not ready.

    Attributes:
        question: What the person is asked.
        has_outputs: ``False``.
    """

    question: Question
    has_outputs: ClassVar[Literal[False]] = False


#: What happened to one call: one of the five kinds of outcome.
Outcome = Done | Answered | Skipped | NotCalled | Waiting

def outputs_when_skipped(outputs: Mapping[str, Field]) -> dict[str, Any]:
    """The ``outputs`` of a ``Skipped`` outcome of a node with these outputs: ``SKIPPED`` for each.

    For ``shout``: ``{"result": SKIPPED}``.
    """
    return dict.fromkeys(outputs, SKIPPED)


def outputs_when_not_called(outputs: Mapping[str, Field]) -> dict[str, Any]:
    """The ``outputs`` of a ``NotCalled`` outcome of a node with these outputs.

    For ``shout``: ``{"result": None}``. For ``route``, whose two outputs are branches, and a node with one more
    output ``note`` that is not: ``{"short": SKIPPED, "long": SKIPPED, "note": None}``. An output is a branch
    when its ``Param`` gives it an ``exclusive_group``.
    """
    return {name: SKIPPED if output.param.exclusive_group is not None else None for name, output in outputs.items()}


def all_item_indexes(
    outcomes: dict[str, dict[ItemIndexes, Outcome]], loops: tuple[tuple[FlatFunctionNode, str], ...], prefix: ItemIndexes,
) -> tuple[list[ItemIndexes], bool]:
    """The item indexes of every call a node will have, and whether that list is complete yet.

    ``shout`` is inside one loop, over ``lines.result``. Once ``lines`` has returned three lines,
    ``all_item_indexes(outcomes, shout.loops, ())`` returns ``([(0,), (1,), (2,)], True)``. Before ``lines`` has
    returned it returns ``([], False)``: no call is known yet, and more will come. ``lines`` is inside no loop,
    so for it the result is always ``([()], True)``: one call.

    Args:
        outcomes: The run's outcomes, by node name and then by item indexes. Every node has an entry, empty
            until its first outcome.
        loops: The loops the node is inside, outermost first. Each is the node that returns the series and the
            name of that output.
        prefix: Return only the item indexes that start with these. ``()`` returns all of them. With two nested
            loops, ``(1,)`` returns the calls for item 1 of the outer loop: ``(1, 0)``, ``(1, 1)``, ...

    Returns:
        The item indexes of each call known so far, and ``True`` when every series the node loops over exists.
    """
    start, deepest = len(prefix), len(loops)
    if start == deepest:  # A shortcut: the loop below returns the same. It saves about 1% of a run of a chain of 500 nodes.
        return [prefix], True
    every_item_indexes, complete = [prefix], True
    for depth in range(start, deepest):
        series_node, output = loops[depth]
        of_series_node = outcomes[series_node.name]
        one_loop_deeper: list[ItemIndexes] = []
        for item_indexes in every_item_indexes:
            outcome = of_series_node.get(item_indexes)
            if outcome is None or not outcome.has_outputs:
                complete = False
                continue
            one_loop_deeper.extend((*item_indexes, index) for index in range(length(outcome.outputs[output])))
        every_item_indexes = one_loop_deeper
    return every_item_indexes, complete


def calls_expected(outcomes: dict[str, dict[ItemIndexes, Outcome]], loops: tuple[tuple[FlatFunctionNode, str], ...]) -> tuple[int, bool]:
    """How many calls a node inside ``loops`` will have in this run, and whether that number is final yet.

    A node inside no loop has one call: for ``lines`` and ``join`` the result is ``(1, True)``. ``shout`` is
    inside the loop over ``lines.result``: ``(3, True)`` once ``lines`` has returned three lines, ``(0, False)``
    before. With two nested loops the inner series may differ in length; the number is the sum of their lengths.

    Args:
        outcomes: The run's outcomes, by node name and then by item indexes.
        loops: The loops the node is inside, outermost first, as ``all_item_indexes`` takes them.
    """
    if not loops:
        return 1, True
    calls_of_series_node, complete = all_item_indexes(outcomes, loops[:-1], ())
    series_node, output = loops[-1]
    of_series_node = outcomes[series_node.name]
    expected = 0
    for item_indexes in calls_of_series_node:
        outcome = of_series_node.get(item_indexes)
        if outcome is None or not outcome.has_outputs:
            complete = False
        else:
            expected += length(outcome.outputs[output])
    return expected, complete


def has_every_outcome(outcomes: dict[str, dict[ItemIndexes, Outcome]], node: FlatFunctionNode) -> bool:
    """Whether ``node`` is complete: it has an outcome for every call it will have.

    ``shout`` is complete when ``lines`` has returned and ``shout`` has three outcomes. A ``Waiting`` outcome
    counts: the call was made, though no answer has come.
    """
    if not node.loops:  # A shortcut: calls_expected says one call, so any outcome is every outcome. It saves 2% of to_json on a chain of 500 nodes.
        return bool(outcomes[node.name])
    expected, complete = calls_expected(outcomes, node.loops)
    return complete and len(outcomes[node.name]) == expected


def length(value: Any) -> int:
    """How many items the value of a ``Series`` output has: 0 when the value is ``None`` or ``SKIPPED``."""
    return 0 if value is SKIPPED or value is None else len(value)
