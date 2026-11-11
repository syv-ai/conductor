"""What happened to one node for one item: the five outcomes, and the two reasons a node cannot be called yet.

The example graph used in this file::

    load() -> Series[str]                 # returns the documents ["a b", "c"]
    clean(document: str) -> str           # wired to load: called once per document
    report(cleaned: Series[str]) -> str   # wired to clean: called once, with ["A B", "C"]

``clean`` is called once per document, so it has one outcome per document. Which document a call is for is
its item path: ``(0,)`` for "a b" and ``(1,)`` for "c". ``load`` and ``report`` are inside no loop, so their
item path is ``()``. When the run has finished, its outcomes are this dict, by node name and then by item path::

    {"load":   {(): Done({"result": ["a b", "c"]})},
     "clean":  {(0,): Done({"result": "A B"}), (1,): Done({"result": "C"})},
     "report": {(): Done({"result": "A B, C"})}}

An outcome is one of five classes:

- ``Done``: the function was called and returned.
- ``Answered``: the function asked a person a question (it raised ``Ask``), and the person answered.
- ``Skipped``: the function was not called, because the node is on a branch that was not chosen.
- ``NotCalled``: the function was not called, because an argument was ``None`` and the parameter does not
  accept ``None``.
- ``Asking``: the function asked a person a question, and the answer has not come yet.

The first four have ``out``, the node's outputs by name. Code that needs an output reads ``outcome.out[name]``
and does not look at which of the four classes it has. ``Asking`` has no outputs yet, so a node that needs
one of them has to wait.
"""

from __future__ import annotations

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, final

from conductor._internal.address import ItemPath
from conductor._internal.ask import Question
from conductor._internal.field import Field
from conductor._internal.sentinels import SKIPPED

if TYPE_CHECKING:
    from conductor._internal.checked_graph.flat_nodes import FlatFunctionNode



class NotReady(enum.Enum):
    """Why a node cannot be called yet for one item path.

    Nothing in this file returns these. The code that works out the arguments of a call returns one of them
    in place of the arguments, and the caller then tries the node again later.

    Attributes:
        NOT_READY: A value the call needs does not exist yet: the node that makes it has not run, or is waiting
            for a person. In the example, ``clean`` is not ready before ``load`` has returned.
        NOT_ITS_TURN: An answer a person gave in an earlier run is kept for this node, and another call of the
            node has to be tried first. Example: in an earlier run, ``clean`` asked a person about two
            documents with the same text, and the person gave two different answers. This run uses the
            answers again, in the same order: the first for ``(0,)``, the second for ``(1,)``. If ``(1,)``
            has its arguments first, it is ``NOT_ITS_TURN`` until ``(0,)`` has been tried.
    """

    NOT_READY = "not_ready"
    NOT_ITS_TURN = "not_its_turn"



@final
@dataclass(frozen=True, slots=True)
class Done:
    """The function was called and returned. In the example: ``Done({"result": "C"})`` for ``clean`` at ``(1,)``.

    Attributes:
        out: The node's outputs by name. An output is usually the value the function returned for it, which
            may be ``None``. It is the marker ``SKIPPED`` when the function returned ``SKIPPED`` for that
            output, to say that the branch the output leads to is not chosen. For an output declared as a
            ``Series`` it is a ``StoredSeries``: an object that wraps the returned list and has ``len()``.
    """

    out: dict[str, Any]


@final
@dataclass(frozen=True, slots=True)
class Answered:
    """The function asked a person a question, and the person answered. The answer is the node's outputs.

    Attributes:
        out: The node's outputs by name, read from the answer.
        question: What the person was asked.
    """

    out: dict[str, Any]
    question: Question


@final
@dataclass(frozen=True, slots=True)
class Skipped:
    """The function was not called, because the node is on a branch that was not chosen.

    Example: a node ``decide`` has the outputs ``yes`` and ``no`` and returns a value for ``yes`` only. A node
    wired to ``decide.no`` is skipped, and so is every node that needs that node's output.

    ``Skipped`` and ``NotCalled`` are read the same way, through ``out``. They are two classes because the
    caller of the graph is told about them differently: a run lists its ``NotCalled`` outcomes for the
    caller and not its ``Skipped`` ones, and a saved run records which of the two an outcome was.

    Attributes:
        out: The node's outputs by name. Each is ``SKIPPED``.
    """

    out: dict[str, Any]


@final
@dataclass(frozen=True, slots=True)
class NotCalled:
    """The function was not called, because an argument was ``None`` and the parameter does not accept ``None``.

    In the example: if ``load`` returns ``["a b", None]``, then ``clean`` at ``(1,)`` is
    ``NotCalled({"result": None}, input="document")``, as ``document: str`` does not accept ``None``.

    Attributes:
        out: The node's outputs by name. Each is ``None``. The exception is an output in an exclusive group
            (outputs of which the function picks one, like ``yes`` and ``no``): it is ``SKIPPED``, since the
            function never picked.
        input: The name of the input that was ``None``.
    """

    out: dict[str, Any]
    input: str


@final
@dataclass(frozen=True, slots=True)
class Asking:
    """The function asked a person a question (it raised ``Ask``), and the answer has not come yet.

    There are no outputs yet, so a node that needs one of them is not ready.

    Attributes:
        question: What the person is asked.
    """

    question: Question


#: What happened to one node for one item path. An outcome is never changed after it is made.
Outcome = Done | Answered | Skipped | NotCalled | Asking

def outputs_when_skipped(outputs: Mapping[str, Field]) -> dict[str, Any]:
    """The ``out`` of a ``Skipped`` outcome for a node with these outputs: ``SKIPPED`` for each.

    For ``clean``: ``{"result": SKIPPED}``.
    """
    return dict.fromkeys(outputs, SKIPPED)


def outputs_when_not_called(outputs: Mapping[str, Field]) -> dict[str, Any]:
    """The ``out`` of a ``NotCalled`` outcome for a node with these outputs.

    For ``clean``: ``{"result": None}``. For a node with the outputs ``yes`` and ``no`` in one exclusive group
    and an output ``note`` in none: ``{"yes": SKIPPED, "no": SKIPPED, "note": None}``.
    """
    return {name: SKIPPED if output.param.exclusive_group is not None else None for name, output in outputs.items()}


def item_paths(outcomes: dict[str, dict[ItemPath, Outcome]], loops: tuple[tuple[FlatFunctionNode, str], ...], under: ItemPath, *, counted: bool = False) -> tuple[list[ItemPath], bool]:
    """The item paths a node is called for, and whether that list is complete yet.

    In the example, ``clean`` is inside one loop, over ``load.result``. After ``load`` has returned two
    documents, ``item_paths(outcomes, clean.maps, ())`` returns ``([(0,), (1,)], True)``. Before ``load`` has
    returned, or while it is waiting for a person, it returns ``([], False)``: there are no item paths so far,
    and more may come.

    Args:
        outcomes: The run's outcomes, by node name and then by item path.
        loops: The loops the node is inside, outermost first. Each is the node that returns the list and the
            name of the output that is the list.
        under: Return only the item paths that start with this one. ``()`` returns all of them. With two
            nested loops, ``(1,)`` returns the paths inside item 1 of the outer list: ``(1, 0)``, ``(1, 1)``, ...
        counted: Count the items of the innermost loop without making a path for each. The list returned
            then has at most one entry, and it is a count, not a path: ``[(2,)]`` means two item paths in all.

    Returns:
        The item paths, and ``True`` when every list the node loops over exists.
    """
    start, deepest = len(under), len(loops)
    if start >= deepest:
        return [under[:deepest]], True
    paths, complete = [under], True
    for depth in range(start, deepest):
        series_node, output = loops[depth]
        of_series_node = outcomes[series_node.name]
        below: list[ItemPath] = []
        for parent in paths:
            outcome = of_series_node.get(parent)
            if outcome is None or type(outcome) is Asking:
                complete = False
                continue
            count = length(outcome.out[output])
            if counted and depth == deepest - 1:
                below = [(below[0][0] + count,)] if below else [(count,)]
            else:
                below.extend((*parent, i) for i in range(count))
        paths = below
    return paths, complete


def has_every_outcome(outcomes: dict[str, dict[ItemPath, Outcome]], node: FlatFunctionNode) -> bool:
    """Whether ``node`` has an outcome for every item path it will ever have.

    In the example, ``clean`` has every outcome when ``load`` has returned and ``clean`` has one outcome per
    document. An ``Asking`` outcome counts: the call was made, though its answer has not come.
    """
    of_node = outcomes[node.name]
    if not node.loops:
        return bool(of_node)
    counted, complete = item_paths(outcomes, node.loops, (), counted=True)
    return complete and len(of_node) == (counted[0][0] if counted else 0)


def length(value: Any) -> int:
    """How many items the value of a ``Series`` output has: 0 when the value is ``None`` or ``SKIPPED``."""
    return 0 if value is SKIPPED or value is None else len(value)
