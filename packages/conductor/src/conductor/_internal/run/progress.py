"""How far a run is: how many calls each node has an outcome for, and how many it will have; whether a node is
complete (``is_complete``), the one rule for it; and the item indexes of every call a node will have
(``all_item_indexes``), which the scheduler and a saved run read too.

The example graph::

    lines(text: str) -> Series[str]          # called once; returns ["a", "b", "c"]
    shout(line: str, mark: str = "!") -> str # wired to lines.result: called once per line
    join(texts: Series[str]) -> str          # wired to shout.result: called once, with ["A!", "B!", "C!"]

Before any call, ``graph.with_inputs(text="a\\nb\\nc").progress`` is::

    Progress(finished=0, waiting=0, known=2, total=None, nodes={
        "lines": CallCount(finished=0, waiting=0, known=1, total=1),
        "shout": CallCount(finished=0, waiting=0, known=0, total=None),
        "join":  CallCount(finished=0, waiting=0, known=1, total=1)})

How many calls ``shout`` has depends on how many lines ``lines`` returns, so its ``total`` is ``None``. Once
``lines`` has returned three lines, ``shout`` has ``known=3, total=3``, and the run ``known=5, total=5``. If
``lines`` costs nothing to call, ``run.estimate()`` calls it first, so a host has the totals before any call
that costs something.

Nothing here is saved or counted as the run goes: every count is read from the run's outcomes when
``run.progress`` is read. So a run read back with ``Run.from_json`` in another process has the same progress.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from conductor._internal.base import SKIPPED, ConductorModel, ItemIndexes, NodePath, place_text

from .blueprint import BlueprintFunctionNode, BlueprintSubgraphNode, RunBlueprint
from .outcome import Outcome, Unanswered


class CallCount(ConductorModel):
    """How many calls of one node have an outcome, and how many it will have: ``CallCount(finished=1, waiting=0,
    known=3, total=3)`` for ``shout`` after its first line.

    A subgraph node is never called itself: its count is the sum of the counts of the nodes inside it.

    Attributes:
        finished: How many calls are finished: they have an outcome that is not waiting (done, answered, skipped
            or not called).
        waiting: How many calls wait for a person.
        known: How many calls the node has so far: one per item of every series it is looped over that exists.
            It only grows as the run goes.
        total: How many calls the node will have. ``None`` while that depends on a series that does not exist
            yet; then ``known`` is the least it can be.
    """

    finished: int
    waiting: int
    known: int
    total: int | None


class Progress(CallCount):
    """How far a run is: ``run.progress``. Its own four numbers are the sums over every function node of the
    graph and of its subgraphs, and ``nodes`` has the count of each node.

    ``total`` is ``None`` while the total of any node is. ``total`` counts calls, so it is the most calls the run
    can make: a call that is skipped, not called or reused from the previous run is counted, and costs nothing.

    Attributes:
        nodes: The count of each node, by node path as text: ``"shout"``, and ``"loud/shout"`` for ``shout``
            inside the subgraph node ``loud``, with ``"loud"`` itself as the sum of the nodes inside it.
    """

    nodes: dict[str, CallCount]


def call_count(outcomes: dict[NodePath, dict[ItemIndexes, Outcome]], node: BlueprintFunctionNode) -> CallCount:
    """The count of one function node, read from a run's outcomes.

    Args:
        outcomes: The run's outcomes, by node path and then by item indexes.
        node: The node.

    Returns:
        Its count: ``CallCount(finished=0, waiting=0, known=0, total=None)`` for ``shout`` before ``lines`` has
        returned.
    """
    of_node = outcomes[node.node_path]
    waiting = sum(1 for outcome in of_node.values() if type(outcome) is Unanswered)
    known, final = calls_expected(outcomes, node.loops)
    return CallCount.model_construct(finished=len(of_node) - waiting, waiting=waiting, known=known, total=known if final else None)


def progress_of(blueprint: RunBlueprint, outcomes: dict[NodePath, dict[ItemIndexes, Outcome]]) -> Progress:
    """The progress of a run with these outcomes: what ``run.progress`` returns.

    Args:
        blueprint: The blueprint of the run's graph.
        outcomes: The run's outcomes, by node path and then by item indexes.

    Returns:
        The count of each node, in the order of the blueprint with a subgraph node after the nodes inside it,
        and their sums.
    """
    nodes: dict[str, CallCount] = {}
    for path, node in blueprint.nodes.items():
        if isinstance(node, BlueprintSubgraphNode):  # the nodes inside it come before it, so their counts are made
            nodes[place_text(path)] = _sum(nodes[place_text(inner.node_path)] for inner in node.inner_nodes.values())
        else:
            nodes[place_text(path)] = call_count(outcomes, node)
    run = _sum(nodes[place_text(path)] for path in blueprint.function_nodes)
    return Progress.model_construct(finished=run.finished, waiting=run.waiting, known=run.known, total=run.total, nodes=nodes)


def _sum(counts: Iterable[CallCount]) -> CallCount:
    """The sum of several counts: ``total`` is ``None`` when any of theirs is."""
    finished = waiting = known = 0
    total: int | None = 0
    for count in counts:
        finished, waiting, known = finished + count.finished, waiting + count.waiting, known + count.known
        total = None if total is None or count.total is None else total + count.total
    return CallCount.model_construct(finished=finished, waiting=waiting, known=known, total=total)


def all_item_indexes(
    outcomes: dict[NodePath, dict[ItemIndexes, Outcome]], loops: tuple[tuple[NodePath, str], ...], prefix: ItemIndexes,
) -> tuple[list[ItemIndexes], bool]:
    """The item indexes of every call a node will have, and whether that list is complete yet.

    ``shout`` is inside one loop, over ``lines.result``. Once ``lines`` has returned three lines,
    ``all_item_indexes(outcomes, shout.loops, ())`` returns ``([(0,), (1,), (2,)], True)``. Before ``lines`` has
    returned it returns ``([], False)``: no call is known yet, and more will come. ``lines`` is inside no loop,
    so for it the result is always ``([()], True)``: one call.

    Args:
        outcomes: The run's outcomes, by node path and then by item indexes. Every node has an entry, empty
            until its first outcome.
        loops: The loops the node is inside, outermost first. Each is the path of the node that returns the series and
            the name of that output.
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
        of_series_node = outcomes[series_node]
        one_loop_deeper: list[ItemIndexes] = []
        for item_indexes in every_item_indexes:
            outcome = of_series_node.get(item_indexes)
            if outcome is None or not outcome.has_outputs:
                complete = False
                continue
            one_loop_deeper.extend((*item_indexes, index) for index in range(length(outcome.outputs[output])))
        every_item_indexes = one_loop_deeper
    return every_item_indexes, complete


def calls_expected(outcomes: dict[NodePath, dict[ItemIndexes, Outcome]], loops: tuple[tuple[NodePath, str], ...]) -> tuple[int, bool]:
    """How many calls a node inside ``loops`` will have in this run, and whether that number is final yet.

    A node inside no loop has one call: for ``lines`` and ``join`` the result is ``(1, True)``. ``shout`` is
    inside the loop over ``lines.result``: ``(3, True)`` once ``lines`` has returned three lines, ``(0, False)``
    before. With two nested loops the inner series may differ in length; the number is the sum of their lengths.

    Args:
        outcomes: The run's outcomes, by node path and then by item indexes.
        loops: The loops the node is inside, outermost first, as ``all_item_indexes`` takes them.
    """
    if not loops:
        return 1, True
    calls_of_series_node, complete = all_item_indexes(outcomes, loops[:-1], ())
    series_node, output = loops[-1]
    of_series_node = outcomes[series_node]
    expected = 0
    for item_indexes in calls_of_series_node:
        outcome = of_series_node.get(item_indexes)
        if outcome is None or not outcome.has_outputs:
            complete = False
        else:
            expected += length(outcome.outputs[output])
    return expected, complete


def is_complete(outcomes: dict[NodePath, dict[ItemIndexes, Outcome]], node: BlueprintFunctionNode) -> bool:
    """Whether ``node`` is complete: it has an outcome for every call it will have, and none of them waits for a person.

    ``shout`` is complete when ``lines`` has returned and ``shout`` has three outcomes. ``approve`` with an
    ``Unanswered`` outcome is not, until the person answers.
    """
    of_node = outcomes[node.node_path]
    if not node.loops:  # A shortcut: one call, so its one outcome decides. It saves 2% of a run of a chain of 500 nodes.
        return bool(of_node) and type(of_node[()]) is not Unanswered
    return has_every_outcome(outcomes, node) and all(type(outcome) is not Unanswered for outcome in of_node.values())


def has_every_outcome(outcomes: dict[NodePath, dict[ItemIndexes, Outcome]], node: BlueprintFunctionNode) -> bool:
    """Whether ``node`` has an outcome for every call it will have.

    ``shout`` has when ``lines`` has returned and ``shout`` has three outcomes. An ``Unanswered`` outcome counts:
    the call was made, though no answer has come.
    """
    if not node.loops:  # A shortcut: calls_expected says one call, so any outcome is every outcome. It saves 2% of to_json on a chain of 500 nodes.
        return bool(outcomes[node.node_path])
    expected, complete = calls_expected(outcomes, node.loops)
    return complete and len(outcomes[node.node_path]) == expected


def length(value: Any) -> int:
    """How many items the value of a ``Series`` output has: 0 when the value is ``None`` or ``SKIPPED``."""
    return 0 if value is SKIPPED or value is None else len(value)
