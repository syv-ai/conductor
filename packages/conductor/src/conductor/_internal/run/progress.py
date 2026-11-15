"""How far a run is: for each node, how many calls are finished and how many there will be.

Take this graph::

    lines(text: str) -> Series[str]          # called once; returns ["a", "b", "c"]
    shout(line: str, mark: str = "!") -> str # wired to lines.result: called once per line
    join(texts: Series[str]) -> str          # wired to shout.result: called once, with ["A!", "B!", "C!"]

Before any call, ``graph.with_inputs(text="a\\nb\\nc").progress`` is::

    RunProgress(finished=0, waiting=0, known=2, total=None, nodes={
        "lines": CallCount(finished=0, waiting=0, known=1, total=1),
        "shout": CallCount(finished=0, waiting=0, known=0, total=None),
        "join":  CallCount(finished=0, waiting=0, known=1, total=1)})

How many calls ``shout`` has depends on how many lines ``lines`` returns, so its ``total`` is ``None`` for now.
Once ``lines`` has returned three lines, ``shout`` has ``known=3, total=3``, and the run has
``known=5, total=5``. If ``lines`` costs nothing, ``run.resume(only_costs_nothing=True)`` calls it first, so a
host learns the totals before any call that costs something.

Nothing here is saved or counted as the run goes. Every number is worked out from the run's outcomes when
``run.progress`` is read, so a run read back with ``Run.from_json`` has the same progress.

The module also has two helpers the rest of the run uses. ``is_complete`` is the one rule for whether a node is
complete. ``all_item_indexes`` lists the item indexes of every call a node will have.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, assert_never

from conductor._internal.base import SKIPPED, ConductorModel, ItemIndexes, NodePath, place_text

from .blueprint import BlueprintFunctionNode, BlueprintSubgraphNode
from .outcome import Done, NoneInput, Outcome, Skipped, Unanswered

if TYPE_CHECKING:
    from .type import Run


class CallCount(ConductorModel):
    """How many calls of one node have an outcome, and how many it will have.

    ``CallCount(finished=1, waiting=0, known=3, total=3)`` is ``shout`` after its first line. A subgraph node is
    never called itself, so its count is the sum of the counts of the nodes inside it.

    Attributes:
        finished: How many calls have an outcome other than waiting: done, answered, skipped or not called.
        waiting: How many calls wait for a person.
        known: How many calls the node has so far: one per item of every series it loops over that exists. It only
            grows as the run goes.
        total: How many calls the node will have. ``None`` while that depends on a series that doesn't exist yet.
            Until then, ``known`` is the fewest it can be.
    """

    finished: int
    waiting: int
    known: int
    total: int | None


class RunProgress(CallCount):
    """How far a run is, as ``run.progress`` gives it.

    Its own four numbers add up the counts of every function node, in the graph and in its subgraphs. ``nodes``
    has the count of each node.

    ``total`` is ``None`` while any node's total is. It counts calls, so it is the most calls the run can make. A
    call that is skipped, not called, or reused from a previous run is counted too, though it costs nothing.

    Attributes:
        nodes: The count of each node, by its path as text. ``"shout"`` is a node of the graph, and
            ``"loud/shout"`` is ``shout`` inside the subgraph node ``loud``. ``"loud"`` itself has the sum of the
            nodes inside it.
    """

    nodes: dict[str, CallCount]


def call_count(
    outcomes: dict[NodePath, dict[ItemIndexes, Outcome]], node: BlueprintFunctionNode
) -> CallCount:
    """Returns the count of one function node, from a run's outcomes.

    Before ``lines`` has returned, ``shout``'s count is ``CallCount(finished=0, waiting=0, known=0, total=None)``.

    Args:
        outcomes: The run's outcomes, by node path and then by item indexes.
        node: The node.
    """
    of_node = outcomes[node.node_path]
    waiting = 0
    for outcome in of_node.values():
        match outcome:
            case Done() | Skipped() | NoneInput():
                pass
            case Unanswered():
                waiting += 1
            case _:
                assert_never(outcome)
    calls, final = all_item_indexes(outcomes, node.loops, ())
    known = len(calls)
    return CallCount.model_construct(
        finished=len(of_node) - waiting,
        waiting=waiting,
        known=known,
        total=known if final else None,
    )


def progress_of(run: Run[Any]) -> RunProgress:
    """Returns the progress of ``run``, which is what ``run.progress`` gives.

    The nodes are in blueprint order, and each subgraph node comes right after the nodes inside it.
    """
    blueprint, outcomes = run._blueprint, run._outcomes
    nodes: dict[str, CallCount] = {}
    for path, node in blueprint.nodes.items():
        if isinstance(
            node, BlueprintSubgraphNode
        ):  # the nodes inside it come before it, so their counts are already there
            nodes[place_text(path)] = _sum(
                nodes[place_text(inner.node_path)] for inner in node.inner_nodes.values()
            )
        else:
            nodes[place_text(path)] = call_count(outcomes, node)
    whole = _sum(nodes[place_text(path)] for path in blueprint.function_nodes)
    return RunProgress.model_construct(
        finished=whole.finished,
        waiting=whole.waiting,
        known=whole.known,
        total=whole.total,
        nodes=nodes,
    )


def _sum(counts: Iterable[CallCount]) -> CallCount:
    """Returns the sum of several counts. The ``total`` is ``None`` if any of theirs is."""
    finished = waiting = known = 0
    total: int | None = 0
    for count in counts:
        finished, waiting, known = (
            finished + count.finished,
            waiting + count.waiting,
            known + count.known,
        )
        total = None if total is None or count.total is None else total + count.total
    return CallCount.model_construct(finished=finished, waiting=waiting, known=known, total=total)


def all_item_indexes(
    outcomes: dict[NodePath, dict[ItemIndexes, Outcome]],
    loops: tuple[tuple[NodePath, str], ...],
    prefix: ItemIndexes,
) -> tuple[list[ItemIndexes], bool]:
    """Returns the item indexes of every call a node will have, and whether that list is final yet.

    ``shout`` is inside one loop, over ``lines.result``. Once ``lines`` has returned three lines,
    ``all_item_indexes(outcomes, shout.loops, ())`` returns ``([(0,), (1,), (2,)], True)``. Before ``lines`` has
    returned, it returns ``([], False)``: no call is known yet, and more will come. ``lines`` is inside no loop, so
    for it the result is always ``([()], True)``, one call.

    Args:
        outcomes: The run's outcomes, by node path and then by item indexes. Every node has an entry, empty until
            its first outcome.
        loops: The loops the node is inside, outermost first. Each is the path of the node that returns the series,
            and that output's name.
        prefix: Only return the item indexes that start with these. ``()`` returns all of them. With two nested
            loops, ``(1,)`` returns the calls for item 1 of the outer loop: ``(1, 0)``, ``(1, 1)``, and so on.

    Returns:
        The item indexes of each call known so far, and ``True`` once every series the node loops over exists.
    """
    start, deepest = len(prefix), len(loops)
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
            one_loop_deeper.extend(
                (*item_indexes, index) for index in range(length(outcome.outputs[output]))
            )
        every_item_indexes = one_loop_deeper
    return every_item_indexes, complete


def is_complete(
    outcomes: dict[NodePath, dict[ItemIndexes, Outcome]], node: BlueprintFunctionNode
) -> bool:
    """Returns whether ``node`` is complete: it has an outcome for every call it will have, and none is waiting.

    ``shout`` is complete once ``lines`` has returned and ``shout`` has three outcomes. A node with an
    ``Unanswered`` outcome isn't complete until the person answers.
    """
    calls, every_call_known = all_item_indexes(outcomes, node.loops, ())
    of_node = outcomes[node.node_path]
    if not every_call_known or len(of_node) != len(calls):
        return False
    for outcome in of_node.values():
        match outcome:
            case Done() | Skipped() | NoneInput():
                pass
            case Unanswered():
                return False
            case _:
                assert_never(outcome)
    return True


def length(value: Any) -> int:
    """Returns how many items the value of a ``Series`` output has. A value of ``None`` or ``SKIPPED`` has 0."""
    return 0 if value is SKIPPED or value is None else len(value)
