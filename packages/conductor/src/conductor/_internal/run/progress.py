"""How far a run is: for each node, how many calls are finished and how many there will be.

Take this graph::

    lines(text: str) -> Series[str]          # called once; returns ["a", "b", "c"]
    shout(line: str, mark: str = "!") -> str # wired to lines.result: called once per line
    join(texts: Series[str]) -> str          # wired to shout.result: called once, with ["A!", "B!", "C!"]

Before any call, ``graph.with_inputs({"text": "a\\nb\\nc"}).progress`` is::

    RunProgress(finished=0, waiting=0, skipped=0, none_input=0, known=2, total=None, nodes={
        "lines": CallCount(finished=0, waiting=0, skipped=0, none_input=0, known=1, total=1),
        "shout": CallCount(finished=0, waiting=0, skipped=0, none_input=0, known=0, total=None),
        "join":  CallCount(finished=0, waiting=0, skipped=0, none_input=0, known=1, total=1)})

How many calls ``shout`` has depends on how many lines ``lines`` returns, so its ``total`` is ``None`` for now.
Once ``lines`` has returned three lines, ``shout`` has ``known=3, total=3``, and the run has
``known=5, total=5``. If ``lines`` costs nothing, ``run.resume(only_costs_nothing=True)`` calls it first, so a
host learns the totals before any call that costs something.

Nothing here is saved or counted as the run goes. Every number is worked out from the run's outcomes when
``run.progress`` is read, so a run read back with ``Run.from_json`` has the same progress.

``call_count`` is the one count of a node. ``run.progress``, the ``NodeProgress`` and ``NodeComplete`` events, and
a run's status all read it, so they never disagree. ``is_complete`` is the one rule for whether a node is complete,
read from its count.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, assert_never

from conductor._internal.base import SKIPPED, ConductorModel, place_text

from .blueprint import BlueprintFunctionNode, BlueprintNode, BlueprintSubgraphNode
from .outcome import Done, NoneInput, Skipped, Unanswered
from .outcomes import RunOutcomes

if TYPE_CHECKING:
    from .type import Run


class CallCount(ConductorModel):
    """How many calls of one node have an outcome, and how many it will have.

    ``CallCount(finished=1, waiting=0, skipped=0, none_input=0, known=3, total=3)`` is ``shout`` after its first
    line. A subgraph node is never called itself, so its count is the sum of the counts of the nodes inside it.

    Attributes:
        finished: How many calls have an outcome other than waiting: done, answered, skipped or a ``None`` input.
        waiting: How many calls wait for a person.
        skipped: How many of the finished calls were skipped: the node is downstream of a branch that wasn't
            chosen, or its function returned ``SKIPPED`` for every output. A node was skipped when ``finished`` is
            more than 0 and ``skipped`` equals it.
        none_input: How many of the finished calls weren't made because an argument was ``None`` and its input
            doesn't take ``None``.
        known: How many calls the node has so far: one per item of every series it loops over that exists. It only
            grows as the run goes.
        total: How many calls the node will have. ``None`` while that depends on a series that doesn't exist yet.
            Until then, ``known`` is the fewest it can be.
    """

    finished: int
    waiting: int
    skipped: int
    none_input: int
    known: int
    total: int | None


class RunProgress(CallCount):
    """How far a run is, as ``run.progress`` gives it.

    Its own numbers add up the counts of every function node, in the graph and in its subgraphs. ``nodes`` has the
    count of each node.

    ``total`` is ``None`` while any node's total is. It counts calls, so it is the most calls the run can make. A
    call that is skipped, not made for a ``None`` input, or reused from a previous run is counted too, though it costs nothing.

    Attributes:
        nodes: The count of each node, by its path as text. ``"shout"`` is a node of the graph, and
            ``"loud/shout"`` is ``shout`` inside the subgraph node ``loud``. ``"loud"`` itself has the sum of the
            nodes inside it.
    """

    nodes: dict[str, CallCount]


def call_count(outcomes: RunOutcomes, node: BlueprintNode) -> CallCount:
    """Returns the count of one node, from a run's outcomes.

    Before ``lines`` has returned, ``shout``'s count is
    ``CallCount(finished=0, waiting=0, skipped=0, none_input=0, known=0, total=None)``. A subgraph node's count is
    the sum of the counts of the nodes inside it.

    Args:
        outcomes: The run's outcomes.
        node: The node.
    """
    match node:
        case BlueprintSubgraphNode():
            return _sum(call_count(outcomes, inner) for inner in node.inner_nodes.values())
        case BlueprintFunctionNode():
            pass
        case _:
            assert_never(node)
    of_node = outcomes.of(node.node_path)
    waiting = skipped = none_input = 0
    for outcome in of_node.values():
        match outcome:
            case Unanswered():
                waiting += 1
            case NoneInput():
                none_input += 1
            case Skipped():
                skipped += 1
            case Done():
                skipped += bool(outcome.outputs) and all(
                    value is SKIPPED for value in outcome.outputs.values()
                )
            case _:
                assert_never(outcome)
    calls, final = outcomes.item_indexes(node.loops)
    known = len(calls)
    return CallCount.model_construct(
        finished=len(of_node) - waiting,
        waiting=waiting,
        skipped=skipped,
        none_input=none_input,
        known=known,
        total=known if final else None,
    )


def is_complete(count: CallCount) -> bool:
    """Returns whether a node with this count is complete: every call it will have is finished.

    ``shout`` is complete once ``lines`` has returned and ``shout`` has three finished calls. A node with a call
    waiting for a person isn't complete until the person answers.
    """
    return count.total is not None and count.finished == count.total


def progress_of(run: Run[Any]) -> RunProgress:
    """Returns the progress of ``run``, which is what ``run.progress`` gives.

    The nodes are in blueprint order, and each subgraph node comes right after the nodes inside it.
    """
    blueprint, outcomes = run.graph._blueprint, run._outcomes
    nodes = {place_text(path): call_count(outcomes, node) for path, node in blueprint.nodes.items()}
    whole = _sum(nodes[place_text(path)] for path in blueprint.function_nodes)
    return RunProgress.model_construct(**dict(whole), nodes=nodes)


def _sum(counts: Iterable[CallCount]) -> CallCount:
    """Returns the sum of several counts. The ``total`` is ``None`` if any of theirs is."""
    finished = waiting = skipped = none_input = known = 0
    total: int | None = 0
    for count in counts:
        finished += count.finished
        waiting += count.waiting
        skipped += count.skipped
        none_input += count.none_input
        known += count.known
        total = None if total is None or count.total is None else total + count.total
    return CallCount.model_construct(
        finished=finished,
        waiting=waiting,
        skipped=skipped,
        none_input=none_input,
        known=known,
        total=total,
    )
