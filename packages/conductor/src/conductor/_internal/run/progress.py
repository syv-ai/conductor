"""How far a run is: how many calls each node has an outcome for, and how many it will have.

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
``lines`` costs nothing to call, ``run.resume(only_costs_nothing=True)`` calls it first, so a host has the
totals before any call that costs something.

Nothing here is saved or counted as the run goes: every count is read from the run's outcomes when
``run.progress`` is read. So a run read back with ``Run.from_json`` in another process has the same progress.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

from conductor._internal.compiled import FlatFunctionNode, FlatSubgraph
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.run.outcome import ItemIndexes, Outcome, Unanswered, calls_expected

if TYPE_CHECKING:
    from conductor._internal.graph import Graph


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


def call_count(outcomes: dict[str, dict[ItemIndexes, Outcome]], node: FlatFunctionNode) -> CallCount:
    """The count of one function node, read from a run's outcomes.

    Args:
        outcomes: The run's outcomes, by node name and then by item indexes.
        node: The node.

    Returns:
        Its count: ``CallCount(finished=0, waiting=0, known=0, total=None)`` for ``shout`` before ``lines`` has
        returned.
    """
    of_node = outcomes[node.name]
    waiting = sum(1 for outcome in of_node.values() if type(outcome) is Unanswered)
    known, final = calls_expected(outcomes, node.loops)
    return CallCount.model_construct(finished=len(of_node) - waiting, waiting=waiting, known=known, total=known if final else None)


def progress_of(graph: Graph, outcomes: dict[str, dict[ItemIndexes, Outcome]]) -> Progress:
    """The progress of a run of ``graph`` with these outcomes: what ``run.progress`` returns.

    Args:
        graph: The run's graph.
        outcomes: The run's outcomes, by node name and then by item indexes.

    Returns:
        The count of each node, in the order of the flat graph with a subgraph node after the nodes inside it,
        and their sums.
    """
    nodes: dict[str, CallCount] = {}
    for name, flat in graph._nodes_by_name.items():
        if isinstance(flat, FlatSubgraph):  # the nodes inside it come before it, so their counts are made
            nodes[name] = _sum(nodes[inner.name] for inner in flat.inner_nodes.values())
        else:
            nodes[name] = call_count(outcomes, flat)
    run = _sum(nodes[node.name] for node in graph._flat_function_nodes)
    return Progress.model_construct(finished=run.finished, waiting=run.waiting, known=run.known, total=run.total, nodes=nodes)


def _sum(counts: Iterable[CallCount]) -> CallCount:
    """The sum of several counts: ``total`` is ``None`` when any of theirs is."""
    finished = waiting = known = 0
    total: int | None = 0
    for count in counts:
        finished, waiting, known = finished + count.finished, waiting + count.waiting, known + count.known
        total = None if total is None or count.total is None else total + count.total
    return CallCount.model_construct(finished=finished, waiting=waiting, known=known, total=total)
