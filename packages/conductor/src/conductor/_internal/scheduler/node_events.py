"""Works out the events about each node from what the scheduler tells it, and hands them to the listener.

The example graph::

    lines(text: str) -> Series[str]          # called once; returns ["a", "b", "c"]
    shout(line: str, mark: str = "!") -> str # wired to lines.result: called once per line
    join(texts: Series[str]) -> str          # wired to shout.result: called once, with ["A!", "B!", "C!"]

``run.stream()`` on it yields these events about its nodes, in this order::

    NodeStart(lines)
    NodeStart(shout)
    NodeProgress(shout, done=1, total=3)
    NodeStart(join)
    NodeComplete(lines, outcomes=1)
    NodeProgress(shout, done=3, total=3)
    NodeComplete(shout, outcomes=3)
    NodeComplete(join, outcomes=1)

The scheduler tells a ``NodeEvents`` what happens, one method for each thing: a call is about to be made, the next
outcome is reused from the previous run, an outcome was recorded, a call will be retried, a call failed, no call is
ready, and the scheduler has ended. ``NodeEvents`` decides which events follow:

- ``NodeStart``, once per node: before its first call, or with its first outcome that is not ``Skipped``. A node
  inside a subgraph starts the subgraph node first.
- ``NodeWaiting``, with every ``Waiting`` outcome.
- ``NodeProgress``, for a node inside a loop: after an outcome, at most four times a second, and a last one with
  the final count.
- ``NodeComplete`` or ``NodeSkipped``, once per node, when the scheduler says that no call is ready or that it
  has ended: every call of the node has an outcome, none is waiting for a person, and no call of the node or of
  a node upstream of it is running. Above, no node is called on a thread, so the scheduler first says "no call
  is ready" after ``join``: that is why ``lines`` is complete only after ``join`` has started.
- ``NodeRetry`` and ``NodeFailed``, as the scheduler says them.

Nothing here starts a thread, and the only clock read is the one that spaces ``NodeProgress``. A test makes a
``NodeEvents`` with a list's ``append`` as the listener and calls the methods by hand
(``tests/test_core/test_node_events.py``).
"""

from __future__ import annotations

import operator
import time
from collections import defaultdict
from collections.abc import Callable, Iterable

from conductor._internal.address import ItemIndexes, NodePath
from conductor._internal.ask import waiting_record
from conductor._internal.compiled import FlatFunctionNode, FlatNode, all_upstream, outcome_loc
from conductor._internal.markers import SKIPPED
from conductor._internal.outcomes import (
    Answered,
    Done,
    NotCalled,
    Outcome,
    Skipped,
    Waiting,
    all_item_indexes,
    calls_expected,
    has_every_outcome,
)
from conductor._internal.problem import Problem
from conductor._internal.scheduler.events import (
    EventBase,
    NodeComplete,
    NodeFailed,
    NodeProgress,
    NodeRetry,
    NodeSkipped,
    NodeStart,
    NodeWaiting,
)

#: The least time between two ``NodeProgress`` of one node.
_SECONDS_BETWEEN_PROGRESS = 0.25

#: A node's position in topological order.
_POSITION = operator.attrgetter("position")


class NodeEvents:
    """The events about the nodes of one call of ``run.resume()`` or ``run.stream()``.

    Attributes:
        emit: The listener: it is called with each event.
        nodes: Every function node of the graph and of its subgraphs, in topological order.
        outcomes: The run's outcomes, by node name and item indexes. The scheduler writes them; they are only read here.
        started: The nodes that have had ``NodeStart``, in this call or an earlier one.
        complete: The nodes that have had ``NodeComplete`` or ``NodeSkipped``, in this call or an earlier one.
        skipped: The nodes that had ``NodeSkipped`` in this call.
        changed: The nodes with an outcome recorded since they were last looked at for ``NodeComplete``.
        reused: How many outcomes each node took from the previous run, for ``NodeComplete.reused``.
        progress_at: When each node last had ``NodeProgress``, as ``time.monotonic()``.
        progress_done: The ``done`` of each node's last ``NodeProgress``.
    """

    def __init__(
        self,
        emit: Callable[[EventBase], None],
        nodes: list[FlatFunctionNode],
        outcomes: dict[str, dict[ItemIndexes, Outcome]],
        finished: frozenset[NodePath],
    ) -> None:
        """
        Args:
            emit: The listener.
            nodes: Every function node, in topological order.
            outcomes: The run's outcomes, by node name and item indexes.
            finished: The node paths of the nodes that were complete when the run was last resumed.
        """
        self.emit, self.nodes, self.outcomes = emit, nodes, outcomes
        with_outcomes: set[FlatFunctionNode] = {node for node in nodes if outcomes[node.name]}
        self.started: set[FlatNode] = set(with_outcomes)
        for node in with_outcomes:
            subgraph = node.parent
            while subgraph is not None:
                self.started.add(subgraph)
                subgraph = subgraph.parent
        complete_before: set[FlatFunctionNode] = {node for node in nodes if node.node_path in finished}
        self.complete: set[FlatNode] = set(complete_before)
        for node in complete_before:
            self._complete_subgraphs_around(node, announce=False)
        self.skipped: set[FlatNode] = set()
        self.changed = with_outcomes - complete_before
        self.reused: defaultdict[FlatFunctionNode, int] = defaultdict(int)
        self.progress_at: dict[FlatFunctionNode, float] = {}
        self.progress_done: dict[FlatFunctionNode, int] = {}

    def call_starts(self, node: FlatNode) -> None:
        """A call of ``node`` is about to be made: ``NodeStart``, unless the node has started."""
        if node not in self.started:
            if node.parent is not None:
                self.call_starts(node.parent)
            self.started.add(node)
            self.emit(NodeStart(loc=node.node_path))

    def outcome_reused(self, node: FlatFunctionNode) -> None:
        """The outcome recorded next for ``node`` is taken from the previous run."""
        self.reused[node] += 1

    def outcome_recorded(self, node: FlatFunctionNode, item_indexes: ItemIndexes, outcome: Outcome) -> None:
        """``outcome`` was written for the call ``item_indexes`` of ``node``.

        An outcome that was not a call (reused, or ``NotCalled``) starts its node as a call does. ``Skipped`` does not.
        """
        self.changed.add(node)
        if type(outcome) is not Skipped:
            self.call_starts(node)
        if type(outcome) is Waiting:
            self.emit(NodeWaiting(waiting=waiting_record(node.compiled_node, outcome_loc(node, item_indexes), outcome)))
        if node.loops:
            self._progress(node)

    def call_retries(self, node: FlatFunctionNode, item_indexes: ItemIndexes, attempt: int, delay: float, cause: Problem) -> None:
        """A call failed and is made again after ``delay`` seconds, as try number ``attempt`` after the first."""
        self.emit(NodeRetry(loc=outcome_loc(node, item_indexes), attempt=attempt, retries=node.policy.retries, delay=delay, cause=cause))

    def call_failed(self, node: FlatFunctionNode, cause: Problem) -> None:
        """A call of ``node`` failed and is not retried: the run fails with ``cause``."""
        self.emit(NodeFailed(loc=node.node_path, cause=cause))

    def nothing_is_ready(self, running: Iterable[tuple[FlatFunctionNode, ItemIndexes]]) -> None:
        """No call is ready to be made: say ``NodeComplete`` or ``NodeSkipped`` for every node that is now complete.

        Args:
            running: The calls that are on a thread, waiting for one, or waiting to be retried.
        """
        if not self.changed:
            return
        nodes_running = {node for node, _ in running}
        for node in sorted(self.changed, key=_POSITION):
            if node in nodes_running or nodes_running & all_upstream(node):
                continue
            self.changed.discard(node)
            outcomes = self.outcomes[node.name]
            if node in self.complete or Waiting in {type(outcome) for outcome in outcomes.values()} or not has_every_outcome(self.outcomes, node):
                continue
            self.complete.add(node)
            if node.loops and node in self.progress_done:
                self._progress(node, last=True)
            if outcomes and all(_is_skipped(outcome) for outcome in outcomes.values()):
                self.skipped.add(node)
                self.emit(NodeSkipped(loc=node.node_path))
            else:
                not_called = sum(type(outcome) is NotCalled for outcome in outcomes.values())
                self.emit(NodeComplete(loc=node.node_path, outcomes=len(outcomes), not_called=not_called, reused=self.reused[node]))
            self._complete_subgraphs_around(node, announce=True)

    def scheduler_ended(self) -> None:
        """The scheduler makes no more calls and none is running: say which nodes are complete, and the last
        ``NodeProgress`` of each node that had one and is not complete."""
        self.changed.update(self.nodes)
        self.nothing_is_ready(())
        for node in self.nodes:
            if node.loops and node in self.progress_done:
                self._progress(node, last=True)

    def _progress(self, node: FlatFunctionNode, last: bool = False) -> None:
        """Say ``NodeProgress`` for a node inside a loop, unless the last one is under a quarter of a second old or
        had the same count. ``last`` leaves out the clock: it is the node's final count."""
        now = time.monotonic()
        if not (last or now - self.progress_at.get(node, -1.0) >= _SECONDS_BETWEEN_PROGRESS):
            return
        done = sum(1 for outcome in self.outcomes[node.name].values() if outcome.has_outputs)
        if self.progress_done.get(node) != done:
            self.progress_at[node], self.progress_done[node] = now, done
            every_item_indexes, known = all_item_indexes(self.outcomes, node.loops, ())
            self.emit(NodeProgress(loc=node.node_path, done=done, total=len(every_item_indexes) if known else None))

    def _complete_subgraphs_around(self, node: FlatNode, announce: bool) -> None:
        """Mark the subgraph node ``node`` is inside as complete when every node inside it is, then the subgraph
        node that one is inside, and so on.

        With ``announce``, each gets ``NodeSkipped`` when every node inside it was skipped, else ``NodeComplete``
        with one outcome per call of the subgraph.
        """
        subgraph = node.parent
        while subgraph is not None and subgraph not in self.complete and all(inner in self.complete for inner in subgraph.inner_nodes.values()):
            self.complete.add(subgraph)
            if announce:
                if all(inner in self.skipped for inner in subgraph.inner_nodes.values()):
                    self.skipped.add(subgraph)
                    self.emit(NodeSkipped(loc=subgraph.node_path))
                else:
                    self.emit(NodeComplete(loc=subgraph.node_path, outcomes=calls_expected(self.outcomes, subgraph.loops)[0]))
            subgraph = subgraph.parent


def _is_skipped(outcome: Outcome) -> bool:
    """Whether a call was skipped: the node was not called because a branch upstream was not chosen, or it was
    called and returned ``SKIPPED`` on every output."""
    return type(outcome) is Skipped or (
        (type(outcome) is Done or type(outcome) is Answered) and bool(outcome.outputs) and all(value is SKIPPED for value in outcome.outputs.values())
    )
