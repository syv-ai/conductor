"""Works out the events about each node from what the scheduler reports, and hands them to the listener.

The scheduler only knows about calls. A host listening to ``run.stream()`` wants to hear about nodes: when one
starts, how far it has got, when it is done. ``NodeEvents`` sits between the two. The scheduler tells it what
happens, one method per kind of news, and it decides which node events follow. The run events, ``RunStart``,
``RunEnd`` and ``RunCheckpoint``, are made elsewhere.

Take this example graph::

    lines(text: str) -> Series[str]          # called once; returns ["a", "b", "c"]
    shout(line: str, mark: str = "!") -> str # wired to lines.result: called once per line
    join(texts: Series[str]) -> str          # wired to shout.result: called once, with ["A!", "B!", "C!"]

``run.stream()`` on it gives these events about its nodes, in this order::

    NodeStart(lines)
    NodeStart(shout)
    NodeProgress(shout, count=CallCount(finished=1, waiting=0, skipped=0, none_input=0, known=3, total=3))
    NodeStart(join)
    NodeComplete(lines, count=CallCount(finished=1, waiting=0, skipped=0, none_input=0, known=1, total=1))
    NodeProgress(shout, count=CallCount(finished=3, waiting=0, skipped=0, none_input=0, known=3, total=3))
    NodeComplete(shout, count=CallCount(finished=3, waiting=0, skipped=0, none_input=0, known=3, total=3))
    NodeComplete(join, count=CallCount(finished=1, waiting=0, skipped=0, none_input=0, known=1, total=1))

The scheduler reports seven kinds of news: a call is about to be made, the next outcome is reused from the previous
run, an outcome was recorded, a call will be retried, a call failed, no call is ready, and the scheduler has ended.
From those, ``NodeEvents`` makes:

- ``NodeStart``, once per node. It comes before the node's first call, or with its first outcome that isn't
  ``Skipped``. A node inside a subgraph starts the subgraph node first.
- ``NodeWaiting``, with every ``Unanswered`` outcome.
- ``NodeProgress``, for a node inside a loop. It comes after an outcome, at most four times a second, and once
  more with the final count.
- ``NodeComplete``, once per node, when the scheduler says no call is ready, or that it has ended. By then every
  call of the node has an outcome, none is waiting for a person, and no call of the node or of a node upstream of
  it is still running. In the example, no node runs on a thread. So the scheduler first finds that no call is
  ready only after ``join``. That's why ``lines`` only completes after ``join`` has started.
- A ``NodeProgress`` and a ``NodeComplete`` carry the node's ``CallCount``, the same count ``run.progress`` has
  for it. A subgraph node completes once every node inside it has, and its count is theirs added up.
- ``NodeRetry`` and ``NodeFailed``, passed on as the scheduler reports them.

Nothing here starts a thread. The only clock it reads is the one that spaces out ``NodeProgress``. So a test can
make a ``NodeEvents`` with a list's ``append`` as the listener, and call its methods by hand
(``tests/test_core/test_node_events.py``).
"""

from __future__ import annotations

import operator
import time
from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import assert_never

from conductor._internal.base import ItemIndexes, NodePath
from conductor._internal.problem import Problem

from .arguments import ResultReader
from .blueprint import BlueprintFunctionNode, BlueprintNode, all_upstream
from .events import (
    EventBase,
    NodeComplete,
    NodeFailed,
    NodeProgress,
    NodeRetry,
    NodeStart,
    NodeWaiting,
)
from .outcome import Done, NoneInput, Outcome, Skipped, Unanswered
from .progress import CallCount, call_count, is_complete
from .waiting import waiting_record

#: The shortest time between two ``NodeProgress`` events for one node, in seconds.
_SECONDS_BETWEEN_PROGRESS = 0.25

#: A node's position in topological order.
_POSITION = operator.attrgetter("position")


class NodeEvents:
    """Makes the events about the nodes, for one call of ``run.resume()`` or ``run.stream()``.

    The module docstring says which events come when.

    Attributes:
        emit: The listener. It is called with each event.
        nodes: Every function node of the graph and of its subgraphs, in topological order.
        outcomes: The run's outcomes, by node path and item indexes. The scheduler writes them, and this class only
            looks at them.
        reader: Looks up the run's outcomes, to find what a waiting call was given.
        started: The nodes that have had ``NodeStart``, in this call or an earlier one.
        complete: The nodes that have had ``NodeComplete``, in this call or an earlier one.
        changed: The nodes that got an outcome since they were last checked for ``NodeComplete``.
        reused: How many outcomes each node took from the previous run, for ``NodeComplete.reused``. A subgraph
            node has the outcomes of the nodes inside it.
        progress_at: When each node last had ``NodeProgress``, by ``time.monotonic()``.
        progress_count: The count in each node's last ``NodeProgress``.
    """

    def __init__(
        self,
        emit: Callable[[EventBase], None],
        nodes: list[BlueprintFunctionNode],
        outcomes: dict[NodePath, dict[ItemIndexes, Outcome]],
        complete: frozenset[NodePath],
        reader: ResultReader,
    ) -> None:
        """Sets up the events for one call of ``run.resume()`` or ``run.stream()``.

        A run may have been resumed before. So the nodes that already have outcomes count as started, and the nodes
        that were complete at the end of the last ``resume()`` count as complete. They get no second event.

        Args:
            emit: The listener.
            nodes: Every function node, in topological order.
            outcomes: The run's outcomes, by node path and item indexes.
            complete: The node paths of the nodes that were complete when the last ``resume()`` ended. A stream
                has already said so, or would have.
            reader: Looks up the run's outcomes, to find what a waiting call was given.
        """
        self.emit, self.nodes, self.outcomes, self.reader = emit, nodes, outcomes, reader
        with_outcomes: set[BlueprintFunctionNode] = {
            node for node in nodes if outcomes[node.node_path]
        }
        self.started: set[BlueprintNode] = set(with_outcomes)
        for node in with_outcomes:
            subgraph = node.parent
            while subgraph is not None:
                self.started.add(subgraph)
                subgraph = subgraph.parent
        complete_before: set[BlueprintFunctionNode] = {
            node for node in nodes if node.node_path in complete
        }
        self.complete: set[BlueprintNode] = set(complete_before)
        for node in complete_before:
            self._complete_subgraphs_around(node, announce=False)
        self.changed = with_outcomes - complete_before
        self.reused: defaultdict[BlueprintNode, int] = defaultdict(int)
        self.progress_at: dict[BlueprintFunctionNode, float] = {}
        self.progress_count: dict[BlueprintFunctionNode, CallCount] = {}

    def call_starts(self, node: BlueprintNode) -> None:
        """A call of ``node`` is about to be made. Sends ``NodeStart``, unless the node has started already."""
        if node not in self.started:
            if node.parent is not None:
                self.call_starts(node.parent)
            self.started.add(node)
            self.emit(NodeStart(loc=node.node_path))

    def outcome_reused(self, node: BlueprintFunctionNode) -> None:
        """The next outcome recorded for ``node`` is reused from the previous run.

        Counts it for the ``NodeComplete`` of ``node`` and of each subgraph node around it.
        """
        around: BlueprintNode | None = node
        while around is not None:
            self.reused[around] += 1
            around = around.parent

    def outcome_recorded(
        self, node: BlueprintFunctionNode, item_indexes: ItemIndexes, outcome: Outcome
    ) -> None:
        """``outcome`` was written for the call ``item_indexes`` of ``node``.

        An outcome that came without a call, because it was reused or is ``NoneInput``, starts its node just as a
        call would. A ``Skipped`` outcome doesn't. An ``Unanswered`` one also sends ``NodeWaiting``.
        """
        self.changed.add(node)
        match outcome:
            case Skipped():
                pass
            case Done() | NoneInput():
                self.call_starts(node)
            case Unanswered():
                self.call_starts(node)
                self.emit(
                    NodeWaiting(
                        loc=node.node_path,
                        waiting=waiting_record(node, item_indexes, outcome, self.reader),
                    )
                )
            case _:
                assert_never(outcome)
        if node.loops:
            self._progress(node)

    def call_retries(
        self, node: BlueprintFunctionNode, attempt: int, delay: float, cause: Problem
    ) -> None:
        """A call of ``node`` failed with ``cause``, and will be retried after ``delay`` seconds.

        ``attempt`` is the number of the retry: 1 for the first.
        """
        self.emit(
            NodeRetry(
                loc=node.node_path,
                attempt=attempt,
                retries=node.policy.retries,
                delay=delay,
                cause=cause,
            )
        )

    def call_failed(self, node: BlueprintFunctionNode, cause: Problem) -> None:
        """A call of ``node`` failed with ``cause``, and won't be retried.

        The run then fails, or ends ``cancelled`` if it was cancelled by then.
        """
        self.emit(NodeFailed(loc=node.node_path, cause=cause))

    def nothing_is_ready(
        self, running: Iterable[tuple[BlueprintFunctionNode, ItemIndexes]]
    ) -> None:
        """No call is ready to be made. Sends ``NodeComplete`` for every node that is now complete.

        Only the nodes that got an outcome since the last check are looked at. A node that still has a call
        running, or a running node upstream, is left for a later check.

        Args:
            running: The calls that are on a thread, waiting to start, or waiting to be retried.
        """
        if not self.changed:
            return
        nodes_running = {node for node, _ in running}
        for node in sorted(self.changed, key=_POSITION):
            if node in nodes_running or nodes_running & all_upstream(node):
                continue
            self.changed.discard(node)
            if node in self.complete:
                continue
            count = call_count(self.outcomes, node)
            if not is_complete(count):
                continue
            self.complete.add(node)
            if node.loops and node in self.progress_count:
                self._progress(node, last=True)
            self.emit(NodeComplete(loc=node.node_path, count=count, reused=self.reused[node]))
            self._complete_subgraphs_around(node, announce=True)

    def scheduler_ended(self) -> None:
        """The scheduler has ended: it makes no more calls, and none is running.

        Sends ``NodeComplete`` for every node that is complete. Every node that had a ``NodeProgress`` and isn't
        complete gets one last ``NodeProgress`` with its final count.
        """
        self.changed.update(self.nodes)
        self.nothing_is_ready(())
        for node in self.nodes:
            if node.loops and node in self.progress_count:
                self._progress(node, last=True)

    def _progress(self, node: BlueprintFunctionNode, last: bool = False) -> None:
        """Sends ``NodeProgress`` for a node inside a loop.

        Nothing is sent if the last one is less than a quarter of a second old, or had the same count. With
        ``last``, the clock is ignored, since this is the node's final count. The count is the node's
        ``CallCount``, the same as in ``run.progress``.
        """
        now = time.monotonic()
        if not (last or now - self.progress_at.get(node, -1.0) >= _SECONDS_BETWEEN_PROGRESS):
            return
        count = call_count(self.outcomes, node)
        if self.progress_count.get(node) != count:
            self.progress_at[node], self.progress_count[node] = now, count
            self.emit(NodeProgress(loc=node.node_path, count=count))

    def _complete_subgraphs_around(self, node: BlueprintNode, announce: bool) -> None:
        """Marks the subgraph node around ``node`` as complete, once every node inside it is.

        Then it does the same for the subgraph node around that one, and so on outwards. With ``announce``, each
        subgraph node marked gets a ``NodeComplete``.
        """
        subgraph = node.parent
        while (
            subgraph is not None
            and subgraph not in self.complete
            and all(inner in self.complete for inner in subgraph.inner_nodes.values())
        ):
            self.complete.add(subgraph)
            if announce:
                self.emit(
                    NodeComplete(
                        loc=subgraph.node_path,
                        count=call_count(self.outcomes, subgraph),
                        reused=self.reused[subgraph],
                    )
                )
            subgraph = subgraph.parent
