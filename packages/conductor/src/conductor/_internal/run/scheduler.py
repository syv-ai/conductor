"""Calls every node of a run that is ready, and writes one outcome per call into the run.

``Scheduler(...).run()`` is what one call of ``run.resume()`` or ``run.stream()`` does. It returns when the run
is complete, waiting for a person, failed or stopped.

The example graph::

    lines(text: str) -> Series[str]          # called once; returns ["a", "b", "c"]
    shout(line: str, mark: str = "!") -> str # wired to lines.result: called once per line
    join(texts: Series[str]) -> str          # wired to shout.result: called once, with ["A!", "B!", "C!"]

A run of it has five calls. A call is a node and item indexes: ``lines ()``, ``shout (0,)``, ``shout (1,)``,
``shout (2,)`` and ``join ()``. For each, the scheduler asks ``arguments_for(node, item_indexes)``. For
``shout (1,)`` the answer is ``NOT_READY`` until ``lines`` has an outcome, and ``{"line": "b", "mark": "!"}``
after. The scheduler then calls ``shout`` with those arguments and writes what it returns as the outcome
``outcomes[("shout",)][(1,)] = Done({"result": "B!"})``. Outcomes are the only thing it changes in a run.

``Scheduler``'s jobs follow, and its methods are in the same order.

1. **Choosing the next call.** ``run`` first tries every call that has no outcome, node by node in topological
   order (``_try_every_call``). When no node is called on a thread that is the whole run: each call is made
   where it is tried, so its outcome exists before the next node is tried. When ``lines`` is called on a
   thread, the pass finds ``shout`` and ``join`` not ready and ends. From then on a recorded outcome says what
   to try: the calls of downstream nodes that read it go on ``ready`` (``_make_downstream_ready``), and the
   loop in ``run`` takes them from there one at a time (``_start_if_ready``). A call that collects a loop, as
   ``join ()`` does, is not tried after every ``shout``. It is tried once, when no call upstream of it is
   running (``_release_collectors``).

2. **Threads, retries and timeouts.** A node whose ``Policy`` has ``concurrency`` 1, no ``attempt_timeout`` and
   no ``retries`` is called on the scheduler's own thread, where it is tried. Any other call is made as an
   ``_Attempt``. With ``concurrency`` above 1 or an ``attempt_timeout`` the attempt runs on a pool thread, at
   most ``concurrency`` of one node at a time, and comes back through the queue ``returned``. A failed attempt
   that the policy retries is made again after ``retry_delay`` seconds, doubled for each attempt before it.
   An attempt that is still on its thread after ``attempt_timeout`` seconds fails the run; its thread cannot be
   stopped. Only node functions run on pool threads. Outcomes are written on the scheduler's thread.

3. **Reusing answers in item order.** A run made with ``graph.with_inputs(previous=run)`` takes an outcome of
   the earlier run in place of a call with the same arguments (``run/reuse.py``). A person's answer is taken
   once, by the first call in item order that has those arguments. Calls on threads come back in any order, so
   a call that could take an answer is left untried until every call of its node before it has been tried
   (``_in_turn``), and ``_try_every_call`` runs again for it.

4. **Stopping, the item limit and checkpoints.** The first call that fails ends the run as ``failed``. So does
   ``stop`` (a cancel, a hand back, the run's ``timeout``) with its own state, and a series with more items than
   ``max_items`` as ``stopped``. In each case ``end`` is called: no more calls start, and the attempts
   already on threads are waited for and their outcomes kept, so a later ``resume()`` never calls them again.
   With ``checkpoint_every`` the host is handed a copy of the run at most that often.

5. **Telling a listener.** For ``run.stream()`` the scheduler tells a ``NodeEvents`` when a call starts, when
   an outcome is recorded, when a call is retried or fails, and when nothing is ready. ``NodeEvents`` makes the
   events (``node_events.py``).

6. **Only the nodes that cost nothing.** ``run.estimate()`` calls only the nodes whose
   ``Policy`` says ``costs_nothing``: say ``lines``, so that ``run.progress`` knows ``shout`` has three calls
   before ``shout`` is called. Every other node is not ready (``_only_costs_nothing``). A call that raises is
   left without an outcome, and the run ends ``ready``.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue
from typing import Any, Literal

from conductor._internal.ask import Ask
from conductor._internal.base import ItemIndexes, NodePath, place_text
from conductor._internal.problem import (
    ExternalError,
    InvalidCallError,
    NodeError,
    Problem,
    run_problems,
)

from .call import call, jsonable
from .events import EventBase
from .flat import FlatFunctionNode, all_upstream, outcome_loc
from .node_events import NodeEvents
from .outcome import (
    Answered,
    Done,
    NoneInput,
    NotReady,
    Outcome,
    Skipped,
    Unanswered,
)
from .progress import all_item_indexes, has_every_outcome, length
from .reuse import ReusableOutcomes
from .state import COMPLETE, READY, RUNNING, WAITING, EndedEarly, RunState

#: The exceptions of a call that is never retried, whatever the ``retry_on`` of its ``Policy`` names: the node's
#: own failure and a question for a person. Calling again would raise the same.
_NEVER_RETRIED = (NodeError, Ask)

#: How long the scheduler waits for a thread before it asks ``stop`` again, in seconds.
_SECONDS_BETWEEN_STOP_CHECKS = 0.05

_logger = logging.getLogger("conductor")


@dataclass(slots=True)
class _Attempt:
    """One attempt at one call that runs on a thread or may be retried.

    Example: ``_Attempt(shout, (1,), {"line": "b", "mark": "!"}, 0)`` is the first attempt at ``shout (1,)``.

    Attributes:
        node: The node that is called.
        item_indexes: Which call of the node it is.
        arguments: The arguments of the call.
        number: 0 for the first attempt at the call, 1 for its first retry.
    """

    node: FlatFunctionNode
    item_indexes: ItemIndexes
    arguments: dict[str, Any]
    number: int


@dataclass(slots=True)
class _Returned:
    """What a pool thread hands back when a node's function has returned or raised.

    Attributes:
        attempt: The attempt that was made.
        outputs: The node's outputs by name, or None when ``error`` is set.
        error: What the function raised, or the problem a door of the call refused it with, or None.
    """

    attempt: _Attempt
    outputs: Any
    error: BaseException | Problem | None


@dataclass(slots=True, order=True)
class _Timer:
    """Something the scheduler does at a set time. Timers sort by ``when``, then by ``order``.

    Attributes:
        when: The ``time.monotonic()`` at which it is due.
        order: A counter, so two timers due at the same time are taken in the order they were set.
        kind: ``"retry"``: make ``attempt``. ``"deadline"``: ``attempt`` is on a thread and its ``attempt_timeout`` is up.
        attempt: The attempt to make, or the attempt whose time is up.
        seconds: How long after it was set the timer is due.
    """

    when: float
    order: int
    kind: Literal["retry", "deadline"] = field(compare=False)
    attempt: _Attempt = field(compare=False)
    seconds: float = field(compare=False)


@dataclass(slots=True)
class _AnswerOrder:
    """How far the calls of one node have been tried in item order, for a node that takes a person's answers
    from an earlier run.

    Attributes:
        order: The item indexes of the node's calls, in item order.
        position: The position of each of them in ``order``.
        reached: How many calls, counted from the first in ``order``, have been tried.
        passed: The calls counted as tried because they cannot be tried in this ``resume()``.
    """

    order: list[ItemIndexes]
    position: dict[ItemIndexes, int]
    reached: int = 0
    passed: set[ItemIndexes] = field(default_factory=set)


@dataclass(slots=True)
class _Checkpoints:
    """Hands the host a copy of the run at most every ``every`` seconds, while outcomes are recorded.

    Attributes:
        save: Called with the run's state. It copies the run and hands the copy to the host.
        every: The least number of seconds between two checkpoints.
        next_at: The ``time.monotonic()`` from which the next recorded outcome is followed by a checkpoint.
    """

    save: Callable[[RunState], None]
    every: float
    next_at: float


class Scheduler:
    """One call of ``run.resume()`` or ``run.stream()``: ``Scheduler(...).run()`` calls every node that is ready.

    The module docstring says what it does, job by job. The attributes are listed by job in ``__init__``.
    """

    def __init__(
        self,
        *,
        nodes: list[FlatFunctionNode],
        threads: int,
        outcomes: dict[NodePath, dict[ItemIndexes, Outcome]],
        reusable: ReusableOutcomes,
        finished: frozenset[NodePath],
        max_items: int | None,
        arguments_for: Callable[[FlatFunctionNode, ItemIndexes], dict[str, Any] | Skipped | NoneInput | Literal[NotReady.NOT_READY]],
        emit: Callable[[EventBase], None] | None,
        stop: Callable[[], RunState | None] | None,
        context: Any,
        checkpoint: Callable[[RunState], None] | None = None,
        checkpoint_every: float | None = None,
        only_costs_nothing: bool = False,
    ) -> None:
        """
        Args:
            nodes: Every function node of the graph and of its subgraphs, in topological order.
            threads: How many pool threads the graph needs: the sum of ``concurrency`` over its nodes that are
                called on threads.
            outcomes: The run's outcomes, by node path and item indexes. The scheduler writes into this dict.
            reusable: The outcomes of the previous run that a call may reuse. The scheduler takes out the answers it
                reuses and drops a node's outcomes once the node is complete.
            finished: The node paths of the nodes that were complete when the run was last resumed.
            max_items: The item limit: the most items a series that a loop runs over may have. None for no limit.
            arguments_for: Returns the arguments of one call, or why it cannot be called (``run/arguments.py``).
            emit: The listener of ``run.stream()``, called with each event about a node. None for ``run.resume()``.
            stop: Asked before each call and while waiting for threads. It returns the state the run ends with
                when the scheduler must stop now: an ``EndedEarly`` (cancelled, timed out), or ``RUNNING`` when
                the host takes the run back to resume it elsewhere. Otherwise it returns None.
            context: The run's context, passed to each node that takes it; ``None`` when the host passed none.
            checkpoint: Called with the run's state to hand the host a copy of the run.
            checkpoint_every: The least number of seconds between two checkpoints. None for no checkpoints.
            only_costs_nothing: Whether only the nodes whose ``Policy`` says ``costs_nothing`` are called (job 6).
        """
        if only_costs_nothing:
            arguments_for = _only_costs_nothing(arguments_for)
        self.nodes, self.threads, self.all_outcomes, self.max_items, self.arguments_for = nodes, threads, outcomes, max_items, arguments_for
        self.stop, self.context, self.only_costs_nothing = stop, context, only_costs_nothing

        # 1. Choosing the next call.
        #: The state ``run`` returns: ``RUNNING`` until ``end`` or ``_close`` sets another.
        self.state: RunState = RUNNING
        #: False once ``end`` was called: no more calls are started.
        self.going = True
        #: The node paths of the nodes that are complete: as given, and from ``_close`` as this ``resume()`` leaves them.
        self.finished = finished
        #: Each node's outcomes by item indexes. These are the dicts of ``all_outcomes``, by node in place of node path.
        self.outcomes = {node: outcomes[node.node_path] for node in self.nodes}
        #: The nodes with an outcome written before one that is earlier in item order. ``_put_in_order`` sorts them.
        self.out_of_order: set[FlatFunctionNode] = set()
        #: The calls to try, as a heap of (node position, item indexes): the first node in topological order first.
        self.ready: list[tuple[int, ItemIndexes]] = []
        #: Whether an outcome was recorded, or every call was tried, since ``_when_nothing_is_ready`` last ran.
        self.changed = True
        #: The collecting calls to try once no call upstream of them is running: the node, and the item indexes
        #: that the calls it collects start with.
        self.collectors: set[tuple[FlatFunctionNode, ItemIndexes]] = set()
        #: The calls with an attempt on a thread, waiting for a thread, or waiting to be retried, each with that attempt.
        #: speed only: a dict where a set would do. It is listed whenever nothing is ready, and a set that had 2,000 calls
        #: in it is slow to list for the rest of its life: 18% of a run of 6,000 calls on threads.
        self.running: dict[tuple[FlatFunctionNode, ItemIndexes], _Attempt] = {}

        # 2. Threads, retries and timeouts.
        #: The pool the attempts run on. It is made when the first attempt goes to a thread.
        self.pool: ThreadPoolExecutor | None = None
        #: Each node's attempts that wait for a thread, first in first out.
        self.waiting_for_thread: defaultdict[FlatFunctionNode, deque[_Attempt]] = defaultdict(deque)
        #: How many attempts of each node are on threads. The node's ``concurrency`` is the most there may be.
        self.on_threads: defaultdict[FlatFunctionNode, int] = defaultdict(int)
        #: The number of the attempt that each call last put on a thread.
        self.attempts: dict[tuple[FlatFunctionNode, ItemIndexes], int] = {}
        #: What the pool threads hand back: an attempt with its outputs or its error, or an exception that ``call``
        #: itself raised.
        self.returned: SimpleQueue[_Returned | BaseException] = SimpleQueue()
        #: The retries to make and the deadlines of the attempts on threads, as a heap: the one due first is first.
        self.timers: list[_Timer] = []
        #: Gives each timer its ``order``.
        self.timer_order = itertools.count()

        # 3. Reusing answers in item order.
        #: The outcomes of the previous run that a call may reuse.
        self.reusable = reusable
        #: How far each node that takes a person's earlier answers has been tried in item order.
        self.answer_orders: dict[FlatFunctionNode, _AnswerOrder] = {}
        #: Whether a call was left untried because a call of its node before it has not been tried yet.
        self.out_of_turn = False
        #: How many outcomes were recorded since ``_try_every_call`` last ran, not counting its own.
        self.recorded_since_last_pass = 0

        # 4. Stopping, the item limit and checkpoints.
        #: When the host is handed a copy of the run. None when it asked for no checkpoints.
        self.checkpoints = (
            _Checkpoints(checkpoint, checkpoint_every, time.monotonic() + checkpoint_every)
            if checkpoint is not None and checkpoint_every is not None
            else None
        )

        # 5. Telling a listener.
        #: Makes the events about each node. None when nobody listens.
        self.node_events = NodeEvents(emit, nodes, outcomes, finished, self.arguments_for) if emit is not None else None

    def run(self) -> tuple[RunState, frozenset[NodePath]]:
        """Call every node that is ready, until none is and no attempt is on a thread, or until ``end`` is called.

        Returns:
            The run's state (complete, waiting for a person, ended early with its cause, or ``RUNNING`` when
            the host took the run back), and the node paths of the nodes that are complete.
        """
        try:
            self._check_item_limit()
            self._try_every_call()
            while self.going:
                if self.stop is not None and (stopped := self.stop()) is not None:
                    self.end(stopped)
                elif self.ready:
                    position, item_indexes = heapq.heappop(self.ready)
                    self._start_if_ready(self.nodes[position], item_indexes)
                elif self.changed:
                    self.changed = False
                    if not self._when_nothing_is_ready():
                        break
                else:
                    self._wait()
            if self.checkpoints is not None and self.running:  # the process may die before the attempts on threads are back
                self._checkpoint(self.checkpoints)
            while self.running:  # a call that was made is never made again: wait for its outcome
                self._wait()
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=False, cancel_futures=True)
        self._close()
        return self.state, self.finished

    def end(self, state: RunState) -> None:
        """Start no more calls. The attempts that wait for a thread or for a retry are dropped; ``run`` waits for
        the attempts on threads and keeps their outcomes.

        Args:
            state: The state ``run`` returns: an ``EndedEarly`` with its cause, or ``RUNNING`` when the host
                takes the run back.
        """
        self.state, self.going = state, False
        for node, waiting in self.waiting_for_thread.items():
            for attempt in waiting:
                self.running.pop((node, attempt.item_indexes), None)
            waiting.clear()
        for timer in self.timers:
            if timer.kind == "retry":
                self.running.pop((timer.attempt.node, timer.attempt.item_indexes), None)

    # -- 1. Choosing the next call --------------------------------------------------------------------------------

    def _try_every_call(self) -> None:
        """Try every call that has no outcome and is not running, node by node in topological order.

        ``run`` does this once at the start. It is done again only for calls left untried out of turn (job 3).
        """
        self.out_of_turn, self.recorded_since_last_pass = False, 0
        running = self.running
        for node in self.nodes:
            outcomes = self.outcomes[node]
            for item_indexes in all_item_indexes(self.all_outcomes, node.loops, ())[0]:
                if item_indexes not in outcomes and (node, item_indexes) not in running:
                    if not self.going or (self.stop is not None and self.stop() is not None):
                        return
                    self._start_if_ready(node, item_indexes, trying_every_call=True)

    def _start_if_ready(self, node: FlatFunctionNode, item_indexes: ItemIndexes, trying_every_call: bool = False) -> None:
        """Try one call: ask for its arguments, and if it is ready, call the node or record why it is not called.

        Example: ``_start_if_ready(shout, (1,))`` does nothing while ``lines`` has no outcome. After, it calls
        ``shout(line="b", mark="!")`` and records ``Done({"result": "B!"})``.

        One of five things happens:

        - The call is not ready, or is out of turn for an answer it could reuse: nothing.
        - The call is skipped or not called (``arguments_for`` returns that outcome): the outcome is recorded.
        - An earlier run has an outcome for these arguments: that outcome is recorded, and the node is not called.
        - The node is called on this thread, and the outcome is recorded.
        - The node is called on a pool thread or may be retried: an ``_Attempt`` is made, and the outcome is
          recorded when it is back.

        Args:
            node: The node to call.
            item_indexes: Which call of the node.
            trying_every_call: True from ``_try_every_call``, which has checked that the call has no outcome and
                is not running, and which reaches the downstream calls itself.
        """
        if not trying_every_call and (item_indexes in self.outcomes[node] or (node, item_indexes) in self.running):
            return
        try:
            arguments = self.arguments_for(node, item_indexes)
        except InvalidCallError as missing:  # an edge reads a field that the record it reads from does not have
            (problem,) = missing.problems
            self._record(node, item_indexes, self._failure(node, item_indexes, problem), trying_every_call)
            return
        if arguments is NotReady.NOT_READY:
            return
        policy = node.policy
        if not isinstance(arguments, dict):  # a Skipped or NoneInput outcome
            self._record(node, item_indexes, arguments, trying_every_call)
        elif (previous := self.reusable.take(node, item_indexes, arguments, self.all_outcomes, lambda: self._in_turn(node, item_indexes))) is not None:
            if previous is NotReady.NOT_ITS_TURN:
                self.out_of_turn = True
                return
            if self.node_events is not None:
                self.node_events.outcome_reused(node)
            self._record(node, item_indexes, previous, trying_every_call)
        else:
            if self.node_events is not None:
                self.node_events.call_starts(node)
            if policy.threaded or policy.retries:
                self._make_attempt(_Attempt(node, item_indexes, arguments, 0), trying_every_call)
            else:
                # speed only: what `call` returns goes straight to `_outcome` and is not kept in a local here. An exception in a
                # local of this frame is a reference cycle (its traceback has the frame of `call`, which points back to this
                # one), left for the garbage collector. That costs 20% of a run of 2,000 calls that each ask a person.
                outcome = self._outcome(node, item_indexes, arguments, *call(node, arguments, self.context))
                self._record(node, item_indexes, outcome, trying_every_call)

    def _record(
        self, node: FlatFunctionNode, item_indexes: ItemIndexes, outcome: Outcome | Problem | None, trying_every_call: bool = False
    ) -> None:
        """One call is over: write its outcome into the run, and do what follows from it.

        Example: ``_record(lines, (), Done({"result": ["a", "b", "c"]}))`` writes ``outcomes[("lines",)][()]`` and
        puts ``shout (0,)``, ``shout (1,)`` and ``shout (2,)`` on ``ready``.

        What follows, in this order: a checkpoint if one is due, the item limit, the listener, and the downstream
        calls. A ``Problem`` in place of an outcome means the call failed: nothing is written, and the run ends
        as ``failed``. ``None`` means the call raised while only the nodes that cost nothing are called (job 6):
        nothing is written, and the run goes on.

        Args:
            node: The node of the call.
            item_indexes: Which call of the node.
            outcome: The call's outcome, the problem it failed with, or ``None``.
            trying_every_call: True from ``_try_every_call``, which reaches the downstream calls itself.
        """
        # speed only: isinstance of a pydantic model costs four times as much, and no class derives from Problem
        if type(outcome) is Problem:
            self._fail(node, outcome)
            return
        if outcome is None:
            return
        outcomes = self.outcomes[node]
        if outcomes and item_indexes not in outcomes and item_indexes < next(reversed(outcomes)):
            self.out_of_order.add(node)
        outcomes[item_indexes] = outcome
        if not trying_every_call:
            self.recorded_since_last_pass += 1
            self.changed = True
        if (checkpoints := self.checkpoints) is not None and time.monotonic() >= checkpoints.next_at:
            self._checkpoint(checkpoints)
        if self.max_items is not None and node.looped_outputs and outcome.has_outputs:
            if self._stop_if_over_item_limit(node, item_indexes, outcome, self.max_items):
                return
        if self.node_events is not None:
            self.node_events.outcome_recorded(node, item_indexes, outcome)
        if not trying_every_call and self.going and outcome.has_outputs:  # no call that reads a waiting outcome is ready
            self._make_downstream_ready(node, item_indexes)

    def _fail(self, node: FlatFunctionNode, cause: Problem) -> None:
        """A call of ``node`` failed. The first failure ends the run as ``failed``; a later one changes nothing."""
        if self.going:
            self.end(EndedEarly("failed", cause))
            if self.node_events is not None:
                self.node_events.call_failed(node, cause)

    def _make_downstream_ready(self, node: FlatFunctionNode, item_indexes: ItemIndexes) -> None:
        """Put the calls that read the outcome of ``node`` for ``item_indexes`` on ``ready``, or on ``collectors``.

        Example: after ``shout (1,)``, the downstream node ``join`` collects every call of ``shout``, so
        ``(join, ())`` goes on ``collectors``. After ``lines ()``, ``shout`` is one loop deeper, so every call of
        ``shout`` goes on ``ready``. A downstream node inside the same loops has one call with the same item
        indexes, and that call goes on ``ready``.
        """
        depth = len(item_indexes)
        for downstream, shared_indexes in node.downstream:
            if shared_indexes < depth:
                self.collectors.add((downstream, item_indexes[:shared_indexes]))
            # speed only: the branch below finds this one call too. Worth about 2% of a run whose 6,000 calls all come back from threads.
            elif len(downstream.loops) == depth:
                heapq.heappush(self.ready, (downstream.position, item_indexes))
            else:
                for downstream_indexes in all_item_indexes(self.all_outcomes, downstream.loops, item_indexes)[0]:
                    heapq.heappush(self.ready, (downstream.position, downstream_indexes))

    def _when_nothing_is_ready(self) -> bool:
        """``ready`` is empty: find more to try, and say whether ``run`` goes on.

        In this order:

        1. Put the collecting calls that no running call is upstream of on ``ready``. If there is one, go on.
        2. Tell the listener, which says which nodes are complete.
        3. If a call is running, go on: ``run`` waits for it.
        4. If a call was left untried out of turn (job 3), try every call again and go on. This is done only
           when an outcome was recorded since the last time, or when calls that cannot be tried in this
           ``resume()`` were just counted as tried: otherwise trying again would find the same.

        Returns:
            False when nothing more can be called in this ``resume()``.
        """
        if self.collectors and self._release_collectors():
            self.changed = True
            return True
        if self.node_events is not None:
            self.node_events.nothing_is_ready(self.running)
        if self.running:
            return True
        if self.out_of_turn and (self.recorded_since_last_pass or self._skip_calls_that_cannot_start()):
            self._try_every_call()
            self.changed = True
            return True
        return False

    def _release_collectors(self) -> bool:
        """Put on ``ready`` the collecting calls that no running call is upstream of.

        Example: ``(join, ())`` is in ``collectors`` after the first ``shout`` is recorded. While another call of
        ``shout`` is running it stays there. When none is, ``join ()`` goes on ``ready``.

        Returns:
            Whether any call was put on ``ready``.
        """
        running = list(self.running)
        released = False
        for node, item_indexes in list(self.collectors):
            upstream, depth = all_upstream(node), len(item_indexes)
            if not any(running_node in upstream and running_indexes[:depth] == item_indexes for running_node, running_indexes in running):
                self.collectors.discard((node, item_indexes))
                released = True
                for call_indexes in all_item_indexes(self.all_outcomes, node.loops, item_indexes)[0]:
                    heapq.heappush(self.ready, (node.position, call_indexes))
        return released

    def _close(self) -> None:
        """Set the state and the complete nodes that ``run`` returns, once no call is running.

        A node is complete when it has an outcome for every call and none of them is waiting for a person. When
        ``end`` was not called, the run is ``COMPLETE`` when every node is complete, ``WAITING`` when a node is
        waiting, and ``READY`` when only the nodes that cost nothing were called and a node lacks an outcome.
        Otherwise it has the state ``end`` was given.

        Raises:
            RuntimeError: ``end`` was not called, every node was called, no call is waiting for a person, and a
                node lacks an outcome. The scheduler failed to try a call that was ready.
        """
        self._put_in_order()
        waiting: set[FlatFunctionNode] = set()
        unfinished: set[FlatFunctionNode] = set()
        for node, outcomes in self.outcomes.items():
            if node.node_path in self.finished:
                continue
            if Unanswered in {type(outcome) for outcome in outcomes.values()}:
                waiting.add(node)
            # speed only: has_every_outcome says the same of a node inside no loop. Not calling it is worth 2% of a run of 500 chained nodes.
            elif not (has_every_outcome(self.all_outcomes, node) if node.loops else outcomes):
                unfinished.add(node)
        if self.going and not waiting and not unfinished:
            self.state, self.finished = COMPLETE, frozenset(node.node_path for node in self.nodes)
        elif self.going and not waiting and not self.only_costs_nothing:
            lacking = sorted(node.node_path for node in unfinished)
            raise RuntimeError(f"the scheduler has nothing to try, and no outcome for every call of {lacking}")
        else:
            self.finished = frozenset(node.node_path for node in self.nodes if node not in waiting and node not in unfinished)
            if self.going:
                self.state = WAITING if waiting else READY
        self.reusable.drop(self.finished)  # a complete node has no call left to take them
        if self.node_events is not None:
            self.node_events.scheduler_ended()

    def _put_in_order(self) -> None:
        """Sort the outcomes of each node in ``out_of_order`` by item indexes.

        Attempts on threads come back in any order. The run has every node's outcomes in item order whenever a
        host can see it: in a checkpoint's copy, and when ``run`` returns.
        """
        for node in self.out_of_order:
            outcomes = self.outcomes[node]
            in_order = sorted(outcomes.items())
            outcomes.clear()
            outcomes.update(in_order)
        self.out_of_order.clear()

    # -- 2. Threads, retries and timeouts -------------------------------------------------------------------------

    def _make_attempt(self, attempt: _Attempt, trying_every_call: bool = False) -> None:
        """Make one attempt at a call, for a node that is called on threads or has ``retries``.

        A node called on threads: the attempt joins the node's queue for a thread. Any other node is called here,
        on the scheduler's thread, and its outcome is recorded unless the attempt is retried.
        """
        node, item_indexes, arguments = attempt.node, attempt.item_indexes, attempt.arguments
        if not node.policy.threaded:
            outputs, error = call(node, arguments, self.context)
            if not self._retry(attempt, error):
                if attempt.number:
                    self.running.pop((node, item_indexes), None)
                self._record(node, item_indexes, self._outcome(node, item_indexes, arguments, outputs, error), trying_every_call)
            return
        self.running[(node, item_indexes)] = attempt
        self.waiting_for_thread[node].append(attempt)
        self._start_waiting_calls(node)

    def _start_waiting_calls(self, node: FlatFunctionNode) -> None:
        """Put the attempts of ``node`` that wait for a thread on threads, until ``concurrency`` of them are on
        threads. An attempt with an ``attempt_timeout`` gets its deadline timer here, when its thread starts."""
        waiting, policy = self.waiting_for_thread[node], node.policy
        while waiting and self.on_threads[node] < policy.concurrency and self.going:
            attempt = waiting.popleft()
            self.on_threads[node] += 1
            self.attempts[(node, attempt.item_indexes)] = attempt.number
            if self.pool is None:
                self.pool = ThreadPoolExecutor(self.threads, "conductor")
            if policy.attempt_timeout is not None:
                self._set_timer(policy.attempt_timeout, "deadline", attempt)
            self.pool.submit(self._on_thread, attempt)

    def _on_thread(self, attempt: _Attempt) -> None:
        """Runs on a pool thread: call the node and put what it returned or raised on ``returned``.

        ``call`` returns a node's own exception as ``error``. What ``call`` itself raises (``SystemExit``, a bug in
        a type's own validation) is put on ``returned`` as it is, and ``_wait`` raises it on the scheduler's
        thread, as it would be raised for a node called there.
        """
        try:
            outputs, error = call(attempt.node, attempt.arguments, self.context)
        except BaseException as raised:  # noqa: BLE001  the pool would swallow it, and the scheduler would wait for the attempt for ever
            self.returned.put(raised)
        else:
            self.returned.put(_Returned(attempt, outputs, error))

    def _wait(self) -> None:
        """Wait until an attempt is back from its thread, a timer is due, or it is time to ask ``stop`` again.
        Then handle the attempt that is back, if any, and every timer that is due."""
        timeout = self.timers[0].when - time.monotonic() if self.timers else None
        if self.stop is not None:
            timeout = _SECONDS_BETWEEN_STOP_CHECKS if timeout is None else min(timeout, _SECONDS_BETWEEN_STOP_CHECKS)
        try:
            if timeout is None:
                returned = self.returned.get()
            else:
                try:
                    # CPython 3.12's SimpleQueue.get(timeout=) never returns when its timeout runs out before its first
                    # look at an empty queue, so the first look is one that doesn't wait.
                    returned = self.returned.get(block=False)
                except Empty:
                    if timeout <= 0:
                        raise
                    returned = self.returned.get(timeout=timeout)
            if isinstance(returned, BaseException):
                raise returned
            self._call_returned(returned)
        except Empty:
            pass
        now = time.monotonic()
        while self.timers and self.timers[0].when <= now:
            self._due(heapq.heappop(self.timers))

    def _call_returned(self, returned: _Returned) -> None:
        """An attempt is back from its thread: start the node's next waiting attempt, then retry this one or
        record its outcome.

        The outcome is dropped when the attempt's deadline has passed, since the run has failed with ``timeout``
        for it, or when ``end`` dropped the call. One case is kept: the deadline has passed, the function returned
        without an error, and the run is ending. The call was made, so its outcome is recorded and a later
        ``resume()`` does not make it again.
        """
        attempt, outputs, error = returned.attempt, returned.outputs, returned.error
        node, item_indexes = attempt.node, attempt.item_indexes
        self.on_threads[node] -= 1
        latest = self.attempts.get((node, item_indexes)) == attempt.number
        current = latest and (node, item_indexes) in self.running
        self._start_waiting_calls(node)
        if not current:
            if latest and error is None and not self.going and item_indexes not in self.outcomes[node]:
                self.attempts.pop((node, item_indexes))
                self._record(node, item_indexes, self._outcome(node, item_indexes, attempt.arguments, outputs, None))
            return
        self.attempts.pop((node, item_indexes))
        if not self._retry(attempt, error):
            del self.running[(node, item_indexes)]
            self._record(node, item_indexes, self._outcome(node, item_indexes, attempt.arguments, outputs, error))

    def _retry(self, attempt: _Attempt, error: BaseException | Problem | None) -> bool:
        """Set a timer to make the call again, if ``attempt`` failed and its node's ``Policy`` retries the error.

        An ``ExternalError`` is retried, and so is an exception of a class in ``retry_on``. A ``NodeError`` that is
        not an ``ExternalError``, a value a door of the call refused and an ``Ask`` are never retried. The delay is
        ``retry_delay`` seconds, doubled for each attempt before this one.

        Returns:
            Whether the call will be made again. If not, the caller records the outcome of ``attempt``.
        """
        node, item_indexes = attempt.node, attempt.item_indexes
        policy = node.policy
        if error is None or isinstance(error, Problem) or attempt.number >= policy.retries or not self.going:
            return False
        if not isinstance(error, ExternalError) and (isinstance(error, _NEVER_RETRIED) or not isinstance(error, policy.retry_on)):
            return False
        delay = policy.retry_delay * 2**attempt.number
        retry = _Attempt(node, item_indexes, attempt.arguments, attempt.number + 1)
        self.running[(node, item_indexes)] = retry
        self._set_timer(delay, "retry", retry)
        if self.node_events is not None:
            self.node_events.call_retries(node, item_indexes, attempt.number + 1, delay, self._failure(node, item_indexes, error))
        return True

    def _set_timer(self, seconds: float, kind: Literal["retry", "deadline"], attempt: _Attempt) -> None:
        """Set a timer that is due ``seconds`` from now: ``"retry"`` to make ``attempt``, or ``"deadline"`` for
        the ``attempt_timeout`` of ``attempt``, which starts on a thread now."""
        heapq.heappush(self.timers, _Timer(time.monotonic() + seconds, next(self.timer_order), kind, attempt, seconds))

    def _due(self, timer: _Timer) -> None:
        """A timer is due. A retry: make the attempt, unless ``end`` was called. A deadline: if the attempt is
        still on its thread, the call fails with the problem ``timeout``, which ends the run."""
        attempt = timer.attempt
        node, item_indexes = attempt.node, attempt.item_indexes
        if timer.kind == "retry":
            if self.going:
                self._make_attempt(attempt)
        elif self.attempts.get((node, item_indexes)) == attempt.number and (node, item_indexes) in self.running:
            del self.running[(node, item_indexes)]
            self._record(node, item_indexes, run_problems.timeout(*outcome_loc(node, item_indexes), seconds=timer.seconds))

    def _outcome(
        self, node: FlatFunctionNode, item_indexes: ItemIndexes, arguments: dict[str, Any], outputs: Any, error: BaseException | Problem | None
    ) -> Outcome | Problem | None:
        """The outcome of a call from what ``call`` returned: ``Done`` with the outputs, ``Unanswered`` with what the
        ``Ask`` said when the function raised one, or the problem the call failed with.

        A person who is asked is shown what the call received, as JSON, so arguments with no JSON form are a problem,
        and so is an ``Ask`` that does not fit the node's outputs. While only the nodes that cost nothing
        are called, a call that raised has no outcome (``None``): the next ``resume()`` that calls every node
        calls it again, and it fails or asks then.
        """
        if error is None:
            return Done(outputs)
        if self.only_costs_nothing:
            return None
        if isinstance(error, Ask):
            try:
                jsonable(arguments)  # a person is shown what the call received with the question (run.waiting)
                choices, defaults = node.compiled_node.ask_door(error)
            except InvalidCallError as refused:  # arguments with no JSON form, or the Ask door
                (problem,) = refused.problems
                return self._failure(node, item_indexes, problem)
            return Unanswered(error.prompt, choices, defaults)
        return self._failure(node, item_indexes, error)

    def _failure(self, node: FlatFunctionNode, item_indexes: ItemIndexes, error: BaseException | Problem) -> Problem:
        """The problem a call failed with, with the call's loc: a door's problem, or one for what the function raised.

        A door's problem is kept, and a ``NodeError`` gives its own. Any other exception, an ``InvalidCallError`` the
        function let escape included, is written to the log with its traceback; the problem has only its class name,
        as its text may quote values: ``external_failed`` for a class in the node's ``retry_on``, else
        ``execution_failed``.
        """
        loc = outcome_loc(node, item_indexes)
        if isinstance(error, NodeError):
            return Problem.model_construct(type=error.code, loc=loc, msg=error.message, ctx=error.ctx)  # ctx was validated when the error was made
        if isinstance(error, Problem):
            return error.model_copy(update={"loc": loc})
        _logger.error("node %r failed for item %r", place_text(node.node_path), item_indexes, exc_info=error)
        if isinstance(error, node.policy.retry_on):
            return run_problems.external_failed(*loc, error=type(error).__name__)
        return run_problems.execution_failed(*loc, error=type(error).__name__)

    # -- 3. Reusing answers in item order -------------------------------------------------------------------------

    def _in_turn(self, node: FlatFunctionNode, item_indexes: ItemIndexes) -> bool:
        """Whether every call of ``node`` before ``item_indexes`` in item order has been tried.

        Example: the earlier run has one answer for ``approve(line="a")``, and this run has the line ``"a"`` at
        ``(0,)`` and at ``(2,)``. The answer belongs to ``approve (0,)``. If ``approve (2,)`` is tried first,
        because its upstream call came back from a thread first, it is not in turn and takes nothing.

        A call counts as tried when it has an outcome, is running, or is in ``passed``
        (``_skip_calls_that_cannot_start``).
        """
        turn = self.answer_orders.get(node)
        if turn is None or item_indexes not in turn.position:
            order = all_item_indexes(self.all_outcomes, node.loops, ())[0]
            turn = self.answer_orders[node] = _AnswerOrder(order, {indexes: position for position, indexes in enumerate(order)})
        order, outcomes, passed, reached = turn.order, self.outcomes[node], turn.passed, turn.reached
        while reached < len(order):
            next_indexes = order[reached]
            if not (next_indexes in outcomes or (node, next_indexes) in self.running or next_indexes in passed):
                break
            reached += 1
        turn.reached = reached
        return turn.position.get(item_indexes, 0) <= reached

    def _skip_calls_that_cannot_start(self) -> bool:
        """Count as tried every call, of a node in ``answer_orders``, that has not been tried and has no outcome.

        This is done when nothing is running and nothing is ready. Such a call is not ready in this ``resume()``:
        an upstream call is waiting for a person. The calls after it would otherwise never be in turn. If the
        call becomes ready in a later ``resume()``, it takes what answers are left.

        Returns:
            Whether any call was newly counted as tried.
        """
        passed_any = False
        for node, turn in self.answer_orders.items():
            outcomes = self.outcomes[node]
            untried = {indexes for indexes in turn.order[turn.reached :] if indexes not in outcomes and indexes not in turn.passed}
            turn.passed |= untried
            passed_any = passed_any or bool(untried)
        return passed_any

    # -- 4. Stopping, the item limit and checkpoints --------------------------------------------------------------

    def _check_item_limit(self) -> None:
        """At the start: end the run as ``stopped`` if a series that a loop runs over already has more items
        than ``max_items``. A run that stopped at the limit and is resumed with the same limit stops here."""
        item_limit = self.max_items
        if item_limit is None:
            return
        for node in self.nodes:
            for output in node.looped_outputs:
                for item_indexes, outcome in self.outcomes[node].items():
                    if outcome.has_outputs and length(outcome.outputs[output]) > item_limit:
                        self._stop_at_item_limit(node, item_indexes, output, length(outcome.outputs[output]), item_limit)
                        return

    def _stop_if_over_item_limit(
        self, node: FlatFunctionNode, item_indexes: ItemIndexes, outcome: Done | Answered | Skipped | NoneInput, item_limit: int
    ) -> bool:
        """End the run as ``stopped`` if a series in ``outcome`` that a loop runs over has more items than
        ``item_limit``.

        Returns:
            Whether a series has more.
        """
        for output in node.looped_outputs:
            items = length(outcome.outputs[output])
            if items > item_limit:
                self._stop_at_item_limit(node, item_indexes, output, items, item_limit)
                return True
        return False

    def _stop_at_item_limit(self, node: FlatFunctionNode, item_indexes: ItemIndexes, output: str, items: int, item_limit: int) -> None:
        """End the run as ``stopped`` with the problem ``item_limit``, unless ``end`` was called before: the first
        cause stays.

        The problem names the series as the graph with the loop names it. ``node.looped_outputs[output]`` is
        that name: the node and output themselves, or, when the loop is outside a subgraph the node is inside,
        the subgraph node and its output. Its loc has the item indexes of that node.
        """
        if self.going:
            named, series_output = node.looped_outputs[output]
            loc = outcome_loc(named, item_indexes[: len(named.parent.loops) if named.parent is not None else 0])
            series = place_text((named.compiled_node.id,), series_output)
            self.end(EndedEarly("stopped", run_problems.item_limit(*loc, series=series, items=items, max_items=item_limit)))

    def _checkpoint(self, checkpoints: _Checkpoints) -> None:
        """Hand the host a copy of the run now, with its outcomes in item order, and set when the next is due."""
        checkpoints.next_at = time.monotonic() + checkpoints.every
        self._put_in_order()
        checkpoints.save(self.state)


def _only_costs_nothing(
    arguments_for: Callable[[FlatFunctionNode, ItemIndexes], dict[str, Any] | Skipped | NoneInput | Literal[NotReady.NOT_READY]],
) -> Callable[[FlatFunctionNode, ItemIndexes], dict[str, Any] | Skipped | NoneInput | Literal[NotReady.NOT_READY]]:
    """``arguments_for``, with every call of a node whose ``Policy`` does not say ``costs_nothing`` not ready.

    So a ``resume()`` restricted to the nodes that cost nothing goes as any other, and never calls ``shout`` when
    only ``lines`` costs nothing.
    """

    def restricted(node: FlatFunctionNode, item_indexes: ItemIndexes) -> dict[str, Any] | Skipped | NoneInput | Literal[NotReady.NOT_READY]:
        return arguments_for(node, item_indexes) if node.policy.costs_nothing else NotReady.NOT_READY

    return restricted
