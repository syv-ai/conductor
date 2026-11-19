"""The scheduler, which calls every node of a run that is ready, and writes down how each call went.

Each call of ``run.resume()`` or ``run.stream()`` makes one ``Scheduler`` and calls its ``run()``. That returns
when the run is complete, is waiting for a person, has failed, or was stopped.

Here is an example graph::

    lines(text: str) -> Series[str]          # called once; returns ["a", "b", "c"]
    shout(line: str, mark: str = "!") -> str # wired to lines.result: called once per line
    join(texts: Series[str]) -> str          # wired to shout.result: called once, with ["A!", "B!", "C!"]

A run of it has five calls. A call is a node plus its item indexes, which say which item of each loop it is for:
``lines ()``, ``shout (0,)``, ``shout (1,)``, ``shout (2,)`` and ``join ()``.

For each call, the scheduler asks ``arguments_for(node, item_indexes)`` for its arguments. For ``shout (1,)`` the
answer is ``NOT_READY`` until ``lines`` has an outcome, and ``{"line": "b", "mark": "!"}`` after that. The scheduler
then calls ``shout`` with those arguments, and writes what it returns as the call's outcome::

    outcomes.record(shout, (1,), Done({"result": "B!"}))

Outcomes are the only thing the scheduler writes, into the copy of the run's outcomes (``RunOutcomes``) it is given.
The one other thing it changes is the pool of outcomes from an earlier run that its calls may reuse. When a call
reuses an outcome of a node that may ask a person, that outcome is taken out of the pool, so no other call reuses it
too (job 3).

The scheduler has five jobs, described below. Its methods come in the same order.

1. **Choosing the next call.** ``run`` starts by putting every call that has no outcome on ``ready``
   (``_put_every_call_on_ready``). In the example that's ``lines ()`` and ``join ()``, since no call of ``shout``
   is known until ``lines`` has returned. The loop in ``run`` then takes calls off ``ready`` one at a time, the
   node that comes first in topological order first (``_start_if_ready``). From then on, each new outcome says what
   to try next: the downstream calls that read it go on ``ready`` (``_make_downstream_ready``).

   A call that collects a loop, as ``join ()`` does, isn't tried after every ``shout``. It is tried at the start,
   like every call. After that, ``_release_collectors`` puts it back on ``ready`` when no call upstream of it in its
   loop is running.

2. **Threads, retries and timeouts.** Every call is made as an ``_Attempt``. An attempt waits in
   ``waiting_to_start`` until fewer than ``most_at_once`` calls of its node are running (``_count_of``).

   Who counts the running calls depends on the node. A node that sets ``concurrency`` has its calls counted across
   the whole process, whatever run and whatever place in a graph they are for. Its definition does that counting
   (``RunningCalls``). Any other node has its calls counted by this run alone, separately for each place in the
   graph.

   While an attempt waits, the scheduler goes on calling other nodes. When a counted call ends, in this run or in
   another, ``None`` is put on the ``returned`` queue of every run waiting for that node. Each of those runs then
   tries again, and asks ``stop`` before each call it starts. The first to ask is the one that starts. A run that
   has just made a call on its own thread lets the runs it woke ask first, if they were parked (waiting in
   ``_wait``). It doesn't wait for a run that is busy with something else.

   An attempt runs on a pool thread when its node has ``concurrency`` above 1 or an ``attempt_timeout``, and comes
   back through ``returned``. Any other attempt is made on the scheduler's own thread. A call counts as running
   until its function returns, whatever thread it runs on.

   When an attempt fails and the policy retries the error, the attempt is made again after ``retry_delay`` seconds,
   doubled for each attempt before it. It doesn't count as running while it waits.

   An attempt that is still on its thread after ``attempt_timeout`` seconds fails the run. Python can't stop a
   thread, so the call keeps its place in the count until its function returns. With ``concurrency`` set, every run
   in the process counts it against that limit. Without it, only this ``resume()`` counts it, at that place in the
   graph.

   Only node functions run on pool threads. Outcomes are always written on the scheduler's thread. Between any two
   calls it makes, the scheduler takes in the attempts back from their threads and the timers that are due
   (``_take_news``), so a slot on a thread is filled, a retry is made and a deadline is kept while it makes calls
   of its own. An attempt it sees back only after its deadline, because it was busy with such a call, is timed
   out by when its function returned.

3. **Reusing in item order.** A run made with ``graph.with_inputs(previous=run)`` reuses an outcome of the earlier
   run in place of a call with the same arguments (``run/reuse.py``). For a node that may ask a person, each of its
   outcomes, an answer or a return, is reused only once. It goes to the first call in item order that has those
   arguments.

   Calls on threads come back in any order. So a call that could reuse such an outcome is left untried until every
   call of its node before it has been tried (``_in_turn``). It goes back on ``ready``
   (``_put_every_call_on_ready``) once nothing is ready, nothing is running, and an outcome was recorded since it
   was left. If a call before it waits on a person's answer upstream, it waits for that answer too.

4. **Stopping, the item limit and checkpoints.** The first call that fails ends the run as ``failed``. The
   exception is a run that was cancelled by the time the failure comes back. A node can be told of a cancel, and
   it stops by raising. So the run ends ``cancelled``, and the failed call is made again on resume.

   ``stop`` reports a cancel, a hand back, or the run's ``timeout``. The first time it says to stop, the run ends
   with the state it gives, even if it says otherwise later. A series with more items than ``max_items`` ends the
   run as ``stopped``.

   In each case ``end`` is called, and no more calls start. The attempts already on threads are waited for, and
   their outcomes are kept, so a later ``resume()`` never makes those calls again. With ``checkpoint_every``, the
   host is handed a copy of the run at most that often.

5. **Telling a listener.** For ``run.stream()``, the scheduler tells a ``NodeEvents`` what happens: a call starts,
   an outcome is recorded, a call is retried or fails, or nothing is ready. ``NodeEvents`` makes the events from
   that (``node_events.py``).
"""

from __future__ import annotations

import heapq
import itertools
import logging
import time
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Empty, SimpleQueue
from typing import Any, Literal

from conductor._internal.ask import Ask
from conductor._internal.base import ItemIndexes, place_text
from conductor._internal.node import LONGEST_WAIT, RunningCalls
from conductor._internal.problem import (
    ExternalError,
    InvalidCallError,
    NodeError,
    Problem,
    author_code_problem,
    run_problems,
)

from .blueprint import BlueprintFunctionNode, all_upstream, outcome_loc
from .call import call, jsonable
from .node_events import NodeEvents
from .outcome import Done, NoneInput, NotReady, Outcome, Skipped, Unanswered
from .outcomes import RunOutcomes, length
from .reuse import ReusableOutcomes
from .state import EndedEarly, LastResume

#: The exceptions that are never retried, whatever the node's ``Policy.retry_on`` names. They are the node's own
#: failure and a question for a person. Calling again would only raise the same thing.
_NEVER_RETRIED = (NodeError, Ask)

#: How long the scheduler waits for a thread before it asks ``stop`` again, in seconds.
_SECONDS_BETWEEN_STOP_CHECKS = 0.05


def _each[T](items: list[T], action: Callable[[T], object]) -> None:
    """Calls ``action`` on each of ``items``, and finishes the rest even if one call raises.

    It is for clean-up. If an exception is raised on this thread partway through, a ``KeyboardInterrupt`` say,
    ``action`` is still called on the remaining items. Then the exception goes on up.
    """
    rest = iter(items)
    try:
        for item in rest:
            action(item)
    finally:
        for item in rest:
            action(item)


@dataclass(
    slots=True, eq=False
)  # compared and hashed by identity, since ``on_threads`` uses the attempt as a key
class _Attempt:
    """One attempt at one call: the first try, or a retry.

    For example, ``_Attempt(shout, (1,), {"line": "b", "mark": "!"}, 0)`` is the first attempt at ``shout (1,)``.

    Attributes:
        node: The node to call.
        item_indexes: Which call of the node it is.
        arguments: The call's arguments.
        number: 0 for the first attempt at the call, 1 for its first retry, and so on.
        future: The pool's handle on the attempt, once it is handed to a thread. It lets an attempt still in the
            pool's queue be cancelled. It is ``None`` before that, and for an attempt made on the scheduler's thread.
        deadline: The timer for its ``attempt_timeout``, for an attempt on a thread whose node has one. ``None``
            otherwise. It is taken off the timers when the attempt comes back.
        returned_at: The ``time.monotonic()`` at which its function returned or raised on its thread. The
            scheduler may see the attempt back later than that, while it makes a call of its own.
    """

    node: BlueprintFunctionNode
    item_indexes: ItemIndexes
    arguments: dict[str, Any]
    number: int
    future: Future[None] | None = None
    deadline: _Timer | None = None
    returned_at: float | None = None


@dataclass(slots=True, order=True)
class _Timer:
    """Something the scheduler does at a set time.

    Timers sort by ``when``, then by ``order``.

    Attributes:
        when: The ``time.monotonic()`` at which it is due.
        order: A counter, so two timers due at the same time go in the order they were set.
        kind: ``"retry"`` makes ``attempt`` when it is due. ``"deadline"`` means ``attempt`` is on a thread, and
            its ``attempt_timeout`` runs out when the timer is due.
        attempt: The attempt to make, or the attempt whose time runs out.
    """

    when: float
    order: int
    kind: Literal["retry", "deadline"] = field(compare=False)
    attempt: _Attempt = field(compare=False)


@dataclass(slots=True)
class _ItemOrder:
    """How far the calls of one node have been tried, in item order.

    It is kept for a node that may ask a person and reuses outcomes from an earlier run (job 3).

    Attributes:
        order: The item indexes of the node's calls, in item order.
        position: The position of each of them in ``order``.
        reached: How many calls have been tried, counting from the first in ``order``.
    """

    order: list[ItemIndexes]
    position: dict[ItemIndexes, int]
    reached: int = 0


@dataclass(slots=True)
class _Checkpoints:
    """Hands the host a copy of the run as outcomes are recorded, at most every ``every`` seconds.

    Attributes:
        save: Copies the run and hands the copy to the host. It is called with why the scheduler stopped, if it did.
        every: The fewest seconds between two checkpoints.
        next_at: The ``time.monotonic()`` from which the next recorded outcome brings a checkpoint.
    """

    save: Callable[[LastResume], None]
    every: float
    next_at: float


class Scheduler:
    """Makes the calls for one call of ``run.resume()`` or ``run.stream()``.

    ``Scheduler(...).run()`` calls every node that is ready, until nothing more can be called. The module docstring
    describes its five jobs. ``__init__`` sets up the attributes in the same groups, one group per job.
    """

    def __init__(
        self,
        *,
        outcomes: RunOutcomes,
        reusable: ReusableOutcomes,
        max_items: int | None,
        arguments_for: Callable[
            [BlueprintFunctionNode, ItemIndexes],
            dict[str, Any] | Skipped | NoneInput | Literal[NotReady.NOT_READY],
        ],
        node_events: NodeEvents | None,
        stop: Callable[[], LastResume] | None,
        context: Any,
        checkpoint: Callable[[LastResume], None] | None = None,
        checkpoint_every: float | None = None,
    ) -> None:
        """Sets up the scheduler. Nothing is called until ``run()``.

        Args:
            outcomes: The run's outcomes, with the blueprint of its nodes. The scheduler records into them.
            reusable: The outcomes of the previous run that a call may reuse. Some can only be reused once, and the
                scheduler takes those out as it reuses them.
            max_items: The item limit: the most items a series that a loop runs over may have. ``None`` for no limit.
            arguments_for: Returns the arguments of one call, or why it can't be called yet
                (``RunOutcomes.arguments_for``).
            node_events: Makes the events about each node, for the listener of ``run.stream()``. ``None`` for
                ``run.resume()``, which has no listener.
            stop: Asked before each call, and every so often while waiting for threads. It returns ``None`` to go
                on. To stop, it returns an ``EndedEarly`` (cancelled, or timed out), or ``"running"`` when the host
                takes the run back to go on with it somewhere else.
            context: The run's context, passed to each node that takes it. ``None`` when the host passed none.
            checkpoint: Hands the host a copy of the run. It is called with why the scheduler stopped, if it did.
            checkpoint_every: The fewest seconds between two checkpoints. ``None`` for no checkpoints.
        """
        #: Every function node of the graph and of its subgraphs, in topological order.
        self.nodes = list(outcomes.blueprint.function_nodes.values())
        #: How many pool threads the graph needs (``RunBlueprint.threads``).
        self.threads = outcomes.blueprint.threads
        self.outcomes, self.max_items, self.arguments_for = outcomes, max_items, arguments_for
        self.stop, self.context = stop, context

        # 1. Choosing the next call.
        #: Why ``run`` stopped before every ready call was made. It is ``None`` until ``end`` is called. Once it isn't
        #: ``None``, ``going`` is false and no more calls start.
        self.stopped: LastResume = None
        #: The calls to try, as a heap of (node position, item indexes), so the first node in topological order
        #: comes off first.
        self.ready: list[tuple[int, ItemIndexes]] = []
        #: Whether an outcome was recorded, or every call was put on ``ready`` again, since
        #: ``_when_nothing_is_ready`` last ran. When ``ready`` is empty, ``run`` only looks for more to try if this is
        #: set. If not, it waits for a thread or a timer. That way a wake-up that recorded nothing doesn't look
        #: through the collecting calls again for nothing.
        self.changed = True
        #: The collecting calls to try once no call upstream of them is running. Each is the node, and the item
        #: indexes that the calls it collects start with.
        self.collectors: set[tuple[BlueprintFunctionNode, ItemIndexes]] = set()
        #: The calls that are running, each with its current attempt. A call is running while its attempt is on a
        #: thread, waiting to start, or waiting to be retried. When an attempt comes back from its thread, or its
        #: deadline is due, it only counts if it is still the call's attempt here.
        self.running: dict[tuple[BlueprintFunctionNode, ItemIndexes], _Attempt] = {}

        # 2. Threads, retries and timeouts.
        #: The thread pool the attempts run on. It is only made when the first attempt goes to a thread.
        self.pool: ThreadPoolExecutor | None = None
        #: Each node's attempts that wait to start until fewer than ``concurrency`` calls of the node run. First in,
        #: first out.
        self.waiting_to_start: defaultdict[BlueprintFunctionNode, deque[_Attempt]] = defaultdict(
            deque
        )
        #: What the pool threads hand back. Usually it's an attempt with what ``call`` returned for it: the node's
        #: outputs, and what the function raised or the problem a validator refused the call with. It can also be
        #: an exception that ``call`` itself raised. Any thread puts ``None`` here when a call ends of a node whose
        #: attempts wait here.
        self.returned: SimpleQueue[
            tuple[_Attempt, Any, BaseException | Problem | None] | BaseException | None
        ] = SimpleQueue()
        #: Whether this run's thread is waiting on ``returned`` right now. A node's ``RunningCalls`` lets a parked
        #: run have the next turn.
        self.parked = False
        #: The attempts handed to the pool that still count among their node's running calls, each with the
        #: ``RunningCalls`` that counts it. Three places can take an attempt out: ``_on_thread``, ``_cancel_queued``,
        #: and an exception in ``_start_waiting_calls``. Whichever comes first counts it out (``_count_out``), so it
        #: is counted out only once.
        self.on_threads: dict[_Attempt, RunningCalls] = {}
        #: The running calls of each node that sets no ``concurrency``. This run counts them on its own, and lets
        #: one run at a time.
        self.counted_in_this_run: defaultdict[BlueprintFunctionNode, RunningCalls] = defaultdict(
            RunningCalls
        )
        #: The retries to make, and the deadlines of the attempts on threads. A heap, so the one due first comes
        #: off first.
        self.timers: list[_Timer] = []
        #: Gives each timer its ``order``.
        self.timer_order = itertools.count()

        # 3. Reusing in item order.
        #: The outcomes of the previous run that a call may reuse.
        self.reusable = reusable
        #: How far each node that may ask, and reuses earlier outcomes, has been tried in item order.
        self.item_orders: dict[BlueprintFunctionNode, _ItemOrder] = {}
        #: Whether a call was left untried because an earlier call of its node hasn't been tried yet.
        self.out_of_turn = False
        #: Whether an outcome was recorded since ``_put_every_call_on_ready`` last ran.
        self.recorded_since_last_pass = False

        # 4. Stopping, the item limit and checkpoints.
        #: When the host is handed a copy of the run. ``None`` when it asked for no checkpoints.
        self.checkpoints = (
            _Checkpoints(checkpoint, checkpoint_every, time.monotonic() + checkpoint_every)
            if checkpoint is not None and checkpoint_every is not None
            else None
        )

        # 5. Telling a listener.
        #: Makes the events about each node. ``None`` when nobody listens.
        self.node_events = node_events

    def run(self) -> LastResume:
        """Calls every node that is ready, until nothing more can be called or ``end`` is called.

        Nothing more can be called when no call is ready and no attempt is on a thread. After ``end``, it still
        waits for the attempts on threads, and keeps their outcomes.

        Returns:
            ``None`` if it didn't stop early. Otherwise, how the run ended early, or ``"running"`` if the host took
            the run back.
        """
        try:
            self._check_item_limit()
            self._put_every_call_on_ready()
            while True:
                # Between two calls, so a call back from its thread or a timer that fell due is seen at once. It can
                # end the run, so stop is asked after it.
                self._take_news()
                if self._stop_if_asked():
                    break
                if self.ready:
                    position, item_indexes = heapq.heappop(self.ready)
                    self._start_if_ready(self.nodes[position], item_indexes)
                elif self.changed:
                    self.changed = False
                    if not self._when_nothing_is_ready():
                        break
                else:
                    self._wait()
            # Checkpoint now, since the process may die before the attempts on threads are back.
            if self.checkpoints is not None and self.running:
                self._checkpoint(self.checkpoints)
            while self.running:  # a call that was made is never made again, so wait for its outcome
                self._wait()
        finally:
            # Attempts are only left here when run ends by an exception. Each step runs even if another one raises.
            # A node's RunningCalls should only keep this run while an attempt of the node waits to start here.
            _each(list(self.running.values()), self._cancel_queued)
            _each(self._nodes_waiting(), lambda node: self._count_of(node).stop_waiting(self))
            if self.pool is not None:
                self.pool.shutdown(wait=False, cancel_futures=True)
        if self.node_events is not None:
            self.node_events.scheduler_ended()
        return self.stopped

    def end(self, stopped: EndedEarly | Literal["running"]) -> None:
        """Stops starting calls.

        The attempts that are waiting to start, or waiting to be retried, are dropped. ``run`` still waits for the
        attempts on threads, and keeps their outcomes.

        Args:
            stopped: Why: how the run ends early, with its cause, or ``"running"`` when the host takes the run back.
        """
        self.stopped = stopped
        for node, waiting in self.waiting_to_start.items():
            # This comes first. If it raises, the attempts are still waiting, so run's finally tells the node again.
            self._count_of(node).stop_waiting(self)
            for attempt in waiting:
                self.running.pop((node, attempt.item_indexes), None)
            waiting.clear()
        for timer in self.timers:
            if timer.kind == "retry":
                self.running.pop((timer.attempt.node, timer.attempt.item_indexes), None)

    @property
    def going(self) -> bool:
        """Whether calls are still being started, which is until ``end`` is called."""
        return self.stopped is None

    # -- 1. Choosing the next call --------------------------------------------------------------------------------

    def _put_every_call_on_ready(self) -> None:
        """Puts every call that has no outcome and isn't running on ``ready``.

        ``run`` does this once at the start. It is only done again for calls that were left untried because they
        were out of turn (job 3).
        """
        self.out_of_turn, self.recorded_since_last_pass = False, False
        for node in self.nodes:
            outcomes = self.outcomes.of(node.node_path)
            for item_indexes in self.outcomes.item_indexes(node.loops)[0]:
                if item_indexes not in outcomes and (node, item_indexes) not in self.running:
                    heapq.heappush(self.ready, (node.position, item_indexes))

    def _start_if_ready(self, node: BlueprintFunctionNode, item_indexes: ItemIndexes) -> None:
        """Tries one call: gets its arguments, and if it is ready, makes the call or records why it isn't made.

        For example, ``_start_if_ready(shout, (1,))`` does nothing while ``lines`` has no outcome. Once it has one,
        it calls ``shout(line="b", mark="!")`` and records ``Done({"result": "B!"})``.

        One of four things happens:

        - Nothing, if the call already has an outcome, is running, isn't ready, or is out of turn for an answer it
          could reuse.
        - If the call is skipped or has a ``None`` input, ``arguments_for`` returns that outcome, and it is recorded.
        - If an earlier run has an outcome for these arguments, that outcome is recorded, and the node isn't called.
        - Otherwise an ``_Attempt`` is made, and the outcome is recorded once the node has returned.

        Args:
            node: The node to call.
            item_indexes: Which call of the node.
        """
        if item_indexes in self.outcomes.of(node.node_path) or (node, item_indexes) in self.running:
            return
        try:
            arguments = self.arguments_for(node, item_indexes)
        # An edge reads a field that the record it reads from doesn't have.
        except InvalidCallError as missing:
            (problem,) = missing.problems
            self._record(node, item_indexes, self._failure(node, item_indexes, problem))
            return
        if arguments is NotReady.NOT_READY:
            return
        if not isinstance(arguments, dict):  # a Skipped or NoneInput outcome
            self._record(node, item_indexes, arguments)
        elif (
            previous := self.reusable.take(
                node,
                item_indexes,
                arguments,
                self.outcomes,
                lambda: self._in_turn(node, item_indexes),
            )
        ) is not None:
            if previous is NotReady.NOT_ITS_TURN:
                self.out_of_turn = True
                return
            if self.node_events is not None:
                self.node_events.outcome_reused(node)
            self._record(node, item_indexes, previous)
        else:
            self._make_attempt(_Attempt(node, item_indexes, arguments, 0))

    def _record(
        self, node: BlueprintFunctionNode, item_indexes: ItemIndexes, outcome: Outcome | Problem
    ) -> None:
        """One call is over. Writes its outcome into the run, and does what follows from it.

        For example, ``_record(lines, (), Done({"result": ["a", "b", "c"]}))`` records the outcome of ``lines ()``,
        and puts ``shout (0,)``, ``shout (1,)`` and ``shout (2,)`` on ``ready``.

        What follows comes in this order: a checkpoint if one is due, the listener, the item limit check, and the
        downstream calls. The listener hears of every outcome written, so a reused one over the item limit still
        starts its node. A ``Problem`` instead of an outcome means the call failed. Then nothing is written, and
        the run ends as ``failed``.

        Args:
            node: The node of the call.
            item_indexes: Which call of the node.
            outcome: The call's outcome, or the problem it failed with.
        """
        # Only for speed: isinstance on a pydantic model costs four times as much, and no class derives from Problem.
        if type(outcome) is Problem:
            self._fail(node, outcome)
            return
        self.outcomes.record(node, item_indexes, outcome)
        self.recorded_since_last_pass = self.changed = True
        if (
            checkpoints := self.checkpoints
        ) is not None and time.monotonic() >= checkpoints.next_at:
            self._checkpoint(checkpoints)
        if self.node_events is not None:
            self.node_events.outcome_recorded(node, item_indexes, outcome)
        if self.max_items is not None and node.looped_outputs and outcome.has_outputs:
            if self._stop_if_over_item_limit(node, item_indexes, outcome, self.max_items):
                return
        if self.going and outcome.has_outputs:  # a call that reads a waiting outcome isn't ready
            self._make_downstream_ready(node, item_indexes)

    def _fail(self, node: BlueprintFunctionNode, cause: Problem) -> None:
        """A call of ``node`` failed. The first failure ends the run, and any later one changes nothing.

        The run usually ends as ``failed``. But if it was cancelled by then, it ends ``cancelled``. That's because a
        node can be told of a cancel: a host can put its cancel event in the context. A node that stops a slow call
        when told does so by raising, so its failure is the cancel's doing.

        A node is never told that the ``timeout`` passed, or that the run is being handed back. So a failure then is
        the node's own, and the run ends ``failed``. Either way the call has no outcome, and a resume makes it again.
        """
        if self.going:
            asked = self.stop() if self.stop is not None else None
            if isinstance(asked, EndedEarly) and asked.status == "cancelled":
                self.end(asked)
            else:
                self.end(EndedEarly(status="failed", cause=cause))
            if self.node_events is not None:
                self.node_events.call_failed(node, cause)

    def _make_downstream_ready(
        self, node: BlueprintFunctionNode, item_indexes: ItemIndexes
    ) -> None:
        """Puts the calls that read the outcome of ``node`` for ``item_indexes`` on ``ready``, or on ``collectors``.

        Some examples. After ``shout (1,)``, the downstream node ``join`` collects every call of ``shout``, so
        ``(join, ())`` goes on ``collectors``. After ``lines ()``, ``shout`` is one loop deeper, so every call of
        ``shout`` goes on ``ready``. A downstream node inside the same loops has one call with the same item
        indexes, and that call goes on ``ready``.
        """
        depth = len(item_indexes)
        for downstream, shared_indexes in node.downstream:
            if shared_indexes < depth:
                self.collectors.add((downstream, item_indexes[:shared_indexes]))
            else:
                for downstream_indexes in self.outcomes.item_indexes(
                    downstream.loops, item_indexes
                )[0]:
                    heapq.heappush(self.ready, (downstream.position, downstream_indexes))

    def _when_nothing_is_ready(self) -> bool:
        """Called when ``ready`` is empty. Finds more to try, and says whether ``run`` goes on.

        It goes through these steps in order:

        1. Put on ``ready`` the collecting calls that no running call is upstream of. If there is one, go on.
        2. Tell the listener, which works out which nodes are complete.
        3. If a call is running, go on. ``run`` waits for it.
        4. If a call was left untried because it was out of turn (job 3), put every call on ``ready`` again and go
           on. This is only done if an outcome was recorded since the last time. Otherwise trying again would find
           the same.

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
        if self.out_of_turn and self.recorded_since_last_pass:
            self._put_every_call_on_ready()
            self.changed = True
            return True
        return False

    def _release_collectors(self) -> bool:
        """Puts on ``ready`` the collecting calls that no running call is upstream of.

        For example, ``(join, ())`` is in ``collectors`` after the first ``shout`` is recorded. It stays there while
        another call of ``shout`` is running. Once none is, ``join ()`` goes on ``ready``.

        Returns:
            Whether any call was put on ``ready``.
        """
        running = list(self.running)
        released = False
        for node, item_indexes in list(self.collectors):
            upstream, depth = all_upstream(node), len(item_indexes)
            if not any(
                running_node in upstream and running_indexes[:depth] == item_indexes
                for running_node, running_indexes in running
            ):
                self.collectors.discard((node, item_indexes))
                released = True
                for call_indexes in self.outcomes.item_indexes(node.loops, item_indexes)[0]:
                    heapq.heappush(self.ready, (node.position, call_indexes))
        return released

    # -- 2. Threads, retries and timeouts -------------------------------------------------------------------------

    def _make_attempt(self, attempt: _Attempt) -> None:
        """Makes one attempt at a call.

        The attempt joins the node's attempts that wait to start. It starts at once if fewer than ``most_at_once``
        calls of the node are running.
        """
        self.running[(attempt.node, attempt.item_indexes)] = attempt
        self.waiting_to_start[attempt.node].append(attempt)
        self._start_waiting_calls(attempt.node)

    def _start_waiting_calls(self, node: BlueprintFunctionNode) -> None:
        """Starts the attempts of ``node`` that are waiting to start, while fewer than ``most_at_once`` calls of the
        node are running. First in, first out.

        ``stop`` is asked before each attempt, just as before each call, and the run ends if it says to stop.

        An attempt of a node that is called on threads goes to a pool thread. If the node has an
        ``attempt_timeout``, the attempt gets its deadline timer here. Any other attempt is made right here, on the
        scheduler's thread, and its outcome is recorded unless it is retried. Only one is made per call of this
        method: if more wait, the run wakes itself, so the run loop takes in what came back and fell due before it
        makes the next. If other runs are waiting for the
        node, the ones that were parked (waiting for news) get to ask before this run asks again. While no attempt
        can start, the node's ``RunningCalls`` keeps hold of this run, so it can ``wake`` it.

        Each attempt that ``try_start`` counted in is counted out exactly once, however it ends:

        - here, when the function returns or raises;
        - on its pool thread, by ``_on_thread``;
        - or by whoever cancels it (``_cancel_queued``), if it is still queued for a thread when its deadline passes
          or ``run`` ends.
        """
        waiting, policy, running_calls = (
            self.waiting_to_start[node],
            node.policy,
            self._count_of(node),
        )
        while (
            waiting
            and not self._stop_if_asked()
            and running_calls.try_start(policy.most_at_once, self)
        ):
            counted_here: RunningCalls | None = (
                running_calls  # who counts the attempt out if this raises: here, or ``on_threads``
            )
            try:  # nothing is called between try_start and here, so nothing can raise before this try
                attempt = waiting.popleft()
                if self.node_events is not None:
                    self.node_events.call_starts(node)
                if policy.threaded:
                    if self.pool is None:
                        self.pool = ThreadPoolExecutor(self.threads, "conductor")
                    self.on_threads[attempt] = running_calls
                    counted_here = None
                    attempt.future = self.pool.submit(self._on_thread, attempt)
                else:
                    outputs, error = call(node, attempt.arguments, self.context)
            except BaseException:
                if counted_here is not None:
                    counted_here.finish()
                else:
                    self._count_out(attempt)
                raise
            if policy.threaded:
                if policy.attempt_timeout is not None:
                    attempt.deadline = self._set_timer(policy.attempt_timeout, "deadline", attempt)
                continue
            running_calls.finish(give_parked_runs_a_turn=True)
            self._attempt_over(attempt, outputs, error)
            # An exception kept in a local of this frame makes a reference cycle through its traceback. That keeps
            # the call's frames and their values alive until the garbage collector runs.
            del error
            if waiting:
                # One call on this thread at a time: the run loop takes in what came meanwhile, then this wake
                # brings it back for the next.
                self.wake()
                return

    def _cancel_queued(self, attempt: _Attempt) -> None:
        """Takes ``attempt`` off the pool's queue if no thread has started it yet, and counts it out, since it will
        never run."""
        if attempt.future is not None and attempt.future.cancel():
            self._count_out(attempt)

    def _count_out(self, attempt: _Attempt) -> None:
        """Counts ``attempt`` out of its node's running calls, unless that was done already.

        It is for an attempt handed to the pool that is over, or will never run.
        """
        if (running_calls := self.on_threads.pop(attempt, None)) is not None:
            running_calls.finish()

    def _stop_if_asked(self) -> bool:
        """Asks ``stop`` whether to stop starting calls, and if it says so, ends the run here.

        Once a stop is seen, it holds. ``cancel`` and ``hand_back`` are the host's events, and the host may clear
        them again. If the stop were taken back, an attempt that was left waiting to start would wait forever.

        ``stop`` isn't asked once the run has ended, so the first reason to stop is the one that counts. A cancel
        set after a call failed, or after a series went over the item limit, leaves the run ``failed`` or
        ``stopped``. But a call that fails while a cancel is set ends the run ``cancelled`` (see ``_fail``).

        Returns:
            Whether the run has ended.
        """
        if self.going and self.stop is not None and (stopped := self.stop()) is not None:
            self.end(stopped)
        return not self.going

    def wake(self) -> None:
        """Puts ``None`` on ``returned``, to tell this run that a call ended of a node whose attempts wait here.

        The call may be in this run or another. The node's ``RunningCalls`` calls this, from any thread.
        """
        self.returned.put(None)

    def _nodes_waiting(self) -> list[BlueprintFunctionNode]:
        """Returns the nodes that have attempts waiting to start."""
        return [node for node, waiting in self.waiting_to_start.items() if waiting]

    def _start_every_waiting_call(self) -> None:
        """A call ended of a node whose attempts wait here, in this run or another. Starts what can start now."""
        for node in self._nodes_waiting():
            self._start_waiting_calls(node)

    def _count_of(self, node: BlueprintFunctionNode) -> RunningCalls:
        """Returns what counts the running calls of ``node``.

        If the node's policy sets ``concurrency``, that's its definition's ``RunningCalls``, shared by every run in
        the process. Otherwise it is this run's own count, for this place in the graph.
        """
        if node.policy.concurrency is not None:
            return node.compiled_node.definition.running_calls
        return self.counted_in_this_run[node]

    def _attempt_over(
        self, attempt: _Attempt, outputs: Any, error: BaseException | Problem | None
    ) -> None:
        """``attempt`` is over. Records its outcome, unless it is retried.

        The attempt may have run on the scheduler's thread or come back from its own. It is still the call's attempt
        in ``running``.
        """
        node, item_indexes = attempt.node, attempt.item_indexes
        if not self._retry(attempt, error):
            del self.running[(node, item_indexes)]
            self._record(
                node,
                item_indexes,
                self._outcome(node, item_indexes, attempt.arguments, outputs, error),
            )

    def _on_thread(self, attempt: _Attempt) -> None:
        """Runs on a pool thread. Calls the node, counts the call out, and puts what came back on ``returned``.

        ``call`` returns a node's own exception as ``error``. But ``call`` itself can raise, with a ``SystemExit``
        say, or a bug in a type's own validation. That exception is put on ``returned`` as it is, and ``_wait``
        raises it on the scheduler's thread, just as it would be raised for a node called there.

        The call is counted out before anything is put on ``returned``. So when the scheduler reads what came back,
        the node's next waiting attempt can start.
        """
        returned: tuple[_Attempt, Any, BaseException | Problem | None] | BaseException
        try:
            outputs, error = call(attempt.node, attempt.arguments, self.context)
        except BaseException as raised:  # noqa: BLE001  the pool would swallow it, and the scheduler would wait for the attempt forever
            returned = raised
        else:
            returned = (attempt, outputs, error)
        finally:
            attempt.returned_at = time.monotonic()
            self._count_out(attempt)
        self.returned.put(returned)

    def _wait(self) -> None:
        """Waits until an attempt is back from its thread, a timer is due, or it is time to ask ``stop`` again.

        Then it takes in everything that came meanwhile (``_take_news``).
        """
        timeout = self.timers[0].when - time.monotonic() if self.timers else None
        if self.stop is not None:
            timeout = (
                _SECONDS_BETWEEN_STOP_CHECKS
                if timeout is None
                else min(timeout, _SECONDS_BETWEEN_STOP_CHECKS)
            )
        try:
            self.parked = True
            try:
                if timeout is None:
                    returned = self.returned.get()
                else:
                    try:
                        # A CPython 3.12 bug (every release; 3.13 rewrote the method in python/cpython#114161):
                        # get(timeout=) on an empty SimpleQueue that is new, or has just given an item, can wait
                        # forever. If the timeout runs out during its first look, its next look waits with no limit.
                        # So the first look is made here without waiting, and get(timeout=) after it returns in time.
                        returned = self.returned.get(block=False)
                    except Empty:
                        if timeout <= 0:
                            raise
                        returned = self.returned.get(timeout=timeout)
            finally:
                self.parked = False
        except Empty:
            pass
        else:
            self._returned(returned)
        self._take_news()

    def _take_news(self) -> None:
        """Takes in, without waiting, every attempt back from its thread, then every timer that is due.

        The attempts come first. One that came back before its deadline is then out of ``running``, so its deadline
        timer finds nothing to time out, however late the scheduler sees it. One that came back after its deadline
        is timed out by ``_call_returned``.
        """
        while not self.returned.empty():
            self._returned(self.returned.get(block=False))
        now = time.monotonic()
        while self.timers and self.timers[0].when <= now:
            self._due(heapq.heappop(self.timers))

    def _returned(
        self, returned: tuple[_Attempt, Any, BaseException | Problem | None] | BaseException | None
    ) -> None:
        """Handles one thing from ``returned``: ``None`` for a call that ended of a node whose attempts wait here,
        an attempt back from its thread, or what ``call`` raised on a thread, which is raised here."""
        if returned is None:
            self._start_every_waiting_call()
            return
        for node in (
            self._nodes_waiting()
        ):  # busy with other news, so a run that woke it doesn't wait for its turn
            self._count_of(node).turned_away(self)
        if isinstance(returned, BaseException):
            raise returned
        self._call_returned(returned)

    def _call_returned(
        self, returned: tuple[_Attempt, Any, BaseException | Problem | None]
    ) -> None:
        """An attempt is back from its thread. Starts the node's next waiting attempt, then retries this one or
        records its outcome.

        Its deadline timer, if it has one, is taken off the timers, so the timer doesn't keep the attempt and its
        arguments until it falls due. An attempt that came back after its deadline, while the scheduler was busy
        with a call of its own, is timed out as its deadline timer would have done in time, and its outcome is
        dropped. Otherwise its outcome is dropped if its deadline has passed, since the run has already failed with
        ``timeout`` for it. It is also dropped if ``end`` dropped the call. There's one exception: the deadline has passed, the function returned without an error, and the run is
        ending. The call was made, so its outcome is recorded, and a later ``resume()`` doesn't make it again.
        """
        attempt, outputs, error = returned
        node, item_indexes = attempt.node, attempt.item_indexes
        if (deadline := attempt.deadline) is not None:
            self._drop_timer(deadline)
            if (
                attempt.returned_at is not None
                and attempt.returned_at > deadline.when
                and self.running.get((node, item_indexes)) is attempt
            ):
                self._time_out(attempt)
                return
        self._start_waiting_calls(node)
        if (
            self.running.get((node, item_indexes)) is not attempt
        ):  # only a deadline that passed takes an attempt on a thread out of ``running``
            if (
                error is None
                and not self.going
                and item_indexes not in self.outcomes.of(node.node_path)
            ):
                self._record(
                    node,
                    item_indexes,
                    self._outcome(node, item_indexes, attempt.arguments, outputs, None),
                )
            return
        self._attempt_over(attempt, outputs, error)

    def _retry(self, attempt: _Attempt, error: BaseException | Problem | None) -> bool:
        """Sets a timer to make the call again, if ``attempt`` failed and its node's ``Policy`` retries the error.

        The rules for which errors are retried are in the reference: ``python -m conductor.about
        retries-and-timeouts``. The delay is ``retry_delay`` seconds, doubled for each attempt before this one, up
        to ``LONGEST_WAIT``.

        Returns:
            Whether the call will be made again. If not, the caller records the outcome of ``attempt``.
        """
        node, item_indexes = attempt.node, attempt.item_indexes
        policy = node.policy
        if (
            error is None
            or isinstance(error, Problem)
            or attempt.number >= policy.retries
            or not self.going
        ):
            return False
        if not isinstance(error, ExternalError) and (
            isinstance(error, _NEVER_RETRIED) or not isinstance(error, policy.retry_on)
        ):
            return False
        # 2.0 ** 1024 is past what a float holds. A product past it is inf, which the cap brings down.
        delay = min(policy.retry_delay * 2.0 ** min(attempt.number, 1023), LONGEST_WAIT)
        retry = _Attempt(node, item_indexes, attempt.arguments, attempt.number + 1)
        self.running[(node, item_indexes)] = retry
        self._set_timer(delay, "retry", retry)
        # Made with or without a listener, since making it logs the error.
        failure = self._failure(node, item_indexes, error, retry=(retry, delay))
        if self.node_events is not None:
            self.node_events.call_retries(node, attempt.number + 1, delay, failure)
        return True

    def _set_timer(
        self, seconds: float, kind: Literal["retry", "deadline"], attempt: _Attempt
    ) -> _Timer:
        """Sets a timer that is due ``seconds`` from now, and returns it.

        A ``"retry"`` timer makes ``attempt`` when it is due. A ``"deadline"`` timer is for the ``attempt_timeout``
        of ``attempt``, which starts on a thread now.
        """
        timer = _Timer(time.monotonic() + seconds, next(self.timer_order), kind, attempt)
        heapq.heappush(self.timers, timer)
        return timer

    def _drop_timer(self, timer: _Timer) -> None:
        """Takes ``timer`` off the timers, if it is still there. Timers sort by ``when`` and ``order``, and no two
        have the same ``order``, so the one taken off is this one."""
        if timer in self.timers:
            self.timers.remove(timer)
            heapq.heapify(self.timers)

    def _due(self, timer: _Timer) -> None:
        """Handles a timer that is due.

        For a retry, it makes the attempt, unless ``end`` was called. For a deadline, if the attempt is still on its
        thread, the call fails with the problem ``timeout``, which ends the run. An attempt whose function returned
        by then is left alone: it is on its way back, and ``_call_returned`` takes it.
        """
        attempt = timer.attempt
        if timer.kind == "retry":
            if self.going:
                self._make_attempt(attempt)
        elif attempt.returned_at is None or attempt.returned_at > timer.when:
            self._time_out(attempt)

    def _time_out(self, attempt: _Attempt) -> None:
        """The ``attempt_timeout`` of ``attempt`` has run out. If the attempt is still the call's, the call fails
        with the problem ``timeout``, which ends the run ``failed``.

        It does so even when a cancel is asked, or ended the run already. A failure after a cancel is usually the
        cancel's doing (see ``_fail``), but a timeout never is: the attempt's thread is abandoned, not stopped.
        """
        node, item_indexes = attempt.node, attempt.item_indexes
        if self.running.get((node, item_indexes)) is not attempt:
            return
        del self.running[(node, item_indexes)]
        self._cancel_queued(attempt)
        seconds = node.policy.attempt_timeout
        assert seconds is not None, (
            "only an attempt whose node has an attempt_timeout has a deadline"
        )
        cause = run_problems.timeout(*outcome_loc(node, item_indexes), seconds=seconds)
        if self.going:
            self.end(EndedEarly(status="failed", cause=cause))
        elif isinstance(self.stopped, EndedEarly) and self.stopped.status == "cancelled":
            self.stopped = EndedEarly(status="failed", cause=cause)
        else:  # the run ended for another reason first, and that stands
            return
        if self.node_events is not None:
            self.node_events.call_failed(node, cause)

    def _outcome(
        self,
        node: BlueprintFunctionNode,
        item_indexes: ItemIndexes,
        arguments: dict[str, Any],
        outputs: Any,
        error: BaseException | Problem | None,
    ) -> Outcome | Problem:
        """Returns a call's outcome, made from what ``call`` returned.

        That's ``Done`` with the outputs, or ``Unanswered`` with what the ``Ask`` said if the function raised one,
        or else the problem the call failed with.

        A person who is asked is shown the call's arguments, as JSON. So arguments that can't be turned into JSON
        are a problem. So is an ``Ask`` that doesn't fit the node's outputs.
        """
        if error is None:
            return Done(outputs)
        if isinstance(error, Ask):
            try:
                # a person is shown the call's arguments with the question (run.waiting)
                jsonable(arguments)
                choices, defaults = node.compiled_node.ask_validator(error)
            except InvalidCallError as refused:  # arguments with no JSON form, or the Ask validator
                (problem,) = refused.problems
                return self._failure(node, item_indexes, problem)
            return Unanswered(error.prompt, choices, defaults)
        return self._failure(node, item_indexes, error)

    def _failure(
        self,
        node: BlueprintFunctionNode,
        item_indexes: ItemIndexes,
        error: BaseException | Problem,
        retry: tuple[_Attempt, float] | None = None,
    ) -> Problem:
        """Returns the problem a call failed with, with the call's loc.

        A validator's problem is kept as it is. An exception is reported by ``author_code_problem``: the problem a
        ``NodeError`` carries is kept, and any other exception is written to the log, with its traceback. That
        includes an ``InvalidCallError`` the function let escape. The problem then only names the exception's class.
        It is ``external_failed`` for a class in the node's ``retry_on``, and ``execution_failed`` for anything else.

        Args:
            node: The node whose call failed.
            item_indexes: The call's item indexes.
            error: What the call raised, or the problem it failed with.
            retry: For an attempt that is retried, the attempt made next and the seconds until it is due. The
                exception is then logged as a warning that counts the attempts, since the call may still succeed;
                else as an error. The warning says only what is known when it is written: the run may end before
                the next attempt is due.
        """
        loc = outcome_loc(node, item_indexes)
        if isinstance(error, Problem):
            return error.model_copy(update={"loc": loc})
        kind = (
            run_problems.external_failed
            if isinstance(error, node.policy.retry_on)
            else run_problems.execution_failed
        )
        where = place_text(node.node_path)
        if retry is None:
            problem = author_code_problem(
                error, "node %r failed for item %r", where, item_indexes, raised=kind
            )
        else:
            next_attempt, due_in = retry
            problem = author_code_problem(
                error,
                "node %r failed for item %r on attempt %d of %d; the next attempt is due in %s seconds",
                where,
                item_indexes,
                next_attempt.number,
                node.policy.retries + 1,
                due_in,
                raised=kind,
                log_level=logging.WARNING,
            )
        return problem.model_copy(update={"loc": loc})

    # -- 3. Reusing in item order ---------------------------------------------------------------------------------

    def _in_turn(self, node: BlueprintFunctionNode, item_indexes: ItemIndexes) -> bool:
        """Whether every call of ``node`` that comes before ``item_indexes`` in item order has been tried.

        Say the earlier run has one answer for ``approve(line="a")``, and this run has the line ``"a"`` at ``(0,)``
        and at ``(2,)``. The answer belongs to ``approve (0,)``. If ``approve (2,)`` is tried first, because its
        upstream call came back from a thread first, it isn't in turn, and takes nothing.

        A call counts as tried when it has an outcome or is running. A call that isn't ready, because an upstream
        call waits for a person, isn't tried in this ``resume()``. So no call after it is in turn, and the run waits
        for that person too.

        No call is in turn either while a series the node loops over is still to come, since an earlier call may be
        inside it. With two pages, say, page 1's lines wait until page 0's lines are known.
        """
        turn = self.item_orders.get(node)
        if turn is None:
            order, every_call_known = self.outcomes.item_indexes(node.loops)
            if not every_call_known:
                return False
            turn = self.item_orders[node] = _ItemOrder(
                order, {indexes: position for position, indexes in enumerate(order)}
            )
        order, outcomes, reached = turn.order, self.outcomes.of(node.node_path), turn.reached
        while reached < len(order):
            next_indexes = order[reached]
            if not (next_indexes in outcomes or (node, next_indexes) in self.running):
                break
            reached += 1
        turn.reached = reached
        return turn.position[item_indexes] <= reached

    # -- 4. Stopping, the item limit and checkpoints --------------------------------------------------------------

    def _check_item_limit(self) -> None:
        """At the start, ends the run as ``stopped`` if a series that a loop runs over already has too many items.

        Too many is more than ``max_items``. So a run that stopped at the limit, and is resumed with the same
        limit, stops again here.
        """
        item_limit = self.max_items
        if item_limit is None:
            return
        for node in self.nodes:
            if node.looped_outputs:
                for item_indexes, outcome in self.outcomes.of(node.node_path).items():
                    if outcome.has_outputs and self._stop_if_over_item_limit(
                        node, item_indexes, outcome, item_limit
                    ):
                        return

    def _stop_if_over_item_limit(
        self,
        node: BlueprintFunctionNode,
        item_indexes: ItemIndexes,
        outcome: Done | Skipped | NoneInput,
        item_limit: int,
    ) -> bool:
        """Ends the run as ``stopped`` if a series in ``outcome`` that a loop runs over has too many items.

        Too many is more than ``item_limit``. The run ends with the problem ``item_limit``, unless ``end`` was
        called before. In that case the first cause stays.

        The problem names the series the way the graph that has the loop names it. ``node.looped_outputs[output]``
        gives that name. It's usually the node and output themselves. But if the loop is outside a subgraph the
        node is inside, it's the subgraph node and its output. The problem's loc is the call that returned the
        series, the loc a failure of that call would have.

        Returns:
            Whether a series has too many items.
        """
        for output, series in node.looped_outputs.items():
            items = length(outcome.outputs[output])
            if items > item_limit:
                if self.going:
                    loc = outcome_loc(node, item_indexes)
                    self.end(
                        EndedEarly(
                            status="stopped",
                            cause=run_problems.item_limit(
                                *loc,
                                series=place_text((series.node,), series.output),
                                items=items,
                                max_items=item_limit,
                            ),
                        )
                    )
                return True
        return False

    def _checkpoint(self, checkpoints: _Checkpoints) -> None:
        """Hands the host a copy of the run now, and sets when the next is due."""
        checkpoints.next_at = time.monotonic() + checkpoints.every
        checkpoints.save(self.stopped)
