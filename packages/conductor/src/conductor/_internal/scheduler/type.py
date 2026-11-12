"""One leg of a run: start every unit whose inputs exist, record what comes of each, and say so as events.

A leg walks every function of the run's graph and of the graphs embedded in
it, each where it stands (``graph._flat_function_nodes``). A node inside a subgraph is
one of them, with a unit per item of the subgraph node and of its own scope,
so it is called, retried, timed out and cancelled as the same node drawn in
the graph around it. A subgraph's node is not among them: it is never
called and holds no unit, and a listener hears that it started and is
complete from the nodes inside it.

The leg first walks the nodes in order and tries every unit that can exist,
as a later leg would. A node whose policy has ``concurrency`` 1 and no
timeout is called on the spot; any other is handed to threads, at most
``concurrency`` of its units at once, and the walk goes on. When a call comes
back, the units that read it are tried at once: the reader's units under,
at, or above the unit's instance. So item 1 can finish the chain while item
10 is still being made. What reads a unit is every node whose units are made
from it (a node's ``upstream``, which is what its inputs' plans read): a node
that gathers reads the Series as well as what fills its items, so a Series
of no items makes its gather ready as the last item of a long one does.

A failure the node's ``Policy`` may retry waits its delay on a timer, and the
leg goes on meanwhile; so does an attempt with a timeout, whose clock starts
with its thread. The first failure that is final stops the leg from starting
units, and the calls already on threads are waited for and kept, so a resume
never makes them twice.

A unit that gathers (a ``Series`` or ``Table`` input fed one value per item)
is tried only once nothing it waits on is still working under its instance,
so it is tried once and not once per item it reads. When nothing is working,
the leg walks every node once more if a unit was left waiting on another and
one has been recorded since the last walk, and ends when nothing more can start.
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

from conductor._internal.address import ItemIndexes, NodePath
from conductor._internal.ask import Ask, InvalidAskError, question_asked
from conductor._internal.compiled import FlatFunctionNode, all_upstream, outcome_loc
from conductor._internal.outcomes import (
    Done,
    NotCalled,
    NotReady,
    Outcome,
    Skipped,
    Waiting,
    all_item_indexes,
    has_every_outcome,
    length,
)
from conductor._internal.problem import CallRejectedError, ExternalError, NodeError, Problem
from conductor._internal.problem import catalogue as problem
from conductor._internal.run.reuse import ReusableOutcomes, reused_outcome
from conductor._internal.run.state import COMPLETE, RUNNING, WAITING, EndedEarly, RunState
from conductor._internal.scheduler.call import call, jsonable
from conductor._internal.scheduler.events import EventBase
from conductor._internal.scheduler.node_events import NodeEvents

#: What a call can raise that is never tried again, whatever a ``Policy``'s ``retry_on`` names: the node's own failure,
#: an output its type refused, a question for a person. Repeating the call repeats it.
_NEVER_RETRIED = (NodeError, CallRejectedError, Ask)

_logger = logging.getLogger("conductor")


@dataclass(slots=True)
class _AnswerOrder:
    """How far a leg has tried a node's units in order, so a person's answers are taken in order.

    Attributes:
        order: The node's instances, in order.
        place: Each instance's position in ``order``.
        reached: How many instances from the first have been tried.
        passed: Instances counted as tried because nothing in the leg could start them (they wait on a person).
    """

    order: list[ItemIndexes]
    place: dict[ItemIndexes, int]
    reached: int = 0
    passed: set[ItemIndexes] = field(default_factory=set)


@dataclass(slots=True)
class _Attempt:
    """One try of one call that runs on a thread or may be retried.

    Attributes:
        node: The node that is called.
        item_indexes: Which call of the node it is.
        arguments: The arguments of the call.
        number: 0 for the first try of the call, 1 for its first retry.
    """

    node: FlatFunctionNode
    item_indexes: ItemIndexes
    arguments: dict[str, Any]
    number: int


@dataclass(slots=True)
class _Returned:
    """What a thread hands back when a node's function has returned or raised.

    Attributes:
        attempt: The try that was made.
        outputs: The node's outputs by name, or None when ``error`` is set.
        error: What the function raised, or None.
    """

    attempt: _Attempt
    outputs: Any
    error: BaseException | None


@dataclass(slots=True, order=True)
class _Timer:
    """Something the scheduler does at a set time. Timers sort by ``when``, then by ``order``.

    Attributes:
        when: The ``time.monotonic()`` at which it is due.
        order: A counter, so two timers due at the same time are taken in the order they were set.
        kind: ``"retry"``: make ``attempt``. ``"deadline"``: ``attempt`` is on a thread, and its ``attempt_timeout`` runs out.
        attempt: The try to make, or the try that times out.
        seconds: How long after it was set it is due.
    """

    when: float
    order: int
    kind: Literal["retry", "deadline"] = field(compare=False)
    attempt: _Attempt = field(compare=False)
    seconds: float = field(compare=False)


class Scheduler:
    """One leg over the units of a run, which it records into.

    The leg walks every function of the run's graph and of the graphs embedded in it, where they stand: it holds
    which of their units can start, the threads, each node's slots on them, the timers, ``stop``, the listener
    and the checkpoint clock. Only node functions run on threads; the run is changed on the leg's own thread.

    ``emit`` is None when nobody listens. ``stop`` is asked between units, and says how the run stands when the
    leg must end now: ended early (a cancel, the deadline), or still ``running`` for a leg its host hands back;
    else None.
    """

    def __init__(
        self, *, nodes: list[FlatFunctionNode], threads: int, outcomes: dict[str, dict[ItemIndexes, Outcome]], reusable: dict[NodePath, ReusableOutcomes], finished: frozenset[NodePath],
        max_items: int | None, arguments_for: Callable[[FlatFunctionNode, ItemIndexes], dict[str, Any] | Skipped | NotCalled | Literal[NotReady.NOT_READY]],
        emit: Callable[[EventBase], None] | None, stop: Callable[[], RunState | None] | None, context: dict[type, Any],
        checkpoint: Callable[[RunState], None] | None = None, checkpoint_every: float | None = None,
    ) -> None:
        self.nodes, self.threads, self.all_outcomes, self.max_items, self.arguments_for = nodes, threads, outcomes, max_items, arguments_for
        self.stop, self.context, self.checkpoint = stop, context, checkpoint
        #: What tells the listener about each node; None when nobody listens.
        self.node_events = NodeEvents(emit, nodes, outcomes, finished) if emit is not None else None
        #: How the run stands for this leg: running, until the leg ends it early or closes.
        self.state: RunState = RUNNING
        #: The nodes every unit of which is finished: those the run came with, then as the leg leaves them.
        self.finished = finished
        #: Each node's units by instance: the run's own, recorded into.
        self.outcomes = {node: outcomes[node.name] for node in self.nodes}
        #: The nodes whose units were recorded out of instance order, as calls on threads finish when they finish.
        self.out_of_order: set[FlatFunctionNode] = set()
        #: What each node may take from the run this one went on from, by its path.
        self.reusable = reusable
        #: Whether nothing has ended the leg early; ``end`` is the one place that does.
        self.going = True
        self.checkpoint_every = checkpoint_every if checkpoint is not None else None
        self.next_checkpoint = time.monotonic() + checkpoint_every if checkpoint_every is not None else 0.0
        #: (node position, instance): the units to try, in that order.
        self.ready: list[tuple[int, ItemIndexes]] = []
        #: Whether a walk ended or a unit was recorded outside one since the leg last looked over the run.
        self.changed = True
        #: (node, an instance above its own): a node that gathers, started in the instances under that one once
        #: nothing it reads still works there.
        self.collectors: set[tuple[FlatFunctionNode, ItemIndexes]] = set()
        self.running: dict[tuple[FlatFunctionNode, ItemIndexes], dict[str, Any]] = {}  # units handed to threads, and their inputs
        self.attempts: dict[tuple[FlatFunctionNode, ItemIndexes], int] = {}  # the attempt each unit on a thread is on
        #: Each node's tries waiting for a thread, first come first.
        self.waiting_for_thread: defaultdict[FlatFunctionNode, deque[_Attempt]] = defaultdict(deque)
        #: How many of each node's calls are on threads, which its ``concurrency`` bounds.
        self.on_threads: defaultdict[FlatFunctionNode, int] = defaultdict(int)
        #: What came back from threads: a try with its outputs or its error, or what ``call`` itself raised.
        self.returned: SimpleQueue[_Returned | BaseException] = SimpleQueue()
        #: The retries to make and the deadlines of the tries on threads, as a heap: the first is due first.
        self.timers: list[_Timer] = []
        self.timer_order = itertools.count()
        self.pool: ThreadPoolExecutor | None = None
        #: For a node taking a person's earlier answers, how far through its units in order the leg has tried.
        self.answer_orders: dict[FlatFunctionNode, _AnswerOrder] = {}
        #: Units recorded since the last walk over every unit ended.
        self.recorded_since_last_pass = 0
        #: Whether a unit was left untried because a unit before it may take the earlier answer it would take.
        self.out_of_turn = False

    def run(self) -> tuple[RunState, frozenset[NodePath]]:
        """Run the leg: try what is ready, look over the run when something happened in it, and wait for the
        threads, until nothing more can start or something ends the leg early.

        Returns:
            How the run stands, and the nodes every unit of which is finished.
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
            if self.checkpoint_every is not None and self.running:  # stopped with calls on threads: the drain may outlast the process
                self._checkpoint()
            while self.running:  # calls already made had their effect: keep them, as long as their timeouts allow
                self._wait()
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=False, cancel_futures=True)
        self._close()
        return self.state, self.finished

    def end(self, state: RunState) -> None:
        """End the leg early: nothing more starts. The units waiting for a thread and the retries not yet due are
        dropped; the calls on threads are waited for and kept.

        Args:
            state: How the run stands for it: ended early with its cause, or ``running`` for a leg handed back.
        """
        self.state, self.going = state, False
        for node, waiting in self.waiting_for_thread.items():
            for attempt in waiting:
                self.running.pop((node, attempt.item_indexes), None)
            waiting.clear()
        for timer in self.timers:
            if timer.kind == "retry":
                self.running.pop((timer.attempt.node, timer.attempt.item_indexes), None)

    def _wait(self) -> None:
        """Wait for a thread to come back, a timer to fire, or the next look at ``stop``."""
        timeout = self.timers[0].when - time.monotonic() if self.timers else None
        if self.stop is not None:
            timeout = 0.05 if timeout is None else min(timeout, 0.05)
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
            self._returned(returned)
        except Empty:
            pass
        now = time.monotonic()
        while self.timers and self.timers[0].when <= now:
            self._due(heapq.heappop(self.timers))

    def _after(self, seconds: float, kind: Literal["retry", "deadline"], attempt: _Attempt) -> None:
        """Set a timer ``seconds`` from now: the retry ``attempt``, or the deadline of ``attempt``, which is on a thread."""
        heapq.heappush(self.timers, _Timer(time.monotonic() + seconds, next(self.timer_order), kind, attempt, seconds))

    def _start_waiting_calls(self, node: FlatFunctionNode) -> None:
        """Put the node's waiting units on threads, as far as its ``concurrency`` allows."""
        waiting, policy = self.waiting_for_thread[node], node.policy
        while waiting and self.on_threads[node] < policy.concurrency and self.going:
            attempt = waiting.popleft()
            self.on_threads[node] += 1
            self.attempts[(node, attempt.item_indexes)] = attempt.number
            if self.pool is None:
                self.pool = ThreadPoolExecutor(self.threads, "conductor")
            if policy.attempt_timeout is not None:
                self._after(policy.attempt_timeout, "deadline", attempt)
            self.pool.submit(self._on_thread, attempt)

    def _checkpoint(self) -> None:
        """Hand the listener a copy of the run as it stands, and set when the next is due."""
        assert self.checkpoint is not None and self.checkpoint_every is not None
        self.next_checkpoint = time.monotonic() + self.checkpoint_every
        self._put_in_order()
        self.checkpoint(self.state)

    def _put_in_order(self) -> None:
        """Put the units of each node recorded out of instance order back in it, so the run holds every node's
        units in instance order whenever it leaves the leg: at its end, and as a checkpoint's copy."""
        for node in self.out_of_order:
            outcomes = self.outcomes[node]
            in_order = sorted(outcomes.items())
            outcomes.clear()
            outcomes.update(in_order)
        self.out_of_order.clear()

    def _when_nothing_is_ready(self) -> bool:
        """With nothing ready to try: start the gathering units nothing works under any more and say which nodes
        are complete. Once nothing is working, walk every unit again if one was left out of its turn.

        A unit is made ready by the units it is made from (a node's ``upstream``), so nothing else needs a walk
        after the first. A unit that takes a person's earlier answer waits for the units of its node before it,
        which no ``upstream`` holds: it is tried again by a walk, once a unit was recorded since the last one or
        the units that can't be tried in this leg are counted as tried.

        Returns:
            False once nothing more can start in this leg.
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

    def _try_every_call(self) -> None:
        """Try every unit of every node, in order: as a leg begins, and again for units left out of their turn."""
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
        """Start ``node`` on ``item_indexes`` if what it reads exists."""
        if not trying_every_call and (item_indexes in self.outcomes[node] or (node, item_indexes) in self.running):
            return
        try:
            arguments = self.arguments_for(node, item_indexes)
        except CallRejectedError as missing:  # an edge names a field the record it brings doesn't have
            self._record(node, item_indexes, self._failure(node, item_indexes, missing), trying_every_call)
            return
        if arguments is NotReady.NOT_READY:
            return
        policy = node.policy
        if not isinstance(arguments, dict):  # skipped or blank: it comes to that uncalled
            self._record(node, item_indexes, arguments, trying_every_call)
        elif (
            (reusable := self.reusable) and policy.reuse_when_inputs_match and (of_node := reusable.get(node.node_path)) is not None
            and (previous := reused_outcome(of_node, node, item_indexes, arguments, self.all_outcomes, lambda: self._in_turn(node, item_indexes))) is not None
        ):
            if previous is NotReady.NOT_ITS_TURN:
                self.out_of_turn = True  # a unit before it may take this answer; the next walk tries it again
                return
            if self.node_events is not None:
                self.node_events.outcome_reused(node)
            self._record(node, item_indexes, previous, trying_every_call)
        else:
            if self.node_events is not None:
                self.node_events.call_starts(node)
            if policy.threaded or policy.retries:
                self._make(_Attempt(node, item_indexes, arguments, 0), trying_every_call)
            else:
                self._record(node, item_indexes, self._outcome(node, item_indexes, arguments, *call(node, arguments, self.context)), trying_every_call)

    def _close(self) -> None:
        """Say how the run stands now the leg is over, and which of its nodes are finished."""
        self._put_in_order()
        # one look at every node not yet said complete: it waits for a person, or lacks a unit, or is finished
        waiting: set[FlatFunctionNode] = set()
        unfinished: set[FlatFunctionNode] = set()
        for node, outcomes in self.outcomes.items():
            if node.node_path in self.finished:
                continue
            if Waiting in {type(outcome) for outcome in outcomes.values()}:
                waiting.add(node)
            elif not (has_every_outcome(self.all_outcomes, node) if node.loops else outcomes):  # speed only: has_every_outcome says the same of a node inside no loop. Not calling it is worth 2% of a run of 500 chained nodes.
                unfinished.add(node)
        if self.going and not waiting:  # nothing more can start and nothing waits: every unit of every node is there
            if unfinished:  # a unit nothing made ready: the engine's own fault, never a run to hand on as complete
                raise RuntimeError(f"the leg ended with nothing to try and no unit for every instance of {sorted(node.node_path for node in unfinished)}")
            self.state, self.finished = COMPLETE, frozenset(node.node_path for node in self.nodes)
            self.reusable.clear()
        else:
            self.finished = frozenset(node.node_path for node in self.nodes if node not in waiting and node not in unfinished)
            for path in [path for path in self.reusable if path in self.finished]:  # the earlier work a finished node didn't take, no unit of this run will
                del self.reusable[path]
            if self.going:
                self.state = WAITING
        if self.node_events is not None:
            self.node_events.scheduler_ended()

    def _make(self, attempt: _Attempt, trying_every_call: bool = False) -> None:
        """Make one try of a call: on the spot for a node that is not called on threads, else on a thread as soon as
        the node's ``concurrency`` allows."""
        node, item_indexes, arguments = attempt.node, attempt.item_indexes, attempt.arguments
        if not node.policy.threaded:
            out, error = call(node, arguments, self.context)
            if not self._retry(attempt, error):
                if attempt.number:
                    self.running.pop((node, item_indexes), None)
                self._record(node, item_indexes, self._outcome(node, item_indexes, arguments, out, error), trying_every_call)
            return
        self.running[(node, item_indexes)] = arguments
        self.waiting_for_thread[node].append(attempt)
        self._start_waiting_calls(node)

    def _on_thread(self, attempt: _Attempt) -> None:
        """Make the call and hand what it came to back to the leg. What ``call`` lets out (an exit, a defect in a
        type's own check) goes back too and is raised on the leg's thread, as it is for a node called there."""
        try:
            out, error = call(attempt.node, attempt.arguments, self.context)
        except BaseException as raised:  # noqa: BLE001  a pool thread would keep it, and the leg would wait for the call for ever
            self.returned.put(raised)
        else:
            self.returned.put(_Returned(attempt, out, error))

    def _due(self, timer: _Timer) -> None:
        """A timer fired: the retry of an attempt, or the deadline of the attempt on a thread."""
        attempt = timer.attempt
        node, item_indexes = attempt.node, attempt.item_indexes
        if timer.kind == "retry":
            if self.going:
                self._make(attempt)
        elif self.attempts.get((node, item_indexes)) == attempt.number and (node, item_indexes) in self.running:  # the attempt still runs
            del self.running[(node, item_indexes)]
            self._record(node, item_indexes, problem.timeout(*outcome_loc(node, item_indexes), seconds=timer.seconds))

    def _returned(self, returned: _Returned) -> None:
        """A call came back from its thread: free its slot, and keep what it came to unless its attempt was
        given up."""
        attempt, out, error = returned.attempt, returned.outputs, returned.error
        node, item_indexes = attempt.node, attempt.item_indexes
        self.on_threads[node] -= 1
        latest = self.attempts.get((node, item_indexes)) == attempt.number
        current = latest and (node, item_indexes) in self.running
        self._start_waiting_calls(node)
        if not current:
            if latest and error is None and not self.going and item_indexes not in self.outcomes[node]:
                self.attempts.pop((node, item_indexes))
                self._record(node, item_indexes, Done(out))  # past its timeout, but the leg is draining: keep it, never call it again
            return  # timed out while the leg runs on, or the leg stopped
        self.attempts.pop((node, item_indexes))
        if not self._retry(attempt, error):
            del self.running[(node, item_indexes)]
            self._record(node, item_indexes, self._outcome(node, item_indexes, attempt.arguments, out, error))

    def _retry(self, attempt: _Attempt, error: BaseException | None) -> bool:
        """Put a failed attempt on a timer if the policy says to try again: an ``ExternalError``, or a class ``retry_on``
        names, never a node's own ``NodeError``, an output its type refused or an ``Ask``."""
        node, item_indexes = attempt.node, attempt.item_indexes
        policy = node.policy
        if error is None or attempt.number >= policy.retries or not self.going:
            return False
        if not isinstance(error, ExternalError) and (isinstance(error, _NEVER_RETRIED) or not isinstance(error, policy.retry_on)):
            return False
        delay = policy.retry_delay * 2**attempt.number
        self.running[(node, item_indexes)] = attempt.arguments
        self._after(delay, "retry", _Attempt(node, item_indexes, attempt.arguments, attempt.number + 1))
        if self.node_events is not None:
            self.node_events.call_retries(node, item_indexes, attempt.number + 1, delay, self._failure(node, item_indexes, error))
        return True

    def _outcome(self, node: FlatFunctionNode, item_indexes: ItemIndexes, arguments: dict[str, Any], out: Any, error: BaseException | None) -> Outcome | Problem:
        if error is None:
            return Done(out)
        if isinstance(error, Ask):
            try:
                inputs = jsonable(arguments)
            except CallRejectedError as unwritable:  # a person can't be shown what the node received
                return self._failure(node, item_indexes, unwritable)
            try:
                return Waiting(question_asked(error, node.compiled_node.outputs, inputs))
            except InvalidAskError as invalid:
                return problem.invalid_ask(*outcome_loc(node, item_indexes), reason=str(invalid))
        return self._failure(node, item_indexes, error)

    def _failure(self, node: FlatFunctionNode, item_indexes: ItemIndexes, error: BaseException) -> Problem:
        """The problem a node's exception is: its own words for a ``NodeError``; for any other, only its class, the text to the log."""
        loc = outcome_loc(node, item_indexes)
        if isinstance(error, NodeError):
            return error._as_problem_at(*loc)
        if isinstance(error, CallRejectedError):
            return error.problem.model_copy(update={"loc": loc})
        _logger.error("node %r failed for item %r", node.name, item_indexes, exc_info=error)
        if isinstance(error, node.policy.retry_on):
            return problem.external_failed(*loc, error=type(error).__name__)
        return problem.execution_failed(*loc, error=type(error).__name__)

    def _record(self, node: FlatFunctionNode, item_indexes: ItemIndexes, outcome: Outcome | Problem, trying_every_call: bool = False) -> None:
        """Keep what a unit came to, and make ready what reads it."""
        if type(outcome) is Problem:  # its exact class, which nothing derives from: isinstance of a pydantic model costs four times as much
            if self.going:  # the first failure is the cause; a later one changes nothing
                self.end(EndedEarly("failed", outcome))
                if self.node_events is not None:
                    self.node_events.call_failed(node, outcome)
            return
        outcomes = self.outcomes[node]
        if outcomes and item_indexes not in outcomes and item_indexes < next(reversed(outcomes)):
            self.out_of_order.add(node)
        outcomes[item_indexes] = outcome
        if not trying_every_call:
            self.recorded_since_last_pass += 1
            self.changed = True
        if self.checkpoint_every is not None and time.monotonic() >= self.next_checkpoint:
            self._checkpoint()
        if self.max_items is not None and node.looped_outputs and outcome.has_outputs:
            for output in node.looped_outputs:
                items = length(outcome.outputs[output])
                if items > self.max_items:
                    self._stop_at_item_limit(node, item_indexes, output, items, self.max_items)
                    return
        if self.node_events is not None:
            self.node_events.outcome_recorded(node, item_indexes, outcome)
        if trying_every_call or not self.going or not outcome.has_outputs:  # nothing reads a unit that waits for a person
            return
        depth = len(item_indexes)
        for reader, places in node.downstream:
            if places < depth:  # it is made from every unit under those places: tried once nothing it reads works under them
                self.collectors.add((reader, item_indexes[:places]))
            elif len(reader.loops) == depth:  # speed only: the branch below finds this one call too. Worth about 2% of a run whose 6,000 calls all come back from threads.
                heapq.heappush(self.ready, (reader.position, item_indexes))
            else:
                for downstream_indexes in all_item_indexes(self.all_outcomes, reader.loops, item_indexes)[0]:
                    heapq.heappush(self.ready, (reader.position, downstream_indexes))

    def _in_turn(self, node: FlatFunctionNode, item_indexes: ItemIndexes) -> bool:
        """Whether every unit of ``node`` before ``item_indexes`` has been tried, so a person's answers are taken in order,
        whatever thread finished first.

        The k-th of identical units is the k-th in order among those the leg could try: one that waits on a person
        counts as tried once nothing else can start (``_skip_calls_that_cannot_start``).
        """
        turn = self.answer_orders.get(node)
        if turn is None or item_indexes not in turn.place:
            order = all_item_indexes(self.all_outcomes, node.loops, ())[0]
            turn = self.answer_orders[node] = _AnswerOrder(order, {k: i for i, k in enumerate(order)})
        order, outcomes, passed, reached = turn.order, self.outcomes[node], turn.passed, turn.reached
        while reached < len(order) and ((next_indexes := order[reached]) in outcomes or (node, next_indexes) in self.running or next_indexes in passed):
            reached += 1
        turn.reached = reached
        return turn.place.get(item_indexes, 0) <= reached

    def _skip_calls_that_cannot_start(self) -> bool:
        """With nothing left to start, count every unit a node taking earlier answers hasn't tried as tried, so the
        ones after it take theirs.

        Such a unit waits on a person, so this leg can't learn what it receives; if it does later, it takes what is left.
        """
        passed_any = False
        for node, turn in self.answer_orders.items():
            outcomes = self.outcomes[node]
            untried = {k for k in turn.order[turn.reached :] if k not in outcomes and k not in turn.passed}
            turn.passed |= untried
            passed_any = passed_any or bool(untried)
        return passed_any

    def _release_collectors(self) -> bool:
        """Make ready each gathering unit nothing above it is still working under; False when there is none."""
        working = list(self.running)
        released = False
        for node, item_indexes in list(self.collectors):
            above = all_upstream(node)
            if not any(n in above and k[: len(item_indexes)] == item_indexes for n, k in working):
                self.collectors.discard((node, item_indexes))
                released = True
                for call_indexes in all_item_indexes(self.all_outcomes, node.loops, item_indexes)[0]:
                    heapq.heappush(self.ready, (node.position, call_indexes))
        return released

    def _check_item_limit(self) -> None:
        """Stop the leg if a Series some node works through already holds more items than ``max_items``, as the
        leg begins."""
        item_limit = self.max_items
        if item_limit is None:
            return
        for node in self.nodes:
            for output in node.looped_outputs:
                for item_indexes, outcome in self.outcomes[node].items():
                    if outcome.has_outputs and length(outcome.outputs[output]) > item_limit:
                        self._stop_at_item_limit(node, item_indexes, output, length(outcome.outputs[output]), item_limit)
                        return

    def _stop_at_item_limit(self, node: FlatFunctionNode, item_indexes: ItemIndexes, output: str, items: int, item_limit: int) -> None:
        """Stop the leg at the item ceiling, unless it already ended for a failure, a cancel or its timeout, which
        stays the cause. The cause names the Series as the graph of the reader that works through it does,
        inside the item of each subgraph's node around that."""
        if self.going:
            named, series_output = node.looped_outputs[output]
            loc = outcome_loc(named, item_indexes[: len(named.parent.loops) if named.parent is not None else 0])
            self.end(EndedEarly("stopped", problem.item_limit(*loc, series=f"{named.compiled_node.id}.{series_output}", items=items, max_items=item_limit)))
