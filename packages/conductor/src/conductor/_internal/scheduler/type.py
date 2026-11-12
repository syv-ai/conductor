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
import operator
import queue
import time
from collections import defaultdict, deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from queue import Empty
from typing import Any, Literal

from conductor._internal.address import ItemIndexes, NodePath
from conductor._internal.ask import Ask, InvalidAskError, question_asked, waiting_record
from conductor._internal.compiled import (
    FlatFunctionNode,
    FlatNode,
    FlatSubgraph,
    all_upstream,
    outcome_loc,
)
from conductor._internal.markers import SKIPPED
from conductor._internal.outcomes import (
    Answered,
    Done,
    NotCalled,
    NotReady,
    Outcome,
    Skipped,
    Waiting,
    all_item_indexes,
    calls_expected,
    has_every_outcome,
    length,
)
from conductor._internal.problem import CallRejectedError, ExternalError, NodeError, Problem
from conductor._internal.problem import catalogue as problem
from conductor._internal.run.reuse import ReusableOutcomes, reused_outcome
from conductor._internal.run.state import COMPLETE, RUNNING, WAITING, EndedEarly, RunState
from conductor._internal.scheduler.call import call, jsonable
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

#: What a call can raise that is never tried again, whatever a ``Policy``'s ``retry_on`` names: the node's own failure,
#: an output its type refused, a question for a person. Repeating the call repeats it.
_NEVER_RETRIED = (NodeError, CallRejectedError, Ask)

#: The least time between two ``NodeProgress`` of one node.
_SECONDS_BETWEEN_PROGRESS = 0.25

_logger = logging.getLogger("conductor")

#: A node's place in the order a leg tries the nodes in.
_POSITION = operator.attrgetter("position")


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
        self.emit, self.stop, self.context, self.checkpoint = emit, stop, context, checkpoint
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
        #: Each node's units waiting for a thread: (instance, what it received, attempt).
        self.waiting_for_thread: defaultdict[FlatFunctionNode, deque[tuple[ItemIndexes, dict[str, Any], int]]] = defaultdict(deque)
        #: How many of each node's calls are on threads, which its ``concurrency`` bounds.
        self.on_threads: defaultdict[FlatFunctionNode, int] = defaultdict(int)
        #: What came back from threads: a call's (node, instance, attempt, outputs, what went wrong), or what ``call`` itself raised.
        self.returned: queue.SimpleQueue[tuple[FlatFunctionNode, ItemIndexes, int, Any, BaseException | None] | BaseException] = queue.SimpleQueue()
        #: (when, order, what is due, node, instance, attempt): a retry of the attempt, or the deadline of the attempt on a thread.
        self.timers: list[tuple[float, int, Literal["retry", "deadline"], FlatFunctionNode, ItemIndexes, int, float]] = []
        self.timer_order = itertools.count()
        self.pool: ThreadPoolExecutor | None = None
        #: For a node taking a person's earlier answers, how far through its units in order the leg has tried.
        self.answer_orders: dict[FlatFunctionNode, _AnswerOrder] = {}
        #: Units recorded since the last walk over every unit ended.
        self.recorded_since_last_pass = 0
        #: Whether a unit was left untried because a unit before it may take the earlier answer it would take.
        self.out_of_turn = False
        # what only a listener needs
        with_outcomes: set[FlatFunctionNode] = {node for node, held in self.outcomes.items() if held}
        self.started: set[FlatNode] = with_outcomes | {around for node in with_outcomes for around in _parents(node)}
        #: The nodes said complete, on this leg or on an earlier one; a node an answer finished between legs is said on
        #: this one. A subgraph's node is complete when every node inside it is.
        self.complete: set[FlatNode] = {node for node in self.nodes if node.node_path in finished} if finished else set()
        if self.complete and emit is not None:
            for node in list(self.complete):
                self._complete_parents(node, announce=False)
        #: The nodes said skipped: every unit a branch not taken.
        self.skipped: set[FlatNode] = set()
        self.touched = with_outcomes - self.complete
        self.reused: defaultdict[FlatFunctionNode, int] = defaultdict(int)
        self.progress_at: dict[FlatFunctionNode, float] = {}
        self.progress_said: dict[FlatFunctionNode, int] = {}

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
            for item_indexes, _, _ in waiting:
                self.running.pop((node, item_indexes), None)
            waiting.clear()
        for _, _, due, node, item_indexes, _, _ in self.timers:
            if due == "retry":
                self.running.pop((node, item_indexes), None)

    def _wait(self) -> None:
        """Wait for a thread to come back, a timer to fire, or the next look at ``stop``."""
        timeout = self.timers[0][0] - time.monotonic() if self.timers else None
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
            self._returned(*returned)
        except Empty:
            pass
        now = time.monotonic()
        while self.timers and self.timers[0][0] <= now:
            _, _, due, node, item_indexes, attempt, seconds = heapq.heappop(self.timers)
            self._due(due, node, item_indexes, attempt, seconds)

    def _after(self, seconds: float, due: Literal["retry", "deadline"], node: FlatFunctionNode, item_indexes: ItemIndexes, attempt: int) -> None:
        """Set a timer: the retry of an attempt, or the deadline of the attempt on a thread."""
        heapq.heappush(self.timers, (time.monotonic() + seconds, next(self.timer_order), due, node, item_indexes, attempt, seconds))

    def _start_waiting_calls(self, node: FlatFunctionNode) -> None:
        """Put the node's waiting units on threads, as far as its ``concurrency`` allows."""
        waiting, policy = self.waiting_for_thread[node], node.policy
        while waiting and self.on_threads[node] < policy.concurrency and self.going:
            item_indexes, arguments, attempt = waiting.popleft()
            self.on_threads[node] += 1
            self.attempts[(node, item_indexes)] = attempt
            if self.pool is None:
                self.pool = ThreadPoolExecutor(self.threads, "conductor")
            if policy.attempt_timeout is not None:
                self._after(policy.attempt_timeout, "deadline", node, item_indexes, attempt)
            self.pool.submit(self._on_thread, node, item_indexes, arguments, attempt)

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
        if self.emit and self.touched:
            self._completions()
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
            self.reused[node] += 1
            self._record(node, item_indexes, previous, trying_every_call)
        else:
            if self.emit:
                self._announce(node)
            if policy.threaded or policy.retries:
                self._attempt(node, item_indexes, arguments, 0, trying_every_call)
            else:
                self._record(node, item_indexes, self._outcome(node, item_indexes, arguments, *call(node, arguments, self.context)), trying_every_call)

    def _close(self) -> None:
        """Say how the run stands now the leg is over, and which of its nodes are finished."""
        self._put_in_order()
        # one look at every node not yet said complete: it waits for a person, or lacks a unit, or is finished
        waiting: set[FlatFunctionNode] = set()
        unfinished: set[FlatFunctionNode] = set()
        for node, outcomes in self.outcomes.items():
            if node in self.complete:
                continue
            if Waiting in {type(outcome) for outcome in outcomes.values()}:
                waiting.add(node)
            elif not (has_every_outcome(self.all_outcomes, node) if node.loops else outcomes):  # a node that runs once has its one unit or none: no count, on a leg of many nodes
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
        if self.emit:
            self.touched.update(self.nodes)
            self._completions()
            for node in self.nodes:  # a node the leg left part done says how far it got
                if node.loops and node in self.progress_said:
                    self._progress(node, last=True)

    def _attempt(self, node: FlatFunctionNode, item_indexes: ItemIndexes, arguments: dict[str, Any], attempt: int, trying_every_call: bool = False) -> None:
        if not node.policy.threaded:
            out, error = call(node, arguments, self.context)
            if not self._retry(node, item_indexes, arguments, attempt, error):
                if attempt:
                    self.running.pop((node, item_indexes), None)
                self._record(node, item_indexes, self._outcome(node, item_indexes, arguments, out, error), trying_every_call)
            return
        self.running[(node, item_indexes)] = arguments
        self.waiting_for_thread[node].append((item_indexes, arguments, attempt))
        self._start_waiting_calls(node)

    def _on_thread(self, node: FlatFunctionNode, item_indexes: ItemIndexes, arguments: dict[str, Any], attempt: int) -> None:
        """Make the call and hand what it came to back to the leg. What ``call`` lets out (an exit, a defect in a
        type's own check) goes back too and is raised on the leg's thread, as it is for a node called there."""
        try:
            out, error = call(node, arguments, self.context)
        except BaseException as raised:  # noqa: BLE001  a pool thread would keep it, and the leg would wait for the call for ever
            self.returned.put(raised)
        else:
            self.returned.put((node, item_indexes, attempt, out, error))

    def _due(self, due: Literal["retry", "deadline"], node: FlatFunctionNode, item_indexes: ItemIndexes, attempt: int, seconds: float) -> None:
        """A timer set for ``seconds`` fired: the retry of an attempt, or the deadline of the attempt on a thread."""
        if due == "retry":
            if self.going:
                self._attempt(node, item_indexes, self.running[(node, item_indexes)], attempt)
        elif self.attempts.get((node, item_indexes)) == attempt and (node, item_indexes) in self.running:  # the attempt still runs
            del self.running[(node, item_indexes)]
            self._record(node, item_indexes, problem.timeout(*outcome_loc(node, item_indexes), seconds=seconds))

    def _returned(self, node: FlatFunctionNode, item_indexes: ItemIndexes, attempt: int, out: Any, error: BaseException | None) -> None:
        """A call came back from its thread: free its slot, and keep what it came to unless its attempt was
        given up."""
        self.on_threads[node] -= 1
        latest = self.attempts.get((node, item_indexes)) == attempt
        current = latest and (node, item_indexes) in self.running
        self._start_waiting_calls(node)
        if not current:
            if latest and error is None and not self.going and item_indexes not in self.outcomes[node]:
                self.attempts.pop((node, item_indexes))
                self._record(node, item_indexes, self._outcome(node, item_indexes, {}, out, None))  # past its timeout, but the leg is draining: keep it, never call it again
            return  # timed out while the leg runs on, or the leg stopped
        self.attempts.pop((node, item_indexes))
        arguments = self.running[(node, item_indexes)]
        if not self._retry(node, item_indexes, arguments, attempt, error):
            del self.running[(node, item_indexes)]
            self._record(node, item_indexes, self._outcome(node, item_indexes, arguments, out, error))

    def _retry(self, node: FlatFunctionNode, item_indexes: ItemIndexes, arguments: dict[str, Any], attempt: int, error: BaseException | None) -> bool:
        """Put a failed attempt on a timer if the policy says to try again: an ``ExternalError``, or a class ``retry_on``
        names, never a node's own ``NodeError``, an output its type refused or an ``Ask``."""
        policy = node.policy
        if error is None or attempt >= policy.retries or not self.going:
            return False
        if not isinstance(error, ExternalError) and (isinstance(error, _NEVER_RETRIED) or not isinstance(error, policy.retry_on)):
            return False
        delay = policy.retry_delay * 2**attempt
        self.running[(node, item_indexes)] = arguments
        self._after(delay, "retry", node, item_indexes, attempt + 1)
        if self.emit:
            self.emit(NodeRetry(loc=outcome_loc(node, item_indexes), attempt=attempt + 1, retries=policy.retries, delay=delay, cause=self._failure(node, item_indexes, error)))
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
                if self.emit:
                    self.emit(NodeFailed(loc=node.node_path, cause=outcome))
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
        if self.max_items is not None and node.looped_outputs and type(outcome) is not Waiting:
            for output in node.looped_outputs:
                items = length(outcome.outputs[output])
                if items > self.max_items:
                    self._stop_at_item_limit(node, item_indexes, output, items, self.max_items)
                    return
        if self.emit:
            self.touched.add(node)
            if type(outcome) is not Skipped:
                self._announce(node)  # a unit taken from earlier work, or a blank one, starts its node as a call does
            if type(outcome) is Waiting:
                self.emit(NodeWaiting(waiting=waiting_record(node.compiled_node, outcome_loc(node, item_indexes), outcome)))
            if node.loops:
                self._progress(node)
        if trying_every_call or not self.going or type(outcome) is Waiting:  # nothing reads a unit that waits for a person
            return
        depth = len(item_indexes)
        for reader, places in node.downstream:
            if places < depth:  # it is made from every unit under those places: tried once nothing it reads works under them
                self.collectors.add((reader, item_indexes[:places]))
            elif len(reader.loops) == depth:
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
                    if type(outcome) is not Waiting and length(outcome.outputs[output]) > item_limit:
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

    # -- what the leg says --------------------------------------------------------------------

    def _announce(self, node: FlatNode) -> None:
        """Say that ``node`` started, once, after each subgraph's node it is inside."""
        if node not in self.started:
            if node.parent is not None:
                self._announce(node.parent)
            self.started.add(node)
            if self.emit:
                self.emit(NodeStart(loc=node.node_path))

    def _progress(self, node: FlatFunctionNode, last: bool = False) -> None:
        """Say how far a node running per item has got, at most four times a second; ``last`` says its final count, unless already said."""
        now = time.monotonic()
        emit = self.emit
        if not emit or not (last or now - self.progress_at.get(node, -1.0) >= _SECONDS_BETWEEN_PROGRESS):
            return
        done = sum(1 for outcome in self.outcomes[node].values() if type(outcome) is not Waiting)  # counted only when said, at most four times a second
        if self.progress_said.get(node) != done:
            self.progress_at[node], self.progress_said[node] = now, done
            every_item_indexes, known = all_item_indexes(self.all_outcomes, node.loops, ())
            emit(NodeProgress(loc=node.node_path, done=done, total=len(every_item_indexes) if known else None))

    def _completions(self) -> None:
        """Say which touched nodes are complete: nothing they wait on is working, and every unit is done."""
        emit = self.emit
        assert emit is not None
        working = {node for node, _ in self.running}
        for node in sorted(self.touched, key=_POSITION):
            if node in working or working & all_upstream(node):
                continue
            self.touched.discard(node)
            outcomes = self.outcomes[node]
            if node in self.complete or not self._finished(node):
                continue
            self.complete.add(node)
            if node.loops and node in self.progress_said:
                self._progress(node, last=True)
            if outcomes and all(_is_skipped_or_not_called(outcome) for outcome in outcomes.values()):
                self.skipped.add(node)
                emit(NodeSkipped(loc=node.node_path))
            else:
                not_called = sum(type(outcome) is NotCalled for outcome in outcomes.values())
                emit(NodeComplete(loc=node.node_path, outcomes=len(outcomes), not_called=not_called, reused=self.reused[node]))
            self._complete_parents(node, announce=True)

    def _complete_parents(self, node: FlatNode, announce: bool) -> None:
        """Count the subgraph's node around ``node`` as complete once every node inside it is, and so on
        outwards; with ``announce``, tell the listener: skipped when every node inside was, else complete, with one
        unit for each item the subgraph's node runs for."""
        around = node.parent
        while around is not None and around not in self.complete and all(inner in self.complete for inner in around.nodes):
            self.complete.add(around)
            if announce and self.emit is not None:
                if all(inner in self.skipped for inner in around.nodes):
                    self.skipped.add(around)
                    self.emit(NodeSkipped(loc=around.node_path))
                else:
                    self.emit(NodeComplete(loc=around.node_path, outcomes=calls_expected(self.all_outcomes, around.loops)[0]))
            around = around.parent

    def _finished(self, node: FlatFunctionNode, may_wait: bool = True) -> bool:
        """Whether every unit of ``node`` is known and none of them is still to run or waiting; ``may_wait`` False
        when the caller already knows none waits."""
        if may_wait and Waiting in {type(outcome) for outcome in self.outcomes[node].values()}:
            return False
        return has_every_outcome(self.all_outcomes, node)


def _is_skipped_or_not_called(outcome: Outcome) -> bool:
    """Whether a unit is a branch not taken: its node wasn't called for one above it, or was called and returned
    ``SKIPPED`` on every output."""
    return type(outcome) is Skipped or ((type(outcome) is Done or type(outcome) is Answered) and bool(outcome.outputs) and all(value is SKIPPED for value in outcome.outputs.values()))


def _parents(node: FlatNode) -> list[FlatSubgraph]:
    """The subgraphs' nodes ``node`` is inside, innermost first."""
    around = []
    while node.parent is not None:
        node = node.parent
        around.append(node)
    return around
