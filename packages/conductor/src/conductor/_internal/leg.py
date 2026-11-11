"""One leg of a run: start every unit whose inputs exist, record what comes of each, and say so as events.

A leg walks every node of the run's graph and of the graphs embedded in it,
each where it stands (``graph._tree``). A node inside an embedded graph is
one of them, with a unit per item of the embedding node and of its own scope,
so it is called, retried, timed out and cancelled as the same node drawn in
the graph around it. An embedded graph's node is never called: its unit is
what the outputs of the graph inside it came to for that item.

The leg first walks the nodes in order and tries every unit that can exist,
as a later leg would. A node whose policy has ``concurrency`` 1 and no
timeout is called on the spot; any other is handed to threads, at most
``concurrency`` of its units at once, and the walk goes on. When a call comes
back, the units that read it are tried at once: the reader's units under,
at, or above the unit's instance. So item 1 can finish the chain while item
10 is still being made.

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
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from conductor._internal.ask import Ask, InvalidAskError, Waiting, question_asked
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.fields import SKIPPED, FieldValueError
from conductor._internal.graph import FunctionAtPath, GraphAtPath, NodeAtPath, waits_on
from conductor._internal.problem import ExternalError, NodeError, Problem, make_problem
from conductor._internal.run import (
    _COMPLETE,
    _RUNNING,
    _WAITING,
    PENDING,
    SKIPPED_UNIT,
    Asking,
    Blank,
    Done,
    EndedEarly,
    Going,
    Key,
    Path,
    Run,
    RunStatus,
    Unit,
    _length,
    jsonable,
    unit_loc,
    waiting_unit,
)

#: What a call can raise that is never tried again, whatever a ``Policy``'s ``retry_on`` names: the node's own failure,
#: an output its type refused, a question for a person. Repeating the call repeats it.
_NEVER_RETRIED = (NodeError, FieldValueError, Ask)

#: The least time between two ``NodeProgress`` of one node.
_SECONDS_BETWEEN_PROGRESS = 0.25

_logger = logging.getLogger("conductor")

#: A node's place in the order a leg tries the nodes in.
_POSITION = operator.attrgetter("position")


# -- events -------------------------------------------------------------------------------------


class _EventBase(ConductorModel):
    """Every event: its schema marks ``type`` and every field a dump always writes as required, so a generated client can rely on them."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True, arbitrary_types_allowed=True, ser_json_inf_nan="strings")


class _NodeEvent(_EventBase):
    """Every event about one node, or one unit of it. Never made on its own.

    Attributes:
        loc: Which node it is: its id, and for a node inside an embedded graph the id of each embedding node
            first, ``("e", "upper")``. A node inside an embedded graph says its events once, for every item of
            the embedding node together, as a node that runs per item does.
    """

    loc: Path


class RunStart(_EventBase):
    """A leg began; ``waiting`` is the units already waiting for a person."""

    type: Literal["run_start"] = "run_start"
    waiting: tuple[Waiting, ...] = ()


class NodeStart(_NodeEvent):
    """A node's first unit started, called or taken from earlier work; never again for that node, on this or a later leg."""

    type: Literal["node_start"] = "node_start"


class NodeProgress(_NodeEvent):
    """How many units of a node that runs per item are done, at most four times a second; ``total`` ``None`` until
    every Series above it is made."""

    type: Literal["node_progress"] = "node_progress"
    done: int
    total: int | None


class NodeComplete(_NodeEvent):
    """Every unit of a node is done.

    Its values stay in the run, as ``run.results`` and ``run.not_called`` read them, so an event is small whatever
    the run holds.

    Attributes:
        units: How many units it has: one, or one per item.
        not_called: How many of them a blank input kept the node from.
        reused_units: How many of them ``run.again`` took from the earlier run.
    """

    type: Literal["node_complete"] = "node_complete"
    units: int
    not_called: int = 0
    reused_units: int = 0


class NodeSkipped(_NodeEvent):
    """Every unit of a node was a branch not taken."""

    type: Literal["node_skipped"] = "node_skipped"


class NodeFailed(_NodeEvent):
    """A unit of the node failed, finally; ``cause`` says why, and its ``loc`` which unit: the node's id, then the
    unit's item, and inside an embedded graph the embedding node and its item first, ``("e", 0, "upper")``."""

    type: Literal["node_failed"] = "node_failed"
    cause: Problem


class NodeRetry(_NodeEvent):
    """An attempt failed on the outside world and is tried again: attempt ``attempt`` of ``retries``, after
    ``delay`` seconds. Its ``loc`` is the unit's, as its ``cause`` has it: the node's id, then the place of the
    unit's item in each Series the node runs per item of; inside an embedded graph, the embedding node and its
    item first."""

    type: Literal["node_retry"] = "node_retry"
    attempt: int
    retries: int
    delay: float
    cause: Problem


class UnitWaiting(_EventBase):
    """A unit started waiting for a person. It waits, whatever reads it waits, and the rest of the graph runs on."""

    type: Literal["unit_waiting"] = "unit_waiting"
    unit: Waiting


class RunEnd(_EventBase):
    """The leg ended, with ``status`` and, for a run that ``failed``, ``stopped``, was ``cancelled`` or ``timed_out``, its ``cause``.

    ``run`` is the run as the leg left it, every value in it; it stays out of what the event writes, for the host to store.
    """

    type: Literal["run_end"] = "run_end"
    status: RunStatus
    cause: Problem | None = None
    run: SkipJsonSchema[Run] = Field(exclude=True)

    @classmethod
    def of(cls, run: Run) -> RunEnd:
        """The end of the leg that left ``run`` as it is."""
        match run.state:
            case Going(status=status):
                return cls(status=status, run=run)
            case EndedEarly(status=status, cause=cause):
                return cls(status=status, cause=cause, run=run)


class RunCheckpoint(_EventBase):
    """The run as it stands mid-leg, for a host to store, every ``checkpoint_every`` seconds and once when the leg is stopped.

    ``run`` is a copy, its ``status`` ``running``, or once stopped how it stopped (``cancelled``, ``failed``,
    ``stopped``, ``timed_out``) with its ``cause``: ``graph.load_run`` takes its JSON and ``resume()`` calls only
    the units it holds no record of, so a process that dies mid-leg repeats at most the calls that were in
    flight. Like ``RunEnd``'s, it stays out of what the event writes.
    """

    type: Literal["run_checkpoint"] = "run_checkpoint"
    run: SkipJsonSchema[Run] = Field(exclude=True)


Event = Annotated[
    RunStart | NodeStart | NodeProgress | NodeComplete | NodeSkipped | NodeFailed | NodeRetry | UnitWaiting | RunCheckpoint | RunEnd,
    Field(discriminator="type"),
]
Event.__doc__ = "Any event ``run.stream()`` yields, told apart by ``type``: ``TypeAdapter(Event)`` reads one back from JSON."


# -- the leg ------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Turn:
    """How far a leg has tried a node's units in order, so a person's answers are taken in order.

    Attributes:
        order: The node's instances, in order.
        place: Each instance's position in ``order``.
        reached: How many instances from the first have been tried.
        passed: Instances counted as tried because nothing in the leg could start them (they wait on a person).
    """

    order: list[Key]
    place: dict[Key, int]
    reached: int = 0
    passed: set[Key] = field(default_factory=set)


class Leg:
    """One leg over ``run``, which it changes in place.

    The leg walks every node of the run's graph and of the graphs embedded in it (``graph._tree``): it holds
    which of their units can start, the threads, each node's slots on them, the timers, ``stop``, the listener
    and the checkpoint clock. Only node functions run on threads; the run is changed on the leg's own thread.

    ``emit`` is None when nobody listens. ``stop`` is asked between units, and says why the leg must end now (a
    cancel, the deadline), or None.
    """

    def __init__(
        self, run: Run, emit: Callable[[_EventBase], None] | None, stop: Callable[[], EndedEarly | None] | None, from_run: dict[type, Any],
        checkpoint_every: float | None = None,
    ) -> None:
        self.run, self.emit, self.stop, self.from_run = run, emit, stop, from_run
        self.nodes = run.graph._tree
        #: Each node's units by instance: the run's own, recorded into.
        self.units = units = {node: run._units[node.inside][node.placed.id] for node in self.nodes}
        #: The nodes whose units were recorded out of instance order, as calls on threads finish when they finish.
        self.out_of_order: set[NodeAtPath] = set()
        #: What each node may take from the run this one went on from, by its path.
        self.carried = run._earlier
        #: Whether nothing has ended the leg early; ``end`` is the one place that does.
        self.going = True
        self.checkpoint_every = checkpoint_every if emit is not None else None
        self.next_checkpoint = time.monotonic() + checkpoint_every if checkpoint_every is not None else 0.0
        #: (node position, instance): the units to try, in that order.
        self.ready: list[tuple[int, Key]] = []
        #: Whether a walk ended or a unit was recorded outside one since the leg last looked over the run.
        self.changed = True
        #: (node, an instance above its own): a node that gathers, started in the instances under that one once
        #: nothing it reads still works there.
        self.gathers: set[tuple[NodeAtPath, Key]] = set()
        self.running: dict[tuple[FunctionAtPath, Key], dict[str, Any]] = {}  # units handed to threads, and their inputs
        self.attempts: dict[tuple[FunctionAtPath, Key], int] = {}  # the attempt each unit on a thread is on
        #: Each node's units waiting for a thread: (instance, what it received, attempt).
        self.slots_waiting: defaultdict[FunctionAtPath, deque[tuple[Key, dict[str, Any], int]]] = defaultdict(deque)
        #: How many of each node's calls are on threads, which its ``concurrency`` bounds.
        self.on_threads: defaultdict[FunctionAtPath, int] = defaultdict(int)
        self.returned: queue.SimpleQueue[tuple[FunctionAtPath, Key, int, Any, BaseException | None]] = queue.SimpleQueue()
        #: (when, order, what is due, node, instance, attempt): a retry of the attempt, or the deadline of the attempt on a thread.
        self.timers: list[tuple[float, int, Literal["retry", "deadline"], FunctionAtPath, Key, int]] = []
        self.timer_order = itertools.count()
        self.pool: ThreadPoolExecutor | None = None
        #: For a node taking a person's earlier answers, how far through its units in order the leg has tried.
        self.turns: dict[FunctionAtPath, _Turn] = {}
        #: Units recorded since the last walk over every unit ended. A walk tries a node after the nodes it reads,
        #: so it starts whatever its own units make ready, and another with none recorded since would find nothing new.
        self.recorded_since_walk = 1
        #: Whether the last walk left a unit whose inputs weren't there yet, so another walk may start it.
        self.left_pending = True
        # what only a listener needs
        with_units: set[NodeAtPath] = {node for node, held in units.items() if held}
        self.started = with_units | {around for node in with_units for around in _around(node)}
        #: The nodes said complete, on this leg or on an earlier one; a node an answer finished between legs is said on this one.
        self.complete = {node for node in self.nodes if node.path in run._complete} if run._complete else set()
        self.touched = with_units - self.complete
        self.reused: defaultdict[NodeAtPath, int] = defaultdict(int)
        self.progress_at: dict[NodeAtPath, float] = {}
        self.progress_said: dict[NodeAtPath, int] = {}

    def go(self) -> None:
        """Run the leg: try what is ready, look over the run when something happened in it, and wait for the
        threads, until nothing more can start or something ends the leg early."""
        run = self.run
        run.state = _RUNNING  # the one running state a leg holds
        if self.emit:
            self.emit(RunStart(waiting=tuple(run.waiting)))
        try:
            self._check_ceiling()
            while self.going:
                if self.stop is not None and (stopped := self.stop()) is not None:
                    self.end(stopped)
                elif self.ready:
                    position, key = heapq.heappop(self.ready)
                    self._try(self.nodes[position], key)
                elif self.changed:
                    self.changed = False
                    if not self._look_over():
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
        if self.emit:
            self.emit(RunEnd.of(run))

    def end(self, state: EndedEarly) -> None:
        """End the leg early: nothing more starts. The units waiting for a thread and the retries not yet due are
        dropped; the calls on threads are waited for and kept."""
        self.run.state, self.going = state, False
        for node, waiting in self.slots_waiting.items():
            for key, _, _ in waiting:
                self.running.pop((node, key), None)
            waiting.clear()
        for _, _, due, node, key, _ in self.timers:
            if due == "retry":
                self.running.pop((node, key), None)

    def _wait(self) -> None:
        """Wait for a thread to come back, a timer to fire, or the next look at ``stop``."""
        timeout = self.timers[0][0] - time.monotonic() if self.timers else None
        if self.stop is not None:
            timeout = 0.05 if timeout is None else min(timeout, 0.05)
        try:
            self._returned(*self.returned.get(timeout=None if timeout is None else max(timeout, 0)))
        except queue.Empty:
            pass
        now = time.monotonic()
        while self.timers and self.timers[0][0] <= now:
            _, _, due, node, key, attempt = heapq.heappop(self.timers)
            self._due(due, node, key, attempt)

    def _after(self, seconds: float, due: Literal["retry", "deadline"], node: FunctionAtPath, key: Key, attempt: int) -> None:
        """Set a timer: the retry of an attempt, or the deadline of the attempt on a thread."""
        heapq.heappush(self.timers, (time.monotonic() + seconds, next(self.timer_order), due, node, key, attempt))

    def _fill_slots(self, node: FunctionAtPath) -> None:
        """Put the node's waiting units on threads, as far as its ``concurrency`` allows."""
        waiting, policy = self.slots_waiting[node], node.policy
        while waiting and self.on_threads[node] < policy.concurrency and self.going:
            key, kwargs, attempt = waiting.popleft()
            self.on_threads[node] += 1
            self.attempts[(node, key)] = attempt
            if self.pool is None:
                self.pool = ThreadPoolExecutor(self.run.graph._threads, "conductor")
            if policy.attempt_timeout is not None:
                self._after(policy.attempt_timeout, "deadline", node, key, attempt)
            self.pool.submit(self._on_thread, node, key, kwargs, attempt)

    def _checkpoint(self) -> None:
        """Hand the listener a copy of the run as it stands, and set when the next is due."""
        assert self.emit is not None and self.checkpoint_every is not None
        self.next_checkpoint = time.monotonic() + self.checkpoint_every
        self._put_in_order()
        copy = self.run._copy()
        copy._settle_embedded()  # the leg makes an embedded graph's unit when it comes to try it; the copy is read now
        self.emit(RunCheckpoint(run=copy))

    def _put_in_order(self) -> None:
        """Put the units of each node recorded out of instance order back in it, so the run holds every node's
        units in instance order whenever it leaves the leg: at its end, and as a checkpoint's copy."""
        for node in self.out_of_order:
            units = self.units[node]
            in_order = sorted(units.items())
            units.clear()
            units.update(in_order)
        self.out_of_order.clear()

    def _look_over(self) -> bool:
        """With nothing ready to try: start the gathering units nothing works under any more and say which nodes
        are complete. Once nothing is working, walk every unit again if that could start one.

        Returns:
            False once nothing more can start in this leg.
        """
        if self.gathers and self._release_gathers():
            self.changed = True
            return True
        if self.emit and self.touched:
            self._completions()
        if self.running:
            return True
        if self.left_pending and (self.recorded_since_walk or (self.turns and self._pass_untried())):
            self._walk()
            self.changed = True
            return True
        return False

    def _walk(self) -> None:
        """Try every unit of every node, in order: the first look, and the last one when nothing runs."""
        self.left_pending, self.recorded_since_walk = False, 0
        run, running = self.run, self.running
        for node in self.nodes:
            units = self.units[node]
            for key in run._instances(node.maps, ())[0]:
                if key not in units and (node, key) not in running:
                    if not self._try(node, key, walking=True):
                        self.left_pending = True
                    if not self.going or (self.stop is not None and self.stop() is not None):
                        return

    def _try(self, node: NodeAtPath, key: Key, walking: bool = False) -> bool:
        """Start ``node`` on ``key`` if what it reads exists; False if it doesn't yet."""
        run = self.run
        if not walking and (key in self.units[node] or (node, key) in self.running):
            return True
        if type(node) is GraphAtPath:  # never called: its unit is what its inner graph's outputs came to
            offered = run._offered(node, key)
            if offered is PENDING:
                return False
            self._record(node, key, Done(offered), walking)
            return True
        kwargs = run._inputs(node, key)
        if kwargs is PENDING:
            return False
        policy = node.policy
        if type(kwargs) is not dict:  # skipped or blank: it comes to that uncalled
            self._record(node, key, kwargs, walking)
        elif (
            (carried := self.carried) and policy.reuse_when_inputs_match and (lent := carried.get(node.path)) is not None
            and (earlier := run._reused(lent, node, key, kwargs, lambda: self._in_turn(node, key))) is not None
        ):
            if earlier is PENDING:
                self.left_pending = True  # a unit before it may take this answer; the next walk tries it again
                return False
            self.reused[node] += 1
            self._record(node, key, earlier, walking)
        else:
            if self.emit:
                self._announce(node)
            if policy.threaded or policy.retries:
                self._attempt(node, key, kwargs, 0, walking)
            else:
                self._record(node, key, self._outcome(node, key, kwargs, *run._call(node, kwargs, self.from_run)), walking)
        return True

    def _close(self) -> None:
        """Say how the run stands now the leg is over, and which of its nodes are finished."""
        run = self.run
        self._put_in_order()
        if not self.going:  # ended early: the units recorded since made nothing ready, an embedded graph's node included
            run._settle_embedded()
        # one look at every unit: a node already said complete has no unit waiting
        waiting = {node for node, units in self.units.items() if node not in self.complete and Asking in {type(unit) for unit in units.values()}}
        if self.going and not waiting:  # every unit of every node is finished, and nothing carried will be taken
            run.state, run._complete, run._earlier = _COMPLETE, frozenset(run.graph._node_at), {}
        else:
            finished = {node for node in self.nodes if node not in self.complete and node not in waiting and self._finished(node, False)}
            run._complete = frozenset(node.path for node in self.complete | finished)
            for path in [path for path in run._earlier if path in run._complete]:  # what a finished node was carried and didn't take, no unit of this run will
                del run._earlier[path]
            if self.going:
                run.state = _WAITING
        if self.emit:
            self.touched.update(self.nodes)
            self._completions()
            for node in self.nodes:  # a node the leg left part done says how far it got
                if node.maps and node in self.progress_said:
                    self._progress(node, last=True)

    def _attempt(self, node: FunctionAtPath, key: Key, kwargs: dict[str, Any], attempt: int, walking: bool = False) -> None:
        if not node.policy.threaded:
            out, error = self.run._call(node, kwargs, self.from_run)
            if not self._retry(node, key, kwargs, attempt, error):
                if attempt:
                    self.running.pop((node, key), None)
                self._record(node, key, self._outcome(node, key, kwargs, out, error), walking)
            return
        self.running[(node, key)] = kwargs
        self.slots_waiting[node].append((key, kwargs, attempt))
        self._fill_slots(node)

    def _on_thread(self, node: FunctionAtPath, key: Key, kwargs: dict[str, Any], attempt: int) -> None:
        self.returned.put((node, key, attempt, *self.run._call(node, kwargs, self.from_run)))

    def _due(self, due: Literal["retry", "deadline"], node: FunctionAtPath, key: Key, attempt: int) -> None:
        """A timer fired: the retry of an attempt, or the deadline of the attempt on a thread."""
        if due == "retry":
            if self.going:
                self._attempt(node, key, self.running[(node, key)], attempt)
        elif self.attempts.get((node, key)) == attempt and (node, key) in self.running:  # the attempt still runs
            del self.running[(node, key)]
            self._record(node, key, make_problem("timeout", *unit_loc(node, key), seconds=node.policy.attempt_timeout))

    def _returned(self, node: FunctionAtPath, key: Key, attempt: int, out: Any, error: BaseException | None) -> None:
        """A call came back from its thread: free its slot, and keep what it came to unless its attempt was
        given up."""
        self.on_threads[node] -= 1
        latest = self.attempts.get((node, key)) == attempt
        current = latest and (node, key) in self.running
        self._fill_slots(node)
        if not current:
            if latest and error is None and not self.going and key not in self.units[node]:
                self.attempts.pop((node, key))
                self._record(node, key, self._outcome(node, key, {}, out, None))  # past its timeout, but the leg is draining: keep it, never call it again
            return  # timed out while the leg runs on, or the leg stopped
        self.attempts.pop((node, key))
        kwargs = self.running[(node, key)]
        if not self._retry(node, key, kwargs, attempt, error):
            del self.running[(node, key)]
            self._record(node, key, self._outcome(node, key, kwargs, out, error))

    def _retry(self, node: FunctionAtPath, key: Key, kwargs: dict[str, Any], attempt: int, error: BaseException | None) -> bool:
        """Put a failed attempt on a timer if the policy says to try again: an ``ExternalError``, or a class ``retry_on``
        names, never a node's own ``NodeError``, an output its type refused or an ``Ask``."""
        policy = node.policy
        if error is None or attempt >= policy.retries or not self.going:
            return False
        if not isinstance(error, ExternalError) and (isinstance(error, _NEVER_RETRIED) or not isinstance(error, policy.retry_on)):
            return False
        delay = policy.retry_delay * 2**attempt
        self.running[(node, key)] = kwargs
        self._after(delay, "retry", node, key, attempt + 1)
        if self.emit:
            self.emit(NodeRetry(loc=unit_loc(node, key), attempt=attempt + 1, retries=policy.retries, delay=delay, cause=self._failure(node, key, error)))
        return True

    def _outcome(self, node: FunctionAtPath, key: Key, kwargs: dict[str, Any], out: Any, error: BaseException | None) -> Unit | Problem:
        if error is None:
            return Done(out)
        if isinstance(error, Ask):
            try:
                inputs = jsonable(kwargs)
            except FieldValueError as unwritable:  # a person can't be shown what the node received
                return self._failure(node, key, unwritable)
            try:
                return Asking(question_asked(error, node.placed.outputs, inputs))
            except InvalidAskError as invalid:
                return make_problem("invalid_ask", *unit_loc(node, key), reason=str(invalid))
        return self._failure(node, key, error)

    def _failure(self, node: FunctionAtPath, key: Key, error: BaseException) -> Problem:
        """The problem a node's exception is: its own words for a ``NodeError``; for any other, only its class, the text to the log."""
        loc = unit_loc(node, key)
        if isinstance(error, NodeError):
            return error._as_problem_at(*loc)
        if isinstance(error, FieldValueError):
            return make_problem(error.code, *loc, reason=error.reason, **error.ctx)
        _logger.error("node %r failed for item %r", "/".join(node.path), key, exc_info=error)
        if isinstance(error, node.policy.retry_on):
            return make_problem("external_failed", *loc, error=type(error).__name__)
        return make_problem("execution_failed", *loc, error=type(error).__name__)

    def _record(self, node: NodeAtPath, key: Key, unit: Unit | Problem, walking: bool = False) -> None:
        """Keep what a unit came to, and make ready what reads it."""
        run = self.run
        if type(unit) is Problem:  # its exact class, which nothing derives from: isinstance of a pydantic model costs four times as much
            if self.going:  # the first failure is the cause; a later one changes nothing
                self.end(EndedEarly("failed", unit))
                if self.emit:
                    self.emit(NodeFailed(loc=node.path, cause=unit))
            return
        units, placed = self.units[node], node.placed
        if units and key not in units and key < next(reversed(units)):
            self.out_of_order.add(node)
        units[key] = unit
        if not walking:
            self.recorded_since_walk += 1
            self.changed = True
        if self.checkpoint_every is not None and time.monotonic() >= self.next_checkpoint:
            self._checkpoint()
        if run.max_items is not None and placed.opens and isinstance(unit, Done):
            for output in placed.opens:
                items = _length(unit.out[output])
                if items > run.max_items:
                    self._stop_at_ceiling(node, key, output, items)
                    return
        if self.emit:
            self.touched.add(node)
            if unit is not SKIPPED_UNIT:
                self._announce(node)  # a unit taken from earlier work, or a blank one, starts its node as a call does
            if isinstance(unit, Asking):
                assert type(node) is FunctionAtPath  # only a function's unit asks
                self.emit(UnitWaiting(unit=waiting_unit(node, key, unit)))
            if node.maps:
                self._progress(node)
        if walking or not self.going or type(unit) is Asking:  # nothing reads a unit that waits for a person
            return
        depth = len(key)
        for reader, shared in node.readers:
            if shared < depth:  # a reader above this instance: tried once nothing it reads works under the one they share
                self.gathers.add((reader, key[:shared]))
            elif len(reader.maps) == depth:
                heapq.heappush(self.ready, (reader.position, key))
            else:
                for instance in run._instances(reader.maps, key)[0]:
                    heapq.heappush(self.ready, (reader.position, instance))

    def _in_turn(self, node: FunctionAtPath, key: Key) -> bool:
        """Whether every unit of ``node`` before ``key`` has been tried, so a person's answers are taken in order,
        whatever thread finished first.

        The k-th of identical units is the k-th in order among those the leg could try: one that waits on a person
        counts as tried once nothing else can start (``_pass_untried``).
        """
        turn = self.turns.get(node)
        if turn is None or key not in turn.place:
            order = self.run._instances(node.maps, ())[0]
            turn = self.turns[node] = _Turn(order, {k: i for i, k in enumerate(order)})
        order, units, passed, reached = turn.order, self.units[node], turn.passed, turn.reached
        while reached < len(order) and ((instance := order[reached]) in units or (node, instance) in self.running or instance in passed):
            reached += 1
        turn.reached = reached
        return turn.place.get(key, 0) <= reached

    def _pass_untried(self) -> bool:
        """With nothing left to start, count every unit a node taking earlier answers hasn't tried as tried, so the
        ones after it take theirs.

        Such a unit waits on a person, so this leg can't learn what it receives; if it does later, it takes what is left.
        """
        passed_any = False
        for node, turn in self.turns.items():
            units = self.units[node]
            untried = {k for k in turn.order[turn.reached :] if k not in units and k not in turn.passed}
            turn.passed |= untried
            passed_any = passed_any or bool(untried)
        return passed_any

    def _release_gathers(self) -> bool:
        """Make ready each gathering unit nothing above it is still working under; False when there is none."""
        working = list(self.running)
        released = False
        for node, key in list(self.gathers):
            above = waits_on(node)
            if not any(n in above and k[: len(key)] == key for n, k in working):
                self.gathers.discard((node, key))
                released = True
                for instance in self.run._instances(node.maps, key)[0] if len(key) < len(node.maps) else (key,):
                    heapq.heappush(self.ready, (node.position, instance))
        return released

    def _check_ceiling(self) -> None:
        """Stop the leg if a Series some node works through already holds more items than ``max_items``, as the
        leg begins."""
        ceiling = self.run.max_items
        if ceiling is None:
            return
        for node in self.nodes:
            for output in node.placed.opens:
                for key, unit in self.units[node].items():
                    if isinstance(unit, Done) and _length(unit.out[output]) > ceiling:
                        self._stop_at_ceiling(node, key, output, _length(unit.out[output]))
                        return

    def _stop_at_ceiling(self, node: NodeAtPath, key: Key, output: str, items: int) -> None:
        """Stop the leg at the item ceiling, unless it already ended for a failure, a cancel or its timeout, which
        stays the cause. The cause names the node, inside the item of each embedded graph's node around it."""
        if self.going:
            loc = unit_loc(node, key[: node.base])
            self.end(EndedEarly("stopped", make_problem("item_ceiling", *loc, series=f"{node.placed.id}.{output}", items=items, max_items=self.run.max_items)))

    # -- what the leg says --------------------------------------------------------------------

    def _announce(self, node: NodeAtPath) -> None:
        """Say that ``node`` started, once, after each embedded graph's node it is inside."""
        if node not in self.started:
            if node.embedding is not None:
                self._announce(node.embedding)
            self.started.add(node)
            if self.emit:
                self.emit(NodeStart(loc=node.path))

    def _progress(self, node: NodeAtPath, last: bool = False) -> None:
        """Say how far a node running per item has got, at most four times a second; ``last`` says its final count, unless already said."""
        now = time.monotonic()
        emit = self.emit
        if not emit or not (last or now - self.progress_at.get(node, -1.0) >= _SECONDS_BETWEEN_PROGRESS):
            return
        done = sum(1 for unit in self.units[node].values() if type(unit) is not Asking)  # counted only when said, at most four times a second
        if self.progress_said.get(node) != done:
            self.progress_at[node], self.progress_said[node] = now, done
            keys, known = self.run._instances(node.maps, ())
            emit(NodeProgress(loc=node.path, done=done, total=len(keys) if known else None))

    def _completions(self) -> None:
        """Say which touched nodes are complete: nothing they wait on is working, and every unit is done."""
        emit = self.emit
        assert emit is not None
        working = {node for node, _ in self.running}
        for node in sorted(self.touched, key=_POSITION):
            if node in working or working & waits_on(node):
                continue
            self.touched.discard(node)
            units = self.units[node]
            if node in self.complete or not self._finished(node):
                continue
            self.complete.add(node)
            if node.maps and node in self.progress_said:
                self._progress(node, last=True)
            if units and all(_came_to_no_value(unit) for unit in units.values()):
                emit(NodeSkipped(loc=node.path))
            else:
                not_called = sum(type(unit) is Blank for unit in units.values())
                emit(NodeComplete(loc=node.path, units=len(units), not_called=not_called, reused_units=self.reused[node]))

    def _finished(self, node: NodeAtPath, may_wait: bool = True) -> bool:
        """Whether every unit of ``node`` is known and none of them is still to run or waiting; ``may_wait`` False
        when the caller already knows none waits."""
        units = self.units[node]
        if not node.maps:
            for unit in units.values():
                return type(unit) is not Asking
            return False
        if may_wait and Asking in {type(unit) for unit in units.values()}:
            return False
        counted, known = self.run._instances(node.maps, (), counted=True)
        return known and len(units) == (counted[0][0] if counted else 0)


def _came_to_no_value(unit: Unit) -> bool:
    """Whether a unit is a branch not taken: its node wasn't called for one above it, or was called and returned
    ``SKIPPED`` on every output."""
    return unit is SKIPPED_UNIT or (isinstance(unit, Done) and bool(unit.out) and all(value is SKIPPED for value in unit.out.values()))


def _around(node: NodeAtPath) -> list[GraphAtPath]:
    """The embedded graphs' nodes ``node`` is inside, innermost first."""
    around = []
    while node.embedding is not None:
        node = node.embedding
        around.append(node)
    return around
