"""One leg of a run: start every unit whose inputs exist, record what comes of each, and say so as events.

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
the leg walks every node once more if any unit was left waiting on another;
it ends when nothing more can start.
"""

from __future__ import annotations

import heapq
import logging
import queue
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from conductor._internal.ask import Ask, InvalidAskError, Waiting, question_asked
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import ExternalError, NodeError, Problem, make_problem
from conductor._internal.fields import SKIPPED, FieldValueError
from conductor._internal.graph import shared_depth
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
    Run,
    RunStatus,
    Unit,
    _length,
    jsonable,
)

if TYPE_CHECKING:
    from conductor._internal.graph import PlacedNode

#: What a call can raise that is never tried again, whatever a ``Policy``'s ``retry_on`` names: the node's own failure,
#: an output its type refused, a question for a person. Repeating the call repeats it.
_NEVER_RETRIED = (NodeError, FieldValueError, Ask)

#: The least time between two ``NodeProgress`` of one node.
_SECONDS_BETWEEN_PROGRESS = 0.25

_logger = logging.getLogger("conductor")


# -- events -------------------------------------------------------------------------------------


class _EventBase(ConductorModel):
    """Every event: its schema marks ``type`` and every field a dump always writes as required, so a generated client can rely on them."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True, arbitrary_types_allowed=True, ser_json_inf_nan="strings")


class RunStart(_EventBase):
    """A leg began; ``waiting`` is the units already waiting for a person."""

    type: Literal["run_start"] = "run_start"
    waiting: tuple[Waiting, ...] = ()


class NodeStart(_EventBase):
    """A node's first unit started, called or taken from earlier work; never again for that node, on this or a later leg."""

    type: Literal["node_start"] = "node_start"
    node_id: str


class NodeProgress(_EventBase):
    """How many units of a node that runs per item are done, at most four times a second; ``total`` ``None`` until
    every Series above it is made."""

    type: Literal["node_progress"] = "node_progress"
    node_id: str
    done: int
    total: int | None


class NodeComplete(_EventBase):
    """Every unit of a node is done.

    Its values stay in the run, as ``run.results`` and ``run.not_called`` read them, so an event is small whatever
    the run holds.

    Attributes:
        node_id: The node.
        units: How many units it has: one, or one per item.
        not_called: How many of them a blank input kept the node from.
        reused_units: How many of them ``run.again`` took from the earlier run.
    """

    type: Literal["node_complete"] = "node_complete"
    node_id: str
    units: int
    not_called: int = 0
    reused_units: int = 0


class NodeSkipped(_EventBase):
    """Every unit of a node was a branch not taken."""

    type: Literal["node_skipped"] = "node_skipped"
    node_id: str


class NodeFailed(_EventBase):
    """A unit of the node failed, finally; ``cause`` says why, at ``(node_id, *item)``."""

    type: Literal["node_failed"] = "node_failed"
    node_id: str
    cause: Problem


class NodeRetry(_EventBase):
    """An attempt failed on the outside world and is tried again: attempt ``attempt`` of ``retries``, for ``item``, after ``delay`` seconds."""

    type: Literal["node_retry"] = "node_retry"
    node_id: str
    item: tuple[int, ...]
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
    """One leg over ``run``, which it changes in place. ``emit`` is None when nobody listens.

    ``stop`` is asked between units, and says why the leg must end now (a cancel, the deadline), or None.
    """

    def __init__(
        self, run: Run, emit: Callable[[_EventBase], None] | None, stop: Callable[[], EndedEarly | None] | None, from_run: dict[type, Any],
        checkpoint_every: float | None = None,
    ) -> None:
        self.run, self.stop, self.from_run = run, stop, from_run
        self.checkpoint_every = checkpoint_every if emit is not None else None
        self.next_checkpoint = time.monotonic() + checkpoint_every if checkpoint_every is not None else 0.0
        self.emit = emit
        self.nodes = list(run.graph._nodes.values())
        self.position = {p.id: i for i, p in enumerate(self.nodes)}
        self.ready: list[tuple[int, Key]] = []  # (node position, instance), tried in that order
        #: (node, an instance above its own): a node that gathers, started in the instances under that one once
        #: nothing it reads still works there.
        self.gathers: set[tuple[str, Key]] = set()
        self.running: dict[tuple[str, Key], dict[str, Any]] = {}  # units handed to threads, and their inputs
        self.slots_waiting: dict[str, deque[tuple[Key, dict[str, Any], int]]] = {p.id: deque() for p in self.nodes}
        self.on_threads = {p.id: 0 for p in self.nodes}
        self.returned: queue.SimpleQueue[tuple[str, Key, int, Any, BaseException | None]] = queue.SimpleQueue()
        #: (when, order, what is due, node, instance, attempt): a retry of the attempt, or the deadline of the attempt on a thread.
        self.timers: list[tuple[float, int, Literal["retry", "deadline"], str, Key, int]] = []
        self.attempts: dict[tuple[str, Key], int] = {}  # the attempt each unit on a thread is on
        self.pool: ThreadPoolExecutor | None = None
        #: (node, a node reading it): how many maps their scopes share, worked out once.
        self.shared: dict[tuple[str, str], int] = {}
        #: For a node taking a person's earlier answers, how far through its units in order the leg has tried.
        self.turns: dict[str, _Turn] = {}
        #: Units recorded since the last walk over every unit; a walk with none to follow would find nothing new.
        self.recorded_since_walk = 1
        #: Whether the last walk left a unit whose inputs weren't there yet, so another walk may start it.
        self.left_pending = True
        # what only a listener needs
        self.started = {n for n, units in run._units.items() if units}
        self.touched: set[str] = set()
        #: The nodes said complete, on this leg or on an earlier one; a node an answer finished between legs is said on this one.
        self.complete: set[str] = set(run._complete)
        if seeded := {n for n, units in run._units.items() if units and n not in self.complete}:
            self.touched.update(seeded)
        self.reused = {p.id: 0 for p in self.nodes}
        self.progress_at: dict[str, float] = {}
        self.progress_said: dict[str, int] = {}

    def go(self) -> None:
        run = self.run
        run.state = _RUNNING  # the one running state a leg holds, so it is asked by identity on each unit
        if self.emit:
            self.emit(RunStart(waiting=tuple(run.waiting)))
        try:
            if not self._over_ceiling():
                self._loop()
            if self.checkpoint_every is not None and self.running:  # stopped with calls on threads: the drain may outlast the process
                self._checkpoint()
            while self.running and self.run.state is not _RUNNING:  # calls already made had their effect: keep them, as long as their timeouts allow
                self._wait()
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=False, cancel_futures=True)
        # one look at every unit: a node already said complete has no unit waiting
        waiting = {n for n, us in run._units.items() if n not in self.complete and any(type(u) is Asking for u in us.values())}
        if run.state is _RUNNING:
            run.state = _WAITING if waiting else _COMPLETE
        if run.status == "complete":  # every unit of every node is finished, and nothing carried will be taken
            run._complete, run._earlier = frozenset(self.position), {}
        else:
            run._complete = frozenset(self.complete | {p.id for p in self.nodes if p.id not in self.complete and p.id not in waiting and self._finished(p, False)})
            for node_id in run._complete if run._earlier else ():  # what a finished node was carried and didn't take, no unit of this run will
                run._earlier.pop(node_id, None)
        if self.emit:
            self.touched.update(self.position)
            self._completions()
            for placed in self.nodes:  # a node the leg left part done says how far it got
                if placed.scope and placed.id in self.progress_said:
                    self._progress(placed, last=True)
            self.emit(RunEnd.of(run))

    def _loop(self) -> None:
        while self.run.state is _RUNNING:
            if self.stop is not None and (stopped := self.stop()) is not None:
                self.run.state = stopped
                self._start_nothing_more()
            elif self.ready:
                position, key = heapq.heappop(self.ready)
                self._try(self.nodes[position], key)
            elif self.gathers:
                self._check_gathers()
            else:
                if self.emit and self.touched:
                    self._completions()
                if self.running:
                    self._wait()
                elif self.left_pending and (self.recorded_since_walk or (self.turns and self._pass_untried())):
                    self._walk()
                else:
                    return

    def _walk(self) -> None:
        """Try every unit of every node, in order: the first look, and the last one when nothing runs."""
        self.left_pending, self.recorded_since_walk = False, 0
        units = self.run._units
        for placed in self.nodes:
            for key in self.run._instances(placed.scope, ())[0]:
                if key not in units[placed.id] and (placed.id, key) not in self.running:
                    if not self._try(placed, key, walking=True):
                        self.left_pending = True
                    if self.run.state is not _RUNNING or (self.stop is not None and self.stop() is not None):
                        return

    def _try(self, placed: PlacedNode, key: Key, walking: bool = False) -> bool:
        """Start ``placed`` on ``key`` if its inputs exist; False if they don't yet."""
        run = self.run
        if not walking and (key in run._units[placed.id] or (placed.id, key) in self.running):
            return True
        kwargs = run._inputs(placed, key)
        if kwargs is PENDING:
            return False
        if type(kwargs) is not dict:  # skipped or blank: it comes to that uncalled
            self._record(placed, key, kwargs, walking)
        elif run._earlier and placed.policy.reuse_when_inputs_match and (
            earlier := run._reused(placed, key, kwargs, lambda: self._in_turn(placed, key))
        ) is not None:
            if earlier is PENDING:
                self.left_pending = True  # a unit before it may take this answer; the next walk tries it again
                return False
            self.reused[placed.id] += 1
            self._record(placed, key, earlier, walking)
        else:
            if self.emit:
                self._announce(placed)
            policy = placed.policy
            if policy.threaded or policy.retries:
                self._attempt(placed, key, kwargs, 0, walking)
            else:
                self._record(placed, key, self._outcome(placed, key, kwargs, *run._call(placed, kwargs, self.from_run)), walking)
        return True

    def _attempt(self, placed: PlacedNode, key: Key, kwargs: dict[str, Any], attempt: int, walking: bool = False) -> None:
        if not placed.policy.threaded:
            out, error = self.run._call(placed, kwargs, self.from_run)
            if not self._retry(placed, key, kwargs, attempt, error):
                if attempt:
                    self.running.pop((placed.id, key), None)
                self._record(placed, key, self._outcome(placed, key, kwargs, out, error), walking)
            return
        self.running[(placed.id, key)] = kwargs
        self.slots_waiting[placed.id].append((key, kwargs, attempt))
        self._fill_slots(placed)

    def _fill_slots(self, placed: PlacedNode) -> None:
        waiting, policy = self.slots_waiting[placed.id], placed.policy
        while waiting and self.on_threads[placed.id] < policy.concurrency and self.run.state is _RUNNING:
            key, kwargs, attempt = waiting.popleft()
            self.on_threads[placed.id] += 1
            self.attempts[(placed.id, key)] = attempt
            if self.pool is None:
                self.pool = ThreadPoolExecutor(sum(p.policy.concurrency for p in self.nodes if p.policy.threaded), "conductor")
            if policy.attempt_timeout is not None:
                heapq.heappush(self.timers, (time.monotonic() + policy.attempt_timeout, len(self.timers), "deadline", placed.id, key, attempt))
            self.pool.submit(self._on_thread, placed, key, kwargs, attempt)

    def _on_thread(self, placed: PlacedNode, key: Key, kwargs: dict[str, Any], attempt: int) -> None:
        self.returned.put((placed.id, key, attempt, *self.run._call(placed, kwargs, self.from_run)))

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
            _, _, due, node_id, key, attempt = heapq.heappop(self.timers)
            placed = self.run.graph._nodes[node_id]
            if due == "retry":
                if self.run.state is _RUNNING:
                    self._attempt(placed, key, self.running[(node_id, key)], attempt)
            elif self.attempts.get((node_id, key)) == attempt and (node_id, key) in self.running:  # the attempt still runs
                del self.running[(node_id, key)]
                seconds = placed.policy.attempt_timeout
                self._record(placed, key, make_problem("timeout", node_id, *key, seconds=seconds))

    def _returned(self, node_id: str, key: Key, attempt: int, out: Any, error: BaseException | None) -> None:
        placed = self.run.graph._nodes[node_id]
        self.on_threads[node_id] -= 1
        latest = self.attempts.get((node_id, key)) == attempt
        current = latest and (node_id, key) in self.running
        self._fill_slots(placed)
        if not current:
            if latest and error is None and self.run.state is not _RUNNING and key not in self.run._units[node_id]:
                self.attempts.pop((node_id, key))
                self._record(placed, key, self._outcome(placed, key, {}, out, None))  # past its timeout, but the leg is draining: keep it, never call it again
            return  # timed out while the leg runs on, or the leg stopped
        self.attempts.pop((node_id, key))
        kwargs = self.running[(node_id, key)]
        if not self._retry(placed, key, kwargs, attempt, error):
            del self.running[(node_id, key)]
            self._record(placed, key, self._outcome(placed, key, kwargs, out, error))

    def _retry(self, placed: PlacedNode, key: Key, kwargs: dict[str, Any], attempt: int, error: BaseException | None) -> bool:
        """Put a failed attempt on a timer if the policy says to try again: an ``ExternalError``, or a class ``retry_on``
        names, never a node's own ``NodeError``, an output its type refused or an ``Ask``."""
        policy = placed.policy
        if error is None or attempt >= policy.retries or self.run.state is not _RUNNING:
            return False
        if not isinstance(error, ExternalError) and (isinstance(error, _NEVER_RETRIED) or not isinstance(error, policy.retry_on)):
            return False
        delay = policy.retry_delay * 2**attempt
        self.running[(placed.id, key)] = kwargs
        heapq.heappush(self.timers, (time.monotonic() + delay, len(self.timers), "retry", placed.id, key, attempt + 1))
        if self.emit:
            self.emit(NodeRetry(node_id=placed.id, item=key, attempt=attempt + 1, retries=policy.retries, delay=delay, cause=self._failure(placed, key, error)))
        return True

    def _outcome(self, placed: PlacedNode, key: Key, kwargs: dict[str, Any], out: Any, error: BaseException | None) -> Unit | Problem:
        if error is None:
            return SKIPPED_UNIT if out is SKIPPED else Done(out)
        if isinstance(error, Ask):
            try:
                inputs = jsonable(kwargs)
            except FieldValueError as unwritable:  # a person can't be shown what the node received
                return self._failure(placed, key, unwritable)
            try:
                return Asking(question_asked(error, placed.outputs, inputs))
            except InvalidAskError as invalid:
                return make_problem("invalid_ask", placed.id, *key, reason=str(invalid))
        return self._failure(placed, key, error)

    def _failure(self, placed: PlacedNode, key: Key, error: BaseException) -> Problem:
        """The problem a node's exception is: its own words for a ``NodeError``; for any other, only its class, the text to the log."""
        if isinstance(error, NodeError):
            return error._as_problem_at(placed.id, *key)
        if isinstance(error, FieldValueError):
            return make_problem(error.code, placed.id, *key, reason=error.reason, **error.ctx)
        _logger.error("node %r failed for item %r", placed.id, key, exc_info=error)
        if isinstance(error, placed.policy.retry_on):
            return make_problem("external_failed", placed.id, *key, error=type(error).__name__)
        return make_problem("execution_failed", placed.id, *key, error=type(error).__name__)

    def _record(self, placed: PlacedNode, key: Key, unit: Unit | Problem, walking: bool = False) -> None:
        """Keep what a unit came to, and make ready what reads it."""
        run = self.run
        if isinstance(unit, Problem):
            match run.state:
                case Going():  # the first failure is the cause; a later one changes nothing
                    run.state = EndedEarly("failed", unit)
                    self._start_nothing_more()
                    if self.emit:
                        self.emit(NodeFailed(node_id=placed.id, cause=unit))
                case EndedEarly():
                    pass
            return
        units = run._units[placed.id]
        if units and key not in units and key < next(reversed(units)):
            run._unordered.add(placed.id)
        units[key] = unit
        self.recorded_since_walk += 1
        if self.checkpoint_every is not None and time.monotonic() >= self.next_checkpoint:
            self._checkpoint()
        if placed.opens and run.max_items is not None and isinstance(unit, Done):
            for output in placed.opens:
                items = _length(unit.out[output])
                if items > run.max_items:
                    run._stop(placed, output, items)
                    self._start_nothing_more()
                    return
        if self.emit:
            self.touched.add(placed.id)
            if unit is not SKIPPED_UNIT:
                self._announce(placed)  # a unit taken from earlier work, or a blank one, starts its node as a call does
            if isinstance(unit, Asking):
                self.emit(UnitWaiting(unit=run._waiting(placed, key, unit)))  # type: ignore[arg-type]
            if placed.scope:
                self._progress(placed)
        if walking or run.state is not _RUNNING:
            return
        for reader in run.graph._readers[placed.id]:
            shared = self._shared(placed, reader)
            if shared < len(key):  # a reader above this instance: tried once nothing it reads works under the one they share
                self.gathers.add((reader.id, key[:shared]))
            else:
                for instance in run._instances(reader.scope, key)[0]:
                    heapq.heappush(self.ready, (self.position[reader.id], instance))

    def _checkpoint(self) -> None:
        """Hand the listener a copy of the run as it stands, and set when the next is due."""
        assert self.emit is not None and self.checkpoint_every is not None
        self.next_checkpoint = time.monotonic() + self.checkpoint_every
        snapshot = self.run._copy()
        self.run._earlier_own = set()  # the snapshot shares what is carried: the leg copies a node's before it next takes from it
        self.emit(RunCheckpoint(run=snapshot))

    def _in_turn(self, placed: PlacedNode, key: Key) -> bool:
        """Whether every unit of ``placed`` before ``key`` has been tried, so a person's answers are taken in order,
        whatever thread finished first.

        The k-th of identical units is the k-th in order among those the leg could try: one that waits on a person
        counts as tried once nothing else can start (``_pass_untried``).
        """
        turn = self.turns.get(placed.id)
        if turn is None or key not in turn.place:
            order = self.run._instances(placed.scope, ())[0]
            turn = self.turns[placed.id] = _Turn(order, {k: i for i, k in enumerate(order)})
        order, units, passed, reached = turn.order, self.run._units[placed.id], turn.passed, turn.reached
        while reached < len(order) and ((instance := order[reached]) in units or (placed.id, instance) in self.running or instance in passed):
            reached += 1
        turn.reached = reached
        return turn.place.get(key, 0) <= reached

    def _pass_untried(self) -> bool:
        """With nothing left to start, count every unit a node taking earlier answers hasn't tried as tried, so the
        ones after it take theirs.

        Such a unit waits on a person, so this leg can't learn what it receives; if it does later, it takes what is left.
        """
        passed_any = False
        for node_id, turn in self.turns.items():
            units = self.run._units[node_id]
            untried = {k for k in turn.order[turn.reached :] if k not in units and k not in turn.passed}
            turn.passed |= untried
            passed_any = passed_any or bool(untried)
        return passed_any

    def _shared(self, placed: PlacedNode, reader: PlacedNode) -> int:
        """How many maps the scopes of ``placed`` and of a node reading it share from the graph's own scope down."""
        found = self.shared.get((placed.id, reader.id))
        if found is None:
            found = self.shared[(placed.id, reader.id)] = shared_depth(placed.scope, reader.scope)
        return found

    def _start_nothing_more(self) -> None:
        """Drop the units not yet on a thread and the retries not yet due; the calls on threads are waited for and kept."""
        for node_id, waiting in self.slots_waiting.items():
            for k, _, _ in waiting:
                self.running.pop((node_id, k), None)
            waiting.clear()
        for _, _, due, node_id, k, _ in self.timers:
            if due == "retry":
                self.running.pop((node_id, k), None)

    def _check_gathers(self) -> None:
        """Make ready each gathering unit nothing above it is still working under; wait if none is."""
        busy = list(self.running)
        for node_id, key in list(self.gathers):
            above = self.run.graph._above_of(node_id)
            if not self.running or not any(n in above and k[: len(key)] == key for n, k in busy):
                self.gathers.discard((node_id, key))
                placed = self.nodes[self.position[node_id]]
                for instance in self.run._instances(placed.scope, key)[0] if len(key) < len(placed.scope) else (key,):
                    heapq.heappush(self.ready, (self.position[node_id], instance))
        if self.gathers and not self.ready:
            self._wait()

    def _over_ceiling(self) -> bool:
        """Whether a Series some node works through already holds more items than ``max_items``, as a leg begins;
        if so the run stops, and starts nothing."""
        run = self.run
        if run.max_items is None:
            return False
        for placed in self.nodes:
            for output in placed.opens:
                longest = max((_length(unit.out[output]) for unit in run._units[placed.id].values() if isinstance(unit, Done)), default=0)
                if longest > run.max_items:
                    run._stop(placed, output, longest)
                    return True
        return False

    # -- what the leg says --------------------------------------------------------------------

    def _announce(self, placed: PlacedNode) -> None:
        if placed.id not in self.started:
            self.started.add(placed.id)
            if self.emit:
                self.emit(NodeStart(node_id=placed.id))

    def _progress(self, placed: PlacedNode, last: bool = False) -> None:
        """Say how far a node running per item has got, at most four times a second; ``last`` says its final count, unless already said."""
        now = time.monotonic()
        if not self.emit or not (last or now - self.progress_at.get(placed.id, -1.0) >= _SECONDS_BETWEEN_PROGRESS):
            return
        done = sum(1 for unit in self.run._units[placed.id].values() if type(unit) is not Asking)  # counted only when said, at most four times a second
        if self.progress_said.get(placed.id) != done:
            self.progress_at[placed.id], self.progress_said[placed.id] = now, done
            keys, known = self.run._instances(placed.scope, ())
            self.emit(NodeProgress(node_id=placed.id, done=done, total=len(keys) if known else None))

    def _completions(self) -> None:
        """Say which touched nodes are complete: nothing they wait on is working, and every unit is done."""
        assert self.emit is not None
        running = {n for n, _ in self.running}
        for node_id in sorted(self.touched, key=self.position.__getitem__):
            if node_id in running or running & self.run.graph._above_of(node_id):
                continue
            self.touched.discard(node_id)
            placed, units = self.run.graph._nodes[node_id], self.run._units[node_id]
            if node_id in self.complete or not self._finished(placed):
                continue
            self.complete.add(node_id)
            if placed.scope and node_id in self.progress_said:
                self._progress(placed, last=True)
            if units and all(unit is SKIPPED_UNIT for unit in units.values()):
                self.emit(NodeSkipped(node_id=node_id))
            else:
                not_called = sum(type(unit) is Blank for unit in units.values())
                self.emit(NodeComplete(node_id=node_id, units=len(units), not_called=not_called, reused_units=self.reused[node_id]))

    def _finished(self, placed: PlacedNode, may_wait: bool = True) -> bool:
        """Whether every unit of ``placed`` is known and none of them is still to run or waiting; ``may_wait`` False
        when the caller already knows none waits."""
        units = self.run._units[placed.id]
        if not placed.scope:
            for unit in units.values():
                return type(unit) is not Asking
            return False
        if may_wait and any(type(unit) is Asking for unit in units.values()):
            return False
        counted, known = self.run._instances(placed.scope, (), counted=True)
        return known and len(units) == (counted[0][0] if counted else 0)
