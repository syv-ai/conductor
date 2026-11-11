"""One leg of a run: start every unit whose inputs exist, record what comes of each, and say so as events.

A leg drives the run it was started on and, for each instance of an embedded
graph, the nested run that is its call. Each run has a frontier of its own;
the threads, each node's slots on them, the timers and the events are the
leg's, so a node inside an embedded graph is called, retried, timed out and
cancelled as the same node drawn in the graph around it.

A frontier first walks its run's nodes in order and tries every unit that can exist,
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
so it is tried once and not once per item it reads. When nothing of a run is
working, its frontier walks every node once more if any unit was left waiting
on another, and closes when nothing more can start: a nested run that is
complete is then the unit of its embedded graph's instance. The leg ends when
the frontier of its own run closes.
"""

from __future__ import annotations

import heapq
import itertools
import logging
import queue
import time
from collections import defaultdict, deque
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
from conductor._internal.graph import PlacedGraph
from conductor._internal.run import (
    _COMPLETE,
    _RUNNING,
    _WAITING,
    NOT_NESTED,
    PENDING,
    SKIPPED_UNIT,
    Asking,
    Blank,
    Done,
    EndedEarly,
    Going,
    Key,
    Nesting,
    NodePath,
    Path,
    Run,
    RunStatus,
    Unit,
    _length,
    jsonable,
    path_text,
    waiting_unit,
)

if TYPE_CHECKING:
    from conductor._internal.graph import PlacedFunction, PlacedNode

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


class _NodeEvent(_EventBase):
    """Every event about one node, or one unit of it. Never made on its own.

    Attributes:
        loc: Where the node is, as a ``Problem``'s ``loc`` says it: the node's id, and for a node inside an
            embedded graph the embedding node and its item first, ``("e", 0, "upper")``. A node inside an embedded
            graph says its events once per nested run, so a host that shows it once folds them by the ids in
            ``loc``.
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
    """A unit of the node failed, finally; ``cause`` says why, and its ``loc`` which unit: this ``loc``, then the
    unit's item."""

    type: Literal["node_failed"] = "node_failed"
    cause: Problem


class NodeRetry(_NodeEvent):
    """An attempt failed on the outside world and is tried again: attempt ``attempt`` of ``retries``, after
    ``delay`` seconds. Its ``loc`` is the unit's: the node's, then the place of the unit's item in each Series the
    node runs per item of."""

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


#: A node whatever nested run it is in: the ids of the embedded graphs' nodes it is inside, outermost first, and
#: its own. What its ``concurrency`` counts.
_Slots = tuple[NodePath, str]


@dataclass(slots=True)
class _Turn:
    """How far a frontier has tried a node's units in order, so a person's answers are taken in order.

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
    """One leg over ``run``, which it changes in place, and over every nested run under it.

    The leg holds what the runs it drives share: the units ready to try, the threads, each node's slots on them,
    the timers, ``stop``, the listener and the checkpoint clock. Each run, the one the leg was started on and the
    nested run of each instance of an embedded graph, has its own ``Frontier``. Only node functions run on
    threads; every run is changed on the leg's own thread.

    ``emit`` is None when nobody listens. ``stop`` is asked between units, and says why the leg must end now (a
    cancel, the deadline), or None.
    """

    def __init__(
        self, run: Run, emit: Callable[[_EventBase], None] | None, stop: Callable[[], EndedEarly | None] | None, from_run: dict[type, Any],
        checkpoint_every: float | None = None,
    ) -> None:
        self.run, self.emit, self.stop, self.from_run = run, emit, stop, from_run
        #: What each node may take from the run this one went on from, by node whatever nested run it is in.
        self.carried = run._earlier
        #: Whether nothing has ended the leg early; ``end`` is the one place that does.
        self.going = True
        self.checkpoint_every = checkpoint_every if emit is not None else None
        self.next_checkpoint = time.monotonic() + checkpoint_every if checkpoint_every is not None else 0.0
        #: The frontier of each run being driven, by its number: the order they were opened in, the leg's own run first.
        self.frontiers: dict[int, Frontier] = {}
        self.opened = 0
        #: (frontier's number, node position, instance): the units to try, in that order.
        self.ready: list[tuple[int, int, Key]] = []
        #: The frontiers a unit was recorded in since each was last looked over, in that order.
        self.changed: dict[Frontier, None] = {}
        #: By a node whatever nested run it is in, ``(("e",), "fetch")``: its units waiting for a thread.
        self.slots_waiting: defaultdict[_Slots, deque[tuple[Frontier, Key, dict[str, Any], int]]] = defaultdict(deque)
        #: By a node whatever nested run it is in: how many of its calls are on threads, so ``concurrency`` bounds
        #: the node across nested runs.
        self.on_threads: defaultdict[_Slots, int] = defaultdict(int)
        self.returned: queue.SimpleQueue[tuple[Frontier, PlacedFunction, Key, int, Any, BaseException | None]] = queue.SimpleQueue()
        #: (when, order, what is due, frontier, node, instance, attempt): a retry of the attempt, or the deadline of the attempt on a thread.
        self.timers: list[tuple[float, int, Literal["retry", "deadline"], Frontier, PlacedFunction, Key, int]] = []
        self.timer_order = itertools.count()
        self.pool: ThreadPoolExecutor | None = None

    def go(self) -> None:
        """Run the leg: try what is ready, look over each run something happened in, and wait for the threads, until
        the leg's own run closes or something ends the leg early."""
        run = self.run
        run.state = _RUNNING  # the one running state a leg holds
        if self.emit:
            self.emit(RunStart(waiting=tuple(run.waiting)))
        try:
            self.open(run, NOT_NESTED, None)
            while self.frontiers and self.going:
                if self.stop is not None and (stopped := self.stop()) is not None:
                    self.end(stopped)
                elif self.ready:
                    number, position, key = heapq.heappop(self.ready)
                    frontier = self.frontiers[number]
                    frontier._try(frontier.nodes[position], key)
                elif self.changed:
                    frontier = next(iter(self.changed))
                    del self.changed[frontier]
                    frontier._look_over()
                else:
                    self._wait()
            if self.checkpoint_every is not None and self._calls_on_threads():  # stopped with calls on threads: the drain may outlast the process
                self._checkpoint()
            while self._calls_on_threads():  # calls already made had their effect: keep them, as long as their timeouts allow
                self._wait()
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=False, cancel_futures=True)
        for frontier in reversed(list(self.frontiers.values())):  # ended early: each nested run before the run it is a call of
            frontier._close()
        if self.emit:
            self.emit(RunEnd.of(run))

    def open(self, run: Run, nesting: Nesting, embedded_as: tuple[Frontier, PlacedGraph, Key] | None) -> Frontier:
        """Start driving ``run``. Its units are first tried when it is looked over; a Series it already holds over
        the item ceiling stops the leg.

        Args:
            run: The run.
            nesting: Where it sits in the leg's own run.
            embedded_as: For a nested run, the frontier of the run around it, and the embedded graph and instance
                it is the call of.
        """
        frontier = self.frontiers[self.opened] = Frontier(self, self.opened, run, nesting, embedded_as)
        self.opened += 1
        self.changed[frontier] = None
        frontier._check_ceiling()
        return frontier

    def end(self, state: EndedEarly) -> None:
        """End the leg early: nothing more starts in any run it drives. The units waiting for a thread and the
        retries not yet due are dropped; the calls on threads are waited for and kept."""
        self.run.state, self.going = state, False
        for (_, node_id), waiting in self.slots_waiting.items():
            for frontier, key, _, _ in waiting:
                frontier.running.pop((node_id, key), None)
            waiting.clear()
        for _, _, due, frontier, placed, key, _ in self.timers:
            if due == "retry":
                frontier.running.pop((placed.id, key), None)

    def _calls_on_threads(self) -> bool:
        """Whether a call of any run is still on a thread, within its timeout, once the leg has ended early."""
        return any(frontier.running for frontier in self.frontiers.values())

    def _wait(self) -> None:
        """Wait for a thread to come back, a timer to fire, or the next look at ``stop``."""
        timeout = self.timers[0][0] - time.monotonic() if self.timers else None
        if self.stop is not None:
            timeout = 0.05 if timeout is None else min(timeout, 0.05)
        try:
            frontier, placed, key, attempt, out, error = self.returned.get(timeout=None if timeout is None else max(timeout, 0))
            frontier._returned(placed, key, attempt, out, error)
        except queue.Empty:
            pass
        now = time.monotonic()
        while self.timers and self.timers[0][0] <= now:
            _, _, due, frontier, placed, key, attempt = heapq.heappop(self.timers)
            frontier._due(due, placed, key, attempt)

    def _after(self, seconds: float, due: Literal["retry", "deadline"], frontier: Frontier, placed: PlacedFunction, key: Key, attempt: int) -> None:
        """Set a timer: the retry of an attempt, or the deadline of the attempt on a thread."""
        heapq.heappush(self.timers, (time.monotonic() + seconds, next(self.timer_order), due, frontier, placed, key, attempt))

    def _fill_slots(self, slots: _Slots, placed: PlacedFunction) -> None:
        """Put the node's waiting units on threads, as far as its ``concurrency`` allows across every run it is in."""
        waiting, policy = self.slots_waiting[slots], placed.definition.policy
        while waiting and self.on_threads[slots] < policy.concurrency and self.going:
            frontier, key, kwargs, attempt = waiting.popleft()
            self.on_threads[slots] += 1
            frontier.attempts[(placed.id, key)] = attempt
            if self.pool is None:
                self.pool = ThreadPoolExecutor(self.run.graph._threads, "conductor")
            if policy.attempt_timeout is not None:
                self._after(policy.attempt_timeout, "deadline", frontier, placed, key, attempt)
            self.pool.submit(frontier._on_thread, placed, key, kwargs, attempt)

    def _checkpoint(self) -> None:
        """Hand the listener a copy of the run as it stands, its nested runs with it, and set when the next is due."""
        assert self.emit is not None and self.checkpoint_every is not None
        self.next_checkpoint = time.monotonic() + self.checkpoint_every
        self.emit(RunCheckpoint(run=self.run._copy()))


class Frontier:
    """One run in a leg: which of its units can start, and what came of each.

    A nested run has a frontier like the run around it; only where it sits differs. When nothing more of it can
    start in the leg, it closes: complete, it is the unit of the embedded graph's instance it is the call of.

    Attributes:
        leg: The leg, and what every run in it shares.
        number: The frontier's place among the leg's, in the order they were opened.
        run: The run, changed in place.
        nesting: Where the run sits in the leg's own run: the node and item of each embedded graph it is nested
            in, which every event, problem and waiting id of the run starts with.
        embedded_as: For a nested run, the frontier of the run around it, and the embedded graph and instance it
            is the call of; ``None`` for the run the leg was started on.
    """

    def __init__(self, leg: Leg, number: int, run: Run, nesting: Nesting, embedded_as: tuple[Frontier, PlacedGraph, Key] | None) -> None:
        self.leg, self.number, self.run, self.nesting, self.embedded_as = leg, number, run, nesting, embedded_as
        self.inside, self.node_path = nesting.path, nesting.nodes
        self.nodes, self.position = run.graph._order, run.graph._position
        #: (node, an instance above its own): a node that gathers, started in the instances under that one once
        #: nothing it reads still works there.
        self.gathers: set[tuple[str, Key]] = set()
        self.running: dict[tuple[str, Key], dict[str, Any]] = {}  # units handed to threads, and their inputs
        #: The frontier of each nested run open under this one, by the embedded graph's node and instance.
        self.nested: dict[tuple[str, Key], Frontier] = {}
        self.attempts: dict[tuple[str, Key], int] = {}  # the attempt each unit on a thread is on
        #: For a node taking a person's earlier answers, how far through its units in order the frontier has tried.
        self.turns: dict[str, _Turn] = {}
        #: Units recorded since the last walk over every unit; a walk with none to follow would find nothing new.
        self.recorded_since_walk = 1
        #: Whether the last walk left a unit whose inputs weren't there yet, so another walk may start it.
        self.left_pending = True
        #: Whether a unit of a nested run is left until the instances of its embedded graph before this one have
        #: taken their earlier answers.
        self.awaits_turn = False
        # what only a listener needs
        self.started = {n for n, units in run._units.items() if units} | {n for n, runs in run._nested.items() if runs}
        self.touched: set[str] = set()
        #: The nodes said complete, on this leg or on an earlier one; a node an answer finished between legs is said on this one.
        self.complete: set[str] = set(run._complete)
        if seeded := {n for n, units in run._units.items() if units and n not in self.complete}:
            self.touched.update(seeded)
        self.reused: defaultdict[str, int] = defaultdict(int)
        self.progress_at: dict[str, float] = {}
        self.progress_said: dict[str, int] = {}

    def _look_over(self) -> None:
        """With nothing ready to try: start the gathering units nothing works under any more and say which nodes
        are complete. Once nothing of the run is working, walk its units again if that could start one, and else
        close it."""
        if self.gathers and self._release_gathers():
            self.leg.changed[self] = None
            return
        if self.leg.emit and self.touched:
            self._completions()
        if self.running or self.nested:
            return
        if self.left_pending and (self.recorded_since_walk or (self.turns and not self.awaits_turn and self._pass_untried())):
            self._walk()
            self.leg.changed[self] = None
        else:
            self._close()

    def _walk(self) -> None:
        """Try every unit of every node, in order: the first look, and the last one when nothing runs."""
        self.left_pending, self.awaits_turn, self.recorded_since_walk = False, False, 0
        leg, units = self.leg, self.run._units
        for placed in self.nodes:
            for key in self.run._instances(placed.scope, ())[0]:
                if key not in units[placed.id] and (placed.id, key) not in self.running and (placed.id, key) not in self.nested:
                    if not self._try(placed, key, walking=True):
                        self.left_pending = True
                    if not leg.going or (leg.stop is not None and leg.stop() is not None):
                        return

    def _try(self, placed: PlacedNode, key: Key, walking: bool = False) -> bool:
        """Start ``placed`` on ``key`` if its inputs exist; False if they don't yet."""
        run = self.run
        if not walking and (key in run._units[placed.id] or (placed.id, key) in self.running or (placed.id, key) in self.nested):
            return True
        kwargs = run._inputs(placed, key)
        if kwargs is PENDING:
            return False
        if type(kwargs) is not dict:  # skipped or blank: it comes to that uncalled
            self._record(placed, key, kwargs, walking)
        elif type(placed) is PlacedGraph:  # its call is a nested run, whose nodes take earlier work one by one
            self._open(placed, key, kwargs)
        elif (
            (carried := self.leg.carried) and placed.definition.policy.reuse_when_inputs_match
            and (lent := carried.get((*self.node_path, placed.id))) is not None  # by node, whatever nested run it is in
            and (earlier := run._reused(lent, placed, key, kwargs, lambda: self._in_turn(placed, key), self.nesting)) is not None
        ):
            if earlier is PENDING:
                self.left_pending = True  # a unit before it may take this answer; the next walk tries it again
                return False
            self.reused[placed.id] += 1
            self._record(placed, key, earlier, walking)
        else:
            if self.leg.emit:
                self._announce(placed)
            policy = placed.definition.policy
            if policy.threaded or policy.retries:
                self._attempt(placed, key, kwargs, 0, walking)
            else:
                self._record(placed, key, self._outcome(placed, key, kwargs, *run._call(placed, kwargs, self.leg.from_run)), walking)
        return True

    def _open(self, placed: PlacedGraph, key: Key, kwargs: dict[str, Any]) -> None:
        """Start driving the nested run of an embedded graph's instance: the one the run keeps from an earlier
        leg, else one of what the instance received."""
        try:
            nested = self.run._nested_run(placed, key, kwargs)
        except FieldValueError as refused:  # an edge brought what the inner graph's input doesn't take
            self._record(placed, key, make_problem(refused.code, placed.id, *key, reason=refused.reason, **refused.ctx))
            return
        if self.leg.emit:
            self._announce(placed)
        self.nested[(placed.id, key)] = self.leg.open(nested, self.run._nesting_of(self.nesting, placed, key), (self, placed, key))

    def _close(self) -> None:
        """Stop driving the run: say how it stands and which of its nodes are finished. A nested run that is
        complete becomes the unit of the embedded graph's instance it is the call of, its outputs the unit's."""
        run, leg = self.run, self.leg
        # one look at every unit: a node already said complete has no unit waiting; an embedded graph waits while a nested run of it does
        waiting = {n for n, us in run._units.items() if n not in self.complete and Asking in {type(u) for u in us.values()}}
        waiting |= {n for n, runs in run._nested.items() if any(nested.state is not _COMPLETE for nested in runs.values())}
        if leg.going and not waiting and not self.awaits_turn:  # every unit of every node is finished, and nothing carried will be taken
            run.state, run._complete, run._earlier = _COMPLETE, frozenset(self.position), {}
        else:
            run._complete = frozenset(self.complete | {p.id for p in self.nodes if p.id not in self.complete and p.id not in waiting and self._finished(p, False)})
            for path in [path for path in run._earlier if path[0] in run._complete]:  # what a finished node was carried and didn't take, no unit of this run will
                del run._earlier[path]
            if leg.going:
                run.state = _WAITING
            elif self.embedded_as is not None:  # a nested run stands as the leg that ended early does, unless the calls it waited for finished it
                run.state = _COMPLETE if len(run._complete) == len(self.nodes) else leg.run.state
        if leg.emit:
            self.touched.update(self.position)
            self._completions()
            for placed in self.nodes:  # a node the leg left part done says how far it got
                if placed.scope and placed.id in self.progress_said:
                    self._progress(placed, last=True)
        del leg.frontiers[self.number]
        if self.embedded_as is not None:
            around, embedded, key = self.embedded_as
            del around.nested[(embedded.id, key)]
            leg.changed[around] = None
            if self.awaits_turn:  # the run around tries it again, once the instances before it are through
                around.left_pending = True
            if run.state is _COMPLETE:
                around._record(embedded, key, run._as_unit())

    def _attempt(self, placed: PlacedFunction, key: Key, kwargs: dict[str, Any], attempt: int, walking: bool = False) -> None:
        if not placed.definition.policy.threaded:
            out, error = self.run._call(placed, kwargs, self.leg.from_run)
            if not self._retry(placed, key, kwargs, attempt, error):
                if attempt:
                    self.running.pop((placed.id, key), None)
                self._record(placed, key, self._outcome(placed, key, kwargs, out, error), walking)
            return
        self.running[(placed.id, key)] = kwargs
        slots = (self.node_path, placed.id)
        self.leg.slots_waiting[slots].append((self, key, kwargs, attempt))
        self.leg._fill_slots(slots, placed)

    def _on_thread(self, placed: PlacedFunction, key: Key, kwargs: dict[str, Any], attempt: int) -> None:
        self.leg.returned.put((self, placed, key, attempt, *self.run._call(placed, kwargs, self.leg.from_run)))

    def _due(self, due: Literal["retry", "deadline"], placed: PlacedFunction, key: Key, attempt: int) -> None:
        """A timer of this run fired: the retry of an attempt, or the deadline of the attempt on a thread."""
        if due == "retry":
            if self.leg.going:
                self._attempt(placed, key, self.running[(placed.id, key)], attempt)
        elif self.attempts.get((placed.id, key)) == attempt and (placed.id, key) in self.running:  # the attempt still runs
            del self.running[(placed.id, key)]
            self._record(placed, key, make_problem("timeout", placed.id, *key, seconds=placed.definition.policy.attempt_timeout))

    def _returned(self, placed: PlacedFunction, key: Key, attempt: int, out: Any, error: BaseException | None) -> None:
        """A call of this run came back from its thread: free its slot, and keep what it came to unless its attempt
        was given up."""
        slots = (self.node_path, placed.id)
        self.leg.on_threads[slots] -= 1
        latest = self.attempts.get((placed.id, key)) == attempt
        current = latest and (placed.id, key) in self.running
        self.leg._fill_slots(slots, placed)
        if not current:
            if latest and error is None and not self.leg.going and key not in self.run._units[placed.id]:
                self.attempts.pop((placed.id, key))
                self._record(placed, key, self._outcome(placed, key, {}, out, None))  # past its timeout, but the leg is draining: keep it, never call it again
            return  # timed out while the leg runs on, or the leg stopped
        self.attempts.pop((placed.id, key))
        kwargs = self.running[(placed.id, key)]
        if not self._retry(placed, key, kwargs, attempt, error):
            del self.running[(placed.id, key)]
            self._record(placed, key, self._outcome(placed, key, kwargs, out, error))

    def _retry(self, placed: PlacedFunction, key: Key, kwargs: dict[str, Any], attempt: int, error: BaseException | None) -> bool:
        """Put a failed attempt on a timer if the policy says to try again: an ``ExternalError``, or a class ``retry_on``
        names, never a node's own ``NodeError``, an output its type refused or an ``Ask``."""
        policy, leg = placed.definition.policy, self.leg
        if error is None or attempt >= policy.retries or not leg.going:
            return False
        if not isinstance(error, ExternalError) and (isinstance(error, _NEVER_RETRIED) or not isinstance(error, policy.retry_on)):
            return False
        delay = policy.retry_delay * 2**attempt
        self.running[(placed.id, key)] = kwargs
        leg._after(delay, "retry", self, placed, key, attempt + 1)
        if leg.emit:
            cause = self._here(self._failure(placed, key, error))
            leg.emit(NodeRetry(loc=(*self.inside, placed.id, *key), attempt=attempt + 1, retries=policy.retries, delay=delay, cause=cause))
        return True

    def _outcome(self, placed: PlacedFunction, key: Key, kwargs: dict[str, Any], out: Any, error: BaseException | None) -> Unit | Problem:
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

    def _failure(self, placed: PlacedFunction, key: Key, error: BaseException) -> Problem:
        """The problem a node's exception is: its own words for a ``NodeError``; for any other, only its class, the text to the log."""
        if isinstance(error, NodeError):
            return error._as_problem_at(placed.id, *key)
        if isinstance(error, FieldValueError):
            return make_problem(error.code, placed.id, *key, reason=error.reason, **error.ctx)
        _logger.error("node %r failed for item %r", path_text((*self.inside, placed.id)), key, exc_info=error)
        if isinstance(error, placed.definition.policy.retry_on):
            return make_problem("external_failed", placed.id, *key, error=type(error).__name__)
        return make_problem("execution_failed", placed.id, *key, error=type(error).__name__)

    def _here(self, problem: Problem) -> Problem:
        """A problem placed in this run, placed by its whole path: each embedded graph's node and item in front."""
        return problem.model_copy(update={"loc": (*self.inside, *problem.loc)}) if self.inside else problem

    def _record(self, placed: PlacedNode, key: Key, unit: Unit | Problem, walking: bool = False) -> None:
        """Keep what a unit came to, and make ready what reads it."""
        run, leg = self.run, self.leg
        if isinstance(unit, Problem):
            if leg.going:  # the first failure is the cause; a later one changes nothing
                cause = self._here(unit)
                leg.end(EndedEarly("failed", cause))
                if leg.emit:
                    leg.emit(NodeFailed(loc=(*self.inside, placed.id), cause=cause))
            return
        units = run._units[placed.id]
        if units and key not in units and key < next(reversed(units)):
            run._unordered.add(placed.id)
        units[key] = unit
        self.recorded_since_walk += 1
        leg.changed[self] = None
        if leg.checkpoint_every is not None and time.monotonic() >= leg.next_checkpoint:
            leg._checkpoint()
        if placed.opens and run.max_items is not None and isinstance(unit, Done):
            for output in placed.opens:
                items = _length(unit.out[output])
                if items > run.max_items:
                    self._stop_at_ceiling(placed, output, items)
                    return
        if leg.emit:
            self.touched.add(placed.id)
            if unit is not SKIPPED_UNIT:
                self._announce(placed)  # a unit taken from earlier work, or a blank one, starts its node as a call does
            if isinstance(unit, Asking):
                leg.emit(UnitWaiting(unit=waiting_unit(self.inside, placed, key, unit)))  # type: ignore[arg-type]
            if placed.scope:
                self._progress(placed)
        if walking or not leg.going:
            return
        graph = run.graph
        for reader in graph._readers[placed.id]:
            shared = graph._shared_maps(placed, reader)
            if shared < len(key):  # a reader above this instance: tried once nothing it reads works under the one they share
                self.gathers.add((reader.id, key[:shared]))
            else:
                for instance in run._instances(reader.scope, key)[0]:
                    heapq.heappush(leg.ready, (self.number, self.position[reader.id], instance))

    def _in_turn(self, placed: PlacedNode, key: Key) -> bool:
        """Whether every unit of ``placed`` before ``key`` has been tried, so a person's answers are taken in order,
        whatever thread finished first.

        The k-th of identical units is the k-th in order among those the leg could try: one that waits on a person
        counts as tried once nothing else can start (``_pass_untried``). In a nested run, the order goes on from
        the instances of its embedded graph before this one, which share what the node may take: each is through
        (complete, or counted as tried) before a unit here takes an answer.
        """
        if self.embedded_as is not None:
            around, embedded, instance_around = self.embedded_as
            if not around._in_turn(embedded, instance_around):
                self.awaits_turn = True
                return False
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

    def _release_gathers(self) -> bool:
        """Make ready each gathering unit nothing above it is still working under; False when there is none."""
        working = [*self.running, *self.nested]
        released = False
        for node_id, key in list(self.gathers):
            above = self.run.graph._above_of(node_id)
            if not any(n in above and k[: len(key)] == key for n, k in working):
                self.gathers.discard((node_id, key))
                released = True
                position = self.position[node_id]
                placed = self.nodes[position]
                for instance in self.run._instances(placed.scope, key)[0] if len(key) < len(placed.scope) else (key,):
                    heapq.heappush(self.leg.ready, (self.number, position, instance))
        return released

    def _check_ceiling(self) -> None:
        """Stop the leg if a Series some node works through already holds more items than ``max_items``, as the run
        is opened."""
        run = self.run
        if run.max_items is None:
            return
        for placed in self.nodes:
            for output in placed.opens:
                longest = max((_length(unit.out[output]) for unit in run._units[placed.id].values() if isinstance(unit, Done)), default=0)
                if longest > run.max_items:
                    self._stop_at_ceiling(placed, output, longest)
                    return

    def _stop_at_ceiling(self, placed: PlacedNode, output: str, items: int) -> None:
        """Stop the leg at the item ceiling, unless it already ended for a failure, a cancel or its timeout, which
        stays the cause."""
        if self.leg.going:
            ceiling = make_problem("item_ceiling", placed.id, series=f"{placed.id}.{output}", items=items, max_items=self.run.max_items)
            self.leg.end(EndedEarly("stopped", self._here(ceiling)))

    # -- what the frontier says ---------------------------------------------------------------

    def _announce(self, placed: PlacedNode) -> None:
        if placed.id not in self.started:
            self.started.add(placed.id)
            if self.leg.emit:
                self.leg.emit(NodeStart(loc=(*self.inside, placed.id)))

    def _progress(self, placed: PlacedNode, last: bool = False) -> None:
        """Say how far a node running per item has got, at most four times a second; ``last`` says its final count, unless already said."""
        now = time.monotonic()
        emit = self.leg.emit
        if not emit or not (last or now - self.progress_at.get(placed.id, -1.0) >= _SECONDS_BETWEEN_PROGRESS):
            return
        done = sum(1 for unit in self.run._units[placed.id].values() if type(unit) is not Asking)  # counted only when said, at most four times a second
        if self.progress_said.get(placed.id) != done:
            self.progress_at[placed.id], self.progress_said[placed.id] = now, done
            keys, known = self.run._instances(placed.scope, ())
            emit(NodeProgress(loc=(*self.inside, placed.id), done=done, total=len(keys) if known else None))

    def _completions(self) -> None:
        """Say which touched nodes are complete: nothing they wait on is working, and every unit is done."""
        emit = self.leg.emit
        assert emit is not None
        working = {n for n, _ in self.running} | {n for n, _ in self.nested}
        for node_id in sorted(self.touched, key=self.position.__getitem__):
            if node_id in working or working & self.run.graph._above_of(node_id):
                continue
            self.touched.discard(node_id)
            placed, units = self.run.graph._nodes[node_id], self.run._units[node_id]
            if node_id in self.complete or not self._finished(placed):
                continue
            self.complete.add(node_id)
            if placed.scope and node_id in self.progress_said:
                self._progress(placed, last=True)
            if units and all(unit is SKIPPED_UNIT for unit in units.values()):
                emit(NodeSkipped(loc=(*self.inside, node_id)))
            else:
                not_called = sum(type(unit) is Blank for unit in units.values())
                emit(NodeComplete(loc=(*self.inside, node_id), units=len(units), not_called=not_called, reused_units=self.reused[node_id]))

    def _finished(self, placed: PlacedNode, may_wait: bool = True) -> bool:
        """Whether every unit of ``placed`` is known and none of them is still to run or waiting; ``may_wait`` False
        when the caller already knows none waits."""
        units = self.run._units[placed.id]
        if not placed.scope:
            for unit in units.values():
                return type(unit) is not Asking
            return False
        if may_wait and Asking in {type(unit) for unit in units.values()}:
            return False
        counted, known = self.run._instances(placed.scope, (), counted=True)
        return known and len(units) == (counted[0][0] if counted else 0)
