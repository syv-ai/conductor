"""One leg of a run: start every unit whose inputs exist, record what comes of each, and say so as events.

The leg first walks the nodes in order and tries every row that exists, as a
later leg would. A node whose policy has ``concurrency`` 1 and no timeout is
called on the spot; any other is handed to threads, at most ``concurrency``
of its rows at once, and the walk goes on. When a call comes back, the units
that read it are tried at once: the reader's rows under, at, or above the
unit's row. So row 1 can finish the chain while row 10 is still being made.

A failure the node's ``Policy`` may retry waits its delay on a timer, and the
leg goes on meanwhile; so does an attempt with a timeout, whose clock starts
with its thread. The first failure that is final stops the leg from starting
units, and the calls already on threads are waited for and kept, so a resume
never makes them twice.

A unit that gathers rows (a ``Series`` or ``Table`` input fed from deeper
rows) is tried only once nothing it waits on is still working under its row,
so it is tried once and not once per row it reads. When nothing is working,
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
from typing import TYPE_CHECKING, Annotated, Any, Literal

from pydantic import ConfigDict, Field
from pydantic.json_schema import SkipJsonSchema

from conductor._internal.ask import Ask, Waiting, WrongAsk, asked
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import ExternalFailure, NodeError, Problem, make_problem
from conductor._internal.fields import SKIPPED, Refused
from conductor._internal.graph import shared_depth
from conductor._internal.run import (
    PENDING,
    Key,
    Run,
    RunStatus,
    SkippedAt,
    Unit,
    _length,
    jsonable,
)

if TYPE_CHECKING:
    from conductor._internal.graph import PlacedNode

#: The least time between two ``NodeProgress`` of one node.
_SECONDS_BETWEEN_PROGRESS = 0.25

_logger = logging.getLogger("conductor")


# -- events -------------------------------------------------------------------------------------


class _EventBase(ConductorModel):
    """Every event: its schema marks ``type`` and every field a dump always writes as required, so a generated client can rely on them."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True, arbitrary_types_allowed=True, ser_json_inf_nan="strings")


class RunStart(_EventBase):
    """A leg began; ``waiting`` is the rows already waiting for a person."""

    type: Literal["run_start"] = "run_start"
    waiting: tuple[Waiting, ...] = ()


class NodeStart(_EventBase):
    """A node's first unit started; never again for that node, on this or a later leg."""

    type: Literal["node_start"] = "node_start"
    node_id: str


class NodeProgress(_EventBase):
    """How many rows of a node are done, at most four times a second; ``total`` ``None`` until its rows are known."""

    type: Literal["node_progress"] = "node_progress"
    node_id: str
    done: int
    total: int | None


class NodeComplete(_EventBase):
    """Every row of a node is done. ``outputs`` is as JSON: one value, or a list in row order.

    ``not_called`` is each row a blank input kept the node from, with that
    input; ``reused_units`` how many rows ``run.again`` took from the earlier run.
    """

    type: Literal["node_complete"] = "node_complete"
    node_id: str
    outputs: dict[str, Any]
    not_called: list[dict[str, Any]] = Field(default_factory=list)
    reused_units: int = 0


class NodeSkipped(_EventBase):
    """Every row of a node was a branch not taken."""

    type: Literal["node_skipped"] = "node_skipped"
    node_id: str


class NodeFailed(_EventBase):
    """A unit of the node failed, finally; ``cause`` says why, at ``(node_id, *row)``."""

    type: Literal["node_failed"] = "node_failed"
    node_id: str
    cause: Problem


class NodeRetry(_EventBase):
    """An attempt failed on the outside world and is tried again: attempt ``attempt`` of ``retries``, on ``row``, after ``delay`` seconds."""

    type: Literal["node_retry"] = "node_retry"
    node_id: str
    row: tuple[int, ...]
    attempt: int
    retries: int
    delay: float
    cause: Problem


class UnitWaiting(_EventBase):
    """A row started waiting for a person. The row waits, whatever reads it waits, and the rest of the graph runs on."""

    type: Literal["unit_waiting"] = "unit_waiting"
    unit: Waiting


class RunEnd(_EventBase):
    """The leg ended, with ``status``. ``run`` is the run as the leg left it; it stays out of what the event writes, as every value already went out on its own event."""

    type: Literal["run_end"] = "run_end"
    status: RunStatus
    run: SkipJsonSchema[Run] = Field(exclude=True)


#: Any event, told apart by ``type``.
Event = Annotated[
    RunStart | NodeStart | NodeProgress | NodeComplete | NodeSkipped | NodeFailed | NodeRetry | UnitWaiting | RunEnd,
    Field(discriminator="type"),
]


# -- the leg ------------------------------------------------------------------------------------


class Leg:
    """One leg over ``run``, which it changes in place. ``emit`` is None when nobody listens.

    ``stop`` is asked between units, and says why the leg must end now (a cancel, the deadline), or None.
    """

    def __init__(self, run: Run, emit: Callable[[_EventBase], None] | None, stop: Callable[[], Problem | None] | None, from_run: dict[type, Any]) -> None:
        self.run, self.emit, self.stop, self.from_run = run, emit, stop, from_run
        self.nodes = list(run.graph._nodes.values())
        self.position = {p.id: i for i, p in enumerate(self.nodes)}
        self.ready: list[tuple[int, Key]] = []  # (node position, row), tried in that order
        self.gathers: set[tuple[str, Key]] = set()
        self.running: dict[tuple[str, Key], dict[str, Any]] = {}  # units handed to threads, and their inputs
        self.slots_waiting: dict[str, deque[tuple[Key, dict[str, Any], int]]] = {p.id: deque() for p in self.nodes}
        self.on_threads = {p.id: 0 for p in self.nodes}
        self.returned: queue.SimpleQueue[tuple[str, Key, int, Any, BaseException | None]] = queue.SimpleQueue()
        self.timers: list[tuple[float, int, str, Key, int]] = []  # (when, order, node, row, attempt): a retry due, or an attempt's deadline
        self.attempts: dict[tuple[str, Key], int] = {}  # the attempt each unit on a thread is on
        self.pool: ThreadPoolExecutor | None = None
        self.made = run._rows_made()
        self.shared: dict[tuple[str, str], int] = {}
        #: For a node taking earlier work, its rows in order, each row's place, and how many from the first have been tried.
        self.turns: dict[str, list[Any]] = {}
        self.recorded_since_walk = 1
        self.left_pending = True
        # what only a listener needs
        self.started = {run.graph.outward(n) for n, units in run._units.items() if units}
        #: For an embedding, what each of its nodes came to once complete: all skipped, the rows not called, the rows reused.
        self.inside: dict[str, dict[str, tuple[bool, list[dict[str, Any]], int]]] = {}
        self.touched: set[str] = set()
        self.complete: set[str] = set()
        self.reused = {p.id: 0 for p in self.nodes}
        self.progress_at: dict[str, float] = {}
        self.progress_said: dict[str, int] = {}

    def go(self) -> None:
        run = self.run
        run.status, run.cause = "running", None
        if self.emit:
            self.emit(RunStart(waiting=tuple(run.waiting)))
        try:
            if not self._over_ceiling():
                self._loop()
            while self.running and self.run.status != "running":  # calls already made had their effect: keep them, as long as their timeouts allow
                self._wait()
        finally:
            if self.pool is not None:
                self.pool.shutdown(wait=False, cancel_futures=True)
        if run.status == "running":
            run.status = "waiting" if any(u.state == "waiting" for us in run._units.values() for u in us.values()) else "complete"
        if self.emit:
            self.touched.update(self.position)
            self._completions()
            for placed in self.nodes:  # a node the leg left part done says how far it got
                if placed.level and placed.id in self.progress_said:
                    self._progress(placed, last=True)
            self.emit(RunEnd(status=run.status, run=run))

    def _loop(self) -> None:
        while self.run.status == "running":
            if self.stop is not None and (stopped := self.stop()) is not None:
                self.run.status, self.run.cause = stopped.type.removeprefix("run_"), stopped  # type: ignore[assignment]
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
                elif self.left_pending and self.recorded_since_walk:
                    self._walk()
                else:
                    return

    def _walk(self) -> None:
        """Try every row of every node, in order: the first look, and the last one when nothing runs."""
        self.left_pending, self.recorded_since_walk = False, 0
        units = self.run._units
        for placed in self.nodes:
            for key in self.run._rows(placed.level, ())[0]:
                if key not in units[placed.id] and (placed.id, key) not in self.running:
                    if not self._try(placed, key, walking=True):
                        self.left_pending = True
                    if self.run.status != "running" or (self.stop is not None and self.stop() is not None):
                        return

    def _try(self, placed: PlacedNode, key: Key, walking: bool = False) -> bool:
        """Start ``placed`` on ``key`` if its inputs exist; False if they don't yet."""
        run = self.run
        if not walking and (key in run._units[placed.id] or (placed.id, key) in self.running):
            return True
        kwargs = run._inputs(placed, key)
        if kwargs is PENDING:
            return False
        if isinstance(kwargs, SkippedAt):
            self._record(placed, key, Unit("skipped", question={"from": kwargs.depth}), walking)
        elif isinstance(kwargs, str):
            self._record(placed, key, Unit("blank", question={"input": kwargs}), walking)
        elif run._earlier and placed.definition.policy.reuse_when_inputs_match and (
            earlier := run._reused(placed, key, kwargs, lambda: self._in_turn(placed, key))
        ) is not None:
            if earlier is PENDING:
                return False  # a row before it may take this earlier unit; the walk tries it again
            self.reused[placed.id] += 1
            self._record(placed, key, earlier, walking)
        else:
            if self.emit:
                self._announce(placed)
            policy = placed.definition.policy
            if policy.threaded or policy.retries:
                self._attempt(placed, key, kwargs, 0, walking)
            else:
                self._record(placed, key, self._outcome(placed, key, kwargs, *run._call(placed, kwargs, self.from_run)), walking)
        return True

    def _attempt(self, placed: PlacedNode, key: Key, kwargs: dict[str, Any], attempt: int, walking: bool = False) -> None:
        if not placed.definition.policy.threaded:
            out, error = self.run._call(placed, kwargs, self.from_run)
            if error is None or isinstance(error, Ask) or not self._retry(placed, key, kwargs, attempt, error):
                if attempt:
                    self.running.pop((placed.id, key), None)
                self._record(placed, key, self._outcome(placed, key, kwargs, out, error), walking)
            return
        self.running[(placed.id, key)] = kwargs
        self.slots_waiting[placed.id].append((key, kwargs, attempt))
        self._fill_slots(placed)

    def _fill_slots(self, placed: PlacedNode) -> None:
        waiting, policy = self.slots_waiting[placed.id], placed.definition.policy
        while waiting and self.on_threads[placed.id] < policy.concurrency and self.run.status == "running":
            key, kwargs, attempt = waiting.popleft()
            self.on_threads[placed.id] += 1
            self.attempts[(placed.id, key)] = attempt
            if self.pool is None:
                self.pool = ThreadPoolExecutor(sum(p.definition.policy.concurrency for p in self.nodes if p.definition.policy.threaded), "conductor")
            if policy.attempt_timeout is not None:
                heapq.heappush(self.timers, (time.monotonic() + policy.attempt_timeout, len(self.timers), placed.id, key, -1 - attempt))
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
            _, _, node_id, key, attempt = heapq.heappop(self.timers)
            placed = self.run.graph._nodes[node_id]
            if attempt >= 0:  # a retry is due
                if self.run.status == "running":
                    self._attempt(placed, key, self.running[(node_id, key)], attempt)
            elif self.attempts.get((node_id, key)) == -1 - attempt and (node_id, key) in self.running:  # the attempt still runs
                del self.running[(node_id, key)]
                seconds = placed.definition.policy.attempt_timeout
                self._record(placed, key, make_problem("timeout", self.run.graph.outward(node_id), *key, seconds=seconds))

    def _returned(self, node_id: str, key: Key, attempt: int, out: Any, error: BaseException | None) -> None:
        placed = self.run.graph._nodes[node_id]
        self.on_threads[node_id] -= 1
        current = self.attempts.get((node_id, key)) == attempt and (node_id, key) in self.running
        self._fill_slots(placed)
        if not current:
            return  # timed out, or the leg stopped
        self.attempts.pop((node_id, key))
        kwargs = self.running[(node_id, key)]
        if not self._retry(placed, key, kwargs, attempt, error):
            del self.running[(node_id, key)]
            self._record(placed, key, self._outcome(placed, key, kwargs, out, error))

    def _retry(self, placed: PlacedNode, key: Key, kwargs: dict[str, Any], attempt: int, error: BaseException | None) -> bool:
        """Put a failed attempt on a timer if the policy says to try again."""
        policy = placed.definition.policy
        if error is None or attempt >= policy.retries or self.run.status != "running" or not isinstance(error, (ExternalFailure, *policy.retry_on)):
            return False
        delay = policy.retry_delay * 2**attempt
        self.running[(placed.id, key)] = kwargs
        heapq.heappush(self.timers, (time.monotonic() + delay, len(self.timers), placed.id, key, attempt + 1))
        if self.emit:
            self.emit(NodeRetry(node_id=self.run.graph.outward(placed.id), row=key, attempt=attempt + 1, retries=policy.retries, delay=delay, cause=self._failure(placed, key, error)))
        return True

    def _outcome(self, placed: PlacedNode, key: Key, kwargs: dict[str, Any], out: Any, error: BaseException | None) -> Unit | Problem:
        if error is None:
            return Unit("skipped") if out is SKIPPED else Unit("done", out)
        if isinstance(error, Ask):
            question = {"prompt": error.prompt, "inputs": jsonable(kwargs)}
            if error.given:
                try:
                    question["asked"] = asked(error, placed.definition)
                except WrongAsk as wrong:
                    return make_problem("invalid_ask", self.run.graph.outward(placed.id), *key, reason=str(wrong))
            return Unit("waiting", None, question)
        return self._failure(placed, key, error)

    def _failure(self, placed: PlacedNode, key: Key, error: BaseException) -> Problem:
        """The problem a node's exception is: its own words for a ``NodeError``; for any other, only its class, the text to the log."""
        outer = self.run.graph.outward(placed.id)  # a node of an embedding fails as the embedding, from outside
        if isinstance(error, NodeError):
            return error._as_problem_at(outer, *key)
        if isinstance(error, Refused):
            return make_problem(error.code, outer, *key, reason=error.reason, **error.ctx)
        _logger.error("node %r failed on row %r", placed.id, key, exc_info=error)
        if isinstance(error, placed.definition.policy.retry_on):
            return make_problem("external_failed", outer, *key, error=type(error).__name__)
        return make_problem("execution_failed", outer, *key, error=type(error).__name__)

    def _record(self, placed: PlacedNode, key: Key, unit: Unit | Problem, walking: bool = False) -> None:
        """Keep what a unit came to, and make ready what reads it."""
        run = self.run
        if isinstance(unit, Problem):
            if run.cause is None:
                run.status, run.cause = "failed", unit
                self._start_nothing_more()
                if self.emit:
                    self.emit(NodeFailed(node_id=self.run.graph.outward(placed.id), cause=unit))
            return
        units = run._units[placed.id]
        if units and key not in units and key < next(reversed(units)):
            run._unordered.add(placed.id)
        units[key] = unit
        self.recorded_since_walk += 1
        if unit.state == "done" and placed.makes:
            for output, level in placed.makes.items():
                self.made[level[-1]] = self.made.get(level[-1], 0) + _length(unit.out[output])  # type: ignore[index]
            if self._over_ceiling():
                return
        if self.emit:
            self.touched.add(placed.id)
            if unit.state == "waiting":
                self._announce(placed)
                self.emit(UnitWaiting(unit=run._waiting(placed.id, key, unit)))
            if placed.level:
                self._progress(placed)
        if walking or run.status != "running":
            return
        for reader in run.graph._readers[placed.id]:
            shared = self._shared(placed, reader)
            if shared < len(key):  # a reader above this row, or beside it: tried once nothing it reads works under the row they share
                self.gathers.add((reader.id, key[:shared]))
            else:
                for row in run._rows(reader.level, key)[0]:
                    heapq.heappush(self.ready, (self.position[reader.id], row))

    def _in_turn(self, placed: PlacedNode, key: Key) -> bool:
        """Whether every row of ``placed`` before ``key`` has been tried, so identical rows take earlier work in row order, whatever thread finished first."""
        turn = self.turns.get(placed.id)
        if turn is None or key not in turn[1]:
            order = self.run._rows(placed.level, ())[0]
            turn = self.turns[placed.id] = [order, {k: i for i, k in enumerate(order)}, 0]
        order, index, reached = turn
        units = self.run._units[placed.id]
        while reached < len(order) and (order[reached] in units or (placed.id, order[reached]) in self.running):
            reached += 1
        turn[2] = reached
        return index.get(key, 0) <= reached

    def _shared(self, placed: PlacedNode, reader: PlacedNode) -> int:
        """How many tables the rows of ``placed`` and of a node reading it share from the graph's own row down."""
        found = self.shared.get((placed.id, reader.id))
        if found is None:
            found = self.shared[(placed.id, reader.id)] = shared_depth(placed.level, reader.level)
        return found

    def _start_nothing_more(self) -> None:
        """Drop the units not yet on a thread and the retries not yet due; the calls on threads are waited for and kept."""
        for node_id, waiting in self.slots_waiting.items():
            for k, _, _ in waiting:
                self.running.pop((node_id, k), None)
            waiting.clear()
        for _, _, node_id, k, attempt in self.timers:
            if attempt >= 0:
                self.running.pop((node_id, k), None)

    def _check_gathers(self) -> None:
        """Make ready each gathering unit nothing above it is still working under; wait if none is."""
        busy = list(self.running)
        for node_id, key in list(self.gathers):
            above = self.run.graph._above_of(node_id)
            if not self.running or not any(n in above and k[: len(key)] == key for n, k in busy):
                self.gathers.discard((node_id, key))
                placed = self.nodes[self.position[node_id]]
                for row in self.run._rows(placed.level, key)[0] if len(key) < len(placed.level) else (key,):
                    heapq.heappush(self.ready, (self.position[node_id], row))
        if self.gathers and not self.ready:
            self._wait()

    def _over_ceiling(self) -> bool:
        limit = self.run.max_rows
        if limit is not None:
            for table, count in self.made.items():
                if count > limit:
                    self.run._stop(table, count)
                    return True
        return False

    # -- what the leg says --------------------------------------------------------------------

    def _announce(self, placed: PlacedNode) -> None:
        outer = self.run.graph.outward(placed.id)
        if outer not in self.started:
            self.started.add(outer)
            if self.emit:
                self.emit(NodeStart(node_id=outer))

    def _progress(self, placed: PlacedNode, last: bool = False) -> None:
        """Say how far a node running per row has got, at most four times a second; ``last`` says its final count, unless already said."""
        now = time.monotonic()
        if placed.id in self.run.graph._spliced:
            return  # an embedding's own nodes are not seen from outside; its completion is
        if not self.emit or not (last or now - self.progress_at.get(placed.id, -1.0) >= _SECONDS_BETWEEN_PROGRESS):
            return
        done = sum(1 for unit in self.run._units[placed.id].values() if unit.state != "waiting")  # counted only when said, at most four times a second
        if self.progress_said.get(placed.id) != done:
            self.progress_at[placed.id], self.progress_said[placed.id] = now, done
            keys, known = self.run._rows(placed.level, ())
            self.emit(NodeProgress(node_id=placed.id, done=done, total=len(keys) if known else None))

    def _completions(self) -> None:
        """Say which touched nodes are complete: nothing they wait on is working, and every row is done."""
        assert self.emit is not None
        running = {n for n, _ in self.running}
        for node_id in sorted(self.touched, key=self.position.__getitem__):
            if node_id in running or running & self.run.graph._above_of(node_id):
                continue
            self.touched.discard(node_id)
            placed, units = self.run.graph._nodes[node_id], self.run._units[node_id]
            keys, known = self.run._rows(placed.level, ())
            if node_id in self.complete or not known or any(units.get(k) is None or units[k].state == "waiting" for k in keys):
                continue
            self.complete.add(node_id)
            if placed.level and node_id in self.progress_said:
                self._progress(placed, last=True)
            skipped = bool(keys) and all(units[k].state == "skipped" for k in keys)
            not_called = [{"row": list(k), "input": units[k].question["input"]} for k in keys if units[k].state == "blank" and units[k].question]  # type: ignore[index]
            if node_id in self.run.graph._spliced:
                self._inner_complete(node_id, skipped, not_called)
            elif skipped:
                self.emit(NodeSkipped(node_id=node_id))
            else:
                self.emit(NodeComplete(node_id=node_id, outputs=self.run._outputs_json(placed), not_called=not_called, reused_units=self.reused[node_id]))

    def _inner_complete(self, node_id: str, skipped: bool, not_called: list[dict[str, Any]]) -> None:
        """A node of an embedding is complete: once all of them are, the embedding is, as one node with the outputs it offers."""
        assert self.emit is not None
        graph = self.run.graph
        outer = graph.outward(node_id)
        inside = self.inside.setdefault(outer, {})
        inside[node_id] = (skipped, [{**n, "input": graph.outward(f"{node_id}.{n['input']}").partition(".")[2] or n["input"]} for n in not_called], self.reused[node_id])
        if len(inside) < len(graph._placed_as(outer)):
            return
        if all(s for s, _, _ in inside.values()):
            self.emit(NodeSkipped(node_id=outer))
            return
        outputs = {name: self.run._outputs_json(graph._nodes[n])[o] for name, (n, o) in graph._readable[outer].items()}
        self.emit(NodeComplete(node_id=outer, outputs=outputs, not_called=[n for _, ns, _ in inside.values() for n in ns], reused_units=sum(r for _, _, r in inside.values())))
