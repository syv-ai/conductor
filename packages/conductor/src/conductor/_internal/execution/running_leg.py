"""One leg, start to finish: the loop that starts every ready unit, and each unit's run.

The engine doesn't schedule whole nodes. It schedules *units*: one call of a
node's ``run``, a node on one row. A node that runs once is one unit, on the
empty row ``()``; a node that runs per row of a series is one unit per row, so
row 3 can move on down the graph while row 10 is still waiting its turn.

A *leg* is one call of ``CompiledGraph.run`` or ``stream``: it starts every
unit it can, and stops when nothing more can start. The loop asks the run's
ledger (``conductor._internal.execution.run_ledger``) which units are ready
and queues each on its node; a unit is started when its node has a free
concurrency slot, and each record answers with the units it made ready.

**One coroutine does all of it.** The event stream is the loop: it starts
units and records what comes of them, one thing at a time, synchronously, and
the ledger is touched nowhere else. Everything from outside is posted to the
loop's queue: a thread's outcome, a timer's firing (a timeout, a due retry),
the host's cancel, the leg's deadline. What is posted is handled before the
next unit starts, so a failure is seen before anything more begins.

**The leg owns its work.** Every call of a node (``call_node``: its inputs
checked, its ``run``, what it returned checked) happens in a thread the leg
owns, made when every thread so far is busy, so a unit never waits for a
thread and a node's timeout counts only the time its thread ran. Leaving the
``async with`` of a stream stops every unit before the block is gone. What a
leg cannot do is interrupt a thread: a ``run`` that has started finishes on
its own and what it returns is dropped if the leg has moved on. A timed-out
attempt is therefore final, and the thread keeps the node's concurrency slot
until it returns.

**The event loop stays the host's.** A unit that needs no thread (a skip, a
reused result) would never let the event loop go on its own, so the leg hands
it back whenever it has held it for a few milliseconds.

**A run has legs.** A unit whose node returned ``AskPerson`` is neither done
nor failed: it waits, whatever reads it waits, and the rest of the graph runs
on. Nothing resumes: the next leg is an ordinary run over a ledger that
already holds what earlier legs produced, and a node the graph has changed
since, or that reads one, runs again, except on a row whose inputs match a row
it ran on before: that row's result is used, and the node isn't called.

**A leg can save itself while it runs.** Asked for checkpoints, the loop
compares a monotonic clock each time a unit finishes and, when the interval has
passed, emits the state as ``RunCheckpoint``. Only a finished unit changes the
state, so no timer is needed. Building the state runs on the event loop and
costs a pass over every value so far; the clock restarts once the consumer
comes back for the next event, so a slow save cannot crowd out the leg.

**Progress is paced.** A node running per row reports at most every quarter
second, whatever its rows cost, and its count is reported once more when the
node completes or the leg ends, so a reader always has the last one.

**Every leg ends the same way:** one ``RunEnd`` carrying the state. A failed
unit fails the leg: nothing more starts, the calls already on a thread are
waited for and recorded, as far as each one's attempt timeout allows, since
they have had their effect and a later leg must not repeat them; then the leg
ends, with the cause and its row on ``state.failure``. A node with no timeout
that never returns holds a failed leg open as it holds any other. A cancel or
the leg's deadline ends the leg at once: what threads have already handed back
is recorded, and a call still on a thread is not.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from conductor._internal.errors import InvalidCallError, Problem, make_problem
from conductor._internal.execution.call_node import CallFailed, call_node
from conductor._internal.execution.events import (
    Event,
    NodeFailed,
    NodeProgress,
    NodeRetry,
    NodeStart,
    RunCheckpoint,
    RunEnd,
    RunStart,
    UnitWaiting,
)
from conductor._internal.execution.run_ledger import (
    CallTheNode,
    CouldNotReadInputs,
    FinishedWithoutACall,
    RunLedger,
    SkippedWithoutACall,
    Unit,
    UnitRecorded,
    WaitsOnAPersonAgain,
)
from conductor._internal.execution.run_state import RunState, WaitingUnit

#: The least time between two ``NodeProgress`` of one node. A person can't read faster, and a host may write each to a database.
_SECONDS_BETWEEN_PROGRESS = 0.25
#: The longest the leg holds the event loop before handing it back, so a host's other coroutines keep running beside it.
_SECONDS_HOLDING_THE_EVENT_LOOP = 0.005

logger = logging.getLogger("conductor")

if TYPE_CHECKING:
    from conductor._internal.graph.compiled_graph import CompiledGraph
    from conductor._internal.graph.compiled_node import CompiledNode
    from conductor._internal.node_definition import Policy

# -- what is posted to the loop --------------------------------------------------


@dataclass(slots=True, eq=False)
class _CallOnAThread:
    """One attempt handed to a thread: what the thread needs, and the timer on its attempt."""

    unit: Unit
    attempt: int
    node: CompiledNode
    inputs: dict[str, Any]
    input_hash: str
    attempt_timer: asyncio.TimerHandle | None = None


#: A handler, the unit it's about, and the handler's other arguments; ``None`` only wakes the loop to look at the cancel or the deadline.
_Posted = tuple[Callable[..., None], Unit, Any] | None


# -- the leg ---------------------------------------------------------------------


class RunningLeg:
    """One leg in flight: the coroutine that starts ready units and records what comes of them.

    What ``CompiledGraph.stream`` returns. Building one makes every check a
    start needs, so a start that can't go raises at the call. It is an async context
    manager: entering it arms the watch on the host's cancel event and the
    leg's deadline and gives the event stream; leaving it — however the leg
    ended, the consumer dropping out included — tells the threads to take no
    more and drops the timers, so nothing outlives the ``async with``.

    What the leg's coroutine keeps: the ledger, the attempts that have a slot
    and are about to start (``startable``), the calls on threads whose outcome
    it still wants (``calls_on_threads``), the retries waiting for their timer
    (``retries_waiting``), the queue everything is posted to (``posted``), and
    the events not yet given to the consumer (``out``). The call chain is
    ``_run_loop`` → ``_start_attempt`` (``ledger.start_unit``) → a thread
    (``_take_calls``, ``call_node``, the whole call). Nothing reaches a node from here
    that its signature does not name.

    A node holds at most its policy's concurrency of attempts at once: the
    rest wait, as ``(unit, attempt)``, in its ``ready_units``. Threads are made as
    calls need them, so an idle graph costs nothing and no call queues for
    one. On the way out the threads take no more calls; a thread mid-run
    finishes on its own.
    """

    started_at: float
    #: When the last ``RunCheckpoint`` went out and its consumer came back for
    #: more; when the leg began, before the first.
    last_checkpoint_at: float

    def __init__(
        self,
        compiled: CompiledGraph,
        *,
        state: RunState | None,
        from_run: Mapping[type, Any],
        timeout: float | None,
        cancel: asyncio.Event,
        checkpoint_every: float | None = None,
    ) -> None:
        self.compiled = compiled
        if state is not None and not isinstance(state, RunState):
            raise InvalidCallError([make_problem("state_unreadable", "state", reason=f"A state is a RunState, not a {type(state).__name__}.")])
        self.ledger = RunLedger(compiled) if state is None else RunLedger.restore(compiled, state)
        self.changed_nodes = [] if state is None else compiled._nodes_a_leg_runs_again(state)
        self.free_slots: dict[str, int] = {}
        problems = [make_problem("from_run_not_a_type", "from_run", repr(key), key=repr(key)) for key in from_run if not isinstance(key, type)]
        for node_id in compiled.execution_order:
            version = compiled.node(node_id).version
            problems += [
                make_problem("from_run_missing", "from_run", needed.__name__, node=node_id, parameter=name, needs=needed.__name__)
                for name, needed in version.from_run.items() if needed not in from_run
            ]
            self.free_slots[node_id] = version.policy.concurrency
        if problems:
            raise InvalidCallError(problems)
        self.from_run = dict(from_run)
        self.timeout = timeout
        self.cancel = cancel
        self.checkpoint_every = checkpoint_every
        self.nodes_announced_started = self.ledger.completed_nodes()
        self.runs_per_row_of = compiled._engine_lookups.runs_per_row_of
        self.calls: queue.SimpleQueue[_CallOnAThread | None] = queue.SimpleQueue()
        self.calls_in_threads = 0
        self.threads_made = 0
        self.stopped = False
        self.ready_units: dict[str, deque[tuple[Unit, int]]] = {node_id: deque() for node_id in self.free_slots}
        #: Attempts holding a slot of their node, about to start, in the order they got one.
        self.startable: deque[tuple[Callable[..., None], Unit, int]] = deque()
        self.calls_on_threads: set[_CallOnAThread] = set()
        self.retries_waiting: dict[Unit, asyncio.TimerHandle] = {}
        self.posted: asyncio.Queue[_Posted] = asyncio.Queue()
        self.out: deque[Event] = deque()
        #: Cleared by the first failure, a cancel or the deadline: nothing starts after that.
        self.units_may_start = True
        self.failure: Problem | None = None
        self.timed_out_after: float | None = None
        #: A unit finished since the last ``RunCheckpoint``; only that changes the state.
        self.a_unit_finished = False
        #: When the last ``NodeProgress`` of each node went out, and the nodes with finished rows that none has counted yet.
        self.progress_sent_at: dict[str, float] = {}
        self.nodes_with_unsent_progress: set[str] = set()

    def __aiter__(self) -> None:
        raise TypeError("a stream is read with: async with compiled.stream() as events: async for event in events: ...")

    async def __aenter__(self) -> AsyncGenerator[Event, None]:
        self.started_at = self.last_checkpoint_at = time.monotonic()
        self._loop = asyncio.get_running_loop()
        self._cancel_watch = asyncio.ensure_future(self._report_cancel())
        self._deadline = None if self.timeout is None else self._loop.call_later(self.timeout, self._report_deadline)
        self._stream = self._run_loop()
        return self._stream

    async def __aexit__(self, *exc: object) -> None:
        """Let the threads go and drop the timers.

        The threads are not waited for: one mid-run finishes on its own, what
        it returns is dropped, and a call that never started is not started."""
        await self._stream.aclose()
        self.stopped = True
        self._cancel_watch.cancel()
        await asyncio.gather(self._cancel_watch, return_exceptions=True)
        if self._deadline is not None:
            self._deadline.cancel()
        for call in self.calls_on_threads:
            if call.attempt_timer is not None:
                call.attempt_timer.cancel()
        for timer in self.retries_waiting.values():
            timer.cancel()
        for _ in range(self.threads_made):
            self.calls.put(None)

    async def _report_cancel(self) -> None:
        """Wake the loop when the host sets its cancel event: the event has no callback of its own."""
        await self.cancel.wait()
        self.posted.put_nowait(None)

    def _report_deadline(self) -> None:
        """The leg's deadline passed: the loop stops at its next look, and this wakes it."""
        self.timed_out_after = self.timeout
        self.posted.put_nowait(None)

    # -- the loop ----------------------------------------------------------------

    async def _run_loop(self) -> AsyncGenerator[Event, None]:
        """Say what this leg starts from, then handle what is posted and start what can start until the leg ends."""
        ledger = self.ledger
        yield RunStart(
            reused_nodes=[node_id for node_id in self.compiled.execution_order if node_id in self.nodes_announced_started],
            changed_nodes=self.changed_nodes,
            waiting=ledger.waiting(),
        )
        held_since = time.monotonic()
        self._queue_units(ledger.units_ready_to_start())
        out, posted, startable = self.out, self.posted, self.startable
        stopped_early = False
        while startable or self.calls_on_threads or self.retries_waiting:
            if self.cancel.is_set() or self.timed_out_after is not None:
                stopped_early = True
                break
            if self.a_unit_finished and self.checkpoint_every is not None and self.units_may_start and time.monotonic() - self.last_checkpoint_at >= self.checkpoint_every:
                self.a_unit_finished = False
                yield RunCheckpoint(state=ledger.state("running"))
                self.last_checkpoint_at = time.monotonic()
            if not posted.empty():
                self._handle(posted.get_nowait())
            elif startable:
                self._handle(startable.popleft())
            else:
                message = await posted.get()
                held_since = time.monotonic()
                self._handle(message)
            held_long_enough = time.monotonic() - held_since >= _SECONDS_HOLDING_THE_EVENT_LOOP
            if out and (held_long_enough or not startable):  # a consumer that takes its time doesn't hold up units about to start
                while out:
                    yield out.popleft()
            if held_long_enough:
                await asyncio.sleep(0)
                held_since = time.monotonic()
        await asyncio.sleep(0)  # the state is the longest single stretch on the event loop; and what a thread handed back before a stop reaches the queue
        if stopped_early:
            self._stop_starting_units()
            while not posted.empty():
                self._handle(posted.get_nowait())
            while out:
                yield out.popleft()
        ending = self._ending_state(stopped_early)
        for node_id in sorted(self.nodes_with_unsent_progress):
            yield self._progress_of(node_id)
        yield RunEnd(state=ending)

    def _ending_state(self, stopped_early: bool) -> RunState:
        """The state the leg ends with: the first failure stands; a stop says which it was, a cancel over a deadline."""
        ledger = self.ledger
        if self.failure is not None:
            return ledger.state("failed", self.failure)
        if not stopped_early:
            return ledger.state("waiting" if ledger.waiting() else "complete")
        if self.cancel.is_set():
            return ledger.state("cancelled", make_problem("run_cancelled"))
        assert self.timed_out_after is not None
        return ledger.state("timed_out", make_problem("run_timed_out", elapsed_seconds=time.monotonic() - self.started_at, timeout_seconds=self.timed_out_after))

    def _handle(self, message: _Posted) -> None:
        """Do what this message says. An exception out of a handler is a defect in the
        engine, the unit's ``engine_error``: a handler that died silently would leave the loop waiting."""
        if message is None:
            return
        handler, unit, *details = message
        try:
            handler(unit, *details)
        except Exception as raised:
            self._fail_unit(unit, _engine_error(unit, raised))

    def _queue_units(self, units: list[Unit]) -> None:
        for unit in units:
            self._queue_attempt(unit, 0)

    def _queue_attempt(self, unit: Unit, attempt: int) -> None:
        if not self.units_may_start:
            return
        node_id, _ = unit
        self.ready_units[node_id].append((unit, attempt))
        self._fill_free_slots(node_id)

    def _fill_free_slots(self, node_id: str) -> None:
        """Make startable as many of this node's waiting attempts as it has free slots."""
        ready = self.ready_units[node_id]
        while ready and self.free_slots[node_id]:
            unit, attempt = ready.popleft()
            self.free_slots[node_id] -= 1
            self.startable.append((self._start_attempt, unit, attempt))

    def _free_slot(self, node_id: str) -> None:
        self.free_slots[node_id] += 1
        self._fill_free_slots(node_id)

    def _stop_starting_units(self) -> None:
        self.units_may_start = False
        self.startable.clear()
        for ready in self.ready_units.values():
            ready.clear()
        for timer in self.retries_waiting.values():
            timer.cancel()
        self.retries_waiting.clear()

    # -- one unit ----------------------------------------------------------------

    def _start_attempt(self, unit: Unit, attempt: int) -> None:
        """This attempt's turn: hand it to a thread, or finish it here if it needs no call."""
        node_id, _ = unit
        match self.ledger.start_unit(unit):
            case CallTheNode(inputs, input_hash):
                self._announce_node_started(node_id)
                self._hand_to_a_thread(unit, attempt, inputs, input_hash)
            case SkippedWithoutACall(recorded):
                self._free_slot(node_id)
                self._report_recorded(unit, recorded)
            case FinishedWithoutACall(recorded):
                self._announce_node_started(node_id)
                self._free_slot(node_id)
                self._report_recorded(unit, recorded)
            case WaitsOnAPersonAgain(waiting_unit):
                self._announce_node_started(node_id)
                self._free_slot(node_id)
                self._report_waiting(waiting_unit)
            case CouldNotReadInputs(problem):
                self._fail_unit(unit, problem)

    def _announce_node_started(self, node_id: str) -> None:
        if node_id not in self.nodes_announced_started:
            self.nodes_announced_started.add(node_id)
            self.out.append(NodeStart(node_id=node_id))

    def _hand_to_a_thread(self, unit: Unit, attempt: int, inputs: dict[str, Any], input_hash: str) -> None:
        """Hand this call to a thread, making one only when every thread made so far is busy.

        The count is the loop's own, so its peak is the number of calls in
        flight at once. The attempt's clock starts here, with the thread."""
        call = _CallOnAThread(unit, attempt, self.compiled.node(unit[0]), inputs, input_hash)
        attempt_timeout = self._policy_of(unit[0]).attempt_timeout
        if attempt_timeout is not None:
            call.attempt_timer = self._loop.call_later(attempt_timeout, self._post, self._attempt_timed_out, unit, call)
        self.calls_on_threads.add(call)
        self.calls.put(call)
        self.calls_in_threads += 1
        if self.calls_in_threads > self.threads_made:
            self.threads_made += 1
            threading.Thread(target=self._take_calls, name="conductor", daemon=True).start()

    def _policy_of(self, node_id: str) -> Policy:
        version = self.compiled.node(node_id).version
        assert version is not None and version.policy is not None  # the engine runs only a runnable graph
        return version.policy

    def _post(self, handler: Callable[..., None], unit: Unit, *details: Any) -> None:
        self.posted.put_nowait((handler, unit, *details))

    def _take_calls(self) -> None:
        """One thread's life: make each call it takes and post what came of it to the loop.

        ``call_node`` turns a node's failures into values; what it raises
        anyway is the engine's own defect, or an output's type failing
        rather than refusing a value, and is the unit's ``engine_error``."""
        row_ids_must_survive = self.compiled._row_ids_must_survive
        while not self.stopped and (call := self.calls.get()) is not None:
            unit = call.unit
            try:
                outcome = call_node(call.node, unit[1], call.inputs, self.from_run, call.input_hash, row_ids_must_survive)
            except BaseException as raised:
                outcome = CallFailed(_engine_error(unit, raised))
            try:
                self._loop.call_soon_threadsafe(self._post, self._call_returned, unit, call, outcome)
            except RuntimeError:  # the loop is gone, and what the call returned with it
                return

    def _call_returned(self, unit: Unit, call: _CallOnAThread, outcome: dict[str, Any] | WaitingUnit | CallFailed) -> None:
        """A thread is back: its slot opens, and what it did is recorded unless its attempt timed out."""
        node_id, row = unit
        self.calls_in_threads -= 1
        self._free_slot(node_id)
        if call not in self.calls_on_threads:
            return
        self.calls_on_threads.remove(call)
        if call.attempt_timer is not None:
            call.attempt_timer.cancel()
        if outcome.__class__ is dict:
            self._report_recorded(unit, self.ledger.record(unit, outcome, call.input_hash))  # type: ignore[arg-type]
        elif isinstance(outcome, WaitingUnit):
            self._report_waiting(self.ledger.mark_waiting(unit, outcome))
        else:
            assert isinstance(outcome, CallFailed)
            self._retry_or_fail(unit, call.attempt, outcome)

    def _retry_or_fail(self, unit: Unit, attempt: int, failed: CallFailed) -> None:
        node_id, row = unit
        policy = self._policy_of(node_id)
        if not (failed.retryable and attempt < policy.retries and self.units_may_start):
            self._fail_unit(unit, failed.problem)
            return
        delay = policy.retry_delay * (2 ** attempt)
        self.out.append(NodeRetry(node_id=node_id, row=row, attempt=attempt + 1, retries=policy.retries, error=failed.problem.msg, delay=delay))
        self.retries_waiting[unit] = self._loop.call_later(delay, self._post, self._retry_due, unit, attempt + 1)

    def _retry_due(self, unit: Unit, attempt: int) -> None:
        self.retries_waiting.pop(unit, None)
        self._queue_attempt(unit, attempt)

    def _attempt_timed_out(self, unit: Unit, call: _CallOnAThread) -> None:
        """Final: the thread runs on, and a retry beside it would be a second call of the same unit."""
        if call in self.calls_on_threads:
            self.calls_on_threads.remove(call)
            node_id, row = unit
            self._fail_unit(unit, make_problem("timeout", node_id, *row, seconds=self._policy_of(node_id).attempt_timeout))

    def _fail_unit(self, unit: Unit, problem: Problem) -> None:
        """The first failure fails the leg; anything failing after it is only in the log."""
        if self.failure is None:
            self.failure = problem
            self.out.append(NodeFailed(node_id=unit[0], cause=problem))
            self._stop_starting_units()

    def _report_waiting(self, waiting_unit: WaitingUnit) -> None:
        """This unit waits on a person, and whatever reads it waits too."""
        self.a_unit_finished = True
        self.out.append(UnitWaiting(unit=waiting_unit))

    def _report_recorded(self, unit: Unit, recorded: UnitRecorded) -> None:
        """Report this unit's progress and every node its record completed, and queue what it made ready."""
        node_id, _ = unit
        self.a_unit_finished = True
        if self.runs_per_row_of[node_id] is not None:
            self._report_progress_when_due(node_id)
        for event in recorded.node_events:
            if event.node_id in self.nodes_with_unsent_progress:
                self.out.append(self._progress_of(event.node_id))
            self.out.append(event)
        self._queue_units(recorded.units_ready)

    def _report_progress_when_due(self, node_id: str) -> None:
        """One more row of this node finished: say so unless the node reported in the last quarter second.

        Most rows of a fast node stop at the clock, before anything is counted or built."""
        now = time.monotonic()
        if now - self.progress_sent_at.get(node_id, -_SECONDS_BETWEEN_PROGRESS) < _SECONDS_BETWEEN_PROGRESS:
            self.nodes_with_unsent_progress.add(node_id)
            return
        self.progress_sent_at[node_id] = now
        self.out.append(self._progress_of(node_id))

    def _progress_of(self, node_id: str) -> NodeProgress:
        self.nodes_with_unsent_progress.discard(node_id)
        done, total = self.ledger.progress(node_id)
        return NodeProgress(node_id=node_id, done=done, total=total)


def _engine_error(unit: Unit, raised: BaseException) -> Problem:
    """The failure of a unit brought down by the engine's own defect. The
    exception's text can hold a value the node returned, so the problem carries
    the generic line and the class name; the log has the rest."""
    node_id, row = unit
    logger.error("the engine failed on node %r", node_id, exc_info=raised)
    return make_problem("engine_error", node_id, *row, exception=type(raised).__name__)
