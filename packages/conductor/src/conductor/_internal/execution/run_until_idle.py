"""One leg, start to finish: the loop that starts every ready unit, and each unit's run.

The engine doesn't schedule whole nodes. It schedules *units*: one call of a
node's ``run``, a node on one row. A node that runs once is one unit, on the
empty row ``()``; a node that runs per row of a series is one unit per row, so
row 3 can move on down the graph while row 10 is still waiting its turn.

A *leg* is one call of ``CompiledGraph.run`` or ``stream``: it starts every
unit it can, and stops when nothing more can start. The loop asks the run's
ledger (``conductor._internal.execution.run_ledger``) which units are ready
and queues each on its node; a unit runs as an asyncio task once its node has a free
concurrency slot, and each record answers with the units it made ready.

**The leg owns its work.** Every node's ``run`` is called in a thread the leg
owns, made when every thread so far is busy, so a unit never waits for a
thread and a node's timeout counts only the time its thread ran. Leaving the
``async with`` of a stream stops every unit before the block is gone. What a
leg cannot do is interrupt a thread: a ``run`` that has started finishes on
its own and what it returns is dropped. A timed-out attempt is therefore
final, and the thread keeps the node's concurrency slot until it returns.

**A run has legs.** A unit whose node returned ``AskPerson`` is neither done
nor failed: it waits, whatever reads it waits, and the rest of the graph runs
on. Nothing resumes: the next leg is an ordinary run over a ledger that
already holds what earlier legs produced, and a node the graph has changed
since, or that reads one, runs again.

**A leg can save itself while it runs.** Asked for checkpoints, the loop
compares a monotonic clock each time a unit finishes and, when the interval has
passed, emits the state as ``RunCheckpoint``. Only a finished unit changes the
state, so no timer is needed. Building the state runs on the event loop and
costs a pass over every value so far; the clock restarts once the consumer
comes back for the next event, so a slow save cannot crowd out the leg.

**Every leg ends the same way:** one ``RunEnd`` carrying the state. A failed
unit fails the leg: the other units are stopped and the cause, with the row,
is on ``state.failure``.
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from collections import deque
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from conductor._internal.errors import (
    PROBLEM_CATALOGUE,
    ExternalFailure,
    InvalidCallError,
    NodeError,
    NodeExecutionError,
    NodeTimeoutError,
    NodeValidationError,
    Problem,
    make_problem,
)
from conductor._internal.execution.events import (
    Event,
    NodeComplete,
    NodeFailed,
    NodeProgress,
    NodeRetry,
    NodeSkipped,
    NodeStart,
    RunCheckpoint,
    RunEnd,
    RunStart,
    UnitWaiting,
)
from conductor._internal.execution.run_ledger import (
    BlankInputsStopCall,
    RunLedger,
    SkippedAtRow,
    Unit,
    UnitRecorded,
)
from conductor._internal.execution.run_state import RunState
from conductor._internal.execution.validate_unit_outputs import took_a_row_in, validate_unit_outputs
from conductor._internal.execution.waiting_units import waiting_unit_for
from conductor._internal.node_signature import split_return_into_outputs
from conductor._internal.skip_and_ask import AskPerson
from conductor._internal.table import missing_column_in

logger = logging.getLogger("conductor")

if TYPE_CHECKING:
    from conductor._internal.graph.compiled_graph import CompiledGraph

# -- what the loop reads besides events --------------------------------------------


@dataclass
class _UnitDone:
    """A unit finished, so the loop should start what it made ready.

    The only thing a unit's task puts on the queue besides events.
    ``units_made_ready`` holds the units the unit's record made ready, which the loop
    starts. ``error`` is set when the unit failed, which fails the leg.
    """

    unit: Unit
    units_made_ready: list[Unit] = field(default_factory=list)
    error: NodeFailed | None = None


@dataclass(frozen=True)
class _Cancelled:
    """The host set its cancel event; the leg ends ``cancelled``."""


@dataclass(frozen=True)
class _TimedOut:
    """The leg's own deadline passed; the leg ends ``timed_out``.

    ``seconds`` is the timeout that ran out, which the state's failure reports."""

    seconds: float


# -- the leg ---------------------------------------------------------------------


class RunningLeg:
    """One leg in flight: the loop that starts ready units, and each unit's run.

    What ``CompiledGraph.stream`` returns. Building one makes every check a
    start needs, so a start that can't go raises at the call. It is an async context
    manager: entering it arms the watch on the host's cancel event and the
    leg's deadline and gives the event stream; leaving it — however the leg
    ended, the consumer dropping out included — stops every unit and tells
    the threads to take no more, so nothing outlives the ``async with``.

    It holds what the loop and the unit tasks share: the ledger, the
    calls handed to its threads (``calls``), the nodes that
    have emitted ``node_start`` (``nodes_announced_started``), the queue the unit tasks
    report on, and each unit between being made ready and being reported
    (``in_flight``, ``None`` until it has a task). The call chain is
    ``events`` (the loop) → ``_run_unit_reporting_engine_errors`` (one attempt) → ``_validate_inputs_and_call_node`` (the
    node, in a worker thread). Nothing reaches a node from here that its
    signature does not name.

    A node holds at most its policy's concurrency of units as tasks at
    once: the rest wait, as ``(unit, attempt)``, in its ``ready_units``, so a
    wide series costs a task per slot and not per row. Threads are made as
    calls need them, so an idle graph costs nothing and no call queues for
    one. On the way out every unit task is cancelled and awaited and the
    threads take no more calls; a thread mid-run finishes on its own.
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
        self.changed_nodes = [] if state is None else compiled.nodes_changed_since(state)
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
        self.calls: queue.SimpleQueue[tuple[asyncio.Future[Any], Unit, dict[str, Any]] | None] = queue.SimpleQueue()
        self.calls_in_threads = 0
        self.threads_made = 0
        self.stopped = False
        self.ready_units: dict[str, deque[tuple[Unit, int]]] = {node_id: deque() for node_id in self.free_slots}
        self.queue: asyncio.Queue[Event | _UnitDone | _Cancelled | _TimedOut] = asyncio.Queue()
        self.in_flight: dict[Unit, asyncio.Task[None] | None] = {}
        self.timed_out: _TimedOut | None = None

    def __aiter__(self) -> None:
        raise TypeError("a stream is read with: async with compiled.stream() as events: async for event in events: ...")

    async def __aenter__(self) -> AsyncGenerator[Event, None]:
        self.started_at = self.last_checkpoint_at = time.monotonic()
        self._cancel_watch = asyncio.ensure_future(self._report_cancel())
        self._deadline = None if self.timeout is None else asyncio.get_running_loop().call_later(self.timeout, self._report_deadline)
        self._stream = self._run_loop()
        return self._stream

    async def __aexit__(self, *exc: object) -> None:
        """Stop every unit and the watches, and let the threads go.

        A unit task is cancelled where it waits — on its thread or in a
        retry's sleep — and awaited, so no task outlives the leg, and no unit
        still queued gets one. The threads are not waited for: one mid-run
        finishes on its own, what it returns is dropped, and a call that never
        started is not started."""
        await self._stream.aclose()
        if self._deadline is not None:
            self._deadline.cancel()
        for ready in self.ready_units.values():
            ready.clear()
        tasks = [task for task in self.in_flight.values() if task is not None]
        tasks.append(self._cancel_watch)
        self.in_flight.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.stopped = True
        for _ in range(self.threads_made):
            self.calls.put(None)

    async def _report_cancel(self) -> None:
        """Tell the loop when the host sets its cancel event."""
        await self.cancel.wait()
        self.queue.put_nowait(_Cancelled())

    def _report_deadline(self) -> None:
        """The leg's deadline passed: the loop stops at its next message, and this is one."""
        assert self.timeout is not None
        self.timed_out = _TimedOut(self.timeout)
        self.queue.put_nowait(self.timed_out)

    # -- the loop ----------------------------------------------------------------

    async def _run_loop(self) -> AsyncGenerator[Event, None]:
        """Say what this leg starts from, then start every ready unit until the leg ends."""
        ledger, order = self.ledger, self.compiled.execution_order
        yield RunStart(
            reused_nodes=[node_id for node_id in order if node_id in self.nodes_announced_started],
            changed_nodes=self.changed_nodes,
            waiting=ledger.waiting(),
        )
        self._start_units(ledger.units_ready_to_start())
        while self.in_flight:
            message = await self._next_message()
            if isinstance(message, _Cancelled):
                yield RunEnd(state=ledger.state("cancelled", make_problem("run_cancelled")))
                return
            if isinstance(message, _TimedOut):
                elapsed = time.monotonic() - self.started_at
                failure = make_problem("run_timed_out", elapsed_seconds=elapsed, timeout_seconds=message.seconds)
                yield RunEnd(state=ledger.state("timed_out", failure))
                return
            if not isinstance(message, _UnitDone):
                yield message
                continue
            self.in_flight.pop(message.unit, None)
            if message.error is not None:
                yield message.error
                yield RunEnd(state=ledger.state("failed", message.error.cause))
                return
            self._start_units(message.units_made_ready)
            if self.checkpoint_every is not None and self.in_flight and time.monotonic() - self.last_checkpoint_at >= self.checkpoint_every:
                yield RunCheckpoint(state=ledger.state("running"))
                self.last_checkpoint_at = time.monotonic()

        yield RunEnd(state=ledger.state("waiting" if ledger.waiting() else "complete"))

    async def _next_message(self) -> Event | _UnitDone | _Cancelled | _TimedOut:
        """The next thing the loop acts on: a message from a unit, or the reason to stop.

        A message already queued comes at once, with no task or wait for it. A
        stop wins over any message that is there too, a cancel over a timeout."""
        message = await self.queue.get()
        if self.cancel.is_set():
            return _Cancelled()
        return self.timed_out or message

    def _start_units(self, units: list[Unit]) -> None:
        """Queue each of these ready units not already in flight, done or waiting.

        A unit two records both made ready is in both lists; the second finds it in flight."""
        ledger = self.ledger
        for unit in units:
            if unit in self.in_flight or ledger.is_done(unit) or ledger.is_waiting(unit):
                continue
            self._queue_attempt(unit, 0)

    def _queue_attempt(self, unit: Unit, attempt: int) -> None:
        node_id, _ = unit
        self.in_flight[unit] = None
        self.ready_units[node_id].append((unit, attempt))
        self._fill_free_slots(node_id)

    def _fill_free_slots(self, node_id: str) -> None:
        """Give a task to as many of this node's waiting attempts as it has free slots."""
        ready = self.ready_units[node_id]
        while ready and self.free_slots[node_id]:
            unit, attempt = ready.popleft()
            self.free_slots[node_id] -= 1
            self.in_flight[unit] = asyncio.create_task(self._run_unit_reporting_engine_errors(unit, attempt))

    def _free_slot(self, node_id: str) -> None:
        self.free_slots[node_id] += 1
        self._fill_free_slots(node_id)

    # -- the threads -------------------------------------------------------------

    def _call_on_a_thread(self, unit: Unit, inputs: dict[str, Any]) -> asyncio.Future[Any]:
        """Hand this call to a thread, making one only when every thread made so far is busy.

        The count is the loop's own, so a thread that has returned its result
        counts as free the moment the loop hears of it, however busy the loop
        keeps the processor. Its peak is the number of calls in flight at once."""
        loop = asyncio.get_running_loop()
        call: asyncio.Future[Any] = loop.create_future()
        self.calls.put((call, unit, inputs))
        self.calls_in_threads += 1
        if self.calls_in_threads > self.threads_made:
            self.threads_made += 1
            threading.Thread(target=self._take_calls, args=(loop,), name="conductor", daemon=True).start()
        return call

    def _take_calls(self, loop: asyncio.AbstractEventLoop) -> None:
        """One thread's life: run each call it takes and hand the outcome to the loop."""
        while not self.stopped and (taken := self.calls.get()) is not None:
            call, unit, inputs = taken
            value, raised = None, None
            try:
                value = self._validate_inputs_and_call_node(unit, inputs)
            except BaseException as failure:
                raised = failure
            try:
                loop.call_soon_threadsafe(self._call_returned, call, unit[0], value, raised)
            except RuntimeError:  # the loop is gone, and what the call returned with it
                return

    def _call_returned(self, call: asyncio.Future[Any], node_id: str, value: Any, raised: BaseException | None) -> None:
        """A thread is back: its slot opens now, however long ago the leg stopped waiting for it."""
        self.calls_in_threads -= 1
        self._free_slot(node_id)
        if not call.done():
            if raised is None:
                call.set_result(value)
            else:
                call.set_exception(raised)

    # -- one unit ----------------------------------------------------------------

    async def _run_unit_reporting_engine_errors(self, unit: Unit, attempt: int) -> None:
        """One attempt of one unit, start to finish, and its report to the loop.

        An exception out of the unit's own machinery — the ledger, the
        events, this module — is a defect in the engine, and is reported as
        the unit's failure with code ``engine_error``, since a task that died
        silently would leave the loop waiting. The task's cancellation is not
        that: it means the leg is over."""
        try:
            await self._run_unit(unit, attempt)
        except asyncio.CancelledError:
            raise
        except BaseException as raised:
            # The exception's text can hold a value the node returned, so
            # the event carries the generic line and the class name; the log has the rest.
            node_id, row = unit
            logger.error("the engine failed on node %r", node_id, exc_info=raised)
            failure = NodeExecutionError(
                PROBLEM_CATALOGUE["engine_error"].message, node_id=node_id, code="engine_error", ctx={"exception": type(raised).__name__},
            )
            self.queue.put_nowait(_UnitDone(unit, error=self._node_failed_event(node_id, failure, row)))

    async def _run_unit(self, unit: Unit, attempt: int) -> None:
        """One attempt: its inputs, the node's call, and what it produced.

        The attempt holds a slot of its node from when its task starts until
        its call's thread returns, however long ago the leg stopped waiting
        for it. An attempt that needs no thread gives its slot back here. A
        failed attempt the policy retries holds no slot while it waits."""
        node_id, row = unit
        compiled, ledger, queue = self.compiled, self.ledger, self.queue
        try:
            inputs = ledger.inputs_for(unit)
        except NodeError as failure:  # two edges into one input both supplied this row
            self._free_slot(node_id)
            queue.put_nowait(_UnitDone(unit, error=self._node_failed_event(node_id, failure, row)))
            return
        if not isinstance(inputs, SkippedAtRow) and node_id not in self.nodes_announced_started:
            self.nodes_announced_started.add(node_id)
            queue.put_nowait(NodeStart(node_id=node_id))
        if isinstance(inputs, (SkippedAtRow, BlankInputsStopCall)):
            self._free_slot(node_id)
            self._report_recorded(unit, ledger.record(unit, inputs))
            return

        node = compiled.node(node_id)
        version = node.version
        policy = version.policy
        call = self._call_on_a_thread(unit, inputs)
        try:
            # With no timeout the wait is on the call itself: only the leg's end cancels it.
            value = await (call if policy.timeout is None else asyncio.wait_for(asyncio.shield(call), timeout=policy.timeout))
        except TimeoutError:
            # Final: the thread runs on, and a retry beside it would be a
            # second call of the same unit.
            failure = NodeTimeoutError(PROBLEM_CATALOGUE["timeout"].message, node_id=node_id, code="timeout", ctx={"seconds": policy.timeout})
            queue.put_nowait(_UnitDone(unit, error=self._node_failed_event(node_id, failure, row)))
            return
        except NodeError as raised:
            if not isinstance(raised, ExternalFailure) or attempt >= policy.retries:
                queue.put_nowait(_UnitDone(unit, error=self._node_failed_event(node_id, raised, row)))
                return
            delay = policy.delay * (2 ** attempt)
            queue.put_nowait(NodeRetry(
                node_id=node_id, row=row,
                attempt=attempt + 1, retries=policy.retries, error=str(raised), delay=delay,
            ))
            await asyncio.sleep(delay)
            self._queue_attempt(unit, attempt + 1)
            return

        try:
            if isinstance(value, AskPerson):
                waiting_unit = waiting_unit_for(node, row, value, inputs)
                ledger.mark_waiting(unit, waiting_unit)
                queue.put_nowait(UnitWaiting(unit=waiting_unit))
                queue.put_nowait(_UnitDone(unit))
                return
            must_survive = compiled._row_ids_must_survive
            if must_survive and not took_a_row_in(inputs):
                must_survive = frozenset()
            value_by_output = split_return_into_outputs(version.return_shape, value, node.interface.outputs, version.interface.outputs)
            cells, problems = validate_unit_outputs(node, row, value_by_output, must_survive)
        except ValueError as invalid:  # the wrong shape of return, or an ask that isn't sound
            queue.put_nowait(_UnitDone(unit, error=self._invalid_output_event(node_id, row, str(invalid), code="invalid_output")))
            return
        if problems:
            reason = "; ".join(problem.msg for problem in problems)
            errors = [error for problem in problems for error in problem.ctx["errors"]]
            code = next((problem.type for problem in problems if problem.type != "invalid_output"), "invalid_output")
            queue.put_nowait(_UnitDone(unit, error=self._invalid_output_event(node_id, row, reason, errors, code=code)))
            return
        self._report_recorded(unit, ledger.record(unit, cells))

    def _validate_inputs_and_call_node(self, unit: Unit, inputs: dict[str, Any]) -> Any:
        """Validate the inputs against the node's interface, then call the node.

        The interface is the list of inputs the node actually has, which can be
        more than its declaration: a ``compute_inputs`` hook may add some.
        Validating against the interface rather than the declaration means those
        added inputs are validated like any other. ``FromRun`` parameters
        (values the host supplies by type) come from the run. Runs in a
        worker thread. A ``NodeError`` the node raises passes through; anything
        else a node or an input's type raises, ``SystemExit`` and a stray
        ``CancelledError`` included, is the unit's failure: wrapped as an
        ``ExternalFailure`` when the policy's ``retry_on`` names its class, as a
        ``NodeExecutionError`` otherwise, with the exception in the log, its
        class name in ``ctx`` and its text on no event.
        """
        node_id, row = unit
        node = self.compiled.node(node_id)
        version = node.version
        try:
            kwargs = node._validate_call_inputs(inputs)
        except ValidationError as invalid:
            reason = self._validation_error_as_text(invalid, node.interface.inputs)
            column = missing_column_in(invalid)
            if column is not None:
                raise NodeValidationError(reason, node_id=node_id, code="table_column_missing", ctx={"column": column}) from invalid
            raise NodeValidationError(
                reason, node_id=node_id, code="invalid_input", ctx={"reason": reason},
            ) from invalid
        except BaseException as raised:
            raise self._node_failed(node_id, raised) from raised
        kwargs.update({name: self.from_run[needed] for name, needed in version.from_run.items()})
        node_function = version.run
        try:
            return node_function(**kwargs)
        except NodeError:
            raise
        except version.policy.retry_on as raised:
            logger.warning("node %r: an outside service failed", node_id, exc_info=raised)
            raise ExternalFailure(
                PROBLEM_CATALOGUE["external_failed"].message, node_id=node_id, code="external_failed", ctx={"exception": type(raised).__name__},
            ) from raised
        except BaseException as raised:
            raise self._node_failed(node_id, raised) from raised

    @staticmethod
    def _node_failed(node_id: str, raised: BaseException) -> NodeExecutionError:
        logger.error("node %r failed", node_id, exc_info=raised)
        return NodeExecutionError(
            PROBLEM_CATALOGUE["execution_failed"].message, node_id=node_id, code="execution_failed", ctx={"exception": type(raised).__name__},
        )

    def _report_recorded(self, unit: Unit, recorded: UnitRecorded) -> None:
        """Report this unit's progress and every node its record completed, then tell the loop what is ready."""
        node_id, _ = unit
        ledger = self.ledger
        if ledger.node_runs_per_row(node_id):
            done, total = ledger.progress(node_id)
            self.queue.put_nowait(NodeProgress(node_id=node_id, done=done, total=total))
        for completed_node_id in recorded.nodes_completed:
            result = ledger.result_of(completed_node_id)
            rows_not_called = ledger.rows_not_called(completed_node_id)
            if result is None and not rows_not_called:
                self.queue.put_nowait(NodeSkipped(node_id=completed_node_id))
            else:
                self.queue.put_nowait(NodeComplete(node_id=completed_node_id, result=result or {}, rows_not_called=rows_not_called))
        self.queue.put_nowait(_UnitDone(unit, recorded.units_ready))

    # -- failures ----------------------------------------------------------------

    def _invalid_output_event(self, node_id: str, row: Any, reason: str, errors: list[Any] | None = None, *, code: str) -> NodeFailed:
        """The failure of a node that returned what it did not declare."""
        details: dict[str, Any] = {"reason": reason}
        if errors:
            details["errors"] = errors
        failure = NodeExecutionError(PROBLEM_CATALOGUE[code].message, node_id=node_id, code=code, ctx=details)
        return self._node_failed_event(node_id, failure, row)

    @staticmethod
    def _node_failed_event(node_id: str, failure: NodeError, row: Any) -> NodeFailed:
        """The event for a failed unit. A ``NodeError`` the node raised without
        a code is ``external_failed`` for an ``ExternalFailure``, ``failed`` for
        the rest, with the message the node wrote."""
        code = failure.code or ("external_failed" if isinstance(failure, ExternalFailure) else "failed")
        return NodeFailed(node_id=node_id, cause=Problem(type=code, msg=str(failure), loc=(node_id, *row), ctx=failure.ctx))

    @staticmethod
    def _validation_error_as_text(invalid: ValidationError, inputs: tuple[Any, ...]) -> str:
        """A validation error as one line per field, using the field's title."""
        titles = {input_.name: input_.title for input_ in inputs}
        parts = []
        for err in invalid.errors():
            loc = [segment for segment in err.get("loc", ()) if isinstance(segment, str)]
            name = loc[0] if loc else "?"
            parts.append(f"{titles.get(name, name)}: {err.get('msg', '')}")
        return "Invalid input — " + "; ".join(parts)
