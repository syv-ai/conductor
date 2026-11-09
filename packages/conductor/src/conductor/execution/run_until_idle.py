"""One leg, start to finish: the loop that starts every ready unit, and each unit's run.

The engine doesn't schedule whole nodes. It schedules *units*: one call of a
node's ``run``, a node on one row. A node that runs once is one unit, on the
empty row ``()``; a node that runs per row of a series is one unit per row, so
row 3 can move on down the graph while row 10 is still waiting its turn.

A *leg* is one call of ``CompiledGraph.run`` or ``stream``: it starts every
unit it can, and stops when nothing more can start. The loop asks the run's
live values (``conductor.execution.live_run_values``) which units are ready,
starts each under its node's concurrency limit and records what comes back;
each record answers with the units it made ready, and the loop starts those.

**The leg owns its work.** Every node's ``run`` is called in a thread the leg
owns, from a pool with one worker per unit that may be in flight, so a unit
never waits for a worker and a node's timeout counts only the time its thread
ran. Leaving the ``async with`` of a stream stops every unit before the block
is gone. What a leg cannot do is interrupt a thread: a ``run`` that has
started finishes on its own and what it returns is dropped. A timed-out attempt
is therefore final, and the thread keeps the node's concurrency slot until it
returns.

**A run has legs.** A unit whose node returned ``AskPerson`` is neither done
nor failed: it waits, whatever reads it waits, and the rest of the graph runs
on. Nothing resumes: the next leg is an ordinary run over live values that
already hold what earlier legs produced, and a node the graph has changed
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
import time
from collections.abc import AsyncGenerator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from conductor._sentinel import AskPerson
from conductor.errors import (
    CATALOGUE,
    ExternalFailure,
    NodeError,
    NodeExecutionError,
    NodeTimeoutError,
    NodeValidationError,
    Problem,
    problem,
)
from conductor.execution.events import (
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
from conductor.execution.live_run_values import LiveRunValues, NotCalled, Skip, Unit
from conductor.execution.run_state import RunState
from conductor.execution.validate_unit_outputs import validate_unit_outputs
from conductor.execution.waiting_units import waiting_unit_for
from conductor.interface import unpack

if TYPE_CHECKING:
    from conductor.graph.compiled import CompiledGraph

# -- what the loop reads besides events --------------------------------------------


@dataclass
class _UnitDone:
    """A unit finished, so the loop should start what it made ready.

    The only thing a unit's task puts on the queue besides events.
    ``ready`` holds the units the unit's record made ready, which the loop
    starts. ``error`` is set when the unit failed, which fails the leg.
    """

    unit: Unit
    ready: list[Unit] = field(default_factory=list)
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


class Leg:
    """One leg in flight: the loop that starts ready units, and each unit's run.

    What ``CompiledGraph.stream`` returns. Building one makes every check a
    start needs, so a refused start raises at the call. It is an async context
    manager: entering it starts the watchers for the host's cancel event and
    the leg's deadline and gives the event stream; leaving it — however the leg
    ended, the consumer dropping out included — stops every unit, both watchers
    and the thread pool, so nothing outlives the ``async with``.

    It holds what the loop and the unit tasks share: the live values, the
    thread pool every ``run`` is called on (``executor``), one semaphore
    per node sized by its policy's concurrency (``gates``), the nodes that
    have emitted ``node_start`` (``started``), the queue the unit tasks
    report on, and the tasks still running. The call chain is ``events``
    (the loop) → ``_run_unit`` (one unit, retries included) → ``_call``
    (the node, in a worker thread). Nothing reaches a node from here that
    its signature does not name.

    The pool has one worker per unit that may be in flight — the sum of
    the nodes' concurrency — so no unit ever queues for a thread and a
    node's timeout counts only the time its thread ran. Threads are made
    as needed, so an idle graph costs nothing. On the way out every unit
    task is cancelled and awaited and the pool is told to start nothing
    more; a thread mid-run finishes on its own.
    """

    #: The two things that end the loop early — the host's cancel event
    #: and the deadline — awaited beside the queue rather than polled.
    #: Made on entering, where the loop's event loop runs.
    _watchers: list[asyncio.Task[_Cancelled | _TimedOut]]
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
        self.live = LiveRunValues(compiled) if state is None else LiveRunValues.restore(compiled, state)
        self.changed_nodes = [] if state is None else compiled.nodes_changed_since(state)
        for node_id in compiled.execution_order:
            for name, needed in compiled.node(node_id).version.interface.needs.items():
                if needed not in from_run:
                    raise TypeError(
                        f"'{node_id}' needs a {needed.__name__} for '{name}', and run() was not given one"
                    )
        self.from_run = dict(from_run)
        self.timeout = timeout
        self.cancel = cancel
        self.checkpoint_every = checkpoint_every
        self.started = self.live.completed_nodes()
        in_flight = sum(compiled.node(node_id).version.policy.concurrency for node_id in compiled.execution_order)
        self.executor = ThreadPoolExecutor(max_workers=max(1, in_flight), thread_name_prefix="conductor")
        self.gates: dict[str, asyncio.Semaphore] = {}
        self.queue: asyncio.Queue[Event | _UnitDone] = asyncio.Queue()
        self.running: dict[Unit, asyncio.Task[None]] = {}

    async def __aenter__(self) -> AsyncGenerator[Event, None]:
        self.started_at = self.last_checkpoint_at = time.monotonic()
        self._watchers = [asyncio.ensure_future(self._cancelled())]
        if self.timeout is not None:
            self._watchers.append(asyncio.ensure_future(self._timed_out(self.timeout)))
        self._stream = self._events()
        return self._stream

    async def __aexit__(self, *exc: object) -> None:
        """Stop every unit and the watchers, and let the threads go.

        A unit task is cancelled where it waits — on a gate, on its thread,
        in a retry's sleep — and awaited, so no task outlives the leg. The
        pool is shut down without waiting: a thread mid-run finishes on its
        own and what it returns is dropped, and a call that never started
        is not started."""
        await self._stream.aclose()
        tasks = [*self.running.values(), *self._watchers]
        self.running.clear()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self.executor.shutdown(wait=False, cancel_futures=True)

    async def _cancelled(self) -> _Cancelled:
        await self.cancel.wait()
        return _Cancelled()

    @staticmethod
    async def _timed_out(seconds: float) -> _TimedOut:
        await asyncio.sleep(seconds)
        return _TimedOut(seconds)

    # -- the loop ----------------------------------------------------------------

    async def _events(self) -> AsyncGenerator[Event, None]:
        """Say what this leg starts from, then start every ready unit until the leg ends."""
        ledger, order = self.live, self.compiled.execution_order
        yield RunStart(
            reused_nodes=[node_id for node_id in order if node_id in self.started],
            changed_nodes=self.changed_nodes,
            waiting=ledger.waiting(),
        )
        self._start(ledger.runnable())
        while self.running:
            message = await self._next()
            if isinstance(message, _Cancelled):
                yield RunEnd(state=ledger.state("cancelled", problem("run_cancelled")))
                return
            if isinstance(message, _TimedOut):
                elapsed = time.monotonic() - self.started_at
                failure = problem("run_timed_out", elapsed_seconds=elapsed, timeout_seconds=message.seconds)
                yield RunEnd(state=ledger.state("timed_out", failure))
                return
            if not isinstance(message, _UnitDone):
                yield message
                continue
            self.running.pop(message.unit, None)
            if message.error is not None:
                yield message.error
                yield RunEnd(state=ledger.state("failed", message.error.cause))
                return
            self._start(message.ready)
            if self.checkpoint_every is not None and self.running and time.monotonic() - self.last_checkpoint_at >= self.checkpoint_every:
                yield RunCheckpoint(state=ledger.state("running"))
                self.last_checkpoint_at = time.monotonic()

        # Quiescence: nothing in flight. Nothing may be runnable either; a
        # ready unit nobody started is a wake that was missed.
        missed = ledger.runnable()
        if missed:
            raise RuntimeError(f"{missed} are ready but were never started — a missed wake, an engine bug")
        if ledger.waiting():
            yield RunEnd(state=ledger.state("waiting"))
            return
        unfinished = [node_id for node_id in order if not ledger.complete(node_id)]
        if unfinished:
            raise RuntimeError(f"nothing left to run, but {unfinished} did not complete — an engine bug")
        yield RunEnd(state=ledger.state("complete"))

    async def _next(self) -> Event | _UnitDone | _Cancelled | _TimedOut:
        """The next thing the loop acts on: a message from a unit, or the reason to stop.

        The queue and the watchers are awaited together; whichever is first
        wins, and a stop wins over a message that arrived in the same
        moment — a cancel over a timeout. A getter left waiting is
        cancelled, and the queue keeps the item for the next one."""
        getter = asyncio.ensure_future(self.queue.get())
        try:
            done, _ = await asyncio.wait([getter, *self._watchers], return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not getter.done():
                getter.cancel()
        for watcher in self._watchers:
            if watcher in done:
                return watcher.result()
        return getter.result()

    def _start(self, units: list[Unit]) -> None:
        """Start a task for each of these ready units not already running, done or waiting.

        A unit two records both made ready is in both lists; the second finds it running."""
        ledger = self.live
        for unit in units:
            if unit in self.running or ledger.is_done(unit) or ledger.is_waiting(unit):
                continue
            self.running[unit] = asyncio.create_task(self._run_unit(unit), name=f"unit-{unit}")

    def _gate(self, node_id: str) -> asyncio.Semaphore:
        if node_id not in self.gates:
            self.gates[node_id] = asyncio.Semaphore(self.compiled.node(node_id).version.policy.concurrency)
        return self.gates[node_id]

    # -- one unit ----------------------------------------------------------------

    async def _run_unit(self, unit: Unit) -> None:
        """One unit, start to finish, and its report to the loop.

        An exception out of the unit's own machinery — the ledger, the
        events, this module — is a defect in the engine, and is reported as
        the unit's failure with code ``engine_error``, since a task that died
        silently would leave the loop waiting."""
        try:
            await self._unit(unit)
        except Exception as raised:
            # The exception's text can hold a value the node returned, so
            # the event carries the generic line and the class name; the
            # exception itself is on ``original`` for a log.
            node_id, row = unit
            failure = NodeExecutionError(
                CATALOGUE["engine_error"][0], node_id=node_id, original=raised, code="engine_error", ctx={"exception": type(raised).__name__},
            )
            self.queue.put_nowait(_UnitDone(unit, error=self._error_event(node_id, failure, row)))

    async def _unit(self, unit: Unit) -> None:
        """One unit: its inputs, its attempts under the node's policy, and what it produced."""
        node_id, row = unit
        compiled, ledger, queue = self.compiled, self.live, self.queue
        try:
            inputs = ledger.inputs_for(unit)
        except NodeError as failure:  # two edges into one input both supplied this row
            await queue.put(_UnitDone(unit, error=self._error_event(node_id, failure, row)))
            return
        if isinstance(inputs, Skip):
            ready = ledger.record(unit, inputs)
            await self._after(unit)
            await queue.put(_UnitDone(unit, ready))
            return

        if node_id not in self.started:
            self.started.add(node_id)
            await queue.put(NodeStart(node_id=node_id))

        if isinstance(inputs, NotCalled):
            ready = ledger.record(unit, inputs)
            await self._after(unit)
            await queue.put(_UnitDone(unit, ready))
            return

        version = compiled.node(node_id).version
        policy = version.policy
        gate = self._gate(node_id)
        loop = asyncio.get_running_loop()
        attempt = 0
        while True:
            await gate.acquire()
            call = loop.run_in_executor(self.executor, self._call, unit, inputs)
            # The slot is the thread's, not the wait's: it opens when the
            # thread returns, however long ago the leg stopped waiting for it.
            call.add_done_callback(lambda _: gate.release())
            try:
                value = await asyncio.wait_for(asyncio.shield(call), timeout=policy.timeout)
                break
            except TimeoutError:
                # Final: the thread runs on, and a retry beside it would be a
                # second call of the same unit.
                failure: NodeError = NodeTimeoutError(
                    CATALOGUE["timeout"][0], node_id=node_id, code="timeout", ctx={"seconds": policy.timeout},
                )
            except NodeError as raised:
                failure = raised
            if not isinstance(failure, ExternalFailure) or attempt >= policy.retries:
                await queue.put(_UnitDone(unit, error=self._error_event(node_id, failure, row)))
                return
            attempt += 1
            delay = policy.delay * (2 ** (attempt - 1))
            await queue.put(NodeRetry(
                node_id=node_id, row=row,
                attempt=attempt, retries=policy.retries, error=str(failure), delay=delay,
            ))
            await asyncio.sleep(delay)

        node = compiled.node(node_id)
        try:
            if isinstance(value, AskPerson):
                asked = waiting_unit_for(node, row, value)
                ledger.mark_waiting(unit, asked)
                await queue.put(UnitWaiting(unit=asked))
                await queue.put(_UnitDone(unit))
                return
            cells, faults = validate_unit_outputs(node, row, unpack(version.returns, value, node.interface.outputs, version.interface.outputs))
        except ValueError as invalid:  # the wrong shape of return, or an ask naming no output
            await queue.put(_UnitDone(unit, error=self._invalid_output(node_id, row, str(invalid), invalid, code="invalid_output")))
            return
        if faults:
            reason = "; ".join(fault.message for fault in faults)
            await queue.put(_UnitDone(unit, error=self._invalid_output(node_id, row, reason, None, [e for fault in faults for e in fault.errors], code=next((fault.code for fault in faults if fault.code != "invalid_output"), "invalid_output"))))
            return
        ready = ledger.record(unit, cells)
        await self._after(unit)
        await queue.put(_UnitDone(unit, ready))

    def _call(self, unit: Unit, inputs: dict[str, Any]) -> Any:
        """Validate the inputs against the node's interface, then call the node.

        The interface is the list of inputs the node actually has, which can be
        more than its declaration: a ``compute_inputs`` hook may add some.
        Validating against the interface rather than the declaration means those
        added inputs are validated like any other. ``FromRun`` parameters
        (values the host supplies by type) come from the run. Runs in a
        worker thread. A ``NodeError`` the node raises passes through; any
        other exception is wrapped — as an
        ``ExternalFailure`` when the policy's ``retry_on`` names its class,
        as a ``NodeExecutionError`` otherwise — and the exception itself
        goes on ``original``, its text on no event.
        """
        node_id, row = unit
        node = self.compiled.node(node_id)
        version = node.version
        try:
            kwargs = node.validate(inputs)
        except ValidationError as invalid:
            reason = self._describe(invalid, node.interface.inputs)
            raise NodeValidationError(
                reason, node_id=node_id, original=invalid, code="invalid_input", ctx={"reason": reason},
            ) from invalid
        kwargs.update({name: self.from_run[needed] for name, needed in version.interface.needs.items()})
        runner = node.runner
        try:
            return runner(**kwargs)
        except NodeError:
            raise
        except version.policy.retry_on as raised:
            raise ExternalFailure(
                CATALOGUE["external_failed"][0], node_id=node_id, original=raised, code="external_failed",
            ) from raised
        except Exception as raised:
            raise NodeExecutionError(
                CATALOGUE["execution_failed"][0], node_id=node_id, original=raised, code="execution_failed",
            ) from raised

    async def _after(self, unit: Unit) -> None:
        """Report progress, and completion when this was the node's last unit."""
        node_id, _ = unit
        ledger = self.live
        if ledger.on_a_row(unit):
            done, total = ledger.progress(node_id)
            await self.queue.put(NodeProgress(node_id=node_id, done=done, total=total))
        if not ledger.complete(node_id):
            return
        result = ledger.result_of(node_id)
        rows_not_called = ledger.rows_not_called(node_id)
        if result is None and not rows_not_called:
            await self.queue.put(NodeSkipped(node_id=node_id))
            return
        await self.queue.put(NodeComplete(node_id=node_id, result=result or {}, rows_not_called=rows_not_called))

    # -- failures ----------------------------------------------------------------

    def _invalid_output(self, node_id: str, row: Any, reason: str, original: Exception | None, errors: list[Any] | None = None, *, code: str) -> NodeFailed:
        """The failure of a node that returned what it did not declare."""
        details: dict[str, Any] = {"reason": reason}
        if errors:
            details["errors"] = errors
        failure = NodeExecutionError(CATALOGUE[code][0], node_id=node_id, original=original, code=code, ctx=details)
        return self._error_event(node_id, failure, row)

    @staticmethod
    def _error_event(node_id: str, failure: NodeError, row: Any) -> NodeFailed:
        """The event for a failed unit. A ``NodeError`` the node raised without
        a code is ``external_failed`` for an ``ExternalFailure``, ``failed`` for
        the rest, with the message the node wrote."""
        code = failure.code or ("external_failed" if isinstance(failure, ExternalFailure) else "failed")
        return NodeFailed(node_id=node_id, error=str(failure), cause=Problem(type=code, msg=str(failure), loc=(node_id, *row), ctx=failure.ctx))

    @staticmethod
    def _describe(invalid: ValidationError, inputs: tuple[Any, ...]) -> str:
        """A validation error as one line per field, using the field's title."""
        titles = {inp.name: inp.title for inp in inputs}
        parts = []
        for err in invalid.errors():
            loc = [segment for segment in err.get("loc", ()) if isinstance(segment, str)]
            name = loc[0] if loc else "?"
            parts.append(f"{titles.get(name, name)}: {err.get('msg', '')}")
        return "Invalid input — " + "; ".join(parts)
