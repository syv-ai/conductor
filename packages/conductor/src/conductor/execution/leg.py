"""One leg in flight: the loop that starts ready units, and each unit's run.

``execute`` in ``conductor.execution.engine`` is the public way in; it
makes a ``Leg`` per call, enters it, and hands its events on. This module
is what runs underneath: the loop, the tasks it starts for each unit, the
thread each node's ``run`` is called in, and the teardown that stops all
of them when the leg ends. What a unit and a leg are, and why a leg
ends, is the engine module's docstring.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncGenerator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from conductor._sentinel import is_asking
from conductor.errors import (
    MESSAGES,
    ErrorCause,
    ExternalFailure,
    NodeError,
    NodeExecutionError,
    NodeTimeoutError,
    NodeValidationError,
)
from conductor.execution.events import (
    ExecutionEvent,
    GraphCancelledEvent,
    GraphCompleteEvent,
    GraphErrorEvent,
    GraphPendingEvent,
    GraphTimeoutEvent,
    NodeCompleteEvent,
    NodeErrorEvent,
    NodeProgressEvent,
    NodeRetryEvent,
    NodeSkippedEvent,
    NodeStartEvent,
)
from conductor.execution.ledger import Ledger, Skip, Unit
from conductor.execution.record import RunRecord
from conductor.graph.compiled import CompiledGraph
from conductor.returns import unpack

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
    error: NodeErrorEvent | None = None


@dataclass(frozen=True)
class _Cancelled:
    """The host set its cancel event; the leg ends ``graph_cancelled``."""


@dataclass(frozen=True)
class _TimedOut:
    """The leg's own deadline passed; the leg ends ``graph_timeout``.

    ``seconds`` is the timeout that ran out, which the ending reports."""

    seconds: float


# -- the leg ---------------------------------------------------------------------


class Leg:
    """One leg in flight: the loop that starts ready units, and each unit's run.

    ``execute`` makes one per call and drops it when the leg ends; the
    ledger outlives it as the record every ending carries. It is an async
    context manager: entering it starts the watchers for the host's cancel
    event and the leg's deadline, and leaving it — however the leg ended,
    the consumer closing the stream included — stops every unit, both
    watchers and the thread pool, so nothing outlives the ``async with``.
    ``events`` is the loop, and runs only inside it.

    It holds what the loop and the unit tasks share: the ledger, the
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

    def __init__(
        self,
        compiled: CompiledGraph,
        *,
        record: RunRecord | None,
        from_run: Mapping[type, Any],
        timeout: float | None,
        cancel: asyncio.Event,
    ) -> None:
        self.compiled = compiled
        self.ledger = Ledger(compiled) if record is None else Ledger.restore(compiled, record)
        for node_id in compiled.execution_order:
            for name, needed in compiled.node(node_id).version.interface.needs.items():
                if needed not in from_run:
                    raise TypeError(
                        f"'{node_id}' needs a {needed.__name__} for '{name}', and execute() was not given one"
                    )
        self.from_run = dict(from_run)
        self.timeout = timeout
        self.cancel = cancel
        self.started = self.ledger.completed_nodes()
        in_flight = sum(compiled.node(node_id).version.policy.concurrency for node_id in compiled.execution_order)
        self.executor = ThreadPoolExecutor(max_workers=max(1, in_flight), thread_name_prefix="conductor")
        self.gates: dict[str, asyncio.Semaphore] = {}
        self.queue: asyncio.Queue[ExecutionEvent | _UnitDone] = asyncio.Queue()
        self.running: dict[Unit, asyncio.Task[None]] = {}

    async def __aenter__(self) -> Leg:
        self.started_at = time.monotonic()
        self._watchers = [asyncio.ensure_future(self._cancelled())]
        if self.timeout is not None:
            self._watchers.append(asyncio.ensure_future(self._timed_out(self.timeout)))
        return self

    async def __aexit__(self, *exc: object) -> None:
        """Stop every unit and the watchers, and let the threads go.

        A unit task is cancelled where it waits — on a gate, on its thread,
        in a retry's sleep — and awaited, so no task outlives the leg. The
        pool is shut down without waiting: a thread mid-run finishes on its
        own and what it returns is dropped, and a call that never started
        is not started."""
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

    async def events(self, cache: dict[str, dict[str, Any]]) -> AsyncGenerator[ExecutionEvent, None]:
        """Seed the cached nodes, then start every ready unit until the leg ends."""
        for node_id, outputs in cache.items():
            self.ledger.inject(node_id, outputs)
            self.started.add(node_id)
            if not self.ledger.complete(node_id):
                done, total = self.ledger.progress(node_id)
                yield NodeProgressEvent(type="node_progress", node_id=node_id, done=done, total=total)
                continue
            result = self.ledger.result_of(node_id)
            if result is None:
                yield NodeSkippedEvent(type="node_skipped", node_id=node_id)
            else:
                yield NodeCompleteEvent(type="node_complete", node_id=node_id, result=result, cached=True)

        self._start(self.ledger.runnable())
        while self.running:
            message = await self._next()
            if isinstance(message, _Cancelled):
                yield GraphCancelledEvent(type="graph_cancelled", results=self.ledger.results(), record=self.ledger.cells())
                return
            if isinstance(message, _TimedOut):
                yield GraphTimeoutEvent(
                    type="graph_timeout",
                    results=self.ledger.results(),
                    record=self.ledger.cells(),
                    elapsed_seconds=time.monotonic() - self.started_at,
                    timeout_seconds=message.seconds,
                )
                return
            if not isinstance(message, _UnitDone):
                yield message
                continue
            self.running.pop(message.unit, None)
            if message.error is not None:
                yield message.error
                yield GraphErrorEvent(
                    type="graph_error",
                    node_id=message.error.node_id,
                    error=message.error.error,
                    cause=message.error.cause,
                    results=self.ledger.results(),
                    record=self.ledger.cells(),
                )
                return
            self._start(message.ready)

        # Quiescence: nothing in flight. Nothing may be runnable either; a
        # ready unit nobody started is a wake the ledger missed.
        missed = self.ledger.runnable()
        if missed:
            raise RuntimeError(f"{missed} are ready but were never started — a missed wake, an engine bug")
        pending = self.ledger.pending()
        if pending:
            yield GraphPendingEvent(
                type="graph_pending", pending=pending, results=self.ledger.results(), record=self.ledger.cells()
            )
            return
        unfinished = [node_id for node_id in self.compiled.execution_order if not self.ledger.complete(node_id)]
        if unfinished:
            raise RuntimeError(f"nothing left to run, but {unfinished} did not complete — an engine bug")
        yield GraphCompleteEvent(type="graph_complete", results=self.ledger.results(), record=self.ledger.cells())

    async def _next(self) -> ExecutionEvent | _UnitDone | _Cancelled | _TimedOut:
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
        ledger = self.ledger
        for unit in units:
            if unit in self.running or ledger.is_done(unit) or ledger.is_pending(unit):
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
                MESSAGES["engine_error"], node_id=node_id, original=raised,
                cause=self._cause(code="engine_error", row=row, details={"exception": type(raised).__name__}),
            )
            self.queue.put_nowait(_UnitDone(unit, error=self._error_event(node_id, failure, row)))

    async def _unit(self, unit: Unit) -> None:
        """One unit: its inputs, its attempts under the node's policy, and what it produced."""
        node_id, row = unit
        compiled, ledger, queue = self.compiled, self.ledger, self.queue
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
            await queue.put(NodeStartEvent(type="node_start", node_id=node_id))

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
                    MESSAGES["timeout"], node_id=node_id,
                    cause=self._cause(code="timeout", row=row, details={"seconds": policy.timeout}),
                )
            except NodeError as raised:
                failure = raised
            if not isinstance(failure, ExternalFailure) or attempt >= policy.retries:
                await queue.put(_UnitDone(unit, error=self._error_event(node_id, failure, row)))
                return
            attempt += 1
            delay = policy.delay * (2 ** (attempt - 1))
            await queue.put(NodeRetryEvent(
                type="node_retry", node_id=node_id, row=row,
                attempt=attempt, retries=policy.retries, error=str(failure), delay=delay,
            ))
            await asyncio.sleep(delay)

        if is_asking(value):
            # A person must answer. The unit waits; the rest of the leg runs
            # on and the leg ends pending once quiet. A node with no
            # questions of its own asks for its declared outputs.
            questions = value.questions or tuple(out.question() for out in compiled.node(node_id).interface.outputs)
            ledger.pend(unit, questions, value.prompt)
            await queue.put(_UnitDone(unit))
            return
        try:
            outputs = unpack(version.interface.returns, value, compiled.node(node_id).interface.outputs)
            ready = ledger.record(unit, outputs)
        except ValueError as invalid:
            # What the node returned does not fit what it declared — the
            # wrong type, the wrong shape, series outputs of two lengths.
            failure = NodeExecutionError(
                MESSAGES["invalid_output"], node_id=node_id, original=invalid,
                cause=self._cause(code="invalid_output", row=row, details={"reason": str(invalid)}),
            )
            await queue.put(_UnitDone(unit, error=self._error_event(node_id, failure, row)))
            return
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
        other exception is wrapped with the row in its cause — as an
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
                reason, node_id=node_id, original=invalid,
                cause=self._cause(code="invalid_input", row=row, details={"reason": reason}),
            ) from invalid
        kwargs.update({name: self.from_run[needed] for name, needed in version.interface.needs.items()})
        runner = node.runner
        try:
            return runner(**kwargs)
        except NodeError:
            raise
        except version.policy.retry_on as raised:
            raise ExternalFailure(
                MESSAGES["external_failed"], node_id=node_id, original=raised,
                cause=self._cause(code="external_failed", row=row),
            ) from raised
        except Exception as raised:
            raise NodeExecutionError(
                MESSAGES["execution_failed"], node_id=node_id, original=raised,
                cause=self._cause(code="execution_failed", row=row),
            ) from raised

    async def _after(self, unit: Unit) -> None:
        """Report progress, and completion when this was the node's last unit."""
        node_id, row = unit
        ledger = self.ledger
        if row is not None:
            done, total = ledger.progress(node_id)
            await self.queue.put(NodeProgressEvent(type="node_progress", node_id=node_id, done=done, total=total))
        if not ledger.complete(node_id):
            return
        result = ledger.result_of(node_id)
        if result is None:
            await self.queue.put(NodeSkippedEvent(type="node_skipped", node_id=node_id))
            return
        await self.queue.put(NodeCompleteEvent(type="node_complete", node_id=node_id, result=result))

    # -- failures ----------------------------------------------------------------

    @staticmethod
    def _cause(*, code: str, row: Any, details: Mapping[str, Any] | None = None) -> ErrorCause:
        """A cause the engine writes itself, with the generic message for its code."""
        return ErrorCause(code=code, message=MESSAGES[code], details=details or {}, row=row)

    @staticmethod
    def _error_event(node_id: str, failure: NodeError, row: Any) -> NodeErrorEvent:
        """The event for a failed unit. A ``NodeError`` the node raised without
        a cause gets one with the message the node wrote: code ``external_failed``
        for an ``ExternalFailure``, ``failed`` for the rest."""
        if failure.cause is not None:
            cause = failure.cause
        elif isinstance(failure, ExternalFailure):
            cause = ErrorCause(code="external_failed", message=str(failure), row=row)
        else:
            cause = ErrorCause(code="failed", message=str(failure), row=row)
        if cause.row is None and row is not None:
            cause = cause.model_copy(update={"row": row})
        return NodeErrorEvent(type="node_error", node_id=node_id, error=str(failure), cause=cause)

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
