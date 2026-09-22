"""Run a compiled graph: every ready unit, as soon as it is ready.

A unit is ``(node, row)``: a node that runs once is one unit with row
``None``; a node that runs once per row of a series — a value with many
rows, whose *index* names where the rows come from — is one unit per
row. The loop asks the ledger (the run's record of every value produced
so far, ``conductor.execution.ledger``) which units are ready, starts
each under its node's concurrency limit and records what comes back; the
ledger answers each record with the units it made ready, and the loop
starts those — a unit finishing is what makes other units ready. Row 1
of a chain can finish before row 10 of the first node has started; a
node that receives a whole group of rows at once (a reduction) has its
unit ready once the whole group is.

A failed unit fails the run: the other units are cancelled and the cause,
with the row, goes out on the event stream.

**The leg owns its work.** Every node's ``run`` is called in a thread the
leg owns, from a pool with one worker per unit that may be in flight, so
a unit never waits for a worker and a node's timeout counts only the time
its thread ran. Closing the event stream — ``aclose()``, or a cancelled
consumer — stops every unit before the stream is gone. What
the leg cannot do is interrupt a thread: a ``run`` that has started
finishes on its own, and what it returns is dropped. A timed-out attempt
is therefore final, and the thread keeps the node's concurrency slot until
it returns.

**A run has legs.** One call of ``execute`` is one leg, and it runs until
nothing is runnable and nothing is in flight. A unit whose node returned
``Asks`` is neither done nor failed: it waits, everything that reads it
waits, and the rest of the graph runs on. If anything is waiting when the
leg goes quiet, the leg ends with ``graph_pending`` carrying every waiting
unit's questions, plus what the leg completed and the ledger's record (a
``RunRecord``: every cell in wire form, and a fingerprint per node). The
next leg is ``execute`` again, with ``record`` restoring the ledger and
the answers in ``cache`` as the asking node's outputs. Nothing is
checkpointed and nothing resumes: a leg is an ordinary run over a ledger
that already holds what earlier legs produced — and a node the graph has
changed since, or that reads one, runs again.

**Every ending has one shape.** ``graph_complete``, ``graph_pending``,
``graph_error``, ``graph_cancelled`` and ``graph_timeout`` all carry the
results so far and the ledger's record beside their reason, so a host can
start a new run from any of them.
"""

from __future__ import annotations

import asyncio
import enum
import time
from collections.abc import AsyncGenerator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from conductor._sentinel import is_asking
from conductor.errors import (
    MESSAGES,
    CompilationError,
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

# -- entry points ---------------------------------------------------------------


async def execute(
    compiled: CompiledGraph,
    *,
    record: RunRecord | Mapping[str, Any] | None = None,
    cache: dict[str, dict[str, Any]] | None = None,
    from_run: Mapping[type, Any] | None = None,
    timeout: float | None = None,
    cancel: asyncio.Event | None = None,
) -> AsyncGenerator[ExecutionEvent, None]:
    """Run one leg of ``compiled`` and yield events as it goes.

    ``from_run`` supplies values to nodes by type: a ``run`` parameter
    annotated ``Annotated[X, FromRun()]`` receives ``from_run[X]``. A graph
    needing a type the host did not provide is refused before anything
    runs. ``record`` restores the ledger of an earlier leg, cell by cell —
    the ``RunRecord`` an ending carried, or its ``model_dump()`` as a host
    stored it; a node the graph has changed since that leg, and everything
    reading it, is left out and runs again. ``cache`` records outputs by node id without running the node — a
    person's answers to a pending unit, or an earlier run's results a caller
    reuses. For a node running per row each output is a series, and only
    the rows it names are recorded. A node the cache completes is reported
    as ``node_complete`` (``cached=True``); a unit already done, or a row
    not yet produced, is refused.
    ``timeout`` bounds the whole leg, in seconds, and ends it with
    ``graph_timeout``; ``None``, the default, lets the leg run until it is
    quiet. ``cancel`` is an event the host sets to stop the leg, which ends
    it with ``graph_cancelled`` at once::

        async for event in execute(compiled, from_run={Clock: clock}):
            ...

    Closing the stream — ``aclose()`` on the generator, or cancelling the
    task that reads it — stops every unit before the generator is gone; a
    bare ``break`` leaves the generator open until it is closed or
    collected, so close it.
    """
    if not compiled.is_runnable:
        raise CompilationError("the graph cannot run", problems=compiled.problems)
    if record is not None and not isinstance(record, RunRecord):
        record = RunRecord.model_validate(record)
    leg = _Leg(
        compiled,
        record=record,
        from_run=from_run or {},
        timeout=timeout,
        cancel=cancel or asyncio.Event(),
    )
    events = leg.events(cache or {})
    try:
        async for event in events:
            yield event
    finally:
        # Close the leg's own generator here, under the consumer's close,
        # rather than leaving it to the garbage collector's finalizer: the
        # units are stopped by the time the consumer's ``aclose()`` returns.
        await events.aclose()


async def run(compiled: CompiledGraph, **kwargs: Any) -> ExecutionEvent:
    """Run one leg to its end and return the event it ended on.

    ``execute`` with the same arguments, drained: the ending is
    ``graph_complete`` (``results`` and ``record``), ``graph_pending`` (the
    questions a person must answer, and the ``record`` the next leg
    takes), ``graph_error``, ``graph_cancelled`` or ``graph_timeout``. A
    pause is an ending like any other, not an exception: the caller reads
    ``type`` and, for a pending leg, calls again with ``record`` and
    ``cache``. For code with no event loop, ``run_sync``.
    """
    ending: ExecutionEvent | None = None
    async for ending in execute(compiled, **kwargs):
        pass
    if ending is None:
        raise RuntimeError("the leg ended without an event")
    return ending


def run_sync(compiled: CompiledGraph, **kwargs: Any) -> ExecutionEvent:
    """``run`` for a script or a test: the same call under ``asyncio.run``.

    Inside a running event loop — a notebook, a server — it refuses,
    since ``asyncio.run`` cannot nest: ``await conductor.run(...)`` is the
    call there.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run(compiled, **kwargs))
    raise RuntimeError("a loop is running; use await conductor.run(...)")


# -- the leg ---------------------------------------------------------------------


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


class _Stop(enum.Enum):
    """Why the loop must end before the leg is quiet: the host's cancel
    event was set, or the leg's own deadline passed. What ``_Leg._next``
    returns in place of a message when one of its watchers fires first."""

    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"


class _Leg:
    """One leg in flight: the loop that starts ready units, and each unit's run.

    ``execute`` creates one per call and drops it when the leg ends; the
    ledger outlives it as the record every ending carries. It holds what the
    loop and the unit tasks share: the ledger, the thread pool every
    ``run`` is called on (``executor``), one semaphore per node sized by
    its policy's concurrency (``gates``), the nodes that have emitted
    ``node_start`` (``started``), the queue the unit tasks report on, and
    the tasks still running. The call chain is ``events`` (the loop) →
    ``_run_unit`` (one unit, retries included) → ``_call`` (the node, in a
    worker thread). Nothing reaches a node from here that its signature
    does not name.

    The pool has one worker per unit that may be in flight — the sum of
    the nodes' concurrency — so no unit ever queues for a thread and a
    node's timeout counts only the time its thread ran. Threads are made
    as needed, so an idle graph costs nothing. When the loop ends, for
    whatever reason, every unit task is cancelled and awaited and the pool
    is told to start nothing more; a thread mid-run finishes on its own.
    """

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
        self.started_at = time.monotonic()
        self.started = {node_id for node_id in compiled.execution_order if self.ledger.complete(node_id)}
        in_flight = sum(compiled.node(node_id).version.policy.concurrency for node_id in compiled.execution_order)
        self.executor = ThreadPoolExecutor(max_workers=max(1, in_flight), thread_name_prefix="conductor")
        self.gates: dict[str, asyncio.Semaphore] = {}
        self.queue: asyncio.Queue[ExecutionEvent | _UnitDone] = asyncio.Queue()
        self.running: dict[Unit, asyncio.Task[None]] = {}
        #: The two things that end the loop early, awaited beside the queue
        #: rather than polled; made in ``events``, where the loop runs.
        self._cancelled: asyncio.Task[bool] | None = None
        self._deadline: asyncio.Task[None] | None = None

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

        self._cancelled = asyncio.ensure_future(self.cancel.wait())
        self._deadline = None if self.timeout is None else asyncio.ensure_future(asyncio.sleep(self.timeout))
        try:
            self._start(self.ledger.runnable())
            while self.running:
                message = await self._next()
                if message is _Stop.CANCELLED:
                    yield GraphCancelledEvent(type="graph_cancelled", results=self.ledger.results(), record=self.ledger.cells())
                    return
                if message is _Stop.TIMED_OUT:
                    assert self.timeout is not None
                    yield GraphTimeoutEvent(
                        type="graph_timeout",
                        results=self.ledger.results(),
                        record=self.ledger.cells(),
                        elapsed_seconds=time.monotonic() - self.started_at,
                        timeout_seconds=self.timeout,
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
                        node_id=message.error["node_id"],
                        error=message.error["error"],
                        cause=message.error["cause"],
                        results=self.ledger.results(),
                        record=self.ledger.cells(),
                    )
                    return
                self._start(message.ready)
        finally:
            # Whatever ended the loop — quiet, an ending, or the consumer
            # closing the stream at a yield above — nothing outlives the leg.
            await self._teardown()

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

    async def _next(self) -> ExecutionEvent | _UnitDone | _Stop:
        """The next thing the loop acts on: a message from a unit, or the reason to stop.

        The queue, the cancel event and the deadline are awaited together;
        whichever is first wins, and a stop wins over a message that arrived
        in the same moment. A getter left waiting is cancelled, and the
        queue keeps the item for the next one."""
        assert self._cancelled is not None
        getter = asyncio.ensure_future(self.queue.get())
        watched = [getter, self._cancelled] + ([self._deadline] if self._deadline is not None else [])
        try:
            done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not getter.done():
                getter.cancel()
        if self._cancelled in done:
            return _Stop.CANCELLED
        if self._deadline in done:
            return _Stop.TIMED_OUT
        return getter.result()

    async def _teardown(self) -> None:
        """Stop every unit and the watchers, and let the threads go.

        Runs when the loop ends for any reason. A unit task is cancelled
        where it waits — on a gate, on its thread, in a retry's sleep — and
        awaited, so no task outlives the leg. The pool is shut down without
        waiting: a thread mid-run finishes on its own and what it returns
        is dropped, and a call that never started is not started."""
        tasks = list(self.running.values())
        self.running.clear()
        watchers = [watcher for watcher in (self._cancelled, self._deadline) if watcher is not None]
        for task in [*tasks, *watchers]:
            task.cancel()
        await asyncio.gather(*tasks, *watchers, return_exceptions=True)
        self.executor.shutdown(wait=False, cancel_futures=True)

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
