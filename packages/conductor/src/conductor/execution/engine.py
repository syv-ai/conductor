"""Run a compiled graph: every ready unit, as soon as it is ready.

A unit is ``(node, row)``: a node that runs once is one unit with row
``None``; a node that runs once per row of a series — a value with many
rows, whose *index* names where the rows come from — is one unit per
row. The loop asks the ledger (the run's record of every value produced
so far, ``conductor.execution.ledger``) which units are ready, starts
each under its node's concurrency limit, records what comes back and
asks again — a unit finishing is what makes other units ready. Row 1 of
a chain can finish before row 10 of the first node has started; a node
that receives a whole group of rows at once (a reduction) has its unit
ready once the whole group is.

A failed unit fails the run: the other units are cancelled and the cause,
with the row, goes out on the event stream.

**A run has legs.** One call of ``execute`` is one leg, and it runs until
nothing is runnable and nothing is in flight. A unit whose node returned
``Asks`` is neither done nor failed: it waits, everything that reads it
waits, and the rest of the graph runs on. If anything is waiting when the
leg goes quiet, the leg ends with ``graph_pending`` carrying every waiting
unit's questions, plus what the leg completed and the ledger's cells (its
whole record, row by row). The next leg is ``execute`` again, with
``cells`` restoring the ledger and the
answers in ``cache`` as the asking node's outputs. Nothing is checkpointed
and nothing resumes: a leg is an ordinary run over a ledger that already
holds what earlier legs produced.

**Every ending has one shape.** ``graph_complete``, ``graph_pending``,
``graph_error``, ``graph_cancelled`` and ``graph_timeout`` all carry the
results so far and the ledger's cells beside their reason, so a host can
start a new run from any of them.
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import AsyncGenerator, Mapping
from dataclasses import dataclass, replace
from typing import Any

from pydantic import ValidationError

from conductor._sentinel import is_asking
from conductor.errors import (
    CompilationError,
    ErrorCause,
    GraphExecutionError,
    GraphPendingError,
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
from conductor.graph.compiled import CompiledGraph
from conductor.interface import model_of
from conductor.returns import unpack

# -- entry points ---------------------------------------------------------------


async def execute(
    compiled: CompiledGraph,
    *,
    timeout_seconds: int = 300,
    from_run: Mapping[type, Any] | None = None,
    cache: dict[str, dict[str, Any]] | None = None,
    cells: dict[str, Any] | None = None,
    cancel: asyncio.Event | None = None,
) -> AsyncGenerator[ExecutionEvent, None]:
    """Run one leg of ``compiled`` and yield events as it goes.

    ``from_run`` supplies values to nodes by type: a ``run`` parameter
    annotated ``Annotated[X, FromRun()]`` receives ``from_run[X]``. A graph
    needing a type the host did not provide is refused before anything
    runs. ``cells`` restores the ledger of an earlier leg, cell by cell.
    ``cache`` pre-seeds nodes with complete outputs by node id — a person's
    answers to a pending unit, or an earlier run's results a caller reuses;
    they are reported as ``node_complete`` (``cached=True``) and not run.
    ``cancel`` is an event the host sets to stop the leg::

        async for event in execute(compiled, from_run={Clock: clock}):
            ...
    """
    if not compiled.is_runnable:
        raise CompilationError("the flow cannot run", problems=compiled.problems)
    leg = _Leg(
        compiled,
        cells=cells,
        from_run=from_run or {},
        timeout_seconds=timeout_seconds,
        cancel=cancel or asyncio.Event(),
    )
    async for event in leg.events(cache or {}):
        yield event


async def collect(events: AsyncGenerator[ExecutionEvent, None]) -> dict[str, dict[str, Any]]:
    """Drain an event stream and return the results, or raise for the ending that stopped it."""
    async for event in events:
        kind = event["type"]
        if kind == "graph_complete":
            return event["results"]
        if kind == "graph_pending":
            raise GraphPendingError(event["pending"], event["cells"])
        if kind == "graph_error":
            raise GraphExecutionError(event["error"], node_id=event.get("node_id"), cause=event.get("cause"))
        if kind in ("graph_cancelled", "graph_timeout"):
            raise GraphExecutionError(f"The flow was stopped ({kind}).")
    raise GraphExecutionError("The flow ended without a result.")


def execute_sync(compiled: CompiledGraph, **kwargs: Any) -> dict[str, dict[str, Any]]:
    """Run one leg synchronously and return ``{node_id: {output: value}}``.

    A convenience for tests and scripts. A leg that ends pending raises
    ``GraphPendingError`` with the questions and the cells; the caller
    answers and calls again with ``cells`` and ``cache``.
    """
    return asyncio.run(collect(execute(compiled, **kwargs)))


# -- the leg ---------------------------------------------------------------------


@dataclass
class _UnitDone:
    """A unit finished, so the loop should dispatch again.

    The only thing a unit's task puts on the queue besides events.
    ``error`` is set when the unit failed, which fails the leg.
    """

    unit: Unit
    error: NodeErrorEvent | None = None


class _Leg:
    """One leg in flight: the loop that starts ready units, and each unit's run.

    ``execute`` creates one per call and drops it when the leg ends; the
    ledger outlives it as the cells every ending carries. It holds what the
    loop and the unit tasks share: the ledger, one semaphore per node sized
    by its policy's concurrency (``gates``), the nodes that have emitted
    ``node_start`` (``started``), the queue the unit tasks report on, and
    the tasks still running. The call chain is ``events`` (the loop) →
    ``_run_unit`` (one unit, retries included) → ``_call`` (the node, in a
    worker thread). Nothing reaches a node from here that its signature
    does not name.
    """

    def __init__(
        self,
        compiled: CompiledGraph,
        *,
        cells: dict[str, Any] | None,
        from_run: Mapping[type, Any],
        timeout_seconds: int,
        cancel: asyncio.Event,
    ) -> None:
        self.compiled = compiled
        self.ledger = Ledger(compiled) if cells is None else Ledger.restore(compiled, cells)
        for node_id in compiled.execution_order():
            for name, needed in compiled.node(node_id).version.interface.needs.items():
                if needed not in from_run:
                    raise TypeError(
                        f"'{node_id}' needs a {needed.__name__} for '{name}', and execute() was not given one"
                    )
        self.from_run = dict(from_run)
        self.timeout_seconds = timeout_seconds
        self.cancel = cancel
        self.started_at = time.monotonic()
        self.started = {node_id for node_id in compiled.execution_order() if self.ledger.complete(node_id)}
        self.gates: dict[str, asyncio.Semaphore] = {}
        self.queue: asyncio.Queue[ExecutionEvent | _UnitDone] = asyncio.Queue()
        self.running: dict[Unit, asyncio.Task[None]] = {}

    # -- the loop ----------------------------------------------------------------

    async def events(self, cache: dict[str, dict[str, Any]]) -> AsyncGenerator[ExecutionEvent, None]:
        """Seed the cached nodes, then start every ready unit until the leg ends."""
        for node_id, outputs in cache.items():
            self.ledger.inject(node_id, outputs)
            self.started.add(node_id)
            yield NodeCompleteEvent(type="node_complete", node_id=node_id, result=outputs, cached=True)

        self._dispatch()
        while self.running:
            if self.cancel.is_set():
                self._stop()
                yield GraphCancelledEvent(type="graph_cancelled", results=self.ledger.results(), cells=self.ledger.cells())
                return
            if self._remaining() == 0:
                self._stop()
                yield GraphTimeoutEvent(
                    type="graph_timeout",
                    results=self.ledger.results(),
                    cells=self.ledger.cells(),
                    elapsed_seconds=time.monotonic() - self.started_at,
                    timeout_seconds=self.timeout_seconds,
                )
                return
            try:
                message = await asyncio.wait_for(self.queue.get(), timeout=0.5)
            except TimeoutError:
                continue
            if not isinstance(message, _UnitDone):
                yield message
                continue
            self.running.pop(message.unit, None)
            if message.error is not None:
                self._stop()
                yield message.error
                yield GraphErrorEvent(
                    type="graph_error",
                    node_id=message.error["node_id"],
                    error=message.error["error"],
                    cause=message.error["cause"],
                    results=self.ledger.results(),
                    cells=self.ledger.cells(),
                )
                return
            self._dispatch()

        # Quiescence: nothing runnable, nothing in flight.
        pending = self.ledger.pending()
        if pending:
            yield GraphPendingEvent(
                type="graph_pending", pending=pending, results=self.ledger.results(), cells=self.ledger.cells()
            )
            return
        unfinished = [node_id for node_id in self.compiled.execution_order() if not self.ledger.complete(node_id)]
        if unfinished:
            raise RuntimeError(f"nothing left to run, but {unfinished} did not complete — an engine bug")
        yield GraphCompleteEvent(type="graph_complete", results=self.ledger.results(), cells=self.ledger.cells())

    def _dispatch(self) -> None:
        """Start a task for every unit that is ready and not already running, done or waiting."""
        ledger = self.ledger
        for node_id in self.compiled.execution_order():
            if ledger.complete(node_id):
                continue
            for unit in ledger.units(node_id):
                if unit in self.running or ledger.is_done(unit) or ledger.is_pending(unit) or not ledger.ready(unit):
                    continue
                task = asyncio.create_task(self._run_unit(unit), name=f"unit-{unit}")
                task.add_done_callback(functools.partial(self._settled, unit))
                self.running[unit] = task

    def _settled(self, unit: Unit, task: asyncio.Task[None]) -> None:
        """A unit's task ended. One that raised is a defect in the engine, and
        is reported as the unit's failure, since a task that died silently
        would leave the loop waiting."""
        if task.cancelled():
            return
        raised = task.exception()
        if raised is None:
            return
        node_id, row = unit
        failure = NodeExecutionError(
            f"{type(raised).__name__}: {raised}", node_id=node_id, original=raised,
            cause=self._cause(code="engine_error", message="An error in the engine stopped the run.", row=row),
        )
        self.queue.put_nowait(_UnitDone(unit, error=self._error_event(node_id, failure, row)))

    def _stop(self) -> None:
        for task in self.running.values():
            task.cancel()
        self.running.clear()

    def _remaining(self) -> float:
        return max(0.0, self.timeout_seconds - (time.monotonic() - self.started_at))

    def _gate(self, node_id: str) -> asyncio.Semaphore:
        if node_id not in self.gates:
            self.gates[node_id] = asyncio.Semaphore(self.compiled.node(node_id).version.policy.concurrency)
        return self.gates[node_id]

    # -- one unit ----------------------------------------------------------------

    async def _run_unit(self, unit: Unit) -> None:
        """One unit, start to finish: its inputs, its retries, and what it produced."""
        node_id, row = unit
        compiled, ledger, queue = self.compiled, self.ledger, self.queue
        try:
            inputs = ledger.inputs_for(unit)
        except NodeError as failure:  # two edges into one input both supplied this row
            await queue.put(_UnitDone(unit, error=self._error_event(node_id, failure, row)))
            return
        if isinstance(inputs, Skip):
            ledger.record(unit, inputs)
            await self._after(unit)
            await queue.put(_UnitDone(unit))
            return

        if node_id not in self.started:
            self.started.add(node_id)
            await queue.put(NodeStartEvent(type="node_start", node_id=node_id))

        version = compiled.node(node_id).version
        policy = version.policy
        attempt = 0
        async with self._gate(node_id):
            while True:
                try:
                    budget = self._remaining() if policy.timeout is None else min(self._remaining(), policy.timeout)
                    value = await asyncio.wait_for(asyncio.to_thread(self._call, unit, inputs), timeout=max(0.05, budget))
                    break
                except TimeoutError:
                    if self._remaining() == 0:
                        return  # the loop reports the timeout
                    failure: NodeError = NodeTimeoutError(
                        f"'{node_id}' exceeded its time limit of {policy.timeout}s", node_id=node_id,
                        cause=self._cause(code="timeout", message="The node did not answer in time.", row=row, details={"seconds": policy.timeout}),
                    )
                except NodeError as raised:
                    failure = raised
                if not failure.retryable or attempt >= policy.retries:
                    await queue.put(_UnitDone(unit, error=self._error_event(node_id, failure, row)))
                    return
                attempt += 1
                delay = policy.delay * (2 ** (attempt - 1))
                await queue.put(NodeRetryEvent(
                    type="node_retry", node_id=node_id, row=None if row is None else list(row),
                    attempt=attempt, retries=policy.retries, error=str(failure), delay=delay,
                ))
                await asyncio.sleep(delay)

        if is_asking(value):
            # A person must answer. The unit waits; the rest of the leg runs
            # on and the leg ends pending once quiet.
            ledger.pend(unit, value.questions, value.prompt)
            await queue.put(_UnitDone(unit))
            return
        outputs = unpack(version.interface.returns, value, compiled.node(node_id).interface.outputs)
        ledger.record(unit, outputs)
        await self._after(unit)
        await queue.put(_UnitDone(unit))

    def _call(self, unit: Unit, inputs: dict[str, Any]) -> Any:
        """Validate the inputs against the node's interface, then call the node.

        The interface is the list of inputs the node actually has, which can be
        more than its declaration: a ``compute_inputs`` hook may add some.
        Validating against the interface rather than the declaration means those
        added inputs are validated like any other. ``FromRun`` parameters
        (values the host supplies by type) come from the run. Runs in a
        worker thread. A ``NodeError`` the node raises passes through; any
        other exception becomes a ``NodeExecutionError`` with the row in its
        cause.
        """
        node_id, row = unit
        node = self.compiled.node(node_id)
        version = node.version
        try:
            validated = model_of(node.interface.inputs)(**inputs)
        except ValidationError as invalid:
            reason = self._describe(invalid, node.interface.inputs)
            raise NodeValidationError(
                reason, node_id=node_id, original=invalid,
                cause=self._cause(code="invalid_input", message="The node received a value it cannot use.", row=row, details={"reason": reason}),
            ) from invalid
        kwargs = {name: getattr(validated, name) for name in type(validated).model_fields}
        kwargs.update({name: self.from_run[needed] for name, needed in version.interface.needs.items()})
        runner = node.runner
        try:
            return runner(**kwargs)
        except NodeError:
            raise
        except Exception as raised:
            raise NodeExecutionError(
                f"{type(raised).__name__}: {raised}", node_id=node_id, original=raised,
                cause=self._cause(code="execution_failed", message=str(raised), row=row),
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
    def _cause(*, code: str, message: str, row: Any, details: Mapping[str, Any] | None = None) -> ErrorCause:
        return ErrorCause(code=code, message=message, details=details or {}, row=row)

    @staticmethod
    def _error_event(node_id: str, failure: NodeError, row: Any) -> NodeErrorEvent:
        cause = failure.cause if failure.cause is not None else _Leg._cause(code="failed", message=str(failure), row=row)
        if cause.row is None and row is not None:
            cause = replace(cause, row=row)
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
