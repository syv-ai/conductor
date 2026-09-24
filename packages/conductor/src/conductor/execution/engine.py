"""Run a compiled graph: every ready unit, as soon as it is ready.

The engine doesn't schedule whole nodes. It schedules *units* — one call
of a node's ``run``. A node that runs once is one unit. A node that runs
once for each item in a list — each row of a series — is one unit per
item, so item 3 can move on down the graph while item 10 is still
waiting its turn. In the code a unit is ``(node_id, row)``, with row
``None`` for a node that runs once; a series is a value with many rows,
whose *index* names where the rows come from.

A *leg* is one call of ``execute``: it starts every unit it can, and
stops when nothing more can start — because the graph is done, or
because it's waiting for a person. The next leg picks up from what this
one recorded.

The leg's loop asks the ledger (the run's live state, every value produced
so far, ``conductor.execution.ledger``) which units are ready, starts
each under its node's concurrency limit and records what comes back; the
ledger answers each record with the units it made ready, and the loop
starts those — a unit finishing is what makes other units ready. A node
that receives a whole group of rows at once (a reduction) has its unit
ready once the whole group is.

A failed unit fails the run: the other units are cancelled and the cause,
with the row, goes out on the event stream.

**The leg owns its work.** Every node's ``run`` is called in a thread the
leg owns, from a pool with one worker per unit that may be in flight, so
a unit never waits for a worker and a node's timeout counts only the time
its thread ran. Closing the event stream — ``aclose()``, or a cancelled
consumer — stops every unit before the stream is gone. What the leg
cannot do is interrupt a thread: a ``run`` that has started
finishes on its own, and what it returns is dropped. A timed-out attempt
is therefore final, and the thread keeps the node's concurrency slot until
it returns.

**A run has legs.** A leg runs until nothing is runnable and nothing is
in flight. A unit whose node returned ``Asks`` is neither done nor
failed: it waits, everything that reads it waits, and the rest of the
graph runs on. If anything is waiting when the
leg goes quiet, the leg ends with ``graph_pending`` carrying every waiting
unit's questions, plus the run's state (a ``RunState``: every value in
wire form, and a fingerprint per node). The next leg is ``execute``
again, with ``state`` restoring the ledger and
the answers in ``cache`` as the asking node's outputs. Nothing is
checkpointed and nothing resumes: a leg is an ordinary run over a ledger
that already holds what earlier legs produced — and a node the graph has
changed since, or that reads one, runs again.

**Every ending has one shape.** ``graph_complete``, ``graph_pending``,
``graph_error``, ``graph_cancelled`` and ``graph_timeout`` all carry the
run's state beside their reason, so a host can start a new run from any
of them.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Mapping
from typing import Any

from conductor.errors import CompilationError
from conductor.execution.events import EndingEvent, ExecutionEvent
from conductor.execution.leg import Leg
from conductor.execution.state import RunState
from conductor.graph.compiled import CompiledGraph

# -- entry points ---------------------------------------------------------------


async def execute(
    compiled: CompiledGraph,
    *,
    state: RunState | None = None,
    cache: dict[str, dict[str, Any]] | None = None,
    from_run: Mapping[type, Any] | None = None,
    timeout: float | None = None,
    cancel: asyncio.Event | None = None,
) -> AsyncGenerator[ExecutionEvent, None]:
    """Run one leg of ``compiled`` and yield events as it goes.

    ``from_run`` supplies values to nodes by type: a ``run`` parameter
    annotated ``Annotated[X, FromRun()]`` receives ``from_run[X]``. A graph
    needing a type the host did not provide is refused before anything
    runs. ``state`` restores the ledger of an earlier leg, value by value —
    the ``RunState`` an ending carried; a host that stored its dump reads
    it back with ``RunState.model_validate``. A node the graph has changed
    since that leg, and everything reading it, is left out and runs again.
    ``cache`` records outputs by node id without running the node — a
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
    leg = Leg(
        compiled,
        state=state,
        from_run=from_run or {},
        timeout=timeout,
        cancel=cancel or asyncio.Event(),
    )
    async with leg:
        events = leg.events(cache or {})
        try:
            async for event in events:
                yield event
        finally:
            # Close the loop's generator here, under the consumer's close,
            # rather than leaving it to the garbage collector's finalizer,
            # so the leg's exit stops the units with nothing suspended in
            # it: they are stopped by the time the consumer's ``aclose()``
            # returns.
            await events.aclose()


async def run(compiled: CompiledGraph, **kwargs: Any) -> EndingEvent:
    """Run one leg to its end and return the event it ended on.

    ``execute`` with the same arguments, drained: the ending, one of
    ``EndingEvent`` from ``conductor.execution.events``, is ``graph_complete`` (``state``), ``graph_pending`` (the questions a
    person must answer, and the ``state`` the next leg
    takes), ``graph_error``, ``graph_cancelled`` or ``graph_timeout``. A
    pause is an ending like any other, not an exception: the caller reads
    ``type`` and, for a pending leg, calls again with ``state`` and
    ``cache``. For code with no event loop, ``run_sync``.
    """
    ending: ExecutionEvent | None = None
    async for ending in execute(compiled, **kwargs):
        pass
    if not isinstance(ending, EndingEvent):
        raise RuntimeError(f"the leg ended on {ending!r}, not an ending — an engine bug")
    return ending


def run_sync(compiled: CompiledGraph, **kwargs: Any) -> EndingEvent:
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
