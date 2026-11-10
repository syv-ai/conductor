"""``run_router``: the routes that start, follow, read, answer, resume and cancel runs, on any number of workers.

A worker runs each leg in the background, holds a lease on its run while it
does, and writes a checkpoint every few seconds. Another worker takes over a
run whose lease ran out, from its last checkpoint, and a worker that shuts
down hands its runs back. Every write is a compare-and-set on the run's
``rev``, so of two workers acting on one run, one goes on and the other is
told so.
"""

import asyncio
import contextlib
import json
import logging
import time
import typing
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Annotated, Any, TypeVar

import polars as pl
from conductor import (
    GraphDescription,
    GraphRecord,
    InvalidCallError,
    NodeRegistry,
    Problem,
    RegistryDescription,
    Run,
    RunStatus,
    Waiting,
)
from conductor.events import RunCheckpoint, RunEnd
from fastapi import APIRouter, Body, Depends, FastAPI, Header, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, PositiveInt, TypeAdapter
from pydantic_core import to_json

from conductor_providers.fastapi.store import RunChanges, RunStore, StoredRun

C = TypeVar("C")

log = logging.getLogger("conductor_providers.fastapi")

#: The statuses a run can be resumed from; ``complete`` is the only one that is final.
_RESUMABLE = ("timed_out", "stopped", "cancelled", "failed")


class StartRun(BaseModel):
    """What ``POST /runs`` takes: a graph and its inputs, by name or by address."""

    graph: GraphRecord
    inputs: dict[str, Any] = {}


class RunView(BaseModel):
    """What a client reads about a run."""

    id: str
    status: RunStatus
    waiting: list[Waiting] = []
    outputs: dict[str, Any] | None = None
    #: Answers sent while a leg ran that were refused when it ended, by waiting id.
    refused: dict[str, list[Problem]] = {}
    #: Why the run failed, stopped, was cancelled or timed out; ``None`` while it runs, waits or once complete.
    cause: Problem | None = None


class Started(BaseModel):
    """A run that was started; ``GET /runs/{id}`` reads it."""

    id: str


class Answered(BaseModel):
    """``started``: the answers started the next leg. ``queued``: a leg was running, so they are applied when it ends."""

    answers: typing.Literal["started", "queued"]


class Cancelled(BaseModel):
    """Whether a running leg was asked to stop; a run that isn't running has nothing to cancel."""

    cancelled: bool


@dataclass
class _Leg:
    """A leg this worker runs: what stops it, and the ``rev`` its checkpoints and renewals last wrote."""

    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    rev: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def run_router(
    *,
    store: RunStore,
    registry: NodeRegistry | Callable[[C], NodeRegistry],
    caller: Callable[..., C] | None = None,
    from_run: Callable[[C], Mapping[type, Any]] | None = None,
    may_see: Callable[[C, C], bool] | None = None,
    published: Callable[..., GraphRecord] | None = None,
    max_rows: int | None = None,
    lease: float = 30.0,
    checkpoint_every: float = 5.0,
    leg_timeout: float | None = 300.0,
    poll: float = 0.1,
) -> APIRouter:
    """The runs routes, with the worker that runs their legs; mount them with ``app.include_router(...)``.

    - ``store`` keeps runs and events (``RunStore``).
    - ``registry`` is the nodes graphs may use, or a function of the caller giving them, for a registry per tenant.
      It is called again for every request and every leg, so a function caches what it builds.
    - ``caller`` is a FastAPI dependency giving who is calling: your login. The type it gives (a class used as the
      dependency, or a function's return annotation, which it must have) is kept on the run as JSON, so a run is
      built again on any worker.
    - ``from_run`` gives a leg its ``FromRun`` values from the caller who started it. A leg runs on threads of its
      own, so a value is a way to connect, never an open connection.
    - ``may_see(caller, started_by)`` says whether a caller may read and act on a run; by default only the caller
      who started it may.
    - ``published`` is a FastAPI dependency taking the path's ``name`` and giving a ``GraphRecord``, raising
      ``HTTPException(404)`` for one it doesn't offer; with it, ``POST /graphs/{name}/runs`` starts that graph on
      inputs alone.
    - ``max_rows`` is the ceiling on the rows of one table a run starts with; a resume may raise it for that run,
      and the raised ceiling holds for its later legs. ``lease`` the seconds a worker holds a run between
      renewals; ``checkpoint_every`` the seconds between saves of a running leg; ``leg_timeout`` the seconds one leg
      may run; ``poll`` the seconds between a follower's reads of new events.
    """
    worker = uuid.uuid4().hex
    legs: dict[str, _Leg] = {}
    tasks: set[asyncio.Task[None]] = set()
    stopping = False
    caller_dependency: Callable[..., Any] = caller if caller is not None else _nobody
    callers: TypeAdapter[Any] = TypeAdapter(_caller_type(caller_dependency))
    sees = may_see if may_see is not None else _same

    def nodes_for(who: Any) -> NodeRegistry:
        return registry if isinstance(registry, NodeRegistry) else registry(who)

    def started_by(row: StoredRun) -> Any:
        return callers.validate_json(row.caller)

    def load(row: StoredRun, who: Any = None) -> Run:
        """The stored run, on the nodes of ``who``, or of the caller who started it."""
        return nodes_for(who if who is not None else started_by(row)).load(row.graph).load_run(row.run)

    # -- legs -------------------------------------------------------------------------------------

    async def claim(row: StoredRun, changes: RunChanges | None = None) -> bool:
        """Take ``row`` for this worker if nobody holds it and it is still at the ``rev`` read, writing ``changes`` too.
        Of two workers, one wins. A cancel asked of the run stays asked: only a resume clears it."""
        if row.owner is not None and row.lease_until >= time.time():
            return False
        return await store.update(row.id, row.rev, {**held(), **(changes or {})})

    def held() -> RunChanges:
        """What a row says while this worker holds it: running, with no cause yet."""
        return {"status": "running", "owner": worker, "lease_until": time.time() + lease, "cause": None}

    def launch(run_id: str, rev: int, ceiling: int | None = None, cancelled: bool = False) -> None:
        """Run the next leg of a run this worker has just claimed at ``rev``, in the background; ``cancelled`` when a
        cancel was asked of it, which the leg then hears before its first unit."""
        leg = legs[run_id] = _Leg(rev=rev)
        if cancelled:
            leg.cancel.set()
        task = asyncio.get_running_loop().create_task(drive(run_id, leg, ceiling))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def write(run_id: str, leg: _Leg, changes: RunChanges) -> bool:
        """Write a checkpoint or a renewal while this worker holds the run. Another writer (an answer queued, a
        cancel) moves ``rev`` on: read again and go on. A run another worker took, or that stopped running, stops the
        leg and writes nothing."""
        async with leg.lock:
            while True:
                if await store.update(run_id, leg.rev, changes):
                    leg.rev += 1
                    return True
                row = await store.get(run_id)
                if row is None or row.owner != worker or row.status != "running":
                    leg.cancel.set()
                    return False
                if row.cancel:
                    leg.cancel.set()
                leg.rev = row.rev

    async def drive(run_id: str, leg: _Leg, ceiling: int | None) -> None:
        """Run the leg, and write its end while its lease is still renewed; a leg that raised ends ``failed``."""
        keep_lease = asyncio.create_task(hold(run_id, leg))
        try:
            row = await store.get(run_id)
            assert row is not None
            run = await asyncio.to_thread(load, row)
            values = from_run(started_by(row)) if from_run is not None else None
            stream = run.stream(
                from_run=values, max_rows=ceiling, timeout=leg_timeout,
                cancel=leg.cancel, checkpoint_every=checkpoint_every,
            )
            async with contextlib.aclosing(stream):  # a raise in this loop ends the leg before its next unit
                async for event in stream:
                    if isinstance(event, RunCheckpoint):
                        await write(run_id, leg, {"run": await asyncio.to_thread(event.run.to_json)})
                        continue
                    if await store.add_event(run_id, worker, event.model_dump_json()) is None:
                        leg.cancel.set()  # another worker holds the run
                    if isinstance(event, RunEnd):
                        await finish(run_id, leg, keep_lease, event.run)
        except Exception:
            log.exception("the leg of run %s failed", run_id)
            await store.add_event(run_id, worker, '{"type":"run_end","status":"failed","cause":null}')
            await finish(run_id, leg, keep_lease, None)
        finally:
            keep_lease.cancel()
            if legs.get(run_id) is leg:  # the answers queued during this leg may have launched the next one already
                del legs[run_id]

    async def finish(run_id: str, leg: _Leg, keep_lease: asyncio.Task[None], run: Run | None) -> None:
        """Write how a leg ended with ``run`` (``None`` when it raised), decided from the row as read for each try of
        the write, so a cancel or an answer that lands meanwhile is never written over.

        A leg this worker stopped to shut down, with no cancel asked of the run, is handed back for any worker to go on
        with. Otherwise the answers queued during the leg are taken when it ended ``waiting``, and the next leg starts
        on them; when it ended needing a resume, each is refused as ``run_not_waiting``. A run another worker took is
        left as it is.

        ``keep_lease`` renews the lease until the end is written, and each JSON of a run is made once, outside the
        write's tries: a run however long it takes to write as JSON stays this worker's, and a try lost to a renewal
        or a queued answer costs a read and a write."""
        status: RunStatus = run.status if run is not None else "failed"
        written: RunChanges = await asyncio.to_thread(_ended, run) if run is not None else {"cause": None}
        handed_back: RunChanges = {"status": "running", "owner": None, "lease_until": 0.0}
        as_ended: str | None = None
        answered_from: tuple[str, str] | None = None
        answered: tuple[str | None, dict[str, Any]] = (None, {})
        try:
            while True:
                row = await store.get(run_id)
                if row is None or row.owner != worker or row.status != "running":
                    return
                saved = None
                if stopping and status == "cancelled" and not row.cancel:
                    changes: RunChanges = {**written, **handed_back}
                else:
                    changes = {**written, "status": status, "owner": None, "lease_until": 0.0, "cancel": False}
                    if row.pending != "{}":
                        if answered_from != (row.pending, row.refused):
                            answered_from = (row.pending, row.refused)
                            answered = await asyncio.to_thread(_answered, run, json.loads(row.pending), json.loads(row.refused))
                        saved, refused = answered
                        changes.update({"pending": "{}", "refused": to_json(refused).decode()})
                    if saved is not None:  # a worker shutting down hands the next leg back rather than start it
                        changes.update({"run": saved, **(handed_back if stopping else held()), "cause": None})
                if saved is None and run is not None:
                    if as_ended is None:
                        as_ended = await asyncio.to_thread(run.to_json)
                    changes["run"] = as_ended
                async with leg.lock:  # no renewal lands between this write and the lease's last renewal
                    ended = await store.update(run_id, row.rev, changes)
                    if ended:
                        keep_lease.cancel()
                if ended:
                    await _stopped(keep_lease)
                    if saved is not None and not stopping:
                        launch(run_id, row.rev + 1)
                    return
        finally:
            await _stopped(keep_lease)

    async def hold(run_id: str, leg: _Leg) -> None:
        """Renew the lease until the leg ends; a renewal that finds the run taken stops the leg."""
        while True:
            await asyncio.sleep(lease / 3)
            if not await write(run_id, leg, {"lease_until": time.time() + lease}):
                return

    async def sweep() -> None:
        """Go on with the runs whose worker died or handed them back."""
        while True:
            for row in await store.expired(time.time()):
                if row.id not in legs and await claim(row):
                    launch(row.id, row.rev + 1, cancelled=row.cancel)
            await asyncio.sleep(lease / 3)

    async def start(graph: GraphRecord, inputs: Mapping[str, Any], who: Any) -> Started:
        try:
            saved = await asyncio.to_thread(lambda: nodes_for(who).load(graph).with_inputs(inputs, max_rows=max_rows).to_json())
        except InvalidCallError as error:
            raise _unprocessable(error) from None
        return await begin(graph.model_dump_json(), who, saved)

    async def begin(graph: str, who: Any, saved: str) -> Started:
        """Store a run of ``who``'s, held by this worker, and run its first leg."""
        run_id = uuid.uuid4().hex
        await store.create(replace(StoredRun(id=run_id, graph=graph, caller=callers.dump_json(who).decode(), run=saved, status="running"), **held()))
        launch(run_id, 0)
        return Started(id=run_id)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        nonlocal stopping
        sweeping = asyncio.create_task(sweep())
        yield
        stopping = True
        sweeping.cancel()
        for leg in legs.values():
            leg.cancel.set()  # a cancel keeps the calls already made, so the next worker repeats none
        await asyncio.gather(*tasks, return_exceptions=True)

    # -- routes -----------------------------------------------------------------------------------

    router = APIRouter(lifespan=lifespan)
    Caller = Annotated[Any, Depends(caller_dependency)]

    async def stored(run_id: str, who: Caller) -> StoredRun:
        row = await store.get(run_id)
        if row is None or not sees(who, started_by(row)):
            raise HTTPException(404, "no such run")
        return row

    Stored = Annotated[StoredRun, Depends(stored)]

    @router.get("/nodes")
    def palette(who: Caller) -> RegistryDescription:
        """The nodes the caller may use, as an editor's palette reads them."""
        return nodes_for(who).describe()

    @router.post("/load")
    def check(graph: GraphRecord, who: Caller) -> GraphDescription:
        """The graph as loading reads it: each node's fields, where it runs, and every problem."""
        return nodes_for(who).load(graph).describe()

    @router.post("/runs", status_code=202)
    async def start_run(body: StartRun, who: Caller) -> Started:
        """Start a run of any graph; this is an author's route, so guard it with your own dependency."""
        return await start(body.graph, body.inputs, who)

    if published is not None:
        Published = Annotated[GraphRecord, Depends(published)]

        @router.post("/graphs/{name}/runs", status_code=202)
        async def start_published(graph: Published, inputs: dict[str, Any], who: Caller) -> Started:
            """Start a graph you offer by name, on inputs alone."""
            return await start(graph, inputs, who)

    @router.get("/runs/{run_id}")
    def read(row: Stored) -> RunView:
        return RunView(
            id=row.id, status=row.status, waiting=json.loads(row.waiting),
            outputs=json.loads(row.outputs) if row.outputs is not None else None, refused=json.loads(row.refused),
            cause=json.loads(row.cause) if row.cause is not None else None,
        )

    @router.get("/runs/{run_id}/events", response_class=EventSourceResponse)
    async def follow(row: Stored, last_event_id: Annotated[int | None, Header()] = None) -> AsyncIterable[ServerSentEvent]:
        """The run's events, from whichever worker runs it, until it stops running; ``Last-Event-ID`` goes on after a reconnect."""
        seen = -1 if last_event_id is None else last_event_id
        while True:
            new = await store.events(row.id, seen)
            for seq, body in new:
                seen = seq
                yield ServerSentEvent(raw_data=body, id=str(seq))
            if not new:
                now = await store.get(row.id)
                if now is None or now.status != "running":
                    for seq, body in await store.events(row.id, seen):  # written before the status changed
                        yield ServerSentEvent(raw_data=body, id=str(seq))
                    return
                await asyncio.sleep(poll)

    @router.post("/runs/{run_id}/answers", status_code=202)
    async def answer(row: Stored, answers: dict[str, Any]) -> Answered:
        """Answer waiting rows by their ids. While a leg runs, the answers wait for it to end, and a row answered once
        meanwhile is refused a second answer. An answer to a row whose question changed since it was read is refused by
        its waiting id (``answer_stale``)."""
        while row.status == "running":
            queued = json.loads(row.pending)
            if queued.keys() & answers.keys():
                raise HTTPException(409, "the row is already answered")
            queued.update(answers)
            if await store.update(row.id, row.rev, {"pending": to_json(queued).decode()}):
                return Answered(answers="queued")  # the leg's end, written after this, reads them
            again = await store.get(row.id)
            if again is None:
                raise HTTPException(404, "no such run")
            row = again
        if row.status != "waiting":
            raise HTTPException(409, _not_waiting(row.status).msg)
        try:  # every wrong answer is refused at once
            answered = await asyncio.to_thread(lambda: load(row).answer(answers).to_json())
        except InvalidCallError as error:
            raise _unprocessable(error) from None
        if not await claim(row, {"run": answered}):
            raise HTTPException(409, "the run changed since it was read")
        launch(row.id, row.rev + 1)
        return Answered(answers="started")

    @router.post("/runs/{run_id}/resume", status_code=202)
    async def resume(row: Stored, max_rows: Annotated[PositiveInt | None, Body(embed=True)] = None) -> Started:
        """Go on with a run that timed out, stopped at its ceiling, was cancelled or failed; nothing finished is called
        again. ``{"max_rows": ...}`` raises the ceiling of a run that stopped at it; a resume needs no body."""
        if row.status not in _RESUMABLE:
            raise HTTPException(409, f"the run is {row.status}")
        if not await claim(row, {"cancel": False}):
            raise HTTPException(409, "the run changed since it was read")
        launch(row.id, row.rev + 1, max_rows)
        return Started(id=row.id)

    @router.post("/runs/{run_id}/again", status_code=202)
    async def again(row: Stored, inputs: dict[str, Any], who: Caller) -> Started:
        """A new run of the caller's with these inputs changed, on the caller's own nodes; a row that receives what it
        received before isn't called again."""
        try:
            saved = await asyncio.to_thread(lambda: load(row, who).with_inputs(inputs).to_json())
        except InvalidCallError as error:
            raise _unprocessable(error) from None
        return await begin(row.graph, who, saved)

    @router.post("/runs/{run_id}/cancel")
    async def cancel(row: Stored) -> Cancelled:
        """Stop a running leg before its next unit; the calls already made are kept, and ``resume`` goes on."""
        while row.status == "running":
            if await store.update(row.id, row.rev, {"cancel": True}):
                if row.id in legs:  # on this worker at once; on another at its next renewal
                    legs[row.id].cancel.set()
                return Cancelled(cancelled=True)
            again = await store.get(row.id)
            if again is None:
                break
            row = again
        return Cancelled(cancelled=False)

    return router


def _ended(run: Run) -> RunChanges:
    """What a leg's end writes of ``run`` beside the run itself: its waiting rows, its outputs once complete (a table as
    records, wherever it sits, a float that isn't finite as text, as every JSON conductor writes), and its cause."""
    changes: RunChanges = {
        "waiting": to_json([unit.model_dump(mode="json") for unit in run.waiting]).decode(),
        "cause": run.cause.model_dump_json() if run.cause is not None else None,
    }
    if run.status == "complete":
        changes["outputs"] = to_json(run.outputs, fallback=_records, inf_nan_mode="strings").decode()
    return changes


async def _stopped(task: asyncio.Task[None]) -> None:
    """Cancel ``task`` and wait until it has stopped."""
    task.cancel()
    await asyncio.wait([task])


def _answered(run: Run | None, pending: dict[str, Any], refused: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """The answers queued during a leg that ended with ``run`` (``None`` when it raised), each on its own: the run they
    give as JSON when any was taken, else ``None``, and ``refused`` with the problems of each one that wasn't. A run
    that ended needing a resume takes none (``run_not_waiting``); otherwise a wrong answer, or one whose row waits for
    nothing (``not_waiting``, as on a complete run), keeps its own problems."""
    if run is None or run.status in _RESUMABLE:
        problem = _not_waiting(run.status if run is not None else "failed").model_dump(mode="json")
        return None, {**refused, **{unit_id: [problem] for unit_id in pending}}
    taken = False
    for unit_id, value in pending.items():
        try:
            run, taken = run.answer(unit_id, value), True
        except InvalidCallError as error:
            refused[unit_id] = [problem.model_dump(mode="json") for problem in error.errors()]
    return (run.to_json() if taken else None), refused


def _not_waiting(status: RunStatus) -> Problem:
    """Why a run that isn't ``waiting`` takes no answers: a leg starts only on a run that waits for them. The code and
    its message are in ``PROBLEM_CODES`` and conductor's catalogue, as every code conductor writes."""
    return Problem(type="run_not_waiting", msg=f"The run is {status}, so it takes no answers; resume it first, then answer.", ctx={"status": status})


def _records(value: Any) -> Any:
    if isinstance(value, pl.DataFrame):
        return value.to_dicts()
    raise TypeError(type(value).__name__)


def _caller_type(caller: Callable[..., Any]) -> Any:
    """The type a ``caller`` dependency gives: the class itself, or a function's return annotation, which it must have."""
    if isinstance(caller, type):
        return caller
    returns = typing.get_type_hints(caller).get("return")
    if returns is None:
        raise TypeError(f"caller {caller!r} has no return annotation; annotate it with the type it gives, so a run keeps its caller as that type")
    return returns


def _nobody() -> None:
    """The caller when the host names none."""
    return None


def _same(caller: Any, started_by: Any) -> bool:
    return bool(caller == started_by)


def _unprocessable(error: InvalidCallError) -> HTTPException:
    return HTTPException(422, [problem.model_dump(mode="json") for problem in error.errors()])


