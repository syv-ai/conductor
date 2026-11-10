"""``run_router``: the routes that start, follow, read, answer, resume and cancel runs, on any number of workers.

A worker runs each leg in the background, holds a lease on its run while it
does, and writes a checkpoint every few seconds. Another worker takes over a
run whose lease ran out, from its last checkpoint, and a worker that shuts
down hands its runs back. Every write is a compare-and-set on the run's
``rev``, so of two workers acting on one run, one goes on and the other is
told so.
"""

import asyncio
import json
import logging
import time
import typing
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
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
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, TypeAdapter
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


class Resume(BaseModel):
    """What ``POST /runs/{id}/resume`` takes: a higher ceiling for a run that stopped at its ``max_rows``."""

    max_rows: int | None = None


class RunView(BaseModel):
    """What a client reads about a run. ``rev`` goes in ``If-Match`` to answer only the run as read."""

    id: str
    status: RunStatus
    rev: int
    waiting: list[Waiting] = []
    outputs: dict[str, Any] | None = None
    #: Answers sent while a leg ran that were refused when it ended, by waiting id.
    refused: dict[str, list[Problem]] = {}


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
    """A leg this worker runs: what stops it, the ``rev`` it last wrote, and why it was stopped."""

    cancel: asyncio.Event = field(default_factory=asyncio.Event)
    rev: int = 0
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    asked: bool = False  # someone cancelled the run, as opposed to this worker shutting down


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
    - ``caller`` is a FastAPI dependency giving who is calling: your login. Its return annotation is the type kept
      on the run as JSON, so a run is built again on any worker.
    - ``from_run`` gives a leg its ``FromRun`` values from the caller who started it. A leg runs on threads of its
      own, so a value is a way to connect, never an open connection.
    - ``may_see(caller, started_by)`` says whether a caller may read and act on a run; by default only the caller
      who started it may.
    - ``published`` is a FastAPI dependency taking the path's ``name`` and giving a ``GraphRecord``, raising
      ``HTTPException(404)`` for one it doesn't offer; with it, ``POST /graphs/{name}/runs`` starts that graph on
      inputs alone.
    - ``max_rows`` is every run's ceiling on the rows of one table; ``lease`` the seconds a worker holds a run between
      renewals; ``checkpoint_every`` the seconds between saves of a running leg; ``leg_timeout`` the seconds one leg
      may run; ``poll`` the seconds between a follower's reads of new events.
    """
    worker = uuid.uuid4().hex
    legs: dict[str, _Leg] = {}
    tasks: set[asyncio.Task[None]] = set()
    stopping = False
    caller_dependency: Callable[..., Any] = caller if caller is not None else _nobody
    caller_type = typing.get_type_hints(caller_dependency).get("return", Any)
    callers: TypeAdapter[Any] = TypeAdapter(caller_type)
    sees = may_see if may_see is not None else _same

    def nodes_for(who: Any) -> NodeRegistry:
        return registry if isinstance(registry, NodeRegistry) else registry(who)

    def started_by(row: StoredRun) -> Any:
        return callers.validate_json(row.caller)

    def load(row: StoredRun, who: Any = None) -> Run:
        """The stored run, on the nodes of ``who``, or of the caller who started it."""
        return nodes_for(who if who is not None else started_by(row)).load(row.graph).load_run(row.run)

    # -- legs -------------------------------------------------------------------------------------

    async def claim(row: StoredRun, run: str | None = None) -> bool:
        """Take ``row`` for this worker if nobody holds it and it is still at the ``rev`` read. Of two workers, one wins."""
        now = time.time()
        if row.owner is not None and row.lease_until >= now:
            return False
        changes: RunChanges = {"status": "running", "owner": worker, "lease_until": now + lease, "cancel": False}
        if run is not None:
            changes["run"] = run
        return await store.update(row.id, row.rev, changes)

    def launch(run_id: str, rev: int, ceiling: int | None = None) -> None:
        """Run the next leg of a run this worker has just claimed at ``rev``, in the background."""
        legs[run_id] = _Leg(rev=rev)
        task = asyncio.get_running_loop().create_task(drive(run_id, legs[run_id], ceiling))
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    async def write(run_id: str, leg: _Leg, changes: RunChanges) -> bool:
        """Write while this worker holds the run. Another writer (an answer queued, a cancel) moves ``rev`` on: read
        again and go on. A run another worker took, or that stopped running, stops the leg and writes nothing."""
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
                    leg.asked = True
                    leg.cancel.set()
                leg.rev = row.rev

    async def drive(run_id: str, leg: _Leg, ceiling: int | None) -> None:
        keep_lease = asyncio.create_task(hold(run_id, leg))
        try:
            row = await store.get(run_id)
            assert row is not None
            run = load(row)
            values = from_run(started_by(row)) if from_run is not None else None
            stream = run.stream(
                from_run=values, max_rows=ceiling if ceiling is not None else max_rows, timeout=leg_timeout,
                cancel=leg.cancel, checkpoint_every=checkpoint_every,
            )
            async for event in stream:
                if isinstance(event, RunCheckpoint):
                    await write(run_id, leg, {"run": await asyncio.to_thread(event.run.to_json)})
                    continue
                if await store.add_event(run_id, worker, event.model_dump_json()) is None:
                    leg.cancel.set()  # another worker holds the run
                if isinstance(event, RunEnd):
                    await finish(run_id, leg, event.run)
        except Exception:
            log.exception("the leg of run %s failed", run_id)
            await store.add_event(run_id, worker, '{"type":"run_end","status":"failed"}')
            await write(run_id, leg, {"status": "failed", "owner": None})
        finally:
            keep_lease.cancel()
            legs.pop(run_id, None)

    async def finish(run_id: str, leg: _Leg, run: Run) -> None:
        """Write how the leg ended. A leg this worker stopped to shut down is handed back for any worker to go on with."""
        changes: RunChanges = {"status": run.status, "owner": None, "lease_until": 0.0, "cancel": False}
        if stopping and run.status == "cancelled" and not leg.asked:
            changes["status"] = "running"
        changes["waiting"] = to_json([unit.model_dump(mode="json") for unit in run.waiting]).decode()
        if run.status == "complete":
            changes["outputs"] = to_json({a: v.to_dicts() if isinstance(v, pl.DataFrame) else v for a, v in run.outputs.items()}).decode()
        changes["run"] = await asyncio.to_thread(run.to_json)
        if await write(run_id, leg, changes):
            await apply_pending(run_id)

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
                    launch(row.id, row.rev + 1)
            await asyncio.sleep(lease / 3)

    async def apply_pending(run_id: str) -> None:
        """Answers that came while a leg ran, each on its own once the leg has ended: a wrong one, or one whose row
        doesn't wait, keeps its problems for the client in ``refused``, and the rest start the next leg."""
        row = await store.get(run_id)
        pending: dict[str, Any] = json.loads(row.pending) if row is not None else {}
        if row is None or row.status == "running" or not pending:
            return
        run = await asyncio.to_thread(load, row)
        taken, refused = False, json.loads(row.refused)
        for unit_id, value in pending.items():
            try:
                run, taken = run.answer(unit_id, value), True
            except InvalidCallError as error:
                refused[unit_id] = [problem.model_dump(mode="json") for problem in error.errors()]
        changes: RunChanges = {"pending": "{}", "refused": to_json(refused).decode()}
        if taken:
            changes.update({"status": "running", "owner": worker, "lease_until": time.time() + lease, "run": await asyncio.to_thread(run.to_json)})
        if await store.update(run_id, row.rev, changes) and taken:
            launch(run_id, row.rev + 1)
        # a failed update means another worker changed the run first: the answers stay queued for the end of its leg

    async def start(graph: GraphRecord, inputs: Mapping[str, Any], who: Any) -> Started:
        try:
            run = nodes_for(who).load(graph).with_inputs(inputs, max_rows=max_rows)
        except InvalidCallError as error:
            raise _unprocessable(error) from None
        run_id = uuid.uuid4().hex
        await store.create(StoredRun(
            id=run_id, graph=graph.model_dump_json(), caller=callers.dump_json(who).decode(), run=run.to_json(),
            status="running", owner=worker, lease_until=time.time() + lease,
        ))
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
            id=row.id, status=row.status, rev=row.rev, waiting=json.loads(row.waiting),
            outputs=json.loads(row.outputs) if row.outputs is not None else None, refused=json.loads(row.refused),
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
    async def answer(row: Stored, answers: dict[str, Any], if_match: Annotated[int | None, Header()] = None) -> Answered:
        """Answer waiting rows by their ids. While a leg runs, the answers wait for it to end, and a row answered once
        meanwhile is refused a second answer."""
        while row.status == "running":
            queued = json.loads(row.pending)
            if queued.keys() & answers.keys():
                raise HTTPException(409, "the row is already answered")
            queued.update(answers)
            if await store.update(row.id, row.rev, {"pending": to_json(queued).decode()}):
                await apply_pending(row.id)  # the leg may have ended meanwhile
                return Answered(answers="queued")
            again = await store.get(row.id)
            if again is None:
                raise HTTPException(404, "no such run")
            row = again
        if row.status != "waiting":
            raise HTTPException(409, f"the run is {row.status}; resume it first")
        try:  # every wrong answer is refused at once
            answered = await asyncio.to_thread(lambda: load(row).answer(answers).to_json())
        except InvalidCallError as error:
            raise _unprocessable(error) from None
        if (if_match is not None and if_match != row.rev) or not await claim(row, answered):
            raise HTTPException(409, "the run changed since it was read")
        launch(row.id, row.rev + 1)
        return Answered(answers="started")

    @router.post("/runs/{run_id}/resume", status_code=202)
    async def resume(row: Stored, body: Resume) -> Started:
        """Go on with a run that timed out, stopped at its ceiling, was cancelled or failed; nothing finished is called again."""
        if row.status not in _RESUMABLE:
            raise HTTPException(409, f"the run is {row.status}")
        if not await claim(row):
            raise HTTPException(409, "the run changed since it was read")
        launch(row.id, row.rev + 1, body.max_rows)
        return Started(id=row.id)

    @router.post("/runs/{run_id}/again", status_code=202)
    async def again(row: Stored, inputs: dict[str, Any], who: Caller) -> Started:
        """A new run of the caller's with these inputs changed, on the caller's own nodes; a row that receives what it
        received before isn't called again."""
        try:
            run = await asyncio.to_thread(lambda: load(row, who).with_inputs(inputs))
        except InvalidCallError as error:
            raise _unprocessable(error) from None
        run_id = uuid.uuid4().hex
        await store.create(StoredRun(
            id=run_id, graph=row.graph, caller=callers.dump_json(who).decode(), run=run.to_json(),
            status="running", owner=worker, lease_until=time.time() + lease,
        ))
        launch(run_id, 0)
        return Started(id=run_id)

    @router.post("/runs/{run_id}/cancel")
    async def cancel(row: Stored) -> Cancelled:
        """Stop a running leg before its next unit; the calls already made are kept, and ``resume`` goes on."""
        while row.status == "running":
            if await store.update(row.id, row.rev, {"cancel": True}):
                if row.id in legs:  # on this worker at once; on another at its next renewal
                    legs[row.id].asked = True
                    legs[row.id].cancel.set()
                return Cancelled(cancelled=True)
            again = await store.get(row.id)
            if again is None:
                break
            row = again
        return Cancelled(cancelled=False)

    return router


def _nobody() -> None:
    """The caller when the host names none."""
    return None


def _same(caller: Any, started_by: Any) -> bool:
    return bool(caller == started_by)


def _unprocessable(error: InvalidCallError) -> HTTPException:
    return HTTPException(422, [problem.model_dump(mode="json") for problem in error.errors()])


