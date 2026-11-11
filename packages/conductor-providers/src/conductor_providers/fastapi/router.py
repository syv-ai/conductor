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
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Annotated, Any, TypeVar, Unpack, overload

from conductor import (
    EndedEarly,
    Going,
    GraphDescription,
    GraphRecord,
    InvalidCallError,
    NodeRegistry,
    Problem,
    RegistryDescription,
    Run,
    RunState,
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

#: The English of each code the router itself writes, beside the codes of the runs it serves.
_ROUTER_PROBLEMS: dict[str, str] = {
    "no_such_run": "There is no run by that id that the caller may see; check the id and who is calling.",
    "already_answered": "An answer to {waiting_id!r} already waits for the running leg to end; read the run once it has, and answer what still waits.",
    "run_complete": "The run is complete, so nothing in it waits for an answer; start a new run to ask again.",
    "run_not_resumable": "The run is {status}, and only a run that failed, stopped, was cancelled or timed out is resumed; read the run to see what it needs.",
    "run_changed": "The run changed since it was read, so nothing was done; read it again and repeat the call if it still applies.",
    "leg_failed": "The run's worker failed before the leg could end; its operator finds why in the server log. Resume it to carry on.",
}

#: The codes the router itself writes, for a host translating by code to cover beside conductor's ``PROBLEM_CODES``.
ROUTER_PROBLEM_CODES: frozenset[str] = frozenset(_ROUTER_PROBLEMS)


class StartRun(BaseModel):
    """The body of ``POST /runs``: a graph to run and the values it starts on.

    Attributes:
        graph: The graph, as its stored record; loaded on the caller's registry, so a node it lacks is a 422.
        inputs: Values for the graph's open inputs, each by its name in ``graph.inputs`` or its address
            ``node.input``, checked as ``graph.with_inputs`` checks them.
    """

    graph: GraphRecord
    inputs: dict[str, Any] = {}


class RunView(BaseModel):
    """The body of ``GET /runs/{id}``: a run as its last leg left it, read without loading the run.

    Attributes:
        id: The run's id, as ``POST /runs`` gave it.
        status: Where the run is: ``running`` while a leg runs, ``waiting`` for answers, ``complete``, or
            ``timed_out``, ``stopped``, ``cancelled`` or ``failed``, which ``POST /runs/{id}/resume`` goes on from.
        waiting: What waits for a person, as of the last leg's end; each ``id`` is what
            ``POST /runs/{id}/answers`` takes.
        outputs: The graph's outputs by address once the run is complete, each as its type writes it (a table as its
            columns); ``None`` until then.
        refused: Answers sent while a leg ran that were refused when it ended, by waiting id, each with its problems.
        cause: Why the run failed, stopped, was cancelled or timed out; ``None`` while it runs, waits or once complete.
    """

    id: str
    status: RunStatus
    waiting: list[Waiting] = []
    outputs: dict[str, Any] | None = None
    refused: dict[str, list[Problem]] = {}
    cause: Problem | None = None


class Refused(BaseModel):
    """The body of a call the router refused, as a 404, 409 or 422.

    Attributes:
        detail: Every problem with the call. A 404 and a 409 hold the router's own codes
            (``ROUTER_PROBLEM_CODES``); a 422 holds the codes of the graph or the run that refused it.
    """

    detail: list[Problem]


#: What a route declares beside its own answer: a run that isn't there for the caller, and a run in another state.
_NOT_FOUND: dict[int | str, dict[str, Any]] = {404: {"model": Refused, "description": "No such run for the caller: `no_such_run`."}}
_CONFLICT: dict[int | str, dict[str, Any]] = {**_NOT_FOUND, 409: {"model": Refused, "description": "The run is in a state that doesn't take the call."}}


class Started(BaseModel):
    """A run that was started; ``GET /runs/{id}`` reads it."""

    id: str


class Answered(BaseModel):
    """The body of a ``POST /runs/{id}/answers`` that was taken.

    Attributes:
        answers: ``started`` when the answers were checked and started the next leg; ``queued`` when a leg was
            running, so they are checked when it ends, and any refused then is in ``RunView.refused``.
    """

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


def _nobody() -> None:
    """The caller when the host names none."""
    return None


def _same(caller: object, started_by: object) -> bool:
    return caller == started_by


def _no_values(caller: object) -> Mapping[type, Any]:
    """The ``FromRun`` values when the host gives none."""
    return {}


@overload
def run_router(
    *,
    store: RunStore,
    registry: NodeRegistry | Callable[[C], NodeRegistry],
    caller: Callable[..., C],
    from_run: Callable[[C], Mapping[type, Any]] = ...,
    may_see: Callable[[C, C], bool] = ...,
    published: Callable[..., GraphRecord] | None = None,
    max_items: int | None = None,
    lease: float = 30.0,
    checkpoint_every: float = 5.0,
    leg_timeout: float | None = 300.0,
    poll: float = 0.1,
) -> APIRouter: ...


@overload
def run_router(
    *,
    store: RunStore,
    registry: NodeRegistry | Callable[[None], NodeRegistry],
    from_run: Callable[[None], Mapping[type, Any]] = ...,
    may_see: Callable[[None, None], bool] = ...,
    published: Callable[..., GraphRecord] | None = None,
    max_items: int | None = None,
    lease: float = 30.0,
    checkpoint_every: float = 5.0,
    leg_timeout: float | None = 300.0,
    poll: float = 0.1,
) -> APIRouter: ...


def run_router(
    *,
    store: RunStore,
    registry: NodeRegistry | Callable[[C], NodeRegistry],
    caller: Callable[..., Any] = _nobody,
    from_run: Callable[[C], Mapping[type, Any]] = _no_values,
    may_see: Callable[[C, C], bool] = _same,
    published: Callable[..., GraphRecord] | None = None,
    max_items: int | None = None,
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
    - ``max_items`` is the ceiling on the items of any one Series a node works through, which a run starts with; a resume may raise it for that run,
      and the raised ceiling holds for its later legs. ``lease`` the seconds a worker holds a run between
      renewals; ``checkpoint_every`` the seconds between saves of a running leg; ``leg_timeout`` the seconds one leg
      may run; ``poll`` the seconds between a follower's reads of new events.
    """
    worker = uuid.uuid4().hex
    legs: dict[str, _Leg] = {}
    tasks: set[asyncio.Task[None]] = set()
    stopping = False
    callers: TypeAdapter[C] = TypeAdapter(_caller_type(caller))
    nodes_for: Callable[[C], NodeRegistry] = _always(registry) if isinstance(registry, NodeRegistry) else registry

    def started_by(row: StoredRun) -> C:
        return callers.validate_json(row.caller)

    def load(row: StoredRun, nodes: NodeRegistry) -> Run:
        """The stored run, on ``nodes``."""
        return nodes.load(row.graph).load_run(row.run)

    # -- legs -------------------------------------------------------------------------------------

    async def claim(row: StoredRun, ceiling: int | None = None, **changes: Unpack[RunChanges]) -> bool:
        """Take ``row`` for this worker if it is still at the ``rev`` read, writing ``changes`` too, and run its next
        leg. Of two workers, one wins. A cancel asked of the run stays asked: only a resume clears it."""
        written: RunChanges = {**held(), **changes}
        if not await store.update(row.id, row.rev, written):
            return False
        launch(replace(row, **written, rev=row.rev + 1), ceiling)
        return True

    def held() -> RunChanges:
        """What a row says while this worker holds it: running, with no cause yet."""
        return {**_standing(Going("running")), "owner": worker, "lease_until": time.time() + lease}

    def launch(row: StoredRun, ceiling: int | None = None) -> None:
        """Run the next leg of ``row``, as this worker has just written it, in the background; a cancel asked of it the
        leg hears before its first unit."""
        leg = legs[row.id] = _Leg(rev=row.rev)
        if row.cancel:
            leg.cancel.set()
        task = asyncio.get_running_loop().create_task(drive(row, leg, ceiling))
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

    async def drive(row: StoredRun, leg: _Leg, ceiling: int | None) -> None:
        """Run the leg and store its events, then have ``finish`` write its end while its lease is still renewed; a
        leg that raised ends ``failed``."""
        run_id = row.id
        keep_lease = asyncio.create_task(hold(run_id, leg))
        try:
            who = started_by(row)
            run = await asyncio.to_thread(load, row, nodes_for(who))
            values = from_run(who)
            stream = run.stream(
                from_run=values, max_items=ceiling, timeout=leg_timeout,
                cancel=leg.cancel, checkpoint_every=checkpoint_every,
            )
            async with contextlib.aclosing(stream):  # a raise in this loop ends the leg before its next unit
                async for event in stream:
                    if isinstance(event, RunCheckpoint):
                        await write(run_id, leg, {"run": await asyncio.to_thread(event.run.to_json)})
                        continue
                    if isinstance(event, RunEnd):
                        await finish(run_id, leg, keep_lease, await asyncio.to_thread(_Ended, event))
                    elif await store.add_event(run_id, worker, event.model_dump_json()) is None:
                        leg.cancel.set()  # another worker holds the run
        except Exception:
            log.exception("the leg of run %s failed", run_id)
            await finish(run_id, leg, keep_lease, _Raised())
        finally:
            keep_lease.cancel()
            if legs.get(run_id) is leg:  # the answers queued during this leg may have launched the next one already
                del legs[run_id]

    async def finish(run_id: str, leg: _Leg, keep_lease: asyncio.Task[None], ending: "_Ended | _Raised") -> None:
        """Write how a leg ended, decided from the row as read for each try of
        the write, so a cancel or an answer that lands meanwhile is never written over.

        A leg this worker stopped to shut down, with no cancel asked of the run, is handed back for any worker to go on
        with: the run did not end, so no ``run_end`` is stored for it. Otherwise the leg's ``run_end`` is stored, once
        and before the row says the run stopped running, so a follower that reads the new status has the event to
        read; then the answers queued during the leg are given to the run it ended with, and the next leg starts
        on those it took; a leg that raised ended with no run, so its queued answers stay queued for the next leg's
        end. A run another worker took is left as it is.

        ``keep_lease`` renews the lease until the end is written, and each JSON of a run is made once, outside the
        write's tries: a run however long it takes to write as JSON stays this worker's, and a try lost to a renewal
        or a queued answer costs a read and a write."""
        status = ending.status
        written = ending.written
        handed_back: RunChanges = {**_standing(Going("running")), "owner": None, "lease_until": 0.0}  # it did not end early
        answered_from: tuple[str, str] | None = None
        answered: tuple[str | None, dict[str, Any]] = (None, {})
        end_stored = False
        try:
            while True:
                row = await store.get(run_id)
                if row is None or row.owner != worker or row.status != "running":
                    return
                saved = None
                if stopping and status == "cancelled" and not row.cancel:
                    changes: RunChanges = {**written, **handed_back}
                else:
                    if not end_stored:  # a later try can't turn into a hand-back: only a resume clears a cancel
                        await store.add_event(run_id, worker, ending.event)
                        end_stored = True
                    changes = {**written, "status": status, "owner": None, "lease_until": 0.0, "cancel": False}
                    if row.pending != "{}" and isinstance(ending, _Ended):
                        if answered_from != (row.pending, row.refused):
                            answered_from = (row.pending, row.refused)
                            answered = await asyncio.to_thread(ending.answered, json.loads(row.pending), json.loads(row.refused))
                        saved, refused = answered
                        changes.update({"pending": "{}", "refused": to_json(refused).decode()})
                    if saved is not None:  # a worker shutting down hands the next leg back rather than start it
                        changes.update({"run": saved, **(handed_back if stopping else held())})
                if saved is None:
                    changes.update(await asyncio.to_thread(lambda: ending.as_saved))
                async with leg.lock:  # no renewal lands between this write and the lease's last renewal
                    ended = await store.update(run_id, row.rev, changes)
                    if ended:
                        keep_lease.cancel()
                if ended:
                    await _stopped(keep_lease)
                    if saved is not None and not stopping:
                        launch(replace(row, **changes, rev=row.rev + 1))
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
                if row.id not in legs:
                    await claim(row)
            await asyncio.sleep(lease / 3)

    async def start(graph: GraphRecord, inputs: Mapping[str, Any], who: C) -> Started:
        try:
            saved = await asyncio.to_thread(lambda: nodes_for(who).load(graph).with_inputs(inputs, max_items=max_items).to_json())
        except InvalidCallError as error:
            raise _refused(422, *error.errors()) from None
        return await begin(graph.model_dump_json(), who, saved)

    async def begin(graph: str, who: C, saved: str) -> Started:
        """Store a run of ``who``'s, held by this worker, and run its first leg."""
        row = replace(StoredRun(id=uuid.uuid4().hex, graph=graph, caller=callers.dump_json(who).decode(), run=saved, status="running"), **held())
        await store.create(row)
        launch(row)
        return Started(id=row.id)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None]:
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
    async def stored(run_id: str, who: Annotated[C, Depends(caller)]) -> StoredRun:
        row = await store.get(run_id)
        if row is None or not may_see(who, started_by(row)):
            raise _refused(404, _router_problem("no_such_run"))
        return row

    @router.get("/nodes")
    def palette(who: Annotated[C, Depends(caller)]) -> RegistryDescription:
        """The nodes the caller may use, as an editor's palette reads them."""
        return nodes_for(who).describe()

    @router.post("/load")
    def check(graph: GraphRecord, who: Annotated[C, Depends(caller)]) -> GraphDescription:
        """The graph as loading reads it: each node's fields, where it runs, and every problem."""
        return nodes_for(who).load(graph).describe()

    @router.post("/runs", status_code=202)
    async def start_run(body: StartRun, who: Annotated[C, Depends(caller)]) -> Started:
        """Start a run of any graph; this is an author's route, so guard it with your own dependency."""
        return await start(body.graph, body.inputs, who)

    if published is not None:
        @router.post("/graphs/{name}/runs", status_code=202)
        async def start_published(graph: Annotated[GraphRecord, Depends(published)], inputs: dict[str, Any], who: Annotated[C, Depends(caller)]) -> Started:
            """Start a graph you offer by name, on inputs alone."""
            return await start(graph, inputs, who)

    @router.get("/runs/{run_id}", responses=_NOT_FOUND)
    def read(row: Annotated[StoredRun, Depends(stored)]) -> RunView:
        return RunView(
            id=row.id, status=row.status, waiting=json.loads(row.waiting),
            outputs=json.loads(row.outputs) if row.outputs is not None else None, refused=json.loads(row.refused),
            cause=json.loads(row.cause) if row.cause is not None else None,
        )

    @router.get("/runs/{run_id}/events", response_class=EventSourceResponse, responses=_NOT_FOUND)
    async def follow(row: Annotated[StoredRun, Depends(stored)], last_event_id: Annotated[int, Header()] = -1) -> AsyncIterable[ServerSentEvent]:
        """The run's events, from whichever worker runs it, until it stops running; ``Last-Event-ID`` goes on after a reconnect."""
        seen = last_event_id
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

    @router.post("/runs/{run_id}/answers", status_code=202, responses=_CONFLICT)
    async def answer(row: Annotated[StoredRun, Depends(stored)], answers: dict[str, Any]) -> Answered:
        """Answer what waits by its ids, and start the next leg on the answers.

        The run takes them as ``run.answer`` does, whatever its status: a run that failed, stopped or timed out with
        questions waiting is answered and goes on, and every answer the run refuses is in the 422. A complete run
        has nothing to answer and is a 409. While a leg runs, the answers wait for it to end, and a question
        answered once meanwhile is refused a second answer.
        """
        while row.status == "running":
            queued = json.loads(row.pending)
            if twice := queued.keys() & answers.keys():
                raise _refused(409, *(_router_problem("already_answered", waiting_id=waiting_id) for waiting_id in sorted(twice)))
            queued.update(answers)
            if await store.update(row.id, row.rev, {"pending": to_json(queued).decode()}):
                return Answered(answers="queued")  # the leg's end, written after this, reads them
            again = await store.get(row.id)
            if again is None:
                raise _refused(404, _router_problem("no_such_run"))
            row = again
        if row.status == "complete":
            raise _refused(409, _router_problem("run_complete"))
        try:  # every answer the run refuses is refused at once
            answered = await asyncio.to_thread(lambda: load(row, nodes_for(started_by(row))).answer(answers).to_json())
        except InvalidCallError as error:
            raise _refused(422, *error.errors()) from None
        if not await claim(row, run=answered):
            raise _refused(409, _router_problem("run_changed"))
        return Answered(answers="started")

    @router.post("/runs/{run_id}/resume", status_code=202, responses=_CONFLICT)
    async def resume(row: Annotated[StoredRun, Depends(stored)], max_items: Annotated[PositiveInt | None, Body(embed=True)] = None) -> Started:
        """Go on with a run that timed out, stopped at its ceiling, was cancelled or failed; nothing finished is called
        again. ``{"max_items": ...}`` raises the ceiling of a run that stopped at it; a resume needs no body."""
        if row.status not in _RESUMABLE:
            raise _refused(409, _router_problem("run_not_resumable", status=row.status))
        if not await claim(row, max_items, cancel=False):
            raise _refused(409, _router_problem("run_changed"))
        return Started(id=row.id)

    @router.post("/runs/{run_id}/again", status_code=202, responses=_NOT_FOUND)
    async def again(row: Annotated[StoredRun, Depends(stored)], inputs: dict[str, Any], who: Annotated[C, Depends(caller)]) -> Started:
        """A new run of the caller's with these inputs changed, on the caller's own nodes; a call that receives what it
        received before isn't made again."""
        try:
            saved = await asyncio.to_thread(lambda: load(row, nodes_for(who)).with_inputs(inputs).to_json())
        except InvalidCallError as error:
            raise _refused(422, *error.errors()) from None
        return await begin(row.graph, who, saved)

    @router.post("/runs/{run_id}/cancel", responses=_NOT_FOUND)
    async def cancel(row: Annotated[StoredRun, Depends(stored)]) -> Cancelled:
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


def _always(registry: NodeRegistry) -> Callable[[object], NodeRegistry]:
    """``registry``, whoever calls."""
    return lambda caller: registry


class _Ended:
    """A leg that ended with a run: its status, its ``run_end`` event as JSON, and what its end writes of the run
    beside the run itself (what waits for a person, its outputs once complete, each written by its type as the run
    writes it, and its status and cause)."""

    def __init__(self, end: RunEnd) -> None:
        run = self.run = end.run
        self.status: RunStatus = run.status
        self.event = end.model_dump_json()
        self.written: RunChanges = {
            "waiting": to_json(run.waiting).decode(),
            **_standing(run.state),
        }
        if run.status == "complete":
            self.written["outputs"] = run.outputs_json()

    @cached_property
    def as_saved(self) -> RunChanges:
        """The run as JSON, made once however many tries its write takes."""
        return {"run": self.run.to_json()}

    def answered(self, pending: dict[str, Any], refused: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
        """The answers queued during the leg, each on its own: the run they give as JSON when any was taken, else
        ``None``, and ``refused`` with the problems of each one that wasn't. The run decides, as ``run.answer`` does
        whatever its status: a wrong answer, one for something that isn't waiting (``not_waiting``, as on a complete
        run) or one to a cancelled run (``run_cancelled``) keeps its own problems."""
        run, taken = self.run, False
        for unit_id, value in pending.items():
            try:
                run, taken = run.answer(unit_id, value), True
            except InvalidCallError as error:
                refused[unit_id] = [problem.model_dump(mode="json") for problem in error.errors()]
        return (run.to_json() if taken else None), refused


class _Raised:
    """A leg that raised: it ends ``failed`` with ``leg_failed`` as its cause and keeps the run as last saved; answers
    queued during it stay queued."""

    status: RunStatus = "failed"

    def __init__(self) -> None:
        cause = _router_problem("leg_failed")
        self.written: RunChanges = _standing(EndedEarly("failed", cause))
        self.as_saved: RunChanges = {}
        # the event holds no run, which it never writes: a leg that raised left none
        self.event = RunEnd.model_construct(status="failed", cause=cause).model_dump_json()


def _standing(state: RunState) -> RunChanges:
    """How a run stands, as the row's ``status`` and ``cause`` columns write it: a cause only for one that ended early."""
    match state:
        case EndedEarly(status=status, cause=cause):
            return {"status": status, "cause": cause.model_dump_json()}
        case Going(status=status):
            return {"status": status, "cause": None}


def _router_problem(code: str, **ctx: Any) -> Problem:
    """The problem ``code`` of the router's own, about the run as a whole, its message made from ``ctx``."""
    return Problem(type=code, msg=_ROUTER_PROBLEMS[code].format(**ctx), ctx=ctx)


async def _stopped(task: asyncio.Task[None]) -> None:
    """Cancel ``task`` and wait until it has stopped."""
    task.cancel()
    await asyncio.wait([task])


def _caller_type(caller: Callable[..., Any]) -> Any:
    """The type a ``caller`` dependency gives: the class itself, or a function's return annotation, which it must have."""
    if isinstance(caller, type):
        return caller
    returns = typing.get_type_hints(caller).get("return")
    if returns is None:
        raise TypeError(f"caller {caller!r} has no return annotation; annotate it with the type it gives, so a run keeps its caller as that type")
    return returns


def _refused(status_code: int, *problems: Problem) -> HTTPException:
    """The refusal of a call: ``problems`` as the ``detail`` of a response with ``status_code``."""
    return HTTPException(status_code, [problem.model_dump(mode="json") for problem in problems])


