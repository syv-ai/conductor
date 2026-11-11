"""``run_router``: the routes that start, follow, read, answer, resume and cancel runs, on any number of workers.

Each route parses its call, asks the process's ``Worker`` and says what it
answered; the worker runs the legs.
"""

import asyncio
import json
from collections.abc import AsyncIterable, Callable, Coroutine, Mapping
from typing import Annotated, Any, Literal, TypeVar, get_type_hints, overload

from conductor import (
    GraphDescription,
    GraphRecord,
    InvalidCallError,
    NodeRegistry,
    Problem,
    RegistryDescription,
)
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request, Response
from fastapi.routing import APIRoute
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, PositiveInt

from conductor_providers.fastapi.store import RunStore, StoredRun
from conductor_providers.fastapi.worker import NoSuchRunError, RefusedError, RunView, Worker

C = TypeVar("C")


class StartRun(BaseModel):
    """The body of ``POST /runs``: a graph to run and the values it starts on.

    Attributes:
        graph: The graph, as its stored record; loaded on the caller's registry, so a node it lacks is a 422.
        inputs: Values for the graph's open inputs, each by its name in ``graph.inputs`` or its address
            ``node.input``, checked as ``graph.with_inputs`` checks them.
    """

    graph: GraphRecord
    inputs: dict[str, Any] = {}


class Refused(BaseModel):
    """The body of a 404 or a 409: a call the router itself refused.

    Attributes:
        detail: Every problem with the call, each with one of the router's own codes (``ROUTER_PROBLEM_CODES``).
    """

    detail: list[Problem]


#: What a route declares beside its own answer: a run that isn't there for the caller, and a run in another state.
_NOT_FOUND: dict[int | str, dict[str, Any]] = {404: {"model": Refused, "description": "No such run for the caller: `no_such_run`."}}
_CONFLICT: dict[int | str, dict[str, Any]] = {409: {"model": Refused, "description": "The run is in a state that doesn't take the call."}}
#: The same 404 for a route that answers with a stream: its refusal is JSON, where a model would be declared as the
#: stream's type, so it names the component the other routes' model declares.
_NOT_FOUND_FOR_A_STREAM: dict[int | str, dict[str, Any]] = {
    404: {"description": _NOT_FOUND[404]["description"], "content": {"application/json": {"schema": {"$ref": "#/components/schemas/Refused"}}}},
}


class Started(BaseModel):
    """A run that was started; ``GET /runs/{id}`` reads it."""

    id: str


class Answered(BaseModel):
    """The body of a ``POST /runs/{id}/answers`` that was taken.

    Attributes:
        answers: ``started`` when the answers were checked and started the next leg; ``queued`` when a leg was
            running, so they are checked when it ends, and any refused then is in ``RunView.refused``.
    """

    answers: Literal["started", "queued"]


class Resume(BaseModel):
    """The body of ``POST /runs/{id}/resume``; a resume needs none.

    Attributes:
        max_items: A higher ceiling on the items of any one Series, for a run that stopped at its ceiling; it
            holds for the run's later legs. Left out, the run keeps the ceiling it has.
    """

    max_items: PositiveInt | None = None


class Cancelled(BaseModel):
    """Whether a running leg was asked to stop; a run that isn't running has nothing to cancel."""

    cancelled: bool


class _RefusingRoute(APIRoute):
    """A route whose refused call answers with the problems: a 422 for what the graph or the run refused, a 404 for a
    run that isn't there for the caller, a 409 for a run in a state that doesn't take the call."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        """The route's handler, with each refusal raised under it answered as its status and problems."""
        handle = super().get_route_handler()

        async def refusing(request: Request) -> Response:
            try:
                return await handle(request)
            except InvalidCallError as error:
                raise _refusal(422, error.errors()) from None
            except NoSuchRunError as error:
                raise _refusal(404, error.problems) from None
            except RefusedError as error:
                raise _refusal(409, error.problems) from None

        return refusing


def _refusal(status_code: int, problems: list[Problem]) -> HTTPException:
    """The refusal of a call: ``problems`` as the ``detail`` of a response with ``status_code``."""
    return HTTPException(status_code, [problem.model_dump(mode="json") for problem in problems])


def _nobody() -> None:
    """The caller when the host names none."""
    return None


def _same(caller: object, started_by: object) -> bool:
    """Whether a caller is the one who started a run: who may see it when the host says nothing else."""
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
    nodes_for: Callable[[C], NodeRegistry] = (lambda who: registry) if isinstance(registry, NodeRegistry) else registry
    worker: Worker[C] = Worker(
        store=store, nodes_for=nodes_for, caller_type=_caller_type(caller), from_run=from_run,
        max_items=max_items, lease=lease, checkpoint_every=checkpoint_every, leg_timeout=leg_timeout,
    )
    router = APIRouter(lifespan=lambda _: worker.running(), route_class=_RefusingRoute)

    async def stored(run_id: str, who: Annotated[C, Depends(caller)]) -> StoredRun:
        row = await worker.read(run_id)
        if not may_see(who, worker.started_by(row)):
            raise NoSuchRunError
        return row

    @router.get("/nodes")
    def list_nodes(who: Annotated[C, Depends(caller)]) -> RegistryDescription:
        """The nodes the caller may use, as an editor's palette reads them."""
        return nodes_for(who).describe()

    @router.post("/load")
    def load_graph(graph: GraphRecord, who: Annotated[C, Depends(caller)]) -> GraphDescription:
        """The graph as loading reads it: each node's fields, where it runs, and every problem."""
        return nodes_for(who).load(graph).describe()

    @router.post("/runs", status_code=202)
    async def start_run(body: StartRun, who: Annotated[C, Depends(caller)]) -> Started:
        """Start a run of any graph; this is an author's route, so guard it with your own dependency."""
        return Started(id=await worker.begin(body.graph, body.inputs, who))

    if published is not None:
        @router.post("/graphs/{name}/runs", status_code=202)
        async def start_published(
            graph: Annotated[GraphRecord, Depends(published)],
            inputs: Annotated[dict[str, Any], Body(description="Values for the graph's open inputs, each by its name or its address `node.input`.")],
            who: Annotated[C, Depends(caller)],
        ) -> Started:
            """Start a graph you offer by name, on inputs alone."""
            return Started(id=await worker.begin(graph, inputs, who))

    @router.get("/runs/{run_id}", responses=_NOT_FOUND)
    def read_run(row: Annotated[StoredRun, Depends(stored)]) -> RunView:
        """The run as its last leg left it: its status, what waits for a person, its outputs once complete, the
        queued answers that were refused, and why it ended early if it did."""
        return RunView.of(row)

    @router.get("/runs/{run_id}/events", response_class=EventSourceResponse, responses=_NOT_FOUND_FOR_A_STREAM)
    async def follow_run(row: Annotated[StoredRun, Depends(stored)], last_event_id: Annotated[int, Header()] = -1) -> AsyncIterable[ServerSentEvent]:
        """The run's events, from whichever worker runs it, until it stops running; ``Last-Event-ID`` goes on after a reconnect.

        A run stops running only after its leg's ``run_end`` is stored, so the stream reads the run itself only once
        it has sent a ``run_end`` or has had nothing to send, and otherwise reads events alone.
        """
        seen = last_event_id
        may_have_ended = True
        while True:
            new = await store.events(row.id, seen)
            for seq, body in new:
                seen = seq
                yield ServerSentEvent(raw_data=body, id=str(seq))
            if new:
                may_have_ended = json.loads(new[-1][1])["type"] == "run_end"
                continue
            if may_have_ended:  # the row follows its run_end, so it may still read running for a poll or two
                read_again = await store.get(row.id)
                if read_again is None or read_again.status != "running":
                    for seq, body in await store.events(row.id, seen):  # written before the status changed
                        yield ServerSentEvent(raw_data=body, id=str(seq))
                    return
            await asyncio.sleep(poll)

    @router.post("/runs/{run_id}/answers", status_code=202, responses={**_NOT_FOUND, **_CONFLICT})
    async def answer_run(
        row: Annotated[StoredRun, Depends(stored)],
        answers: Annotated[dict[str, Any], Body(description="An answer for each waiting unit, by its `id` in the run's `waiting`.")],
    ) -> Answered:
        """Answer what waits by its ids, and start the next leg on the answers.

        The run takes them as ``run.answer`` does, whatever its status: a run that failed, stopped or timed out with
        questions waiting is answered and goes on, and every answer the run refuses is in the 422. A complete run
        has nothing to answer and is a 409. While a leg runs, the answers wait for it to end, and a question
        answered once meanwhile is refused a second answer.
        """
        return Answered(answers=await worker.answer(row, answers))

    @router.post("/runs/{run_id}/resume", status_code=202, responses={**_NOT_FOUND, **_CONFLICT})
    async def resume_run(row: Annotated[StoredRun, Depends(stored)], body: Resume = Resume()) -> Started:
        """Go on with a run whose last leg ended early; nothing finished is called again. ``{"max_items": ...}`` raises
        the ceiling of a run that stopped at it; a resume needs no body."""
        await worker.resume(row, body.max_items)
        return Started(id=row.id)

    @router.post("/runs/{run_id}/again", status_code=202, responses=_NOT_FOUND)
    async def run_again(
        row: Annotated[StoredRun, Depends(stored)],
        inputs: Annotated[dict[str, Any], Body(description="The inputs to change, each by its name or its address `node.input`; the rest stay as the run had them.")],
        who: Annotated[C, Depends(caller)],
    ) -> Started:
        """A new run of the caller's with these inputs changed, on the caller's own nodes; a call that receives what it
        received before isn't made again."""
        return Started(id=await worker.again(row, inputs, who))

    @router.post("/runs/{run_id}/cancel", responses=_NOT_FOUND)
    async def cancel_run(row: Annotated[StoredRun, Depends(stored)]) -> Cancelled:
        """Stop a running leg before its next unit; the calls already made are kept, and ``resume`` goes on."""
        return Cancelled(cancelled=await worker.cancel(row))

    return router


def _caller_type(caller: Callable[..., Any]) -> Any:
    """The type a ``caller`` dependency gives: the class itself, or a function's return annotation, which it must have."""
    if isinstance(caller, type):
        return caller
    returns = get_type_hints(caller).get("return")
    if returns is None:
        raise TypeError(f"caller {caller!r} has no return annotation; annotate it with the type it gives, so a run keeps its caller as that type")
    return returns
