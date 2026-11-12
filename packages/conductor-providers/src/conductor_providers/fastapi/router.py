"""``run_router``: the routes that start, follow, read, answer, resume and cancel runs, on any number of workers.

Each route parses its call, asks the process's ``Worker`` and says what it
answered; the worker resumes the runs.
"""

from collections.abc import AsyncIterable, Callable, Coroutine, Mapping
from typing import Annotated, Any, Literal, TypeVar, get_type_hints, overload

from conductor import (
    Graph,
    GraphDescription,
    GraphRecord,
    InvalidCallError,
    NodeRegistry,
    Problem,
    RegistryDescription,
    RunStatus,
)
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request, Response
from fastapi.routing import APIRoute
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, PositiveInt, TypeAdapter

from conductor_providers.fastapi.router_problems import NoSuchRunError, RefusedError
from conductor_providers.fastapi.store import RunStore
from conductor_providers.fastapi.stored_run import Outcome, RunAsRead
from conductor_providers.fastapi.worker import Worker

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
        detail: Every problem with the call, each with one of the router's own codes (``ROUTER_PROBLEMS``).
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
        answers: ``started`` when the answers were checked and the run was resumed with them; ``queued`` when the
            run was running, so they are checked when it stops running, and any refused then is in
            ``RunView.refused``.
    """

    answers: Literal["started", "queued"]


class Resume(BaseModel):
    """The body of ``POST /runs/{id}/resume``; a resume needs none.

    Attributes:
        max_items: A higher ceiling on the items of any one Series, for a run that stopped at its ceiling; it
            holds for the run from then on. Left out, the run keeps the ceiling it has.
    """

    max_items: PositiveInt | None = None


class Cancelled(BaseModel):
    """Whether a running run was asked to stop; a run that isn't running has nothing to cancel."""

    cancelled: bool


class RunView(Outcome):
    """The body of ``GET /runs/{id}``: a run as it was when it last stopped running, read without loading the run.

    Attributes:
        waiting: What waits for a person, as of when the run last stopped running; each ``id`` is what
            ``POST /runs/{id}/answers`` takes. When the worker raised while resuming the run, what waits in the
            run's last checkpoint.
        outputs: The graph's outputs by address once the run is complete, each as its type writes it (a table as its
            columns); ``None`` until then.
        id: The run's id, as ``POST /runs`` gave it.
        status: Where the run is: ``running`` while a worker resumes it, ``waiting`` for answers, ``complete``, or
            how it ended early (``failed``, ``stopped``, ``cancelled``, ``timed_out``), which
            ``POST /runs/{id}/resume`` goes on from.
        refused: Answers sent while the run was running that were refused when it stopped, by waiting id, each
            with its problems.
        cause: Why the run ended early; ``None`` while it runs, waits or once complete.
    """

    id: str
    status: RunStatus
    refused: dict[str, list[Problem]]
    cause: Problem | None

    @classmethod
    def of(cls, run: RunAsRead) -> "RunView":
        """The view of a run, from its state and its row's ``outcome``.

        Args:
            run: The run as read.

        Returns:
            What ``GET /runs/{id}`` answers for it.
        """
        outcome = Outcome.model_validate_json(run.row.outcome)
        return cls(id=run.row.id, status=run.state.run_state.status, refused=run.state.refused, cause=run.state.cause, waiting=outcome.waiting, outputs=outcome.outputs)


class _RefusingRoute(APIRoute):
    """A route whose refused call answers with the problems: a 422 for what the graph or the run refused, a 404 for a
    run that isn't there for the caller, a 409 for a run in a state that doesn't take the call."""

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        """The route's handler, with each refusal raised under it answered as its status and problems.

        Returns:
            The handler FastAPI calls for a request.
        """
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
    """The refusal of a call.

    Args:
        status_code: The status of the response.
        problems: Why the call is refused; the response's ``detail``.

    Returns:
        The exception FastAPI answers with.
    """
    return HTTPException(status_code, [problem.model_dump(mode="json") for problem in problems])


def _nobody() -> None:
    """The caller when the host names none."""
    return None


def _everyone(caller: None, started_by: None) -> bool:
    """Who may see a run when the host names no caller: every request, since each is the same caller, nobody."""
    return True


def _no_values(caller: object) -> Mapping[type, Any]:
    """The ``Context`` values when the host gives none."""
    return {}


@overload
def run_router(
    *,
    store: RunStore,
    registry: NodeRegistry | Callable[[C], NodeRegistry],
    caller: Callable[..., C],
    may_see: Callable[[C, C], bool],
    context: Callable[[C], Mapping[type, Any]] = ...,
    published: Callable[..., GraphRecord] | None = None,
    max_items: int | None = None,
    lease: float = 30.0,
    checkpoint_every: float = 5.0,
    resume_timeout: float | None = 300.0,
    poll: float = 0.1,
) -> APIRouter: ...


@overload
def run_router(
    *,
    store: RunStore,
    registry: NodeRegistry | Callable[[None], NodeRegistry],
    context: Callable[[None], Mapping[type, Any]] = ...,
    published: Callable[..., GraphRecord] | None = None,
    max_items: int | None = None,
    lease: float = 30.0,
    checkpoint_every: float = 5.0,
    resume_timeout: float | None = 300.0,
    poll: float = 0.1,
) -> APIRouter: ...


def run_router(
    *,
    store: RunStore,
    registry: NodeRegistry | Callable[[C], NodeRegistry],
    caller: Callable[..., Any] = _nobody,
    may_see: Callable[[Any, Any], bool] = _everyone,
    context: Callable[[C], Mapping[type, Any]] = _no_values,
    published: Callable[..., GraphRecord] | None = None,
    max_items: int | None = None,
    lease: float = 30.0,
    checkpoint_every: float = 5.0,
    resume_timeout: float | None = 300.0,
    poll: float = 0.1,
) -> APIRouter:
    """The runs routes, with the worker that resumes their runs; mount them with ``app.include_router(...)``.

    Args:
        store: Where runs and events are kept.
        registry: The nodes graphs may use, or a function of the caller giving them, for a registry per tenant. It
            is called again for every request and every time a run is resumed, so a function caches what it builds.
        caller: A FastAPI dependency giving who is calling: your login. The type it gives (a class used as the
            dependency, or a function's return annotation, which it must have) is kept on the run as JSON, so a
            run is built again on any worker. Let it hold what identifies the caller and nothing that changes: a
            run keeps its caller as they were when it started.
        may_see: ``may_see(caller, started_by)`` says whether a caller may read and act on a run. Given whenever
            ``caller`` is: ``lambda caller, started_by: caller.id == started_by.id`` for the starter alone.
        context: Gives a run its ``Context`` values, from the caller who started it, each time it is resumed. A
            run's nodes are called on threads of their own, so a value is a way to connect, never an open
            connection.
        published: A FastAPI dependency taking the path's ``name`` and giving a ``GraphRecord``, raising
            ``HTTPException(404)`` for one it doesn't offer; with it, ``POST /graphs/{name}/runs`` starts that
            graph on inputs alone.
        max_items: The ceiling on the items of any one Series a node works through, which a run starts with; a
            resume may raise it for that run, and the raised ceiling holds for that run from then on.
        lease: The seconds a worker holds a run between renewals.
        checkpoint_every: The seconds between saves of a running run.
        resume_timeout: The seconds one call of ``resume()`` may take before the run ends ``timed_out``; ``None`` for no limit.
        poll: The seconds between a follower's reads of new events.

    Returns:
        The router, whose lifespan runs the worker.

    Raises:
        TypeError: ``caller`` is given without ``may_see``, or is a function with no return annotation.
    """
    if caller is not _nobody and may_see is _everyone:
        raise TypeError("run_router(caller=...) needs may_see=: say who may see a run, as lambda caller, started_by: caller.id == started_by.id")
    nodes_for: Callable[[C], NodeRegistry] = (lambda who: registry) if isinstance(registry, NodeRegistry) else registry
    worker: Worker[C] = Worker(
        store=store, nodes_for=nodes_for, callers=TypeAdapter(_caller_type(caller)), context=context,
        max_items=max_items, lease=lease, checkpoint_every=checkpoint_every, resume_timeout=resume_timeout, poll=poll,
    )
    router = APIRouter(lifespan=lambda _: worker.serving(), route_class=_RefusingRoute)

    async def stored(run_id: str, who: Annotated[C, Depends(caller)]) -> RunAsRead:
        """The run of a route under ``/runs/{run_id}``, for the caller: the access rule of every such route.

        Args:
            run_id: The run's id, from the path.
            who: The caller.

        Returns:
            The run as the store has it.

        Raises:
            NoSuchRunError: No run has the id, or ``may_see`` says the caller may not see it. The two read the
                same (a 404), so a caller learns nothing about a run that isn't theirs.
        """
        run = await worker.read(run_id)
        if not may_see(who, worker.started_by(run.row)):
            raise NoSuchRunError
        return run

    @router.get("/nodes")
    def list_nodes(who: Annotated[C, Depends(caller)]) -> RegistryDescription:
        """The nodes the caller may use, as an editor's palette reads them."""
        return nodes_for(who).describe()

    @router.post("/compile")
    def compile_graph(graph: GraphRecord, who: Annotated[C, Depends(caller)]) -> GraphDescription:
        """Compile the graph record and describe the result: each node's inputs and outputs, how often it is called,
        and every problem. Nothing is run or stored."""
        return Graph.compile(graph, nodes_for(who)).describe()

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
    def read_run(run: Annotated[RunAsRead, Depends(stored)]) -> RunView:
        """The run as it was when it last stopped running: its status, what waits for a person, its outputs once complete, the
        queued answers that were refused, and why it ended early if it did."""
        return RunView.of(run)

    @router.get("/runs/{run_id}/events", response_class=EventSourceResponse, responses=_NOT_FOUND_FOR_A_STREAM)
    async def follow_run(run: Annotated[RunAsRead, Depends(stored)], last_event_id: Annotated[int, Header()] = -1) -> AsyncIterable[ServerSentEvent]:
        """The run's events, from whichever worker runs it, until it stops running; ``Last-Event-ID`` goes on after a reconnect."""
        async for number, body in worker.events(run, last_event_id):
            yield ServerSentEvent(raw_data=body, id=str(number))

    @router.post("/runs/{run_id}/answers", status_code=202, responses={**_NOT_FOUND, **_CONFLICT})
    async def answer_run(
        run: Annotated[RunAsRead, Depends(stored)],
        answers: Annotated[dict[str, Any], Body(description="An answer for each waiting call, by its `id` in the run's `waiting`.")],
    ) -> Answered:
        """Answer what waits by its ids, and resume the run with the answers.

        The run takes them as ``run.answer`` does, whatever its status: a run that failed, stopped, timed out or was
        cancelled with questions waiting is answered and goes on, and every answer the run refuses is in the 422. A complete run
        has nothing to answer and is a 409. While the run is running, the answers wait until it stops, and a question
        answered once meanwhile is refused a second answer.
        """
        return Answered(answers=await worker.answer(run, answers))

    @router.post("/runs/{run_id}/resume", status_code=202, responses={**_NOT_FOUND, **_CONFLICT})
    async def resume_run(run: Annotated[RunAsRead, Depends(stored)], body: Resume = Resume()) -> Started:
        """Resume a run that ended early; no node call that finished is made again. ``{"max_items": ...}`` raises
        the ceiling of a run that stopped at it; a resume needs no body."""
        await worker.resume(run, body.max_items)
        return Started(id=run.row.id)

    @router.post("/runs/{run_id}/again", status_code=202, responses=_NOT_FOUND)
    async def run_again(
        run: Annotated[RunAsRead, Depends(stored)],
        inputs: Annotated[dict[str, Any], Body(description="The inputs to change, each by its name or its address `node.input`; the rest stay as the run had them.")],
        who: Annotated[C, Depends(caller)],
    ) -> Started:
        """A new run of the caller's with these inputs changed, on the caller's own nodes; a call that receives what it
        received before isn't made again."""
        return Started(id=await worker.again(run, inputs, who))

    @router.post("/runs/{run_id}/cancel", responses=_NOT_FOUND)
    async def cancel_run(run: Annotated[RunAsRead, Depends(stored)]) -> Cancelled:
        """Stop a running run before its next node call; the calls already made are kept, and ``resume`` goes on."""
        return Cancelled(cancelled=await worker.cancel(run))

    return router


def _caller_type(caller: Callable[..., Any]) -> Any:
    """The type a ``caller`` dependency gives.

    Args:
        caller: The dependency: a class, or a function.

    Returns:
        The class itself, or the function's return annotation.

    Raises:
        TypeError: The function has no return annotation.
    """
    if isinstance(caller, type):
        return caller
    returns = get_type_hints(caller).get("return")
    if returns is None:
        raise TypeError(f"caller {caller!r} has no return annotation; annotate it with the type it gives, so a run keeps its caller as that type")
    return returns
