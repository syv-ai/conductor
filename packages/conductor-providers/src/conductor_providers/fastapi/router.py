"""``run_router``: the HTTP routes over a ``Worker``.

Each route reads its request, calls one method of the worker and answers with what it gave back. The worker does
the real work: it keeps runs in the store, resumes them, and takes over the runs of a worker that died.
"""

from collections.abc import AsyncIterable, AsyncIterator, Callable, Coroutine
from operator import attrgetter
from typing import Annotated, Any

from conductor import (
    Graph,
    GraphDescription,
    GraphRecord,
    InvalidCallError,
    ProblemKind,
    RegistryDescription,
    RunProgress,
)
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.routing import APIRoute
from fastapi.sse import EventSourceResponse, ServerSentEvent

from conductor_providers.fastapi.bodies import Decline, Resume, StartRun, declared
from conductor_providers.fastapi.problems import STATUS, no_such_run, run_not_resumable
from conductor_providers.fastapi.state import M, RunView
from conductor_providers.fastapi.worker import Worker, X

#: A refusal's body: FastAPI's own ``HTTPValidationError``, the shape of a 422, which a conductor ``Problem`` fits, so
#: a client reads every refusal one way.
_REFUSED: dict[str, Any] = {
    "content": {
        "application/json": {"schema": {"$ref": "#/components/schemas/HTTPValidationError"}}
    }
}


def _refusals(*kinds: ProblemKind) -> dict[int | str, dict[str, Any]]:
    """The ``responses`` a route declares for the worker's own codes it can refuse with, each at the status
    ``STATUS`` gives it: ``_refusals(no_such_run, run_not_resumable)`` declares a 404 and a 409.

    The 422 for what conductor or FastAPI refuses is declared where a route's input is: FastAPI declares it for every
    parameter it reads, and ``_body`` for a body read as text.

    Args:
        kinds: The worker's own kinds of problem the route can refuse with.
    """
    return {STATUS[kind.code]: _REFUSED for kind in kinds}


def _body(schema: dict[str, Any]) -> dict[str, Any]:
    """The ``openapi_extra`` of a route that reads its body as text: a required JSON body of ``schema``, and the 422
    a body conductor refuses is answered with, which FastAPI doesn't declare for a body it doesn't read."""
    return {
        "requestBody": {"required": True, "content": {"application/json": {"schema": schema}}},
        # FastAPI's own words for the 422 it declares for a parameter, so every 422 reads alike
        "responses": {"422": {"description": "Validation Error", **_REFUSED}},
    }


async def _refused_as_responses() -> AsyncIterator[None]:
    """Answers every call a route's worker refuses with the problems.

    The worker raises conductor's ``InvalidCallError`` for every refusal. Its first problem's code picks the status
    from ``STATUS`` (404 or 409), and any other code is a 422.

    Yields:
        Nothing; the route runs inside it.

    Raises:
        HTTPException: For a refused call, with the problems as its ``detail``.
    """
    try:
        yield
    except InvalidCallError as refused:
        status = STATUS.get(refused.problems[0].type, 422)
        detail = [problem.model_dump(mode="json") for problem in refused.problems]
        raise HTTPException(status, detail) from None


class _ValueFreeRoute(APIRoute):
    """A route whose 422 for a request FastAPI or a body's reader refuses quotes nothing the request sent.

    Pydantic keeps the refused value in each error's ``input``, and FastAPI writes it into the 422: the whole body
    for a body that isn't an object, or a header's text. A refusal never quotes a value
    (``python -m conductor.about errors``), so every error is answered without it, on every route.
    """

    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        handler = super().get_route_handler()

        async def handled(request: Request) -> Response:
            try:
                return await handler(request)
            except RequestValidationError as refused:
                errors = [
                    {key: value for key, value in error.items() if key != "input"}
                    for error in refused.errors()
                ]
                raise RequestValidationError(errors) from None

        return handled


def run_router(
    worker: Worker[M, X],
    *,
    metadata: Callable[..., M],
    run_filter: Callable[..., Callable[[M], bool]],
) -> APIRouter:
    """The runs routes over ``worker``; mount them with ``app.include_router(...)``.

    Who may reach the routes that take any graph (``POST /runs``, ``/estimate``, ``/compile``, ``/nodes``) is your
    call, made the usual FastAPI way: ``app.include_router(router, dependencies=[Depends(is_author)])``.

    Args:
        worker: The worker the routes call: where runs are kept, the nodes and context a run gets from its
            metadata, and the settings of its leases.
        metadata: A FastAPI dependency giving the metadata for this request, of the worker's ``metadata_type``.
            A run started by the request keeps it, as it is then, and its nodes and context are built from it on
            every resume. On the routes that store no run (``/nodes``, ``/compile``, ``/estimate``, and ``again``'s
            check of the new run), it picks the nodes this request may use. Let it hold what identifies the
            request, such as a user and a team, and nothing that changes.
        run_filter: A FastAPI dependency giving, for this request, a test of a run's metadata: whether the
            request may read and act on that run. ``lambda metadata: metadata["user"] == user.id`` lets a person
            see only their own runs. A run it turns away is a 404, as one that isn't there.

    Returns:
        The router, whose lifespan runs the worker.
    """
    router = APIRouter(
        lifespan=lambda _: worker.serving(),
        dependencies=[Depends(_refused_as_responses)],
        generate_unique_id_function=attrgetter("name"),
        route_class=_ValueFreeRoute,
    )
    start_body = _body(declared(StartRun))

    async def followed(
        run_id: str,
        allows: Annotated[Callable[[M], bool], Depends(run_filter)],
        last_event_id: Annotated[int, Header()] = -1,
    ) -> AsyncIterator[tuple[int, str]]:
        """The run's events, checked before the stream starts, so a run the request may not see is a 404."""
        return await worker.events(run_id, last_event_id, allows=allows)

    @router.get("/nodes")
    def list_nodes(requester: Annotated[M, Depends(metadata)]) -> RegistryDescription:
        """The nodes this request may use, as an editor's palette reads them."""
        return worker.nodes_for(requester).describe()

    @router.post("/compile", openapi_extra=_body(declared(GraphRecord)))
    async def compile_graph(
        request: Request, requester: Annotated[M, Depends(metadata)]
    ) -> GraphDescription:
        """Compile the graph record and describe the result: each node's inputs and outputs, how often it is called,
        and every problem. Nothing is run or stored."""
        return Graph.compile(await request.body(), worker.nodes_for(requester)).describe()

    @router.post(
        "/runs", status_code=202, openapi_extra=start_body, responses=_refusals(no_such_run)
    )
    async def start_run(
        request: Request,
        requester: Annotated[M, Depends(metadata)],
        allows: Annotated[Callable[[M], bool], Depends(run_filter)],
    ) -> RunView:
        """Start a run of any graph. This is an author's route, so guard it with a dependency of your own."""
        sent = StartRun.read(await request.body())
        return await worker.begin(
            sent.graph, sent.inputs, requester, previous=sent.previous, allows=allows
        )

    @router.post("/estimate", openapi_extra=start_body, responses=_refusals(no_such_run))
    async def estimate_run(
        request: Request,
        requester: Annotated[M, Depends(metadata)],
        allows: Annotated[Callable[[M], bool], Depends(run_filter)],
    ) -> RunProgress:
        """How many calls the run ``POST /runs`` would start will make: the nodes whose ``Policy`` says
        ``costs_nothing`` are called, and nothing is stored. A node's ``total`` is ``null`` where it depends on a
        call that costs something."""
        sent = StartRun.read(await request.body())
        return await worker.estimate(
            sent.graph, sent.inputs, requester, previous=sent.previous, allows=allows
        )

    @router.get("/runs/{run_id}", responses=_refusals(no_such_run))
    async def read_run(
        run_id: str, allows: Annotated[Callable[[M], bool], Depends(run_filter)]
    ) -> RunView:
        """The run as it was when it last stopped running: its status, what waits for a person, its outputs once
        complete, the answers that were refused, and why it ended early if it did."""
        return await worker.read(run_id, allows=allows)

    @router.get(
        "/runs/{run_id}/events",
        response_class=EventSourceResponse,
        responses=_refusals(no_such_run),
    )
    async def follow_run(
        events: Annotated[AsyncIterator[tuple[int, str]], Depends(followed)],
    ) -> AsyncIterable[ServerSentEvent]:
        """The run's events, from whichever worker runs it, until it stops running. ``Last-Event-ID`` goes on after
        a reconnect."""
        async for number, body in events:
            yield ServerSentEvent(raw_data=body, id=str(number))

    @router.post("/runs/{run_id}/answers", status_code=202, responses=_refusals(no_such_run))
    async def answer_run(
        run_id: str,
        allows: Annotated[Callable[[M], bool], Depends(run_filter)],
        answers: Annotated[
            dict[str, Any],
            Body(
                description="An answer for each waiting call, by its `id` in the run's `waiting`."
            ),
        ],
    ) -> RunView:
        """Answer what waits by its ids, whatever the run's status: the run goes on with them, and an answer it refuses
        is a 422, such as any answer to a ``declined`` run; while the run is running, the answers are queued until it
        stops, and one it refuses then is in its ``refused``."""
        return await worker.answer(run_id, answers, allows=allows)

    @router.post("/runs/{run_id}/declines", status_code=202, responses=_refusals(no_such_run))
    async def decline_run(
        run_id: str,
        allows: Annotated[Callable[[M], bool], Depends(run_filter)],
        body: Decline,
    ) -> RunView:
        """Decline to answer what waits under ``waiting_id``: a run that takes it ends ``declined`` with
        ``answer_declined``, for good, and one that refuses it is a 422. While the run is running, the decline is
        queued until it stops, and a refusal then is in its ``refused``."""
        return await worker.decline(run_id, body.waiting_id, reason=body.reason, allows=allows)

    @router.post(
        "/runs/{run_id}/resume",
        status_code=202,
        responses=_refusals(no_such_run, run_not_resumable),
    )
    async def resume_run(
        run_id: str,
        allows: Annotated[Callable[[M], bool], Depends(run_filter)],
        body: Resume = Resume(),
    ) -> RunView:
        """Resume a run that ended early; no node call that finished is made again, and ``{"max_items": ...}`` raises
        the item limit of a run that stopped at it. A ``declined`` run has ended for good, and is refused."""
        return await worker.resume(run_id, body.max_items, allows=allows)

    @router.post("/runs/{run_id}/again", status_code=202, responses=_refusals(no_such_run))
    async def run_again(
        run_id: str,
        allows: Annotated[Callable[[M], bool], Depends(run_filter)],
        inputs: Annotated[
            dict[str, Any],
            Body(
                description="The inputs to change, each by its name; the rest stay as the run had them."
            ),
        ],
        requester: Annotated[M, Depends(metadata)],
    ) -> RunView:
        """A new run with these inputs changed, of this request's own and on its own nodes. A call that gets what it
        got before isn't made again."""
        return await worker.again(run_id, inputs, requester, allows=allows)

    @router.post("/runs/{run_id}/cancel", responses=_refusals(no_such_run))
    async def cancel_run(
        run_id: str, allows: Annotated[Callable[[M], bool], Depends(run_filter)]
    ) -> RunView:
        """Ask a running or waiting run to stop before its next node call, so that it ends ``cancelled``. A run that
        answers ``running`` ends ``cancelled``, unless it ends early first, even if its calls all finish first. The
        calls already made are kept, its questions still wait, and an answer or a resume goes on. A waiting run is
        resumed for it, with no node call, so like any resume it can end ``failed``. A run that is complete or ended
        early is answered as it is."""
        return await worker.cancel(run_id, allows=allows)

    return router
