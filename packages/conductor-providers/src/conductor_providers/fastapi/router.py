"""``run_router``: the HTTP routes over a ``Worker``.

Each route reads its request, calls one method of the worker and answers with what it gave back. The worker does
the real work: it keeps runs in the store, resumes them, and takes over the runs of a worker that died.
"""

from collections.abc import AsyncIterable, AsyncIterator, Callable
from operator import attrgetter
from typing import Annotated, Any

from conductor import (
    Graph,
    GraphDescription,
    GraphRecord,
    InvalidCallError,
    RegistryDescription,
    RunProgress,
)
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent

from conductor_providers.fastapi.bodies import Resume, StartRun, declared
from conductor_providers.fastapi.problems import STATUS, no_such_run
from conductor_providers.fastapi.stored_run import M, RunView
from conductor_providers.fastapi.worker import Worker, X

#: How a route declares a refusal beside its own answer: the 404 of a run that isn't there for the request, and the
#: 409 of a run in a state that doesn't take the call. Its body is FastAPI's own ``HTTPValidationError``, the shape of
#: a 422, which a conductor ``Problem`` fits, so a client reads every refusal one way.
_REFUSED: dict[str, Any] = {
    "content": {
        "application/json": {"schema": {"$ref": "#/components/schemas/HTTPValidationError"}}
    }
}
_NOT_FOUND: dict[int | str, dict[str, Any]] = {404: _REFUSED}
_NOT_FOUND_OR_CONFLICT: dict[int | str, dict[str, Any]] = {404: _REFUSED, 409: _REFUSED}


def _body(schema: dict[str, Any]) -> dict[str, Any]:
    """The ``openapi_extra`` of a route that reads its body as text: a required JSON body of ``schema``."""
    return {"requestBody": {"required": True, "content": {"application/json": {"schema": schema}}}}


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
    )
    # What every route that acts on a run answers with: its view, with metadata of the worker's type. mypy reads a
    # subscript as a type, and the type is only known at run time.
    view = RunView[worker.metadata_type]  # type: ignore[name-defined]
    start_body = _body(declared(StartRun))

    async def check_visible(run_id: str, allows: Callable[[M], bool]) -> None:
        """Refuse a run the run filter doesn't let this request see, as if it weren't there.

        Only the run's metadata is read, so the check stays cheap however much waits in the run.

        Raises:
            InvalidCallError: No run has the id, or the filter turns the request away (``no_such_run``). The two
                read the same (a 404), so a request learns nothing about a run it may not see.
        """
        if not allows(await worker.metadata_of(run_id)):
            raise InvalidCallError([no_such_run()])

    async def visible(
        run_id: str, allows: Annotated[Callable[[M], bool], Depends(run_filter)]
    ) -> str:
        """The run id of a route under ``/runs/{run_id}``, once the run filter lets this request see the run."""
        await check_visible(run_id, allows)
        return run_id

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

    @router.post("/runs", status_code=202, response_model=view, openapi_extra=start_body)
    async def start_run(
        request: Request,
        requester: Annotated[M, Depends(metadata)],
        allows: Annotated[Callable[[M], bool], Depends(run_filter)],
    ) -> Any:
        """Start a run of any graph. This is an author's route, so guard it with a dependency of your own."""
        sent = StartRun.read(await request.body())
        if sent.previous is not None:
            await check_visible(sent.previous, allows)
        return await worker.begin(sent.graph, sent.inputs, requester, previous=sent.previous)

    @router.post("/estimate", openapi_extra=start_body)
    async def estimate_run(
        request: Request,
        requester: Annotated[M, Depends(metadata)],
        allows: Annotated[Callable[[M], bool], Depends(run_filter)],
    ) -> RunProgress:
        """How many calls the run ``POST /runs`` would start will make: the nodes whose ``Policy`` says
        ``costs_nothing`` are called, and nothing is stored. A node's ``total`` is ``null`` where it depends on a
        call that costs something."""
        sent = StartRun.read(await request.body())
        if sent.previous is not None:
            await check_visible(sent.previous, allows)
        return await worker.estimate(sent.graph, sent.inputs, requester, previous=sent.previous)

    @router.get("/runs/{run_id}", response_model=view, responses=_NOT_FOUND)
    async def read_run(run_id: Annotated[str, Depends(visible)]) -> Any:
        """The run as it was when it last stopped running: its status, what waits for a person, its outputs once
        complete, the answers that were refused, and why it ended early if it did."""
        return await worker.read(run_id)

    @router.get("/runs/{run_id}/events", response_class=EventSourceResponse, responses=_NOT_FOUND)
    async def follow_run(
        run_id: Annotated[str, Depends(visible)], last_event_id: Annotated[int, Header()] = -1
    ) -> AsyncIterable[ServerSentEvent]:
        """The run's events, from whichever worker runs it, until it stops running. ``Last-Event-ID`` goes on after
        a reconnect."""
        async for number, body in worker.events(run_id, last_event_id):
            yield ServerSentEvent(raw_data=body, id=str(number))

    @router.post(
        "/runs/{run_id}/answers",
        status_code=202,
        response_model=view,
        responses=_NOT_FOUND_OR_CONFLICT,
    )
    async def answer_run(
        run_id: Annotated[str, Depends(visible)],
        answers: Annotated[
            dict[str, Any],
            Body(
                description="An answer for each waiting call, by its `id` in the run's `waiting`."
            ),
        ],
    ) -> Any:
        """Answer what waits by its ids, and resume the run with the answers.

        The run takes them whatever its status, so a run that failed, stopped, timed out or was cancelled with
        questions waiting is answered and goes on. Every answer it refuses is in the 422. While the run is running,
        the answers wait until it stops and are given to it then; one it refuses then is in the run's ``refused``.
        """
        return await worker.answer(run_id, answers)

    @router.post(
        "/runs/{run_id}/resume",
        status_code=202,
        response_model=view,
        responses=_NOT_FOUND_OR_CONFLICT,
    )
    async def resume_run(run_id: Annotated[str, Depends(visible)], body: Resume = Resume()) -> Any:
        """Resume a run that ended early; no node call that finished is made again. ``{"max_items": ...}`` raises
        the item limit of a run that stopped at it, and a resume needs no body."""
        return await worker.resume(run_id, body.max_items)

    @router.post("/runs/{run_id}/again", status_code=202, response_model=view, responses=_NOT_FOUND)
    async def run_again(
        run_id: Annotated[str, Depends(visible)],
        inputs: Annotated[
            dict[str, Any],
            Body(
                description="The inputs to change, each by its name; the rest stay as the run had them."
            ),
        ],
        requester: Annotated[M, Depends(metadata)],
    ) -> Any:
        """A new run with these inputs changed, of this request's own and on its own nodes. A call that gets what it
        got before isn't made again."""
        return await worker.again(run_id, inputs, requester)

    @router.post("/runs/{run_id}/cancel", response_model=view, responses=_NOT_FOUND)
    async def cancel_run(run_id: Annotated[str, Depends(visible)]) -> Any:
        """Ask a running or waiting run to stop before its next node call, so that it ends ``cancelled``. The calls
        already made are kept, its questions still wait, and an answer or a resume goes on. A waiting run is resumed
        for it, with no node call, so like any resume it can end ``failed``. A run that is complete or ended early is
        answered as it is."""
        return await worker.cancel(run_id)

    return router
