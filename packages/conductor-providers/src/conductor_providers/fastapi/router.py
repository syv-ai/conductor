"""``run_router``: the routes that start, follow, read, answer, resume and cancel runs, on any number of workers.

Each route parses its call, asks the process's ``Worker`` and says what it
answered; the worker resumes the runs.
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
    read_json,
)
from fastapi import APIRouter, Body, Depends, Header, HTTPException, Request
from fastapi.sse import EventSourceResponse, ServerSentEvent
from pydantic import BaseModel, PlainValidator, StrictInt

from conductor_providers.fastapi.problems import STATUS, no_such_run
from conductor_providers.fastapi.stored_run import M, RunView
from conductor_providers.fastapi.worker import Worker, X

#: A graph record as a caller sent it: read by conductor's record validator when the graph is compiled, so every fault in
#: it is refused in conductor's words; the schema says it is a ``GraphRecord``.
GraphAsSent = Annotated[Any, PlainValidator(lambda sent: sent, json_schema_input_type=GraphRecord)]


class StartRun(BaseModel):
    """The body of ``POST /runs``: a graph to run and the values it starts on.

    Attributes:
        graph: The graph, as its stored record; compiled on the request's registry, so a node it lacks is a 422.
        inputs: Values for the graph's inputs, each by its name in ``graph.inputs``, checked as
            ``graph.with_inputs`` checks them.
    """

    graph: GraphAsSent
    inputs: dict[str, Any] = {}


#: What a route under ``/runs/{run_id}`` declares for a refusal beside its own answer: the 404 of a run that isn't there
#: for the request, and the 409 of a run in a state that doesn't take the call. A refusal's body is FastAPI's own
#: ``HTTPValidationError``, which a conductor ``Problem`` fits, so a client reads every refusal one way.
_REFUSED: dict[str, Any] = {
    "content": {
        "application/json": {"schema": {"$ref": "#/components/schemas/HTTPValidationError"}}
    }
}
_NOT_FOUND: dict[int | str, dict[str, Any]] = {404: _REFUSED}
_NOT_FOUND_OR_CONFLICT: dict[int | str, dict[str, Any]] = {404: _REFUSED, 409: _REFUSED}


class Resume(BaseModel):
    """The body of ``POST /runs/{id}/resume``; a resume needs none.

    Attributes:
        max_items: A higher ceiling on the items of any one Series, for a run that stopped at its ceiling; it
            holds for the run from then on. Left out, the run keeps the ceiling it has. The body says only that it
            is an integer (``true`` is not one); that it is at least 1 is conductor's rule, so ``0`` is refused
            as ``Run.with_max_items`` refuses it, with ``invalid_limit``.
    """

    max_items: StrictInt | None = None


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
    published: Callable[..., GraphRecord] | None = None,
) -> APIRouter:
    """The runs routes over ``worker``, which resumes their runs; mount them with ``app.include_router(...)``.

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
        published: A FastAPI dependency taking the path's ``name`` and giving a ``GraphRecord``, raising
            ``HTTPException(404)`` for one it doesn't offer; with it, ``POST /graphs/{name}/runs`` starts that
            graph on inputs alone.

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

    async def allowed(
        run_id: str, allows: Annotated[Callable[[M], bool], Depends(run_filter)]
    ) -> str:
        """The id of a route under ``/runs/{run_id}``, if the run filter lets this request see the run.

        Only the run's metadata is read to decide, so the check stays cheap however much waits in the run.

        Args:
            run_id: The run's id, from the path.
            allows: The run filter for this request.

        Returns:
            The run's id.

        Raises:
            InvalidCallError: No run has the id, or the filter turns the request away (``no_such_run``). The two
                read the same (a 404), so a request learns nothing about a run it may not see.
        """
        if not allows(await worker.metadata_of(run_id)):
            raise InvalidCallError([no_such_run()])
        return run_id

    @router.get("/nodes")
    def list_nodes(requester: Annotated[M, Depends(metadata)]) -> RegistryDescription:
        """The nodes this request may use, as an editor's palette reads them."""
        return worker.nodes_for(requester).describe()

    @router.post("/compile")
    async def compile_graph(
        graph: Annotated[GraphAsSent, Body()],
        request: Request,
        requester: Annotated[M, Depends(metadata)],
    ) -> GraphDescription:
        """Compile the graph record and describe the result: each node's inputs and outputs, how often it is called,
        and every problem. Nothing is run or stored."""
        return Graph.compile(
            await request.body(), worker.nodes_for(requester)
        ).describe()  # the record validator reads the body's text

    @router.post("/runs", status_code=202, response_model=view)
    async def start_run(
        body: StartRun, request: Request, requester: Annotated[M, Depends(metadata)]
    ) -> Any:
        """Start a run of any graph; this is an author's route, so guard it with your own dependency."""
        sent = StartRun.model_validate(
            read_json(await request.body())
        )  # FastAPI's ``body`` would keep the last of two equal keys
        return await worker.begin(sent.graph, sent.inputs, requester)

    @router.post("/estimate")
    async def estimate_run(
        body: StartRun, request: Request, requester: Annotated[M, Depends(metadata)]
    ) -> RunProgress:
        """How many calls a run of the graph on these inputs will make: the nodes whose ``Policy`` says
        ``costs_nothing`` are called, and nothing is stored. A node's ``total`` is ``null`` where it depends on a
        call that costs something."""
        sent = StartRun.model_validate(
            read_json(await request.body())
        )  # FastAPI's ``body`` would keep the last of two equal keys
        return await worker.estimate(sent.graph, sent.inputs, requester)

    if published is not None:

        @router.post("/graphs/{name}/runs", status_code=202, response_model=view)
        async def start_published(
            graph: Annotated[GraphRecord, Depends(published)],
            inputs: Annotated[
                dict[str, Any], Body(description="Values for the graph's inputs, each by its name.")
            ],
            requester: Annotated[M, Depends(metadata)],
        ) -> Any:
            """Start a graph you offer by name, on inputs alone."""
            return await worker.begin(graph, inputs, requester)

        @router.post("/graphs/{name}/estimate")
        async def estimate_published(
            graph: Annotated[GraphRecord, Depends(published)],
            inputs: Annotated[
                dict[str, Any], Body(description="Values for the graph's inputs, each by its name.")
            ],
            requester: Annotated[M, Depends(metadata)],
        ) -> RunProgress:
            """How many calls a run of a graph you offer by name will make on these inputs, as ``POST /estimate``."""
            return await worker.estimate(graph, inputs, requester)

    @router.get("/runs/{run_id}", response_model=view, responses=_NOT_FOUND)
    async def read_run(run_id: Annotated[str, Depends(allowed)]) -> Any:
        """The run as it was when it last stopped running: its status, what waits for a person, its outputs once
        complete, the answers that were refused, and why it ended early if it did."""
        return await worker.read(run_id)

    @router.get(
        "/runs/{run_id}/events",
        response_class=EventSourceResponse,
        responses=_NOT_FOUND,
    )
    async def follow_run(
        run_id: Annotated[str, Depends(allowed)], last_event_id: Annotated[int, Header()] = -1
    ) -> AsyncIterable[ServerSentEvent]:
        """The run's events, from whichever worker runs it, until it stops running; ``Last-Event-ID`` goes on after a
        reconnect."""
        async for number, body in worker.events(run_id, last_event_id):
            yield ServerSentEvent(raw_data=body, id=str(number))

    @router.post(
        "/runs/{run_id}/answers",
        status_code=202,
        response_model=view,
        responses=_NOT_FOUND_OR_CONFLICT,
    )
    async def answer_run(
        run_id: Annotated[str, Depends(allowed)],
        answers: Annotated[
            dict[str, Any],
            Body(
                description="An answer for each waiting call, by its `id` in the run's `waiting`."
            ),
        ],
    ) -> Any:
        """Answer what waits by its ids, and resume the run with the answers.

        The run takes them as ``run.answer`` does, whatever its status: a run that failed, stopped, timed out or was
        cancelled with questions waiting is answered and goes on, and every answer the run refuses is in the 422.
        While the run is running, the answers wait until it stops, and are given to it then; one it refuses then is
        in the run's ``refused``.
        """
        return await worker.answer(run_id, answers)

    @router.post(
        "/runs/{run_id}/resume",
        status_code=202,
        response_model=view,
        responses=_NOT_FOUND_OR_CONFLICT,
    )
    async def resume_run(run_id: Annotated[str, Depends(allowed)], body: Resume = Resume()) -> Any:
        """Resume a run that ended early; no node call that finished is made again. ``{"max_items": ...}`` raises
        the ceiling of a run that stopped at it; a resume needs no body."""
        return await worker.resume(run_id, body.max_items)

    @router.post("/runs/{run_id}/again", status_code=202, response_model=view, responses=_NOT_FOUND)
    async def run_again(
        run_id: Annotated[str, Depends(allowed)],
        inputs: Annotated[
            dict[str, Any],
            Body(
                description="The inputs to change, each by its name; the rest stay as the run had them."
            ),
        ],
        requester: Annotated[M, Depends(metadata)],
    ) -> Any:
        """A new run with these inputs changed, of this request's own and on its own nodes; a call that receives what
        it received before isn't made again."""
        return await worker.again(run_id, inputs, requester)

    @router.post("/runs/{run_id}/cancel", response_model=view, responses=_NOT_FOUND)
    async def cancel_run(run_id: Annotated[str, Depends(allowed)]) -> Any:
        """Ask a running or waiting run to stop before its next node call, so that it ends ``cancelled``; the calls
        already made are kept, its questions still wait, and an answer or ``resume`` goes on. A waiting run is
        resumed for it, with no node call, so like any resume it can end ``failed``. A run that is complete or
        ended early is answered as it is."""
        return await worker.cancel(run_id)

    return router
