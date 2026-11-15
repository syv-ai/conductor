"""``run_router``: the routes that start, follow, read, answer, resume and cancel runs, on any number of workers.

Each route parses its call, asks the process's ``Worker`` and says what it
answered; the worker resumes the runs.
"""

from collections.abc import AsyncIterable, AsyncIterator, Callable
from operator import attrgetter
from typing import Annotated, Any, Literal, TypeVar

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

from conductor_providers.fastapi.router_problems import STATUS, no_such_run
from conductor_providers.fastapi.stored_run import RunAsRead, RunView
from conductor_providers.fastapi.worker import Worker

C = TypeVar("C")


#: A graph record as a caller sent it: read by conductor's record validator when the graph is compiled, so every fault in
#: it is refused in conductor's words; the schema says it is a ``GraphRecord``.
GraphAsSent = Annotated[Any, PlainValidator(lambda sent: sent, json_schema_input_type=GraphRecord)]


class StartRun(BaseModel):
    """The body of ``POST /runs``: a graph to run and the values it starts on.

    Attributes:
        graph: The graph, as its stored record; compiled on the caller's registry, so a node it lacks is a 422.
        inputs: Values for the graph's inputs, each by its name in ``graph.inputs``, checked as
            ``graph.with_inputs`` checks them.
    """

    graph: GraphAsSent
    inputs: dict[str, Any] = {}


#: What a route under ``/runs/{run_id}`` declares for a refusal beside its own answer: the 404 of a run that isn't there
#: for the caller, and the 409 of a run in a state that doesn't take the call. A refusal's body is FastAPI's own
#: ``HTTPValidationError``, which a conductor ``Problem`` fits, so a client reads every refusal one way.
_REFUSED: dict[str, Any] = {
    "content": {
        "application/json": {"schema": {"$ref": "#/components/schemas/HTTPValidationError"}}
    }
}
_NOT_FOUND: dict[int | str, dict[str, Any]] = {404: _REFUSED}
_NOT_FOUND_OR_CONFLICT: dict[int | str, dict[str, Any]] = {404: _REFUSED, 409: _REFUSED}


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
            holds for the run from then on. Left out, the run keeps the ceiling it has. The body says only that it
            is an integer (``true`` is not one); that it is at least 1 is conductor's rule, so ``0`` is refused
            as ``Run.with_max_items`` refuses it, with ``invalid_limit``.
    """

    max_items: StrictInt | None = None


class Cancelled(BaseModel):
    """The body of a ``POST /runs/{id}/cancel``.

    Attributes:
        cancelled: Whether the cancel was asked: the run was running or waiting, as read or after another call
            changed it first. It is not that the run has ended ``cancelled``: a waiting run is resumed, with no
            node call, and like any resume call that can end it ``failed`` (its context or a node can't be had);
            ``GET /runs/{id}`` says how it ended. ``false`` only for a run that is complete or already ended
            early, which has nothing to cancel.
    """

    cancelled: bool


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
    worker: Worker[C],
    *,
    caller: Callable[..., C],
    may_see: Callable[[C, C], bool],
    published: Callable[..., GraphRecord] | None = None,
) -> APIRouter:
    """The runs routes over ``worker``, which resumes their runs; mount them with ``app.include_router(...)``.

    Args:
        worker: The worker the routes call: where runs are kept, the nodes and context a run's caller gets, and the
            settings of its leases.
        caller: A FastAPI dependency giving who is calling: your login, of the worker's ``caller_type``. A run
            keeps the caller who started it as they were then, so let it hold what identifies the caller and nothing
            that changes. A host with no logins gives a function that returns ``None``.
        may_see: ``may_see(caller, started_by)`` says whether a caller may read and act on a run:
            ``lambda caller, started_by: caller.id == started_by.id`` for the starter alone,
            ``lambda caller, started_by: True`` for everyone.
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

    async def stored(run_id: str, who: Annotated[C, Depends(caller)]) -> RunAsRead:
        """The run of a route under ``/runs/{run_id}``, for the caller: the access rule of every such route.

        Args:
            run_id: The run's id, from the path.
            who: The caller.

        Returns:
            The run as the store has it.

        Raises:
            InvalidCallError: No run has the id, or ``may_see`` says the caller may not see it (``no_such_run``).
                The two read the same (a 404), so a caller learns nothing about a run that isn't theirs.
        """
        run = await worker.read(run_id)
        if not may_see(who, worker.caller_of(run)):
            raise InvalidCallError([no_such_run()])
        return run

    @router.get("/nodes")
    def list_nodes(who: Annotated[C, Depends(caller)]) -> RegistryDescription:
        """The nodes the caller may use, as an editor's palette reads them."""
        return worker.nodes_for(who).describe()

    @router.post("/compile")
    async def compile_graph(
        graph: Annotated[GraphAsSent, Body()], request: Request, who: Annotated[C, Depends(caller)]
    ) -> GraphDescription:
        """Compile the graph record and describe the result: each node's inputs and outputs, how often it is called,
        and every problem. Nothing is run or stored."""
        return Graph.compile(
            await request.body(), worker.nodes_for(who)
        ).describe()  # the record validator reads the body's text

    @router.post("/runs", status_code=202)
    async def start_run(
        body: StartRun, request: Request, who: Annotated[C, Depends(caller)]
    ) -> Started:
        """Start a run of any graph; this is an author's route, so guard it with your own dependency."""
        sent = StartRun.model_validate(
            read_json(await request.body())
        )  # FastAPI's ``body`` would keep the last of two equal keys
        return Started(id=await worker.begin(sent.graph, sent.inputs, who))

    @router.post("/estimate")
    async def estimate_run(
        body: StartRun, request: Request, who: Annotated[C, Depends(caller)]
    ) -> RunProgress:
        """How many calls a run of the graph on these inputs will make: the nodes whose ``Policy`` says
        ``costs_nothing`` are called, and nothing is stored. A node's ``total`` is ``null`` where it depends on a
        call that costs something."""
        sent = StartRun.model_validate(
            read_json(await request.body())
        )  # FastAPI's ``body`` would keep the last of two equal keys
        return await worker.estimate(sent.graph, sent.inputs, who)

    if published is not None:

        @router.post("/graphs/{name}/runs", status_code=202)
        async def start_published(
            graph: Annotated[GraphRecord, Depends(published)],
            inputs: Annotated[
                dict[str, Any], Body(description="Values for the graph's inputs, each by its name.")
            ],
            who: Annotated[C, Depends(caller)],
        ) -> Started:
            """Start a graph you offer by name, on inputs alone."""
            return Started(id=await worker.begin(graph, inputs, who))

        @router.post("/graphs/{name}/estimate")
        async def estimate_published(
            graph: Annotated[GraphRecord, Depends(published)],
            inputs: Annotated[
                dict[str, Any], Body(description="Values for the graph's inputs, each by its name.")
            ],
            who: Annotated[C, Depends(caller)],
        ) -> RunProgress:
            """How many calls a run of a graph you offer by name will make on these inputs, as ``POST /estimate``."""
            return await worker.estimate(graph, inputs, who)

    @router.get("/runs/{run_id}", responses=_NOT_FOUND)
    def read_run(run: Annotated[RunAsRead, Depends(stored)]) -> RunView:
        """The run as it was when it last stopped running: its status, what waits for a person, its outputs once complete, the
        queued answers that were refused, and why it ended early if it did."""
        return run.view()

    @router.get(
        "/runs/{run_id}/events",
        response_class=EventSourceResponse,
        responses=_NOT_FOUND,
    )
    async def follow_run(
        run: Annotated[RunAsRead, Depends(stored)], last_event_id: Annotated[int, Header()] = -1
    ) -> AsyncIterable[ServerSentEvent]:
        """The run's events, from whichever worker runs it, until it stops running; ``Last-Event-ID`` goes on after a reconnect."""
        async for number, body in worker.events(run, last_event_id):
            yield ServerSentEvent(raw_data=body, id=str(number))

    @router.post("/runs/{run_id}/answers", status_code=202, responses=_NOT_FOUND_OR_CONFLICT)
    async def answer_run(
        run: Annotated[RunAsRead, Depends(stored)],
        answers: Annotated[
            dict[str, Any],
            Body(
                description="An answer for each waiting call, by its `id` in the run's `waiting`."
            ),
        ],
    ) -> Answered:
        """Answer what waits by its ids, and resume the run with the answers.

        The run takes them as ``run.answer`` does, whatever its status: a run that failed, stopped, timed out or was
        cancelled with questions waiting is answered and goes on, and every answer the run refuses is in the 422. A complete run
        has nothing to answer and is a 409. While the run is running, the answers wait until it stops, and a question
        answered once meanwhile is refused a second answer.
        """
        return Answered(answers=await worker.answer(run, answers))

    @router.post("/runs/{run_id}/resume", status_code=202, responses=_NOT_FOUND_OR_CONFLICT)
    async def resume_run(
        run: Annotated[RunAsRead, Depends(stored)], body: Resume = Resume()
    ) -> Started:
        """Resume a run that ended early; no node call that finished is made again. ``{"max_items": ...}`` raises
        the ceiling of a run that stopped at it; a resume needs no body."""
        await worker.resume(run, body.max_items)
        return Started(id=run.id)

    @router.post("/runs/{run_id}/again", status_code=202, responses=_NOT_FOUND)
    async def run_again(
        run: Annotated[RunAsRead, Depends(stored)],
        inputs: Annotated[
            dict[str, Any],
            Body(
                description="The inputs to change, each by its name; the rest stay as the run had them."
            ),
        ],
        who: Annotated[C, Depends(caller)],
    ) -> Started:
        """A new run of the caller's with these inputs changed, on the caller's own nodes; a call that receives what it
        received before isn't made again."""
        return Started(id=await worker.again(run, inputs, who))

    @router.post("/runs/{run_id}/cancel", responses=_NOT_FOUND)
    async def cancel_run(run: Annotated[RunAsRead, Depends(stored)]) -> Cancelled:
        """Ask a running or waiting run to stop before its next node call, so that it ends ``cancelled``; the calls
        already made are kept, its questions still wait, and an answer or ``resume`` goes on. A waiting run is
        resumed for it, with no node call, so like any resume it can end ``failed``."""
        return Cancelled(cancelled=await worker.cancel(run))

    return router
