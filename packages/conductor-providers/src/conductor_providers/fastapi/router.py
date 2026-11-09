"""FastAPI ``APIRouter`` factory for conductor."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from conductor import (
    CompiledGraph,
    Graph,
    GraphDescription,
    InvalidCallError,
    NodeRegistry,
    RegistryDescription,
    RunState,
    to_jsonable,
)
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from conductor_providers.fastapi.models import RunRequest, RunResult


def conductor_router(
    registry: NodeRegistry,
    *,
    prefix: str = "",
    tags: list[str] | None = None,
    dependencies: Sequence[Any] | None = None,
    from_run: Callable[[Request], Mapping[type, Any]] | None = None,
) -> APIRouter:
    """Build a FastAPI ``APIRouter`` serving conductor's standard endpoints.

    Mounts:

    - ``GET  {prefix}/nodes``           — ``registry.describe()``: every node and type
    - ``POST {prefix}/run``             — one leg; returns ``state`` and every
      node's ``results``, however the leg ended. ``inputs`` in the body fills the
      graph's inputs, by name or address, as ``with_inputs`` does
    - ``POST {prefix}/run-stream``      — SSE stream of event frames

    A run that asks goes on in legs: send its ``state`` back with the
    ``answers`` (``RunRequest``).
    - ``POST {prefix}/compile``         — takes a ``Graph``, compiles without
      executing, and answers ``compiled.describe()``

    Args:
        registry: The populated ``NodeRegistry`` to serve.
        prefix: Path prefix applied to every route (FastAPI convention).
        tags: OpenAPI tags attached to every route.
        dependencies: FastAPI dependencies applied to every route (auth, rate
            limiting, anything ``Depends(...)`` can express).
        from_run: Optional hook invoked per request on ``/run`` and
            ``/run-stream``. Receives the FastAPI ``Request`` and returns
            the values the run supplies by type, ``run(from_run=...)``: a
            parameter annotated ``FromRun[X]`` receives
            the value keyed by ``X``.

    A request the run cannot start from is the caller's fault and a 422,
    with ``detail`` the ``InvalidCallError``'s ``errors()``: a graph with a
    fatal problem, a state that does not read, a wrong answer, inputs it can't take. Anything else
    raised before the leg starts is the server's and stays a 500. ``/compile`` answers a broken graph with 200 and its
    problems, since describing it is what was asked.
    """
    router = APIRouter(
        prefix=prefix,
        tags=tags or ["conductor"],
        dependencies=list(dependencies) if dependencies else None,
    )
    def _prepared(req: RunRequest) -> tuple[CompiledGraph, RunState | None]:
        """The compiled graph and the state this leg goes on from, the answers written in."""
        compiled = CompiledGraph.from_graph(req.graph, registry)
        if req.inputs:
            compiled = compiled.with_inputs(req.inputs)
        if not req.answers:
            return compiled, req.state
        return compiled, compiled.answer(req.state or RunState(status="waiting"), req.answers)

    @router.get("/nodes", response_model=RegistryDescription)
    def list_nodes() -> RegistryDescription:
        """Every node and every type of the registry: what a palette and an edge check read."""
        return registry.describe()

    @router.post("/run", response_model=RunResult)
    async def run_graph(req: RunRequest, request: Request) -> RunResult:
        """Run one leg, and return its ``state`` and every node's ``results``."""
        try:
            compiled, state = _prepared(req)
            state = await compiled.run(state, from_run=from_run(request) if from_run else None)
        except InvalidCallError as invalid_call:
            raise _unprocessable(invalid_call) from invalid_call
        return RunResult(state=state, results=to_jsonable(compiled.results(state)))

    @router.post("/run-stream")
    async def run_graph_stream(req: RunRequest, request: Request) -> StreamingResponse:
        """Run one leg and stream its events as Server-Sent Events.

        Every check is made before the response starts, so a refused run fails
        the request as it does on ``/run``, instead of a 200 with an empty stream.
        """
        try:
            compiled, state = _prepared(req)
            leg = compiled.stream(state, from_run=from_run(request) if from_run else None)
        except InvalidCallError as invalid_call:
            raise _unprocessable(invalid_call) from invalid_call

        async def event_stream() -> Any:
            async with leg as events:
                async for event in events:
                    yield f"data: {event.model_dump_json()}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @router.post("/compile", response_model=GraphDescription)
    def compile_only(graph: Graph) -> GraphDescription:
        """Compile a graph without executing it, and describe it for an editor:
        every problem, each anchored on a node (none means the graph runs),
        where each node runs, and the fields each node has.

        Debounce-friendly (~10-30 ms): hosts can poll this on every graph
        edit to paint type mismatches and cycles in real time.
        """
        return CompiledGraph.from_graph(graph, registry).describe()

    return router


def _unprocessable(invalid_call: InvalidCallError) -> HTTPException:
    return HTTPException(status_code=422, detail=[fault.model_dump(mode="json") for fault in invalid_call.errors()])
