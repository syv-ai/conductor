"""FastAPI ``APIRouter`` factory for conductor."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from conductor import NodeRegistry
from conductor.errors import CompilationError, StartRefused
from conductor.execution.engine import execute
from conductor.execution.events import ExecutionEvent
from conductor.graph.compiled import CompiledGraph
from conductor.graph.problem import Problem
from conductor.node import NodeDescription
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from conductor_providers.fastapi.models import ExecuteRequest
from conductor_providers.fastapi.sse import as_data, sse_frame


def conductor_router(
    registry: NodeRegistry,
    *,
    prefix: str = "",
    tags: list[str] | None = None,
    dependencies: Sequence[Any] | None = None,
    from_run: Callable[[Request], Mapping[type, Any]] | None = None,
    entity_resolver: (
        Callable[[str, Request], list[dict[str, Any]]] | None
    ) = None,
) -> APIRouter:
    """Build a FastAPI ``APIRouter`` serving conductor's standard endpoints.

    Mounts:

    - ``GET  {prefix}/nodes``           — every definition's ``describe()``
    - ``POST {prefix}/execute``         — one leg; returns the frame it ended
      on, ``graph_complete`` or ``graph_pending``
    - ``POST {prefix}/execute-stream``  — SSE stream of ``ExecutionEvent`` frames

    A run that asks goes on in legs: send the ending's ``state`` back with
    the answers in ``cache`` (``ExecuteRequest``).
    - ``POST {prefix}/compile``         — compile without executing; returns
      every ``Problem`` the graph has

    Args:
        registry: The populated ``NodeRegistry`` to serve.
        prefix: Path prefix applied to every route (FastAPI convention).
        tags: OpenAPI tags attached to every route.
        dependencies: FastAPI dependencies applied to every route (auth, rate
            limiting, anything ``Depends(...)`` can express).
        from_run: Optional hook invoked per request on ``/execute`` and
            ``/execute-stream``. Receives the FastAPI ``Request`` and returns
            the values the run supplies by type, ``execute(from_run=...)``: a
            ``run`` parameter annotated ``Annotated[X, FromRun()]`` receives
            the value keyed by ``X``.
        entity_resolver: Optional hook backing the ``EntityDropdown`` widget.
            Receives the entity kind (e.g. ``"document"``) and the FastAPI
            ``Request``; returns a list of ``{"id": ..., "label": ...}``
            dicts the frontend renders as choices. ``GET {prefix}/entities/{kind}``
            is mounted only when one is given.

    A request the run cannot start from is the caller's fault and a 422:
    a graph with a fatal problem (``detail.problems``, the same records
    ``/compile`` answers with), or a ``cache`` or ``state`` the run refuses
    (``StartRefused``; ``detail`` is the reason). Anything else raised
    before the first event is the server's and stays a 500. ``/compile`` answers a broken graph with 200 and its
    problems, since describing it is what was asked.
    """
    router = APIRouter(
        prefix=prefix,
        tags=tags or ["conductor"],
        dependencies=list(dependencies) if dependencies else None,
    )
    def _from_run(request: Request) -> Mapping[type, Any] | None:
        return from_run(request) if from_run else None

    def _leg(req: ExecuteRequest, request: Request) -> Any:
        compiled = CompiledGraph.from_graph(req.graph, registry)
        return execute(
            compiled,
            from_run=_from_run(request),
            cache=req.cache or None,
            state=req.state,
        )

    async def _started(req: ExecuteRequest, request: Request) -> tuple[Any, ExecutionEvent]:
        """The leg and its first event, or the 422 a refused start is."""
        events = _leg(req, request)
        try:
            return events, await anext(events)
        except CompilationError as refused:
            raise HTTPException(
                status_code=422,
                detail={"message": "the graph cannot run", "problems": [p.model_dump(mode="json") for p in refused.problems]},
            ) from refused
        except StartRefused as refused:
            raise HTTPException(status_code=422, detail=str(refused)) from refused

    @router.get("/nodes", response_model=list[NodeDescription])
    def list_nodes() -> list[NodeDescription]:
        """Every registered definition as a record — the palette."""
        return [cls.describe() for cls in registry.nodes]

    @router.post("/execute")
    async def execute_graph(req: ExecuteRequest, request: Request) -> dict[str, Any]:
        """Run one leg and return the frame it ended on.

        Every ending is an answer: ``graph_complete`` carries the results,
        ``graph_pending`` the questions and the state the next request
        sends back, ``graph_error``, ``graph_cancelled`` and
        ``graph_timeout`` why the leg stopped.
        """
        events, ending = await _started(req, request)
        async for ending in events:
            pass
        return as_data(ending)

    @router.post("/execute-stream")
    async def execute_graph_stream(
        req: ExecuteRequest, request: Request
    ) -> StreamingResponse:
        """Run a graph and stream ``ExecutionEvent``s as Server-Sent Events.

        The first event is taken before the response starts, so a run that
        is refused before anything runs (a graph that cannot run, a
        ``FromRun`` value the hook did not supply) fails the request, as it
        does on ``/execute``, instead of a 200 with an empty stream.
        """
        events, first = await _started(req, request)

        async def event_stream() -> Any:
            yield sse_frame(first)
            async for event in events:
                yield sse_frame(event)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    if entity_resolver is not None:

        @router.get("/entities/{kind}")
        def list_entities(kind: str, request: Request) -> list[dict[str, Any]]:
            """Return candidate entities of ``kind`` for the current request.

            Backs the ``EntityDropdown`` widget in conductor-aware frontends.
            The exact shape of each entry is host-defined, but the frontend
            convention is ``{"id": "...", "label": "..."}``.
            """
            return entity_resolver(kind, request)

    @router.post("/compile", response_model=list[Problem])
    def compile_only(req: ExecuteRequest) -> list[Problem]:
        """Compile a graph without executing it. Returns every problem the
        graph has, fatal or not, each anchored on a node; an empty list
        means the graph runs.

        Debounce-friendly (~10-30 ms): hosts can poll this on every graph
        edit to paint type mismatches and cycles in real time.
        """
        return list(CompiledGraph.from_graph(req.graph, registry).problems)

    return router
