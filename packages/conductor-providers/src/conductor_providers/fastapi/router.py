"""FastAPI ``APIRouter`` factory for conductor."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from conductor import NodeRegistry
from conductor.errors import GraphPendingError
from conductor.execution.engine import collect, execute
from conductor.execution.events import ExecutionEvent
from conductor.graph.compiled import CompiledGraph
from conductor.graph.problem import Problem
from conductor.node import NodeDescription
from conductor.series import Series
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from conductor_providers.fastapi.models import ExecuteRequest
from conductor_providers.fastapi.sse import _jsonable, sse_frame


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

    A run that asks goes on in legs: send the ending's ``record`` back with
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
            dicts the frontend renders as choices. If unset, the mounted
            ``GET {prefix}/entities/{kind}`` route returns 501.
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
            cache=_cache(compiled, req.cache or {}) or None,
            record=req.record,
        )

    @router.get("/nodes", response_model=list[NodeDescription])
    def list_nodes() -> list[NodeDescription]:
        """Every registered definition as a record — the palette."""
        return [cls.describe() for cls in registry.nodes]

    @router.post("/execute")
    async def execute_graph(req: ExecuteRequest, request: Request) -> dict[str, Any]:
        """Run one leg and return the frame it ended on.

        ``graph_complete`` and ``graph_pending`` are answers — a pending
        frame carries the questions and the record the next request sends
        back. A leg that fails, is cancelled or times out fails the request,
        as ``execute_sync`` raises for it.
        """
        ending: ExecutionEvent | None = None

        async def watched() -> Any:
            nonlocal ending
            async for event in _leg(req, request):
                ending = event
                yield event

        try:
            await collect(watched())
        except GraphPendingError:
            pass
        return _jsonable(ending)

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
        events = _leg(req, request)
        first = await anext(events)

        async def event_stream() -> Any:
            yield sse_frame(first)
            async for event in events:
                yield sse_frame(event)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @router.get("/entities/{kind}")
    def list_entities(kind: str, request: Request) -> list[dict[str, Any]]:
        """Return candidate entities of ``kind`` for the current request.

        Backs the ``EntityDropdown`` widget in conductor-aware frontends.
        Hosts provide the list via the ``entity_resolver`` hook; the
        exact shape of each entry is host-defined, but the frontend
        convention is ``{"id": "...", "label": "..."}``.
        """
        if entity_resolver is None:
            from fastapi import HTTPException

            raise HTTPException(
                status_code=501,
                detail=(
                    "No entity_resolver configured. Pass one to "
                    "conductor_router(entity_resolver=...) to enable "
                    f"/entities/{kind} lookups."
                ),
            )
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


def _cache(compiled: CompiledGraph, cache: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """The request's ``cache`` as ``execute`` takes it.

    A node that runs once is given its outputs as they came. A node that
    runs per row is given a ``Series`` per output on the node's own index,
    built from the ``rows`` and ``values`` the request sent; any ``index``
    it sent beside them is not read, since the compiled graph already says
    which index the node runs on.
    """
    taken: dict[str, dict[str, Any]] = {}
    for node_id, outputs in cache.items():
        index = compiled.node(node_id).iterates_on
        if index is None:
            taken[node_id] = outputs
            continue
        taken[node_id] = {
            name: Series(index, sent["values"], rows=[tuple(row) for row in sent["rows"]])
            for name, sent in outputs.items()
        }
    return taken


# Silence "imported but unused" warnings: Depends is a documented option for
# callers to import alongside `conductor_router`, not used in this module.
_ = Depends
