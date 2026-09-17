"""FastAPI ``APIRouter`` factory for conductor."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from conductor import NodeRegistry
from conductor.execution.engine import execute, execute_sync
from conductor.graph.compiled import CompiledGraph
from conductor.graph.problem import Problem
from conductor.node import NodeDescription
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
    - ``POST {prefix}/execute``         — sync execution, returns aggregated results
    - ``POST {prefix}/execute-stream``  — SSE stream of ``ExecutionEvent`` frames
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

    @router.get("/nodes", response_model=list[NodeDescription])
    def list_nodes() -> list[NodeDescription]:
        """Every registered definition as a record — the palette."""
        return [cls.describe() for cls in registry.definitions()]

    @router.post("/execute")
    def execute_graph(req: ExecuteRequest, request: Request) -> dict[str, Any]:
        """Run a graph synchronously and return the aggregated results dict."""
        compiled = CompiledGraph.from_graph(req.graph, registry)
        results = execute_sync(
            compiled, from_run=_from_run(request), cache=req.cache or None
        )
        return {"results": _jsonable(results)}

    @router.post("/execute-stream")
    async def execute_graph_stream(
        req: ExecuteRequest, request: Request
    ) -> StreamingResponse:
        """Run a graph and stream ``ExecutionEvent``s as Server-Sent Events."""
        compiled = CompiledGraph.from_graph(req.graph, registry)
        supplied = _from_run(request)

        async def event_stream() -> Any:
            async for event in execute(
                compiled, from_run=supplied, cache=req.cache or None
            ):
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


# Silence "imported but unused" warnings: Depends is a documented option for
# callers to import alongside `conductor_router`, not used in this module.
_ = Depends
