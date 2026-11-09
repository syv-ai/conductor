"""FastAPI ``APIRouter`` factory for conductor."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from conductor import (
    CompiledGraph,
    InvalidCallError,
    NodeDescription,
    NodeRegistry,
    Problem,
    RunState,
)
from conductor.events import Event, RunEnd
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from conductor_providers.fastapi.models import RunRequest
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
    - ``POST {prefix}/run``             — one leg; returns ``state`` and every
      node's ``results``, however the leg ended
    - ``POST {prefix}/run-stream``      — SSE stream of event frames

    A run that asks goes on in legs: send its ``state`` back with the
    ``answers`` (``RunRequest``).
    - ``POST {prefix}/compile``         — compile without executing; returns
      every ``Problem`` the graph has

    Args:
        registry: The populated ``NodeRegistry`` to serve.
        prefix: Path prefix applied to every route (FastAPI convention).
        tags: OpenAPI tags attached to every route.
        dependencies: FastAPI dependencies applied to every route (auth, rate
            limiting, anything ``Depends(...)`` can express).
        from_run: Optional hook invoked per request on ``/run`` and
            ``/run-stream``. Receives the FastAPI ``Request`` and returns
            the values the run supplies by type, ``run(from_run=...)``: a
            ``run`` parameter annotated ``Annotated[X, FromRun()]`` receives
            the value keyed by ``X``.
        entity_resolver: Optional hook backing the ``EntityDropdown`` widget.
            Receives the entity kind (e.g. ``"document"``) and the FastAPI
            ``Request``; returns a list of ``{"id": ..., "label": ...}``
            dicts the frontend renders as choices. ``GET {prefix}/entities/{kind}``
            is mounted only when one is given.

    A request the run cannot start from is the caller's fault and a 422,
    with ``detail`` the ``InvalidCallError``'s ``errors()``: a graph with a
    fatal problem, a state that does not read, a wrong answer. Anything else
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
        if not req.answers:
            return compiled, req.state
        return compiled, compiled.answer(req.state or RunState(status="waiting"), req.answers)

    @router.get("/nodes", response_model=list[NodeDescription])
    def list_nodes() -> list[NodeDescription]:
        """Every registered definition as a record — the palette."""
        return [cls.describe() for cls in registry.nodes]

    @router.post("/run")
    async def run_graph(req: RunRequest, request: Request) -> dict[str, Any]:
        """Run one leg, and return its ``state`` and every node's ``results``."""
        try:
            compiled, state = _prepared(req)
            state = await compiled.run(state, from_run=from_run(request) if from_run else None)
        except InvalidCallError as refused:
            raise _unprocessable(refused) from refused
        return {"state": as_data(state), "results": as_data(compiled.results(state))}

    @router.post("/run-stream")
    async def run_graph_stream(req: RunRequest, request: Request) -> StreamingResponse:
        """Run one leg and stream its events as Server-Sent Events.

        Every check is made before the response starts, so a refused run fails
        the request as it does on ``/run``, instead of a 200 with an empty stream.
        """
        try:
            compiled, state = _prepared(req)
            leg = compiled.stream(state, from_run=from_run(request) if from_run else None)
        except InvalidCallError as refused:
            raise _unprocessable(refused) from refused

        async def event_stream() -> Any:
            async with leg as events:
                async for event in events:
                    yield sse_frame(_answered(compiled, event))

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
    def compile_only(req: RunRequest) -> list[Problem]:
        """Compile a graph without executing it. Returns every problem the
        graph has, fatal or not, each anchored on a node; an empty list
        means the graph runs.

        Debounce-friendly (~10-30 ms): hosts can poll this on every graph
        edit to paint type mismatches and cycles in real time.
        """
        return list(CompiledGraph.from_graph(req.graph, registry).problems)

    return router



def _unprocessable(refused: InvalidCallError) -> HTTPException:
    return HTTPException(status_code=422, detail=refused.errors())


def _answered(compiled: CompiledGraph, event: Event) -> dict[str, Any]:
    """An event as this provider's JSON: a ``RunEnd`` also carries ``results``,
    every node's values read from its state, since a client over HTTP has
    no compiled graph to ask."""
    data = as_data(event)
    if isinstance(event, RunEnd):
        data["results"] = as_data(compiled.results(event.state))
    return data
