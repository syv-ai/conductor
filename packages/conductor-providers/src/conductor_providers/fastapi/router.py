"""FastAPI ``APIRouter`` factory for conductor."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from conductor import NodeRegistry
from conductor.errors import CompilationError
from conductor.execution.engine import execute, execute_sync
from conductor.graph.compiler import compile as compile_graph
from conductor.registry.schema import serialize_registry

from conductor_providers.fastapi.compile import CompileResult, CompileWarning
from conductor_providers.fastapi.models import ExecuteRequest
from conductor_providers.fastapi.sse import sse_frame


def conductor_router(
    registry: NodeRegistry,
    *,
    prefix: str = "",
    tags: list[str] | None = None,
    dependencies: Sequence[Any] | None = None,
    compound_types: list[type] | None = None,
    context_factory: Callable[[Request], dict[str, Any]] | None = None,
    strict_types: bool = False,
) -> APIRouter:
    """Build a FastAPI ``APIRouter`` serving conductor's standard endpoints.

    Mounts:

    - ``GET  {prefix}/nodes``           — serialized node catalog
    - ``POST {prefix}/execute``         — sync execution, returns aggregated results
    - ``POST {prefix}/execute-stream``  — SSE stream of ``ExecutionEvent`` frames
    - ``POST {prefix}/compile``         — validation without executing; returns
      ``CompileResult`` with errors (hard) and warnings (soft type mismatches)

    Args:
        registry: The populated ``NodeRegistry`` to serve.
        prefix: Path prefix applied to every route (FastAPI convention).
        tags: OpenAPI tags attached to every route.
        dependencies: FastAPI dependencies applied to every route (auth, rate
            limiting, anything ``Depends(...)`` can express).
        compound_types: Compound types passed to ``compile()`` (e.g.
            ``[FOR_EACH]``).
        context_factory: Optional hook invoked per-request on ``/execute`` and
            ``/execute-stream``. Receives the FastAPI ``Request`` and returns
            a dict that seeds the node ``FlowStore``. Node functions declaring
            ``store: FlowStore`` see the seeded keys.
        strict_types: Passed through to ``compile()``. When True, type
            warnings become ``CompilationError``s on ``/execute`` and
            ``/execute-stream`` (``/compile`` always returns warnings as soft).
    """
    router = APIRouter(
        prefix=prefix,
        tags=tags or ["conductor"],
        dependencies=list(dependencies) if dependencies else None,
    )
    compound_types = compound_types or []

    def _store_data(request: Request) -> dict[str, Any] | None:
        return context_factory(request) if context_factory else None

    @router.get("/nodes")
    def list_nodes() -> list[dict[str, Any]]:
        """Return every registered node as serialized catalog entries."""
        return serialize_registry(registry)

    @router.post("/execute")
    def execute_flow(req: ExecuteRequest, request: Request) -> dict[str, Any]:
        """Run a flow synchronously and return the aggregated results dict."""
        nodes, edges = req.to_graph()
        compiled = compile_graph(
            nodes=nodes,
            edges=edges,
            registry=registry,
            compound_types=compound_types,
            strict_types=strict_types,
        )
        results = execute_sync(compiled, store_data=_store_data(request))
        return {"results": results}

    @router.post("/execute-stream")
    async def execute_flow_stream(
        req: ExecuteRequest, request: Request
    ) -> StreamingResponse:
        """Run a flow and stream ``ExecutionEvent``s as Server-Sent Events."""
        nodes, edges = req.to_graph()
        compiled = compile_graph(
            nodes=nodes,
            edges=edges,
            registry=registry,
            compound_types=compound_types,
            strict_types=strict_types,
        )
        store_data = _store_data(request)

        async def event_stream() -> Any:
            async for event in execute(compiled, store_data=store_data):
                yield sse_frame(event)

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    @router.post("/compile")
    def compile_flow(req: ExecuteRequest) -> CompileResult:
        """Validate a graph without executing. Returns errors + type warnings.

        Debounce-friendly (~10-30 ms): hosts can poll this on every graph
        edit to paint type mismatches and cycles in real time.
        """
        nodes, edges = req.to_graph()
        try:
            compiled = compile_graph(
                nodes=nodes,
                edges=edges,
                registry=registry,
                compound_types=compound_types,
                strict_types=False,  # /compile always surfaces warnings as soft
            )
        except CompilationError as e:
            return CompileResult(status="error", errors=[str(e)], warnings=[])

        warnings = [
            CompileWarning(
                edge_id=w.edge_id,
                code=w.code,
                message=w.message,
                source_node=w.source_node,
                source_output=w.source_output,
                source_type=w.source_type,
                target_node=w.target_node,
                target_input=w.target_input,
                target_type=w.target_type,
            )
            for w in compiled.type_warnings
        ]
        return CompileResult(status="ok", errors=[], warnings=warnings)

    return router


# Silence "imported but unused" warnings: Depends is a documented option for
# callers to import alongside `conductor_router`, not used in this module.
_ = Depends
