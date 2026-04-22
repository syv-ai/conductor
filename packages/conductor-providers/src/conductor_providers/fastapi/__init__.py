"""FastAPI adapter for conductor.

Mount conductor's core endpoints (``/nodes``, ``/execute``, ``/execute-stream``,
``/compile``) as a FastAPI ``APIRouter`` without hand-rolling the pydantic
payloads, SSE framing, or compile-result plumbing in every host.

Usage::

    from fastapi import FastAPI, Depends
    from conductor import NodeRegistry
    from conductor.compound.for_each import FOR_EACH
    from conductor_providers.fastapi import conductor_router

    registry = NodeRegistry()
    # ... populate registry ...

    app = FastAPI()
    app.include_router(
        conductor_router(
            registry,
            prefix="/api/v1/flows",
            compound_types=[FOR_EACH],
            dependencies=[Depends(require_admin)],
        )
    )

Optional per-request context injection into the node ``FlowStore``::

    def my_context(request: Request) -> dict[str, Any]:
        return {"user": request.state.user, "session": request.state.session}

    conductor_router(registry, context_factory=my_context)

Requires ``fastapi`` installed (declared as an optional extra:
``pip install conductor-providers[fastapi]``).
"""

from conductor_providers.fastapi.compile import CompileResult, CompileWarning
from conductor_providers.fastapi.models import EdgeInput, ExecuteRequest, NodeInput
from conductor_providers.fastapi.router import conductor_router
from conductor_providers.fastapi.sse import sse_frame

__all__ = [
    "conductor_router",
    "ExecuteRequest",
    "NodeInput",
    "EdgeInput",
    "CompileResult",
    "CompileWarning",
    "sse_frame",
]
