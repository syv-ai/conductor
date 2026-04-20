"""Demo FastAPI app for the flow engine playground.

Serves the node registry and a streaming executor. The frontend is a
separate Next.js app at `demo/web/`; run both independently:

    uv run uvicorn demo.app:app --port 8765 --reload
    cd demo/web && npm run dev        # Next.js on localhost:3000
"""

import json
from typing import Any

from conductor.compound.for_each import FOR_EACH
from conductor.execution.engine import execute, execute_sync
from conductor.graph.compiler import compile
from conductor.graph.model import GraphEdge, GraphNode
from conductor.registry.schema import serialize_registry
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from demo.nodes import registry

app = FastAPI(title="Conductor Playground")

# The Next.js dev server runs on 3000; allow any origin during local dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class NodeInput(BaseModel):
    id: str
    type: str
    data: dict[str, Any] | None = None
    produces: dict[str, str] | None = None
    consumes: dict[str, tuple[str, str]] | None = None


class EdgeInput(BaseModel):
    id: str
    source: str
    target: str
    source_handle: str | None = None
    target_handle: str | None = None


class ExecuteRequest(BaseModel):
    nodes: list[NodeInput]
    edges: list[EdgeInput]


def _build_graph(req: ExecuteRequest) -> tuple[list[GraphNode], list[GraphEdge]]:
    nodes = [
        GraphNode(
            id=n.id,
            type=n.type,
            data=n.data,
            produces=n.produces or None,
            consumes=(
                {k: (v[0], v[1]) for k, v in n.consumes.items()}
                if n.consumes
                else None
            ),
        )
        for n in req.nodes
    ]
    edges = [
        GraphEdge(
            id=e.id,
            source=e.source,
            target=e.target,
            source_handle=e.source_handle,
            target_handle=e.target_handle,
        )
        for e in req.edges
    ]
    return nodes, edges


@app.get("/api/nodes")
def get_nodes() -> list[dict[str, Any]]:
    """Return all registered nodes as JSON for the frontend."""
    return serialize_registry(registry)


@app.post("/api/execute")
def execute_flow(req: ExecuteRequest) -> dict[str, Any]:
    """Execute a flow synchronously and return results."""
    nodes, edges = _build_graph(req)
    compiled = compile(nodes=nodes, edges=edges, registry=registry, compound_types=[FOR_EACH])
    return {"results": execute_sync(compiled)}


@app.post("/api/execute-stream")
async def execute_flow_stream(req: ExecuteRequest) -> StreamingResponse:
    """Execute a flow and stream events as Server-Sent Events."""
    nodes, edges = _build_graph(req)
    compiled = compile(nodes=nodes, edges=edges, registry=registry, compound_types=[FOR_EACH])

    async def event_stream():
        async for event in execute(compiled):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
