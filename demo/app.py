"""Demo FastAPI app for the flow engine playground."""

import json
from typing import Any

from conductor.compound.for_each import FOR_EACH
from conductor.execution.engine import execute, execute_sync
from conductor.graph.compiler import compile
from conductor.graph.model import GraphEdge, GraphNode
from conductor.registry.schema import serialize_registry
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from demo.nodes import registry

app = FastAPI(title="Conductor Playground")
app.mount("/static", StaticFiles(directory="demo/static"), name="static")


class NodeInput(BaseModel):
    id: str
    type: str
    data: dict[str, Any] | None = None


class EdgeInput(BaseModel):
    id: str
    source: str
    target: str
    source_handle: str | None = None
    target_handle: str | None = None


class ExecuteRequest(BaseModel):
    nodes: list[NodeInput]
    edges: list[EdgeInput]


@app.get("/")
async def index():
    with open("demo/static/index.html") as f:
        return HTMLResponse(f.read())


@app.get("/api/nodes")
def get_nodes():
    """Return all registered nodes as JSON for the frontend."""
    return serialize_registry(registry)


@app.post("/api/execute")
def execute_flow(req: ExecuteRequest):
    """Execute a flow synchronously and return results."""
    nodes = [GraphNode(id=n.id, type=n.type, data=n.data) for n in req.nodes]
    edges = [
        GraphEdge(id=e.id, source=e.source, target=e.target,
                  source_handle=e.source_handle, target_handle=e.target_handle)
        for e in req.edges
    ]

    compiled = compile(
        nodes=nodes,
        edges=edges,
        registry=registry,
        compound_types=[FOR_EACH],
    )

    results = execute_sync(compiled)
    return {"results": results}


@app.post("/api/execute-stream")
async def execute_flow_stream(req: ExecuteRequest):
    """Execute a flow with SSE streaming events."""
    nodes = [GraphNode(id=n.id, type=n.type, data=n.data) for n in req.nodes]
    edges = [
        GraphEdge(id=e.id, source=e.source, target=e.target,
                  source_handle=e.source_handle, target_handle=e.target_handle)
        for e in req.edges
    ]

    compiled = compile(
        nodes=nodes,
        edges=edges,
        registry=registry,
        compound_types=[FOR_EACH],
    )

    async def event_stream():
        async for event in execute(compiled):
            yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
