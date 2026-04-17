"""Example 2: Building and executing a flow.

Shows how to create a graph from nodes and edges, compile it,
and run it both synchronously and with streaming events.
"""

import asyncio
from typing import Annotated

from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor.execution.engine import collect, execute, execute_sync
from conductor.widgets import Output, Text, Textarea

# ---------------------------------------------------------------------------
# Register some nodes
# ---------------------------------------------------------------------------

registry = NodeRegistry()


@registry.node("echo", version=1, name="Echo", description="Returns input")
def echo(text: Annotated[str, Text(label="Input")]) -> Annotated[str, Output(label="Output")]:
    return text


@registry.node("upper", version=1, name="Uppercase", description="Uppercases")
def upper(text: Annotated[str, Text(label="Input")]) -> Annotated[str, Output(label="Output")]:
    return text.upper()


@registry.node("join", version=1, name="Join", description="Joins two strings")
def join(
    a: Annotated[str, Text(label="A")],
    b: Annotated[str, Text(label="B")],
    separator: Annotated[str, Text(label="Separator")] = " + ",
) -> Annotated[str, Output(label="Result")]:
    return f"{a}{separator}{b}"


# ---------------------------------------------------------------------------
# Build a flow: echo -> upper -> join (with a second echo)
#
#   echo("hello") ──> upper ──> join ──> result
#   echo("world") ────────────/
# ---------------------------------------------------------------------------

nodes = [
    GraphNode("n1", "echo@1", {"text": "hello"}),
    GraphNode("n2", "echo@1", {"text": "world"}),
    GraphNode("n3", "upper@1", None),  # No static data — input comes from edge
    GraphNode("n4", "join@1", {"separator": " & "}),
]

edges = [
    GraphEdge("e1", "n1", "n3", "result", "text"),   # echo("hello") -> upper
    GraphEdge("e2", "n3", "n4", "result", "a"),       # upper -> join.a
    GraphEdge("e3", "n2", "n4", "result", "b"),       # echo("world") -> join.b
]

# ---------------------------------------------------------------------------
# Compile (validates structure, topological sorts)
# ---------------------------------------------------------------------------

compiled = compile(nodes=nodes, edges=edges, registry=registry)

print("Execution order:", compiled.execution_order)
print()

# ---------------------------------------------------------------------------
# Option A: Synchronous execution (blocking)
# ---------------------------------------------------------------------------

print("=== Sync execution ===")
results = execute_sync(compiled)
for node_id, result in results.items():
    print(f"  {node_id}: {result}")
print()

# ---------------------------------------------------------------------------
# Option B: Async streaming (see events as they happen)
# ---------------------------------------------------------------------------


async def run_streaming():
    print("=== Streaming execution ===")
    async for event in execute(compiled):
        match event["type"]:
            case "node_start":
                print(f"  [start]    {event['node_id']}")
            case "node_complete":
                print(f"  [complete] {event['node_id']}: {event['result']}")
            case "flow_complete":
                print(f"  [done]     Final results: {event['results']}")
            case _:
                print(f"  [{event['type']}]")


if __name__ == "__main__":
    asyncio.run(run_streaming())
