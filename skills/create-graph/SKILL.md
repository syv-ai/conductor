---
name: create-graph
description: Places conductor nodes in a Graph, compiles it with CompiledGraph.from_graph and runs it with execute. Use when building, saving, running, drawing, serving or debugging a conductor graph; binding inputs (From, Static, with_inputs); reading outputs, problems or events; answering Asks (state, cache); or on "build a graph", "run a graph", "connect these nodes".
---

# Creating and running a conductor graph

A graph is placed nodes, each pinning a `type` and `version` and binding its inputs. Compile says what is wrong; `execute` runs it as a stream of events. To declare a node, use the **add-node** skill. The installed library is the authority: `python -m conductor.about bindings` (or `compilation`, `rows`, `legs`, `events`, `errors`).

```python
import conductor_nodes
from conductor import CompiledGraph, From, Graph, GraphNode, run_sync, Static

registry = conductor_nodes.registry(categories=["text"])
graph = Graph(nodes=[
    GraphNode(id="words", type="text-split", version=1),
    GraphNode(id="loud", type="text-uppercase", version=1, bindings={"text": From("words.result")}),
    GraphNode(id="joined", type="text-join", version=1, bindings={"parts": From("loud.result"), "separator": Static(" + ")}),
])

compiled = CompiledGraph.from_graph(graph, registry)
ready = compiled.with_inputs(text="red,green")        # words.text is the graph's open input
if not ready.is_runnable:
    raise ValueError([(p.code, p.node_id, p.message) for p in ready.problems])

ending = run_sync(ready)                               # in a notebook: await run(ready)
assert ending.state.outputs(ready) == {"joined.result": "RED + GREEN"}
results = ending.state.results(ready)                  # {node_id: {output: value}}
assert list(results["loud"]["result"]) == ["RED", "GREEN"]   # ran once per word
assert ready.node("joined").iterates_on is None       # parts is Series[Text]: one call
```

An input holds one binding: `From("node.output")` over an edge, `Static(value)` typed in (a list on a single-value input runs the node once per item), or none for the default. A single output is named `result`. `GraphNode` is keyword-only; a node id may not contain `.`.

## Quick reference

| You want | Do (examples in REFERENCE.md) |
|---|---|
| Two open inputs with one name | `with_inputs(**{"words.text": value})` |
| Events as they happen | `async for event in execute(compiled)` |
| To answer a node that asked | `execute(compiled, state=ending.state, cache={node_id: {output: answer}})` |
| A service or the caller inside a node | `execute(compiled, from_run={Caller: caller})` |
| A bound on the leg | `execute(compiled, timeout=60, cancel=asyncio.Event())` |
| To save | `graph.to_path("g.yaml")`, `Graph.from_path("g.yaml")` |
| A drawing, HTTP routes, a ReactFlow canvas | `conductor_providers.mermaid`, `.fastapi`, `.react` |

## Common mistakes

| Mistake | Fix |
|---|---|
| `try` around `from_graph` | it never raises for a bad graph: read `is_runnable` and `problems` |
| Rebuilding the `Graph` to change a value per run | `compiled.with_inputs(...)`; store `ready.graph` as what ran |
| A loop node, or a `for` around `execute` | bind a series; the engine runs the node once per row |
| `except` around `run_sync` for a pause, error or timeout | they are endings: read `ending.type` |
| `run_sync` in a notebook or server | `await run(compiled)` |

To debug, read `problems`, then print every event of `execute`. A failure's `cause` has a `code` and, on rows, the `row`.
