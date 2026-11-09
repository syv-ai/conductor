---
name: create-graph
description: Puts conductor nodes in a Graph, compiles it with CompiledGraph.from_graph and runs it with compiled.run. Use when building, saving, running, drawing, serving or debugging a conductor graph; binding inputs (From, Static, with_inputs); reading outputs, problems or events; answering a node that asked a person (compiled.answer); or on "build a graph", "run a graph", "connect these nodes".
---

# Creating and running a conductor graph

A graph is a list of nodes, each pinning a `type` and `version` and binding its inputs. Compile says what is wrong; `compiled.run` runs it as a stream of events. To declare a node, use the **add-node** skill. The installed library is the authority: `python -m conductor.about bindings` (or `compilation`, `rows`, `legs`, `events`, `errors`).

```python
import conductor_nodes
from conductor import CompiledGraph, From, Graph, GraphNode, Static

registry = conductor_nodes.registry(categories=["text"])
graph = Graph(nodes=[
    GraphNode(id="words", type="text-split", version=1),
    GraphNode(id="loud", type="text-uppercase", version=1, bindings={"text": From("words.result")}),
    GraphNode(id="joined", type="text-join", version=1, bindings={"parts": From("loud.result"), "separator": Static(" + ")}),
])

compiled = CompiledGraph.from_graph(graph, registry)
ready = compiled.with_inputs(text="red,green")        # words.text is the graph's open input
if not ready.is_runnable:
    raise ValueError([(p.type, p.loc, p.msg) for p in ready.problems])

state = ready.run_sync()                               # in a notebook: await ready.run()
assert ready.outputs(state) == {"joined.result": "RED + GREEN"}
results = ready.results(state)                         # {node_id: {output: value}}; many values are {"rows": [...], "values": [...]}
assert results["loud"]["result"]["values"] == ["RED", "GREEN"]   # ran once per word
assert ready.run_plan.nodes["joined"].runs_per_row_of is None   # parts is Series[str]: one call
```

A node you declared yourself is placed with `upper.graph_node("b", text=From("a.result"), times=2)`, the same record with the function's own signature for a checker to read. A value is typed in, `From(...)` is an edge, and an input left open is written `GraphNode(...)`.

An input holds one binding: `From("node.output")` over an edge, `Static(value)` typed in (a list on a single-value input runs the node once per item), or none for the default. A single output is named `result`. `GraphNode` is keyword-only; a node id may not contain `.`, `/` or `@`.

## Quick reference

| You want | Do (examples in REFERENCE.md) |
|---|---|
| Two open inputs with one name | `with_inputs(**{"words.text": value})` |
| Events as they happen | `async with compiled.stream() as events: async for event in events: ...` |
| To answer a node that asked | `state = compiled.answer(state, {unit.id: {output: answer}})`, then `compiled.run(state)` |
| A service or the caller inside a node | `compiled.run(from_run={Caller: caller})` |
| A bound on the leg | `compiled.run(timeout=60, cancel=asyncio.Event())` |
| To save | `graph.to_path("g.yaml")`, `Graph.from_path("g.yaml")` |
| A drawing, HTTP routes, a record an editor draws from | `conductor_providers.mermaid`, `.fastapi`, `compiled.describe()` |

## Common mistakes

| Mistake | Fix |
|---|---|
| `try` around `from_graph` | it never raises for a bad graph: read `is_runnable` and `problems` |
| Rebuilding the `Graph` to change a value per run | `compiled.with_inputs(...)`; store `ready.graph` as what ran |
| A loop node, or a `for` around `run` | bind a column; the engine runs the node once per row |
| A join node to put columns side by side | wire them all into one `Table` input; they merge on `__row` |
| `except` around `run_sync` for a pause, error or timeout | they are statuses: read `state.status`, `state.failure` |
| `run_sync` in a notebook or server | `await compiled.run()` |

To debug, read `problems`, then print every event of `compiled.stream()`. A failure is a `Problem`: `type`, `msg`, `ctx`, and `loc` is the node and the row. `print(compiled.run_plan)` says where each node runs and what each input receives.
