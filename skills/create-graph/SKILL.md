---
name: create-graph
description: Places conductor nodes in a Graph, compiles it with CompiledGraph.from_graph and runs it with execute. Use when building, saving, compiling or running a conductor graph, binding inputs with From or Static, reading compile problems, streaming events, running a node once per row, answering a node that returned Asks (graph_pending, cells, cache), passing from_run values, serving graphs over HTTP, or debugging a run, or on "build a graph", "run a graph", "connect these nodes".
---

# Creating and running a conductor graph

## Overview

A graph is its nodes: each placed node pins a node `type` and `version` and binds its inputs. Compile turns the graph into an immutable record that says everything about it, including what is wrong; `execute` runs it as a stream of events. For declaring a node, use the **add-node** skill.

## First, read the installed reference

```bash
python -m conductor.about sections     # the section slugs
python -m conductor.about bindings     # one section, by prefix: bindings, compilation, rows, legs, events, errors
```

The installed library wins over this file.

## Core pattern

```python
from conductor import CompiledGraph, From, Graph, GraphNode, Ref, run_sync, Static

graph = Graph(nodes=[
    GraphNode(id="words", type="text-split", version=1, bindings={"text": Static("red,green")}),
    GraphNode(id="loud", type="text-uppercase", version=1, bindings={"text": From(Ref("words", "result"))}),
    GraphNode(id="joined", type="text-join", version=1, bindings={"parts": From(Ref("loud", "result")), "separator": Static(" + ")}),
])

compiled = CompiledGraph.from_graph(graph, registry)
if not compiled.is_runnable:
    raise ValueError([(p.code, p.node_id, p.message) for p in compiled.problems])

results = run_sync(compiled)["results"]          # {node_id: {output_name: value}}
results["joined"]["result"]               # "RED + GREEN"; loud ran once per word
```

## How an input gets its value

Each input holds at most one binding:

- **`From(Ref("node", "output"), ...)`**: over edges. A node with one output names it `result`; a record's outputs are its field names.
- **`Static(...)`**: typed in by the author. A list typed into an input declared for one value runs the node once per value.
- **No binding**: the declared default.

There is no edge list. A `GraphNode` is keyword-only, and a node id may not contain `.`.

## Quick reference

| You want… | Do | Details |
|---|---|---|
| A node's input and output names | `[i.name for i in registry["text-split"].describe().versions[1].inputs]` | add-node → Where to register |
| To know what is wrong | `compiled.problems`, each with `code`, `node_id`, `field`, `fatal` | REFERENCE.md → Compile |
| One node or field after compile | `compiled.node(node_id)`, `compiled.field(Ref(node_id, name))` | REFERENCE.md → Compile |
| Events as they happen | `async for event in execute(compiled)` | REFERENCE.md → Events |
| One call over all rows | declare the input `Series[X]` | REFERENCE.md → Rows |
| To answer a node that asked | `execute(compiled, record=ending["record"], cache={node_id: answers})` | REFERENCE.md → Legs |
| A service or the caller inside a node | `execute(compiled, from_run={Caller: caller})` | REFERENCE.md → Legs |
| A bound on the leg | `execute(compiled, timeout=60, cancel=asyncio.Event())` | REFERENCE.md → Events |
| To save the graph | `graph.to_path("g.yaml")`, `Graph.from_path("g.yaml")` | REFERENCE.md → Saving |
| A definition the registry lacks | `CompiledGraph.from_graph(graph, registry.extended_with({id: cls}))` | REFERENCE.md → Compile |
| HTTP endpoints or a ReactFlow canvas | `conductor_providers.fastapi` / `conductor_providers.react` | REFERENCE.md → Providers |

In a notebook the kernel owns an event loop: `await run(compiled)`, not `run_sync` — the same ending event either way.

## Common mistakes

| Mistake | Fix |
|---|---|
| `GraphNode("a", "echo", 1, {...})` | keywords: `GraphNode(id=, type=, version=, bindings=)` |
| `compile(nodes=..., edges=...)`, `GraphEdge` | `CompiledGraph.from_graph(Graph(nodes=[...]), registry)`; an edge is an `From` binding |
| Wrapping `from_graph` in `try` to catch a bad graph | it does not raise for one: read `is_runnable` and `problems` |
| A loop node, or a `for` around `execute` per item | bind a series; the engine runs the node once per row |
| `run_sync(compiled, retry=...)` | retries belong to the node version's `Policy` |
| `except GraphPendingError` around `run_sync` | nothing raises for a pause: read `ending["type"]`, and answer with `record=ending["record"]` |
| Answering a pending leg with `execute` and no `record` | pass `record=ending["record"]`, or everything runs again |
| An answer for some rows given as a list | a `Series` on `compiled.node(node_id).iterates_on`, with `rows=` |
| Listening for `flow_complete` | endings are `graph_complete`, `graph_pending`, `graph_error`, `graph_cancelled`, `graph_timeout` |

## Debugging a run

1. `compiled.problems` first: a graph that cannot run says why before anything runs.
2. Stream `execute` and print every event: which nodes start, which rows progress, which are skipped.
3. A failure's `cause` has a `code` and, for a node on rows, the `row`: `node_error` / `graph_error` carry it.
4. For a value that is not what you expected, ask `compiled.field(Ref(node_id, input))` for its `binding`, `type` and `index`.
