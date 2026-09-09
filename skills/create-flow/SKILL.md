---
name: create-flow
description: Use when building or running a conductor flow — placing nodes with GraphNode, connecting them through Edges bindings, calling CompiledGraph.from_graph()/execute(), streaming events, or debugging a run. Triggers on phrases like "create a flow", "build a graph", "run a flow", "connect these nodes together", "stream execution events".
---

# Creating and running a conductor flow

Use this skill when the user wants to build or execute a flow in a project that depends on the [`syv-conductor`](https://pypi.org/project/syv-conductor/) library. For defining new node types, use the `add-node` skill.

## First — pull the packaged library reference

Conductor ships its reference text in the wheel. **Run these before writing code** so your advice matches the installed version:

```bash
python -m conductor.about sections         # list section slugs
python -m conductor.about                  # the whole reference
```

Programmatic: `from conductor.about import get_content, list_sections, get_section`.

## Three phases: declare → compile → execute

```python
from conductor import Graph, GraphNode, NodeRegistry, Ref, Edges, Static, CompiledGraph
from conductor.execution.engine import execute_sync

# 1. declare — node classes registered at import (see add-node)
registry = NodeRegistry()
registry.register(Greet)
registry.register(Shout)

# 2. compile — a placement pins a node by type and version, and binds each input
flow = Graph(nodes=[
    GraphNode(id="a", type="greet", version=1, bindings={"name": Static(value="Ada")}),
    GraphNode(id="b", type="shout", version=1, bindings={"text": Edges(refs=(Ref("a", "result"),))}),
])
compiled = CompiledGraph.from_graph(flow, registry)

# 3. execute
results = execute_sync(compiled)
# results["b"]["result"] == "HELLO ADA"
```

A single-output node's output is named `result`; a multi-output node's outputs are the field names of the record it returns. A `Ref("node", "field")` names an output by those names, and the bindings key is the input's name.

## How a node receives a value

One input holds at most one binding:

1. **`Edges(refs=(Ref(...), ...))`** — the value comes from other placements' outputs, in operand order. Several refs into a `Series[X]` input gather into one series; a scalar input takes exactly one.
2. **`Static(value=...)`** — the author typed the value in.
3. **No binding** — the parameter's default.

There is no edge list and no per-edge record: `dependencies_of(flow.nodes)` derives what each node waits for, and a canvas derives its edges. The call is validated through pydantic against the placed node's own interface, and `run` receives instances of the declared dtypes.

## Streaming execution

```python
from conductor.execution.engine import execute

async for event in execute(compiled):
    match event["type"]:
        case "node_start":
            print("start", event["node_id"])
        case "node_complete":
            print("done", event["node_id"], event["result"])
        case "node_error":
            print("fail", event["node_id"], event["error"])
        case "flow_complete":
            return event["results"]
```

Event types: `node_start`, `node_complete`, `node_skipped`, `node_error`, `node_retry`, `runtime_warning`, `flow_complete`, `flow_error`, `flow_timeout`, `flow_cancelled`. A failed node fails the run.

Execution is eager and parallel by default. Independent branches overlap automatically — no flag needed. In a notebook use `await collect(execute(compiled))`; from a script, `execute_sync`.

## Retries

Retries belong to the version's `Policy` (see `add-node`). A run-level fallback covers nodes whose policy sets none:

```python
from conductor.execution.retry import RetryConfig

results = execute_sync(compiled, retry=RetryConfig(max_retries=2, delay=1.0, backoff_factor=2.0))
```

`NodeValidationError` is never retried; `NodeConnectionError` and `NodeExecutionError` are.

## Branches

A node returns `SKIPPED` on the branch it did not take. Whatever is connected to that output is skipped in turn and emits `node_skipped`. The standard library ships `logic-if-empty`, `logic-if-equals` and a `decision` gate — usually you don't write your own.

## Definitions the registry does not hold

A graph may name a definition the static registry lacks (a host's embedded flows, say). The host builds those `NodeDefinition`s itself and hands compile `registry.extended_with({"loaded-id": Loaded})` — a new registry per run; a registered type wins over a loaded one of the same id.

## Cancellation and timeout

`execute(compiled, timeout_seconds=...)` bounds the whole run and emits `flow_timeout`; a version's `Policy(timeout=...)` bounds one node. For caller-driven cancel, wrap `execute` in an asyncio task and cancel it.

## Frontend integration

If the host project has a React-based builder, use `conductor_providers.react`:

```python
from conductor_providers.react import graph_to_react, react_to_graph, palette_from_registry

palette = palette_from_registry(registry)                    # [cls.describe() ...] for the palette
flow = react_to_graph(flow_json)                             # frontend → conductor (a Graph)
flow_json = graph_to_react(flow)                             # conductor → frontend (record under data, edges derived)
```

`conductor_providers.fastapi.conductor_router(registry)` mounts `/nodes`, `/compile`, `/execute`, `/execute-stream` and `/entities/{kind}`.

## Compiled graph — what it answers

`CompiledGraph` is immutable and is asked, never read through. The questions worth knowing:

- `problems_for()` / `is_runnable` — every `Problem` (code, message, `node_id`, `field`, `fatal`), whole or by anchor; a run refuses on the first fatal one.
- `execution_order()` — the node ids in edge order; an embedded flow's inner nodes appear as `placement/inner`.
- `interface_of(node_id)` — the node's inputs and outputs as its hooks answered, every type bound by the edges.
- `type_of(Ref(node_id, field))` — the type that travels on a field; `index_of(Ref(node_id, field))` — for a series, where its rows come from; `lifted_on(node_id)` — the index a node runs once per row of, or `None`.
- `value_source(node_id, input)` — the `Binding` behind an input, or `None` when the declared default applies.
- `condition(Ref(...))` / `decisions()` — under which upstream decisions an output appears.
- `interface` — what the graph takes and returns, named by address.

## Checklist before running a flow

- [ ] Every `GraphNode.type` is registered on the registry passed to `CompiledGraph.from_graph`, and its `version` exists — otherwise `problems_for()` says so.
- [ ] Every `Ref` in an `Edges` names an existing node and one of its outputs, and the bindings key names an input.
- [ ] `Static` values are the declared types (pydantic coerces builtins into the host's dtypes).
- [ ] If long-running, the caller owns cancellation and/or `timeout_seconds`.

## Debugging a failing flow

1. Stream with `execute` (not `execute_sync`) and log every event — reveals scheduling and skip behavior.
2. For node-level errors, catch `FlowExecutionError` (sync) or check `flow_error` events (async); `error.node_id` and `error.original` pinpoint the failure. A `NodeValidationError` names the field and its title.
3. For resolver confusion, ask `compiled.value_source(node_id, input)` and `compiled.type_of(Ref(node_id, input))` for the problem field.

## When your advice diverges from the installed version

Trust `python -m conductor.about` over this skill. If a behavior reported by the user contradicts what's written here, re-fetch the reference before making changes.
