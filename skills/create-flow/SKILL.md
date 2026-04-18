---
name: create-flow
description: Use when building or running a conductor flow — wiring GraphNodes and edges, calling compile()/execute(), streaming events, handling checkpoints, or debugging type warnings. Triggers on phrases like "create a flow", "build a graph", "run a flow", "wire these nodes together", "stream execution events", "pause/resume a flow".
---

# Creating and running a conductor flow

Use this skill when the user wants to build or execute a flow in a project that depends on the [`conductor`](https://pypi.org/project/conductor/) library. For defining new node types, use the `add-node` skill.

## First — pull the authoritative library reference

Conductor ships its reference text in the wheel. **Run these before writing code** so your advice matches the installed version:

```bash
python -m conductor.about sections         # list section slugs
python -m conductor.about compilation      # compile() semantics, type checking
python -m conductor.about execution        # event stream, scheduling, cancellation
python -m conductor.about data-flow        # resolver precedence: edges → consumes → data → defaults
python -m conductor.about shared-references
python -m conductor.about hitl
python -m conductor.about retry
```

`python -m conductor.about` (no args) prints the whole thing. Programmatic: `from conductor.about import get_content, list_sections, get_section`.

## Three phases: register → compile → execute

```python
from conductor import NodeRegistry, GraphNode, GraphEdge, compile
from conductor.execution.engine import execute_sync

# 1. register (usually done in node modules at import time)
registry = NodeRegistry()
# ... @registry.node(...) decorated functions ...

# 2. compile
compiled = compile(
    nodes=[
        GraphNode(id="a", type="greet@1", data={"name": "Ada"}),
        GraphNode(id="b", type="shout@1", data={}),
    ],
    edges=[
        GraphEdge(id="e1", source="a", sourceHandle="result",
                  target="b", targetHandle="text"),
    ],
    registry=registry,
)

# 3. execute
results = execute_sync(compiled)
# results["b"]["result"] == "HELLO ADA"
```

## The three data-flow channels

Resolver precedence — first match wins:

1. **Edge** targeting `(target_id, targetHandle)`.
2. **Consume** declared on the `GraphNode` (`consumes={"input": ("other_node", "result")}`).
3. **Static data** from `GraphNode.data[input_name]`.
4. **Widget default** from the node's Pydantic model.

Use edges for the primary pipeline. Use consumes for fan-out ("ten nodes all read the same mapping"), cross-region access, or when a wire would be visually noisy. Use static data for constants and test fixtures.

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

Event types: `node_start`, `node_complete`, `node_skipped`, `node_error`, `node_retry`, `node_progress`, `flow_complete`, `flow_error`, `flow_timeout`, `flow_paused`, `flow_cancelled`.

Execution is eager and parallel by default. Independent branches overlap automatically — no flag needed.

## Check `type_warnings` before shipping

`compile` validates every edge and every consume binding. In the default (non-strict) mode mismatches become warnings on `compiled.type_warnings`. Surface them to the user or log them:

```python
for w in compiled.type_warnings:
    print(f"{w.code}: {w.message}")
```

For hard validation at compile time, pass `strict_types=True` — real mismatches raise `CompilationError`.

## Retries

Global fallback at execute time:

```python
from conductor.execution.retry import RetryConfig

results = execute_sync(compiled, retry=RetryConfig(
    max_retries=2, delay=1.0, backoff_factor=2.0,
))
```

Per-node `max_retries` / `retry_delay` set on `@registry.node()` always win over the global config. `NodeValidationError` is never retried; `NodeConnectionError` and `NodeExecutionError` are.

## Conditional branches (if-else)

Nodes can return the `SKIPPED` sentinel from `conductor` to skip downstream branches. A node whose inputs are all `SKIPPED` is automatically skipped and emits `node_skipped`. The stdlib `conductor_nodes.logic` ships an `if-else` node — usually you don't write your own.

## For-each (compound nodes)

Iteration is modeled as a compound node region. Register the compound type at compile:

```python
from conductor.compound.for_each import ForEachNode

compiled = compile(nodes, edges, registry, compound_types=[ForEachNode])
```

`conductor_nodes.loop` provides ready-made `for-each-start@1` / `for-each-end@1` marker nodes. Body nodes between the markers run once per iteration. Sequential and parallel modes are configurable.

Details: `python -m conductor.about compound-nodes`.

## Shared references (produce / consume)

Declared per-`GraphNode`, invisible to edges but part of the dependency graph:

```python
GraphNode(
    id="mapper", type="build-map@1", data={"seed": "x"},
    produces={"result": "pseudonym map"},
),
GraphNode(
    id="redactor", type="redact@1", data={"text": "Alice met Bob."},
    consumes={"mapping": ("mapper", "result")},
),
```

Validated at compile. Consumers inside for-each bodies see the same producer value on every iteration (broadcast, not per-iteration).

Full spec: `python -m conductor.about shared-references`.

## Human-in-the-loop

A node raises `HumanInputRequired(prompt, schema=...)` to pause. The engine emits `flow_paused` with a serializable `checkpoint`. Persist it, collect the response, then resume:

```python
from conductor.execution.engine import resume_sync

results = resume_sync(compiled, checkpoint, response={"approved": True})
```

Contract: `python -m conductor.about hitl`.

## Cancellation and timeout

Conductor cancels running tasks on any node failure. For caller-driven cancel, wrap `execute` in an asyncio task and cancel it. For timeout, wrap in `asyncio.wait_for` — conductor does not own a global timeout clock.

## FlowStore — side-channel state

Auto-inject `store: FlowStore` into any node that needs per-run scratch state not carried by the DAG. Survives checkpoint/resume. Useful for caches, counters, accumulated logs:

```python
from conductor import FlowStore

@registry.node("counter", version=1, name="Counter", description="...")
def counter(
    _store: FlowStore,
    step: Annotated[int, Number()] = 1,
) -> Annotated[int, Output()]:
    current = _store.get("n", 0) + step
    _store.set("n", current)
    return current
```

## Frontend integration

If the host project has a React-based builder, use `conductor_providers.react`:

```python
from conductor_providers.react import graph_to_react, react_to_graph, palette_from_registry

payload = palette_from_registry(registry)                    # for /nodes palette
graph = react_to_graph(nodes_json, edges_json)               # frontend → conductor
react_nodes, react_edges = graph_to_react(graph)             # conductor → frontend
```

Providers for other frameworks (Vue, Svelte) can live in sibling subpackages — no ABC to satisfy.

## Compiled graph — what's inside

`CompiledGraph` is immutable. Fields worth knowing:

- `execution_order` — topo-sorted list of node ids.
- `edge_map` — `(target_id, targetHandle) → [(source_id, sourceHandle), ...]`.
- `consume_map` — same shape for shared-reference consumes.
- `compound_nodes`, `managed_ids`, `managed_to_region_start` — compound region bookkeeping.
- `type_warnings` — non-fatal issues.

Treat it as opaque for most use; read it when building custom execution tooling.

## Checklist before running a flow

- [ ] All node types used in `nodes` are registered on the registry passed to `compile`.
- [ ] Every edge's source/target handles exist as outputs/inputs on their nodes.
- [ ] `compiled.type_warnings` is empty, or each warning is intentional.
- [ ] If any node raises `HumanInputRequired`, caller has a persist/resume path.
- [ ] If compound nodes are used, `compound_types=[...]` passed to `compile`.
- [ ] If long-running, caller owns cancellation (task.cancel) and/or timeout (asyncio.wait_for).

## Debugging a failing flow

1. `for w in compiled.type_warnings: print(w)` — catches most wiring mistakes.
2. Stream with `execute` (not `execute_sync`) and log every event — reveals scheduling and skip behavior.
3. For node-level errors, catch `FlowExecutionError` (sync) or check `flow_error` events (async); `error.node_id` and `error.original` pinpoint the failure.
4. For resolver confusion, print `compiled.edge_map` and `compiled.consume_map` for the problem node.

## When your advice diverges from the installed version

Trust `python -m conductor.about` over this skill. If a behavior reported by the user contradicts what's written here, re-fetch the authoritative reference before making changes.
