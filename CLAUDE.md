# Conductor

Reusable DAG execution engine extracted from production flow builders. Provides node registration, graph compilation with type checking, eager parallel streaming execution with retry, and human-in-the-loop checkpointing.

## Repository structure

```
conductor/
├── packages/conductor/        # Core library (the only package so far)
│   └── src/conductor/
│       ├── types.py            # WidgetType, ResultFormat, NodeCategory enums, custom type aliases
│       ├── widgets.py          # Widget ABC + concrete widgets (Text, Dropdown, etc.)
│       ├── metadata.py         # InputMetadata, OutputMetadata frozen dataclasses
│       ├── validation.py       # Pydantic model generation from function signatures
│       ├── errors.py           # Exception hierarchy + HumanInputRequired, FlowPausedError
│       ├── _sentinel.py        # SKIPPED singleton for conditional branches
│       ├── node.py             # BaseNode ABC for class-based nodes
│       ├── registry/           # NodeRegistry, @node decorator, auto-discovery, JSON schema
│       ├── graph/              # GraphNode/Edge, topological sort, compiler, regions, type_check
│       ├── execution/          # Engine (eager+parallel), retry, state, resolver, events, store, checkpoint
│       └── compound/           # CompoundNodeType protocol, ForEachNode
├── tests/test_core/            # 131 tests across 13 files (incl. eager scheduling + retry)
├── demo/                       # FastAPI playground with browser UI
│   ├── app.py                  # FastAPI endpoints (GET /api/nodes, POST /api/execute-stream)
│   ├── nodes.py                # 10 demo nodes (text, number, math, summarizer, loop, etc.)
│   └── static/index.html       # Single-page flow builder UI
├── examples/                   # 6 Jupyter notebooks (nodes, flows, store, control flow, discovery, HITL)
└── docs/                       # Design spec, llms.txt, MkDocs site
```

## Planned packages (not yet built)

- `conductor-nodes` — Reusable node library (text, math, json, regex, conditional, loop markers)
- `conductor-react` — ReactFlow bridge (JSON conversion, frontend schema generation)

## Tech stack

- Python 3.12+, uv workspace monorepo
- pydantic (only hard dependency of conductor core)
- pytest + pytest-asyncio for tests
- FastAPI + uvicorn for demo app

## Key commands

```bash
uv sync                           # Install all deps
uv sync --group demo              # Install with demo deps (FastAPI)
uv run pytest tests/ -v           # Run all 131 tests
uv run pytest tests/test_core/test_execution.py -v  # Run specific file
uv run uvicorn demo.app:app --port 8765 --reload    # Start demo UI
uv run jupyter lab examples/                         # Open the example notebooks
```

## Architecture

Three-phase: `register → compile → execute`.

1. **Registry** — `@registry.node()` decorator introspects function signature at import time. Extracts `Annotated[T, Widget]` metadata into frozen dataclasses, generates Pydantic validation model, stores raw function. Class-based nodes use `BaseNode` ABC + `registry.register_class()`.
2. **Compile** — `compile(nodes, edges, registry)` validates structure, topological sorts, discovers compound regions, type-checks all edge connections. Returns immutable `CompiledGraph` with warnings.
3. **Execute** — `execute(compiled)` is an async generator yielding `ExecutionEvent`s. Nodes are scheduled eagerly: as soon as all dependencies complete, a node's task is created — independent branches run concurrently. Dispatches via 3-way lookup: compound → extension → registry. `execute_sync()` is a blocking wrapper.

### Node types

- **IO nodes** — Plain functions with `@registry.node()`. Data transformation, no effect on execution order.
- **Control nodes** — Same API but with `category=NodeCategory.CONTROL`. If/else uses SKIPPED sentinel. For-each uses compound node regions.
- **Class-based nodes** — Subclass `BaseNode` for complex nodes needing state or custom dispatch.

### Data flow

- **Edges** are the primary data flow (visible as wires in UI). `InputResolver` extracts outputs by handle, overlays onto static node data.
- **FlowStore** is a side-channel key-value cache. Function nodes declare `store: FlowStore` for auto-injection.
- **ConnectionList** widget aggregates N connections into a labeled `dict[str, value]`.
- **SKIPPED sentinel** propagates through conditional branches.
- **ExtensionResolver** protocol lets host apps handle custom node types.

### Compile-time type checking

Every edge is validated: source output type vs target input type. Rules: exact match, numeric interchangeability (int↔float), string coercion (anything→str), list auto-wrap (T→list[T]), ConnectionList accepts all. Default: warnings on `compiled.type_warnings`. With `strict_types=True`: raises `CompilationError`.

### Eager parallel execution

The engine uses a dependency-driven scheduler (`_run_eager` in `execution/engine.py`):
- Each schedulable node tracks an in-degree counter (unfinished deps).
- When in-degree hits 0, `asyncio.create_task` dispatches the node via `asyncio.to_thread` so sync functions don't block the loop.
- Node events flow through an `asyncio.Queue`; the main loop yields them to the caller.
- Failures cancel all running tasks; `flow_paused` also cancels peers and emits a checkpoint.

Independent branches therefore overlap without any per-flow configuration. A chain of 3 × 0.3 s sleeps still serializes to ~0.9 s; two parallel such chains that join still finish in ~0.9 s.

### Retry

Retries are node-level first, global second (`execution/retry.py`):
- Per-node: `@registry.node("fetch", max_retries=3, retry_delay=0.5)` — wins over global.
- Global: `execute(compiled, retry=RetryConfig(max_retries=2, delay=1.0, backoff_factor=2.0))`.
- Delay formula: `delay * backoff_factor ** (attempt - 1)`.
- `NodeValidationError` is **never** retried (bad input won't fix itself).
- `NodeConnectionError` / `NodeExecutionError` are retried.
- `HumanInputRequired` short-circuits retry (pause immediately).
- Each retry emits a `node_retry` event with `{attempt, max_retries, error, delay}`.

### Error hierarchy

All exceptions inherit from `ConductorError` (see `errors.py`):

- `CompilationError` — graph structure invalid
  - `CycleDetectionError`, `TypeCheckError`
- `NodeError` — carries `node_id`, `node_type`, `original`
  - `NodeValidationError` (pydantic failure, never retried)
  - `NodeExecutionError` (node function raised)
  - `NodeTimeoutError`
  - `NodeConnectionError` (raise from node code for transient network/API failures)
- `InputResolutionError` — could not resolve inputs from edges
- `FlowExecutionError` — raised by `execute_sync` when flow fails
- `HumanInputRequired` / `FlowPausedError` — HITL signal + sync-mode counterpart

Legacy aliases (`NodeValidationException`, `NodeExecutionException`, `FlowExecutionException`, `FlowPausedException`) still work but map to the new `*Error` names.

### Human-in-the-loop

- Node raises `HumanInputRequired(prompt, schema=...)` to pause.
- Engine checkpoints to JSON-serializable `FlowCheckpoint`, yields `flow_paused` event.
- `resume(compiled, checkpoint, response)` continues from the paused node.
- FlowStore survives checkpoint/resume. Multiple sequential pauses supported.
- Sync: `FlowPausedException` / `resume_sync()`.

### Custom data types

- `NewType("MyType", str)` → surfaces as `"mytype"` in the frontend JSON schema.
- Built-in: `Base64Str`, `Date`, `NamedFile`, `MultiNamedFile`.
- Host apps define their own — runtime base type, distinct schema string.

## Patterns

### Registering a node
```python
@registry.node("my-node", version=1, name="My Node", description="Does stuff")
def my_node(
    text: Annotated[str, Text(label="Input")],
) -> Annotated[str, Output(label="Result")]:
    return text.upper()
```

### Building and running a flow
```python
compiled = compile(
    nodes=[GraphNode("n1", "my-node@1", {"text": "hello"})],
    edges=[],
    registry=registry,
)
results = execute_sync(compiled)
```

### Checking type warnings
```python
compiled = compile(nodes, edges, registry)
for w in compiled.type_warnings:
    print(f"Warning: {w.message}")
```

### Retry
```python
# Per-node (always applied)
@registry.node("fetch", ..., max_retries=3, retry_delay=0.5)
def fetch(...): ...

# Global fallback
from conductor.execution.retry import RetryConfig
execute_sync(compiled, retry=RetryConfig(max_retries=2, delay=1.0, backoff_factor=2.0))
```

## Conventions

- Nodes are versioned as `base_id@version` (e.g., `echo@2`)
- All node results normalized to dicts: `{"result": value}` for single, `{"output_1": v1, "output_2": v2}` for tuples
- SKIPPED sentinel propagates — if all inputs are SKIPPED, node is skipped
- Widget annotations are the single source of truth for validation AND frontend rendering
- Streaming (async generator) is the only execution path; sync is a wrapper
- Eager scheduling is the default and only mode — there is no sequential-execute switch
- Retries live on the node (`max_retries`, `retry_delay`) or a global `RetryConfig`; node-level wins
- `docs/llms.txt` provides importable AI context for other projects using this library
