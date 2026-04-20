<p align="center">
  <img src="assets/logo-white-background.png" alt="Conductor logo" width="140">
</p>

# Conductor

A reusable, host-agnostic graph execution engine for building DAG-based workflow systems. Register nodes as plain Python functions with type annotations, compile them into a validated execution plan, and run them with **eager parallel streaming execution** and **built-in retry**.

## Highlights

- **Eager parallel scheduling** — independent branches in a DAG run concurrently with no configuration.
- **Node-level and global retry** — exponential backoff with a clean `node_retry` event on every attempt.
- **Shared references (produce / consume)** — per-instance bindings that replace fan-out edges and cross for-each boundaries.
- **Structured error hierarchy** — `NodeValidationError`, `NodeExecutionError`, `NodeConnectionError`, `NodeTimeoutError`, and more, all carrying `node_id` / `node_type` context.
- **Human-in-the-loop** — pause on `HumanInputRequired`, checkpoint to JSON, resume later.
- **Widget-annotated registration** — one `Annotated[T, Widget]` drives validation, execution, and frontend rendering.

## Quick start

```python
from typing import Annotated
from conductor import NodeRegistry, GraphNode, GraphEdge, compile
from conductor.execution.engine import execute_sync
from conductor.execution.retry import RetryConfig
from conductor.widgets import Text, Output

registry = NodeRegistry()

@registry.node(
    "fetch", version=1, name="Fetch", description="HTTP GET",
    max_retries=3, retry_delay=0.5,
)
def fetch(url: Annotated[str, Text(label="URL")]) -> Annotated[str, Output(label="Body")]:
    ...

compiled = compile(
    nodes=[GraphNode("n1", "fetch@1", {"url": "https://example.com"})],
    edges=[],
    registry=registry,
)

results = execute_sync(compiled)
```

## Eager parallel execution

As soon as a node's dependencies complete, its task is dispatched via `asyncio.create_task`. Sync node functions run on `asyncio.to_thread`, so they don't block the event loop. No flag is needed — this is the default (and only) execution mode.

```
  A (0.3s) ──> C (0.3s) ──┐
                           ├──> E (0.3s)
  B (0.3s) ──> D (0.3s) ──┘
```

Sequential: 5 × 0.3 s = 1.5 s. Eager: `A+B` || `C+D` || `E` = ~0.9 s.

## Retry

```python
# Node-level (wins over any global config)
@registry.node("fetch", ..., max_retries=3, retry_delay=0.5)
def fetch(...): ...

# Global — applies to nodes that don't set their own
execute_sync(compiled, retry=RetryConfig(max_retries=2, delay=1.0, backoff_factor=2.0))
```

- Delay: `retry_delay * backoff_factor ** (attempt - 1)`
- Retried: `NodeExecutionError`, `NodeConnectionError`
- Never retried: `NodeValidationError`, `HumanInputRequired`
- Each attempt emits a `node_retry` event: `{attempt, max_retries, error, delay}`

## Shared references (produce / consume)

An alternative to drawn edges for fan-out and cross-region wiring. Declared per-instance on `GraphNode`; validated at compile time; participates in scheduling, cycle detection, and type checking identically to edges.

```python
compiled = compile(
    nodes=[
        GraphNode("mapper", "build-map@1", {"seed": "x"},
                  produces={"result": "pseudonym map"}),
        GraphNode("redactor", "redact@1", {"text": "Alice met Bob."},
                  consumes={"mapping": ("mapper", "result")}),
    ],
    edges=[],     # no edge needed
    registry=registry,
)
```

Reference identity is `(producer_node_id, output_handle)`; the label is UI-only so renames never break subscribers. A consumer inside a for-each body reads the same producer value on every iteration (broadcast, not per-iteration) — this is the idiomatic way to inject a system prompt or a pseudonymisation map into every loop step.

v1 constraint: producers must be top-level; consumers can be anywhere.

Full rules and error cases: [`shared-references.md`](./shared-references.md).

## Error hierarchy

```
ConductorError
├── CompilationError
│   ├── CycleDetectionError
│   └── TypeCheckError
├── NodeError                       # carries node_id, node_type, original
│   ├── NodeValidationError         # pydantic — never retried
│   ├── NodeExecutionError          # node function raised — retried
│   ├── NodeTimeoutError
│   └── NodeConnectionError         # transient network/API — retried
├── InputResolutionError
├── FlowExecutionError              # raised by execute_sync
├── FlowPausedError                 # HITL sync counterpart
└── HumanInputRequired              # pauses execution
```

Raise `NodeConnectionError` from your node code to mark a transient failure as retry-worthy. Legacy aliases (`NodeValidationException`, `NodeExecutionException`, `FlowExecutionException`, `FlowPausedException`) still work.

## Further reading

- [`packages/conductor/src/conductor/about/llms.txt`](../packages/conductor/src/conductor/about/llms.txt) — importable AI context for the whole library (also `python -m conductor.about`).
- [`conductor-design.md`](./conductor-design.md) — full design specification.
- Jupyter notebooks in [`examples/`](https://github.com/syvai/conductor/tree/main/examples) cover nodes, flows, store, control flow, discovery, and HITL.
