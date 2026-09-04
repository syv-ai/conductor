<p align="center">
  <img src="assets/logo-white-background.png" alt="Conductor logo" width="140">
</p>

# Conductor

A reusable, host-agnostic graph execution engine for building DAG-based workflow systems. Declare a node as a class whose typed `run` signature is its interface, compile placements of it into a validated execution plan, and run the plan with **eager parallel streaming execution** and **built-in retry**.

## Highlights

- **One node contract** — a `NodeDefinition` subclass; the signature of `run` is the interface, read once into `Input` and `Output` records that drive validation, execution and the palette.
- **A type vocabulary the host owns** — every wire value is a `DType`; conductor ships the mechanism (`DType`, `Series[X]`, `accepts`) and no vocabulary.
- **Versions with a policy** — `@version(n, policy=Policy(retries=..., timeout=...))`, `@upgrade(1, 2)`, `@deprecated`.
- **Eager parallel scheduling** — independent branches in a DAG run concurrently with no configuration.
- **Retry** — on the version's `Policy` or a run-level `RetryConfig`; a clean `node_retry` event on every attempt.
- **Structured error hierarchy** — `NodeValidationError`, `NodeExecutionError`, `NodeConnectionError`, `NodeTimeoutError`, and more, all carrying `node_id` / `node_type` context.
- **Branching by value** — a node returns `SKIPPED` on the branch not taken; exclusive outputs share a `choice`.

## Quick start

```python
from typing import Annotated
from conductor import GraphNode, NodeDefinition, NodeRegistry, Policy, Result, compile, version
from conductor.execution.engine import execute_sync
from conductor.widgets import Text as TextWidget
from conductor_nodes.types import Text          # or a DType of your own

class Fetch(NodeDefinition):
    id = "fetch"
    title = "Fetch"
    description = "HTTP GET"
    category = "http"

    @version(1, policy=Policy(retries=3, delay=0.5))
    def run(self, url: Annotated[Text, TextWidget(title="URL")]) -> Annotated[Text, Result(title="Body")]:
        ...

registry = NodeRegistry()
registry.register(Fetch)

compiled = compile(
    nodes=[GraphNode("n1", "fetch", 1, {"url": "https://example.com"})],
    edges=[],
    registry=registry,
)

results = execute_sync(compiled)     # results["n1"]["result"]
```

## Eager parallel execution

As soon as a node's dependencies complete, its task is dispatched via `asyncio.create_task`. Sync `run` methods run on `asyncio.to_thread`, so they don't block the event loop. No flag is needed — this is the default (and only) execution mode.

```
  A (0.3s) ──> C (0.3s) ──┐
                           ├──> E (0.3s)
  B (0.3s) ──> D (0.3s) ──┘
```

Sequential: 5 × 0.3 s = 1.5 s. Eager: `A+B` || `C+D` || `E` = ~0.9 s.

## Retry

```python
# On the version — the node author's call
@version(1, policy=Policy(retries=3, delay=0.5))
def run(self, ...): ...

# Run-level — applies to nodes whose policy sets none
execute_sync(compiled, retry=RetryConfig(max_retries=2, delay=1.0, backoff_factor=2.0))
```

- Delay: `delay * backoff_factor ** (attempt - 1)`
- Retried: `NodeExecutionError`, `NodeConnectionError`
- Never retried: `NodeValidationError`
- Each attempt emits a `node_retry` event: `{attempt, max_retries, error, delay}`
- `Policy(timeout=...)` bounds one attempt; expiry is `NodeTimeoutError`

## Shared references (produce / consume)

An alternative to drawn edges for fan-out. Declared per placement on `GraphNode`; participates in scheduling and cycle detection like an edge.

```python
compiled = compile(
    nodes=[
        GraphNode("mapper", "build-map", 1, {"seed": "x"},
                  produces={"result": "pseudonym map"}),
        GraphNode("redactor", "redact", 1, {"text": "Alice met Bob."},
                  consumes={"mapping": ("mapper", "result")}),
    ],
    edges=[],     # no edge needed
    registry=registry,
)
```

Reference identity is `(producer node id, output name)`; the label is UI-only so renames never break subscribers.

## Error hierarchy

```
ConductorError
├── CompilationError
│   └── CycleDetectionError
├── NodeError                       # carries node_id, node_type, original
│   ├── NodeValidationError         # pydantic — never retried
│   ├── NodeExecutionError          # run() raised — retried per policy
│   ├── NodeTimeoutError
│   └── NodeConnectionError         # transient network/API — retried
├── InputResolutionError
└── FlowExecutionError              # raised by execute_sync
```

Raise `NodeConnectionError` from `run` to mark a transient failure as retry-worthy.

## Further reading

- [`packages/conductor/src/conductor/about/llms.txt`](../packages/conductor/src/conductor/about/llms.txt) — the packaged reference (also `python -m conductor.about`).
- [`widgets.md`](./widgets.md) — the controls and how an input declares one.
- Jupyter notebooks in [`examples/`](https://github.com/syvai/conductor/tree/main/examples) cover nodes, flows, discovery and widgets.
