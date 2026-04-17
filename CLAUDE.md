# Conductor

Reusable DAG execution engine extracted from production flow builders. Node registration, graph compilation with type checking, eager parallel streaming execution with retry, shared references across region boundaries, and human-in-the-loop checkpointing.

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
│       ├── graph/              # GraphNode/Edge, topology, compiler, regions, type_check, shared_refs
│       ├── execution/          # Engine (eager+parallel), retry, state, resolver, events, store, checkpoint
│       ├── compound/           # CompoundNodeType protocol, ForEachNode
│       └── about/              # Runnable library context: `python -m conductor.about`
├── packages/conductor-nodes/   # Reusable node library (text, math, logic, loop, json, regex)
│   └── src/conductor_nodes/    # Each module exposes register(reg); top-level register_all()
├── packages/conductor-providers/ # Framework adapters — react subpackage ships today
│   └── src/conductor_providers/
│       └── react/              # graph_to_react / react_to_graph / palette_from_registry
├── tests/test_core/            # 171 tests for conductor core
├── tests/test_nodes/           # 46 tests for conductor-nodes
├── tests/test_providers/       # 18 tests for conductor-providers (React round-trips)
├── demo/                       # FastAPI playground with browser UI
│   ├── app.py                  # FastAPI endpoints (GET /api/nodes, POST /api/execute-stream)
│   ├── nodes.py                # 10 demo nodes (text, number, math, summarizer, loop, etc.)
│   └── static/index.html       # Single-page flow builder UI
├── examples/                   # 7 Jupyter notebooks (nodes, flows, store, control flow, discovery, HITL, shared refs)
├── docs/                       # Design specs, llms.txt, MkDocs site, logo
├── .github/workflows/          # ci.yml (ruff + pytest on PR), docs-audit.yml (weekly)
└── .pre-commit-config.yaml     # nbstripout on *.ipynb
```

## Workspace packages

- **`conductor`** — core engine (compile, execute, registry, widgets, errors, compound nodes, shared refs).
- **`conductor-nodes`** — standard-library nodes. Each category module (`text`, `math`, `logic`, `loop`, `json_ops`, `regex_ops`) exposes `register(registry)`; top-level `register_all(registry, categories=...)` registers everything (or a filtered subset). Node IDs are category-prefixed (`text-uppercase`, `math-add`, …) except the for-each markers which match the `FOR_EACH` compound's discovery prefix.
- **`conductor-providers`** — framework adapters. Ships `conductor_providers.react` today with `graph_to_react` / `react_to_graph` / `palette_from_registry`. New providers (Svelte, Vue, etc.) go in sibling subpackages — no abstract base class to satisfy, each provider shapes itself to its framework.

## Tech stack

- Python 3.12+, uv workspace monorepo
- pydantic (only hard dependency of conductor core)
- pytest + pytest-asyncio for tests
- FastAPI + uvicorn for demo app
- pre-commit + nbstripout for clean notebook diffs
- ruff for linting (config in root pyproject.toml, `uvx ruff check .`); PR-triggered CI in `.github/workflows/ci.yml`

## Key commands

```bash
uv sync                           # Install all deps
uv sync --group demo              # Install with demo deps (FastAPI)
uv run pre-commit install         # Activate the nbstripout hook on your clone
uv run pytest tests/ -v           # Run all 235 tests (core + nodes + providers)
uvx ruff check .                  # Lint (what CI runs on PRs)
uv run python -m conductor.about  # Print the full library reference (llms.txt)
uv run python -m conductor.about sections   # List reference sections
uv run python -m conductor.about retry      # Print one section
uv run pytest tests/test_core/test_shared_references.py -v  # Run specific file
uv run uvicorn demo.app:app --port 8765 --reload    # Start demo UI
uv run jupyter lab examples/                         # Open the example notebooks
```

Slash command: `/docs-audit` — runs a docs review against the last N commits and edits the user-facing docs in place (no commits). Expected hygiene after feature-bearing sessions. See also `.github/workflows/docs-audit.yml` for the weekly CI safety net.

## Architecture

Three-phase: `register → compile → execute`.

1. **Registry** — `@registry.node()` decorator introspects function signature at import time. Extracts `Annotated[T, Widget]` metadata into frozen dataclasses, generates Pydantic validation model, stores raw function. Class-based nodes use `BaseNode` ABC + `registry.register_class()`.
2. **Compile** — `compile(nodes, edges, registry)` validates structure, topological sorts, discovers compound regions, validates shared-reference produce/consume bindings, type-checks every edge + consume binding. Returns immutable `CompiledGraph` with warnings.
3. **Execute** — `execute(compiled)` is an async generator yielding `ExecutionEvent`s. Nodes are scheduled eagerly: as soon as all dependencies complete, a node's task is created — independent branches run concurrently. Dispatches via 3-way lookup: compound → extension → registry. `execute_sync()` is a blocking wrapper.

### Node types

- **IO nodes** — Plain functions with `@registry.node()`. Data transformation, no effect on execution order.
- **Control nodes** — Same API but with `category=NodeCategory.CONTROL`. If/else uses SKIPPED sentinel. For-each uses compound node regions.
- **Class-based nodes** — Subclass `BaseNode` for complex nodes needing state or custom dispatch.

### Data flow

Three ways for a node to receive a value, ordered by resolver precedence (first match wins):

1. **Edges** — the primary mechanism, visible as wires in the UI. `InputResolver` extracts outputs by handle.
2. **Shared references** — per-instance `produces`/`consumes` declarations on `GraphNode`. Invisible to edges, but participate in dependency ordering, cycle detection, and type checking. Can cross compound region boundaries.
3. **Static data** — `GraphNode.data` dict; used when no edge or consume targets the input. Consumes override static data.
4. **Widget default** — Pydantic default if present.

Two additional concepts:

- **FlowStore** — imperative side-channel key/value cache (`store: FlowStore` auto-injected). Useful for per-run scratch data; not part of the DAG.
- **ConnectionList** widget aggregates N edges into a labeled `dict[str, value]`.
- **SKIPPED sentinel** propagates through conditional branches.
- **ExtensionResolver** protocol lets host apps handle custom node types.

### Compile-time type checking

Every edge AND every consume binding is validated: source output type vs target input type. Rules: exact match, numeric interchangeability (int↔float), string coercion (anything→str), list auto-wrap (T→list[T]), ConnectionList accepts all. Default: warnings on `compiled.type_warnings`. With `strict_types=True`: raises `CompilationError` (only for real mismatches; informational warnings like duplicate labels are not fatal).

### Eager parallel execution

The engine uses a dependency-driven scheduler (`_run_eager` in `execution/engine.py`):
- Each schedulable node tracks an in-degree counter (unfinished deps from edges + consumes).
- When in-degree hits 0, `asyncio.create_task` dispatches the node via `asyncio.to_thread` so sync functions don't block the loop.
- Node events flow through an `asyncio.Queue`; the main loop yields them to the caller.
- Failures cancel all running tasks; `flow_paused` also cancels peers and emits a checkpoint.
- Consumers inside compound regions have their dependency redirected onto the region's start node (`managed_to_region_start`), so the region waits for its top-level producers.

Independent branches overlap without any per-flow configuration. A chain of 3 × 0.3 s sleeps still serializes to ~0.9 s; two parallel such chains that join still finish in ~0.9 s.

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

### Shared references (produce / consume)

A first-class alternative to explicit edges for the "fan-out" and "cross-region" cases. Design spec: [`docs/shared-references.md`](docs/shared-references.md). Key points:

- Per-instance opt-in on `GraphNode`:
  - `produces: dict[str, str] | None` — output handle → display label
  - `consumes: dict[str, tuple[str, str]] | None` — input handle → `(producer_id, output_handle)`
- Reference identity is `(producer_node_id, output_handle)`. The label is UI-only; renaming labels never breaks subscribers.
- Validated at compile time: producer handle exists and is top-level (v1 restriction); consumer target exists, points at a declared producer, has no colliding edge on the same handle. Duplicate labels are a non-fatal warning (`code="shared-label-collision"`).
- At runtime, `consume_map` on `CompiledGraph` feeds the scheduler and resolver just like edges do. `InputResolver.resolve` checks edges → consumes → static data → defaults.
- `managed_to_region_start` redirects consume deps from managed body nodes onto their region's start, so compound regions wait for producers.
- Consumers **can** sit inside for-each bodies; they see the same producer value on every iteration (broadcast, not per-iteration).
- Producers inside compound regions are rejected in v1 (semantics TBD).

See `examples/07_shared_references.ipynb` for a walkthrough.

### Human-in-the-loop

- Node raises `HumanInputRequired(prompt, schema=...)` to pause.
- Engine checkpoints to JSON-serializable `FlowCheckpoint`, yields `flow_paused` event.
- `resume(compiled, checkpoint, response)` continues from the paused node.
- FlowStore and shared reference values both survive checkpoint/resume (they live in `state.results`).
- Sync: `FlowPausedException` / `resume_sync()`.

### Custom data types

- `NewType("MyType", str)` → surfaces as `"mytype"` in the frontend JSON schema.
- Built-in: `Base64Str`, `Date`, `NamedFile`, `MultiNamedFile`.
- Host apps define their own — runtime base type, distinct schema string.

### Documentation maintenance

Docs drift is a real failure mode for this project — the whole point of `CLAUDE.md`, `docs/llms.txt`, and `docs/shared-references.md` is that future agent sessions can land with full context. That only works if the docs stay in sync with the code.

Two channels exist for keeping them aligned:

1. **On-demand: `/docs-audit` slash command** (`.claude/commands/docs-audit.md`). Run it at the end of any session that touched public API, added a feature, or changed default behavior. It reads the last N commits (default 10; pass a number or `since-release`), compares against the docs, and edits them in place. It does **not** commit — the user reviews via `git diff` and decides. This is the primary channel.
2. **Weekly safety net: `.github/workflows/docs-audit.yml`**. Every Monday (and on manual `workflow_dispatch`), CI runs the same audit over the last 14 days of commits and opens a PR if anything is out of sync. Needs `ANTHROPIC_API_KEY` as a repo secret. Close the PR without merging if the suggestions are wrong.

**Running `/docs-audit` is expected hygiene at the end of any feature-bearing session** — the CI workflow is a catcher of last resort, not a substitute. If you add a public API, a field on `GraphNode` / `CompiledGraph`, a new error type, a new event, or a new notebook, run the audit.

When the audit flags a discrepancy it can't resolve (commit says X, code does Y), trust the code and surface the discrepancy in the summary — don't write docs for things that don't exist.

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
    print(f"{w.code}: {w.message}")
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

### Shared references
```python
# No edge needed between mapper and redactor
compiled = compile(
    nodes=[
        GraphNode("mapper", "build-map@1", {"seed": "x"},
                  produces={"result": "pseudonym map"}),
        GraphNode("redactor", "redact@1", {"text": "Alice met Bob."},
                  consumes={"mapping": ("mapper", "result")}),
    ],
    edges=[],
    registry=registry,
)
```

## Conventions

- Nodes are versioned as `base_id@version` (e.g., `echo@2`)
- All node results normalized to dicts: `{"result": value}` for single, `{"output_1": v1, "output_2": v2}` for tuples
- SKIPPED sentinel propagates — if all inputs (edges + consumes) are SKIPPED, node is skipped
- Widget annotations are the single source of truth for validation AND frontend rendering
- Streaming (async generator) is the only execution path; sync is a wrapper
- Eager scheduling is the default and only mode — there is no sequential-execute switch
- Retries live on the node (`max_retries`, `retry_delay`) or a global `RetryConfig`; node-level wins
- Shared references are per-instance; the same node *type* can be shared in one flow and not in another
- Notebook outputs are stripped on commit by `nbstripout` — run cells locally to see values
- `docs/shared-references.md` is the authoritative v1 design spec for produce/consume
- `docs/llms.txt` provides importable AI context for other projects using this library — the wheel ships it at `conductor.about.llms.txt` so `python -m conductor.about` works from any installed environment
- After any session that adds or changes public surface area, run `/docs-audit` to keep `CLAUDE.md`, `README.md`, `docs/llms.txt`, `docs/shared-references.md`, and `docs/index.md` in sync; weekly CI catches what the slash command misses
