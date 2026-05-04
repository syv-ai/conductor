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
│       ├── compound/           # CompoundNodeType protocol, ForEachNode, WhileNode, SubprocessNode
│       ├── expr/               # Sandboxed CEL-compatible expression evaluator
│       ├── flow_format/        # YAML / JSON flow file format (Flow ↔ dict)
│       └── about/              # Runnable library context: `python -m conductor.about`
├── packages/conductor-nodes/   # Reusable node library (text, math, logic, loop, json, regex, decision, while_loop, subprocess, signal)
│   └── src/conductor_nodes/    # Each module exposes register(reg); top-level register_all()
├── packages/conductor-providers/ # Framework adapters — react + fastapi subpackages ship today
│   └── src/conductor_providers/
│       ├── react/              # graph_to_react / react_to_graph / palette_from_registry
│       └── fastapi/            # conductor_router factory (/execute, /execute-stream, /compile, /nodes)
├── tests/test_core/            # 363 tests for conductor core (CEL, decision, while, subprocess, signal, compensation, timeout/idempotency, flow format, integration, …)
├── tests/test_nodes/           # conductor-nodes tests
├── tests/test_providers/       # conductor-providers tests (React + FastAPI)
├── demo/                       # Playground — FastAPI backend + Next.js frontend
│   ├── app.py                  # FastAPI endpoints (GET /api/nodes, POST /api/execute, /api/execute-stream) with CORS
│   ├── nodes.py                # Demo nodes: text, uppercase, template, combine, regex, make-list, number, math, if-else, for-each start/end
│   └── web/                    # Next.js 15 + shadcn + Tailwind v4 + @xyflow/react flow builder (independent of aka frontend)
├── examples/                   # 7 Jupyter notebooks (nodes, flows, store, control flow, discovery, HITL, shared refs)
├── docs/                       # Design specs, MkDocs site, logo (llms.txt lives inside the package)
├── .github/workflows/          # ci.yml (ruff + pytest on PR), docs-audit.yml (weekly)
└── .pre-commit-config.yaml     # nbstripout on *.ipynb
```

## Workspace packages

PyPI distribution names are `syv-conductor`, `syv-conductor-nodes`, `syv-conductor-providers` (Apache-2.0). The Python import paths (`conductor`, `conductor_nodes`, `conductor_providers`) are unchanged.

- **`conductor`** (dist: `syv-conductor`) — core engine (compile, execute, registry, widgets, errors, compound nodes, shared refs). Also ships CEL (`conductor.expr`), process-standard primitives (`Flow`, `FlowDependency`, `FlowTrigger`, `Actor`, decision-node guards, compensation, signals), and a YAML flow format (`conductor.flow_format`).
- **`conductor-nodes`** (dist: `syv-conductor-nodes`) — standard-library nodes. Each category module (`text`, `math`, `logic`, `loop`, `json_ops`, `regex_ops`) exposes `register(registry)`; top-level `register_all(registry, categories=...)` registers everything (or a filtered subset). Node IDs are category-prefixed (`text-uppercase`, `math-add`, …) except the for-each markers which match the `FOR_EACH` compound's discovery prefix.
- **`conductor-providers`** (dist: `syv-conductor-providers`) — framework adapters. Ships `conductor_providers.react` today with `graph_to_react` / `react_to_graph` / `palette_from_registry`. New providers (Svelte, Vue, etc.) go in sibling subpackages — no abstract base class to satisfy, each provider shapes itself to its framework.

Tag-driven publishing: pushing a `v*` tag fires `.github/workflows/publish.yml`, which builds wheels + sdists and uploads all three to PyPI (`PYPI_API_TOKEN`, idempotent via `skip-existing`).

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
uv run pytest tests/ -v           # Run all 430 tests (core + nodes + providers)
uvx ruff check .                  # Lint (what CI runs on PRs)
uv run python -m conductor.about  # Print the full library reference (llms.txt)
uv run python -m conductor.about sections   # List reference sections
uv run python -m conductor.about retry      # Print one section
uv run pytest tests/test_core/test_shared_references.py -v  # Run specific file
uv run uvicorn demo.app:app --port 8765 --reload    # Start demo backend (FastAPI)
cd demo/web && npm install && npm run dev           # Start demo frontend (Next.js on :3000)
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
- **ConnectionList** widget aggregates N edges into a labeled `dict[str, value]`. Keys come from `output_labels[handle]` (host-supplied display hint) when unique across the inputs, escalating to `"node_label (output)"` only on collisions; falls back to producer node id + handle otherwise.
- **Display hints** — optional `node_label: str | None` and `output_labels: dict[str, str] | None` on `GraphNode` (mirrored on the FastAPI provider's `NodeInput`). Pure UX; consumed by ConnectionList aggregation and host UIs.
- **SKIPPED sentinel** propagates through conditional branches.
- **ExtensionResolver** protocol lets host apps handle custom node types.

### Widgets and type → widget defaults

Every `WidgetType` enum value in `types.py` has a concrete Python class in `widgets.py`, so a generic frontend can render any widget by reading the registry. The set: `Text`, `Textarea`, `TemplateTextarea`, `CodeEditor`, `Dropdown`, `DependentDropdown`, `Multiselect`, `EntityDropdown`, `Number`, `Range`, `Checkbox`, `Switch`, `DatePicker`, `FileUpload`, `List`, `SchemaBuilder`, `IfElseBuilder`, `ConnectionList`, `Output`.

When a parameter has no widget on its `Annotated[...]`, the registry infers one from the Python type (`str→Text`, `int→Number(integer_only=True)`, `float→Number`, `bool→Checkbox`, `Date→DatePicker`, `list[T]→List(item_widget=default(T))`, `dict→SchemaBuilder`, `Base64Str/NamedFile→FileUpload`). Explicit `Annotated[T, Widget(...)]` always wins. This means `def f(x: str)` is now legal and gets a Text input; constraint-free fields don't need ceremony, but you always can opt in to full widget control.

`List` is the user-authored array widget (each item uses `item_widget`). `ConnectionList` is the multi-edge aggregator — they are not interchangeable.

Full guide for users and contributors: [`docs/widgets.md`](docs/widgets.md) (catalog, defaults, and how to add a new widget). Demo notebook: [`examples/08_widgets.ipynb`](examples/08_widgets.ipynb).

### Compile-time type checking

Every edge AND every consume binding is validated: source output type vs target input type. Rules: exact match, numeric interchangeability (int↔float), string coercion (anything→str), list auto-wrap (T→list[T]), ConnectionList accepts all. Union types (`str | int`) are compared alternative-by-alternative on either side — a match in any pair passes. Default: warnings on `compiled.type_warnings`. With `strict_types=True`: raises `CompilationError` (only for real mismatches; informational warnings like duplicate labels are not fatal).

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
  - `NodeValidationError` (pydantic failure, never retried; renders a one-line-per-field summary, with the original pydantic exception preserved on `.original` for hosts that want structured access)
  - `NodeExecutionError` (node function raised)
  - `NodeTimeoutError`
  - `NodeConnectionError` (raise from node code for transient network/API failures)
  - `SubprocessFailedError` (wraps a sub-flow failure)
- `InputResolutionError` — could not resolve inputs from edges
- `FlowExecutionError` — raised by `execute_sync` when flow fails
- `HumanInputRequired` / `FlowPausedError` — HITL signal + sync-mode counterpart
- `SignalRequired` — raised by signal/event nodes to pause on an external event
- `LoopRunawayError` — raised by `while` compound when `max_iterations` is exceeded

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

### Process-standard features

Conductor ships a full process-standard surface (see `spec.md` for the
design rationale). Each is additive to the core DAG engine:

#### Decision nodes + edge guards

A decision node (register with `is_decision=True`) branches by CEL
expressions on its outgoing edges:

```python
GraphEdge("e1", "d", "high", "result", None, when="amount > 1000", priority=10),
GraphEdge("e2", "d", "low",  "result", None),  # else fallback
```

Compile-time: exactly one else edge (no `when`) and ≥1 guarded edge.
Runtime: the first matching guard (priority-desc) wins; every other
outgoing edge is marked in `state.skipped_edges`, which the resolver
and skip checker honor.

#### CEL expressions (`conductor.expr`)

Self-contained, sandboxed CEL-compatible evaluator used by:
edge guards, `while` conditions, `idempotency_key`, signal
correlation, subprocess input mapping. Literals, arithmetic /
comparison / logical ops, ternary, `in`, dotted/indexed identifiers,
`$` root, and a built-in function library (`size`, `has`,
`contains`, `startsWith`, `endsWith`, `matches`, `string`/`int`/`double`/`bool`,
`exists`, `min`/`max`/`abs`, `lower`/`upper`).

#### Actor metadata

`@registry.node(actor=...)` accepts a bare string (`"human"`), a dict
(`{"kind": "human", "role": "finance_manager"}`), or an `Actor`. Kinds:
`system`, `human`, `agent`, `external_service`. Surfaced in the
registry JSON schema; the engine is indifferent.

#### Per-node timeout and idempotency key

* `timeout=` — seconds (float), ISO 8601 (`"PT30S"`), or shorthand
  (`"30s"`, `"250ms"`). Wraps execution with `asyncio.wait_for` and
  raises `NodeTimeoutError` (retryable) on expiry. Distinguished from
  flow-wide timeout.
* `idempotency_key=` — CEL expression evaluated once per node run.
  The resulting string is surfaced on the `node_start` event and
  injected into the function when it declares an `idempotency_key`
  parameter. Stable across retries.

#### While / until compound region

Type markers `while-start` / `while-end`, discovered by the `WHILE`
compound type. CEL `condition` evaluated each iteration with
`iteration` (count, 1-based after first) and `last` (body's last
return value) in scope. `max_iterations` safety cap raises
`LoopRunawayError`. `negate=True` turns while into until.

#### Subprocess compound

`subprocess-call` node references another flow by `(flow_id, version)`.
Register flows in a `SubprocessRegistry` and pass to
`compile(subprocess_registry=...)`. Runtime depth cap catches
infinite recursion. Errors bubble as `SubprocessFailedError`. Input
mapping via the `inputs:` dict (static values or `$`-prefixed CEL).

#### Compensation / saga

Per-node `compensation=` field points at another node. When the flow
fails, the engine walks `state.completed_order` in reverse and
dispatches each completed node's compensation, giving it
`(target_node_id, original_inputs, original_output)`. Events:
`compensation_start`, `compensation_complete`, `compensation_failed`.
Best-effort — one failure doesn't abort the cascade. Per-node
`on_error` policy (`fail` default, `continue`, `compensate`) controls
the triggering semantics.

#### Signal / event nodes

Nodes raise `SignalRequired(name, correlation=..., timeout_seconds=...)`
to pause the flow. Engine checkpoints `(signal_name, correlation,
signal_timeout_seconds)` and yields `signal_waiting` + `flow_paused`
events. Resume with `resume_sync(compiled, checkpoint, payload)`.
Hosts use `FlowCheckpoint.matches_signal(name, payload)` to route
incoming events — it evaluates the CEL correlation on the host side.

#### Flow-level metadata

`Flow` dataclass wraps nodes/edges with:

* `dependencies: tuple[FlowDependency, ...]` — external systems. Node
  `uses=[...]` lists must reference declared dep ids (compile-time
  check).
* `triggers: tuple[FlowTrigger, ...]` — manual/schedule/event/webhook
  config. Engine stores; host wires.
* `on_error_default` — flow-level default for node `on_error`.

#### YAML / JSON flow format (`conductor.flow_format`)

Round-trip `Flow` ↔ dict via `load_flow` / `flow_to_dict`, and
YAML/JSON files via `yaml_to_flow` / `flow_to_yaml` /
`load_flow_from_path` / `dump_flow`. Defaults (`version=1`,
`on_error_default="fail"`) are omitted on output. Requires PyYAML
(optional extra: `conductor[yaml]`).

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

Docs drift is a real failure mode for this project — the whole point of `CLAUDE.md`, `packages/conductor/src/conductor/about/llms.txt`, and `docs/shared-references.md` is that future agent sessions can land with full context. That only works if the docs stay in sync with the code.

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

### Composing registries
```python
# Pull pre-built nodes into your own registry
mine = NodeRegistry()
# ... register your own nodes ...
mine.merge(conductor_nodes.get_default_registry())

# Chainable; conflict policies: "raise" (default), "skip", "error-summary"
mine.merge(other_registry, on_conflict="skip")
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
- `packages/conductor/src/conductor/about/llms.txt` is the canonical AI context file; because it lives inside the package, installing the wheel is enough for `python -m conductor.about` to work
- After any session that adds or changes public surface area, run `/docs-audit` to keep `CLAUDE.md`, `README.md`, `packages/conductor/src/conductor/about/llms.txt`, `docs/shared-references.md`, and `docs/index.md` in sync; weekly CI catches what the slash command misses
