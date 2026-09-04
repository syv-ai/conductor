# Conductor

Reusable DAG execution engine extracted from production flow builders. One node contract (a class whose typed `run` signature is its interface), a type vocabulary the host declares, graph compilation, eager parallel streaming execution with retry.

## Repository structure

```
conductor/
├── packages/conductor/        # Core library
│   └── src/conductor/
│       ├── node.py             # NodeDefinition ABC, NodeVersion/GraphVersion, Policy, version/upgrade/deprecated, describe()
│       ├── interface.py        # Interface.of(run) — the signature read once; Provided; model_of
│       ├── metadata.py         # Field, Input, Output frozen dataclasses
│       ├── returns.py          # Result (what an author writes on a return); outputs_of / unpack
│       ├── dtype.py            # DType — a value's type; accepts(); Single; registered_dtypes()
│       ├── dtype_ref.py        # DTypeRef — a dtype as a pydantic field
│       ├── series.py           # Series[X], Index, Row
│       ├── ref.py              # Ref — "<node id>.<field>"
│       ├── widgets.py          # Widget ABC + the controls; AnyWidget discriminated union
│       ├── errors.py           # Exception hierarchy
│       ├── _sentinel.py        # SKIPPED
│       ├── registry/           # NodeRegistry, runner_for, discover_nodes
│       ├── graph/              # model (GraphNode/Flow), binding (Sources/Static), views (dependencies, interface), topology, compiler, dynamic_inputs/outputs
│       ├── execution/          # engine (eager+parallel), retry, state, resolver, events
│       ├── flow_format/        # YAML / JSON flow file format (Flow ↔ dict)
│       └── about/              # Runnable library context: `python -m conductor.about`
├── packages/conductor-nodes/   # Standard node library (text, math, logic, json_ops, regex_ops, decision) + its types
│   └── src/conductor_nodes/    # Each module exposes NODES and register(registry); top-level register_all()
├── packages/conductor-providers/ # Framework adapters — react + fastapi subpackages
│   └── src/conductor_providers/
│       ├── react/              # graph_to_react / react_to_graph / palette_from_registry
│       └── fastapi/            # conductor_router factory (/nodes, /compile, /execute, /execute-stream, /entities/{kind})
├── tests/test_core/            # conductor core
├── tests/test_nodes/           # conductor-nodes (types, the catalog contract, every node end to end)
├── tests/test_providers/       # conductor-providers (React + FastAPI)
├── tests/test_stress/          # large graphs, cancellation
├── examples/                   # Jupyter notebooks
├── docs/                       # MkDocs site + design notes (llms.txt lives inside the package)
├── .github/workflows/          # ci.yml (ruff + pytest on PR), docs-audit.yml (weekly)
└── .pre-commit-config.yaml     # nbstripout on *.ipynb
```

## Workspace packages

PyPI distribution names are `syv-conductor`, `syv-conductor-nodes`, `syv-conductor-providers` (Apache-2.0). The Python import paths (`conductor`, `conductor_nodes`, `conductor_providers`) are unchanged.

- **`conductor`** (dist: `syv-conductor`) — the engine: the node contract, the type mechanism (`DType`, `Series`, `Ref`), compile, execute, widgets, errors, a YAML flow format. It ships every mechanism and no vocabulary: which concrete types exist is the host's decision.
- **`conductor-nodes`** (dist: `syv-conductor-nodes`) — standard-library nodes. `conductor_nodes.types` declares the four types the catalog takes (`Text`, `Number`, `Flag`, `Json`) and `StdlibNode`, the base that pins `category` to the package's `Category` literal. Each category module exposes `NODES` and `register(registry)`; `register_all(registry, categories=...)` registers everything or a subset. Node ids are category-prefixed (`text-uppercase`, `math-add`, …).
- **`conductor-providers`** (dist: `syv-conductor-providers`) — framework adapters. `conductor_providers.react` ships `graph_to_react` / `react_to_graph` / `palette_from_registry`; `conductor_providers.fastapi` ships `conductor_router`. New providers go in sibling subpackages — no abstract base class to satisfy.

Tag-driven publishing: pushing a `v*` tag fires `.github/workflows/publish.yml`, which builds wheels + sdists and uploads all three to PyPI (`PYPI_API_TOKEN`, idempotent via `skip-existing`).

## Tech stack

- Python 3.12+, uv workspace monorepo
- pydantic (only hard dependency of conductor core)
- pytest + pytest-asyncio for tests
- pre-commit + nbstripout for clean notebook diffs
- ruff for linting (config in root pyproject.toml, `uvx ruff check .`); PR-triggered CI in `.github/workflows/ci.yml`

## Key commands

```bash
uv sync                           # Install all deps
uv run pre-commit install         # Activate the nbstripout hook on your clone
uv run pytest tests/ -q           # Run the whole suite (core + nodes + providers + stress)
uvx ruff check .                  # Lint (what CI runs on PRs)
uv run python -m conductor.about  # Print the packaged library reference (llms.txt)
uv run python -m conductor.about sections   # List reference sections
uv run pytest tests/test_nodes/test_catalog.py -v   # Run one file
uv run jupyter lab examples/      # Open the example notebooks
```

Slash command: `/docs-audit` — runs a docs review against the last N commits and edits the user-facing docs in place (no commits). Expected hygiene after feature-bearing sessions. See also `.github/workflows/docs-audit.yml` for the weekly CI safety net.

## Architecture

Three phases: `declare → compile → execute`.

1. **Declare** — a node is a `NodeDefinition` subclass. `__init_subclass__` checks `id`, `title`, `description`, `category` and derives one `NodeVersion` per `@version` method (an undecorated `run` is version 1) by reading the signature once with `Interface.of`. `NodeRegistry.register(cls)` files the class under its id and checks the catalogue rules (versions numbered from 1 with no holes, a deprecated current version pointing somewhere, an `alternative` that exists).
2. **Compile** — `compile(flow, registry)` validates node types, that every wire names an existing node, and cycles (over `dependencies_of(flow.nodes)`), and asks each placement's `compute_inputs` / `compute_outputs` for its roster. Returns an immutable `CompiledGraph`. Every definition the flow names must be in the registry; a host that loads one calls `registry.extended_with(...)` first.
3. **Execute** — `execute(compiled)` is an async generator yielding `ExecutionEvent`s. Nodes are scheduled eagerly: as soon as all dependencies complete, a node's task is created — independent branches run concurrently. A call is validated through `model_of(roster)` and dispatched through `runner_for(registry, type, version)`, a fresh instance per call. `execute_sync()` is a blocking wrapper.

### The node contract

```python
class Upper(NodeDefinition):
    id = "upper"                 # the registry id; a placement stores it, so changing it is a data migration
    title = "Upper case"         # what a palette shows; a placement copies title/description and may edit its copy
    description = "Upper-cases a text."
    category = "text"            # where the palette files it; required, a plain string

    def run(self, text: Annotated[Text, Textarea(title="Text")]) -> Annotated[Text, Result(title="Result")]:
        return Text(text.upper())
```

- Every parameter with a handle declares a `DType` (or `Any`) and a widget in `Annotated[...]`; `title`, `description` and `show_handle` are written on the widget and lifted onto the `Input`. A default makes the input optional. `show_handle=False` closes an input to cables; it may then declare any pydantic-validatable type.
- The return annotation is the output declaration: a `DType` with a `Result` is one output named `result`; a frozen dataclass of `Annotated[DType, Result(...)]` fields is one output per field, names = field names; `Mapping[str, Any]` means the placement's computed roster names the outputs.
- `Any` is for a value the node routes without reading; an `Any` output requires a `compute_outputs` override (refused at definition otherwise).
- `Series[X]` is the one collection. A `Series[X]` parameter receives the whole series; a series output is returned as a plain list.
- `SKIPPED` is a value a `run` returns on a branch not taken; outputs that are exclusive alternatives share a `choice`. There is no role, flag or marker on the class that tells the engine what to do.
- `Provided()` marks a parameter the host supplies by type rather than an input; `Interface.needs` lists them.
- Two optional hooks, `compute_inputs(declared, values)` and `compute_outputs(declared, values, arriving)`, are the only home for placement-specific shape. Nothing checks their answers.

### Types

`DType` is a real class, usually on a builtin (`class Text(DType, str)`), registered by id on definition. `target.accepts(source)` is the one wiring question (default `issubclass`; a series is judged by its element). A `DType` does not convert, does not pick a widget, does not format beyond `as_text`. `describe()` is `{"id", "accepted_as"}`; `Series[X].describe()` nests its element. `Ref("node.field")` is the address of one field on one node — a `str` subclass, split only in `node_id` / `field`.

### Data flow

Each input of a placement holds at most one binding (`GraphNode.bindings`):

1. **`Sources(refs=(...))`** — the value arrives from other placements' outputs; `refs` is in operand order and several refs into a `Series[X]` input gather into one series. `InputResolver` reads the outputs by name.
2. **`Static(value=...)`** — the author typed the value in. `GraphNode.data` is the typed-in values as a dict.
3. **No binding** — the parameter's default.

There is no per-cable record: a canvas derives its cables from the bindings, and `dependencies_of(nodes)` derives what each node waits for.

The call is validated through the placement's roster with pydantic (`extra="ignore"`, so stray data keys are dropped), and `run` receives instances of the declared dtypes.

### Widgets

Every control is a frozen, keyword-only dataclass with a `kind` discriminator; `AnyWidget` is the union built from `Widget.__subclasses__()`, so an `Input` dumps its widget with a schema per kind. The set: `Text`, `Textarea`, `TemplateTextarea`, `CodeEditor`, `Dropdown`, `EntityDropdown`, `Number`, `Range`, `Switch`, `DatePicker`, `FileUpload`, `List`, `Tags`, `TableInput`, `SchemaBuilder`, `IfElseBuilder`, `ConnectionList`. Conductor ships no default widget for any type — every input declares its own. Vocabulary inside a control (`Dropdown.choices`, `IfElseBuilder.operators`, `TableInput.column_types`) is the host's, as data. Full guide: [`docs/widgets.md`](docs/widgets.md); demo: [`examples/08_widgets.ipynb`](examples/08_widgets.ipynb).

### Eager parallel execution

The engine uses a dependency-driven scheduler (`_run_eager` in `execution/engine.py`):
- Each schedulable node tracks an in-degree counter (unfinished deps from its `Sources` bindings).
- When in-degree hits 0, `asyncio.create_task` dispatches the node via `asyncio.to_thread` so sync `run` methods don't block the loop.
- Node events flow through an `asyncio.Queue`; the main loop yields them to the caller.
- Failures cancel all running tasks.

### Retry and timeout

Retries live on the version's `Policy` (`retries`, `delay`, `timeout`, `concurrency`), read by the engine; a run-level `RetryConfig(max_retries, delay, backoff_factor)` applies to nodes whose policy sets none.
- Delay formula: `delay * backoff_factor ** (attempt - 1)`.
- `NodeValidationError` is **never** retried (bad input won't fix itself).
- `NodeConnectionError` / `NodeExecutionError` are retried.
- `Policy.timeout` wraps one attempt in `asyncio.wait_for`; expiry is `NodeTimeoutError`.
- Each retry emits a `node_retry` event with `{attempt, max_retries, error, delay}`.

### Error hierarchy

All exceptions inherit from `ConductorError` (see `errors.py`):

- `CompilationError` — graph structure invalid
  - `CycleDetectionError`
- `NodeError` — carries `node_id`, `node_type`, `original`
  - `NodeValidationError` (pydantic failure, never retried; renders one line per failed field)
  - `NodeExecutionError` (`run` raised)
  - `NodeTimeoutError`
  - `NodeConnectionError` (raise from node code for transient network/API failures)
- `InputResolutionError` — could not resolve inputs from the wires
- `FlowExecutionError` — raised by `execute_sync` when the flow fails

### The persisted graph

`Flow` is `nodes` and `display`. A `GraphNode` is behaviour (`type`, `version`, `bindings`, `locked`), content (`title`, `description`, one `FieldContent` per field) and chrome (`display`, stored and returned, never parsed). An id refuses `.` (a `Ref` reads `node.field`) and accepts `/`. What the flow takes and returns is derived, never stored: `graph/views.py`'s `derive_interface(flow, rosters, versions)` returns an `Interface` whose inputs are the unlocked handle-bearing inputs of the input nodes and whose outputs are every output of the nodes nothing consumes, each named by its address. A failed node fails the run; there is no saga.

### YAML / JSON flow format (`conductor.flow_format`)

The record is the schema: the module wraps `TypeAdapter(Flow)` — `load_flow` / `flow_to_dict`, and YAML/JSON files via `yaml_to_flow` / `flow_to_yaml` / `load_flow_from_path` / `dump_flow`. A ref stores as its address, `"node.field"`. Requires PyYAML (optional extra: `syv-conductor[yaml]`).

### Documentation maintenance

Docs drift is a real failure mode for this project — the whole point of `CLAUDE.md` and `packages/conductor/src/conductor/about/llms.txt` is that future agent sessions can land with full context. That only works if the docs stay in sync with the code.

1. **On-demand: `/docs-audit` slash command** (`.claude/commands/docs-audit.md`). Run it at the end of any session that touched public API, added a feature, or changed default behaviour. It reads the last N commits (default 10; pass a number or `since-release`), compares against the docs, and edits them in place. It does **not** commit — the user reviews via `git diff` and decides.
2. **Weekly safety net: `.github/workflows/docs-audit.yml`**. Every Monday (and on manual `workflow_dispatch`), CI runs the same audit over the last 14 days of commits and opens a PR if anything is out of sync. Needs `ANTHROPIC_API_KEY` as a repo secret.

When the audit flags a discrepancy it can't resolve (commit says X, code does Y), trust the code and surface the discrepancy in the summary — don't write docs for things that don't exist.

## Patterns

### Declaring and registering a node
```python
from typing import Annotated
from conductor import NodeDefinition, NodeRegistry, Result
from conductor.widgets import Textarea
from conductor_nodes.types import Text     # or a DType of your own

class MyNode(NodeDefinition):
    id = "my-node"
    title = "My Node"
    description = "Does stuff"
    category = "text"

    def run(self, text: Annotated[Text, Textarea(title="Input")]) -> Annotated[Text, Result(title="Result")]:
        return Text(text.upper())

registry = NodeRegistry()
registry.register(MyNode)
```

### Composing registries
```python
import conductor_nodes

registry = NodeRegistry()
conductor_nodes.register_all(registry, categories=["text", "math"])
registry.register(MyNode)     # ids are unique; registering a second class under one id raises
```

### Building and running a flow
```python
compiled = compile(Flow(nodes=[GraphNode("n1", "my-node", 1, bindings={"text": Static(value="hello")})]), registry)
results = execute_sync(compiled)     # results["n1"]["result"] == "HELLO"
```

### A second version, with a policy and an upgrade
```python
class MyNode(NodeDefinition):
    ...
    @version(1)
    def run_v1(self, text: Annotated[Text, Textarea(title="Input")]) -> Annotated[Text, Result(title="Result")]: ...

    @version(2, policy=Policy(retries=3, delay=0.5))
    def run(self, text: Annotated[Text, Textarea(title="Input")], loud: Annotated[Flag, Switch(title="Loud")] = Flag(False)) -> Annotated[Text, Result(title="Result")]: ...

    @upgrade(1, 2)
    def _add_loud(values: dict) -> dict:
        return {**values, "loud": False}
```

### The palette
```python
palette = [cls.describe() for cls in registry.definitions()]     # NodeDescription records; dump through pydantic for JSON
```

## Conventions

- A placement is `GraphNode(id, type, version, bindings)`; `type` is the node id and `version` the pinned version. There is no `"id@version"` string anywhere, and no edge record: a wire is a `Sources` on the target's input.
- A result is `results[node_id][output_name]`; a single output is named `result`.
- `SKIPPED` propagates — if every wired value is `SKIPPED`, the node is skipped.
- A `run` returns values of its declared dtypes (`Text(...)`, never a bare `str`), because a value arrives downstream as the type the wire carried.
- Identifiers are English; the host's language lives in titles, descriptions and messages.
- No `__all__` in a module: a reader imports a name from the module that defines it. A package `__init__` may declare one for its re-exports.
- Fail loud: no defensive `None` checks or silent defaults where the state means a bug.
- Streaming (async generator) is the only execution path; sync is a wrapper. Eager scheduling is the default and only mode.
- Notebook outputs are stripped on commit by `nbstripout` — run cells locally to see values.
- `packages/conductor/src/conductor/about/llms.txt` ships inside the package, so installing the wheel is enough for `python -m conductor.about` to work.
- After any session that adds or changes public surface area, run `/docs-audit` to keep `CLAUDE.md`, `README.md`, `llms.txt` and `docs/index.md` in sync; weekly CI catches what the slash command misses.
