# Conductor

Reusable DAG execution engine extracted from production node editors. One node contract (a class whose typed `run` signature is its interface), a type vocabulary the host declares, graph compilation, and a streaming engine that runs a node once per row of a series.

## Repository structure

```
conductor/
├── packages/conductor/        # Core library
│   └── src/conductor/
│       ├── node.py             # NodeDefinition ABC, NodeVersion/GraphVersion, Policy, version/upgrade/deprecated, describe(), Refuses
│       ├── interface.py        # Interface.of(run) — the signature read once; FromRun; model_of
│       ├── metadata.py         # Field, Input, Output records
│       ├── model.py            # ConductorModel — the base of every saved or sent record: to_yaml / from_yaml / to_path / from_path
│       ├── returns.py          # Result (what an author writes on a return); outputs_of / unpack
│       ├── dtype.py            # DType — a value's type; accepts(); Single; dtype_of()
│       ├── dtype_ref.py        # DTypeRef — a dtype as a pydantic field
│       ├── series.py           # Series[X], Index, Row
│       ├── ref.py              # Ref — "<node id>.<field>"
│       ├── widgets.py          # Widget + the controls; AnyWidget discriminated union
│       ├── errors.py           # ErrorCause and the exception hierarchy
│       ├── _sentinel.py        # SKIPPED and Asks — the two values a run returns that are not results
│       ├── registry/           # NodeRegistry (register, nodes, runner_for, extended_with, upgraded); discover_nodes
│       ├── graph/              # model (Graph/GraphNode), binding (Edges/Static), compiler + compiled (CompiledGraph and its node/field views), iteration (the edge walk), expand (embedded graphs), conditions, problem, topology, views
│       ├── execution/          # engine (execute, run, run_sync; one leg per call), ledger (what a run produced, and what that makes ready), events
│       └── about/              # Runnable library context: `python -m conductor.about`
├── packages/conductor-nodes/   # Standard node library (text, math, logic, json_ops, regex_ops, decision) + its types
│   └── src/conductor_nodes/    # Each module exposes register(registry); top-level register_all()
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

- **`conductor`** (dist: `syv-conductor`) — the engine: the node contract, the type mechanism (`DType`, `Series`, `Ref`), compile, execute, widgets, errors, records that save themselves as JSON or YAML. It ships every mechanism and no vocabulary: which concrete types exist is the host's decision.
- **`conductor-nodes`** (dist: `syv-conductor-nodes`) — standard-library nodes. `conductor_nodes.types` declares the four types the catalog takes (`Text`, `Number`, `Flag`, `Json`) and `StdlibNode`, the base that pins `category` to the package's `Category` literal. Each category module exposes `register(registry)`, which lists its nodes; `register_all(registry, categories=...)` registers everything or a subset. Node ids are category-prefixed (`text-uppercase`, `math-add`, …).
- **`conductor-providers`** (dist: `syv-conductor-providers`) — framework adapters. `conductor_providers.react` ships `graph_to_react` / `react_to_graph` / `palette_from_registry`; `conductor_providers.fastapi` ships `conductor_router`. New providers go in sibling subpackages — no abstract base class to satisfy.

Tag-driven publishing: pushing a `v*` tag fires `.github/workflows/publish.yml`, which builds wheels + sdists and uploads all three to PyPI (`PYPI_API_TOKEN`, idempotent via `skip-existing`).

## Tech stack

- Python 3.12+, uv workspace monorepo
- pydantic and PyYAML (the two dependencies of conductor core)
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

Tests marked `slow` time the wall clock; they are skipped when `CI` is set.

Slash command: `/docs-audit` — runs a docs review against the last N commits and edits the user-facing docs in place (no commits). Expected hygiene after feature-bearing sessions. See also `.github/workflows/docs-audit.yml` for the weekly CI safety net.

## Architecture

Three phases: `declare → compile → execute`.

1. **Declare** — a node is a `NodeDefinition` subclass. `__init_subclass__` checks `id`, `title`, `description`, `category`, derives one `NodeVersion` per `@version` method (an undecorated `run` is version 1; beside `@version` methods, `run` must carry one too) by reading the signature once with `Interface.of`, and collects its `@upgrade` rewrites into `cls.upgrades`. `NodeRegistry.register(cls)` files the class under its id and checks the catalogue rules (versions numbered from 1 with no holes, a deprecated current version pointing somewhere, an `alternative` that exists).
2. **Compile** — `CompiledGraph.from_graph(graph, registry)` resolves each node's pin, asks `compute_inputs` on the typed statics, validates the stored bindings, expands an embedded graph under its node's name, walks the edges once in order — typing every unconstrained field from what arrives, asking `accepts`, making a node fed a series run once per row of it, asking `compute_outputs` with what arrives — and derives the condition under which each output appears. Returns an immutable `CompiledGraph` that callers ask at three scales — the graph (`problems`, `is_runnable`, `interface`, `execution_order`, `decisions`), one node (`node(id)`: `interface`, `call_model`, `iterates_on`, `statics`, `runner`, `embedded_in`) and one field (`field(ref)`: `type`, `index`, `binding`, `receives`, `condition`); everything wrong with the graph is an anchored `Problem` on it, never an exception. Every definition the graph names must be in the registry; a host that loads one calls `registry.extended_with(...)` first.
3. **Execute** — `execute(compiled)` is an async generator yielding `ExecutionEvent`s. Its unit of work is `(node, row)`: a node that runs once is one unit, a node that runs per row is one unit per row. The `Ledger` holds every value the run has produced, cell by cell, and answers each write with the units it made ready; the engine starts those under each node's `Policy.concurrency`. A call is validated through `model_of(interface.inputs)` and made on a fresh instance (`compiled.node(id).runner`). One call of `execute` is one **leg**: it runs until nothing is runnable and nothing is in flight. `await run(compiled)` drains it and returns the ending event; `run_sync(compiled)` is the same call for a script, and refuses under a running loop.

### The node contract

```python
class Upper(NodeDefinition):
    id = "upper"                 # the registry id; a placement stores it, so changing it is a data migration
    title = "Upper case"         # what a palette shows; a placement copies title/description and may edit its copy
    description = "Upper-cases a text."
    category = "text"            # where the palette files it; required, a plain string

    def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())]) -> Annotated[Text, Result(title="Result")]:
        return Text(text.upper())
```

- Every parameter with a handle declares a `DType` (or `Any`) and a widget in `Annotated[...]`; `title`, `description` and `show_handle` are written on the widget and lifted onto the `Input`. A default makes the input optional. `show_handle=False` closes an input to edges; it may then declare any pydantic-validatable type.
- The return annotation is the output declaration: a `DType` with a `Result` is one output named `result`; a frozen dataclass of `Annotated[DType, Result(...)]` fields is one output per field, names = field names; `Mapping[str, Any]` means the placed node's computed interface names the outputs.
- `Any` is for a value the node routes without reading; an `Any` output requires a `compute_outputs` override (refused at definition otherwise).
- `Series[X]` is the one collection. A `Series[X]` parameter receives the whole series (a reduction); a series arriving on a scalar parameter makes the node run once per row. A series output is returned as a plain list.
- `SKIPPED` is a value a `run` returns on a branch not taken; outputs that are exclusive alternatives share a `choice`. `Asks(questions)` is a value a `run` returns when a person must answer; the return annotation says so, `-> X | Asks`. There is no role, flag or marker on the class that tells the engine what to do.
- `Annotated[T, FromRun()]` marks a parameter the host supplies by type through `execute(from_run={T: value})` rather than an input; `Interface.needs` lists them, and a run that lacks one refuses to start.
- Two optional hooks, `compute_inputs(declared, values)` and `compute_outputs(declared, values, arriving)`, are the only home for placement-specific shape; a hook that cannot answer raises `Refuses(code, message)`, which compile reports as the node's problem.

### Types

`DType` is a real class, usually on a builtin (`class Text(DType, str)`), declaring its own `id` and `title` — declaring one records nothing anywhere, and a subclass that leaves `id` to its parent is refused. `target.accepts(source)` is the one edge question (default `issubclass`; a series is judged by its element). A `DType` does not convert, does not pick a widget, does not format beyond `as_text`. `describe()` is `{"id"}`; `Series[X].describe()` nests its element. Which types exist is the registry's: the words its nodes declare plus `registry.add_types(...)`, read through `registry.types`; `registry.accepted_as(T)` is where a `T` may land over that vocabulary, and `registry.describe()` is the palette — every node's record and one `TypeDescription` per type. `Ref("node.field")` is the address of one field on one node — a `str` subclass, split only in `node_id` / `field`. An `Index` names where a series' rows come from and is compared by id; a row is a path, `(i,)` on a root index and `(i, j)` under parent row `(i,)`.

### Bindings

Each input of a placement holds at most one binding (`GraphNode.bindings`):

1. **`Edges(refs=(...))`** — the value arrives from other placements' outputs; `refs` is in operand order. Several refs into a `Series[X]` input on different indexes gather into one series; several on one index are a union, one value per row.
2. **`Static(value=...)`** — the author typed the value in. A list typed into a scalar input makes the node run once per value.
3. **No binding** — the parameter's default.

There is no per-edge record: a canvas derives its edges from the bindings, and `dependencies_of(nodes)` derives what each node waits for.

The call is validated through the placed node's own interface with pydantic (`extra="ignore"`, so stray data keys are dropped), and `run` receives instances of the declared dtypes.

### Widgets

Every control is a frozen pydantic model with a `kind` discriminator; `AnyWidget` is the union built from `Widget.__subclasses__()`, so an `Input` dumps its widget with a schema per kind. The set: `TextWidget`, `Textarea`, `TemplateTextarea`, `CodeEditor`, `Dropdown`, `EntityDropdown`, `Number`, `Range`, `Switch`, `DatePicker`, `FileUpload`, `List`, `Tags`, `TableInput`, `SchemaBuilder`, `IfElseBuilder`. Conductor ships no default widget for any type — every input declares its own. Vocabulary inside a control (`Dropdown.choices`, `IfElseBuilder.operators`, `TableInput.column_types`) is the host's, as data. Full guide: [`docs/widgets.md`](docs/widgets.md); demo: [`examples/08_widgets.ipynb`](examples/08_widgets.ipynb).

### Rows, skips and legs

- **Iteration and reduction.** A node iterating on an index runs once per row, concurrently up to its policy's `concurrency`; row 1 can finish a whole chain while row 10 is still being produced. A `Series[X]` input on a root index receives the series once it is complete; on a child index, once per parent row.
- **A skip has a depth.** `SKIPPED` at a row skips that row (the series downstream is sparse); `SKIPPED` above a node's rows skips everything under it. A node whose every row was skipped produced an empty series; a node skipped above its rows did not run and is absent from the results.
- **A failed unit fails the run.** No further unit starts (a `run` already in its thread finishes, its result dropped), and the `ErrorCause`, with the row, goes out on `node_error` and `graph_error`.
- **A leg ends pending.** A unit whose node returned `Asks` waits; everything else runs on, and the leg ends `graph_pending` with every waiting unit's questions. The next leg is `execute` again with `record=` (the ledger's record from the last ending) and the answers in `cache=` as the asking node's outputs — for a node on rows, a `Series` naming only the rows it answers; a unit already done or a row not yet produced is refused. Every ending — `graph_complete`, `graph_pending`, `graph_error`, `graph_cancelled`, `graph_timeout` — carries `results` and `record`.
- **Readiness is kept, not recomputed.** A write asks `ready` only of the units it concerns, so a run's time grows with its rows; `Ledger.runnable()` asks every unit at a leg's start and again when it goes quiet, where a ready unit nobody started raises.

### Retry and timeout

Retries live on the version's `Policy` (`retries`, `delay`, `timeout`, `concurrency`, `retry_on`) and nowhere else; each unit retries on its own.
- Two families. A failure is internal unless the outside world caused it. Only an `ExternalFailure` is retried: a node raises one itself, or names its client's exception classes in `Policy(retry_on=(httpx.TransportError, ...))` and the engine wraps a matching foreign exception as one (code `external_failed`). Every other exception from `run` is wrapped as `NodeExecutionError` (code `execution_failed`) and runs once; `NodeValidationError` and a timeout are never retried.
- Delay formula: `delay * 2 ** (attempt - 1)`; each retry emits a `node_retry` event with `{row, attempt, retries, error, delay}`.
- `Policy.timeout` is how long the leg waits on one attempt, counted from the moment the node's thread starts: the leg owns a thread pool with one worker per unit that may be in flight, so no unit ever waits for a worker. It never interrupts the thread. A timed-out attempt is final (`NodeTimeoutError`, code `timeout`); the thread finishes on its own, keeps the node's concurrency slot until it does, and what it returns is dropped. The timeout worth retrying is the client's own, set on the client inside `run`: when the client gives up, the thread has returned and a retry runs nothing twice.
- A `run` that holds the GIL — a regex that never finishes, a tight loop over a huge input — blocks the whole process, and nothing in the engine can stop it. Where legs run, in the API process or in a worker of their own, is the host's decision.
- `execute(timeout=None)` by default; `execute(timeout=60)` bounds the whole leg in seconds and ends it `graph_timeout`; `execute(cancel=event)` stops it with `graph_cancelled` the moment the event is set. Closing the stream — `aclose()`, or cancelling the task that reads it — stops every unit; a bare `break` leaves the generator open until it is closed or collected, so close it.
- What people read: a cause the engine writes carries the generic message for its code (`conductor.errors.MESSAGES`); a `NodeError` the node raised keeps the message the node chose. A foreign exception's text is on the wrapping error's `original` and on no event.

### Error hierarchy

All exceptions inherit from `ConductorError` (see `errors.py`). A run-time failure carries an `ErrorCause` — `code`, `message`, `details`, `row`:

- `CompilationError` — a run was started on a graph compile found not runnable; carries `problems`
- `NodeError` — carries `node_id`, `original` and `cause`; the internal family, never retried
  - `ExternalFailure` (the outside world failed; the one family the engine retries — raise it from `run`, or name the client's classes in `retry_on`)
  - `NodeValidationError` (pydantic failure; renders one line per failed field)
  - `NodeExecutionError` (`run` raised something that is not a `NodeError`; the exception is on `original`)
  - `NodeTimeoutError` (the leg stopped waiting; final)

### The persisted graph

`Graph` is `nodes` and `display`. A `GraphNode` is behaviour (`type`, `version`, `bindings`, `locked`), content (`title`, `description`, one `FieldContent` per field) and chrome (`display`, stored and returned, never parsed). An id refuses `.` (a `Ref` reads `node.field`) and, at compile, `/` (compile names an embedded graph's nodes with it). What the graph takes and returns is derived, never stored: `CompiledGraph.interface` is an `Interface` whose inputs are the unlocked handle-bearing inputs of the input nodes and whose outputs are every output of the nodes nothing consumes, each named by its address.

### Saving a graph

`Graph`, and every record a host saves or sends (`GraphNode`, `Edges`, `Static`, `Problem`, `ErrorCause`, `Policy`, `NodeDescription`, `Input`, `Output`, the widgets, `Index`), is a `ConductorModel`: a frozen pydantic model. JSON is pydantic's own (`model_dump_json` / `model_validate_json`); `to_yaml` / `from_yaml` and `to_path` / `from_path` save and load YAML or JSON by suffix. A saved record reads back what it wrote; what describes a node (`NodeDescription`, `VersionDescription`, `Input`, `Output`, the widgets) is written for an editor and not read back — its `dtype` dumps as a description and a widget's title travels on its `Input`. A ref stores as its address, `"node.field"`. What compile and the engine build per call (`CompiledGraph` and its views, versions, `Interface`, the ledger's records) stays a frozen dataclass.

### Documentation maintenance

Docs drift is a real failure mode for this project — the whole point of `AGENTS.md` and `packages/conductor/src/conductor/about/llms.txt` is that future agent sessions can land with full context. That only works if the docs stay in sync with the code.

1. **On-demand: `/docs-audit` slash command** (`.claude/commands/docs-audit.md`). Run it at the end of any session that touched public API, added a feature, or changed default behaviour. It reads the last N commits (default 10; pass a number or `since-release`), compares against the docs, and edits them in place. It does **not** commit — the user reviews via `git diff` and decides.
2. **Weekly safety net: `.github/workflows/docs-audit.yml`**. Every Monday (and on manual `workflow_dispatch`), CI runs the same audit over the last 14 days of commits and opens a PR if anything is out of sync. Needs `ANTHROPIC_API_KEY` as a repo secret.

When the audit flags a discrepancy it can't resolve (commit says X, code does Y), trust the code and surface the discrepancy in the summary — don't write docs for things that don't exist.

## Patterns

### Declaring and registering a node
```python
from typing import Annotated
from conductor import NodeDefinition, NodeRegistry, Param, Result, run_sync
from conductor.widgets import Textarea
from conductor_nodes.types import Text     # or a DType of your own

class MyNode(NodeDefinition):
    id = "my-node"
    title = "My Node"
    description = "Does stuff"
    category = "text"

    def run(self, text: Annotated[Text, Param(title="Input", widget=Textarea())]) -> Annotated[Text, Result(title="Result")]:
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

### Building and running a graph
```python

compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n1", type="my-node", version=1, bindings={"text": Static(value="hello")})]), registry)
results = run_sync(compiled)["results"]     # results["n1"]["result"] == "HELLO"
```

### A node that asks, and the next leg
```python
class Approve(NodeDefinition):
    ...
    def run(self, proposal: Annotated[Text, Param(title="Proposal", widget=Textarea())]) -> Annotated[Text, Result(title="Decision")] | Asks:
        return Asks(questions=(Input(name="result", dtype=Text, title="Decision", widget=Textarea(), default=proposal, optional=True),))

paused = run_sync(compiled)                                   # paused["type"] == "graph_pending"; paused["pending"]: the questions, by address
results = run_sync(compiled, record=paused["record"], cache={"approve": {"result": Text("yes")}})["results"]
```

### A second version, with a policy and an upgrade
```python
class MyNode(NodeDefinition):
    ...
    @version(1)
    def run_v1(self, text: Annotated[Text, Param(title="Input", widget=Textarea())]) -> Annotated[Text, Result(title="Result")]: ...

    @version(2, policy=Policy(retries=3, delay=0.5))
    def run(self, text: Annotated[Text, Param(title="Input", widget=Textarea())], loud: Annotated[Flag, Param(title="Loud", widget=Switch())] = Flag(False)) -> Annotated[Text, Result(title="Result")]: ...

    @upgrade(1, 2)
    def _add_loud(values: dict) -> dict:
        return {**values, "loud": False}
```

### The palette
```python
palette = [cls.describe() for cls in registry.nodes]     # NodeDescription records; dump through pydantic for JSON
```

## Conventions

- A placement is `GraphNode(id=, type=, version=, bindings=)`; `type` is the node id and `version` the pinned version. There is no `"id@version"` string anywhere, and no edge record: an edge is an `Edges` on the target's input.
- A result is `results[node_id][output_name]`; a single output is named `result`.
- `SKIPPED` propagates by depth — a reader finds it at its own row or above and skips that far; a gather drops skipped sources.
- A `run` returns values of its declared dtypes (`Text(...)`, never a bare `str`), because a value arrives downstream as the type the edge carried.
- Identifiers are English; the host's language lives in titles, descriptions and messages.
- No `__all__` in a module: a reader imports a name from the module that defines it. A package `__init__` may declare one for its re-exports.
- Fail loud: no defensive `None` checks or silent defaults where the state means a bug.
- Streaming (async generator) is the only execution path; sync is a wrapper. Eager scheduling is the default and only mode.
- Notebook outputs are stripped on commit by `nbstripout` — run cells locally to see values.
- `packages/conductor/src/conductor/about/llms.txt` ships inside the package, so installing the wheel is enough for `python -m conductor.about` to work.
- After any session that adds or changes public surface area, run `/docs-audit` to keep `AGENTS.md`, `README.md`, `llms.txt` and `docs/index.md` in sync; weekly CI catches what the slash command misses.
