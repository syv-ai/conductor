<p align="center">
  <img src="logo-white-background.png" alt="Conductor logo" width="140">
</p>

<h1 align="center">Conductor</h1>

<p align="center">
  A reusable, host-agnostic graph execution engine for building DAG-based workflow systems. Declare a node as a class whose typed <code>run</code> signature is its interface, compile placements of it into a validated execution plan, and run the plan with streaming events.
</p>

Built to be the shared core behind visual workflow builders — declare a node once and get validation, execution and the palette a frontend renders from that one declaration.

> Need a short tour to share with a colleague? See [`docs/OVERVIEW.md`](docs/OVERVIEW.md) for a one-page architecture summary.

## Features

- **One node contract** — a node is a `NodeDefinition` subclass; the typed signature of its `run` method *is* its interface. Nothing is declared twice.
- **A type vocabulary you own** — every value on an edge has a `DType`; conductor ships the mechanism and no vocabulary (except `Series[X]`, the one collection). A host declares `Text`, `Number`, `Document`, … as it sees fit.
- **Widgets on the declaration** — `Annotated[Text, Textarea(title="Text")]` says how a person edits an input; the same record drives validation and the palette.
- **Versions with a policy** — several versions live in one class (`@version(2)`); each carries a `Policy` for retries, timeout and concurrency; `@upgrade(1, 2)` rewrites saved values; `@deprecated` retires a node or a version.
- **Compile-then-execute** — everything wrong with a graph is an anchored `Problem` before any node runs.
- **Rows** — a series arriving on a scalar input runs the node once per row, concurrently under its policy; a `Series[X]` input receives the whole series or a group per parent row. A run's time grows with its rows.
- **Retry per version** — on its `Policy`, and nowhere else.
- **Structured failures** — every run-time failure carries an `ErrorCause` (`code`, `message`, `details`, `row`), on the exception and on the event.
- **Streaming execution** — an async generator yields events (`node_start`, `node_progress`, `node_complete`, `node_retry`, `graph_complete`, …).
- **Branching by value** — a node returns `SKIPPED` on the branch it did not take; outputs that are exclusive alternatives share a `choice`.
- **A person in the loop** — a node returns `Asks` with its questions; the run ends pending, and the next call to `execute` carries the answers.
- **Field hooks** — a node whose inputs or outputs depend on its configuration overrides `compute_inputs` / `compute_outputs`.
- **Embedded graphs** — a version whose body is a graph expands under its node's name and runs as nodes of the one run.
- **Auto-discovery** — import a package and every node it registers is in the registry.
- **Records that save themselves** — `Graph.from_path("approval.yaml")`, `graph.to_yaml()`, and pydantic's own JSON.
- **Zero app dependencies** — no FastAPI, no database, no auth in the core; pydantic is the one hard dependency.
- **Standard node library** — `conductor-nodes` ships text, math, logic, JSON and regex nodes and a decision gate, declared in a four-word vocabulary of its own.
- **Framework adapters** — `conductor_providers.react` translates graphs to/from ReactFlow JSON and builds the palette; `conductor_providers.fastapi` mounts `/nodes`, `/compile`, `/execute` and `/execute-stream`.

## Quick start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Install

From PyPI (Apache-2.0):

```bash
uv add syv-conductor                # core engine — import as `conductor`
uv add syv-conductor-nodes          # standard node library — import as `conductor_nodes`
uv add syv-conductor-providers      # framework adapters — import as `conductor_providers`
A `Graph` is a pydantic model and saves itself — and so does every record a host keeps or sends (`Problem`, `ErrorCause`, `Input`, `NodeDescription`, …):

```python
graph.to_path("approval.yaml")            # .json, .yaml or .yml, by suffix
graph = Graph.from_path("approval.yaml")
graph.to_yaml(); Graph.from_yaml(text)
graph.model_dump_json(); Graph.model_validate_json(text)
```

A ref stores as its address, `"node.field"`.

## Widgets

A widget is the control an input is edited with, plus what that control needs. Every widget is a frozen, keyword-only record with a `kind` discriminator; `AnyWidget` is the union of all of them, so pydantic dumps a widget and publishes a JSON schema per kind. `title`, `description` and `show_handle` are written on the widget but belong to the field: `Interface.of` copies them onto the `Input`.

| Widget | Best for | Key options |
|--------|----------|-------------|
| `Text` | Single-line string | `min_length`, `max_length`, `pattern` |
| `Textarea` | Multi-line string | `rows`, `min_length`, `max_length` |
| `TemplateTextarea` | Text with placeholders, each an input | `rows` |
| `CodeEditor` | Source a person writes | `language`, `min_length`, `max_length` |
| `Dropdown` | Pick one of a declared vocabulary of `Choice`s | `choices` |
| `EntityDropdown` | Choices the host resolves | `entity_kind`, `multiple` |
| `Number` | A number typed in | `min_val`, `max_val`, `step`, `integer_only` |
| `Range` | A number on a slider | `min_val`, `max_val`, `step` |
| `Switch` | A boolean | — |
| `DatePicker` | A date from a calendar | `min_date`, `max_date`, `seed` |
| `FileUpload` | Files a person uploads | `accept`, `max_size_mb`, `multiple` |
| `List` | A list of values typed by hand | `min_items`, `max_items` |
| `Tags` | Free-form labels | — |
| `TableInput` | A table typed or pasted in | `min_rows`, `min_columns`, `column_types` |
| `SchemaBuilder` | A schema built field by field | `schema`, `allow_additional`, `field_types` |
| `IfElseBuilder` | Conditions built from the host's operators | `operators` |
| `ConnectionList` | Edited by connecting only — a `Series[X]` or `Any` input | — |

Conductor ships no default widget for any type: `Text` may be a textarea, a single line or a dropdown, so every input declares its own. An input closes itself to edges with `show_handle=False` on its widget, and may then declare any pydantic-validatable type. The set of controls is closed — `AnyWidget` is built from the subclasses in `widgets.py`, and a new control is a change there, since the component that renders each `kind` has to exist in the host's frontend anyway.

**Full widget guide:** [`docs/widgets.md`](docs/widgets.md). Hands-on tour: [`examples/08_widgets.ipynb`](examples/08_widgets.ipynb).

## Execution events

The `execute()` async generator yields these events:

| Event | When |
|-------|------|
| `node_start` | A node's first unit begins |
| `node_progress` | A row of a node running per row finished (`done`, and `total` once every row exists) |
| `node_complete` | Every unit of a node is done (includes its result; a result on rows is a `Series`) |
| `node_skipped` | A node was skipped above its rows and did not run |
| `node_error` | A unit failed for good (includes the `ErrorCause`) |
| `node_retry` | A unit failed and will be retried (includes row, attempt, retries, error, delay) |
| `graph_complete` | Everything ran |
| `graph_pending` | The run is waiting for a person (includes every question) |
| `graph_error` | A unit failed, so the run stopped (includes the cause) |
| `graph_timeout` | The leg exceeded `timeout_seconds` |
| `graph_cancelled` | The `cancel` event was set |

Every `graph_*` ending carries `results` and `cells`.

## Using in other projects

### AI context (llms.txt)

An AI-readable library reference lives inside the package at `packages/conductor/src/conductor/about/llms.txt` and ships as package data in the wheel:

```bash
python -m conductor.about                 # full reference
python -m conductor.about sections        # list section slugs
python -m conductor.about rows            # one section (prefix match)
```

### Keeping docs in sync

- **`/docs-audit` Claude Code slash command** — run it at the end of a session that added public API or changed default behaviour. It diffs the last N commits against `CLAUDE.md`, `README.md`, `llms.txt` and `docs/index.md`, and applies edits in place. Does not commit; you review the diff.
- **Weekly CI audit** — `.github/workflows/docs-audit.yml` runs the same audit every Monday and opens a PR if anything drifted. Requires `ANTHROPIC_API_KEY` as a repo secret.

### Documentation

For full documentation, we recommend [MkDocs Material](https://squidfunk.github.io/mkdocs-material/):

```bash
uv add --group docs mkdocs-material mkdocstrings[python]
uv run mkdocs serve      # Local preview at http://localhost:8000
uv run mkdocs gh-deploy  # Deploy to GitHub Pages
```

## Standard node library (`conductor-nodes`)

A workspace sibling to `conductor` that ships common nodes so downstream graphs don't have to re-author them. Distributed on PyPI as `syv-conductor-nodes`; the Python import path is `conductor_nodes`. Pick the categories you want:

```python
from conductor import NodeRegistry
from conductor_nodes import register_all, get_default_registry, text, math

reg = NodeRegistry()
register_all(reg)                                   # everything
register_all(reg, categories=["text", "math"])      # a subset
text.register(reg)                                  # or per module
reg = get_default_registry()                        # a fresh registry holding everything
```

The library declares the four types its nodes take in `conductor_nodes.types` — `Text`, `Number`, `Flag`, `Json` — because a node library has to say what its nodes take, and conductor itself ships no vocabulary. A host with its own vocabulary declares its own types and does not need these.

| Module | Node ids |
|---|---|
| `text` | `text-uppercase`, `text-lowercase`, `text-trim`, `text-length`, `text-concat`, `text-replace`, `text-contains`, `text-split`, `text-join`, `text-reverse` |
| `math` | `math-add`, `math-subtract`, `math-multiply`, `math-divide`, `math-modulo`, `math-round`, `math-min`, `math-max`, `math-abs` |
| `logic` | `logic-if-empty`, `logic-if-equals`, `logic-not` (the two `if` nodes branch via `SKIPPED`) |
| `json_ops` | `json-parse`, `json-stringify`, `json-get` (dotted path) |
| `regex_ops` | `regex-match`, `regex-replace`, `regex-extract` |
| `decision` | `decision` — routes any value to one of two branches on a `Flag` connected in |

Node ids are category-prefixed to avoid colliding with application-level ids. Registering two different classes under one id raises.

## Frontend providers (`conductor-providers`)

Framework adapters. Each provider is a subpackage translating between conductor's Python objects and the framework's wire format. Distributed on PyPI as `syv-conductor-providers`; the Python import path is `conductor_providers`.

```python
from conductor_providers import react

palette = react.palette_from_registry(registry)   # [cls.describe() for every definition]
wire = react.graph_to_react(graph)                # Graph → ReactFlow JSON (the placement record under each node's data; edges derived; positions laid out if a placement has none)
graph2 = react.react_to_graph(wire)               # ReactFlow JSON → Graph
```

`conductor_providers.fastapi.conductor_router(registry)` returns an APIRouter with `GET /nodes` (the palette), `POST /compile`, `POST /execute`, `POST /execute-stream` (server-sent events) and `GET /entities/{kind}` for `EntityDropdown` choices. Its `from_run` hook turns a request into the values `execute(from_run=...)` supplies.

New providers (Svelte, Vue, Gradio, …) go in sibling subpackages under `conductor_providers.` — no abstract base class to satisfy; each provider picks the shape that matches its framework.

## Examples

The examples are Jupyter notebooks under `examples/` — open them in VS Code, JupyterLab, or any notebook UI and run the cells interactively.

| Notebook | What it covers |
|----------|---------------|
| `01_basic_nodes.ipynb` | Declaring nodes: widgets, defaults, multi-output records, inspecting a registry |
| `02_build_and_run_flow.ipynb` | Placements and edges, collecting results, streaming events |
| `03_class_nodes_and_store.ipynb` | A node with its own methods |
| `05_auto_discovery.ipynb` | Package scanning, versions and deprecation, the palette as JSON |
| `06_human_in_the_loop.ipynb` | A node that asks, the run ending pending, and the next leg with the answer |
| `08_widgets.ipynb` | Every control, inspecting a widget's schema |

```bash
uv sync                       # includes the ipykernel used by the notebooks
uv run jupyter lab examples/  # or open the .ipynb files in VS Code
```

The notebooks use `await collect(execute(compiled))` because the kernel already owns an event loop. From a plain `.py` script, use `execute_sync(compiled)` instead.

## Stability and versioning

From `1.0.0` onward, conductor follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). This is the contract host applications can rely on:

**Public API.** A name is part of the public API if it is exported from a package's `__init__` or documented in this README / `docs/`. Anything else — `_`-prefixed names, modules not re-exported from a public surface — is internal and may change in any release without warning. The public surface:

- Top-level `conductor`: the node contract (`NodeDefinition`, `NodeVersion`, `GraphVersion`, `Policy`, `Deprecation`, `NodeDescription`, `version`, `upgrade`, `deprecated`, `Interface`, `FromRun`, `Input`, `Output`, `AnyWidget`, `SKIPPED`, `Asks`, `is_skipped`, `is_asking`), the type vocabulary (`DType`, `DTypeRef`, `Single`, `dtype_of`, `registered_dtypes`, `Series`, `Index`, `Ref`, `Result`), the registry (`NodeRegistry`), and the graph (`Graph`, `GraphNode`, `FieldContent`, `Binding`, `Edges`, `Static`, `dependencies_of`, `is_input_node`, `CompiledGraph`, `CompiledNode`, `CompiledField`, `Problem`, `Condition`, `Atom`, `ALWAYS`)
- `conductor.execution.engine` (`execute`, `execute_sync`, `collect`), `conductor.errors` (`ErrorCause` and the error classes), `conductor.model` (`ConductorModel`), `conductor.widgets`, `conductor.metadata`, `conductor.execution.events` (the `*Event` `TypedDict`s), `conductor.registry.discovery` (`discover_nodes`)
- `conductor_nodes` (`register_all`, `get_default_registry`, the category modules, `conductor_nodes.types`) and `conductor_providers.react` / `conductor_providers.fastapi`

**Compatibility guarantees from `1.0.0`.**

- *No breaking changes without a major bump.* If a `1.x` release removes a public name, changes a public signature in a way that breaks callers, or alters documented behaviour, the version that ships that change is `2.0.0` (or later).
- *Deprecation policy.* When a public name is scheduled for removal it stays live for **at least one minor release** after deprecation, with a `DeprecationWarning` raised at import or call time. The `CHANGELOG.md` entry that introduces the deprecation lists the target removal version.
- *Internal modules are fair game.* Anything not listed above may be renamed, restructured or removed in any release.
- *The three workspace packages release in lockstep.* `syv-conductor`, `syv-conductor-nodes`, `syv-conductor-providers` share a single version. The `syv-conductor[nodes]` / `[providers]` / `[all]` extras pin sibling packages with `==` to prevent resolver skew.

The `CHANGELOG.md` carries the full history, including the removals that lead up to the next major.

## License

Apache-2.0. See [`LICENSE`](LICENSE) at the repo root; each PyPI wheel ships the same license file.
