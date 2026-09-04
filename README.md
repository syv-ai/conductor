<p align="center">
  <img src="logo-white-background.png" alt="Conductor logo" width="140">
</p>

<h1 align="center">Conductor</h1>

<p align="center">
  A reusable, host-agnostic graph execution engine for building DAG-based workflow systems. Declare a node as a class whose typed <code>run</code> signature is its interface, compile placements of it into a validated execution plan, and run the plan with streaming events.
</p>

Built to be the shared core behind visual flow builders — declare a node once and get validation, execution and the palette a frontend renders from that one declaration.

> Need a short tour to share with a colleague? See [`docs/OVERVIEW.md`](docs/OVERVIEW.md) for a one-page architecture summary.

## Features

- **One node contract** — a node is a `NodeDefinition` subclass; the typed signature of its `run` method *is* its interface. Nothing is declared twice.
- **A type vocabulary you own** — every value on a wire has a `DType`; conductor ships the mechanism and no vocabulary (except `Series[X]`, the one collection). A host declares `Text`, `Number`, `Document`, … as it sees fit.
- **Widgets on the declaration** — `Annotated[Text, Textarea(title="Text")]` says how a person edits an input; the same record drives validation and the palette.
- **Versions with a policy** — several versions live in one class (`@version(2)`); each carries a `Policy` for retries, timeout and concurrency; `@upgrade(1, 2)` rewrites saved values; `@deprecated` retires a node or a version.
- **Compile-then-execute** — structural errors are caught before any node runs.
- **Eager parallel scheduling** — nodes start as soon as their dependencies finish; independent branches run concurrently.
- **Retry** — per version on its `Policy`, or a run-level `RetryConfig`.
- **Structured error hierarchy** — `NodeValidationError`, `NodeExecutionError`, `NodeConnectionError`, `NodeTimeoutError`, and more, all carrying `node_id`/`node_type` context.
- **Streaming execution** — an async generator yields events (`node_start`, `node_complete`, `node_retry`, `flow_complete`, …).
- **Branching by value** — a node returns `SKIPPED` on the branch it did not take; outputs that are exclusive alternatives share a `choice`.
- **Roster hooks** — a node whose inputs or outputs depend on its configuration overrides `compute_inputs` / `compute_outputs`.
- **Shared references** — per-placement produce/consume bindings let one node feed another without an edge.
- **Compensation** — a placement names the node that undoes its work if the flow fails later.
- **Auto-discovery** — import a package and every node it registers is in the registry.
- **YAML / JSON flow format** — `conductor.flow_format` round-trips `Flow` ↔ YAML/JSON/dict.
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
pip install syv-conductor                # core engine — import as `conductor`
pip install syv-conductor-nodes          # standard node library — import as `conductor_nodes`
pip install syv-conductor-providers      # framework adapters — import as `conductor_providers`
pip install "syv-conductor[yaml]"        # optional: YAML/JSON flow format
```

The PyPI distribution names are prefixed with `syv-`; Python imports are unchanged.

For local development (uv workspace):

```bash
git clone <repo-url> conductor
cd conductor
uv sync
uv run pre-commit install   # strip notebook outputs on commit
```

### Run tests

```bash
uv run pytest tests/ -v
```

## Usage

### 1. Declare a vocabulary and some nodes

A value on a wire has a `DType`. Conductor declares none, so start by naming the types your nodes take — or import the standard library's (`conductor_nodes.types`), as this example does.

```python
from typing import Annotated
from conductor import NodeDefinition, NodeRegistry, Result
from conductor.widgets import Textarea, Text as TextWidget
from conductor_nodes.types import Text

class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "Returns the input unchanged"
    category = "text"

    def run(
        self, text: Annotated[Text, Textarea(title="Input", description="Text to echo")]
    ) -> Annotated[Text, Result(title="Output")]:
        return text

class Uppercase(NodeDefinition):
    id = "uppercase"
    title = "Uppercase"
    description = "Converts to uppercase"
    category = "text"

    def run(
        self, text: Annotated[Text, TextWidget(title="Input")]
    ) -> Annotated[Text, Result(title="Result")]:
        return Text(text.upper())

registry = NodeRegistry()
registry.register(Echo)
registry.register(Uppercase)
```

The class is checked the moment it is defined: a missing `id`, `title`, `description` or `category`, a parameter without a widget, or a return without a `Result` fails at import with the traceback at the class.

### 2. Build and execute a flow

A placement pins a node by `type` and `version`:

```python
from conductor import GraphNode, GraphEdge, compile
from conductor.execution.engine import execute_sync

compiled = compile(
    nodes=[
        GraphNode("n1", "echo", 1, {"text": "hello world"}),
        GraphNode("n2", "uppercase", 1),
    ],
    edges=[
        GraphEdge("e1", "n1", "n2", "result", "text"),
    ],
    registry=registry,
)

results = execute_sync(compiled)
print(results["n2"]["result"])  # "HELLO WORLD"
```

A single-output node's output is named `result`; a multi-output node's outputs are the field names of the record it returns (below).

### 3. Stream execution events

```python
from conductor.execution.engine import execute

async for event in execute(compiled):
    match event["type"]:
        case "node_start":
            print(f"Starting {event['node_id']}")
        case "node_complete":
            print(f"Done {event['node_id']}: {event['result']}")
        case "node_retry":
            print(f"Retry {event['node_id']} ({event['attempt']}/{event['max_retries']}): {event['error']}")
        case "flow_complete":
            print(f"Flow done: {event['results']}")
```

## Project structure

```
conductor/
├── packages/
│   ├── conductor/                  # Core library — pip install syv-conductor
│   │   └── src/conductor/
│   │       ├── node.py             # NodeDefinition, NodeVersion, Policy, version/upgrade/deprecated, describe()
│   │       ├── interface.py        # Interface.of(run): the signature read once; Provided; model_of
│   │       ├── metadata.py         # Field, Input, Output records
│   │       ├── returns.py          # Result — what an author writes on a return; outputs_of / unpack
│   │       ├── dtype.py            # DType — a value's type; accepts(); registered_dtypes()
│   │       ├── series.py           # Series[X] and Index — many values of one type
│   │       ├── ref.py              # Ref — the address "<node id>.<field>"
│   │       ├── widgets.py          # The controls: Text, Textarea, Dropdown, …; AnyWidget
│   │       ├── errors.py           # Exception hierarchy (ConductorError, NodeError, …)
│   │       ├── _sentinel.py        # SKIPPED
│   │       ├── registry/           # NodeRegistry, runner_for, discover_nodes
│   │       ├── graph/              # GraphNode/GraphEdge/Flow, topology, compile(), roster resolution
│   │       ├── execution/          # execute(), execute_sync(), the eager scheduler, retry, events
│   │       ├── flow_format/        # YAML / JSON flow files
│   │       └── about/              # Runnable library reference: python -m conductor.about
│   ├── conductor-nodes/            # Standard node library — pip install syv-conductor-nodes
│   └── conductor-providers/        # Framework adapters (react, fastapi) — pip install syv-conductor-providers
├── examples/                       # Jupyter notebooks
├── tests/                          # pytest suite (core, nodes, providers, stress)
├── .github/workflows/              # ci.yml (PR lint + test), docs-audit.yml (weekly)
└── docs/                           # MkDocs site + design notes (llms.txt ships inside the package)
```

## Concepts

### The node contract

A node is a class. It declares its identity and what a palette shows (`id`, `title`, `description`, `category`, optional `tags` and `docs`) and implements `run`, whose typed signature is its interface:

```python
from dataclasses import dataclass
from conductor import NodeDefinition, Result, Series
from conductor.widgets import List, Switch, Textarea
from conductor_nodes.types import Flag, Number, Text

class Length(NodeDefinition):
    id = "length"
    title = "Length"
    description = "Character count of the text"
    category = "text"

    def run(self, text: Annotated[Text, Textarea(title="Text")]) -> Annotated[Number, Result(title="Length")]:
        return Number(len(text))
```

- Every parameter a cable can reach declares a `DType` (or `Any`, below) and a widget. A default makes the input optional.
- The return annotation is the output declaration. A `DType` return declares one output named `result`.
- A node returns a value of the declared type — `Text(...)`, never a bare `str` — because a value arrives downstream as the type the wire carried.
- A collection is `Series[X]`. A node that declares `Series[Text]` receives the whole series at once; a series output is returned as a plain list.

**Several outputs** are a frozen dataclass whose fields are the outputs; `run` returns an instance. The field names are the output names, and nothing is positional:

```python
@dataclass(frozen=True)
class Halves:
    head: Annotated[Text, Result(title="First half")]
    tail: Annotated[Text, Result(title="Second half")]

class Split(NodeDefinition):
    id = "split"
    title = "Split"
    description = "Splits text down the middle"
    category = "text"

    def run(self, text: Annotated[Text, Textarea(title="Text")]) -> Halves:
        mid = len(text) // 2
        return Halves(head=Text(text[:mid]), tail=Text(text[mid:]))
```

**Branching** is a value, not a role. A node that takes one of two branches returns `SKIPPED` on the other; downstream nodes fed only `SKIPPED` are skipped in turn. Outputs that are exclusive alternatives share a `choice`, so an editor knows exactly one of them arrives:

```python
@dataclass(frozen=True)
class Emptiness:
    not_empty: Annotated[Text, Result(title="Not empty", choice="emptiness")]
    empty: Annotated[Text, Result(title="Empty", choice="emptiness")]

class IfEmpty(NodeDefinition):
    id = "if-empty"
    title = "If empty"
    description = "Routes text by whether it is blank"
    category = "control"

    def run(self, text: Annotated[Text, Textarea(title="Text")]) -> Emptiness:
        if text.strip():
            return Emptiness(not_empty=text, empty=SKIPPED)
        return Emptiness(not_empty=SKIPPED, empty=text)
```

**A value the node routes without reading** is annotated `Any` instead of a type. An `Any` output requires the node to override `compute_outputs` so the outputs can be typed from what arrives — the standard library's `decision` gate is the example.

### Types

A `DType` is a real Python class, usually built on a builtin, declared with an `id` (the stable name the persisted graph and the frontend use) and a `title`:

```python
from conductor import DType

class Text(DType, str):
    id = "text"
    title = "Text"

class Number(DType, float):
    id = "number"
    title = "Number"
```

`Text("hello")` is both a `str` and a `Text`; a pydantic model with a `Text` field gives back a `Text`. A type answers one question about wiring — `target.accepts(source)`: may a value of type `source` land on an input declared as `target`? The default is `issubclass`, so a subtype is accepted wherever its parent is. A `DType` does not convert values, does not pick a widget and does not format itself beyond `as_text`. `registered_dtypes()` lists every type declared so far; `describe()` on a type is its JSON-ready record.

`Series[X]` is the one collection: many values of one type on an `Index`, which says where the rows came from. Two series align when they share an index, never by length. `Series[Series[X]]` does not exist.

### Versions

Several versions live in one class as methods marked `@version(n)`; the current one is the method named `run`. Each version has its own signature and `Policy`. `@upgrade(1, 2)` marks the function that rewrites values saved against version 1 into what version 2 expects; `@deprecated` marks a class or a version as going away, optionally naming an `alternative`:

```python
from conductor import Policy, deprecated, upgrade, version

class Greet(NodeDefinition):
    id = "greet"
    title = "Greet"
    description = "Greets a person"
    category = "text"

    @version(1)
    @deprecated(header="Use version 2", migration="The name is now first and last")
    def run_v1(self, name: Annotated[Text, TextWidget(title="Name")]) -> Annotated[Text, Result(title="Greeting")]:
        return Text(f"Hi, {name}!")

    @version(2, policy=Policy(retries=2, delay=0.5))
    def run(
        self,
        first: Annotated[Text, TextWidget(title="First name")],
        last: Annotated[Text, TextWidget(title="Last name")],
    ) -> Annotated[Text, Result(title="Greeting")]:
        return Text(f"Hi, {first} {last}!")

    @upgrade(1, 2)
    def _split_name(values: dict) -> dict:
        first, _, last = values["name"].partition(" ")
        return {**values, "first": first, "last": last}
```

A registered node numbers its versions from 1 with no holes; a placement pins any version up to the current one. `Policy` carries `retries`, `delay`, `timeout` (seconds) and `concurrency`.

### The registry

`NodeRegistry` maps a node id to the class itself — one entry per id, not per version:

```python
registry.register(Greet)
registry.get("greet")                      # the class, or None
registry.contains("greet")                 # True
registry.definitions()                     # every class, in registration order
Greet.versions[2].interface.inputs         # the Input records of version 2
Greet.describe()                           # the palette entry, derived on demand
registry.upgrade_path("greet", 1, 2)       # the @upgrade function, or None
```

`describe()` is the one serialisation of a node: a `NodeDescription` with its versions, fields, policy and deprecation notice, dumped through pydantic when a palette needs JSON. Nothing is stored, so a description is always derived from the live declaration.

**Auto-discovery** imports every module in a package so the registrations in them run:

```python
from conductor.registry.discovery import discover_nodes

discover_nodes("myapp.nodes", registry)    # returns how many definitions were added
```

### Roster hooks

Two optional methods let a node say what one *placement* of it has, when that depends on configuration:

```python
def compute_inputs(self, declared, values) -> tuple[Input, ...]: ...
def compute_outputs(self, declared, values, arriving) -> tuple[Output, ...]: ...
```

`declared` is the pinned version's declaration, `values` what the author typed, `arriving` the type on each wired input where the compiler has recorded one. The default returns `declared`. The compiler asks a fresh instance once per placement and stores the answers on `CompiledGraph.node_inputs` / `node_outputs`. Nothing is checked here: a hook that returns the wrong shape is a node bug and raises where it is found.

### Provided parameters

A parameter marked `Provided()` is not an input — no widget, no handle — but a value the host supplies by type when it runs the flow:

```python
from conductor import Provided

def run(self, text: Annotated[Text, Textarea(title="Text")], who: Annotated[Identity, Provided()]) -> ...:
```

`Interface.needs` lists such parameters by name so a host knows what a flow requires before starting it.

### Eager parallel execution

The engine schedules nodes eagerly: as soon as all of a node's dependencies finish, its task is dispatched. Independent branches run concurrently without any configuration. Sync `run` methods are offloaded to `asyncio.to_thread`, so they don't block the event loop.

```
  A (0.3s) ──> C (0.3s) ──┐
                           ├──> E (0.3s)
  B (0.3s) ──> D (0.3s) ──┘
```

Sequential would be 5 × 0.3 s = 1.5 s. Eager execution: `A + B` in parallel (0.3 s), `C + D` in parallel (0.3 s), `E` (0.3 s) = ~0.9 s. This is the default and only execution mode.

### Retry

Retries belong to the version, on its `Policy`; a run-level `RetryConfig` applies to every node whose policy sets none:

```python
from conductor import Policy, version
from conductor.execution.retry import RetryConfig

class FetchUrl(NodeDefinition):
    id = "fetch-url"
    title = "Fetch"
    description = "HTTP GET"
    category = "http"

    @version(1, policy=Policy(retries=3, delay=0.5))
    def run(self, url: Annotated[Text, TextWidget(title="URL")]) -> Annotated[Text, Result(title="Body")]:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Text(resp.text)

results = execute_sync(compiled, retry=RetryConfig(max_retries=2, delay=1.0, backoff_factor=2.0))
```

Delay between attempts is `delay * backoff_factor ** (attempt - 1)`. `NodeExecutionError` and `NodeConnectionError` are retried; `NodeValidationError` never is — pydantic rejected the inputs, and retrying with the same inputs is pointless. Each retry emits a `node_retry` event with `{attempt, max_retries, error, delay}`. `Policy(timeout=...)` bounds one attempt and raises `NodeTimeoutError` on expiry.

### Error types

All exceptions inherit from `ConductorError` and are importable from `conductor.errors`. Node-level errors carry `node_id`, `node_type` and the `original` exception:

```
ConductorError                     # Base — catch-all for any engine error
├── CompilationError                # Graph structure is invalid
│   └── CycleDetectionError         # Graph contains a cycle
├── NodeError                       # Something went wrong with a specific node
│   ├── NodeValidationError         # Input validation failed (pydantic) — never retried
│   ├── NodeExecutionError          # run() raised — retried if the policy says so
│   ├── NodeTimeoutError            # Node exceeded its policy's timeout
│   └── NodeConnectionError         # External service / network failure inside a node
├── InputResolutionError            # Could not resolve inputs from edges
└── FlowExecutionError              # Flow-level failure (raised by execute_sync)
```

Raise `NodeConnectionError` from `run` to mark a failure as transient and retry-worthy.

### Shared references

A placement can bind one of its inputs to another placement's output without an edge. A producer marks an output as shared in `produces`; a consumer names it in `consumes`. Reference identity is `(producer node id, output name)`; the label is for the UI only:

```python
compiled = compile(
    nodes=[
        GraphNode("mapper", "build-map", 1, {"seed": "x"}, produces={"result": "pseudonym map"}),
        GraphNode("redactor", "redact", 1, {"text": "Alice met Bob."}, consumes={"mapping": ("mapper", "result")}),
    ],
    edges=[],
    registry=registry,
)
```

Resolver precedence, first match wins: an edge into the input, a consume binding, static data on the placement (`GraphNode.data`), the parameter's default.

### Compensation

A placement may name the node that undoes its work: `GraphNode(..., compensation="refund")`. When the flow fails, the engine walks the completed nodes in reverse order and runs each one's compensation with the original inputs and output, emitting `compensation_start` / `compensation_complete` / `compensation_failed`. A placement's `on_error` (`"fail"`, `"continue"`, `"compensate"`) decides what its own failure triggers.

### YAML / JSON flow format

`conductor.flow_format` round-trips a `Flow` (nodes, edges, id, version, name, description) to and from a dict, YAML or a file: `load_flow`, `flow_to_dict`, `yaml_to_flow`, `flow_to_yaml`, `load_flow_from_path`, `dump_flow`. Requires PyYAML (`syv-conductor[yaml]`).

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
| `ConnectionList` | Edited by wiring only — a `Series[X]` or `Any` input | — |

Conductor ships no default widget for any type: `Text` may be a textarea, a single line or a dropdown, so every input declares its own. An input closes itself to cables with `show_handle=False` on its widget, and may then declare any pydantic-validatable type. The set of controls is closed — `AnyWidget` is built from the subclasses in `widgets.py`, and a new control is a change there, since the component that renders each `kind` has to exist in the host's frontend anyway.

**Full widget guide:** [`docs/widgets.md`](docs/widgets.md). Hands-on tour: [`examples/08_widgets.ipynb`](examples/08_widgets.ipynb).

## Execution events

The `execute()` async generator yields these events:

| Event | When |
|-------|------|
| `node_start` | Node begins execution |
| `node_complete` | Node finished (includes result) |
| `node_skipped` | Node skipped (all inputs SKIPPED) |
| `node_error` | Node raised an unretryable (or final) exception |
| `node_retry` | Node failed and will be retried (includes attempt, max_retries, error, delay) |
| `runtime_warning` | The engine noticed something worth surfacing without failing |
| `compensation_start` / `compensation_complete` / `compensation_failed` | The compensation cascade after a failure |
| `flow_complete` | All nodes done (includes all results) |
| `flow_error` | Unrecoverable error |
| `flow_timeout` | Execution exceeded `timeout_seconds` |
| `flow_cancelled` | Execution was cancelled |

## Using in other projects

### AI context (llms.txt)

An AI-readable library reference lives inside the package at `packages/conductor/src/conductor/about/llms.txt` and ships as package data in the wheel:

```bash
python -m conductor.about                 # full reference
python -m conductor.about sections        # list section slugs
python -m conductor.about retry           # one section (prefix match)
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

A workspace sibling to `conductor` that ships common nodes so downstream flows don't have to re-author them. Distributed on PyPI as `syv-conductor-nodes`; the Python import path is `conductor_nodes`. Pick the categories you want:

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
| `decision` | `decision` — routes any value to one of two branches on a wired-in `Flag` |

Node ids are category-prefixed to avoid colliding with application-level ids. Registering two different classes under one id raises.

## Frontend providers (`conductor-providers`)

Framework adapters. Each provider is a subpackage translating between conductor's Python objects and the framework's wire format. Distributed on PyPI as `syv-conductor-providers`; the Python import path is `conductor_providers`.

```python
from conductor_providers import react

palette = react.palette_from_registry(registry)   # [cls.describe() for every definition]
flow_json = react.graph_to_react(nodes, edges)    # GraphNode/GraphEdge → ReactFlow JSON (positions auto-assigned if omitted)
nodes2, edges2 = react.react_to_graph(flow_json)  # ReactFlow JSON → GraphNode/GraphEdge
```

`conductor_providers.fastapi.conductor_router(registry)` returns an APIRouter with `GET /nodes` (the palette), `POST /compile`, `POST /execute`, `POST /execute-stream` (server-sent events) and `GET /entities/{kind}` for `EntityDropdown` choices.

New providers (Svelte, Vue, Gradio, …) go in sibling subpackages under `conductor_providers.` — no abstract base class to satisfy; each provider picks the shape that matches its framework.

## Examples

The examples are Jupyter notebooks under `examples/` — open them in VS Code, JupyterLab, or any notebook UI and run the cells interactively.

| Notebook | What it covers |
|----------|---------------|
| `01_basic_nodes.ipynb` | Declaring nodes: widgets, defaults, multi-output records, inspecting a registry |
| `02_build_and_run_flow.ipynb` | Placements and edges, collecting results, streaming events |
| `03_class_nodes_and_store.ipynb` | A node with its own methods; `Provided` parameters |
| `05_auto_discovery.ipynb` | Package scanning, versions and deprecation, the palette as JSON |
| `08_widgets.ipynb` | Every control, inspecting a widget's schema |

```bash
uv sync                       # includes the ipykernel used by the notebooks
uv run jupyter lab examples/  # or open the .ipynb files in VS Code
```

The notebooks use `await collect(execute(compiled))` because the kernel already owns an event loop. From a plain `.py` script, use `execute_sync(compiled)` instead.

## Stability and versioning

From `1.0.0` onward, conductor follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). This is the contract host applications can rely on:

**Public API.** A name is part of the public API if it is exported from a package's `__init__` or documented in this README / `docs/`. Anything else — `_`-prefixed names, modules not re-exported from a public surface — is internal and may change in any release without warning. The public surface:

- Top-level `conductor`: the node contract (`NodeDefinition`, `NodeVersion`, `GraphVersion`, `Policy`, `Deprecation`, `NodeDescription`, `version`, `upgrade`, `deprecated`, `Interface`, `Provided`, `Input`, `Output`, `AnyWidget`), the type vocabulary (`DType`, `DTypeRef`, `Single`, `dtype_of`, `registered_dtypes`, `Series`, `Index`, `Ref`, `Result`), the registry (`NodeRegistry`, `runner_for`), the graph (`GraphNode`, `GraphEdge`, `Flow`, `compile`, `CompiledGraph`, `resolve_graph_inputs`, `resolve_graph_outputs`), execution (`execute`, `execute_sync`, `RetryConfig`, `SKIPPED`) and the error classes
- `conductor.widgets`, `conductor.metadata`, `conductor.errors`, `conductor.execution.events` (the `*Event` `TypedDict`s), `conductor.registry.discovery` (`discover_nodes`), `conductor.flow_format`
- `conductor_nodes` (`register_all`, `get_default_registry`, the category modules, `conductor_nodes.types`) and `conductor_providers.react` / `conductor_providers.fastapi`

**Compatibility guarantees from `1.0.0`.**

- *No breaking changes without a major bump.* If a `1.x` release removes a public name, changes a public signature in a way that breaks callers, or alters documented behaviour, the version that ships that change is `2.0.0` (or later).
- *Deprecation policy.* When a public name is scheduled for removal it stays live for **at least one minor release** after deprecation, with a `DeprecationWarning` raised at import or call time. The `CHANGELOG.md` entry that introduces the deprecation lists the target removal version.
- *Internal modules are fair game.* Anything not listed above may be renamed, restructured or removed in any release.
- *The three workspace packages release in lockstep.* `syv-conductor`, `syv-conductor-nodes`, `syv-conductor-providers` share a single version. The `syv-conductor[nodes]` / `[providers]` / `[all]` extras pin sibling packages with `==` to prevent resolver skew.

The `CHANGELOG.md` carries the full history, including the removals that lead up to the next major.

## License

Apache-2.0. See [`LICENSE`](LICENSE) at the repo root; each PyPI wheel ships the same license file.
