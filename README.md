<p align="center">
  <img src="logo-white-background.png" alt="Conductor logo" width="140">
</p>

<h1 align="center">Conductor</h1>

<p align="center">
  A reusable, host-agnostic engine that compiles and runs graphs of typed nodes. Declare a node as a class whose typed <code>run</code> signature is its interface, compile placements of it into a validated execution plan, and run the plan with streaming events.
</p>

Built to be the shared core behind visual node editors — declare a node once and get validation, execution and the palette a frontend renders from that one declaration.

> Need a short tour to share with a colleague? See [`docs/OVERVIEW.md`](docs/OVERVIEW.md) for a one-page architecture summary.

## Features

- **One node contract** — a node is a `NodeDefinition` subclass; the typed signature of its `run` method *is* its interface. Nothing is declared twice.
- **A type vocabulary you own** — every value on an edge has a `DType`; conductor ships the mechanism and no vocabulary (except `Series[X]`, the one collection). A host declares `Text`, `Number`, `Document`, … as it sees fit.
- **Widgets on the declaration** — `Annotated[Text, Param(title="Text", widget=Textarea())]` says how a person edits an input; the same record drives validation and the palette.
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
- **One way to wire a package's nodes** — each module exposes `register(registry)`, and the host calls it.
- **Records that save themselves** — `Graph.from_path("approval.yaml")`, `graph.to_yaml()`, and pydantic's own JSON.
- **Zero host dependencies** — no FastAPI, no database, no auth in the core; pydantic is the one hard dependency.
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

A value on an edge has a `DType`. Conductor declares none, so start by naming the types your nodes take — or import the standard library's (`conductor_nodes.types`), as this example does.

```python
from typing import Annotated
from conductor import Asks, CompiledGraph, deprecated, DType, From, FromRun, Graph, GraphNode, Input, NodeDefinition, NodeRegistry, Param, Policy, Ref, Result, run_sync, Series, SKIPPED, Static, upgrade, version
from conductor.widgets import Textarea, TextWidget
from conductor_nodes.types import Text

class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "Returns the input unchanged"
    category = "text"

    def run(
        self, text: Annotated[Text, Param(title="Input", description="Text to echo", widget=Textarea())]
    ) -> Annotated[Text, Result(title="Output")]:
        return text

class Uppercase(NodeDefinition):
    id = "uppercase"
    title = "Uppercase"
    description = "Converts to uppercase"
    category = "text"

    def run(
        self, text: Annotated[Text, Param(title="Input", widget=TextWidget())]
    ) -> Annotated[Text, Result(title="Result")]:
        return Text(text.upper())

registry = NodeRegistry()
registry.register(Echo)
registry.register(Uppercase)
```

The class is checked the moment it is defined: a missing `id`, `title`, `description` or `category`, a parameter without a widget, or a return without a `Result` fails at import with the traceback at the class.

### 2. Build and execute a graph

A placement pins a node by `type` and `version` and says, per input, where its value comes from: an `From` binding names other placements' outputs (an edge), a `Static` binding holds a typed-in value, and an input with no binding takes its declared default. There is no edge list — a graph is its nodes.

```python

graph = Graph(nodes=[
    GraphNode(id="n1", type="echo", version=1, bindings={"text": Static("hello world")}),
    GraphNode(id="n2", type="uppercase", version=1, bindings={"text": From(Ref("n1", "result"))}),
])
compiled = CompiledGraph.from_graph(graph, registry)

results = run_sync(compiled)["results"]
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
        case "node_progress":
            print(f"{event['node_id']}: {event['done']} of {event['total']}")
        case "node_retry":
            print(f"Retry {event['node_id']} ({event['attempt']}/{event['retries']}): {event['error']}")
        case "graph_complete":
            print(f"Done: {event['results']}")
```

A stream you leave early — a `break`, an exception — should be closed, or its units run on until the generator is collected: `async with aclosing(execute(compiled)) as events:` (from `contextlib`) closes it however the block ends, and closing it stops every unit.

## Project structure

```
conductor/
├── packages/
│   ├── conductor/                  # Core library — uv add syv-conductor
│   │   └── src/conductor/
│   │       ├── node.py             # NodeDefinition, NodeVersion, Policy, version/upgrade/deprecated, describe()
│   │       ├── interface.py        # Interface.of(run): the signature read once; FromRun; model_of
│   │       ├── metadata.py         # Field, Input, Output records
│   │       ├── model.py            # ConductorModel — records that save themselves
│   │       ├── returns.py          # Result — what an author writes on a return; outputs_of / unpack
│   │       ├── dtype.py            # DType — a value's type; accepts(); Single; dtype_of()
│   │       ├── series.py           # Series[X] and Index — many values of one type
│   │       ├── ref.py              # Ref — the address "<node id>.<field>"
│   │       ├── widgets.py          # The controls: Text, Textarea, Dropdown, …; AnyWidget
│   │       ├── errors.py           # ErrorCause and the exception hierarchy
│   │       ├── _sentinel.py        # SKIPPED and Asks
│   │       ├── registry/           # NodeRegistry
│   │       ├── graph/              # GraphNode/Graph, the Binding variants, CompiledGraph.from_graph() and the CompiledGraph it returns, iteration, expansion, conditions, Problem
│   │       ├── execution/          # execute(), run(), run_sync(), the ledger, events
│   │       └── about/              # Runnable library reference: python -m conductor.about
│   ├── conductor-nodes/            # Standard node library — uv add syv-conductor-nodes
│   └── conductor-providers/        # Framework adapters (react, fastapi) — uv add syv-conductor-providers
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
from typing import Annotated
from conductor.widgets import ListWidget, Switch, Textarea
from conductor_nodes.types import Flag, Number, Text

class Length(NodeDefinition):
    id = "length"
    title = "Length"
    description = "Character count of the text"
    category = "text"

    def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())]) -> Annotated[Number, Result(title="Length")]:
        return Number(len(text))
```

- Every parameter an edge can reach declares a `DType` (or `Any`, below) and a widget. A default makes the input optional.
- The return annotation is the output declaration. A `DType` return declares one output named `result`.
- A node returns a value of the declared type — `Text(...)`, never a bare `str` — because a value arrives downstream as the type the edge carried.
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

    def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())]) -> Halves:
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

    def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())]) -> Emptiness:
        if text.strip():
            return Emptiness(not_empty=text, empty=SKIPPED)
        return Emptiness(not_empty=SKIPPED, empty=text)
```

**A value the node routes without reading** is annotated `Any` instead of a type. An `Any` output requires the node to override `compute_outputs` so the outputs can be typed from what arrives — the standard library's `decision` gate is the example.

### Types

A `DType` is a real Python class, usually built on a builtin, declared with an `id` (the stable name the persisted graph and the frontend use) and a `title`:

```python

class Text(DType, str):
    id = "text"
    title = "Text"

class Number(DType, float):
    id = "number"
    title = "Number"
```

`Text("hello")` is both a `str` and a `Text`; a pydantic model with a `Text` field gives back a `Text`. A type answers one question about edges — `target.accepts(source)`: may a value of type `source` land on an input declared as `target`? The default is `issubclass`, so a subtype is accepted wherever its parent is. A `DType` does not convert values, does not pick a widget and does not format itself beyond `as_text`. `describe()` on a type is its JSON-ready record, `{"id": ...}`. Declaring a type records nothing anywhere — a notebook cell can declare the same `Text` twice — and a subclass names itself: one that leaves `id` to its parent is refused at definition. Which types exist is a registry's decision (below).

`Series[X]` is the one collection: many values of one type on an `Index`, which says where the rows came from. Two series align when they share an index, never by length. `Series[Series[X]]` does not exist.

### Versions

Several versions live in one class as methods marked `@version(n)`, `run` included; the current one is the highest number, and by convention its method is the one named `run`. Each version has its own signature and `Policy`. `@upgrade(1, 2)` marks the function that rewrites values saved against version 1 into what version 2 expects (`inputs=` and `outputs=` name the fields it renames), and a class with several versions declares one step per adjacent pair or is refused when it is defined; `@deprecated` marks a class or a version as going away, optionally naming an `alternative`:

```python

class Greet(NodeDefinition):
    id = "greet"
    title = "Greet"
    description = "Greets a person"
    category = "text"

    @version(1)
    @deprecated(header="Use version 2", migration="The name is now first and last")
    def run_v1(self, name: Annotated[Text, Param(title="Name", widget=TextWidget())]) -> Annotated[Text, Result(title="Greeting")]:
        return Text(f"Hi, {name}!")

    @version(2, policy=Policy(retries=2, delay=0.5))
    def run(
        self,
        first: Annotated[Text, Param(title="First name", widget=TextWidget())],
        last: Annotated[Text, Param(title="Last name", widget=TextWidget())],
    ) -> Annotated[Text, Result(title="Greeting")]:
        return Text(f"Hi, {first} {last}!")

    @upgrade(1, 2)
    def _split_name(values: dict) -> dict:
        first, _, last = values.pop("name").partition(" ")
        return {**values, "first": first, "last": last}
```

A registered node numbers its versions from 1 with no holes; a placement pins any version up to the current one. `Policy` carries `retries`, `delay`, `timeout` (seconds) and `concurrency`.

### The registry

`NodeRegistry` maps a node id to the class itself — one entry per id, not per version — and owns the **vocabulary**: the types its nodes declare on their inputs and outputs, plus what the host adds, each under its id:

```python
registry = NodeRegistry()
registry.register(Greet)                   # files Greet, and Text under "text"
registry.add_types(Number)                 # a word no node here declares
registry["greet"]                          # the class; an unknown id is a KeyError listing the ids
"greet" in registry                        # True; len(registry) and iterating over the ids work too
registry.nodes                             # every class, in registration order
registry.types                             # {"text": Text, "number": Number}
registry.accepted_as(Text)                 # ("text",) — where a Text may land, over this vocabulary
Greet.versions[2].interface.inputs         # the Input records of version 2
Greet.describe()                           # the palette entry, derived on demand
registry.describe()                        # the palette: every node's record and every type's
# registry.upgraded(graph, "greet-1")      # the graph with one node moved to the current version
```

`describe()` is the one serialisation of a node: a `NodeDescription` with its versions, fields, policy and deprecation notice, dumped through pydantic when a palette needs JSON. Nothing is stored, so a description is always derived from the live declaration. `registry.describe()` is the palette: those records plus one `TypeDescription` per type — `id`, `title` and `accepted_as`, the ids of every type in that registry whose `accepts` admits it — so an editor reads where a value may land once per type, and a field's own record is its id. Two registries in one process may each hold a `text`; one registry refuses a second class under an id it already holds, naming both.

A package of nodes gives each module a `register(registry)` that registers the classes it offers, and the host calls them; importing a module registers nothing:

```python
def register(registry: NodeRegistry) -> None:
    registry.register(Greet)

register(registry)                         # the host decides which registry
```

### Field hooks

Two optional methods let a node say what one *placement* of it has, when that depends on configuration:

```python
def compute_inputs(self, declared, values) -> tuple[Input, ...]: ...
def compute_outputs(self, declared, values, arriving) -> tuple[Output, ...]: ...
```

`declared` is the pinned version's declaration, `values` what the author typed, `arriving` the type on each connected input where the compiler has recorded one. The default returns `declared`. The compiler asks a fresh instance once per node — `compute_inputs` on the typed statics, `compute_outputs` in the edges pass with what arrives — and `CompiledGraph.node(node_id).interface` serves the answers. Nothing is checked here: a hook that returns the wrong shape is a node bug and raises where it is found.

### Parameters the run supplies

A parameter marked `FromRun()` is not an input — no widget, no handle — but a value the host supplies by type when it runs the graph:

```python

def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())], clock: Annotated[Clock, FromRun()]) -> ...:

results = run_sync(compiled, from_run={Clock: SystemClock()})["results"]
```

`Interface.needs` lists such parameters by name, and `execute` refuses to start a graph that needs a type it was not given.

### Rows

A node declared for one value runs once per row when a series reaches it. The engine's unit of work is a node on a row, and it starts every unit as soon as what it reads exists, so row 1 can finish a whole chain while row 10 is still being produced; a node's rows run concurrently up to its policy's `concurrency` (8 by default), in threads from a pool the leg owns with one worker per unit that may be in flight, so no unit waits for a worker. A node declaring `Series[X]` receives the series whole — once for a root series, once per parent row for a child one. Independent branches run concurrently without any configuration, and `run` is a plain function the engine calls in a worker thread; an `async def run` is refused when the class is defined.

A skip has a depth: `SKIPPED` at one row leaves the series downstream sparse, and `SKIPPED` above a node's rows skips everything under it. A failed row fails the run, and its `ErrorCause` names the row.

### A person in the loop

A node that needs a person's answer returns `Asks` instead of a result, and says so in its return annotation. Everything else runs on; the run then ends pending with every question, named by address. Answering is simply the next call:

```python

class Approve(NodeDefinition):
    id = "approve"
    title = "Approve"
    description = "Asks a person to approve the proposal"
    category = "review"

    def run(self, proposal: Annotated[Text, Param(title="Proposal", widget=Textarea())]) -> Annotated[Text, Result(title="Decision")] | Asks:
        return Asks(questions=(Input(name="result", dtype=Text, title="Decision", widget=Textarea(), default=proposal, optional=True),))

registry.register(Approve)
compiled = CompiledGraph.from_graph(Graph(nodes=[
    GraphNode(id="approve", type="approve", version=1, bindings={"proposal": Static("Ship it")}),
]), registry)

paused = run_sync(compiled)                                   # paused["type"] == "graph_pending"; paused["pending"]: the questions, by address
results = run_sync(compiled, record=paused["record"], cache={"approve": {"result": Text("Approved")}})["results"]
```

`record` is the engine's `RunRecord` of everything the earlier leg produced, so nothing is done twice, and a node the graph has changed since runs again; `cache` carries the answers as the asking node's outputs; for a node on rows, a `Series` naming the rows it answers, while a row that ran keeps its value and a row left out asks again. Every ending of a run carries its `results` and `record`, so a host can also start a new run from a failed or stopped one.

### Retry

Retries belong to the version, on its `Policy`, and nowhere else:

```python

class FetchUrl(NodeDefinition):
    id = "fetch-url"
    title = "Fetch"
    description = "HTTP GET"
    category = "http"

    @version(1, policy=Policy(retries=3, delay=0.5, retry_on=(requests.ConnectionError, requests.Timeout)))
    def run(self, url: Annotated[Text, Param(title="URL", widget=TextWidget())]) -> Annotated[Text, Result(title="Body")]:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return Text(resp.text)
```

Two families of failure. A node's failure is internal — a bug, bad data, a refused schema — unless the outside world caused it, and only the outside world's is retried: an `ExternalFailure` the node raises, or a foreign exception whose class the policy's `retry_on` names, which the engine wraps as one (code `external_failed`). Every other exception from `run` is wrapped as `NodeExecutionError` (code `execution_failed`) and runs once; `NodeValidationError` never retries either — pydantic rejected the inputs, and retrying with the same inputs is pointless. Delay between attempts is `delay * 2 ** (attempt - 1)`, each row retries on its own, and each retry emits a `node_retry` event with `{row, attempt, retries, error, delay}`.

`Policy.timeout` is how long the leg waits on one attempt, counted from the moment the node's thread starts: the leg owns a thread pool with one worker per unit that may be in flight, so no unit ever waits for a worker. It never interrupts the thread. A timed-out attempt is final (`NodeTimeoutError`, code `timeout`); the thread finishes on its own, keeps the node's concurrency slot until it does, and what it returns is dropped. The timeout worth retrying is the client's own, set on the client inside `run`: when the client gives up, the thread has returned and a retry runs nothing twice. A `run` that holds the GIL — a regex that never finishes, a tight loop over a huge input — blocks the whole process, and nothing in the engine can stop it. Where legs run, in the API process or in a worker of their own, is the host's decision.

What people read: a cause the engine writes carries the generic message for its code (`conductor.errors.MESSAGES`); a `NodeError` the node raised keeps the message the node chose, since the node wrote it for people. A foreign exception's text is on the wrapping error's `original`, for a log, and on no event — a host streams events to a browser.

### Error types

All exceptions inherit from `ConductorError` and are importable from `conductor.errors`. Node-level errors carry `node_id`, the `original` exception and an `ErrorCause` — `code`, `message`, `details` and the `row` a node running per row failed on:

```
ConductorError                     # Base — catch-all for any engine error
├── CompilationError                # A run was started on a graph compile found not runnable; carries its problems
├── NodeError                       # One node failed; the internal family, never retried
│   ├── ExternalFailure             # The outside world failed; the one family the engine retries
│   ├── NodeValidationError         # Input validation failed (pydantic)
│   ├── NodeExecutionError          # run() raised something that is not a NodeError; it is on `original`
│   └── NodeTimeoutError            # The leg stopped waiting for the node; final
```

Raise `ExternalFailure` from `run` where the node knows the outside world failed, or name the client's exception classes in `Policy(retry_on=...)` and let the engine classify them.

### Bindings

One input holds one binding, so an edge and a typed value can never both claim the same input. `From(Ref("a", "result"), Ref("b", "result"))` is in operand order — into a `Series[X]` input several refs gather into one series. `Static(...)` is what the author typed. An absent binding means the declared default applies. A graph's dependencies (`dependencies_of`) and which placements are its input nodes (`is_input_node`, no edge into any input) are read off the bindings; nothing stores them. A failed node fails the run.

A host that loads definitions the static registry lacks builds them and hands compile `registry.extended_with({...})` — a new registry per run in which a registered type wins over a loaded one.

### Saving a graph

A `Graph` is a pydantic model and saves itself — and so does every record a host keeps (`Problem`, `ErrorCause`, `Policy`, `Index`, …):

```python
graph.to_path("approval.yaml")            # .json, .yaml or .yml, by suffix
graph = Graph.from_path("approval.yaml")
text = graph.to_yaml(); Graph.from_yaml(text)
data = graph.model_dump_json(); Graph.model_validate_json(data)
```

A ref stores as its address, `"node.field"`. What describes a node (`NodeDescription`, `Input`, `Output`, the widgets) is a model too, written for an editor and not read back: call `describe()` again rather than loading a palette.

## Widgets

A widget is the control an input is edited with, plus what that control needs. Every widget is a frozen, keyword-only record with a `kind` discriminator; `AnyWidget` is the union of all of them, so pydantic dumps a widget and publishes a JSON schema per kind. `title`, `description` and `show_handle` are written on the widget but belong to the field: `Interface.of` copies them onto the `Input`.

| Widget | Best for | Key options |
|--------|----------|-------------|
| `TextWidget` | Single-line string | `min_length`, `max_length`, `pattern` |
| `Textarea` | Multi-line string | `rows`, `min_length`, `max_length` |
| `TemplateTextarea` | Text with placeholders, each an input | `rows` |
| `CodeEditor` | Source a person writes | `language`, `min_length`, `max_length` |
| `Dropdown` | Pick one of a declared vocabulary of `Choice`s | `choices` |
| `EntityDropdown` | Choices the host resolves | `entity_kind`, `multiple` |
| `NumberWidget` | A number typed in | `min_val`, `max_val`, `step`, `integer_only` |
| `Range` | A number on a slider | `min_val`, `max_val`, `step` |
| `Switch` | A boolean | — |
| `DatePicker` | A date from a calendar | `min_date`, `max_date`, `seed` |
| `FileUpload` | Files a person uploads | `accept`, `max_size_mb`, `multiple` |
| `ListWidget` | A list of values typed by hand | `min_items`, `max_items` |
| `Tags` | Free-form labels | — |
| `TableInput` | A table typed or pasted in | `min_rows`, `min_columns`, `column_types` |
| `SchemaBuilder` | A schema built field by field | `schema`, `allow_additional`, `field_types` |
| `IfElseBuilder` | Conditions built from the host's operators | `operators` |

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
| `graph_timeout` | The leg ran longer than the `timeout` its caller set (carried as `timeout_seconds`) |
| `graph_cancelled` | The `cancel` event was set |

Every `graph_*` ending carries `results` and `record`.

## Using in other projects

### AI context (llms.txt)

An AI-readable library reference lives inside the package at `packages/conductor/src/conductor/about/llms.txt` and ships as package data in the wheel:

```bash
python -m conductor.about                 # full reference
python -m conductor.about sections        # list section slugs
python -m conductor.about rows            # one section (prefix match)
```

### Keeping docs in sync

- **`/docs-audit` Claude Code slash command** — run it at the end of a session that added public API or changed default behaviour. It diffs the last N commits against `AGENTS.md`, `README.md`, `llms.txt` and `docs/index.md`, and applies edits in place. Does not commit; you review the diff.
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
import conductor_nodes

registry = conductor_nodes.registry()                              # a fresh registry holding everything
registry = conductor_nodes.registry(categories=["text", "math"])   # a subset
conductor_nodes.register_all(my_registry, categories=["logic"])    # added to a registry you already have
conductor_nodes.text.register(my_registry)                         # or one module
```

The library declares the four types its nodes take in `conductor_nodes.types` — `Text`, `Number`, `Flag`, `Json` — because a node library has to say what its nodes take, and conductor itself ships no vocabulary. A host with its own vocabulary declares its own types and does not need these.

| Category | Node ids |
|---|---|
| `text` | `text-uppercase`, `text-lowercase`, `text-trim`, `text-length`, `text-concat`, `text-replace`, `text-contains`, `text-split`, `text-join`, `text-reverse` |
| `math` | `math-add`, `math-subtract`, `math-multiply`, `math-divide`, `math-modulo`, `math-round`, `math-min`, `math-max`, `math-abs` |
| `logic` | `logic-not` |
| `control` | `logic-if-empty`, `logic-if-equals` (both branch via `SKIPPED`), `decision` — routes any value to one of two branches on a `Flag` connected in |
| `json` | `json-parse`, `json-stringify`, `json-get` (dotted path) |
| `regex` | `regex-match`, `regex-replace`, `regex-extract` |

`categories=` filters on each node's own `category`, so `["logic"]` is `logic-not` alone; the branching nodes are `control`.

Node ids are category-prefixed to avoid colliding with application-level ids. Registering two different classes under one id raises.

## Frontend providers (`conductor-providers`)

Framework adapters. Each provider is a subpackage translating between conductor's Python objects and the framework's wire format. Distributed on PyPI as `syv-conductor-providers`; the Python import path is `conductor_providers`.

```python
from conductor_providers import react

palette = registry.describe()                     # the palette is the registry's own
wire = react.graph_to_react(graph)                # Graph → ReactFlow JSON (the placement record under each node's data, display whole; edges derived; positions laid out if a placement has none)
graph2 = react.react_to_graph(wire)               # ReactFlow JSON → Graph
```

`conductor_providers.fastapi.conductor_router(registry)` returns an APIRouter with `GET /nodes` (the palette), `POST /compile`, `POST /execute`, `POST /execute-stream` (server-sent events), and `GET /entities/{kind}` for `EntityDropdown` choices when an `entity_resolver` is given. A graph that cannot run, or a `cache` the run refuses, is a 422 on the execute routes; `/compile` answers a broken graph with 200 and its problems. `/execute` answers with the frame the leg ended on, `graph_complete` or `graph_pending`; a run that asks goes on by posting the ending's `record` back with the answers in `cache` (for a node on rows, `{"rows": [...], "values": [...]}`). Its `from_run` hook turns a request into the values `execute(from_run=...)` supplies.

New providers (Svelte, Vue, Gradio, …) go in sibling subpackages under `conductor_providers.` — no abstract base class to satisfy; each provider picks the shape that matches its framework.

## Examples

The examples are Jupyter notebooks under `examples/` — open them in VS Code, JupyterLab, or any notebook UI and run the cells interactively.

| Notebook | What it covers |
|----------|---------------|
| `01_basic_nodes.ipynb` | Declaring nodes: widgets, defaults, multi-output records, inspecting a registry |
| `02_build_and_run_a_graph.ipynb` | Bindings, asking the compiled graph, problems, collecting results, streaming events, once per row, saving |
| `03_class_nodes.ipynb` | A node with its own methods, and a value the run supplies (`FromRun`) |
| `05_auto_discovery.ipynb` | Versions, upgrades and deprecation, wiring a package's nodes with `register(registry)`, the palette as JSON |
| `06_human_in_the_loop.ipynb` | A node that asks, the run ending pending, and the next leg with the answer |
| `08_widgets.ipynb` | Every control, inspecting a widget's schema |

```bash
uv sync                       # includes the ipykernel used by the notebooks
uv run jupyter lab examples/  # or open the .ipynb files in VS Code
```

The notebooks use `await run(compiled)` — or iterate `execute(compiled)` to see the events — because the kernel already owns an event loop. From a plain `.py` script, `run_sync(compiled)` is the same call; it returns the event the leg ended on.

## Stability and versioning

From `1.0.0` onward, conductor follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). This is the contract host applications can rely on:

**Public API.** A name is part of the public API if it is exported from a package's `__init__` or documented in this README / `docs/`. Anything else — `_`-prefixed names, modules not re-exported from a public surface — is internal and may change in any release without warning. The public surface:

- Top-level `conductor`: the node contract (`NodeDefinition`, `NodeVersion`, `GraphVersion`, `Policy`, `Deprecation`, `NodeDescription`, `version`, `upgrade`, `deprecated`, `Interface`, `FromRun`, `Input`, `Output`, `AnyWidget`, `SKIPPED`, `Asks`, `is_skipped`, `is_asking`), the type vocabulary (`DType`, `DTypeRef`, `Single`, `dtype_of`, `Series`, `Index`, `Ref`, `Result`), the registry (`NodeRegistry`, `RegistryDescription`, `TypeDescription`), and the graph (`Graph`, `GraphNode`, `FieldContent`, `Binding`, `From`, `Static`, `dependencies_of`, `is_input_node`, `CompiledGraph`, `CompiledNode`, `CompiledField`, `Problem`, `Condition`, `Atom`, `ALWAYS`)
- `conductor.execution.engine` (`execute`, `run`, `run_sync`, also exported at the root), `conductor.errors` (`ErrorCause` and the error classes), `conductor.model` (`ConductorModel`), `conductor.widgets`, `conductor.metadata`, `conductor.execution.events` (the `*Event` `TypedDict`s)
- `conductor_nodes` (`registry`, `register_all`, the category modules, `conductor_nodes.types`) and `conductor_providers.react` / `conductor_providers.fastapi`

**Compatibility guarantees from `1.0.0`.**

- *No breaking changes without a major bump.* If a `1.x` release removes a public name, changes a public signature in a way that breaks callers, or alters documented behaviour, the version that ships that change is `2.0.0` (or later).
- *Deprecation policy.* When a public name is scheduled for removal it stays live for **at least one minor release** after deprecation, with a `DeprecationWarning` raised at import or call time. The `CHANGELOG.md` entry that introduces the deprecation lists the target removal version.
- *Internal modules are fair game.* Anything not listed above may be renamed, restructured or removed in any release.
- *The three workspace packages release in lockstep.* `syv-conductor`, `syv-conductor-nodes`, `syv-conductor-providers` share a single version. The `syv-conductor[nodes]` / `[providers]` / `[all]` extras pin sibling packages with `==` to prevent resolver skew.

The `CHANGELOG.md` carries the full history, including the removals that lead up to the next major.

## License

Apache-2.0. See [`LICENSE`](LICENSE) at the repo root; each PyPI wheel ships the same license file.
