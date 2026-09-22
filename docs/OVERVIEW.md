# Conductor — architecture at a glance

## What it is

Conductor is a Python library that compiles and runs graphs of typed nodes. Any tool where nodes are connected together, on a canvas or in a script, can sit on top of it. The core is host-agnostic: no FastAPI, no database, no auth, and no vocabulary of its own. The one hard dependency is pydantic.

Three packages in one uv workspace, released in lockstep:

| Package | What it is |
|---------|------------|
| **`syv-conductor`** (`conductor`) | The engine: the node contract, the type mechanism, compile, the row engine, widgets, errors. |
| **`syv-conductor-nodes`** (`conductor_nodes`) | Standard nodes (`text`, `math`, `logic`, `json_ops`, `regex_ops`, `decision`) in a four-type vocabulary of their own. |
| **`syv-conductor-providers`** (`conductor_providers`) | Adapters. `react` turns a graph into ReactFlow JSON and back and builds the palette; `fastapi` mounts a router. |

## One declaration, three readers

A node is a class whose typed `run` signature is its interface:

```python
class Uppercase(NodeDefinition):
    id = "uppercase"
    title = "Uppercase"
    description = "Capitalizes text"
    category = "text"

    def run(self, text: Annotated[Text, Param(title="Input", widget=Textarea())]) -> Annotated[Text, Result(title="Output")]:
        return Text(text.upper())
```

That one declaration is read three ways. **Execution** calls the method as it is, on a fresh instance per call. **Validation** builds a pydantic model from the `Input` records. **Rendering** reads `describe()`, the palette entry with each field's type, widget and title, dumped through pydantic for the UI. Nothing is declared twice.

Every value on an edge has a `DType`. Conductor declares none: a host says what `Text`, `Number` or `Document` is, and `target.accepts(source)` is the one edge question. `Series[X]` is the one collection, many values of one type on an `Index`.

Several versions live in one class: `@version(1)` on an older method, `@version(2)` on the one named `run`. Each has its own signature and `Policy`; `@upgrade(1, 2)` rewrites saved values, and `@deprecated` retires a node or a version. A placed node pins `type` and `version`, so a saved graph keeps running as the node evolves.

## Widgets

Every control is a frozen pydantic model with a `kind`: `TextWidget`, `Textarea`, `TemplateTextarea`, `CodeEditor`, `Dropdown`, `EntityDropdown`, `NumberWidget`, `Range`, `Switch`, `DatePicker`, `FileUpload`, `ListWidget`, `Tags`, `TableInput`, `SchemaBuilder`, `IfElseBuilder`. `AnyWidget` is their union, so a generic frontend renders any node from the palette.

Conductor ships no default widget for any type, since the same `Text` may be a textarea, a single line or a dropdown. The vocabulary inside a control, a dropdown's `choices` or a condition builder's `operators`, is the host's, carried as data. Full catalog: [`widgets.md`](widgets.md). Hands-on tour: [`examples/08_widgets.ipynb`](../examples/08_widgets.ipynb).

## Declare → compile → execute

- **Declaring** a node checks it when the class is defined. A missing `id`, `title`, `description` or `category`, a widget written bare instead of on a `Param`, a return without a `Result`, an `async def run`: each fails with the traceback at the class. `NodeRegistry.register(cls)` adds the catalogue's rules (versions from 1 with no holes, an `alternative` that exists).
- **`CompiledGraph.from_graph(graph, registry)`** resolves every pin, validates the bindings, types every field from its edges, decides which nodes run once per row, asks the field hooks and expands embedded graphs. It never raises for a fault in the graph: everything wrong is a `Problem` with a stable `code`, anchored on a node, and `is_runnable` says whether a run may start. The result is asked at three scales: the graph, `compiled.node(node_id)` and `compiled.field(ref)`.
- **`execute(compiled)`** runs one leg as an async generator of events: `node_start`, `node_progress`, `node_complete`, `node_retry`, `node_skipped`, `node_error`, and an ending, `graph_complete`, `graph_pending`, `graph_error`, `graph_cancelled` or `graph_timeout`. Every ending carries `results` and `record`. `await run(compiled)` drains it and returns the ending; `run_sync(compiled)` is the same call from a script.

## The row engine

The engine's unit of work is a node on a row. A node that runs once is one unit; a node fed a series on an input declared for one value iterates, one unit per row, and each unit starts as soon as what it reads exists. Row 1 can finish a whole chain while row 10 is still being produced.

```
  split ──> clean (row 0, 1, 2 …) ──> summarise (row 0, 1, 2 …) ──> join (once)
```

A `Series[X]` input is a reduction: it receives the whole series, or the rows under each parent row. A node's rows run at most `Policy.concurrency` at a time, each in a thread from a pool the leg owns with one worker per unit that may be in flight, and a run's time grows with its rows.

Retries live on the version's `Policy`, and each unit retries on its own. Only the outside world's failure is retried — an `ExternalFailure` the node raises, or a foreign exception the policy's `retry_on` names; anything else runs once. `Policy.timeout` is how long the leg waits on one attempt, counted from the moment the node's thread starts: the leg owns a thread pool with one worker per unit that may be in flight, so no unit ever waits for a worker. It never interrupts the thread. A timed-out attempt is final (`NodeTimeoutError`, code `timeout`); the thread finishes on its own, keeps the node's concurrency slot until it does, and what it returns is dropped. The timeout worth retrying is the client's own, set on the client inside `run`: when the client gives up, the thread has returned and a retry runs nothing twice. A `run` that holds the GIL — a regex that never finishes, a tight loop over a huge input — blocks the whole process, and nothing in the engine can stop it. Where legs run, in the API process or in a worker of their own, is the host's decision. Every failure carries an `ErrorCause` (`code`, `message`, `details`, `row`) on the exception and on the event, so a host routes it by code rather than by parsing a message; a cause the engine writes carries the generic message for its code, a node's own keeps its message, and a foreign exception's text reaches no event:

```
ConductorError
├── CompilationError
└── NodeError (External, Validation, Execution, Timeout)
```

**Branching** is a value. A node returns `SKIPPED` on the branch it did not take; a skip at a row leaves the series downstream sparse, and a skip above a node's rows skips everything under it. Outputs that are exclusive alternatives share a `choice`, so an editor knows exactly one arrives.

**A person in the loop** is a value too. A node returns `Asks` with its questions; the rest of the graph runs on, and the leg ends `graph_pending` with every question. Answering is the next leg: `execute(compiled, cells=..., cache=...)`, where `cells` is what the earlier leg produced and `cache` holds the answers as the asking node's outputs. Nothing runs twice and nothing is resumed.

## Bindings — one input, one source

A graph is its nodes; there is no edge list. Each placed node says per input where the value comes from:

```python
GraphNode(id="mapper", type="build-map", version=1, bindings={"seed": Static("x")})
GraphNode(id="redactor", type="redact", version=1, bindings={"mapping": From(Ref("mapper", "result"))})
# the From binding is the edge and the dependency
```

An `From` holds refs in operand order, a `Static` is what the author typed, and an absent binding means the declared default. Dependencies, cycles and what the graph takes and returns are derived from the bindings. A `Graph` saves itself: `graph.to_path("approval.yaml")`, `Graph.from_path(...)`.

## Standard nodes and providers

**`conductor-nodes`** ships the usual nodes so a host need not write them again. `conductor_nodes.registry()` builds a registry holding them, every category or a subset, and `register_all` adds them to a registry a host already has:

```python
import conductor_nodes

registry = conductor_nodes.registry(categories=["text", "math"])
conductor_nodes.register_all(my_registry)
```

The nodes are declared in `conductor_nodes.types` (`Text`, `Number`, `Flag`, `Json`), because a node library has to say what its nodes take. Node ids are prefixed by category (`text-uppercase`, `math-add`), so they don't collide with a host's own.

**`conductor-providers`** sits between conductor's records and a framework:

```python
from conductor_providers import react
from conductor.metadata import Param

palette = registry.describe()                     # the palette is the registry's own
wire = react.graph_to_react(graph)                # Graph → ReactFlow JSON
graph = react.react_to_graph(wire)                # ReactFlow JSON → Graph
```

`conductor_providers.fastapi.conductor_router(registry)` mounts `/nodes`, `/compile`, `/execute`, `/execute-stream` and `/entities/{kind}`; a run that asks goes on in legs over HTTP. A new provider is a sibling subpackage, with no base class to satisfy.

## The reference ships with the library

`python -m conductor.about` prints the packaged reference from inside the installed wheel:

```bash
python -m conductor.about                 # the whole reference
python -m conductor.about sections        # the section slugs
python -m conductor.about rows            # one section (prefix match)
```

The same text in code: `from conductor.about import get_content, get_section`.

## Working agreements

- **CI runs ruff and pytest on every PR** (`.github/workflows/ci.yml`). Locally: `uvx ruff check .` and `uv run pytest tests/`.
- **Conductor stands alone.** `tests/test_core/test_standalone.py` fails on a host's word ("flow", "app", an access model), a host's import or a type the library declares for itself.
- **Docs drift is audited.** The `/docs-audit` slash command runs on demand after a feature, and a weekly CI audit opens a PR as a safety net. `AGENTS.md` and `llms.txt` match the shipped surface.
- **Notebook outputs are stripped on commit** by the `nbstripout` pre-commit hook; run the cells to see values.
- **Registering two classes under one id is an error**, not a silent overwrite.

## Further reading

- [`README.md`](../README.md): install, quick start, concepts.
- [`AGENTS.md`](../AGENTS.md): architecture and conventions, the primary context for agent sessions.
- [`packages/conductor/src/conductor/about/llms.txt`](../packages/conductor/src/conductor/about/llms.txt): the packaged reference.
- [`examples/*.ipynb`](../examples/): tutorial notebooks.
