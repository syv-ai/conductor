<p align="center">
  <img src="logo-white-background.png" alt="Conductor logo" width="140">
</p>

<h1 align="center">Conductor</h1>

<p align="center">
  A host-agnostic Python engine that compiles and runs graphs of typed nodes.
</p>

You declare a node as a class whose typed `run` signature is its interface. You place nodes in a graph, compile it into a record that says everything about it, and run it as a stream of events. The same declaration drives validation, execution and the palette a visual editor renders. The core has no web framework, database or auth in it; pydantic is its one hard dependency.

## Install

Python 3.12+. Three packages, released together at one version:

```bash
uv add syv-conductor                # the engine: import conductor
uv add syv-conductor-nodes          # standard nodes: import conductor_nodes
uv add syv-conductor-providers      # ReactFlow, FastAPI and Mermaid adapters: import conductor_providers
```

To work on the repository:

```bash
git clone https://github.com/syv-ai/conductor && cd conductor
uv sync
uv run pytest tests/ -q
```

## Example

```python
from typing import Annotated
from conductor import CompiledGraph, From, Graph, GraphNode, NodeDefinition, NodeRegistry, Param, Result, run_sync, Static
from conductor.widgets import Textarea, TextWidget
from conductor_nodes.types import Text

class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "Returns the input unchanged"
    category = "text"

    def run(self, text: Annotated[Text, Param(title="Input", widget=Textarea())]) -> Annotated[Text, Result(title="Output")]:
        return text

class Uppercase(NodeDefinition):
    id = "uppercase"
    title = "Uppercase"
    description = "Converts to uppercase"
    category = "text"

    def run(self, text: Annotated[Text, Param(title="Input", widget=TextWidget())]) -> Annotated[Text, Result(title="Result")]:
        return Text(text.upper())

registry = NodeRegistry()
registry.register(Echo)
registry.register(Uppercase)

graph = Graph(nodes=[
    GraphNode(id="n1", type="echo", version=1, bindings={"text": Static("hello world")}),
    GraphNode(id="n2", type="uppercase", version=1, bindings={"text": From("n1.result")}),
])
compiled = CompiledGraph.from_graph(graph, registry)

results = run_sync(compiled).state.results(compiled)
print(results["n2"]["result"])  # HELLO WORLD
```

A graph's interface is the inputs nothing feeds and the outputs nothing reads. `with_inputs` returns a copy with inputs filled, by name or by address, and `state.outputs` reads what the graph returned:

```python
ready = compiled.with_inputs(text="good night")
print(run_sync(ready).state.outputs(ready))  # {'n2.result': 'GOOD NIGHT'}
```

To watch a run, iterate `execute(compiled)`. It yields `node_start`, `node_progress`, `node_complete`, `node_skipped`, `node_retry` and `node_error`, and ends with one of `graph_complete`, `graph_pending`, `graph_error`, `graph_cancelled` or `graph_timeout`. Every ending carries the run's `state`. Close a stream you leave early, with `async with aclosing(execute(compiled)) as events:`, and every unit stops.

## Concepts

**Nodes.** A node is a `NodeDefinition` subclass with `id`, `title`, `description` and `category`. Each `run` parameter is `Annotated[DType, Param(title=..., widget=...)]`; a default makes it optional. One output is `Annotated[DType, Result(title=...)]` and is named `result`. Several outputs are a frozen dataclass whose fields are the outputs. `run` is a plain function that the engine calls in a worker thread. The class is checked when it is defined, so a mistake fails at import.

**Types.** Every value on an edge has a `DType`, a real class such as `class Text(DType, str)` with an `id` and a `title`. Conductor ships the mechanism and no vocabulary: a registry holds the types its nodes declare plus what the host adds with `registry.add_types(...)`. `target.accepts(source)` decides whether an edge may land. `Series[X]` is the one collection.

**Graphs and bindings.** A graph is its nodes. Each input holds at most one binding: `From("node.output")` is an edge, `Static(value)` is a typed-in value, and no binding means the declared default. A `Graph` saves itself with `to_path` / `from_path` as YAML or JSON.

**Compile.** `CompiledGraph.from_graph(graph, registry)` never raises for a fault in the graph. Everything wrong is a `Problem` with a stable `code`, anchored on a node, and `is_runnable` says whether a run may start. You ask the result about the graph, `compiled.node(id)` or `compiled.field(ref)`. Every node has a `state` (`ready`, `not_derived` or `unresolved`) and its own `problems`, so an editor can draw a half-finished graph node by node.

**Rows.** A series arriving on an input declared for one value runs the node once per row, concurrently up to its policy's `concurrency`. A `Series[X]` input receives the whole series.

**Branching.** A node returns `SKIPPED` on the branch it did not take, and nodes fed only `SKIPPED` are skipped in turn. Outputs that are exclusive alternatives share a `choice`.

**A person in the loop.** A node returns `Asks` with its questions, annotated `-> X | Asks`. The rest of the graph runs on and the run ends `graph_pending`. The next call carries the answers: `run_sync(compiled, state=ending.state, cache={node_id: answers})`.

**Versions and retries.** Several versions live in one class as `@version(n)` methods, each with a `Policy` (`retries`, `delay`, `timeout`, `concurrency`, `retry_on`). `@upgrade(1, 2)` rewrites values saved against version 1, and `@deprecated` retires a node or a version. Only an `ExternalFailure`, or an exception class named in `retry_on`, is retried.

**Failures.** Every run-time failure carries an `ErrorCause` (`code`, `message`, `details`, `row`) on the exception and on the event. All exceptions inherit from `ConductorError`.

**Values the host supplies.** A parameter marked `Annotated[T, FromRun()]` is filled by type from `run_sync(compiled, from_run={T: value})`, not from an edge.

**Widgets.** A widget is the control a person edits an input with: `TextWidget`, `Textarea`, `Dropdown`, `NumberWidget`, `Switch`, `FileUpload` and more. Each is a pydantic model with a `kind`. There is no default widget for any type, so every input declares its own.

## Standard nodes and providers

`conductor_nodes.registry(categories=[...])` builds a registry of the standard nodes (`text`, `math`, `logic`, `control`, `json`, `regex`), declared in the types `conductor_nodes.types` ships: `Text`, `Number`, `Flag`, `Json`. `conductor_nodes.register_all(registry)` adds them to a registry you already have.

`conductor_providers.react` turns a graph into ReactFlow JSON and back. `conductor_providers.fastapi.conductor_router(registry)` mounts `/nodes`, `/compile`, `/execute` and `/execute-stream`. `conductor_providers.mermaid.flowchart(compiled)` draws a compiled graph.

## Learn more

- [`docs/OVERVIEW.md`](docs/OVERVIEW.md): the architecture on one page.
- [`docs/widgets.md`](docs/widgets.md): every control and its options.
- [`examples/`](examples/): Jupyter notebooks on nodes, graphs, versions, a person in the loop and widgets.
- [`skills/`](skills/): agent skills for adding a node and building a graph.
- `python -m conductor.about`: the library reference that ships in the wheel.
- [`CHANGELOG.md`](CHANGELOG.md): what changed between releases.

## Versioning

Conductor follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The public API is three doors, `conductor`, `conductor.widgets` and `conductor.events`, each listing its names in `__all__`, plus the `conductor_nodes` and `conductor_providers` packages. Anything else may change in any release. A breaking change to the public API ships in a major release.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
