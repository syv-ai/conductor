<p align="center">
  <img src="logo-white-background.png" alt="Conductor logo" width="140">
</p>

<h1 align="center">Conductor</h1>

<p align="center">
  A host-agnostic Python engine that compiles and runs graphs of typed nodes.
</p>

You declare a node as a class whose typed `run` signature is its interface. You put nodes in a graph, compile it into a record that says everything about it, and run it as a stream of events. The same declaration drives validation, execution and the palette a visual editor renders. The core has no web framework, database or auth in it; pydantic is its one hard dependency.

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
from conductor import CompiledGraph, From, Graph, GraphNode, NodeRegistry, Param, Static, node
from conductor.widgets import Textarea
from conductor_nodes.types import Text

@node("echo", title="Echo", description="Returns the input unchanged", category="text")
def echo(text: Annotated[Text, Param(title="Input", widget=Textarea())]) -> Text:
    return text

@node("uppercase", category="text")
def uppercase(text: Text) -> Text:
    """Converts to uppercase."""
    return Text(text.upper())

assert uppercase(Text("a")) == "A"          # a node is called like its function

registry = NodeRegistry([echo, uppercase])

graph = Graph(nodes=[
    GraphNode(id="n1", type="echo", version=1, bindings={"text": Static("hello world")}),
    GraphNode(id="n2", type="uppercase", version=1, bindings={"text": From("n1.result")}),
])
compiled = CompiledGraph.from_graph(graph, registry)

state = compiled.run_sync()   # in async code: await compiled.run()
results = compiled.results(state)
print(results["n2"]["result"])  # HELLO WORLD
```

A graph's interface is the inputs nothing feeds and the outputs nothing reads. `with_inputs` returns a copy with inputs filled, by name or by address, and `compiled.outputs` reads what the graph returned:

```python
ready = compiled.with_inputs(text="good night")
print(ready.outputs(ready.run_sync()))  # {'n2.result': 'GOOD NIGHT'}
```

To watch a run, `async with compiled.stream() as events:`. It yields `run_start`, `node_start`, `node_progress`, `node_complete`, `node_skipped`, `node_retry`, `node_failed` and `unit_waiting`, and ends with `run_end`, carrying the run's `state`. Leaving the block stops every unit. `compiled.stream(checkpoint_every=5.0)` also yields `run_checkpoint`, a state with status `running` to save as the leg goes: after a crash `compiled.run(state)` calls no finished unit again and calls those in flight again, so a unit runs at least once. A host that resumes one holds a lease of its own, since a saved state can't say whether its leg is still alive.

## Concepts

**Nodes.** `@node("id")` wraps a plain function and returns a frozen, callable `NodeDefinition`. It registers nothing, so `upper(Text("a"))` is the unit test. The title defaults to the function's name and the description to its docstring. Each parameter is `text: Text`, with `Annotated[Text, Param(title=..., widget=...)]` when a person needs to read more; a default makes it optional, and a parameter of a plain Python type is closed to edges by itself. `-> Text` is one output named `result`, several outputs are a frozen dataclass whose fields are the outputs, and `-> None` declares none. The engine calls the function in a worker thread. It is checked by `@node`, so a mistake fails at import.

**Types.** Every value on an edge has a `DType`, a real class such as `class Text(DType, str)` with an `id`. Conductor ships the mechanism and no vocabulary: a registry holds the types its nodes declare plus the `types` a host names. `target.accepts(source)` decides whether an edge may land. `Series[X]` is the one collection, a plain sequence to a type checker, and `Table[Contact]` a whole table whose columns a `TypedDict` names.

**Graphs and bindings.** A graph is its nodes. Each input holds at most one binding: `From("node.output")` is an edge, `Static(value)` is a typed-in value, and no binding means the declared default. A `Graph` saves itself with `to_path` / `from_path` as YAML or JSON.

**Compile.** `CompiledGraph.from_graph(graph, registry)` never raises for a fault in the graph. Everything wrong is a `Problem` with a stable `code`, anchored on a node, and `is_runnable` says whether a run may start. You ask the result about the graph, `compiled.node(id)` or `compiled.field(ref)`. Every node has a `state` (`ready`, `wiring_failed` or `resolution_failed`) and its own `problems`, so an editor can draw a half-finished graph node by node.

**Rows.** A series arriving on an input declared for one value runs the node once per row, concurrently up to its policy's `concurrency`. A `Series[X]` input receives the whole series.

**Blanks.** A value can be blank (`None`, a cell with nothing in it). If an input's type doesn't say `| None`, the node isn't called for that row and its outputs are blank there; to get the blank, write `email: Text | None`. A default is for an input nothing is wired to. `Series[Text]` leaves blanks out row by row, `Series[Text | None]` keeps them. `node_complete` counts the rows not called, per input (`rows_not_called`).

**Branching.** A node returns `SKIPPED` on the branch it did not take, and nodes fed only `SKIPPED` are skipped in turn. Outputs that are exclusive alternatives share a `choice`.

**A person in the loop.** A node returns `AskPerson`, annotated `-> X | AskPerson`, and its declared outputs are what is asked. The rest of the graph runs on and the state ends `waiting`, each waiting unit and its questions in `state.waiting`. `compiled.answer(state, {unit.id: {output: answer}})` checks the answers and returns a new state; `compiled.run_sync(state)` goes on from it.

**Versions and retries.** A node lists its older versions, oldest first: `@node("split", older=[OlderVersion(split_v1, upgrade=Upgrade(renamed_inputs={"sep": "separator"}))])`. Each version has its own `Policy` (`retries`, `delay`, `timeout`, `concurrency`, `retry_on`) and hooks. `upgrade_node(graph, node_id, registry)` moves a node forward, and `Deprecation(...)` retires a node or a version. Only an `ExternalFailure`, or an exception class named in `retry_on`, is retried.

**Failures.** Every run-time failure is a `Problem` (`type`, `loc`, `msg`, `ctx`), the same record a compile reports, on the event and `state.failure`. A call that cannot be acted on raises `InvalidCallError`, every fault listed. All exceptions inherit from `ConductorError`.

**Values the host supplies.** A parameter marked `FromRun[T]` is filled by type from `compiled.run_sync(from_run={T: value})`, not from an edge.

**Widgets.** A widget is the control a person edits an input with: `TextWidget`, `Textarea`, `Dropdown`, `NumberWidget`, `Switch`, `FileUpload` and more. Each is a pydantic model with a `kind`. There is no default widget for any type, so every input declares its own.

## Standard nodes and providers

`conductor_nodes.registry(categories=[...])` builds a registry of the standard nodes (`text`, `math`, `logic`, `control`, `json`, `regex`), declared in the types `conductor_nodes.types` ships: `Text`, `Number`, `Flag`, `Json`. `conductor_nodes.nodes()` lists them for a registry of your own.

`conductor_providers.react` turns a graph into ReactFlow JSON and back. `conductor_providers.fastapi.conductor_router(registry)` mounts `/nodes`, `/compile`, `/run` and `/run-stream`. `conductor_providers.mermaid.flowchart(compiled)` draws a compiled graph.

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
