<p align="center">
  <img src="logo-white-background.png" alt="Conductor logo" width="140">
</p>

<h1 align="center">Conductor</h1>

<p align="center">
  A host-agnostic Python engine that compiles and runs graphs of typed nodes.
</p>

You declare a node as a function whose typed signature is its interface. You put nodes in a graph, compile it into a record that says everything about it, and run it as a stream of events. The same declaration drives validation, execution and the palette a visual editor renders. The core has no web framework, database or auth in it; it depends on pydantic, polars and pyyaml.

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
from conductor import CompiledGraph, From, Graph, NodeRegistry, node


@node("upper")
def upper(text: str) -> str:
    return text.upper()


@node("exclaim")
def exclaim(text: str, times: int = 1) -> str:
    return text + "!" * times


graph = Graph(nodes=[
    upper.graph_node("a", text="hello"),
    exclaim.graph_node("b", text=From("a.result"), times=2),
])
compiled = CompiledGraph.from_graph(graph, NodeRegistry([upper, exclaim]))

state = compiled.run_sync()  # in async code: await compiled.run()
print(compiled.results(state)["b"]["result"])  # HELLO!!
```

`node.graph_node(...)` places a node, its first argument the node's id in the graph. A checker reads the rest as the function's own parameters. A value is typed in; `From("a.result")` is an edge. It returns a plain `GraphNode`, the form a graph is stored in, which is also what you write for an input left open for the caller or a value the run supplies.

A graph's interface is the inputs nothing feeds and the outputs nothing reads. `with_inputs` returns a copy with inputs filled, by name or by address, and `compiled.outputs` reads what the graph returned:

```python
ready = compiled.with_inputs(text="good night")
print(ready.outputs(ready.run_sync()))  # {'b.result': 'GOOD NIGHT!!'}
```

Every level of a graph is a table. A node typed for one value, wired to a column, runs once per row, and its result becomes a column on the same rows:

```python
import io
from typing import TypedDict

import polars as pl
from conductor import CompiledGraph, From, Graph, NodeRegistry, Series, Table, node


class Contact(TypedDict):
    name: str
    email: str


@node("read-csv")
def read_csv(text: str) -> Table[Contact]:
    return pl.read_csv(io.StringIO(text))


@node("domain")
def domain(email: str) -> str:
    return email.split("@")[1]


@node("count")
def count(values: Series[str]) -> int:
    return len(values)


csv = "name,email\nIda,ida@a.dk\nBo,bo@b.dk\n"
graph = Graph(nodes=[
    read_csv.graph_node("csv", text=csv),
    domain.graph_node("domain", email=From("csv.result.email")),
    count.graph_node("how_many", values=From("domain.result")),
])
compiled = CompiledGraph.from_graph(graph, NodeRegistry([read_csv, domain, count]))
results = compiled.results(compiled.run_sync())
print(list(results["domain"]["result"]))        # ['a.dk', 'b.dk']
print(results["how_many"]["result"])            # 2
```

To watch a run, `async with compiled.stream() as events:`. It yields `run_start`, `node_start`, `node_progress`, `node_complete`, `node_skipped`, `node_retry`, `node_failed` and `unit_waiting`, and ends with `run_end`, carrying the run's `state`. Leaving the block stops every unit. `compiled.stream(checkpoint_every=5.0)` also yields `run_checkpoint`, a state with status `running` to save as the leg goes: after a crash `compiled.run(state)` calls no finished unit again and calls those in flight again, so a unit runs at least once. A host that resumes one holds a lease of its own, since a saved state can't say whether its leg is still alive.

## Concepts

**Nodes.** `@node("id")` wraps a plain function and returns a frozen, callable `NodeDefinition`. It registers nothing, so `upper("a")` is the unit test. The title defaults to the function's name and the description to its docstring. Each parameter is `text: str`, with `Annotated[str, Param(title=..., widget=...)]` when a person needs to read more; a default makes it optional, and `Param(connectable=False)` closes it to edges. `-> str` is one output named `result`, several outputs are a frozen dataclass whose fields are the outputs, and `-> None` declares none. The engine calls the function in a worker thread. It is checked by `@node`, so a mistake fails at import.

**Types.** `str`, `int`, `float`, `bool`, `date`, `datetime` and a pydantic model travel on edges as they are. `Series[X]` is the one collection, a plain sequence to a type checker, and `Table[Contact]` a whole table whose columns a `TypedDict` names, a `polars.DataFrame` at run time and to a type checker. A list of something you only type in, `list[Branch]`, closes its input. Write `class Text(DType, str)` with an `id` only for a type of your own: one whose edges should stay among its own, one a palette names, one that formats itself. A subtype lands wherever its parent does. Conductor ships the mechanism and no vocabulary: a registry holds the types its nodes declare plus the `types` a host names, and `target.accepts(source)` decides whether an edge may land.

**Graphs and bindings.** A graph is its nodes. Each input holds at most one binding: `From("node.output")` is an edge, `Static(value)` is a typed-in value, and no binding means the declared default. A saved graph embedded as a node is wired by the names it offers, the bare name where unique and the address on a clash; inside it stays private. A `Graph` saves itself with `to_path` / `from_path` as YAML or JSON.

**Compile.** `CompiledGraph.from_graph(graph, registry)` never raises for a fault in the graph. Everything wrong is a `Problem` with a stable `code`, anchored on a node, and `is_runnable` says whether a run may start. You ask the result about the graph, `compiled.node(id)` or `compiled.field(ref)`. Every node has a `state` (`ready`, `wiring_failed` or `resolution_failed`) and its own `problems`, so an editor can draw a half-finished graph node by node.

**Rows.** A series arriving on an input declared for one value runs the node once per row, concurrently up to its policy's `concurrency`. A `Series[X]` input receives the whole series.

**Blanks.** A value can be blank (`None`, a cell with nothing in it). If an input's type doesn't say `| None`, the node isn't called for that row and its outputs are blank there; to get the blank, write `email: str | None`. A default is for an input nothing is wired to. `Series[str]` leaves blanks out row by row, `Series[str | None]` keeps them. `node_complete` counts the rows not called, per input (`rows_not_called`).

**Branching.** A node returns `SKIPPED` on the branch it did not take, and nodes fed only `SKIPPED` are skipped in turn. Outputs that are exclusive alternatives share a `choice`.

**A person in the loop.** A node returns `AskPerson`, annotated `-> X | AskPerson`, and its declared outputs are what is asked. The rest of the graph runs on and the state ends `waiting`, each waiting unit and its questions in `state.waiting`. `compiled.answer(state, {unit.id: {output: answer}})` checks the answers and returns a new state; `compiled.run_sync(state)` goes on from it.

**Versions and retries.** A node lists its older versions, oldest first: `@node("split", older_versions=[OlderVersion(split_v1, upgrade=Upgrade(renamed_inputs={"sep": "separator"}))])`. Each version has its own `Policy` (`retries`, `delay`, `timeout`, `concurrency`, `retry_on`) and hooks. `upgrade_node(graph, node_id, registry)` moves a node forward, and `Deprecation(...)` retires a node or a version. Only an `ExternalFailure`, or an exception class named in `retry_on`, is retried.

**Failures.** Every run-time failure is a `Problem` (`type`, `loc`, `msg`, `ctx`), the same record a compile reports, on the event and `state.failure`. A node's own exception goes to `logging.getLogger("conductor")` with its traceback, and its class name to `ctx`, never its text. A call that cannot be acted on raises `InvalidCallError`, every fault listed.

**Values the host supplies.** A parameter marked `FromRun[T]` is filled by type from `compiled.run_sync(from_run={T: value})`, not from an edge.

**Widgets.** A widget is the control a person edits an input with: `TextWidget`, `Textarea`, `Dropdown`, `NumberWidget`, `Switch`, `FileUpload` and more. Each is a pydantic model with a `kind`. There is no default widget for any type, so every input declares its own.

## Standard nodes and providers

`conductor_nodes.registry(categories=[...])` builds a registry of the standard nodes (`text`, `math`, `logic`, `control`, `json`, `regex`), declared in plain Python types and `conductor_nodes.types.Json`. `conductor_nodes.nodes()` lists them for a registry of your own.

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
