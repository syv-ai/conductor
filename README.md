<p align="center">
  <img src="logo-white-background.png" alt="Conductor logo" width="140">
</p>

<h1 align="center">Conductor</h1>

<p align="center">
  Compile and run graphs of typed Python functions.
</p>

```python
from conductor import CompiledGraph, NodeRegistry, make_graph

registry = NodeRegistry()


@registry.node("upper")
def upper(text: str) -> str:
    return text.upper()


@registry.node("exclaim")
def exclaim(text: str, times: int = 1) -> str:
    return text + "!" * times


graph = make_graph(upper, exclaim, values={"exclaim.times": 2})
compiled = CompiledGraph.from_graph(graph, registry)

ready = compiled.with_inputs(text="hello")
state = ready.run_sync()  # in async code: await ready.run()
print(ready.outputs(state))  # {'exclaim.result': 'HELLO!!'}
```

A node is a function, and its type hints are its interface. The same hints drive validation, the run and the palette a visual editor shows. A registry is what a host offers: `@registry.node` declares a node and adds it. A graph is data, a `GraphRecord`: its nodes, each naming the node it uses and what is typed into it, and its edges, `{"nodes": [{"id": "upper", "node": "upper"}, ...], "edges": [["upper.result", "exclaim.text"]]}`. That is what an editor writes and a host stores. `make_graph(...)` writes one from Python: with no `edges` the nodes form a chain, each taking its one required open input from the one output before it; with `edges=[("upper.result", "exclaim.text")]` those are every edge. A node's id in the graph is its own id, `upper-1`, `upper-2` when it is used twice. `values` are typed in and fixed.

The core has no web framework, database or auth in it. It depends on pydantic and polars.

## Install

Python 3.12+:

```bash
uv add syv-conductor                # import conductor
```

To work on the repository:

```bash
git clone https://github.com/syv-ai/conductor && cd conductor
uv sync
uv run pytest tests/ -q
```

## Inputs and outputs

A graph takes the inputs of its first nodes, those with no edge into any of their inputs, and returns the outputs nothing reads. `with_inputs` returns a copy with inputs filled, by name or by address, and it takes nothing else, since it is what you hand an end user's values to:

```python
ready = compiled.with_inputs(text="good night")
print(ready.outputs(ready.run_sync()))  # {'exclaim.result': 'GOOD NIGHT!!'}
```

A value the author types in is fixed, and the graph doesn't offer that input. A node's `defaults` fill an input that stays open, so a caller may still change it: `NodeRecord(id="exclaim", node="exclaim", defaults={"times": 3})`.

## Tables

Every level of a graph is a table. A node typed for one value, wired to a column, runs once per row, and its result becomes a column on the same rows:

```python
import io
from typing import TypedDict

import polars as pl
from conductor import CompiledGraph, NodeRegistry, Series, Table, make_graph

registry = NodeRegistry()


class Contact(TypedDict):
    name: str
    email: str


@registry.node("read-csv")
def read_csv(text: str) -> Table[Contact]:
    return pl.read_csv(io.StringIO(text))


@registry.node("domain")
def domain(email: str) -> str:
    return email.split("@")[1]


@registry.node("count")
def count(values: Series[str]) -> int:
    return len(values)


csv = "name,email\nIda,ida@a.dk\nBo,bo@b.dk\n"
graph = make_graph(
    read_csv, domain, count,
    edges=[("read-csv.result.email", "domain.email"), ("domain.result", "count.values")],
    values={"read-csv.text": csv},
)
compiled = CompiledGraph.from_graph(graph, registry)
results = compiled.results(compiled.run_sync())
print(results["domain"]["result"]["values"])    # ['a.dk', 'b.dk']
print(results["count"]["result"])               # 2
```

`print(compiled.run_plan)` says in plain words where each node runs and what each input receives.

## Watching a run

```python
async with compiled.stream() as events:
    async for event in events:
        if event.type == "node_complete":
            print(event.node_id, event.outputs)
        elif event.type == "run_end":
            state = event.state  # the run to save
```

A leg yields `run_start`, `node_start`, `node_progress`, `node_complete`, `node_skipped`, `node_retry`, `node_failed` and `unit_waiting`, and ends with `run_end`, whose `state` stays out of what the event writes. Leaving the block stops every unit. An event writes itself as JSON: `event.model_dump(mode="json")` or `model_dump_json()`, tables whole, and `conductor.to_jsonable(compiled.results(state))` does the same for results. `node_progress` arrives at most four times a second per node, its last count always among them.

`compiled.stream(checkpoint_every=5.0)` also yields `run_checkpoint`, a state with status `running` you can save as the run goes. After a crash, `compiled.run(state)` skips the finished units and calls the ones that were in flight again, so a unit runs at least once. A saved state can't tell whether its run is still alive, so a host that resumes one holds a lease of its own.

## Concepts

**Nodes.** `@node("id")` wraps a plain function and returns a frozen, callable `NodeDefinition`. It registers nothing, so `upper("a")` is the unit test; `@registry.node("id")` takes the same arguments and adds the node to a registry too. The title defaults to the function's name and the description to its docstring. A parameter is `text: str`, or `Annotated[str, Param(title=..., widget=...)]` when a person needs to read more. A default makes it optional, and `Param(connectable=False)` closes it to edges. `-> str` is one output named `result`, a frozen dataclass is one output per field, and `-> None` declares none. The engine calls the function in a worker thread. `@node` checks the function, so a mistake fails at import.

**Types.** `str`, `int`, `float`, `bool`, `date`, `datetime` and a pydantic model travel on edges as they are. `Series[X]` says a field holds many values of `X`: a `Sequence[X]` to a type checker, a plain list to the node. A `list[X]` stays one value. `Table[Contact]` is a whole table whose columns a `TypedDict` names, and a `polars.DataFrame` at run time and to a type checker. A list of something you only type in, `list[Branch]`, closes its input. Write `class Text(DType, str)` with an `id` only for a type of your own: one whose edges stay among its own, one a palette names, one that formats itself. A subtype lands wherever its parent does. A registry holds the types its nodes declare plus the `types` a host names, and `target.accepts(source)` decides whether an edge may land.

**Registries.** A `NodeRegistry` is the nodes a host offers, by id: `"upper" in registry`, `registry["upper"]`, and looping yields the definitions. `registry.add(upper, shout)` adds plain `@node`s, and a second node under one id raises; `replace` and `remove` change one. `registry.add_subregistry(text_nodes)` copies another registry's nodes in, so each module can keep a registry of its own and the host adds them at the top. `NodeRegistry(category="text", tags=["builtin"], extra={...})` sets defaults for every node that enters: tags add up, outermost first, and the node's own `category` and `extra` keys win. `extra` is the host's own JSON about a node, kept and described, never read by conductor. `registry.filter(lambda definition: ...)` is a new registry of the nodes kept, say the ones one customer may place.

**Graph records.** A `GraphRecord` is its `nodes` and its `edges`. A `NodeRecord` is `id`, `node` (the node it uses), `version` (1 when left out), `values` (typed in, fixed), `defaults` (typed in, open to a caller) and `display` (the host's own, never read); empty fields are left out when saved. Edges are `[from, to]` addresses and a set: several edges into one input gather in the order their nodes appear in `nodes`, whatever order the list holds. An input takes an edge, a value or a default, never two; an edge drawn twice or into a node the graph doesn't have is a problem too. An input with none of them takes its declared default. A saved graph embedded as a node is wired by the names it offers, the bare name where unique and the address on a clash; what's inside stays private. A `GraphRecord` saves as JSON with pydantic's own `model_dump_json` and reads back with `GraphRecord.model_validate_json`.

**Compile.** `CompiledGraph.from_graph(graph, registry)` never raises for a fault in the graph. Everything wrong is a `Problem` with a stable `type`, anchored on a node, and `is_runnable` says whether a run may start. Ask the result about the graph with `compiled.node(id)`, `compiled.field(ref)` or `compiled.run_plan`. Every node has a `state` (`ready`, `wiring_failed` or `resolution_failed`) and its own `problems`, so an editor can draw a half-finished graph node by node. `compiled.describe()` is all of that as one JSON-ready record, as `registry.describe()` is for what can be placed.

**Rows.** A series arriving on an input declared for one value runs the node once per row, up to its policy's `concurrency` at a time. A `Series[X]` input receives every value under the node's row, flat, however many levels down. When nothing else says where the node runs, it runs once per row of the level right above the series.

**Merging tables.** Every table carries a `__row` column (`contacts.result:0`) saying which row is which. You never declare it, and a row you return without an id gets a fresh one. Wire several tables or columns into one `Table` input and they are merged on those ids. A parent column beside its children is copied down, so the rows are those of the deepest source and a parent with nothing under it has no row in that table. Sources on the same level keep all their rows, with blanks where one has nothing. A table rebuilt in a node and merged with its source has to keep `__row`, or the node fails with `row_ids_lost`. A column is named after its output (or, from a table, keeps its own name). A column a table carries on from the table it was made from is one column with it, read from the newer table where that has the row. Two columns of one name that aren't related: the later takes its output's name (`name_contacts`, `result_2`). Tables with no rows in common are refused when the graph is checked (`no_rows_in_common`): join them on a key in a node of your own.

**Blanks.** A value can be blank (`None`, a cell with nothing in it). If an input's type doesn't say `| None`, the node isn't called for that row and its outputs are blank there; to get the blank, write `email: str | None`. A default is for an input nothing is wired to. `Series[str]` leaves blanks out row by row, `Series[str | None]` keeps them. A row one column skipped counts as blank there, so `Series` inputs reading one table stay on the same rows whatever a node zips: a row `Series[str]` can't take leaves all of them. Two such inputs reading opposite branches of one decision would lose every row, so that is refused when the graph is checked (`series_inputs_never_share_a_row`): write `Series[str | None]`, or wire both branches into one input. `node_complete` counts the rows not called, per input (`rows_not_called`).

**Branching.** A node returns `SKIPPED` on the branch it did not take, and nodes fed only `SKIPPED` are skipped in turn. Outputs that are alternatives share an `exclusive_group`. Exclusive tables wired into one `Table` input are the branch that ran.

```python
from dataclasses import dataclass
from typing import Annotated

from conductor import SKIPPED, Result, node


@dataclass(frozen=True)
class Routed:
    big: Annotated[str, Result(exclusive_group="size")]
    small: Annotated[str, Result(exclusive_group="size")]


@node("route")
def route(text: str) -> Routed:
    return Routed(text, SKIPPED) if len(text) > 3 else Routed(SKIPPED, text)
```

**A person in the loop.** A node returns `AskPerson`, annotated `-> X | AskPerson`, and its declared outputs are what is asked. The rest of the graph runs on and the state ends `waiting`, each waiting unit and its questions in `state.waiting`.

```python
from conductor import AskPerson, CompiledGraph, NodeRegistry, make_graph


registry = NodeRegistry()


@registry.node("approve")
def approve(invoice: str) -> str | AskPerson:
    return AskPerson(prompt=f"Approve {invoice}?")


compiled = CompiledGraph.from_graph(make_graph(approve, values={"approve.invoice": "INV-1"}), registry)
paused = compiled.run_sync()                                                   # paused.status == "waiting"
answered = compiled.answer(paused, {unit.id: {"result": "yes"} for unit in paused.waiting})  # checks them; no node runs
state = compiled.run_sync(answered)
```

**Running again.** Pass the state of an earlier leg and a node nothing changed above is restored whole; a changed node reuses the result of every row that received the same as before, even if it moved. Add one row to a CSV and one mail goes out: `state = compiled.with_inputs(contacts=edited).run_sync(state)`. A person's answer stays with the row it was asked on: when that row is replaced or any of its columns edited, the person is asked again. A node that reads the world says so with `Policy(reuse_when_inputs_match=False)`: a node reading a file by its path is called again on a new leg from a `complete` state, so an edited file is read, and the rows below it reuse as they always do. A leg going on from a waiting, failed or cancelled state keeps what it read, so answering a person doesn't re-read the file. To read one again in any state, `state.without_nodes("read")`. A state goes back only to the runner it came from.

**Versions and retries.** A newer version names the one it replaces, itself a `@node` of the same id: `@node("split", replaces=split_v1, upgrade=Upgrade(renamed_inputs={"sep": "separator"}))`. Each version has its own `Policy` (`retries`, `retry_delay`, `attempt_timeout`, `concurrency`, `retry_on`, `reuse_when_inputs_match`) and hooks. `upgrade_node(graph, node_id, registry)` moves a node forward, and `Deprecation(...)` retires a version; on the newest, the node. Only an `ExternalFailure`, or an exception class named in `retry_on`, is retried.

**Failures.** Every run-time failure is a `Problem` (`type`, `loc`, `msg`, `ctx`), the same record a compile reports, on the event and on `state.cause`. A node's own exception goes to `logging.getLogger("conductor")` with its traceback, and only its class name reaches `ctx`. A failed unit starts nothing more; the calls already on a thread finish and are kept, so running on from the state doesn't repeat them. A call that can't be acted on raises `InvalidCallError`, with every fault listed.

**Values the host supplies.** A parameter marked `FromRun[T]` is filled by type from `compiled.run_sync(from_run={T: value})`, not from an edge.

**Widgets.** A widget is the control a person edits an input with: `TextWidget`, `Textarea`, `Dropdown`, `NumberWidget`, `Switch`, `FileUpload` and more. Each is a pydantic model with a `type`. No type has a default widget, so an input names its own.

## Serving runs

To serve runs over HTTP, copy [`docs/serving-runs.md`](docs/serving-runs.md) onto your own FastAPI application: start, follow, read, answer and cancel.

## Learn more

- [`docs/OVERVIEW.md`](docs/OVERVIEW.md): the architecture on one page.
- [`docs/widgets.md`](docs/widgets.md): every control and its options.
- [`docs/serving-runs.md`](docs/serving-runs.md): a run over HTTP, as a page to copy.
- [`examples/`](examples/): Jupyter notebooks on nodes, graphs, tables, versions, a person in the loop and widgets.
- [`skills/`](skills/): agent skills for adding a node and building a graph.
- `python -m conductor.about`: the library reference that ships in the wheel.
- [`CHANGELOG.md`](CHANGELOG.md): what changed between releases.

## Versioning

Conductor follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The public API is three modules, `conductor`, `conductor.widgets` and `conductor.events`, each listing its names in `__all__`. Anything else may change in any release. A breaking change to the public API ships in a major release.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
