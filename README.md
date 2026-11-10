<p align="center">
  <img src="logo-white-background.png" alt="Conductor logo" width="140">
</p>

<h1 align="center">Conductor</h1>

<p align="center">
  Run graphs of typed Python functions.
</p>

```python
from conductor import NodeRegistry, make_graph

registry = NodeRegistry()


@registry.node("upper")
def upper(text: str) -> str:
    return text.upper()


@registry.node("exclaim")
def exclaim(text: str, times: int = 1) -> str:
    return text + "!" * times


graph = registry.load(make_graph(upper, exclaim, values={"exclaim.times": 2}))
run = graph.run(text="hello")
print(run.outputs)  # {'exclaim.result': 'HELLO!!'}
```

A node is a function, and its type hints are its interface. The same hints drive validation, the run and the palette a visual editor shows. A registry is what a host offers: `@registry.node` declares a node and adds it. A graph is data, a `GraphRecord`: its nodes, each naming the node it uses and what is typed into it, and its edges, `{"nodes": [{"id": "upper", "node": "upper"}, ...], "edges": [["upper.result", "exclaim.text"]]}`. That is what an editor writes and a host stores. `registry.load(record)` reads one, as a record, JSON text or a mapping, into a `Graph`. `make_graph(...)` writes one from Python: with no `edges` the nodes form a chain, each taking its one required open input from the one output before it; with `edges=[("upper.result", "exclaim.text")]` those are every edge. A node's id in the graph is its own id, `upper-1`, `upper-2` when it is used twice. `values` are typed in and fixed.

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

A graph takes the open inputs of its first nodes, those with no edge into any of their inputs, and returns the outputs nothing reads. `graph.inputs` lists them by name (by address, `node.input`, where two share a name) and `graph.outputs` by address. `graph.run(**inputs)` runs; `graph.with_inputs(**inputs)` checks the inputs and returns a run not yet started, for a host that checks a request now and runs it later. A required input left out raises `InvalidCallError` at the call:

```python
ready = graph.with_inputs(text="good night")
print(ready.resume().outputs)  # {'exclaim.result': 'GOOD NIGHT!!'}
```

A value the author types in is fixed, and the graph doesn't offer that input. A node's `defaults` fill an input that stays open, so a caller may still change it: `NodeRecord(id="exclaim", node="exclaim", defaults={"times": 3})`. An open input named in the node's `many` takes a list from the caller, a row per item: `NodeRecord(id="upper", node="upper", many={"text"})`.

## Tables

A node typed for one value, wired to a column, runs once per row, and its result is a column on the same rows:

```python
import io
from typing import TypedDict

import polars as pl
from conductor import NodeRegistry, Series, Table, make_graph

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
graph = registry.load(make_graph(
    read_csv, domain, count,
    edges=[("read-csv.result.email", "domain.email"), ("domain.result", "count.values")],
    values={"read-csv.text": csv},
))
run = graph.run()
print(run.results["domain"]["result"])    # ['a.dk', 'b.dk']
print(run.results["count"]["result"])     # 2
print(graph.plan)                         # where each node runs
```

## Watching a run

```python
async for event in graph.with_inputs().stream():
    if event.type == "node_complete":
        print(event.node_id, event.outputs)
    elif event.type == "run_end":
        finished = event.run  # the run to save
```

A leg yields `run_start`, `node_start`, `node_progress`, `node_complete`, `node_skipped`, `node_retry`, `node_failed` and `unit_waiting`, and ends with `run_end`, whose `run` stays out of what the event writes. The leg runs on a thread of its own, so a slow node never holds the event loop; leaving the loop, or setting `stream(cancel=event)`, ends it before its next unit. `event.model_dump(mode="json")` is the event as JSON. `node_progress` arrives at most four times a second per node.

## Concepts

**Nodes.** `@node("id")` wraps a plain function and returns a frozen, callable `NodeDefinition`. It registers nothing, so `upper("a")` is the unit test; `@registry.node("id")` takes the same arguments and adds the node to a registry too. The title defaults to the function's name and the description to its docstring. A parameter is `text: str`, or `Annotated[str, Param(title=..., widget=...)]` when a person needs to read more; `Param` also takes pydantic's limits (`ge`, `le`, `max_length`, ...). A default makes it optional. `-> str` is one output named `result`, a frozen dataclass is one output per field, and `-> None` declares none. `@node` checks the function, so a mistake fails at import with a `NodeDeclarationError`.

**Types.** Any type pydantic can check and write as JSON travels on edges: `str`, `int`, `float`, `bool`, `date`, a pydantic model, a class of your own. `Series[X]` says a field holds many values of `X`: a plain list to the node and to a type checker. A `list[X]` stays one value. `Table[Contact]` is a whole table whose columns a `TypedDict` names, a `polars.DataFrame` at run time and to a type checker. A subtype lands wherever its parent does, and a class of your own may declare `id` (the name an editor knows it by), `display_title`, and a classmethod `accepts(source)` that lets values of another type land on it.

**Registries.** A `NodeRegistry` is the nodes a host offers, by id: `"upper" in registry`, `registry["upper"]`, and looping yields the definitions. `registry.add(upper, shout)` adds plain `@node`s, and a second node under one id raises; `replace` and `remove` change one. `registry.add_subregistry(text_nodes)` copies another registry's nodes in, so each module can keep a registry of its own and the host adds them at the top. `NodeRegistry(category="text", tags=["builtin"], extra={...})` sets defaults for every node that enters: tags add up, outermost first, and the node's own `category` and `extra` keys win. `extra` is the host's own JSON about a node, kept and described, never read by conductor. `registry.filter(lambda definition: ...)` is a new registry of the nodes kept, say the ones one customer may place. `registry.describe()` is the palette an editor reads, as JSON.

**Graph records.** A `GraphRecord` is its `nodes` and its `edges`. A `NodeRecord` is `id`, `node` (the node it uses), `version` (1 when left out), `values` (typed in, fixed), `defaults` (typed in, open to a caller), `many` (open inputs that take a list) and `display` (the host's own, never read); empty fields are left out when saved. Edges are `[from, to]` addresses and a set, kept sorted, so their order never tells two records apart: several edges into one input gather in the order their nodes appear in `nodes`, then in the order a node declares its outputs. An input takes an edge, a value or a default, never two; an edge drawn twice or naming a node the graph doesn't have is a problem too. A `GraphRecord` saves as JSON with pydantic's own `model_dump_json`.

**Loading.** `registry.load(record)` never raises for a fault in the graph. Everything wrong is a `Problem` with a stable `type` and a place (`loc`), in `graph.problems`, and `graph.is_runnable` says whether a run may start. An open input of a first node is not a problem: it is an input of the graph. A required input on a later node that nothing feeds is `unbound_on_later_node`. `graph.describe()` is the graph as an editor draws it, as JSON. A loaded graph keeps its own copy of the definitions it used, so a later change to the registry doesn't reach it.

**Rows.** A list or table arriving on an input declared for one value runs the node once per row of it, and each result stays on its row. A `Series[X]` input gathers every value under the node's row. Sources must lie on one path of the row tree: a value from rows above is copied down, and rows of two different tables meet only through a join on a key the author draws in a node (`rows_not_related`). `max_rows=` on a run is a ceiling on any one table's rows, checked before any node is called on them: over it, the run stops with a `row_ceiling` problem, and `run.resume(max_rows=...)` carries on with a higher one.

**Blanks.** A value can be blank (`None`, a cell with nothing in it). If an input's type doesn't say `| None`, the node isn't called for that row and its outputs are blank there; `run.not_called` lists each such row with its node and input. To get the blank, write `email: str | None`. `Series[str]` leaves blanks out row by row, `Series[str | None]` keeps them. Two `Series` inputs reading opposite branches of one decision would lose every row, so that is refused when the graph is loaded (`series_inputs_never_share_a_row`).

**Branching.** A node returns `SKIPPED` on the branch it did not take, and nodes fed only `SKIPPED` are skipped in turn. Outputs that are alternatives share an `exclusive_group`.

```python
from dataclasses import dataclass
from typing import Annotated

from conductor import SKIPPED, Param, node


@dataclass(frozen=True)
class Routed:
    big: Annotated[str, Param(exclusive_group="size")]
    small: Annotated[str, Param(exclusive_group="size")]


@node("route")
def route(text: str) -> Routed:
    return Routed(text, SKIPPED) if len(text) > 3 else Routed(SKIPPED, text)
```

**A person in the loop.** A node raises `Ask`, and its return type says what the answer is. The rest of the graph runs on and the run ends `waiting`, each waiting row in `run.waiting`, with the JSON schema of its answer.

```python
from conductor import Ask, NodeRegistry, make_graph

registry = NodeRegistry()


@registry.node("approve")
def approve(invoice: str) -> bool:
    raise Ask(f"Approve {invoice}?")


graph = registry.load(make_graph(approve, values={"approve.invoice": "INV-1"}))
paused = graph.run()                                     # paused.status == "waiting"
answered = paused.answer(paused.waiting[0], True)        # checked; no node runs
finished = answered.resume()
saved = finished.to_json()                               # graph.load_run(saved) reads it back
```

**Running again.** `run.again(**changed)` runs with changed inputs, the rest as before: a row that receives what it received before is not called again, even if it moved. `run.with_inputs(**changed)` is the same run not yet started, to `resume()` or `stream()`, as `graph.with_inputs` is to `graph.run`. Add one row to a CSV and one mail goes out. A node that reads the world says so with `Policy(reuse_when_inputs_match=False)`, and is called again on every row.

**Versions and retries.** A newer version names the one it replaces, itself a `@node` of the same id: `@node("split", replaces=split_v1, upgrade=Upgrade(renamed_inputs={"sep": "separator"}))`. A stored graph runs the version it was saved at; `registry.upgrade(record, node_id)` moves a node forward, and `Deprecation(...)` marks a version going away. `Policy` sets `concurrency` (rows called at once, on threads), `retries`, `retry_delay`, `attempt_timeout`, `retry_on` and `reuse_when_inputs_match`. Only an `ExternalFailure`, or an exception class named in `retry_on`, is retried.

**Failures.** Every run-time failure is a `Problem` (`type`, `loc`, `msg`, `ctx`), the same record loading reports, on the event and on `run.cause`. A `NodeError` keeps the message its node wrote; any other exception goes to `logging.getLogger("conductor")` with its traceback, and only its class name reaches `ctx`. A failed unit starts nothing more; the calls already on a thread finish and are kept, so resuming doesn't repeat them. A call that can't be acted on raises `InvalidCallError`, with every fault listed.

**Values the host supplies.** A parameter marked `FromRun[T]` is filled by type from `run.resume(from_run={T: value})`, not from an edge.

**Widgets.** A widget is the control a person edits an input with: `TextWidget`, `Textarea`, `Dropdown`, `NumberWidget`, `Switch`, `FileUpload` and more, from `conductor.widgets`. Each is a pydantic model with a `type`. No type has a default widget, so an input names its own.

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
