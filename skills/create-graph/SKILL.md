---
name: create-graph
description: Writes a conductor graph record (GraphRecord, make_graph), compiles it with CompiledGraph.from_graph and runs it with compiled.run. Use when building, saving, running, drawing, serving or debugging a conductor graph; wiring edges, typing in values or defaults, filling inputs (with_inputs); reading outputs, problems or events; answering a node that asked a person (compiled.answer); or on "build a graph", "run a graph", "connect these nodes".
---

# Creating and running a conductor graph

A graph is data, a `GraphRecord`: its nodes, each naming the node it uses and what is typed into it, and its edges. Compile says what is wrong; `compiled.run` runs it as a stream of events. To declare a node, use the **add-node** skill. The installed library is the authority: `python -m conductor.about record` (or `compilation`, `rows`, `legs`, `events`, `errors`).

```python
from conductor import CompiledGraph, GraphRecord, NodeRecord, NodeRegistry, Series

registry = NodeRegistry(category="text")


@registry.node("text-split")
def split(text: str, separator: str = ",") -> Series[str]:
    return text.split(separator)


@registry.node("text-uppercase")
def uppercase(text: str) -> str:
    return text.upper()


@registry.node("text-join")
def join(parts: Series[str], separator: str = "") -> str:
    return separator.join(parts)


graph = GraphRecord(
    nodes=[
        NodeRecord(id="words", node="text-split"),
        NodeRecord(id="loud", node="text-uppercase"),
        NodeRecord(id="joined", node="text-join", values={"separator": " + "}),
    ],
    edges=[("words.result", "loud.text"), ("loud.result", "joined.parts")],
)

compiled = CompiledGraph.from_graph(graph, registry)
ready = compiled.with_inputs(text="red,green")        # words.text is the graph's open input
if not ready.is_runnable:
    raise ValueError([(p.type, p.loc, p.msg) for p in ready.problems])

state = ready.run_sync()                               # in a notebook: await ready.run()
assert ready.outputs(state) == {"joined.result": "RED + GREEN"}
results = ready.results(state)                         # {node_id: {output: value}}; many values are {"rows": [...], "values": [...]}
assert results["loud"]["result"]["values"] == ["RED", "GREEN"]   # ran once per word
assert ready.run_plan.nodes["joined"].runs_per_row_of is None   # parts is Series[str]: one call
```

The same record as JSON is `{"nodes": [{"id": "words", "node": "text-split"}, ...], "edges": [["words.result", "loud.text"], ...]}`, what an editor writes and a host stores. From Python, `make_graph(split, uppercase, join, values={"text-join.separator": " + "})` writes it too: with no `edges` the nodes form a chain, each taking its one required open input from the one output before it; with `edges=[...]` those are every edge, and nothing is matched by name. A node's id there is its own id, `-1`, `-2` when used twice.

An input takes an edge, a value or a default, never two: `values` are fixed (a list on a single-value input runs the node once per item), `defaults` stay open for a caller, and an input with none of them takes its declared default. Several edges into one input gather in the order their nodes appear in `nodes`. A single output is named `result`. A node id may not contain `.`, `/` or `@`.

## Quick reference

| You want | Do (examples in REFERENCE.md) |
|---|---|
| Two open inputs with one name | `with_inputs(**{"words.text": value})` |
| Events as they happen | `async with compiled.stream() as events: async for event in events: ...` |
| To answer a node that asked | `state = compiled.answer(state, {unit.id: {output: answer}})`, then `compiled.run(state)` |
| A service or the caller inside a node | `compiled.run(from_run={Caller: caller})` |
| A bound on the leg | `compiled.run(timeout=60, cancel=asyncio.Event())` |
| To save | `graph.model_dump_json()`, `GraphRecord.model_validate_json(text)` |
| A record an editor draws from | `compiled.describe()`, and `registry.describe()` for the palette |
| A registry for one request | `per_request = NodeRegistry()`, `per_request.add_subregistry(registry)`, `per_request.add(*loaded)` |
| HTTP routes | `docs/serving-runs.md`, a page to copy onto your own FastAPI application |

## Common mistakes

| Mistake | Fix |
|---|---|
| `try` around `from_graph` | it never raises for a bad graph: read `is_runnable` and `problems` |
| Rebuilding the record to change a value per run | `compiled.with_inputs(...)`; store `ready.graph` as what ran |
| A loop node, or a `for` around `run` | bind a column; the engine runs the node once per row |
| A join node to put columns side by side | wire them all into one `Table` input; they merge on `__row` |
| `except` around `run_sync` for a pause, error or timeout | they are statuses: read `state.status`, `state.cause` |
| `run_sync` in a notebook or server | `await compiled.run()` |

To debug, read `problems`, then print every event of `compiled.stream()`. A failure is a `Problem`: `type`, `msg`, `ctx`, and `loc` is the node and the row. `print(compiled.run_plan)` says where each node runs and what each input receives.
