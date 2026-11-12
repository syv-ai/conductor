---
name: create-graph
description: Writes a conductor graph record (GraphRecord, make_graph), compiles it with Graph.compile and runs it with graph.run. Use when building, saving, running, drawing, serving or debugging a conductor graph; wiring edges, setting constants or defaults, giving graph inputs (with_inputs); reading outputs, problems or events; answering a node that asked a person (run.answer); or on "build a graph", "run a graph", "connect these nodes".
---

# Creating and running a conductor graph

A graph is data, a `GraphRecord`: its nodes, each naming the node definition it uses and its constants and defaults, and its edges. `Graph.compile` compiles it against a registry into a `Graph` that says what is wrong; `graph.run` runs it. To declare a node, use the **add-node** skill. The installed library is the authority: `python -m conductor.about record` (or `compiling`, `one-or-many`, `running`, `events`, `errors`).

```python
from conductor import Graph, GraphRecord, NodeRecord, NodeRegistry, Series

registry = NodeRegistry(category="text")


@registry.node("text-split", title="Split", description="Splits a text at a separator.")
def split(text: str, separator: str = ",") -> Series[str]:
    return text.split(separator)


@registry.node("text-uppercase", title="Uppercase", description="Turns a text into capital letters.")
def uppercase(text: str) -> str:
    return text.upper()


@registry.node("text-join", title="Join", description="Joins the pieces into one text.")
def join(parts: Series[str], separator: str = "") -> str:
    return separator.join(parts)


record = GraphRecord(
    nodes=[
        NodeRecord(id="words", node="text-split"),
        NodeRecord(id="loud", node="text-uppercase"),
        NodeRecord(id="joined", node="text-join", constants={"separator": " + "}),
    ],
    edges=[("words.result", "loud.text"), ("loud.result", "joined.parts")],
)

graph = Graph.compile(record, registry)
if not graph.is_runnable:
    raise ValueError([(p.type, p.loc, p.msg) for p in graph.problems])
assert list(graph.inputs) == ["text", "separator"]     # the graph inputs: the inputs of words, the first node

run = graph.run(text="red,green")
assert run.outputs == {"joined.result": "RED + GREEN"}
results = run.results                                  # {node_id: {output: value}}; a node run per item gives a list
assert results["loud"]["result"] == ["RED", "GREEN"]   # ran once per word
assert graph.runs_on["joined"].mode == "once"         # parts is Series[str]: one call
```

The same record as JSON is `{"nodes": [{"id": "words", "node": "text-split"}, ...], "edges": [["words.result", "loud.text"], ...]}`, what an editor writes and a host stores. From Python, `make_graph(split, uppercase, join, constants={"text-join.separator": " + "})` writes it too: with no `edges` the nodes form a chain, each node's one required input with no constant wired to the one output of the node before it; with `edges=[...]` those are every edge, and nothing is matched by name. A node's id there is its own id, `-1`, `-2` when used twice.

An input takes an edge, a value or a default, never two: `constants` are fixed, `defaults` stay open for a caller, and an input with none of them takes its declared default. The types decide one or many: a `Series[X]` output wired to an input of one `X` runs that node once per item, and everything after it; a `Series` or `Table` input collects them again; an outer value, from a node outside the loop, is the same on every call. A list or a table is one value, so a list given as a graph input or a constant at an input of one value is refused (`invalid_input`, `invalid_value`): a node that returns a `Series`, such as `series-split` in `syv-conductor-nodes`, makes the items. An edge may name one field of a record, `series-split.result.email`. Several edges into one input must be branches of one decision (`several_edges_into_one_input`). A single output is named `result`. A node id is not empty and may not contain `.`, `/`, `@` or `~`.

## Quick reference

| You want | Do (examples in REFERENCE.md) |
|---|---|
| Two graph inputs with one name | `graph.run({"words.text": value})`, by address |
| Events as they happen | `async for event in graph.with_inputs(...).stream(): ...` |
| To answer a node that asked | `run = run.answer(run.waiting[0], answer)`, or `run.answer({waiting.id: answer, ...})` for several, then `run.resume()` |
| A service or the caller inside a node | `graph.run(context=AppContext(caller=caller))`, the class named on the registry, `NodeRegistry(context_type=AppContext)` |
| A bound on a run | `run.resume(timeout=60)`, `run.stream(cancel=asyncio.Event())`, `run.stream(hand_back=asyncio.Event())` to stop and leave the run going for another process, `graph.run(max_items=1000)` |
| To save | `record.model_dump_json()`; a run: `run.to_json()`, `Run.from_json(saved, graph)` |
| A record an editor draws from | `graph.describe()`, and `registry.describe()` for the palette |
| A registry for one request | `per_request = NodeRegistry()`, `per_request.add_subregistry(registry)`, `per_request.add(*definitions)` |
| HTTP routes | `app.include_router(run_router(store=, registry=, caller=, may_see=))` from `conductor_providers.fastapi`; `docs/serving-runs.md` |

## Common mistakes

| Mistake | Fix |
|---|---|
| `try` around `Graph.compile` | it never raises for a bad graph: read `is_runnable` and `problems` |
| Rebuilding the record to change a value per run | `graph.run(**inputs)`, or a `defaults` entry the caller may change |
| A loop node, or a `for` around `run` | wire a `Series` output to the node; the engine runs it once per item |
| A table wired to an input of one value (`type_mismatch`, variant `make_a_series`) | put a node that returns the table's records as a `Series` between them, such as `series-split`, and wire its result or one field of it; to join two tables on a key, wire both whole into one node |
| Items of two Series at one node (`series_not_nested`) | make one inside the other: wire the first into the node that makes the second; or give one whole to a `Series` input |
| `except` around `run` for a pause, error or timeout | they are statuses: read `run.status`, and `match run.state` for an `EndedEarly`'s `cause` |
| `graph.run` in a server's event loop | `async for event in run.stream()`: the nodes are called on a thread of their own |

To debug, read `problems`, then print every event of `run.stream()`. A failure is a `Problem`: `type`, `msg`, `ctx`, and `loc` starts with the node. `graph.runs_on` says how often each node runs.
