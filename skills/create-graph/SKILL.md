---
name: create-graph
description: Writes a conductor graph record (GraphRecord, make_graph), compiles it with Graph.compile and runs it with graph.run. Use when building, saving, running, drawing, serving or debugging a conductor graph; wiring edges, setting constants, declaring a graph's inputs and outputs, giving graph inputs (with_inputs); reading outputs, problems or events; answering a node that asked a person (run.with_answers); or on "build a graph", "run a graph", "connect these nodes".
---

# Creating and running a conductor graph

A graph is data, a `GraphRecord`. It lists the graph's nodes by id. Each node names the node definition it uses, and says what each of its inputs gets. The record also names the graph's outputs.

`Graph.compile` checks a record against a registry and gives you a `Graph`, which says what is wrong with it. `graph.run` runs it.

To declare a node, use the **add-node** skill. The installed library is the authority on every rule: `python -m conductor.about record` (or `compiling`, `one-or-many`, `running`, `events`, `errors`).

```python
from conductor import Graph, GraphRecord, NodeRegistry, Series

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


record = GraphRecord.model_validate({   # the record's JSON, as an editor saves it
    "inputs": {"text": {}},
    "nodes": {
        "words": {"node": "text-split", "inputs": {"text": {"graph_input": "text"}}},
        "loud": {"node": "text-uppercase", "inputs": {"text": {"sources": [{"node": "words", "output": "result"}]}}},
        "joined": {"node": "text-join", "inputs": {
            "parts": {"sources": [{"node": "loud", "output": "result"}]},
            "separator": {"constant": " + "},
        }},
    },
    "outputs": {"result": {"node": "joined", "output": "result"}},
})

graph = Graph.compile(record, registry)
if not graph.is_runnable:
    raise ValueError([(p.type, p.loc, p.msg) for p in graph.problems])
assert list(graph.inputs) == ["text"]                  # the graph inputs the record declares

run = graph.run({"text": "red,green"})
assert run.outputs == {"result": "RED + GREEN"}       # the graph outputs, by name
results = run.results                                  # {node_id: {output: value}}; a node run per item gives a list
assert results["loud"]["result"] == ["RED", "GREEN"]   # ran once per word
assert graph.describe().nodes["joined"].per_item_of is None  # parts is Series[str]: one call
```

## The record as JSON

An editor writes the same record as JSON, and a host stores that:

```json
{"inputs": {"text": {}},
 "nodes": {"words": {"node": "text-split", "inputs": {"text": {"graph_input": "text"}}},
           "loud": {"node": "text-uppercase", "inputs": {"text": {"sources": [{"node": "words", "output": "result"}]}}},
           ...},
 "outputs": {"result": {"node": "joined", "output": "result"}}}
```

A graph declares what it takes and returns, the way a function declares its parameters and return value. Each graph input is declared once in `inputs`, with its `default` if it has one (`{"text": {"default": "a,b"}}`), and an entry names it. A graph output exists only in `outputs`. A declared graph input no entry names is `graph_input_not_used`, and an entry naming one the record doesn't declare is `graph_input_not_declared`.

## What an input gets

Each input the record lists gets exactly one of:

- a `constant`, the same on every call;
- a `graph_input`, the caller's value, named as the record declares it in `inputs`;
- `sources`, the edges into it.

An input the record leaves out takes its declared default. A required input left out is `input_has_no_value`.

Some more rules for a record:

- A source may name one field of an output's value: `{"node": "series-split", "output": "result", "field": "email"}`. In `make_graph` that's `"series-split.result.email"`.
- Several sources into one input must be different branches of one function node (outputs whose default is `SKIPPED`), or downstream of them. Two outputs of one subgraph node are never branches, so they can't share an input. Otherwise it's `several_edges_into_one_input`.
- A node with a single output names it `result`.
- A node id, an input, an output and a graph port are names: not empty, and with no `.`, `/`, `@` or `~` in them (`invalid_record`).

## One or many

The types decide how often a node runs:

- A `Series[X]` output wired to an input that takes one `X` runs that node once per item, and every node after it too.
- A `Series` or `Table` input collects the items again.
- A value from a node outside the loop is the same on every call.

A list or a table is one value. So a list given as a graph input or a constant, at an input that takes one value, is refused (`invalid_input_list_for_one_value`, `invalid_value`). To make items from it, use a node that returns a `Series`, such as `series-split` in `syv-conductor-nodes`.

## Writing a record from Python with `make_graph`

`make_graph(split, uppercase, join, constants={"text-join.separator": " + "})` writes a record too.

**Edges.**
- With no `edges=`, the nodes form a chain. Each node's one required input that `constants=`, `defaults=` and `inputs=` don't name is wired to the one output of the node before it.
- With `edges=[...]`, those are all the edges. Nothing is matched by name.

**Node ids.** Each node gets its definition's id. A node used twice gets `-1` and `-2` on the end, such as `text-split-1` and `text-split-2`.

**Graph inputs and outputs.** `make_graph` declares them for you:
- Each required input that gets nothing (no edge, no constant), on any node, becomes a graph input named after the input.
- Each output of a node that no edge leaves becomes a graph output named after the output. A node that any edge leaves gives none, not even an output no edge reads, such as an unwired branch: name that one in `outputs=`.
- An optional input that gets nothing keeps its own default.
- `inputs={"text": ["a.text", "b.text"]}` (one graph input feeding both) and `outputs={"long": "join.result"}` name some of them yourself. These are added on top of the automatic ones. An optional input named in `inputs=` keeps its own default.

**What it raises.**
- Two graph inputs that would share a name raise `graph_inputs_share_a_name`. Two outputs raise `graph_outputs_share_a_name`.
- One input given two of a constant, a graph input and edges, such as `edges=` and `inputs=` both naming `shout.text`, raises `input_given_two_ways`.
- A text that isn't `"node.name"`, or whose id or name has a `/`, `@` or `~` in it, raises `invalid_place`. Its message lists the characters a name can't have.

## Quick reference

| You want | Do (examples in REFERENCE.md) |
|---|---|
| One value for two inputs | name one graph input on both, `{"graph_input": "text"}`. In `make_graph`, `inputs={"text": ["a.text", "b.text"]}` |
| Events as they happen | `async for event in graph.with_inputs(...).stream(): ...` |
| To answer a node that asked | `run = run.with_answers(run.waiting[0], answer)`, or `run.with_answers({waiting.id: answer, ...})` for several, then `run.resume()` |
| To refuse to answer it | `run = run.with_decline(run.waiting[0], reason="...")`: the run is `declined` with `answer_declined`, and has ended for good; `run.again()` starts a new run that asks again |
| A service or the caller inside a node | `graph.run(context=AppContext(caller=caller))`, with the class named on the registry: `NodeRegistry(context_type=AppContext)` |
| A limit on a run | `run.resume(timeout=60)`; `run.stream(cancel=asyncio.Event())`; `run.stream(hand_back=asyncio.Event())` to stop and leave the run going for another process; `graph.run(max_items=1000)` |
| To save | `record.model_dump_json()`. A run: `run.to_json()`, and `Run.from_json(saved, graph)` to read it back |
| What an editor draws from | `graph.describe()`, and `registry.describe()` for the palette |
| A registry for one request | `per_request = NodeRegistry()`, `per_request.add_subregistry(registry)`, `per_request.add(*definitions)` |
| HTTP routes | `worker = Worker(store, nodes_for=)`, then `app.include_router(run_router(worker, metadata=, run_filter=))`, both from `conductor_providers.fastapi`. See `docs/serving-runs.md` |
| HTTP routes of your own | the same `Worker`, called from your routes and entered with `worker.serving()` in the lifespan, on `APIRouter(route_class=RefusalRoute)` so they refuse a call as the router does. See `docs/serving-runs.md#your-own-routes` |

## Common mistakes

| Mistake | Fix |
|---|---|
| `try` around `Graph.compile` | it never raises for a bad graph. Read `is_runnable` and `problems` |
| Rebuilding the record to change a value per run | pass it with `graph.run(inputs)`, or use a graph input with a `default` the caller may change |
| A loop node, or a `for` around `run` | wire a `Series` output to the node. The engine runs it once per item |
| A table wired to an input of one value (`type_mismatch_whole_table_or_list`) | put a node between them that returns the table's rows as a `Series`, such as `series-split`, and wire its result or one field of it. To join two tables on a key, wire both whole into one node |
| One value wired to a `Series` or `Table` input (`type_mismatch_series_given_one_value`) | if the node takes one value per call, declare the input as one `T`. If it should get every item of a loop, collect them first in a node outside the loop, such as `series-collect`, and wire its result in |
| A `Series` wired to a `list[T]` input (`type_mismatch_series_into_list`) | declare the input `Series[T]`, which takes every item as one list |
| Items of two Series at one node (`series_not_nested`) | put one inside the other: wire the first into the node that makes the second. Or give one of them whole to a `Series` input |
| `except` around `run` for a pause, an error or a timeout | these are statuses. Read `run.status`, and `run.cause` for why it ended early |
| `graph.run` in a server's event loop | use `async for event in run.stream()`. The nodes are called on a thread of their own |

## Debugging

Read `graph.problems` first, then print every event of `run.stream()`. A failure is a `Problem` with a `type`, `msg` and `ctx`, and its `loc` starts with the node. `graph.describe()` says how often each node runs.
