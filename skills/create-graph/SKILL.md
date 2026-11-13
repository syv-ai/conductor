---
name: create-graph
description: Writes a conductor graph record (GraphRecord, make_graph), compiles it with Graph.compile and runs it with graph.run. Use when building, saving, running, drawing, serving or debugging a conductor graph; wiring edges, setting constants, declaring a graph's inputs and outputs, giving graph inputs (with_inputs); reading outputs, problems or events; answering a node that asked a person (run.answer); or on "build a graph", "run a graph", "connect these nodes".
---

# Creating and running a conductor graph

A graph is data, a `GraphRecord`: its nodes by id, each naming the node definition it uses and what each of its inputs gets, and the graph's outputs by name. `Graph.compile` compiles it against a registry into a `Graph` that says what is wrong; `graph.run` runs it. To declare a node, use the **add-node** skill. The installed library is the authority: `python -m conductor.about record` (or `compiling`, `one-or-many`, `running`, `events`, `errors`).

```python
from conductor import Address, Graph, GraphRecord, NodeRecord, NodeRegistry, InputEntry, Series

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
    nodes={
        "words": NodeRecord(node="text-split", inputs={"text": InputEntry(graph_input="text")}),
        "loud": NodeRecord(node="text-uppercase", inputs={"text": InputEntry(sources=(Address("words", "result"),))}),
        "joined": NodeRecord(node="text-join", inputs={
            "parts": InputEntry(sources=(Address("loud", "result"),)),
            "separator": InputEntry(constant=" + "),
        }),
    },
    outputs={"result": Address("joined", "result")},
)

graph = Graph.compile(record, registry)
if not graph.is_runnable:
    raise ValueError([(p.type, p.loc, p.msg) for p in graph.problems])
assert list(graph.inputs) == ["text"]                  # the graph inputs the record declares

run = graph.run(text="red,green")
assert run.outputs == {"result": "RED + GREEN"}       # the graph outputs, by name
results = run.results                                  # {node_id: {output: value}}; a node run per item gives a list
assert results["loud"]["result"] == ["RED", "GREEN"]   # ran once per word
assert graph.runs_on["joined"].mode == "once"         # parts is Series[str]: one call
```

The same record as JSON is `{"nodes": {"words": {"node": "text-split", "inputs": {"text": {"graph_input": "text"}}}, "loud": {"node": "text-uppercase", "inputs": {"text": {"sources": [{"node": "words", "output": "result"}]}}}, ...}, "outputs": {"result": {"node": "joined", "output": "result"}}}`, what an editor writes and a host stores. A graph declares what it takes and returns, as a function declares its parameters and its return value: a graph input is there only where an input's entry names it, and a graph output only in `outputs`. From Python, `make_graph(split, uppercase, join, constants={"text-join.separator": " + "})` writes a record too: with no `edges` the nodes form a chain, each node's one required input with no constant wired to the one output of the node before it; with `edges=[...]` those are every edge, and nothing is matched by name. A node's id there is its own id, `-1`, `-2` when used twice. It declares the graph's inputs and outputs: each required input that gets nothing (no edge, no constant), on any node, named by the input, and each output of a node no edge leaves, named by the output; an optional input that gets nothing takes its own default. `inputs={"text": ["a.text", "b.text"]}` (one graph input feeding both) and `outputs={"long": "join.result"}` name some of them otherwise, on top of that. Two that would share a name raise `graph_inputs_share_a_name` or `graph_outputs_share_a_name`; a text that is not `"node.name"` raises `not_a_place`.

Each input the record lists gets exactly one of a `constant` (the same on every call), a `graph_input` (the caller's value, with an optional `default`) or `sources` (the edges into it); an input left out takes its declared default, and a required one left out is `input_has_no_value`. The types decide one or many: a `Series[X]` output wired to an input of one `X` runs that node once per item, and everything after it; a `Series` or `Table` input collects them again; an outer value, from a node outside the loop, is the same on every call. A list or a table is one value, so a list given as a graph input or a constant at an input of one value is refused (`invalid_input`, `invalid_value`): a node that returns a `Series`, such as `series-split` in `syv-conductor-nodes`, makes the items. A source may name one field of a record, `{"node": "series-split", "output": "result", "field": "email"}` (`"series-split.result.email"` in `make_graph`). Several sources of one input must be branches of one decision (`several_edges_into_one_input`). A single output is named `result`. A node id, an input, an output and a graph port are names: not empty, with no `.`, `/`, `@` or `~` (`invalid_record`).

## Quick reference

| You want | Do (examples in REFERENCE.md) |
|---|---|
| One value for two inputs | name one graph input on both, `{"graph_input": "text"}`; in `make_graph`, `inputs={"text": ["a.text", "b.text"]}` |
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
| Rebuilding the record to change a value per run | `graph.run(**inputs)`, or a graph input with a `default` the caller may change |
| A loop node, or a `for` around `run` | wire a `Series` output to the node; the engine runs it once per item |
| A table wired to an input of one value (`type_mismatch`, variant `make_a_series`) | put a node that returns the table's records as a `Series` between them, such as `series-split`, and wire its result or one field of it; to join two tables on a key, wire both whole into one node |
| Items of two Series at one node (`series_not_nested`) | make one inside the other: wire the first into the node that makes the second; or give one whole to a `Series` input |
| `except` around `run` for a pause, error or timeout | they are statuses: read `run.status`, and `match run.state` for an `EndedEarly`'s `cause` |
| `graph.run` in a server's event loop | `async for event in run.stream()`: the nodes are called on a thread of their own |

To debug, read `problems`, then print every event of `run.stream()`. A failure is a `Problem`: `type`, `msg`, `ctx`, and `loc` starts with the node. `graph.runs_on` says how often each node runs.
