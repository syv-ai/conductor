<p align="center">
  <img src="https://raw.githubusercontent.com/syv-ai/conductor/main/docs/assets/logo-white-background.png" alt="Conductor logo" width="140">
</p>

<h1 align="center">Conductor</h1>

<p align="center">
  Run graphs of typed Python functions.
</p>

You write the functions. Someone else draws the graph in an editor, and it reaches you as JSON. Conductor compiles it, tells you what's wrong with it, and runs it: a node runs once per item wherever it is given a list, and pauses where it asks a person. A run is JSON too, so it can wait days for an answer.

The core depends on pydantic, which checks every value, and polars, which holds tables. It has no web framework, database or login in it: those are yours, the host's. The host is your application: it compiles graphs, runs them, stores runs and shows people their questions.

```bash
uv add syv-conductor
```

## The shortest graph

Three functions, a graph of them, a run:

```python
from conductor import Graph, NodeRegistry, Run, Series, make_graph

registry = NodeRegistry()


@registry.node("lines", title="Lines", description="The lines of a text.")
def lines(text: str) -> Series[str]:
    return text.splitlines()


@registry.node("shout", title="Shout", description="Capitals, and a mark.")
def shout(line: str, mark: str = "!") -> str:
    return line.upper() + mark


@registry.node("join", title="Join", description="Joins every text it is given.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


graph = Graph.compile(make_graph(lines, shout, join), registry)
run = graph.run(text="a\nb\nc")
print(run.results["shout"]["result"])  # ['A!', 'B!', 'C!']
print(run.outputs)                     # {'join.result': 'A!+B!+C!'}
```

`lines` returns three texts. `shout` takes one, so it was called three times, once per text. `join` takes all of them, so it was called once. `make_graph` writes the graph record from Python and returns it, a `GraphRecord`. Given no edges, it chains the nodes in order: each node's one required input (`line`; `mark` has a default) gets an edge from the one output of the node before it, here `result`, the name of a function's return value. A node with two required inputs needs its edges written out, as (output, input) address pairs: `edges=[("lines.result", "shout.line")]`. An editor writes the same record as JSON, as the rest of this page shows. `Graph.compile` checks the record against your registry and works out how often each node runs. `run.results` is every node's outputs, by node id and then output name; `run.outputs` is only the graph's outputs, those of its last nodes, by address `node.output`.

## One or many

A node's types say how many values it deals with: `line: str` is one, `Series[str]` is many. To your function a `Series[str]` is a plain list. Each value in it is an item.

- Wire a `Series[T]` output to an input that takes one `T`, and that node runs once per item: a loop. So does every node downstream of it.
- Wire the per-item outputs to a `Series[T]` input, and that node is called once with all of them as a list, in order: it collects the loop. A `Table[R]` input, a whole table that your function gets as a polars `DataFrame` with the columns a `TypedDict` `R` names, collects records into one table the same way ([tables](https://github.com/syv-ai/conductor/blob/main/docs/writing-nodes.md#a-table)).
- A value from outside the loop is the same on every call inside it, as `mark` is for each line.

That is all iteration is. You never write the loop. A `list` or a table is one value until a node returns it as a `Series`: `graph.run(text=["a", "b"])` raises `InvalidCallError`, because `text` takes one text. The [Split node](https://github.com/syv-ai/conductor/tree/main/packages/conductor-nodes) returns the records of a table, or the items of a list, as a `Series` for you.

## Write nodes

A node is a function, and its type hints are its interface:

```python
from typing import TypedDict

from conductor import Ask

registry = NodeRegistry()  # the invoice nodes get a registry of their own


class Invoice(TypedDict):
    name: str
    amount: int


@registry.node("read", title="Read", description="Reads invoices from a CSV text.")
def read(csv: str) -> Series[Invoice]:
    found = [line.split(",") for line in csv.splitlines()[1:]]
    return [Invoice(name=name, amount=int(amount)) for name, amount in found]


@registry.node("approve", title="Approve", description="Approves an invoice, or asks a person to.")
def approve(name: str, amount: int) -> bool:
    if amount < 1000:
        return True
    raise Ask(f"Approve {name}, {amount} kr.?")


@registry.node("summary", title="Summary", description="Tells how many invoices were approved.")
def summary(approved: Series[bool]) -> str:
    return f"{sum(approved)} of {len(approved)} approved"


print(approve("Ida", 100))  # True
```

A node can still be called like the function, so that is its unit test. The registry is the set of nodes your graphs may use.

`read` returns a `Series` of invoices, each a record with a `name` and an `amount`. `raise Ask(...)` stops this call and asks a person. The return type, `bool`, says what they answer: yes or no.

## Compile a graph

This is the graph record an editor saves. Each node has an `id` of its own in this graph, and a `node`, the id of the function it uses. The two are often the same, but a graph that uses `approve` twice has two ids. Each edge goes from an output to an input, written `node.output` and `node.input`, and `read.result.name` is the `name` field of each invoice `read` returns:

```python
record = """{
  "nodes": [
    {"id": "read", "node": "read"},
    {"id": "approve", "node": "approve"},
    {"id": "summary", "node": "summary"}
  ],
  "edges": [
    ["read.result.name", "approve.name"],
    ["read.result.amount", "approve.amount"],
    ["approve.result", "summary.approved"]
  ]
}"""

graph = Graph.compile(record, registry)
print(list(graph.inputs))  # ['csv']
print(graph.outputs)       # ['summary.result']
print({node_id: runs.msg for node_id, runs in graph.runs_on.items()})  # {'read': 'once', 'approve': 'per item of read.result', 'summary': 'once'}
```

Nobody declares what the graph takes or returns. Its inputs are the inputs no edge reaches on the nodes that come first (here `read.csv`, which the caller gives), and its outputs are the outputs of the nodes that come last. `approve` takes one name and one amount, so it runs once per invoice. `summary` takes every answer as a list, so it runs once.

## Show problems

A half-finished graph still compiles. Compiling never raises for a wrong graph: everything wrong is a problem in `graph.problems`, with a `loc` that says which node and input it is on, so an editor can mark it there. Running a graph with problems raises `InvalidCallError`, listing them. Only JSON that isn't a graph record at all, such as a misspelled key, raises `InvalidCallError`:

```python
unfinished = Graph.compile("""{
  "nodes": [
    {"id": "read", "node": "read"},
    {"id": "approve", "node": "approve"},
    {"id": "mail", "node": "send-mail"}
  ],
  "edges": [["read.result.name", "approve.name"]]
}""", registry)

print(unfinished.is_runnable)  # False
for problem in unfinished.problems:
    print(problem.loc, problem.type)
# ('approve', 'amount') unbound_on_later_node
# ('mail',) unknown_node
```

`approve.amount` has no edge, and `approve` has an incoming edge, so it is not a first node and the caller can't give the value either: nothing can. Wiring `read.result.amount` to it fixes it. `send-mail` is not in the registry. Each problem also has a `msg` for a person and a `ctx` with the details. A host that shows another language translates by `type`.

## Run it

```python
csv = "name,amount\nIda,100\nBo,5000\n"
run = graph.run(csv=csv)

print(run.status)  # waiting
for waiting in run.waiting:
    print(waiting.prompt)
# Approve Bo, 5000 kr.?
```

Ida's invoice was approved by the node itself. Bo's waits for a person, and so does `summary`, which needs every answer. Each entry of `run.waiting` is one call that asked: its `id`, its `prompt`, and its `questions`, one per output of the node. `approve` has one output, so it has one question, whose `schema` is `{"type": "boolean"}`: a host draws a yes-or-no from it. A node with several outputs asks a form, one question per output, and is answered with a dict by output name. `run.status` is `waiting` here. It can also be `ready` (not started), `running` (in a checkpoint saved while `resume()` is still calling nodes), `complete`, `failed` (a node raised), `stopped` (a series was longer than the item limit, `graph.run(..., max_items=100)`), `cancelled` (the host cancelled a stream) or `timed_out` (`resume(timeout=...)`, in seconds, ran out).

## Store it

`run.to_json()` returns one JSON text (a `str`): the run's inputs, and what every call returned or asked. Keep it next to the graph record, wherever you keep your data. To read it back you need the same graph record compiled again: a saved run of another graph is refused.

```python
saved = run.to_json()
```

## Answer, later

Days later, in another process, compile the same graph record, read the run back and answer:

```python
run = Run.from_json(saved, Graph.compile(record, registry))
run = run.answer(run.waiting[0], True).resume()

print(run.status)   # complete
print(run.outputs)  # {'summary.result': '2 of 2 approved'}
```

`answer` checks the answer against its question and returns a new run; nothing is called until `resume()`, which blocks until it is done and returns the run. A wrong answer raises `InvalidCallError`, and the run stays as it was. `resume()` calls every node that is ready, until the run is complete, waits for a person again, fails or is stopped. No node is called twice for an invoice.

## Change one invoice

Bo's invoice was for 5500, not 5000. Run again with the corrected CSV:

```python
edited = csv.replace("Bo,5000", "Bo,5500")
run = run.again(csv=edited)

for waiting in run.waiting:
    print(waiting.prompt)
# Approve Bo, 5500 kr.?
```

`again` starts a new run with the changed inputs and reuses what it can from the previous run: a node isn't called again for an item when it would get the same arguments as a call in the previous run, wherever that item was in the list. A person's answers are reused the same way. Ida's invoice is the same, so her call's output is reused, and only Bo's is asked again. Adding an invoice works the same way: Bo's answer is reused, and only the added one is asked.

```python
run = run.answer(run.waiting[0], True).again(csv=edited + "Eva,7000\n")

for waiting in run.waiting:
    print(waiting.prompt)
# Approve Eva, 7000 kr.?
```

`run.with_inputs(...)` is `again` without starting it: a run with the changed inputs that reuses the previous run's outputs. Start it with `resume()`, or with `async for event in run.stream():` to follow each node as it starts, finishes or fails (the events are in `conductor.events`).

## Next

- [Writing nodes for others](https://github.com/syv-ai/conductor/blob/main/docs/writing-nodes.md): types, the widgets an editor shows for an input, outputs that are alternatives (branches), questions, errors, versions, and one test for a whole registry.
- [How it works](https://github.com/syv-ai/conductor/blob/main/docs/OVERVIEW.md): loops, branches and pauses on one page.
- [A graph inside a graph](https://github.com/syv-ai/conductor/blob/main/docs/subgraphs.md): a graph record used as one node of another graph, a subgraph.
- `python -m conductor.about problems`: every problem code, with an example and the fix, printed from the code that declares it.
- [Serving runs over HTTP](https://github.com/syv-ai/conductor/blob/main/docs/serving-runs.md): a FastAPI router from `syv-conductor-providers` that starts, follows and answers runs, on any number of server processes sharing one database.
- [`syv-conductor-nodes`](https://github.com/syv-ai/conductor/tree/main/packages/conductor-nodes): ready nodes for text, numbers, JSON, patterns, tables, and one that chooses a branch.
- [`examples/`](https://github.com/syv-ai/conductor/tree/main/examples): notebooks.
- `python -m conductor.about`: the full reference, shipped in the package.
- [`CHANGELOG.md`](https://github.com/syv-ai/conductor/blob/main/CHANGELOG.md): what changed between releases.

The public API is what `conductor`, `conductor.widgets`, `conductor.events` and `conductor.testing` list in `__all__`, and it follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Licensed under Apache-2.0.
