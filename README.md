<p align="center">
  <img src="https://raw.githubusercontent.com/syv-ai/conductor/main/docs/assets/logo-white-background.png" alt="Conductor logo" width="140">
</p>

<h1 align="center">Conductor</h1>

<p align="center">
  Run graphs of typed Python functions.
</p>

You write the functions. Someone else draws the graph in an editor, and it reaches you as JSON. Conductor loads it, tells you what's wrong with it, and runs it: once per row wherever a table arrives, pausing where a node asks a person. A run is JSON too, so it can wait days for an answer.

The core depends on pydantic and polars. It has no web framework, database or login in it: those are yours.

```bash
uv add syv-conductor
```

## The shortest graph

Two functions, a graph of them, a run:

```python
from conductor import NodeRegistry, Series, make_graph

registry = NodeRegistry()


@registry.node("upper", title="Uppercase", description="Turns a text into capital letters.")
def upper(text: str) -> str:
    return text.upper()


@registry.node("join", title="Join", description="Joins the pieces into one text.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


graph = registry.load(make_graph(upper, join))
print(graph.run(text=["a", "b"]).outputs)  # {'join.result': 'A+B'}
```

`upper` takes one text, so given a list it runs once per item. `join` takes a `Series[str]`, a plain list of every value, so it runs once. That is all iteration is. `make_graph` writes the graph from Python; an editor writes the same thing as JSON, as the rest of this page shows.

## Write nodes

A node is a function, and its type hints are its interface:

```python
import io
from typing import TypedDict

import polars as pl
from conductor import Ask, NodeRegistry, Series, Table

registry = NodeRegistry()  # the invoice nodes get a registry of their own


class Invoice(TypedDict):
    name: str
    amount: int


@registry.node("read", title="Read", description="Reads invoices from a CSV text.")
def read(csv: str) -> Table[Invoice]:
    return pl.read_csv(io.StringIO(csv))


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

A node can still be called like the function, so that is its unit test. The registry holds the nodes your graphs may use.

`Table[Invoice]` is a `polars.DataFrame` with those columns, and `Series[bool]` is a plain list. `raise Ask(...)` stops and asks a person. The return type, `bool`, says what they answer: yes or no.

## Load a graph

This is what an editor saves. Each node has an `id`, which says where it sits in this graph, and a `node`, which says which of your definitions it uses. The two are often the same, but a graph that uses `approve` twice has two ids. Each edge goes from an output to an input, and `read.result.name` is the `name` column of the table `read` returns:

```python
drawing = """{
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

graph = registry.load(drawing)
print(list(graph.inputs))  # ['csv']
print(graph.outputs)       # ['summary.result']
print({node_id: runs.msg for node_id, runs in graph.plan.items()})  # {'read': 'once', 'approve': 'per row of read.result', 'summary': 'once'}
```

Nobody declares what the graph takes or returns. Its inputs are the open inputs of its first nodes, and its outputs are the outputs of the nodes no node reads. `approve` takes one value and is fed a column, so it runs once per row.

## Show problems

A half-finished drawing still loads. Everything wrong is a problem with a place, so an editor can mark it on the node. Only JSON that isn't a graph record at all, such as a misspelled key, raises `InvalidCallError`:

```python
unfinished = registry.load("""{
  "nodes": [
    {"id": "read", "node": "read"},
    {"id": "approve", "node": "approve"},
    {"id": "mail", "node": "send-mail"}
  ],
  "edges": [["read.result.name", "approve.name"]]
}""")

print(unfinished.is_runnable)  # False
for problem in unfinished.problems:
    print(problem.loc, problem.type)
# ('mail',) unknown_node
# ('approve', 'amount') unbound_on_later_node
```

Each problem also has a `msg` for a person and a `ctx` with the details. A host that shows another language translates by `type`.

## Run it

```python
csv = "name,amount\nIda,100\nBo,5000\n"
run = graph.run(csv=csv)

print(run.status)  # waiting
for unit in run.waiting:
    print(unit.row, unit.prompt)
# (1,) Approve Bo, 5000 kr.?
```

Ida's invoice was approved by the node itself. Bo's row waits for a person, and so does `summary`, which reads it. Each waiting row has `questions`, one per output, with a JSON schema to draw a form from.

## Store it

A run is one JSON value. Keep it next to the graph, wherever you keep your data:

```python
saved = run.to_json()
```

## Answer, later

Days later, in another process, load the same graph, read the run back and answer:

```python
run = registry.load(drawing).load_run(saved)
run = run.answer(run.waiting[0], True).resume()

print(run.status)   # complete
print(run.outputs)  # {'summary.result': '2 of 2 approved'}
```

The answer is checked against its question before anything runs. A wrong one raises `InvalidCallError`, and the run stays as it was. `resume()` goes on from where the run stopped, and no node is called twice for a row.

## Edit one row

Bo's invoice was for 5500, not 5000. Run again with the corrected CSV:

```python
edited = csv.replace("Bo,5000", "Bo,5500")
run = run.again(csv=edited)

for unit in run.waiting:
    print(unit.prompt)
# Approve Bo, 5500 kr.?
```

A row that gets what it got before isn't called again. Ida's row is the same, so it keeps its result, and only Bo's row is asked again. Adding a row works the same way: only the new row is asked.

```python
run = run.answer(run.waiting[0], True).again(csv=edited + "Eva,7000\n")

for unit in run.waiting:
    print(unit.prompt)
# Approve Eva, 7000 kr.?
```

`run.with_inputs(...)` is `again` before it starts: the changed inputs and the earlier work, for when you want to `stream()` its events.

## Next

- [Writing nodes for others](https://github.com/syv-ai/conductor/blob/main/docs/writing-nodes.md): types, controls, branches, questions, errors, versions, and one test for a whole registry.
- [How it works](https://github.com/syv-ai/conductor/blob/main/docs/OVERVIEW.md): rows, legs and pauses on one page.
- [A graph inside a graph](https://github.com/syv-ai/conductor/blob/main/docs/embedded-graphs.md): a saved graph used as one node of another.
- [Problem codes](https://github.com/syv-ai/conductor/blob/main/docs/problems.md): every code, with an example and the fix.
- [Serving runs over HTTP](https://github.com/syv-ai/conductor/blob/main/docs/serving-runs.md): a FastAPI router from `syv-conductor-providers`, on as many workers as you run.
- [`syv-conductor-nodes`](https://github.com/syv-ai/conductor/tree/main/packages/conductor-nodes): ready nodes for text, numbers, JSON, patterns and a decision.
- [`examples/`](https://github.com/syv-ai/conductor/tree/main/examples): notebooks.
- `python -m conductor.about`: the full reference, shipped in the package.
- [`CHANGELOG.md`](https://github.com/syv-ai/conductor/blob/main/CHANGELOG.md): what changed between releases.

The public API is what `conductor`, `conductor.widgets`, `conductor.events` and `conductor.testing` list in `__all__`, and it follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Licensed under Apache-2.0.
