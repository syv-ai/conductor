<p align="center">
  <img src="https://raw.githubusercontent.com/syv-ai/conductor/main/docs/assets/logo-white-background.png" alt="Conductor logo" width="140">
</p>

<h1 align="center">Conductor</h1>

<p align="center">
  Run graphs of typed Python functions.
</p>

You write the functions. Someone else draws the graph in an editor, and it reaches you as JSON. Conductor compiles it, tells you what's wrong with it, and runs it: once per item wherever a node returns a list, pausing where a node asks a person. A run is JSON too, so it can wait days for an answer.

The core depends on pydantic, polars and typing-extensions. It has no web framework, database or login in it: those are yours.

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
print(run.outputs)                     # {'result': 'A!+B!+C!'}
```

`lines` returns three texts. `shout` takes one, so it was called three times. `join` takes all of them, so it was called once. `make_graph` writes the graph record from Python, and declares what the graph takes and returns: `text`, the one required input nothing in the graph gives a value, and `result`, the output of the node no edge leaves. An editor writes the same thing as JSON, as the rest of this page shows.

## One or many

A node's types say how many values it deals with: `line: str` is one, `Series[str]` is many. To your function a `Series[str]` is a plain list.

- Wire a `Series[T]` output to an input that takes one `T`, and that node runs once per item: a loop. So does every node downstream of it.
- Wire per-item outputs to a `Series[T]` input, and that node runs once and gets all of them, in order: it collects the loop. A `Table` input collects records the same way.
- A value from outside the loop is the same on every call, as `mark` is for each line.

That is all iteration is. A list or a table is one value until a node returns it as a `Series`: `graph.run(text=["a", "b"])` is refused, because `text` takes one text. The [Split node](https://github.com/syv-ai/conductor/tree/main/packages/conductor-nodes) takes a table or a list apart for you.

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


@registry.node("approve", title="Approve", description="Approves an invoice, or asks a person to.", asks=True)
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

`read` returns a `Series` of invoices, each a record with a `name` and an `amount`. `raise Ask(...)` stops and asks a person, and `asks=True` says the node may. The return type, `bool`, says what they answer: yes or no.

## Compile a graph

This is the graph record an editor saves. Its nodes are keyed by an id of their own in this graph, and each has a `node`, which says which of your definitions it uses. The two are often the same, but a graph that uses `approve` twice has two ids. Each input gets one of three things: a `constant`, a `graph_input` the caller gives, or the `sources` of the edges into it. A source is an output of a node, and `{"node": "read", "output": "result", "field": "name"}` is the `name` of each invoice `read` returns. `outputs` names what the graph returns:

```python
record = """{
  "nodes": {
    "read": {"node": "read", "inputs": {"csv": {"graph_input": "csv"}}},
    "approve": {"node": "approve", "inputs": {
      "name": {"sources": [{"node": "read", "output": "result", "field": "name"}]},
      "amount": {"sources": [{"node": "read", "output": "result", "field": "amount"}]}
    }},
    "summary": {"node": "summary", "inputs": {"approved": {"sources": [{"node": "approve", "output": "result"}]}}}
  },
  "outputs": {"summary": {"node": "summary", "output": "result"}}
}"""

graph = Graph.compile(record, registry)
print(list(graph.inputs))  # ['csv']
print(graph.outputs)       # ['summary']
print(graph.describe().nodes["approve"].per_item_of)  # read.result
```

A graph declares what it takes and returns, as a function declares its parameters and its return value: `csv` is a graph input, and `summary` a graph output. Used as a node of another graph, those are its inputs and outputs. `approve` takes one name and one amount, so it runs once per invoice. `summary` takes every answer, so it runs once.

## Show problems

A half-finished graph still compiles. Everything wrong is a problem with a place, so an editor can mark it on the node. Only JSON that isn't a graph record at all, such as a misspelled key, raises `InvalidCallError`:

```python
unfinished = Graph.compile("""{
  "nodes": {
    "read": {"node": "read", "inputs": {"csv": {"graph_input": "csv"}}},
    "approve": {"node": "approve", "inputs": {"name": {"sources": [{"node": "read", "output": "result", "field": "name"}]}}},
    "mail": {"node": "send-mail"}
  }
}""", registry)

print(unfinished.is_runnable)  # False
for problem in unfinished.problems:
    print(problem.loc, problem.type)
# ('approve', 'amount') input_has_no_value
# ('mail',) unknown_node
```

Each problem also has a `msg` for a person and a `ctx` with the details. A host that shows another language translates by `type`.

## Run it

```python
csv = "name,amount\nIda,100\nBo,5000\n"
run = graph.run(csv=csv)

print(run.status)  # waiting
for waiting in run.waiting:
    print(waiting.prompt)
# Approve Bo, 5000 kr.?
```

Ida's invoice was approved by the node itself. Bo's waits for a person, and so does `summary`, which takes every answer. Each entry of `run.waiting` has `questions`, one per output, with a JSON schema to draw a form from.

## Store it

A run is one JSON text. Keep it next to the graph record, wherever you keep your data:

```python
saved = run.to_json()
```

## Answer, later

Days later, in another process, compile the same graph record, read the run back and answer:

```python
run = Run.from_json(saved, Graph.compile(record, registry))
run = run.answer(run.waiting[0], True).resume()

print(run.status)   # complete
print(run.outputs)  # {'summary': '2 of 2 approved'}
```

The answer is checked against its question before anything runs. A wrong one raises `InvalidCallError`, and the run stays as it was. `resume()` goes on from where the run stopped, and no node is called twice for an invoice.

## Change one invoice

Bo's invoice was for 5500, not 5000. Run again with the corrected CSV:

```python
edited = csv.replace("Bo,5000", "Bo,5500")
run = run.again(csv=edited)

for waiting in run.waiting:
    print(waiting.prompt)
# Approve Bo, 5500 kr.?
```

A node isn't called again for an item that gets the same arguments as in the previous run. Ida's invoice is the same, so her output is reused, and only Bo's is asked again. Adding an invoice works the same way: only the added one is asked.

```python
run = run.answer(run.waiting[0], True).again(csv=edited + "Eva,7000\n")

for waiting in run.waiting:
    print(waiting.prompt)
# Approve Eva, 7000 kr.?
```

`run.with_inputs(...)` is `again` before it starts: the changed inputs and what it reuses from the previous run, for when you want to `stream()` its events.

## Next

- [Writing nodes for others](https://github.com/syv-ai/conductor/blob/main/docs/writing-nodes.md): types, controls, branches, questions, errors, versions, and one test for a whole registry.
- [How it works](https://github.com/syv-ai/conductor/blob/main/docs/OVERVIEW.md): loops, branches and pauses on one page.
- [A graph inside a graph](https://github.com/syv-ai/conductor/blob/main/docs/subgraphs.md): a graph record used as one node of another, a subgraph.
- `python -m conductor.about problems`: every problem code, with an example and the fix, printed from the code that declares it.
- [Serving runs over HTTP](https://github.com/syv-ai/conductor/blob/main/docs/serving-runs.md): a FastAPI router from `syv-conductor-providers`, on as many workers as you run.
- [`syv-conductor-nodes`](https://github.com/syv-ai/conductor/tree/main/packages/conductor-nodes): ready nodes for text, numbers, JSON, patterns, tables and a decision.
- [`examples/`](https://github.com/syv-ai/conductor/tree/main/examples): notebooks.
- `python -m conductor.about`: the full reference, shipped in the package.
- [`CHANGELOG.md`](https://github.com/syv-ai/conductor/blob/main/CHANGELOG.md): what changed between releases.

The public API is what `conductor`, `conductor.widgets`, `conductor.events` and `conductor.testing` list in `__all__`, and it follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Licensed under Apache-2.0.
