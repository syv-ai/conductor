<p align="center">
  <img src="https://raw.githubusercontent.com/syv-ai/conductor/main/docs/assets/logo-white-background.png" alt="Conductor logo" width="140">
</p>

<h1 align="center">Conductor</h1>

<p align="center">
  Run graphs of typed Python functions.
</p>

You write Python functions. Someone else wires them together in an editor, and the graph they draw reaches you as JSON. Conductor checks that graph, tells you what's wrong with it, and runs it.

A few things come for free. When a node returns a list, the nodes after it run once per item. A node can stop and ask a person for an answer. And a run is plain JSON, so it can wait for days and pick up again in another process.

Conductor needs only pydantic, polars and typing-extensions. It has no web framework, database or login in it: those are yours.

```bash
uv add syv-conductor
```

## A first graph

Here are three functions, a graph made of them, and a run:

```python
from conductor import Graph, NodeRegistry, Run, Series, make_graph

registry = NodeRegistry()


@registry.node("lines", title="Lines", description="The lines of a text.")
def lines(text: str) -> Series[str]:
    return text.splitlines()


@registry.node("shout", title="Shout", description="Puts a line in capitals and adds a mark.")
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

A node is a function, and its type hints are its interface. The registry is the set of nodes your graphs may use.

`make_graph` wires the nodes into a chain, each one feeding the next. The graph takes `text`, because nothing else gives `lines` a value. It returns `result`, the output of the last node.

## One or many

The types decide how often a node runs. `line: str` takes one value. `Series[str]` is many, and inside your function it's a plain list.

- `lines` returns three texts as a `Series`. `shout` takes one text, so conductor calls it three times, once per line. That's a loop.
- `join` takes a `Series[str]`, so it's called once with all three results, in order. That ends the loop. An item that is `None` is left out, so `len(texts)` counts the items with a value; write `Series[str | None]` to keep every item's place.
- A value from outside the loop is the same on every call. Here that's `mark`, which keeps its default `"!"`.

That's all there is to looping. A plain `list` or a table is one value until a node returns it as a `Series`. The [Split node](https://github.com/syv-ai/conductor/tree/main/packages/conductor-nodes) does that for you.

A node inside a loop gets one item per call, so a `Series` input there can't collect that loop. To compare each item with all of them, collect the items first in a node outside the loop, such as `series-collect` from the same package, and wire its result in.

## Nodes that ask a person

Now a longer example: approving invoices. These nodes go in a registry of their own:

```python
from typing import TypedDict

from conductor import Ask

registry = NodeRegistry()


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

`read` returns a `Series` of invoices, so `approve` runs once per invoice. A small invoice is approved straight away. For a large one, `raise Ask(...)` stops and asks a person. `asks=True` says up front that the node may do that, and the return type, `bool`, says the answer is a yes or no.

You can still call a node like a plain function. That's its unit test.

## The graph as JSON

This is what an editor saves for the invoice graph:

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
print({name: output.type for name, output in graph.outputs.items()})  # {'summary': 'str'}
print(graph.describe().nodes["approve"].per_item_of)  # read.result
```

Each node has an id of its own in the graph, and `node` says which of your functions it uses. They're often the same, but a graph that uses `approve` twice needs two ids.

Each input gets one of three things:

- a `constant`, the same on every call;
- a `graph_input`, a value the caller gives when they start a run;
- `sources`, the edges into it. `{"node": "read", "output": "result", "field": "name"}` is the `name` of each invoice `read` returns.

`outputs` names what the graph returns. A graph's inputs and outputs work like a function's parameters and return value: here it takes `csv` and returns `summary`.

## When something is wrong

A half-finished graph still compiles. Each mistake becomes a problem tied to a place, so an editor can mark it on the node:

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

Each problem also has a `msg` for a person and a `ctx` with the details. To show it in another language, translate by `type`.

Compiling only raises `InvalidCallError` when the JSON isn't a graph record at all, such as a misspelled key.

## Run it

```python
csv = "name,amount\nIda,100\nBo,5000\n"
run = graph.run(csv=csv)

print(run.status)  # waiting
for waiting in run.waiting:
    print(waiting.prompt)
# Approve Bo, 5000 kr.?
```

The node approved Ida's invoice by itself. Bo's waits for a person, and so does `summary`, since it needs every answer. Each entry in `run.waiting` has `questions`, one per output, each with a JSON schema you can draw a form from.

## Save it, and answer later

A run is one JSON text. Keep it next to the graph record, wherever you keep your data:

```python
saved = run.to_json()
```

Days later, in another process, compile the same record, load the run and answer:

```python
run = Run.from_json(saved, Graph.compile(record, registry))
run = run.answer(run.waiting[0], True).resume()

print(run.status)   # complete
print(run.outputs)  # {'summary': '2 of 2 approved'}
```

A run never changes in place. `answer` and `resume` each return a new run, so keep what they return, as `run = ...` does here. A bare `run.answer(...)` leaves `run` waiting.

The answer is checked before anything runs. A wrong one raises `InvalidCallError`, and the run stays as it was. `resume()` carries on from where the run stopped, and no node is called twice for the same invoice.

To stop without an answer, `run.decline(waiting, reason=...)` ends the run as failed. A decline is not an answer: the next `resume()` asks the same question again. To turn one invoice down and let the run go on, answer it `False` (`python -m conductor.about asking-a-person`).

## Run it again with a change

Bo's invoice was for 5500, not 5000. Run again with the corrected CSV:

```python
edited = csv.replace("Bo,5000", "Bo,5500")
run = run.again(csv=edited)

for waiting in run.waiting:
    print(waiting.prompt)
# Approve Bo, 5500 kr.?
```

A call that gets the same arguments as last time isn't made again. Ida's invoice didn't change, so her result is reused, and only Bo is asked again. Adding an invoice works the same way: only the new one is asked. The code that counts is the node's own function: a template or a helper it reads from outside isn't part of it, so after changing one, call the node again with `run.forget("approve")` (`python -m conductor.about running-again`).

```python
run = run.answer(run.waiting[0], True).again(csv=edited + "Eva,7000\n")

for waiting in run.waiting:
    print(waiting.prompt)
# Approve Eva, 7000 kr.?
```

## Where to go next

- [Writing nodes for others](https://github.com/syv-ai/conductor/blob/main/docs/writing-nodes.md): types, controls, branches, questions, errors, versions, and one test for a whole registry.
- [How it works](https://github.com/syv-ai/conductor/blob/main/docs/OVERVIEW.md): how loops, branches and pauses run, on one page.
- [A graph inside a graph](https://github.com/syv-ai/conductor/blob/main/docs/subgraphs.md): using a whole graph as one node of another.
- [Serving runs over HTTP](https://github.com/syv-ai/conductor/blob/main/docs/serving-runs.md): a FastAPI router from `syv-conductor-providers`, on as many workers as you run.
- [`syv-conductor-nodes`](https://github.com/syv-ai/conductor/tree/main/packages/conductor-nodes): ready-made nodes for text, numbers, JSON, patterns, tables and branches.
- [`examples/`](https://github.com/syv-ai/conductor/tree/main/examples): notebooks.
- `python -m conductor.about`: the full reference, shipped in the package. `python -m conductor.about problems` lists every kind of problem, with an example and the fix.
- [`CHANGELOG.md`](https://github.com/syv-ai/conductor/blob/main/CHANGELOG.md): what changed between releases.

The public API is what `conductor`, `conductor.widgets`, `conductor.events` and `conductor.testing` list in `__all__`. It follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). Licensed under Apache-2.0.
