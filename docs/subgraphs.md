# A graph inside a graph

A graph record can be a node of another graph. An author builds "enrich a contact" once, saves it, and uses it in
every graph that needs it, as one node.

## Make one

`node("id", title=..., description=...)` wraps a function. `subgraph("id", title=..., description=...)` wraps a `GraphRecord` (or its JSON text) the same way:

```python
from conductor import Graph, NodeRegistry, Series, make_graph, node, subgraph

registry = NodeRegistry()


@registry.node("upper", title="Uppercase", description="Turns a text into capital letters.")
def upper(text: str) -> str:
    return text.upper()


@registry.node("exclaim", title="Exclaim", description="Adds an exclamation mark to a text.")
def exclaim(text: str) -> str:
    return text + "!"


@registry.node("split", title="Split", description="Splits a text into words.")
def split(text: str) -> Series[str]:
    return text.split()


@registry.node("join", title="Join", description="Joins the words into one text.")
def join(texts: Series[str]) -> str:
    return " ".join(texts)


shout = subgraph("shout", title="Shout", description="Turns a text into capital letters.")(make_graph(upper, exclaim))
registry.add(shout)
```

`shout` is a `SubgraphDefinition`, the stored-graph kind of `NodeDefinition` (a function makes a `FunctionDefinition`), and it has no function to call. It takes `replaces=`, `version=`, `title=` and the rest, and a newer
version of the subgraph is a newer version of the node. Each version is its own record, so a graph saved against
version 1 keeps running version 1 until [`registry.upgrade`](writing-nodes.md#a-second-version) moves it on.

## Use one

Wire it like any node. Its inputs are its graph's inputs, by their names there, and its outputs are its graph's
outputs, by their addresses there, so an edge reads `shout.exclaim.result`. `make_graph` wants its `edges=` written
out: a chain is drawn only between functions:

```python
graph = Graph.compile(make_graph(split, shout, join, edges=[("split.result", "shout.text"), ("shout.exclaim.result", "join.texts")]), registry)

print(graph.run(text="hi there").outputs)  # {'join.result': 'HI! THERE!'}
print(graph.runs_on["shout"].msg)              # per item of split.result
```

The nodes of a subgraph run where its node stands: here once per word. Embedding a graph changes nothing about
how its nodes behave. Each is called by its own `Policy`, so a node with `concurrency=2` has two calls at once however many items
the graph around it runs for; its retries and its `attempt_timeout` are its own; a cancel and a leg's `timeout` stop
the nodes inside as they stop any node; and `run.again(...)` calls inside only the nodes it would call with the same
nodes drawn in the graph itself. A subgraph takes no `Policy` of its own, because it is never called itself.

## What it offers

From outside, a subgraph is one node:

- **Its inputs** are the open inputs of its first nodes (`graph.inputs` of the subgraph), under the names that
  graph gives them: the input's own name, or `node.input` where two inner nodes share one. A value, a default or an
  edge on it works as on any node. An inner input a hook reads is set inside the subgraph and is not offered.
- **Its outputs** are the outputs of the nodes nothing inside reads (`graph.outputs` of the subgraph), under their
  addresses there, `node.output`.
- An output its graph returns once has the type the inner node gives it. One returned inside loops is one `Series`
  for each loop: an output returned per line of each document is `Series[Series[str]]`, a list of lines per
  document. That is the shape `run.results` shows, and the shape a node wired to it works with, as
  [below](#an-output-returned-inside-two-loops). A `Series` returned once per document is `Series[Series[str]]` the
  same way, and a table returned once per document a `Series` of tables.
- An exclusive group inside is named by the node that decides, `decide.answer`.

```python
side_by_side = subgraph("both", title="Both", description="Runs the capital letters and the exclamation mark side by side.")(make_graph(upper, exclaim, edges=[]))
registry.add(side_by_side)
both = Graph.compile(make_graph(side_by_side), registry)

print(list(both.inputs))  # ['upper.text', 'exclaim.text']
print(both.run({"upper.text": "a", "exclaim.text": "b"}).outputs)
# {'both.upper.result': 'A', 'both.exclaim.result': 'b!'}
```

A node outside reads a subgraph's output as it would read the same node drawn in its own graph. Below, `upper`
and `exclaim` each run once per word inside the subgraph. A node wired to both is called once per word, with
that word's two values:

```python
both_per_word = subgraph("per-word", title="Per word", description="Each word in capital letters, and with an exclamation mark.")(
    make_graph(split, upper, exclaim, edges=[("split.result", "upper.text"), ("split.result", "exclaim.text")])
)
registry.add(both_per_word)


@registry.node("beside", title="Beside", description="Puts two texts side by side.")
def beside(loud: str, keen: str) -> str:
    return f"{loud}/{keen}"


graph = Graph.compile(make_graph(both_per_word, beside, edges=[("per-word.upper.result", "beside.loud"), ("per-word.exclaim.result", "beside.keen")]), registry)
print(graph.run(text="hi there").results["beside"]["result"])  # ['HI/hi!', 'THERE/there!']
print(graph.runs_on["beside"].msg)                             # per item of per-word.split.result
```

The loop is named by the list inside the subgraph that is looped over, `per-word.split.result`. Two outputs
returned in the same loop inside line up outside, as here. Two returned in different loops inside don't: a node
that takes one item of each is refused with `series_not_nested`, as it is for two lists drawn in one graph. Each
output's description lists its `loops`, so an editor can tell which outputs line up.

### An output returned inside two loops

`lines` below is called once per document, and `upper` once per line. Stored as one graph, `upper.result` is a list
of lines for each document, `Series[Series[str]]`:

```python
@registry.node("documents", title="Documents", description="Splits a text into documents at each semicolon.")
def documents(text: str) -> Series[str]:
    return text.split(";")


@registry.node("lines", title="Lines", description="Splits a document into lines at each comma.")
def lines(document: str) -> Series[str]:
    return document.split(",")


per_line = subgraph("per-line", title="Per line", description="Every line of every document, in capital letters.")(
    make_graph(documents, lines, upper, edges=[("documents.result", "lines.document"), ("lines.result", "upper.text")])
)
registry.add(per_line)

graph = Graph.compile(make_graph(per_line, join, edges=[("per-line.upper.result", "join.texts")]), registry)
run = graph.run(text="a,b;c")

print(run.results["per-line"]["upper.result"])  # [['A', 'B'], ['C']]
print(run.results["join"]["result"])            # ['A B', 'C']
print(graph.runs_on["join"].msg)                # per item of per-line.documents.result
print(graph.describe().nodes["per-line"].outputs[0].type)  # Series[str]
```

`join` takes a `Series[str]`, so it collects one loop: it is called once per document, with that document's lines.
A node that takes one `str` would be called once per line. Both are what the same four nodes do drawn in one
graph, and both see the list per document that `run.results` shows. The description says `many` with the type
`Series[str]`, one `Series` for each loop, and its `loops` are `["documents.result", "lines.result"]`.

## A blank or a skip coming in

Nothing is special about a blank that arrives at a subgraph. It reaches the nodes inside that read it, and they
behave as they would drawn in the graph itself:

```python
@registry.node("maybe", title="Maybe", description="Gives a word, or a blank for the word 'none'.")
def maybe(text: str) -> str | None:
    return None if text == "none" else text


graph = Graph.compile(make_graph(split, maybe, shout, edges=[("split.result", "maybe.text"), ("maybe.result", "shout.text")]), registry)
run = graph.run(text="hi none")

print(run.results["shout"])  # {'exclaim.result': ['HI!', None]}
for not_called in run.not_called:
    print(not_called.loc, not_called.input)
# ('shout', 1, 'upper') text
# ('shout', 1, 'exclaim') text
```

The second word is blank. `upper` inside `shout` takes a text and no blank, so it isn't called for that word, and
neither is `exclaim`, which reads `upper`. `run.not_called` names those two nodes. It never names `shout`: a stored
graph is never blank or skipped itself. A branch not taken works the same way and skips the nodes inside that read it.

A node inside that doesn't read the blank runs. `both`, from above, has two inputs. Here only `upper` inside it
gets the blank:

```python
graph = Graph.compile(make_graph(split, maybe, side_by_side, edges=[
    ("split.result", "maybe.text"),
    ("maybe.result", "both.upper.text"),
    ("split.result", "both.exclaim.text"),
]), registry)
print(graph.run(text="hi none").results["both"])  # {'upper.result': ['HI', None], 'exclaim.result': ['hi!', 'none!']}
```

## A subgraph is loaded alone

A subgraph is checked once, before any graph uses it, so it doesn't know what will be wired to it. That matters
when its first node takes any type. `each` gives the records of whatever table arrives. Drawn in a graph, after a
node that gives a table of contacts, it gives contacts. As the first node of a subgraph it has nothing wired to
it, so it gives records of unknown fields:

```python
from typing import TypedDict, cast

from conductor import Table


class Contact(TypedDict):
    name: str


@registry.node("each", title="Each", description="Gives the records of a table, or the items of a list, one at a time.")
def each[T](value: Table[T] | list[T]) -> Series[T]:
    return cast("list[T]", value)


@registry.node("known", title="Known contacts", description="Takes a table of contacts and passes it on.")
def known(contacts: Table[Contact]) -> Table[Contact]:
    return contacts


untyped = subgraph("shout-names", title="Shout names", description="Writes each name in capitals.")(
    make_graph(each, upper, edges=[("each.result.name", "upper.text")])
)
typed = subgraph("shout-contacts", title="Shout contacts", description="Writes each contact's name in capitals.")(
    make_graph(known, each, upper, edges=[("known.result", "each.value"), ("each.result.name", "upper.text")])
)
registry.add(untyped, typed)

palette = {described.id: described for described in registry.describe().nodes}
print(palette["shout-names"].inputs[0].type)     # Table | list
print(palette["shout-contacts"].inputs[0].type)  # Table[name: str]
```

`shout-names` takes any table, so nothing checks `each.result.name` until a run: a misspelled field would load.
`shout-contacts` starts with a node whose input says what it takes. Its edges are checked when it is declared, and a
graph that wires the wrong table to it gets a `type_mismatch`. So give a subgraph a first node with a typed input.

## Where something is inside

`run.results`, `graph.runs_on` and `graph.describe()` show the subgraph as one node. Its results are read from the nodes inside that
make them, as the run stands: it holds none of its own. What happens inside it is told with where it is:
the node and item of each subgraph around it, outermost first.

- A problem's `loc` starts with it. A problem of the subgraph itself is that graph's own problem with the id of
  the node that uses it in front, `("shout", "upper", "text")`; a failure inside for the second word is at
  `("shout", 1, "upper")`.
- A waiting unit and an entry of `run.not_called` have the same `loc`: `("shout", 1, "upper")` is `upper` inside
  `shout`, for the second word.
- A waiting id spells it out: `shout@1/ask~3fa2c0d1` is `ask` inside `shout`, for the second word.
- A node inside says its events once, for every word together, as any node that runs per item does. Their `loc` is
  the node's id behind the id of each subgraph's node around it, `("shout", "upper")`.

A subgraph's palette entry lists the fields it offers in `registry.describe()`, which holds the nodes its
graph uses. A subgraph's definition has no `describe()` of its own, because its fields depend on that registry.

## Graphs saved per request

A host that keeps saved graphs in its database makes their nodes when a request needs them. They go in a registry of
the request's own, so the shared one stays as it was:

```python
# your table of saved graphs: each stored with the title and description a person reads for it
saved_graphs = {"shout": ("Shout", "Writes the text in capitals and adds an exclamation mark.", make_graph(upper, exclaim))}

request_registry = NodeRegistry()
request_registry.add_subregistry(registry)
request_registry.add(*(
    subgraph(name, title=title, description=description)(record)
    for name, (title, description, record) in saved_graphs.items()
    if name not in request_registry
))
```

A graph that embeds one of them then loads through `request_registry`. The loaded `Graph` keeps its own copy of what
it resolved, so the request's registry can go when the request does.
