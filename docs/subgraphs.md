# A graph inside a graph

A graph record can be a node of another graph: a subgraph. An author builds "enrich a contact" once, saves it, and uses it in
every graph that needs it, as one node. The other graph is the outer graph, and the nodes of the subgraph are its inner nodes.

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


shout = subgraph("shout", title="Shout", description="Turns a text into capital letters and adds an exclamation mark.")(make_graph(upper, exclaim))
registry.add(shout)
```

`shout` is a `SubgraphDefinition`, the subgraph kind of `NodeDefinition` (a function makes a `FunctionDefinition`), and it has no function to call. It takes the same arguments as `@node`: `version=`, `replaces=` (the previous version's definition), `title=` and the rest, and a newer
version of the subgraph is a newer version of the node. Each version is its own graph record, so a graph record that names
version 1 keeps running version 1 until [`registry.upgrade`](writing-nodes.md#a-second-version) rewrites it to use version 2.

## Use one

Wire it like any node. Its inputs and outputs are the ones its graph declares, under their names there. `make_graph(upper,
exclaim)` declared `text`, the input of `upper`, and `result`, the output of `exclaim`, so an edge from it reads
`shout.result`. `make_graph` wants its `edges=` written out: a chain from or to a subgraph raises `not_a_chain`:

```python
graph = Graph.compile(make_graph(split, shout, join, edges=[("split.result", "shout.text"), ("shout.result", "join.texts")]), registry)

print(graph.run(text="hi there").outputs)  # {'result': 'HI! THERE!'}
print(graph.runs_on["shout"].msg)              # per item of split.result
```

`runs_on[...].msg` says in words how often a node runs. The subgraph node is inside the loop over `split.result`, so its inner nodes
are called once per word. Using a graph as a subgraph changes nothing about how its nodes behave: they run as they
would if you drew them in the outer graph. Each inner node is called by its own `Policy` (retries, timeouts and how many calls
run at once), so a node with `concurrency=2` has two calls at once however many items the outer graph has. A cancel, and the `timeout` of
`resume()`, stop the inner nodes as they stop any node. `run.again(...)` reuses the previous run's outputs for inner nodes
as it does for any node. A subgraph takes no `Policy` of its own, because it is never called itself: only its inner nodes are.

## What it offers

From outside, a subgraph is one node:

- **Its inputs** are its graph's inputs (`graph.inputs` of the subgraph compiled alone): the graph inputs its record declares,
  under their names there. A constant, a graph input or an edge on one works as on any node. An inner input that a hook reads (a function that
  computes a node's inputs or outputs when the graph is compiled, see [writing nodes](writing-nodes.md#inputs-and-outputs-that-a-constant-decides))
  must be a constant inside the subgraph, so it is not offered.
- **Its outputs** are its graph's outputs (`graph.outputs` of the subgraph): the outputs its record's `outputs` names, under
  their names there.
- An output its graph returns once has the inner node's type. One returned inside loops is one `Series`
  for each loop: an output returned per line of each document is `Series[Series[str]]`, a list of lines per
  document. That is the shape `run.results` shows, and the shape a node wired to it works with, as
  [below](#an-output-returned-inside-two-loops). A `Series` returned once per document is `Series[Series[str]]` the
  same way, and a table returned once per document a `Series` of tables.
- Branches (outputs of one inner node that share an `exclusive_group`, of which a call returns only one) stay
  branches. In the subgraph's description their group is named by that inner node, so the group `answer` of an inner node `decide`
  is `decide.answer`.
- Two outputs downstream of opposite branches of one inner node may be wired into one input outside, as they could
  be if the nodes were drawn in the outer graph: each call gets the value from the branch that was chosen. Wired
  into two `Series` inputs of one node, one per branch, they are refused, as they would be in the outer graph
  (`lists_from_different_loops`): one branch's list would always be empty.

```python
side_by_side = subgraph("both", title="Both", description="Runs the capital letters and the exclamation mark side by side.")(
    make_graph(
        upper, exclaim, edges=[],
        inputs={"loud": "upper.text", "keen": "exclaim.text"},
        outputs={"loud": "upper.result", "keen": "exclaim.result"},
    )
)
registry.add(side_by_side)
both = Graph.compile(make_graph(side_by_side), registry)

print(list(both.inputs))  # ['keen', 'loud']
print(both.run(loud="a", keen="b").outputs)  # {'keen': 'b!', 'loud': 'A'}
```

`upper` and `exclaim` both take `text` and both return `result`, so `make_graph` is told what the graph calls them, as
it would raise `graph_inputs_share_a_name` and `graph_outputs_share_a_name` otherwise.

A node outside gets a subgraph's output as it would get the same node's output if that node were drawn in the outer graph. Below, `upper`
and `exclaim` each run once per word inside the subgraph. A node wired to both is called once per word, with
that word's two values:

```python
both_per_word = subgraph("per-word", title="Per word", description="Each word in capital letters, and with an exclamation mark.")(
    make_graph(
        split, upper, exclaim,
        edges=[("split.result", "upper.text"), ("split.result", "exclaim.text")],
        outputs={"loud": "upper.result", "keen": "exclaim.result"},
    )
)
registry.add(both_per_word)


@registry.node("beside", title="Beside", description="Puts two texts side by side.")
def beside(loud: str, keen: str) -> str:
    return f"{loud}/{keen}"


graph = Graph.compile(make_graph(both_per_word, beside, edges=[("per-word.loud", "beside.loud"), ("per-word.keen", "beside.keen")]), registry)
print(graph.run(text="hi there").results["beside"]["result"])  # ['HI/hi!', 'THERE/there!']
print(graph.runs_on["beside"].msg)                             # per item of per-word/split.result
```

The loop is named by the path of the series inside the subgraph that is looped over, `per-word/split.result`: the node `split` inside `per-word`, and its output. Two outputs
returned in the same loop inside line up outside, item by item, as here. Two returned in different loops inside don't: a node
that takes one item of each is refused with `series_not_nested`, as it is for two series drawn in one graph that are not one inside the other. Each
output's description (`graph.describe()`) lists its `loops`, so an editor can tell which outputs line up.

### An output returned inside two loops

`lines` below is called once per document, and `upper` once per line. As the subgraph's output `result`, what `upper` returns is a list
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

graph = Graph.compile(make_graph(per_line, join, edges=[("per-line.result", "join.texts")]), registry)
run = graph.run(text="a,b;c")

print(run.results["per-line"]["result"])  # [['A', 'B'], ['C']]
print(run.results["join"]["result"])            # ['A B', 'C']
print(graph.runs_on["join"].msg)                # per item of per-line/documents.result
print(graph.describe().nodes["per-line"].outputs[0].type)  # Series[str]
```

`join` takes a `Series[str]`, so it collects one loop, the inner one: it is called once per document, with that document's lines.
A node that takes one `str` would be called once per line. Both are what the same four nodes do drawn in one
graph. In the description the output's `type` is `Series[str]` and `is_series` is true: together they say `Series[Series[str]]`.
Its `loops` are `["documents.result", "lines.result"]`, outermost first.

## A `None` or a skip coming in

Nothing is special about a `None` that arrives at a subgraph. It reaches the inner nodes wired to that input, and they
behave as they would if they were drawn in the outer graph:

```python
@registry.node("maybe", title="Maybe", description="Returns a word, or None for the word 'none'.")
def maybe(text: str) -> str | None:
    return None if text == "none" else text


graph = Graph.compile(make_graph(split, maybe, shout, edges=[("split.result", "maybe.text"), ("maybe.result", "shout.text")]), registry)
run = graph.run(text="hi none")

print(run.results["shout"])  # {'result': ['HI!', None]}
for not_called in run.not_called:
    print(not_called.loc, not_called.input)
# ('shout', 1, 'upper') text
# ('shout', 1, 'exclaim') text
```

For the second word `maybe` returns `None`. `upper` inside `shout` takes a `str`, not `str | None`, so it isn't called for that word, and
neither is `exclaim`, which is wired to `upper`. `run.not_called` lists those two calls, each with its `loc`: the subgraph node, the item index, the inner node. It never lists `shout`: a subgraph
node is never called, so it is never skipped or not called itself. A branch not chosen works the same way and skips the inner nodes wired to it.

An inner node that isn't wired to the `None` runs. `both`, from above, has two inputs. Here only `upper` inside it
gets the `None`:

```python
graph = Graph.compile(make_graph(split, maybe, side_by_side, edges=[
    ("split.result", "maybe.text"),
    ("maybe.result", "both.loud"),
    ("split.result", "both.keen"),
]), registry)
print(graph.run(text="hi none").results["both"])  # {'keen': ['hi!', 'none!'], 'loud': ['HI', None]}
```

## A subgraph is compiled alone

A subgraph is compiled once, on its own, when a registry describes it or a graph that uses it is compiled. So it doesn't know what will be wired to it. That matters
when its first node takes any type. `each` returns the records of whatever table arrives. Drawn in a graph, after a
node that returns a table of contacts, it returns contacts. As the first node of a subgraph it has nothing wired to
it, so it returns records of unknown fields:

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

`shout-names` takes any table, so nothing checks `each.result.name` until a run: a misspelled field would compile.
`shout-contacts` starts with a node whose input says what it takes. Its edges are checked when it is compiled, and a
graph that wires the wrong table to it gets a `type_mismatch`. So give a subgraph a first node whose input has a precise type.

## Where something is inside

`run.results`, `graph.runs_on` and `graph.describe()` show the subgraph as one node. Its outputs in `run.results` are its graph's outputs, by
name, each what the inner node behind it returned; the subgraph node is never called itself. Everything that happens inside it is reported with a `loc` that starts with
the subgraph node's id and, inside a loop, its item index, outermost first.

- A problem of the subgraph's own graph record is that problem with the subgraph node's id in front,
  `("shout", "upper", "text")`. A failure of an inner node for the second word is at `("shout", 1, "upper")`.
- An entry of `run.waiting` (a call that asked a person) and an entry of `run.not_called` have the same kind of `loc`:
  `("shout", 1, "upper")` is `upper` inside `shout`, for the second word.
- The `id` of a waiting call says the same as text: if `shout` had an inner node `ask` that asked a person, `shout@1/ask~3fa2c0d1` would be its call for the second word.
  The part after `~` is a fingerprint of the node and the arguments the call got, so an answer meant for a call whose arguments have since changed is refused.
- The events `run.stream()` yields for an inner node (`node_start`, `node_complete` and the rest) come once for every word together, as for any node
  that runs per item. Their `loc` is the node's id behind the subgraph node's id, `("shout", "upper")`. The subgraph node has events too, `loc` `("shout",)`: `node_start` when its first inner node starts, `node_complete` when its last one is complete.

A subgraph's entry in `registry.describe()` (the list of nodes an editor offers) lists its inputs and outputs, which that registry
works out from the nodes the subgraph uses. A subgraph's definition has no `describe()` of its own, because its inputs and outputs depend on the registry.

## Graphs saved per request

A host that keeps graph records in its database, for its users to use as subgraphs, makes their node definitions when a request needs them. They go in a registry of
the request's own, so the shared one stays as it was:

```python
# your table of graph records: each with the title and description a person reads for it
saved_graphs = {"shout": ("Shout", "Writes the text in capitals and adds an exclamation mark.", make_graph(upper, exclaim))}

request_registry = NodeRegistry()
request_registry.add_subregistry(registry)
request_registry.add(*(
    subgraph(name, title=title, description=description)(record)
    for name, (title, description, record) in saved_graphs.items()
    if name not in request_registry
))
```

`if name not in request_registry` skips a saved graph whose id is already a node of the shared registry, since `add` raises on a
repeated id. A graph that uses one of them as a subgraph is then compiled with `request_registry`. The compiled `Graph` keeps its own copy of
every node definition it uses, so the request's registry can go when the request does.
