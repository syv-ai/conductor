# A graph inside a graph

Say you've built a small graph that does one useful thing, like "clean up a contact". You'd like to use it in other
graphs too, without drawing its nodes again each time. You can: a graph record can be used as one node of another
graph. That node is called a subgraph.

We'll call the graph that uses it the outer graph. The nodes inside the subgraph are its inner nodes.

## Make one

You make a node from a function with `node(...)`. You make a subgraph from a graph record with `subgraph(...)`, which
takes the same arguments. Here are a few small nodes, and a subgraph `shout` made from two of them:

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


shout = subgraph(
    "shout", title="Shout", description="Turns a text into capital letters and adds an exclamation mark."
)(make_graph(upper, exclaim))
registry.add(shout)
```

`shout` is a `SubgraphDefinition`. It's a `NodeDefinition` like any other, except it has no function to call.

A subgraph has versions, just as a function node does. You pass `version=` and `replaces=` as you would to `@node`,
and each version is its own graph record. A graph that uses version 1 keeps using it until
[`registry.upgrade`](writing-nodes.md#a-second-version) moves it to version 2.

## Use one

You wire a subgraph like any other node. Its inputs and outputs are the ones its graph declares, with the same names.
`make_graph(upper, exclaim)` declared the input `text` (from `upper`) and the output `result` (from `exclaim`). So
the outer graph wires `shout.text` and `shout.result`.

`make_graph` chains a subgraph as it chains any node, by those names: `split.result` into `shout.text`, and
`shout.result` into `join.texts`.

```python
graph = Graph.compile(make_graph(split, shout, join), registry)

print(graph.run({"text": "hi there"}).outputs)  # {'result': 'HI! THERE!'}
print(graph.describe().nodes["shout"].per_item_of)  # split.result
```

`split` returns a list of words, and `shout` takes one text. So `shout` runs once per word, and `per_item_of` says
which list it loops over.

The inner nodes behave exactly as they would if you had drawn them straight into the outer graph. Each one keeps its
own `Policy`: its retries, its timeouts and its `concurrency`. A node with `concurrency=2` never has more than two
calls going, however many words there are and however many runs are going on. Cancelling a run stops inner nodes as
it stops any node, and so does the `timeout` of `resume()`. `run.again(...)` reuses their earlier outputs too.

A subgraph takes no `Policy` of its own. It's never called itself. Only its inner nodes are.

## Its inputs and outputs

From outside, a subgraph looks like one node:

- Its inputs are the graph inputs its record declares. You can give each one a constant, a graph input or an edge,
  as on any node.
- Its outputs are the outputs its record's `outputs` names.
- An inner input that takes only a constant (`constant_only` in its description) is set inside the subgraph. It
  isn't offered outside.

Here is a subgraph with two inputs and two outputs:

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
print(both.run({"loud": "a", "keen": "b"}).outputs)  # {'keen': 'b!', 'loud': 'A'}
```

`upper` and `exclaim` both take `text` and both return `result`. Two graph inputs can't share a name, so `inputs=`
and `outputs=` give them names of their own. Without them, `make_graph` raises `graph_inputs_share_a_name` and
`graph_outputs_share_a_name`.

## Outputs that come from inside a loop

A subgraph can loop inside. In `per-word` below, `split` makes a list of words, and `upper` and `exclaim` each run
once per word.

What does a node outside get from that? The same as if those nodes were drawn in the outer graph. `beside` is wired
to both outputs, so it's called once per word, with that word's two values:

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


graph = Graph.compile(
    make_graph(both_per_word, beside, edges=[("per-word.loud", "beside.loud"), ("per-word.keen", "beside.keen")]),
    registry,
)
print(graph.run({"text": "hi there"}).results["beside"]["result"])  # ['HI/hi!', 'THERE/there!']
print(graph.describe().nodes["beside"].per_item_of)  # per-word/split.result
```

The loop is named after the list inside the subgraph: `per-word/split.result` is the output `result` of the node
`split`, inside `per-word`.

The two outputs line up item by item because they come from the same loop. Two outputs from different loops inside
don't line up. A node that takes one item of each is refused with `series_not_nested`, just as it would be in one
graph. To see which outputs share a loop, look at each output's `loops` in `describe()`.

### An output returned inside two loops

Loops can sit inside loops. Below, `documents` splits a text into documents, `lines` splits each document into
lines, and `upper` runs once per line. So the subgraph's output is a list of lines for each document,
`Series[Series[str]]`:

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
run = graph.run({"text": "a,b;c"})

print(run.results["per-line"]["result"])  # [['A', 'B'], ['C']]
print(run.results["join"]["result"])  # ['A B', 'C']
print(graph.describe().nodes["join"].per_item_of)  # per-line/documents.result
print(graph.describe().nodes["per-line"].outputs[0].type)  # str
print(graph.describe().nodes["per-line"].outputs[0].loops)  # ['documents.result', 'lines.result']
```

`join` takes a `Series[str]`, so it collects the inner loop. It's called once per document, with that document's
lines. A node that takes one `str` would be called once per line instead. Again, that's what the same nodes do when
drawn in one graph.

In the description, the output's `type` is `str`: what `upper` returns for one line. `loops` lists the lists it comes
from, outermost first, and each entry is one level of list. A `Series` that the subgraph returns outside any loop has
empty `loops`, and it's just one list.

### Branches

A branch is a function node's output. A subgraph node's outputs are never branches, even one the subgraph reads
from a branch inside, so no two of them may be wired into one input outside: that is `several_edges_into_one_input`.
To join two branches, wire them into one input inside the subgraph, and return what that node gives.

Such an output may still be skipped, so `describe()` shows it as one that may be skipped: `"required": false`, with no
`default`. A skipped branch inside skips the nodes after the subgraph node that read it, as it would drawn flat, and
two lists collected on different branches of one node inside are refused with `series_from_different_branches`, as they
are drawn flat.

## A `None` or a skip coming in

A `None` that reaches a subgraph isn't special. It goes to the inner nodes wired to that input, and they act as they
would in the outer graph:

```python
@registry.node("maybe", title="Maybe", description="Returns a word, or None for the word 'none'.")
def maybe(text: str) -> str | None:
    return None if text == "none" else text


graph = Graph.compile(
    make_graph(split, maybe, shout, edges=[("split.result", "maybe.text"), ("maybe.result", "shout.text")]),
    registry,
)
run = graph.run({"text": "hi none"})

print(run.results["shout"])  # {'result': ['HI!', None]}
for not_called in run.not_called:
    print(not_called.loc, not_called.input)
# ('shout', 1, 'upper') text
# ('shout', 1, 'exclaim') text
```

For the second word, `maybe` returns `None`. `upper` takes a `str`, not `str | None`, so it isn't called for that
word. Neither is `exclaim`, which is wired to `upper`. `run.not_called` lists both calls. Each `loc` says where the
call is: the subgraph node, the item index, then the inner node.

`shout` itself is never listed. A subgraph node is never called, so it's never "not called" or skipped either.

A skipped branch works the same way: it skips the inner nodes wired to it.

Inner nodes that don't get the `None` still run. Here only `upper` inside `both` gets it:

```python
graph = Graph.compile(make_graph(split, maybe, side_by_side, edges=[
    ("split.result", "maybe.text"),
    ("maybe.result", "both.loud"),
    ("split.result", "both.keen"),
]), registry)
print(graph.run({"text": "hi none"}).results["both"])  # {'keen': ['hi!', 'none!'], 'loud': ['HI', None]}
```

## A subgraph is compiled alone

A subgraph is compiled on its own. That happens when a registry describes it, and each time a graph that uses it is
compiled. So it can't know what will be wired into it later.

That matters when its first node takes any type. Take `each` below, which gives the records of whatever table it
gets. In a graph, after a node that returns contacts, it returns contacts. As the first node of a subgraph, nothing
is wired into it yet, so it returns records whose fields aren't known:

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
print(palette["shout-names"].inputs[0].type)  # Table | list
print(palette["shout-contacts"].inputs[0].type)  # Table[name: str]
```

`shout-names` takes any table. So nothing checks the edge `each.result.name` until a run, and a misspelled field
would still compile.

`shout-contacts` starts with `known`, whose input says exactly what it takes. Its edges are checked when it's
compiled. A graph that wires the wrong kind of table into it gets a `type_mismatch`, or `type_mismatch_missing_column` for a table that lacks a column.

So give a subgraph a first node with a precise input type.

## Finding what happened inside

`run.results` and `graph.describe()` show a subgraph as one node. Its outputs in `run.results` are its graph's
outputs, each what the inner node behind it returned.

Everything that happens inside is reported with a `loc`. The `loc` starts with the subgraph node's id. Inside a loop,
the item index comes next. Then comes the inner node.

- A problem in the subgraph's own record gets the subgraph node's id in front: `("shout", "upper", "text")`.
- A failure of `upper` for the second word is at `("shout", 1, "upper")`.
- `run.waiting` (calls waiting for a person) and `run.not_called` use the same kind of `loc`.
- A waiting call's `id` says the same in text. If `shout` had an inner node `ask` that asked a person, its call for
  the second word would be `shout@1/ask~3fa2c0d1`. The part after `~` is a fingerprint of the node, the call's inputs
  and its items, as `python -m conductor.about asking-a-person` says. If they change, an answer meant for the
  old call is refused.
- The events `run.stream()` yields for an inner node come once for all the words together, as for any node that
  runs per item. Their `loc` is `("shout", "upper")`.
- The subgraph node has events too, with the `loc` `("shout",)`. Its `node_start` comes when its first inner node
  starts, and its `node_complete` when the last one is done. Its count, there and in `run.progress`, adds up the
  calls of its inner nodes: `upper` and `exclaim` for two words make four.

In `registry.describe()`, the list of nodes an editor offers, a subgraph's entry shows its inputs and outputs. The
registry works them out from the nodes the subgraph uses. That's also why a `SubgraphDefinition` has no `describe()`
of its own: its inputs and outputs depend on the registry.

All the rules for subgraphs are in the reference: `python -m conductor.about the-graph-record`.

## Graphs saved per request

A host may keep graph records in its database, so its users can use them as subgraphs. The host then makes their
node definitions when a request needs them. They go in a registry made for that request, so the shared registry
stays as it was:

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

`add` raises on an id that's already there. The `if name not in request_registry` skips a saved graph whose id is
already a node of the shared registry.

Compile the graphs of that request with `request_registry`. A compiled `Graph` keeps its own copy of every node
definition it uses, so the request's registry can go away when the request ends.
