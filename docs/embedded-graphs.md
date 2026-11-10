# A graph inside a graph

A stored graph can be a node of another graph. An author builds "enrich a contact" once, saves it, and uses it in
every graph that needs it, as one node.

## Make one

`node("id")` wraps a function, and it wraps a `GraphRecord` the same way:

```python
from conductor import NodeRegistry, Series, make_graph, node

registry = NodeRegistry()


@registry.node("upper")
def upper(text: str) -> str:
    return text.upper()


@registry.node("exclaim")
def exclaim(text: str) -> str:
    return text + "!"


@registry.node("split")
def split(text: str) -> Series[str]:
    return text.split()


@registry.node("join")
def join(texts: Series[str]) -> str:
    return " ".join(texts)


shout = node("shout", title="Shout")(make_graph(upper, exclaim))
registry.add(shout)
```

`shout` is a `NodeDefinition` like any other. It takes `replaces=`, `version=`, `title=` and the rest, and a newer
version of the stored graph is a newer version of the node. Each version is its own record, so a graph saved against
version 1 keeps running version 1 until [`registry.upgrade`](writing-nodes.md#a-second-version) moves it on.

## Use one

Wire it like any node:

```python
graph = registry.load(make_graph(split, shout, join, edges=[("split.result", "shout.text"), ("shout.result", "join.texts")]))

print(graph.run(text="hi there").outputs)  # {'join.result': 'HI! THERE!'}
print(graph.plan["shout"].msg)              # per row of split.result
```

A stored graph runs on each row its node runs on, here once per word. Its own `Series` inputs gather the rows under
that one row.

## What it offers

From outside, a stored graph is one node:

- **Its inputs** are the open inputs of its first nodes.
- **Its outputs** are the outputs of the nodes nothing inside reads.
- Each field goes by its inner name where that name is unique, and by `inner_node/field` where two inner nodes share
  one. With `upper` and `exclaim` side by side, the inputs are `upper/text` and `exclaim/text`.

```python
side_by_side = node("both")(make_graph(upper, exclaim, edges=[]))
registry.add(side_by_side)
both = registry.load(make_graph(side_by_side))

print(list(both.inputs))  # ['upper/text', 'exclaim/text']
print(both.run({"both.upper/text": "a", "both.exclaim/text": "b"}).outputs)
# {'both.upper/result': 'A', 'both.exclaim/result': 'b!'}
```

Everything else names the embedding, never its inner nodes:

- problems
- results
- `graph.plan` and `graph.describe()`
- events: it starts once and completes once
- waiting rows and their questions
- a run's `cause`

A stored graph with problems of its own is one problem, `embedded_graph_broken`, and its `ctx.problems` lists them
(see [problem codes](problems.md#embedded_graph_broken)).

`node.describe()` of a stored graph lists no fields. Its fields depend on the registry its inner nodes come from, so
they show in `graph.describe()` once a graph that uses it loads.

## Graphs saved per request

A host that keeps saved graphs in its database makes their nodes when a request needs them. They go in a registry of
the request's own, so the shared one stays as it was:

```python
saved_records = {"shout": make_graph(upper, exclaim)}  # your table of saved graphs

request_registry = NodeRegistry()
request_registry.add_subregistry(registry)
request_registry.add(*(node(name)(record) for name, record in saved_records.items() if name not in request_registry))
```

A graph that embeds one of them then loads through `request_registry`. The loaded `Graph` keeps its own copy of what
it resolved, so the request's registry can go when the request does.
