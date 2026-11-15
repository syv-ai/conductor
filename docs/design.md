# Conductor 2.0's design: a reviewer's guide

This page is for someone reviewing how conductor is built. It goes topic by topic. For each design choice it says
what the choice is, why it was made, and what it costs. Where a reviewer could fairly push back, it says so under
**Worth questioning**, along with the reason the choice stands.

It assumes you've read the [first page](index.md) and [writing nodes](writing-nodes.md). The exact rules are in the
reference, `python -m conductor.about`, and this page points to its sections rather than repeating them.

Three words come up a lot:

- A **hook** is a plain function you pass to `@node` (`compute_outputs=` or `compute_added_inputs=`). It works out a
  node's inputs or outputs from the graph's constants when the graph is compiled.
- **Branches** are outputs of one node that share an `exclusive_group`. Each call returns one of them, and the others
  are `SKIPPED`.
- An **outcome** is what a run keeps of one call of a node.

The code blocks run in order, as one module. The comment after each `print` is what it prints.

## What conductor is

### A general library, fitted to no host

Conductor runs graphs of typed Python functions. One person writes the functions. Someone else draws the graph in an
editor, and the graph reaches the host as JSON. The core has no web framework, no database and no login.

**Why.** Fitting it to one platform would make the library narrower and the code no better. The source uses none of
a host's words, and `tests/test_core/test_surface.py` checks that.

### Two audiences, two bars

The node author's surface is `@node`, `Param`, `Series`, `Table`, `SKIPPED`, `Ask` and the errors. Other packages
are written against it, so it's kept general and stable. The host's surface is compiling, running, answering, saving
and events. It has only what a real host calls.

Five modules are public: `conductor`, `conductor.widgets`, `conductor.events`, `conductor.testing` and
`conductor.about` (the reference). Anything in `conductor._internal` may move.

## The front door and the registry

### A graph is data

There's no `@graph` decorator, and you don't place nodes in Python. A graph you build in code is the same record an
editor saves.

**Why.** Most graphs are drawn. Two ways to define a graph would mean two sets of rules.

### `@node` wraps a function and registers nothing

`@node` returns a frozen `FunctionDefinition`. You can still call it like the function, so calling it is its unit
test. `subgraph(...)(record)` returns a `SubgraphDefinition`, which has no function to call. A `NodeDefinition` is
either one. `@registry.node(...)` and `registry.subgraph(...)` take the same arguments, and also add the node.

**Why.** One definition can go into any registry. What a host offers is then written in one place: where it builds
its registry.

**Worth questioning:** a node that registers nothing can't be found until someone adds it to a registry. That's the
price of being able to put it in any registry.

### A registry works like a FastAPI router

`add_subregistry` copies another registry's nodes in, so a package can keep one registry per module and join them.

Whenever a node enters a registry, the same merge rule applies. The registry's tags come first, then the node's own.
The node's own `category` wins. `extra` is merged key by key.

**Why.** Replacing `extra` whole would drop a registry's filter key from any node that has an `extra` of its own.

**What it costs.** The registries don't stay linked. A node added to one later doesn't show up in the other.
Nothing has needed that, so it wasn't built.

### One way to do each thing

The decorator is for your own nodes, and `add(*definitions)` is for other people's. The constructor takes no list of
nodes. `add` raises on an id that's already there. To swap or drop a node, you call `replace` or `remove`.

### `filter` gives you a new registry

`registry.filter(predicate)` hands the predicate each whole definition and returns a registry of its own. `extra` is
the host's JSON. `describe()` shows it, and conductor never reads it.

```python
from conductor import Graph, InvalidCallError, NodeRegistry, Series, make_graph

text = NodeRegistry(category="Text", extra={"tier": "free"})


@text.node("words", title="Words", description="The words of a text.")
def words(text: str) -> Series[str]:
    return text.split()


@text.node("upper", title="Uppercase", description="Turns a text into capital letters.", tags=("case",))
def upper(text: str) -> str:
    return text.upper()


@text.node("join", title="Join", description="Joins the pieces into one text.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


registry = NodeRegistry(tags=("core",))
registry.add_subregistry(text)
print(registry["upper"].tags)  # ('core', 'case')
free = registry.filter(lambda d: d.extra.get("tier") == "free")
print(sorted(d.id for d in free))  # ['join', 'upper', 'words']
```

## The graph record and `make_graph`

### A record is its nodes and its outputs

A record is `{"nodes": {id: node, ...}, "outputs": {name: output, ...}}`, and nothing more. A node is
`{"node", "version", "inputs", "display"}`, stored under its id in the graph. Empty fields are left out.

In conductor's JSON, `node` always means the id of a node definition, and `node_id` means a node's id in a graph.

Anything a host keeps about the whole graph, like the canvas viewport, it stores beside the record.

**Why.** Separate fields are simpler than one combined string, which would need parsing everywhere.

The record is checked when it comes in. A key written twice in one JSON object (so two nodes with one id) is refused
as `invalid_record`. So is a node id, input or output that isn't a name: empty, or with `.`, `/`, `@` or `~` in it.

### Compiling never upgrades

`version` defaults to 1. Compiling uses the version the record names. Only `registry.upgrade` moves a graph to a
newer version, and only when asked.

### Each input gets exactly one thing

An input in the record gets one of three things:

- a `constant`, the same on every call of every run;
- a `graph_input`, the caller's value under that name, with a `default` for when the caller gives none;
- or `sources`, the edges that lead into it.

An entry with two kinds is refused as `invalid_record`. An input the record leaves out gets the node's own default.

**Why.** It was chosen over a map per kind of value, and over a `locked` flag beside one map. With one entry per
input, an input can't be given two things, so no rule has to say which one wins.

### Values in a record are JSON

Constants, defaults and a node's `display` are stored as JSON. A date is kept as its ISO text, and a float that isn't
finite as `"NaN"` or `"Infinity"`.

**Why.** A record then always equals its own JSON, so its fingerprint survives being stored and read back.

**What it costs.** A value with no JSON form, like a polars frame, can't be a constant. `make_graph` refuses it as
`invalid_value`. A record built by hand with a float that isn't finite is refused too.

### `display` belongs to the host

Conductor stores a node's `display`, never reads it, and leaves it out of the graph's fingerprint. It sits on the
node so it follows the node through an upgrade.

### The graph decides what's a constant

Whether an input is a constant is decided per graph. A constant in the record is the graph author's value. A graph
input is the caller's.

A node whose input must be known when the graph is drawn says so with `Param(constant_only=True)`. Compiling holds
every graph to it (`input_takes_only_a_constant`), the field's description shows it, and `make_graph` follows it.

**Why.** A node can't know every graph it will be used in.

### An edge lives on the input it leads into

An edge is stored on the input as a source. A source is an output, `{"node": "read", "output": "result"}`, or one
field of the record that output returns, with `"field": "email"` added.

Several sources on one input are alternatives: opposite branches of one decision, never a list to join. So nothing a
run does depends on their order. A source listed twice is refused. An edge from a node the graph doesn't have is the
problem `edge_names_missing_node`.

### A graph declares its inputs and outputs

A graph declares what it takes and what it returns, just as a function declares its parameters and its return value.

A graph input exists only where an input's entry names it. Several inputs may name the same graph input, and then
the caller's one value feeds them all. A graph output exists only in `outputs`, by name. When the graph is used as a
subgraph, these are the node's inputs and outputs.

A graph output that names a node the record doesn't have, or an output that node doesn't have, is the problem
`graph_output_not_found`.

**Why.** Working them out from the first and last nodes was rejected. Adding a node at the end would change what the
graph returns, and a subgraph's inputs and outputs would be places inside it.

### `make_graph` writes a record, like scikit-learn's `make_pipeline`

`make_graph(*nodes, edges=, constants=, defaults=, inputs=, outputs=)` follows `make_pipeline`. Like `Pipeline`, it
also takes a node as `(id, definition)` when you want to choose the id.

With no `edges`, the nodes form a chain: each node's one open required input is wired to the one output of the node
before it. If that doesn't work, it raises `not_a_chain`. With `edges`, those are all the edges.

It declares the graph's inputs and outputs for you. Every required input that gets nothing becomes a graph input, and
every output no edge leaves becomes a graph output. `inputs=` and `outputs=` name some of them differently.

It writes a record and compiles nothing. It raises one `InvalidCallError` for whatever it can't write. Everything
else, like a mistyped input or an edge between types that don't fit, shows up in `Graph.compile(...).problems`.

**Why.** Compiling stays the one place where a graph's faults are found.

**What it costs.** A script that only saves the record hears about those faults only once the record is compiled.

The full rules are in `python -m conductor.about quick-start`.

```python
record = make_graph(words, upper, join)
print(record.model_dump_json())  # {"nodes":{"words":{"node":"words","inputs":{"text":{"graph_input":"text"}}},"upper":{"node":"upper","inputs":{"text":{"sources":[{"node":"words","output":"result"}]}}},"join":{"node":"join","inputs":{"texts":{"sources":[{"node":"upper","output":"result"}]}}}},"outputs":{"result":{"node":"join","output":"result"}}}


@registry.node("half", title="Half", description="Halves a number.")
def half(n: float) -> float:
    return n / 2


typed = make_graph(half, constants={"half.n": float("nan")})
print(typed.nodes["half"].inputs["n"])  # constant='NaN'
```

**Worth questioning:** the chain never matches inputs to outputs by name. Say an output and an input happen to share
a name. A chain that matched by name would wire them without telling you, and leave the input you meant empty. A
chain that refuses to guess can't do that.

**Worth questioning:** a node used twice gets the ids `upper-1` and `upper-2`, not `upper` and `upper-2`. With the
second form, inserting another `upper` earlier would quietly move the id `upper` to a different node.

## `Graph` and `Run`

### Compiling reports faults instead of raising

`Graph.compile(record, registry)` never raises for a fault in the graph. Each fault is a `Problem` (`type`, `loc`,
`msg`, `ctx`) that an editor can mark on the node. It raises only when what it's given isn't a record at all
(`invalid_record`).

**Why.** Getting the shape right is the caller's job. What the record says is the graph's business.

### A graph input with no value yet is fine

A graph input is filled in by the caller when a run starts, so it isn't a problem when compiling. It's like an
unfitted scikit-learn pipeline: not broken, just not run yet. A required input that gets nothing at all (no constant,
no graph input, no edge) is a problem: `input_has_no_value`.

### A compiled graph keeps its own copy of its definitions

**Why.** A later change to the registry can't break a graph that's already compiled.

### One way in

A host never builds a `Graph` or a `Run` itself. `Graph.compile` makes a graph, and a graph makes its runs.

The verbs mirror each other. `graph.with_inputs(...)` checks the inputs and gives you a run that hasn't started.
`graph.run(...)` is the same plus `resume()`. On a run, `with_inputs` and `again` do the same with some inputs
changed, reusing what they can. `run.answer(...)` returns another run and runs nothing.

### Running again reuses by content

A call reuses an earlier output when it gets exactly the arguments a call of the same node got in the previous run,
with the same version, code, inputs and outputs. Identical items share one earlier output.

A node that may ask a person (`asks=True`) is matched another way. Its earlier outcomes are taken in item order,
first to first and second to second. So if an item is inserted above, everyone keeps their own answer. This covers a
call that returned without asking too, since an answer is kept as the call's outputs, just like a return. An
identical item beyond the earlier outcomes calls the function again.

Two more tools build on this. `run.forget(*ids)` calls chosen nodes again, for a host that must check access again.
`edited.with_inputs(..., previous=old)` reuses an old run across an edit of the graph, for the nodes the edit left
alone. The rules are in `python -m conductor.about running-again`.

**Worth questioning:** the code part of the reuse key is the node function's own bytecode and constants. A change to
a helper it calls isn't noticed. A node that depends on more than its inputs sets
`Policy(reuse_when_inputs_match=False)`. Hashing all the code a node can reach would be slow, and it would still
miss the outside world.

**Worth questioning:** earlier outputs a node didn't reuse are dropped once the node is complete. So if you undo an
edit two runs later, the node is called again for that item. Without the drop, a saved run would grow with every
edit. A node with a question still waiting keeps them, so an answer survives an edit and its undo.

### Errors are data

There's one `Problem` record everywhere. Its codes are stable, and `PROBLEMS.codes` lists them so a host can
translate them. A message says how to fix the fault and never quotes a value. A node's own exception text goes to
the log, not to the host.

```python
graph = Graph.compile(record, registry)
print(graph.run(text="a b").outputs)  # {'result': 'A+B'}
```

## One or many

### The types decide how often a node runs

A `Series[T]` output wired to an input that takes one `T` starts a loop. That node runs once per item, and so does
every node downstream of it. A `Series` input collects the loop again: its node is called once, with every item. A
`Table` input collects records the same way. A value from outside the loop is the same on every call. The node
itself never knows any of this.

**Why.** The types already say it. `contract: File` runs per item, and `contracts: Series[File]` runs once. A "loop
over this" flag on an input, or a switch per node, would say it a second time.

### The type decides, never the value

A list given to an input that takes one value is refused. A node that returns many values says so in its return
type, `Series[...]`. So one file and ten files take the same path through the graph.

**Why.** If a caller's list could start a loop, how often a node runs would depend on what each run was given. The
graph would have to be compiled again for every run, and an editor couldn't show it in advance.

### A table is one value

A table never starts a loop by itself, and an edge can't name one of its columns. (An edge can name a field of each
record in a `Series`, though.) To run a node per record, you draw a Split node where you want it.

**Why.** If tables started loops by themselves, how often a node ran would follow from which columns happened to be
wired to it. An author can't see that in the drawing.

**Worth questioning:** you need a Split before a node can run per record. That's on purpose: the Split is where an
author sees the run fan out, and a table that's only passed along costs nothing.

### Two lists never pair up by themselves

Say a graph has ten documents and three questions. A node that takes one of each is refused as `series_not_nested`.
To fix it, make one list come from inside the other's loop (the node that returns the questions takes a document).
Or give one list whole to a `Series` input.

**Why.** Crossing the two lists and pairing them by position were both rejected. Crossing quietly multiplies paid
calls. And which item goes with which is the author's call, so the author draws it.

### Collecting ends a loop

A `Series` input gets the items that have a value. It leaves out `None` and skipped items. It waits for every item.
When no item could have a value, its node is skipped or not called, as any node would be. No items at all is an
empty list, and the node is still called. "0 approved" is an answer.

The full rules are in `python -m conductor.about one-or-many`.

### An item limit before money is spent

`max_items=` caps the length of any one list a loop runs over. It's checked when the list is returned, before any
call for its items. Over the limit, the run is `stopped`. `run.with_max_items(100).resume()` goes on under a higher
limit, without calling anything again.

```python
print(graph.describe().nodes["upper"].per_item_of)  # words.result


@registry.node("greet", title="Greet", description="Greets a person by name.")
def greet(name: str, greeting: str) -> str:
    return f"{greeting} {name}"


names_and_greetings = make_graph(
    words, words, greet,
    edges=[("words-1.result", "greet.name"), ("words-2.result", "greet.greeting")],
    inputs={"names": "words-1.text", "greetings": "words-2.text"},
)
print([problem.type for problem in Graph.compile(names_and_greetings, registry).problems])  # ['series_not_nested']

try:
    graph.run(text=["a", "b"])
except InvalidCallError as e:
    print([p.type for p in e.problems])  # ['invalid_input']
```

**Worth questioning:** a list sent to a `str` input is refused, where conductor could have run the node once per
item. It stands because a graph then runs the same way whatever it's given, and an editor can show how often each
node runs before anyone runs it.

## `None` and values from outside a loop

### A `None` skips that one call

Say an argument is `None` and the input's type doesn't include `| None`. Then the node isn't called for that item.
Its outputs are `None`, and none of its branches is chosen. `run.not_called` records the node, the item and the
input.

**Why.** Pydantic would refuse `None` for a `str`. Failing the whole run, after other items have already had side
effects, would be worse.

**Worth questioning:** the run below ends `complete`, with one message not sent. It stands because the run records
which item and which input. A host can say "Bo was not sent: no middle name" instead of failing everyone.

### Each spelling says one thing

- `Series[X]` leaves `None` items out.
- `Series[X | None]` keeps them.
- `Series[X] | None` is a list that may itself be `None`.

`Table[R] | None` reads the same way: a table that was never returned reaches the input as `None`.

### `SKIPPED` is a branch not chosen, not a `None`

Inside a loop, `SKIPPED` skips the nodes downstream for that item. Outside a loop, it skips them altogether.

A skip wins over everything else. If one input is skipped, the node is skipped, even when another input is `None` or
has a fault. Without a skip, a fault fails the call. Without a fault, a `None` the input doesn't accept makes the
node not called. The order is in `python -m conductor.about one-or-many`.

### A value from outside the loop is the same on every call

Nothing is copied, and it starts no loop of its own. Wiring in a document's name doesn't change how often a node
runs.

```python
middle = {"Ida": "Marie"}


@registry.node("middle-name", title="Middle name", description="Looks up a person's middle name, if there is one.")
def middle_name(name: str) -> str | None:
    return middle.get(name)


@registry.node("send", title="Send", description="Sends a message to a contact.")
def send(middle: str) -> str:
    return f"sent {middle}"


sent = Graph.compile(make_graph(words, middle_name, send), registry).run(text="Ida Bo")
print(sent.status, sent.results["send"]["result"])  # complete ['sent Marie', None]
print([(skipped.loc, skipped.input) for skipped in sent.not_called])  # [(('send', 1), 'middle')]
```

## Asking a person

### `raise Ask(...)` pauses one call

`raise Ask(prompt, choices=, default=)` pauses one call while it waits for a person. That call waits, and so does
whatever reads its outputs. The rest of the run goes on.

The return type says what the answer is and how to ask for it. `bool` is a yes or no, a `Literal` is one of its
values, and a frozen dataclass is a small form.

**Why.** `AskBool`, `AskMulti` or a `type=` argument would declare the answer's type a second time.

**Worth questioning:** `Ask` derives from `BaseException`, not `Exception`. Otherwise a node's `except Exception`
would swallow the pause and carry on with a wrong value.

### A node that asks says so on its `@node`

A node that may ask has `asks=True` on its `@node`. Raising `Ask` without it fails as `ask_not_declared`. In
`registry.describe()`, each node says whether it `asks`. For a subgraph, that's whether any node inside it asks. So
a host knows which nodes can pause without guessing from the code.

**Why on the decorator.** Asking is something the function does, like raising. It isn't something its value is.
Metadata on the return type would be a second word for the same thing, and it would sit oddly on `-> None` or on a
dataclass of outputs.

**Worth questioning:** the signature alone doesn't say everything the node does.

### One way to ask, and the answer is the node's outputs

A node that needs a person's answer reads it from an edge, like any other value. Take "approve, then pay".
`approval(invoice) -> bool` returns `True` for a small invoice and raises `Ask` for a large one. Its output is wired
into `pay(invoice, ok: bool)`. A host that wants the two as one entry in its palette makes them a `subgraph`.

**Why.** An answer has one shape, the call's outputs, as if the node had returned them. And it has one place where
it's kept, saved and reused: with the call that asked. `pay` is called once, with a real `bool`, and
`pay(invoice, ok=True)` is its unit test.

These were rejected:

- A parameter filled in by a function that may ask. That's a second way to ask, its answers would be kept beside the
  outcomes, and an edge couldn't reach it.
- `ok = ask(...)`, replayed. The body would run twice, and nothing would be known when the graph is compiled.
- Calling the node again with `ok=None`. A body that forgets the check never asks.
- Nodes with two phases.

**Worth questioning:** the question shows on `approval`, inside the subgraph, not on the node a person thinks of as
the step.

### `Ask` carries only what the run knows

How a question looks comes from its type's `Param`: a widget, a title, or choices known when the node is written. So
the output types a hook returns carry it too. `Ask` takes no `widget=`, because that would be a second home for it.

`-> str` with a bare `Ask(prompt)` asks for free text. Each argument of `Ask` has the shape of the answer: a value for
one output, or a mapping by output name for several, naming only some if you like. There's no plural `defaults=`.

### A question is a field

A question is the same record as an input or output: the `FieldDescription` a palette reads, with the `choices` and
`default` the `Ask` set. A host draws a question the way it draws an input.

**Why.** One JSON Schema for the whole answer was rejected. It loses a host type's id, whether a field is one value
or many, and exclusive groups. And it has no place for a widget.

### Answers are checked all at once, and strictly

You answer with `run.answer(waiting, value)` or `run.answer({id: value, ...})`. Every fault is refused at once, and
nothing is left half answered.

An answer is read strictly, from its JSON form: `true`, never `"yes"`. A declared choice is read the same way and held
to the field's limits, so an enum's value or a date's ISO text can be a choice. A choice's id is its value as the
field's type writes it in JSON. `Decimal("1.0")` has the id `"1.0"`, and a `datetime` has its ISO text,
`"2026-01-02T03:04:05"`.

### A waiting id names the call and what it was given

A call inside a loop has an id like `approve@1~3fa2c0d1`: an `@` and an index per loop. A node in no loop has one
like `tone~400b7c9a`. The part after `~` is a fingerprint of the node, its code and the call's arguments. An answer
meant for a question that has since changed is refused as `answer_stale`.

```python
from typing import Literal

from conductor import Ask


@registry.node("tone", title="Tone", description="Asks a person which tone a text should have.", asks=True)
def tone(text: str) -> Literal["formal", "casual"]:
    raise Ask(f"Which tone for {text!r}?")


asked = Graph.compile(make_graph(tone), registry).run(text="hello")
print(asked.answer(asked.waiting[0], "casual").resume().outputs)  # {'result': 'casual'}
```

**Worth questioning:** an id without its `~` part still answers the call that waits there. A host stores the full id,
and the fingerprint guards that. The short id is what a person types about the run in front of them.

## The engine

### A small core

Compiling decides once, from the declared types alone, how often each node runs. A run keeps one outcome per call,
keyed by the node and its item indexes (a tuple of numbers, one per loop). A table stays the frame it is.

**Why.** The parts pay off together. Most of the core's speed comes from calling nodes inline, and from checking
values only where a check can fail.

### Nodes run on the run's own thread by default

With `Policy()` as it comes, a node is called on the run's own thread. An author asks for threads by setting
`concurrency` above 1, or an `attempt_timeout`.

**Why.** Threads by default would surprise code that isn't thread-safe. And calling inline is most of the core's
speed.

**Worth questioning:** a host whose slow nodes forget `concurrency` runs them one at a time within a run. It stands
because the slow node knows it's slow, and saying so is one argument.

### `concurrency` counts calls across the whole process

A `concurrency` a node sets counts its calls everywhere: in every graph that uses it, and in every run. So it can
keep to an outside service's quota.

Left at `None`, the default, each run calls the node one item at a time, and runs don't wait for each other.

**Why.** A lock on every call would slow every run, and line up every server's slow nodes, for a quota no node asked
for. A run that has to wait for another run's call doesn't block: it calls its other nodes and is told when a call
ends.

### Values are checked where it matters

A node's return value is read strictly against its type (`invalid_output`). A value on an edge is checked again only
where the source's type doesn't already guarantee it fits. That's decided when compiling, so a plain chain checks
nothing at run time.

### Only some failures are retried

Only an `ExternalError`, or an exception class the node's `retry_on` names, is retried. A node's own `NodeError`, an
output its type refused and an `Ask` are never retried, whatever `retry_on` says.

**Why.** Calling again would just give the same result.

After a failure, calls already running on threads finish, and their outcomes are kept. So resuming repeats none of
them.

### A subgraph is a graph record used as a node

`subgraph("id", title=..., description=...)(record)` makes a node from a graph record. It takes no policy and no
hook, because its inner nodes have their own. From outside it's one node, whose inputs and outputs are the ones its
graph declares.

The subgraph is compiled once, on its own, and stays one node of the compiled outer graph. A run only ever calls
function nodes: the subgraph's inner nodes, each named by its path (`shout/upper` in a saved run). An inner node has
one call per item of every loop it's inside, outside the subgraph and in it. So a `None` or a skip that arrives
reaches the inner nodes wired to it, just as if they were drawn in the outer graph. The subgraph node itself is
never called, skipped or not called. [A graph inside a graph](subgraphs.md) teaches it.

**Why.** A separate run per item of the subgraph was rejected. Measured on 2,000 items, its saved run was 34 times
the size of the same nodes drawn in one graph.

```python
from conductor import subgraph

shout = subgraph("shout", title="Shout", description="Turns a text into capital letters.")(make_graph(upper))
registry.add(shout)
shouted = Graph.compile(make_graph(shout, join, edges=[("shout.result", "join.texts")]), registry)
print(list(shouted.inputs), shouted.outputs)  # ['text'] ['result']
```

## Saved runs, checkpoints and events

### A run is one JSON text

`run.to_json()` writes a run, and `Run.from_json(saved, graph)` reads it back. It checks the saved run against the
graph, and refuses a run saved from another graph by its fingerprint (`saved_run_of_another_graph`).

A few things keep it small and safe:

- It survives a store that reorders keys, such as Postgres `jsonb`.
- It has only the nodes that were called, so a long graph waiting at its start saves small.
- Each node's outcomes are one list per output, one entry per call in item order:
  `"upper": {"states": ["done", "done"], "outputs": {"result": ["A", "B"]}}`. Item indexes are never written.
  Reading works them out again from the lengths of the saved lists. So a call the run never made can't be written.
- A waiting call is saved with its question, so it can't be saved without one.

Its shape is one pydantic schema per graph, made from the schema of each input's and output's type. It's built the
first time a run of the graph is written or read. Writing is one pass through it, and so is reading. A damaged saved
run is one `saved_run_unreadable` problem. Its `ctx.where` is the place pydantic names, like
`outcomes.upper.outputs.result.0`.

### A `Series` of records keeps the table it came as

Split returns a table for a `Series[R]`, so no record has been made yet. Most nodes downstream want one field, and an
edge from `series-split.result.email` reads that column straight from the table. A node that takes the record, or the
whole `Series`, gets plain records. So an author never meets a second kind of value.

The table is checked by its columns when it's returned, and a saved run writes it once, as a table.

**Why.** Making every record up front was the whole cost of Split on a run.

### A float that isn't finite is saved as text

In a saved run, `inf` is written `"Infinity"`, and its declared type reads it back. `float` gives the float back. `Any`,
a `dict` and `float | str` give the text, just as a date under `Any` comes back as its ISO text. An author who wants
the float back declares it.

**Why.** Marking the place of each such float in the saved run was rejected. It would state a value's type a second
time, beside the type.

### Checkpoints come from the stream

`run.stream(checkpoint_every=seconds)` sends `run_checkpoint` events, each with a copy of the run. Resuming one makes
only the calls it has no outcome for. So a process that dies mid-run repeats at most the calls that were in flight.

A host that shuts down hands the run back with `run.stream(hand_back=event)`. The stream stops as it does for a
cancel. Its last event is a checkpoint of the run, still `running`, with no `run_end`. The stored run stays
`running`, and another process goes on from it.

**Why.** A callback argument was rejected as a second channel beside the stream. Cancelling a run and handing it
back are kept as two acts. One event for both would save a run nobody cancelled as `cancelled`.

### Events carry counts, not results

`node_complete` says how many calls a node had. The values stay in the run, so an event stays small however big the
table is.

The exceptions are what a host needs to act on. `node_failed` and `node_retry` carry a `Problem`. `run_end` carries
the run's `cause`, so a run stopped at its item limit says how many items there were, and a client can pick a higher
limit. `node_waiting` carries the `Waiting` call, with its question and the arguments the call got.

```python
import asyncio


async def follow() -> None:
    async for event in graph.with_inputs(text="a b c", max_items=2).stream(checkpoint_every=1.0):
        if event.type == "run_checkpoint":
            pass  # store event.run.to_json()
        if event.type == "run_end":
            print(event.status, event.cause and event.cause.type)  # stopped item_limit


asyncio.run(follow())
```

## Where conductor meets pydantic

### Every record is a frozen pydantic model

`GraphRecord`, `Problem`, the events (one union on `type`), `Waiting` and every `describe()` result are frozen
pydantic models. Each one works as a FastAPI response model as it is.

### `Param` is pydantic metadata

`Param` implements pydantic's schema hooks. So its limits and choices work inside a host's own `BaseModel`, just as
they do on a node's input. A field's limits live in its JSON schema, not in a separate list.

Choices are read by one function, `read_choices`. `Param`'s schema hook, a type's `choices` and an `Ask`'s choices
all use it.

On a node's input or output, `Param` is the only spelling. Pydantic's `Field(...)` in the hint is refused
(`pydantic_field_in_hint`), so a limit is never written two ways. A pydantic type that has its own limits, like
`PositiveInt`, is just a type, read as pydantic reads it.

### Strict where a program made the value, lax where a person wrote it

Answers, declared choices and node outputs are read strictly: `"12"` is never an `int`. Constants, defaults, graph
inputs and edges are read laxly.

Either way, a `bool` never counts as a number. It isn't taken for an `int`, nor for a union that has a number but
no `bool` (`int | float`), whether as a value or in a table's column.

### One reading of a type hint

Each type hint is read once into one object. That object names the type, gives its pydantic schema, writes a value
as JSON and says which types may be wired into it. It doesn't check values itself. Pydantic does all the validating
(see the next section).

The one question pydantic doesn't answer is whether an output of one type may feed an input of another. Conductor
answers that with its own walk over the two types.

**Why.** Answering it from pydantic's JSON schema was tried. It accepted edges whose values were then refused at run
time.

### `Series[T]` and `Table[R]` are plain generic classes

Each has its own pydantic schema. So a table can be part of a union, an item of a `Series` and a field of a host's
model. It has one JSON form wherever a host sees one: `{"columns": [{"name", "type", "values"}]}`.

A value read as `Table[R]` has exactly `R`'s columns, wherever it comes from: a caller, a person's answer, collected
records or a node's own return. A missing column is refused, and extra ones are cut. A bare `Table` keeps every
column.

To a type checker, a `Series[T]` is a `list[T]`, and a `Table[R]` is a frame.

**Worth questioning:** a type checker accepts a `Table[Sale]` where a `Table[Order]` is expected. A separate generic
class would refuse it, but then every plain `polars.DataFrame` return would be a type error.

### A node can be written for any type

`def split[T](value: Table[T] | list[T]) -> Series[T]` takes whatever type arrives on its edge. That's settled when
the graph is compiled, and from then on the node has ordinary inputs and outputs. `series-split`, `series-stack` and
`control-decide` in `syv-conductor-nodes` are written this way, with no hooks.

A hook is only for what a type can't say: inputs or outputs named by a constant. See
`python -m conductor.about a-node-for-any-type`.

### One flat record per input or output

A `FieldDescription` always has a `type` and a `schema`. `is_series` says whether it's a `Series`, and a table's
columns are in its schema.

For a subgraph's output that comes from inside loops, `type` is the value returned inside, and `loops` lists one level
of list per entry. A `Series` returned outside every loop has empty `loops`, and it's one list, as `is_series` says.

`null` in `default` means the value `None` and nothing else. "No default" and "a branch not chosen" both have no
`default` key, and `required` tells them apart. So the record needs no special marker that JSON lacks.

An output is shown as the question its type asks, so a palette and a waiting call say the same thing about it. In
`graph.describe()`, a node's input is `{field, source}`: the field, nested as it is, and where its value comes from.

### `Policy` is a strict pydantic model

`Policy(retries="2")` raises pydantic's own error, right where it's written.

### What a person sees for a type is declared

Conductor's own types get their id and title from one fixed table. A host's type that people will see is declared
with `@host_type("id", title="Title")` above the class, and it's checked as soon as the class is defined. Nothing
falls back to a class's name.

A type's id shows its generic arguments (`list[address]`), so a mismatch can name them. A class used inside a generic
declares its id and title too. A host shows its own words for type ids in its editor.

Every Python block in the docs is type-checked under mypy and pyright.

```python
from typing import Annotated

from pydantic import BaseModel

from conductor import Param


class Settings(BaseModel):
    temperature: Annotated[float, Param(ge=0, le=2)] = 1.0


print(Settings.model_json_schema()["properties"]["temperature"])  # {'default': 1.0, 'maximum': 2, 'minimum': 0, 'title': 'Temperature', 'type': 'number'}
```

**Worth questioning:** `@host_type` is a decorator, not a base class. A base class runs before the class is
finished. That breaks pydantic's `Box[int]`, `@dataclass(slots=True)` and enums, and you can't give a library's
class a new base. The decorator also gives a subclass of a builtin (`class Email(str)`) the schema it lacks. It
puts that on the class, so conductor still knows no host type.

**Worth questioning:** `@host_type(..., accepts=[...])` takes a list of classes whose values may be wired into the
type. A rule that isn't a list of classes can't be said there. But a list is data, which an editor can show before
anything runs.

## Validation: where values are checked

A value from outside is checked once, where it comes in, and never again.

Each place where a value comes in has its own **validator**. There's one for a graph record, a constant or default
in it, a default a node declares, what a hook returns, a caller's graph inputs and limits, the `context=` a host
passes, a person's answer, and what a node returns or raises. There's one for a value an edge gives an input whose
type asks for more than the edge's source promises. A saved run is checked on the way in too.

A validator is built with what it checks a value against, like a graph's inputs or a node's outputs. It's called
with the outside value. It either returns the value as the engine keeps it, or a problem with its own code
(`invalid_input`, `invalid_output`, `invalid_answer` and so on).

Past that wall nothing is checked again. A node gets its arguments as the validators read them, and the engine
trusts them.

### Each validator lives with what it checks

Each validator is in the package of what it checks, in that package's `validator.py`. The record's is in `record/`,
a node's return in `node/`, an answer in `ask/`, and so on. Every one subclasses `Validator`, and one declared
outside a `validator.py` is refused at import.

No other module of the engine builds a validator for a type. `conductor.testing` builds one, to make up values a
node's inputs take. A saved run is read in `run/json/`, beside the code that writes it, since reading and writing
use one schema.

### A type's own rules are in its schema

What a `Series` is, a table's columns, and `Param`'s limits and choices are declared with the type.

Some rules hold for every type. That a yes or no isn't a number is one. The validation package puts such rules into
each type's schema when it builds the type's validator, once per type. So `True` is refused for an `int` input, an
item of a `list[int]`, a field of a `TypedDict` and a cell of a table alike (`bool_is_not_a_number`).

### A validator costs a run as little as it can

A value that's already exactly of a plain class (`"b"` for `str`) skips pydantic. An edge whose source's type
already fits its input isn't checked at all.

```python
from typing import Annotated

from conductor import Param


@registry.node("times", title="Times", description="Repeats a mark a number of times.")
def times(n: Annotated[int, Param(ge=1)]) -> str:
    return "x" * n  # n is an int of at least 1 here: the inputs validator read it, and nothing checks it again


counted = Graph.compile(make_graph(times), registry)
print(counted.run(n="3").outputs)  # {'result': 'xxx'}
try:
    counted.run(n=True)
except InvalidCallError as refused:
    print([found.type for found in refused.problems])  # ['invalid_input']: a yes or no is not a number
```

**Worth questioning:** a validator reports only the first fault in a node's return, but every fault in a caller's
inputs. A return is one call's outcome, so it's one problem. A caller, though, wants to fix all their inputs at
once.

## The companion packages

### `conductor-nodes`

On PyPI it's `syv-conductor-nodes`. It has the 36 standard nodes, a registry per module, and `conductor_nodes.registry`
with all of them. It has no types of its own. A failure a person can act on is a `NodeError` with a code.

A graph record names a node by its id. So an id changes only in a major release, and the changelog lists the old id
and the new one.

### `conductor-providers`

On PyPI it's `syv-conductor-providers`. It has `run_router(worker, caller=, may_see=)`, a FastAPI router over a
`Worker(store=, caller_type=, nodes_for=)`, which you mount with `app.include_router(...)`.

Compiling a graph, starting, estimating, reading, following, answering, resuming, running again and cancelling are
all routes. They work by run id, and any worker can serve them. The routes are a thin layer over `Worker`, which
holds the leases, the checkpoints and the takeovers. A host with its own routes and its own rules for who may start
a run calls `Worker` directly. [Serving runs over HTTP](serving-runs.md) walks through it.

### The host owns storage and login

`RunStore` is six async methods over one row per run. Every change is a compare-and-set on a revision. A lease lets
another worker pick up a dead worker's run from its last checkpoint.

`caller` is a FastAPI dependency. What it returns is kept on the run as JSON, so any worker can build the registry
again for that caller. Who may see a run is the host's call, through `may_see(caller, started_by)`. Both are
required. A host with no logins says so with a caller of `None`, whom everyone may see. `again` by another caller
runs the graph on that caller's own nodes.

**Worth questioning:** shipping no store means every host writes one. Storage is something every host already has,
and the recipe's plain SQL ports to Postgres line by line.

### What the router promises

- An answer sent while a run is running is queued. When the run stops, the write that records it either takes the
  answer or lists it in the run's `refused`. It never drops it. A second answer to a question already queued is a
  409.
- The item limit a run was started or resumed with holds for that run from then on.
- A cancel stays set through a takeover, so only an explicit resume starts the run again.
- One call of `resume()` may take 5 minutes by default (`resume_timeout`). A run that timed out waits for a resume.

### Tested with two real workers

`tests/test_providers/test_serving_runs.py` runs the SQLite store from [Serving runs over HTTP](serving-runs.md) in
two uvicorn processes. When a worker is killed, the other takes over its run and repeats at most 6 of 40 calls. A
deploy repeats none.

### Left out

React and Mermaid adapters (nothing reads them), an `/execute` endpoint (a run lives in a store, by id), and a store
inside the package.

## Where to start reading

Start at `packages/conductor/src/conductor/__init__.py`. Then read `_internal/` from the bottom up: `base`, `types`
and `field`, then `record`, then `graph` (`input`, `node`, `compile`, then `type`), then `run` (`blueprint`,
`outcome`, `json`, `scheduler`, `arguments`, then `type`), and `registry`. The providers' `router.py`, `worker.py`
and `store.py` come after, and `conductor-nodes` last.
