# Conductor 2.0's design: a reviewer's guide

This page is for someone reviewing how conductor is built. Topic by topic, it says what each design choice is, why it
was made, and what it costs. **Worth questioning** marks a choice a reviewer could fairly push back on, with the
reason it stands.

It assumes you've read the [first page](index.md) and [writing nodes](writing-nodes.md). The exact rules are in the
reference, `python -m conductor.about`, so this page points there instead of repeating them. Three words come up a
lot. A *hook* is a plain function passed to `@node` (`compute_outputs=`, `compute_added_inputs=`) that works out a
node's inputs or outputs when the graph is compiled. *Branches* are outputs of one node that share an
`exclusive_group`: each call returns one, and the others are `SKIPPED`. An *outcome* is what a run keeps of one call.

The code blocks run in order, as one module. The comment after each `print` is what it prints.

## What conductor is

Conductor runs graphs of typed Python functions. One person writes the functions, someone else draws the graph in an
editor, and the graph reaches the host as JSON. The core has no web framework, database or login, and it uses none of
a host's words (`tests/test_core/test_surface.py` checks that). Fitting it to one platform would make it narrower and
the code no better.

There are two audiences. Node authors use `@node`, `Param`, `Series`, `Table`, `SKIPPED`, `Ask` and the errors, and
other packages are written against that, so it's kept general and stable. Hosts compile, run, answer, save and follow
events, and that surface has only what a real host calls. Five modules are public: `conductor`, `conductor.widgets`,
`conductor.events`, `conductor.testing` and `conductor.about`. Anything in `conductor._internal` may move.

## The front door and the registry

A graph is data. There's no `@graph` decorator, and a graph built in code is the same record an editor saves. Most
graphs are drawn, and two ways to define one would mean two sets of rules.

`@node` wraps a function and registers nothing. It returns a frozen `FunctionDefinition` you can still call like the
function, so calling it is its unit test. `subgraph(...)(record)` returns a `SubgraphDefinition`, and
`@registry.node(...)` and `registry.subgraph(...)` do the same and add the node. Because a definition belongs to no
registry, it can go into any of them, and what a host offers is written in one place. **Worth questioning:** nobody
can find a node until it's added to a registry.

A registry works like a FastAPI router. `add_subregistry` copies another registry's nodes in, so a package can keep
one registry per module. When a node enters a registry, the registry's tags come first, the node's own `category`
wins, and `extra` is merged key by key, so a registry's filter key isn't lost on a node with an `extra` of its own.
The registries don't stay linked afterwards; nothing has needed that.

There's one way to do each thing. The decorator is for your own nodes, `add(*definitions)` for other people's, and
`add` raises on an id that's already there; `replace` and `remove` are explicit. `registry.filter(predicate)` returns
a new registry. `extra` is the host's JSON, which `describe()` shows and conductor never reads.

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

A record is `{"nodes": {id: node, ...}, "outputs": {name: output, ...}}` and nothing more. A node is
`{"node", "version", "inputs", "display"}`, where `node` is the id of the definition it uses. Separate fields need no
parsing, unlike one combined string. Anything a host keeps about the whole graph, like a canvas viewport, goes beside
the record. The exact shape is in `python -m conductor.about the-graph-record`.

`version` defaults to 1, and compiling never upgrades. Only `registry.upgrade` moves a graph to a newer version, and
only when asked.

Each input gets exactly one thing: a `constant`, a `graph_input` (the caller's value, maybe with a `default`), or the
`sources` of its edges. That was chosen over a map per kind of value or a `locked` flag. An input can't be given two
things, so no rule has to say which one wins. A graph input ignores the default the node declares: only the record's
`default` counts, and a graph input without one is required.

Constants, defaults and `display` are stored as JSON: a date as its ISO text, a float that isn't finite as `"NaN"`.
So a record always equals its own JSON, and its fingerprint survives a store. The cost is that a value with no JSON
form, like a polars frame, can't be a constant. `display` is the host's: conductor stores it, never reads it, and
leaves it out of the fingerprint.

Whether an input is a constant is decided per graph, because a node can't know every graph it will be used in. A node
whose input must be fixed when the graph is drawn says so with `Param(constant_only=True)`, and compiling holds every
graph to it.

An edge is stored on the input it leads into, as a source. Several sources on one input are alternatives, opposite
branches of one decision, never a list to join. So nothing a run does depends on their order.

A graph declares its inputs and outputs, as a function declares its parameters and return value. Working them out
from the first and last nodes was rejected: adding a node at the end would change what the graph returns.

`make_graph` follows scikit-learn's `make_pipeline`. With no `edges`, the nodes form a chain; it declares the graph's
inputs and outputs, and writes a record without compiling it. Every fault it can't see while writing shows up in
`Graph.compile(...).problems`, so compiling stays the one place a graph's faults are found. The cost: a script that
only saves the record hears about them only when it's compiled. The rules are in
`python -m conductor.about writing-a-graph-in-python`.

```python
record = make_graph(words, upper, join)
print(record.model_dump_json())  # {"nodes":{"words":{"node":"words","inputs":{"text":{"graph_input":"text"}}},"upper":{"node":"upper","inputs":{"text":{"sources":[{"node":"words","output":"result"}]}}},"join":{"node":"join","inputs":{"texts":{"sources":[{"node":"upper","output":"result"}]}}}},"outputs":{"result":{"node":"join","output":"result"}}}


@registry.node("half", title="Half", description="Halves a number.")
def half(n: float) -> float:
    return n / 2


typed = make_graph(half, constants={"half.n": float("nan")})
print(typed.nodes["half"].inputs["n"])  # constant='NaN'
```

**Worth questioning:** the chain never matches inputs to outputs by name. If an output and an input happened to share
a name, a chain that matched by name would wire them silently and leave the input you meant empty.

**Worth questioning:** a node used twice gets the ids `upper-1` and `upper-2`, not `upper` and `upper-2`. With the
second form, inserting another `upper` earlier would quietly move the id `upper` to a different node.

## `Graph` and `Run`

`Graph.compile` never raises for a fault in the graph. Each fault is a `Problem` an editor can mark on the node. It
raises only when it isn't given a record at all (`invalid_record`): getting the shape right is the caller's job, and
what the record says is the graph's. A graph input with no value yet is fine, like an unfitted scikit-learn pipeline.
A compiled graph keeps its own copy of its definitions, so a later change to the registry can't break it.

There's one way in. A host never builds a `Graph` or a `Run` itself: `Graph.compile` makes a graph, and a graph
makes its runs. `graph.with_inputs` gives a run that hasn't started, and `graph.run` is that plus `resume()`. On a
run, `with_inputs` and `again` do the same with some inputs changed. `answer` returns another run and runs nothing.

Running again reuses by content. A call reuses an earlier output when it gets exactly the arguments a call of the
same node got before, under the same version and code. A node that may ask a person reuses its earlier outcomes in
item order instead, so inserting an item keeps everyone's answer. The rules, with `forget` and
`with_inputs(previous=)`, are in `python -m conductor.about running-again`.

Errors are data. There's one `Problem` record everywhere, with stable codes in `PROBLEMS.codes` for a host to
translate. A message says how to fix the fault. A problem never quotes a value a graph carries (an input, a constant,
an output, an answer, a saved value) or the text of an exception other than a `NodeError`: a node's or a hook's own exception text goes to the log,
and the problem names only the exception's class. It may name what a developer or a host wrote, such as ids, places
and field names, a hook's field names among them. A `NodeError` keeps its message, which its developer wrote for people. The reference's errors
section has the rule: `python -m conductor.about errors`.

```python
graph = Graph.compile(record, registry)
print(graph.run(text="a b").outputs)  # {'result': 'A+B'}
```

**Worth questioning:** the code in the reuse key is the node function's own source as Python parses it, so a change to
a helper it calls isn't noticed. A node that depends on more than its inputs sets
`Policy(reuse_when_inputs_match=False)`. Hashing all the code a node can reach would be slow, and would still miss the
outside world.

**Worth questioning:** earlier outputs a node didn't reuse are dropped once it's complete, so undoing an edit two runs
later calls the node again. Without the drop, a saved run would grow with every edit. A node with a question still
waiting keeps them.

## One or many

The types decide how often a node runs. A `Series[T]` output wired to an input of one `T` starts a loop: that node
runs once per item, and so does everything downstream. A `Series` or `Table` input collects the loop again. The node
never knows. A "loop over this" flag would say what the types already say: `contract: File` runs per item,
`contracts: Series[File]` once.

The type decides, never the value. A list given to an input of one value is refused. If a caller's list could start a
loop, how often a node runs would depend on each run's inputs, and an editor couldn't show it in advance.

A table is one value. It never starts a loop, and an edge can't name one of its columns. To run per record, you draw
a Split node. If tables started loops by themselves, how often a node runs would depend on which columns happened to
be wired, which an author can't see in the drawing.

Two lists never pair up by themselves. Ten documents and three questions are refused as `series_not_nested` until one
comes from inside the other's loop, or one goes whole into a `Series` input. Crossing them would quietly multiply paid
calls, and which item goes with which is the author's call.

A collecting input gets the items that have a value and waits for all of them. No items is an empty list, and the
node is still called: "0 approved" is an answer. `max_items=` caps any list a loop runs over, checked before any call
for its items, so a run stops before it spends. The rules are in `python -m conductor.about one-or-many`.

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

**Worth questioning:** a list sent to a `str` input is refused, where it could have run the node once per item. It
stands because a graph then runs the same way whatever it's given.

**Worth questioning:** a table needs a Split before a node can run per record. That's where an author sees the run
fan out, and a table that's only passed along costs nothing.

## `None` and values from outside a loop

A `None` skips one call. If an argument is `None` and the input's type lacks `| None`, the node isn't called for that
item, and `run.not_called` records which item and input. Failing the whole run, after other items already had side
effects, would be worse.

Each spelling says one thing: `Series[X]` leaves `None` items out, `Series[X | None]` keeps them, and `Series[X] | None`
is a list that may itself be `None`.

`SKIPPED` is a branch not chosen, not a `None`. It skips the nodes downstream, for one item inside a loop or
altogether outside one. A skip wins over a `None` or a fault on another input.

A value from outside a loop is the same on every call. Nothing is copied, and it starts no loop of its own.

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

**Worth questioning:** the run ends `complete` with one message not sent. It stands because the run records which
item and which input, so a host can say "Bo was not sent: no middle name" instead of failing everyone.

## Asking a person

`raise Ask(prompt, choices=, default=)` pauses one call. That call and whatever reads it wait, and the rest of the
run goes on. The return type says what the answer is: `bool` a yes or no, a `Literal` one of its values, a frozen
dataclass a small form. `AskBool` or a `type=` argument would declare the answer's type a second time.

A node that may ask says `asks=True` on its `@node`, and raising `Ask` without it fails as `ask_not_declared`. In
`registry.describe()` each node says whether it `asks` (for a subgraph, whether any node inside does), so a host knows
which nodes can pause. It's on the decorator because asking is something the function does, like raising, not
something its value is. **Worth questioning:** the signature alone doesn't say everything the node does.

There's one way to ask, and the answer becomes the node's outputs, as if it had returned them. A node that needs the
answer reads it from an edge, like any value: `approval(invoice) -> bool` asks for a large invoice and is wired into
`pay(invoice, ok: bool)`. The answer then has one shape and one place where it's kept, saved and reused. A parameter
filled by a function that asks, a replayed `ok = ask(...)`, and calling the node again with `ok=None` were all
rejected. **Worth questioning:** the question shows on `approval`, not on the step a person thinks of.

A question is the same record as a field: the `FieldDescription` a palette reads, with the `Ask`'s `choices` and
`default`. How it looks comes from the type's `Param`, so `Ask` takes no `widget=`. One JSON Schema for the whole
answer was rejected: it loses a host type's id and exclusive groups, and has no place for a widget.

An answer is read strictly from its JSON form (`true`, never `"yes"`), and every fault is refused at once. A waiting
id like `approve@1~3fa2c0d1` names the call and a fingerprint of the node and the call's inputs; the reference's `asking-a-person` section says what it covers. So an
answer to a place that waits with another fingerprint is refused as `answer_stale`, and one to a place that waits on
nothing as `not_waiting`. The rules are in `python -m conductor.about asking-a-person`.

```python
from typing import Literal

from conductor import Ask


@registry.node("tone", title="Tone", description="Asks a person which tone a text should have.", asks=True)
def tone(text: str) -> Literal["formal", "casual"]:
    raise Ask(f"Which tone for {text!r}?")


asked = Graph.compile(make_graph(tone), registry).run(text="hello")
print(asked.answer(asked.waiting[0], "casual").resume().outputs)  # {'result': 'casual'}
```

**Worth questioning:** `Ask` derives from `BaseException`. Otherwise a node's `except Exception` would swallow the
pause and carry on with a wrong value.

**Worth questioning:** an id without its `~` part still answers the call waiting there. A host stores the full id,
which the fingerprint guards. The short id is what a person types about the run in front of them.

## The engine

The core is small. Compiling decides once, from the types alone, how often each node runs. A run keeps one outcome per
call, keyed by node and item indexes. Most of its speed comes from calling nodes inline and from checking values only
where a check can fail.

Nodes run on the run's own thread by default. An author asks for threads by setting `concurrency` above 1 or an
`attempt_timeout`. Threads by default would surprise code that isn't thread-safe. A `concurrency` a node sets counts
its calls across the whole process, every graph and every run, so it can keep to an outside service's quota. Left at
`None`, runs don't wait for each other, since a lock on every call would slow every run for a quota no node asked for.

A node's return is read strictly against its type. Compiling checks that each edge's types fit. A value on an edge
is checked again only where the source's type doesn't already guarantee it fits the input. Which edges those are is
worked out once, when the graph's first run is made or read, so a plain chain checks nothing at run time.

Only an `ExternalError`, or a class the node's `retry_on` names, is retried. A `NodeError`, a refused output and an
`Ask` never are, because calling again would give the same result. The rules are in
`python -m conductor.about retries-and-timeouts`.

A subgraph is compiled on its own, and stays one node of the outer graph, but a run only ever calls its inner
function nodes (`shout/upper` in a saved run). They have one call per item of every loop they're inside, so they
behave exactly as if drawn in the outer graph. A separate run per item was rejected. Measured on 2,000 items through
a two-node subgraph, a run per item saved about 8 times the bytes of one run (691 KB against 86 KB). [A graph inside a
graph](subgraphs.md) teaches it.

```python
from conductor import subgraph

shout = subgraph("shout", title="Shout", description="Turns a text into capital letters.")(make_graph(upper))
registry.add(shout)
shouted = Graph.compile(make_graph(shout, join, edges=[("shout.result", "join.texts")]), registry)
print(list(shouted.inputs), shouted.outputs)  # ['text'] ['result']
```

**Worth questioning:** a host whose slow nodes forget `concurrency` runs them one at a time within a run. It stands
because the slow node knows it's slow, and saying so is one argument.

## Saved runs, checkpoints and events

A run is one JSON text: `run.to_json()` writes it, and `Run.from_json(saved, graph)` reads it back, refusing a run of
another graph by fingerprint. It survives a store that reorders keys, such as Postgres `jsonb`, and has only the
nodes that were called. Each node's outcomes are one list per output, in item order, like
`"upper": {"states": ["done", "done"], "outputs": {"result": ["A", "B"]}}`. Item indexes aren't written, so a call the
run never made can't be either. A damaged run is one `saved_run_unreadable` problem that names the place.

A `Series` of records keeps the table it came as. Split returns a table, an edge from `series-split.result.email`
reads that column, and a node that takes a record still gets a plain one. Making the records up front would cost most
of the run. On 200,000 rows read by one field, a run took 149 ms with the records made up front and 10 ms with the
table kept.

A float that isn't finite is saved as its text, and its declared type reads it back: `float` gives the float, and
`Any` gives the text. Marking each such float in the saved run would state its type a second time.

Checkpoints come from the stream. `run.stream(checkpoint_every=seconds)` sends `run_checkpoint` events with a copy of
the run. A process that dies repeats the calls in flight, and the calls recorded since the last checkpoint. A host
that shuts down passes `hand_back=` and gets the run back still `running`, for another process to go on with. A
callback was rejected as a second channel, and cancelling and handing back stay two acts so nobody's run is saved as
`cancelled` by mistake.

Events carry counts, not results, so they stay small however big a table is. The exceptions are what a host needs to
act on: `node_failed` and `node_retry` carry a `Problem`, `run_end` the run's `cause`, and `node_waiting` the
`Waiting` call with its question and the arguments the call got. See `python -m conductor.about events`.

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

Every record is a frozen pydantic model: `GraphRecord`, `Problem`, the events, `Waiting` and every `describe()`
result. Each works as a FastAPI response model as it is. `Policy` is one too, so `Policy(retries="2")` raises right
where it's written.

`Param` is pydantic metadata. Its limits and choices work inside a host's own `BaseModel` too, and they live in the
field's JSON schema. On a node's input, `Param` is the only spelling: pydantic's `Field(...)` in the hint is refused
(`pydantic_field_in_hint`), so a limit is never written two ways.

Values a program made (answers, choices, node outputs) are read strictly, and values a person or caller wrote
(constants, defaults, graph inputs, edges) laxly. Either way a `bool` never counts as a number.

Each type hint is read once into one object, and pydantic does all the validating. The one thing pydantic can't say is
whether an output of one type may feed an input of another, so conductor walks the two types itself. Answering that
from pydantic's JSON schema accepted edges whose values were then refused at run time.

`Series[T]` and `Table[R]` are plain generic classes with their own pydantic schema. So a table can sit in a union, a
`Series` or a host's model, with one JSON form everywhere. A value read as `Table[R]` has exactly `R`'s columns.
**Worth questioning:** a type checker accepts a `Table[Sale]` where a `Table[Order]` is expected. A separate generic
would refuse it, but every plain `polars.DataFrame` return would then be a type error.

A node can be written for any type: `def split[T](value: Table[T] | list[T]) -> Series[T]` takes the type on its
edge, settled when compiling. `series-split`, `series-stack` and `control-decide` work this way, with no hooks. A hook
is only for inputs or outputs named by a constant.

A `FieldDescription` is one flat record per input or output. `null` in `default` means `None` and nothing else, and
"no default" and "a branch not chosen" are told apart by `required`, so the record needs no marker JSON lacks. A
`default` on a `required` field is where an editor starts: the one question of a node with one output is answered
with its value, so it is required, unless the output is in an exclusive group, whose outputs never are. In
`graph.describe()`, a node's input is `{field, source}`, and a graph input is a plain `FieldDescription`.

What a person sees for a type is declared. Conductor's own types take their id and title from one fixed table, and a
host's type is declared with `@host_type("id", title="Title")`, checked when the class is defined. Nothing falls back
to a class's name. Every Python block in the docs is type-checked under mypy and pyright.

```python
from typing import Annotated

from pydantic import BaseModel

from conductor import Param


class Settings(BaseModel):
    temperature: Annotated[float, Param(ge=0, le=2)] = 1.0


print(Settings.model_json_schema()["properties"]["temperature"])  # {'default': 1.0, 'maximum': 2, 'minimum': 0, 'title': 'Temperature', 'type': 'number'}
```

**Worth questioning:** `@host_type` is a decorator, not a base class. A base class's hook runs before `@dataclass`
has made the class, so it can't check a dataclass, and a library's class can't be given a base class at all.

**Worth questioning:** `@host_type(..., accepts=[...])` takes a list of classes whose values may be wired into the
type, so a rule that isn't a list can't be said. But a list is data an editor can show before anything runs.

## Validation: where values are checked

A value from outside is checked once, where it comes in, and never again. Each place it comes in has its own
validator: a graph record, a caller's inputs, a person's answer, what a node returns, a saved run, and so on. A
validator returns the value as the engine keeps it, or a problem with its own code (`invalid_input`,
`invalid_output`, ...). Past that wall the engine trusts what it has.

Each validator lives in the package of what it checks, in its `validator.py`, and subclasses `Validator`. One declared
anywhere else is refused at import. A rule that holds for every type, like "a yes or no isn't a number", is built into
each type's schema once, so `True` is refused for an `int` input, a list item and a table cell alike.

A validator costs a run as little as it can. A value already exactly of a plain class skips pydantic, and an edge
whose source already fits isn't checked at all.

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

**Worth questioning:** a validator reports the first fault of a node's return, but every fault of a caller's inputs.
A return is one call's outcome, while a caller wants to fix all their inputs at once.

## The companion packages

`syv-conductor-nodes` has the 36 standard nodes, a registry per module, and `conductor_nodes.registry` with all of
them. It has no types of its own. A graph record names a node by its id, so an id changes only in a major release.

`syv-conductor-providers` has `run_router(worker, metadata=, run_filter=)`, a FastAPI router over a
`Worker(store, nodes_for=)`. Compiling, starting, estimating, reading, following, answering, resuming,
running again and cancelling are routes, and any worker can serve them. Every route but `/compile`, `POST /runs`,
`/estimate` and `/nodes` has a run id in its path. The routes are a thin layer over `Worker`, which holds the
leases, checkpoints and takeovers, so a host with its own routes calls it directly.

The host owns storage and login. `RunStore` is six async methods over one row per run, each change a compare-and-set
on a revision. Each run keeps the host's metadata, and which runs a request may see is the host's `run_filter`, a test of that
metadata. **Worth questioning:** shipping no store
means every host writes one. But every host already has storage, and the recipe's plain SQL ports to Postgres line by
line.

The router never drops an answer: one sent while a run is running is queued, then taken or listed in `refused`. It's
proved with two real workers in `tests/test_providers/test_serving_runs.py`: a killed worker's run is taken over and
repeats at most 6 of 40 calls, and a deploy repeats none. [Serving runs over HTTP](serving-runs.md) walks through it.
React and Mermaid adapters, an `/execute` endpoint and a built-in store were left out.

## Where to start reading

Start at `packages/conductor/src/conductor/__init__.py`. Then read `_internal/` from the bottom up: `base`, `types`
and `field`, then `record`, then `graph` (`input`, `node`, `compile`, then `type`), then `run` (`blueprint`,
`outcome`, `json`, `scheduler`, `arguments`, then `type`), and `registry`. The providers' `router.py`, `store.py`,
`worker.py` and `held_run.py` come after, and `conductor-nodes` last.
