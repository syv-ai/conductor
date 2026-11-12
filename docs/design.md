# Conductor 2.0's design: a reviewer's guide

This page is a reviewer's guide to 2.0's design. It goes topic by topic: what each choice is, why it was made, and what it was chosen over. **Worth questioning** marks a choice a reviewer could fairly push back on, with the reason it stands. It assumes the [first page](index.md) and [writing nodes](writing-nodes.md). Three words recur: a *hook* is a plain function on `@node` (`compute_outputs=`, `compute_added_inputs=`) that works out a node's inputs or outputs from the graph's constants when it is compiled; *branches* are outputs of one node that share an `exclusive_group`, of which each call returns only one, the others `SKIPPED`; and an *outcome* is what a run keeps of one call of a node. The code blocks run in order as one module, and the comment after each `print` is what it prints.

## What conductor is

- **A general library for running graphs of typed Python functions.** Someone writes the functions, someone else draws the graph in an editor, and the graph reaches the host as JSON. The core has no web framework, database or login. Its source uses no host's words, and `tests/test_core/test_surface.py` checks that. Fitting it to one platform would narrow the library and give the code nothing.
- **Two surfaces, two bars.** The node author's surface (`@node`, `Param`, `Series`, `Table`, `SKIPPED`, `Ask`, the errors) is general and stable, because other packages are written against it. The host's surface (compile, run, answer, save, events) has only what a real host calls. Five modules are public: `conductor`, `conductor.widgets`, `conductor.events`, `conductor.testing` and `conductor.about`, the reference. Anything in `conductor._internal` may move.

## The front door and the registry

- **A graph is data.** There is no `@graph` decorator and no placing nodes in Python. A graph built in code is the same record an editor saves. Most graphs are drawn, and two ways to define one would mean two sets of rules.
- **`@node` wraps a function and registers nothing.** It returns a frozen, callable `FunctionDefinition`, so calling it like the function is its unit test; `subgraph(...)(record)` returns a `SubgraphDefinition`, which has no function to call. `NodeDefinition` is either one. `@registry.node(...)` and `registry.subgraph(...)` take the same arguments as the two, and add the node.
- **`NodeRegistry` works like a FastAPI router.** `add_subregistry` copies another registry's nodes in, so a package keeps one registry per module. One merge rule applies wherever a node enters: the registry's tags come first, then the node's; the node's own `category` wins; `extra` merges key by key. Replacing `extra` whole would drop a registry's filter key from any node with an `extra` of its own. Subregistries that stay linked, so a node added to one later shows up in the other, were not built, because nothing needs them.
- **One way each.** The decorator is for your own nodes, `add(*definitions)` for other people's, and the constructor takes no node list. `add` raises on a repeated id; `replace` and `remove` are explicit.
- **`registry.filter(predicate)` returns a registry of its own.** The predicate gets the whole definition. `extra` is the host's JSON, typed as JSON: shown by `describe()`, never read by conductor.

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

**Worth questioning:** a node that registers nothing needs a registry before anyone can find it. This holds because one definition can then go into any registry, and what a host offers is written in one place: where it builds the registry.

## The graph record and `make_graph`

- **A record is `{"nodes": [...], "edges": [[from, to], ...]}`**, nothing more: what a host keeps about the whole graph, such as a canvas viewport, it stores beside the record. A node is `{"id", "node", "version", "constants", "defaults", "display"}`, with empty fields left out. `id` names this node in this graph, and `node` is the id of the node definition it uses. One combined string would need parsing everywhere. In conductor's JSON, `node` always means a node definition's id and `node_id` a node's id in a graph.
- **`version` defaults to 1, and compiling never upgrades.** `registry.upgrade` does, when asked.
- **`constants` are fixed; `defaults` are prefilled and stay open.** There is one map per intent, chosen over a `locked` flag beside one map. A default is allowed only where a caller can give the value: on a graph input, an input with no edge on a node with no incoming edges. Anywhere else it is `default_not_a_graph_input`.
- **Constants, defaults and a node's `display` are JSON.** A constant is kept as its JSON reads back: a date as its ISO text, a float that isn't finite as `"NaN"` or `"Infinity"`. So a record always equals its own JSON, and its fingerprint survives a store. A value with no JSON form, such as a polars frame, is `invalid_value` at the call. A hand-built record with a float that isn't finite in it is `invalid_record`.
- **A node's `display` is the host's.** Conductor stores it, never reads it, and leaves it out of the fingerprint. It sits on the node so it follows the node through an upgrade.
- **There is no flag on an input that makes it a constant.** A node can't know every graph it is used in, so whether an input is a constant is said per graph: a constant in the record is the graph author's value, a graph input is the caller's.
- **Edges are a sorted list of address pairs** (`node.field`, or `node.output.field` for one field of a record). Two records that differ only in edge order are equal, and nothing a run does depends on the list's order: several edges into one input are alternatives, opposite branches of one decision, never a list to join. So a host may sort the list. Edges stored per node were rejected: reconnecting reorders either shape.
- **`make_graph(*nodes, edges=, constants=, defaults=)`** follows scikit-learn's `make_pipeline`. With no `edges` the nodes form a chain: each node takes its one required input with no constant from the one output of the node before it, or the call fails with `not_a_chain`. With `edges`, those are every edge. It writes a record and compiles nothing. It raises one `InvalidCallError` listing what it can't write (a value for a node it wasn't given, a value with no JSON form, an edge that is no address, a chain it can't draw); everything else, a mistyped input or an edge between types that don't meet included, is in `Graph.compile(record, registry).problems`, so compiling is the one place a record's faults are found. A script that only saves the record hears nothing until it is compiled.

```python
record = make_graph(words, upper, join)
print(record.model_dump_json())  # {"nodes":[{"id":"words","node":"words"},{"id":"upper","node":"upper"},{"id":"join","node":"join"}],"edges":[["upper.result","join.texts"],["words.result","upper.text"]]}


@registry.node("half", title="Half", description="Halves a number.")
def half(n: float) -> float:
    return n / 2


typed = make_graph(half, constants={"half.n": float("nan")})
print(typed.nodes[0].constants)  # {'n': 'NaN'}
```

**Worth questioning:** the chain never matches by name. A chain that wired inputs to outputs by their names would, given an output and an input that happen to share a name, wire them without saying so, and leave the input the author meant empty. A chain that refuses to guess can't do that.

**Worth questioning:** a repeated node is `upper-1`, `upper-2`, not `upper`, `upper-2`. The second form silently moves `upper` to whichever node is inserted before it.

## `Graph` and `Run`

- **`Graph.compile(record, registry)` never raises for a fault in the graph.** Each fault is a `Problem` (`type`, `loc`, `msg`, `ctx`) an editor can mark on the node. Only input that isn't a record at all raises (`invalid_record`). The shape is the caller's to get right; what the record says is the graph's.
- **Nobody declares a graph's inputs or outputs.** Its inputs are the inputs with no edge on the nodes with no incoming edges; its outputs are the outputs of the nodes with no outgoing edges. A graph input with no value yet is not a problem, just as an unfitted scikit-learn pipeline isn't broken.
- **A compiled graph keeps its own copy of its node definitions**, so a later change to the registry can't break it.
- **One way in, and verbs that mirror each other.** `Graph(...)` and `Run(...)` refuse to be constructed. `graph.with_inputs` checks inputs and gives a run that hasn't started; `graph.run` is that plus `resume()`. On a run, `with_inputs` and `again` do the same with changed inputs, reusing what they can of this run. `answer` returns another run and runs nothing.
- **`again` reuses by content.** A call reuses a previous call's output when it would get exactly the arguments a call of the same node got in the previous run, under the same version, code, inputs and outputs. Identical items share one previous output. A person's answers are taken in item order, first to first and second to second, so an item inserted above keeps everyone's answer. `run.forget(*ids)` calls chosen nodes again, for a host that must recheck access. `edited.with_inputs(..., previous=old)` reuses the old run across an edit, for the nodes the edit left alone.
- **Errors are data.** One `Problem` record is used everywhere, with stable codes listed in `PROBLEMS.codes` for a host to translate by. A message says how to fix the fault and never quotes a value. A node's own exception text goes to the log. A caller's input named twice, by name and by address, is `input_given_twice` from every verb that takes inputs.

```python
graph = Graph.compile(record, registry)
print(graph.run(text="a b").outputs)  # {'join.result': 'A+B'}
```

**Worth questioning:** the code in the reuse key is the node function's own bytecode and constants, so a changed helper isn't seen. A node that depends on more than its inputs sets `Policy(reuse_when_inputs_match=False)`. Hashing all reachable code would be slow and would still miss the outside world.

**Worth questioning:** reusable outputs from the previous run that a node didn't use are dropped once the node is complete, so an edit undone two runs later calls the node for that item again. Without the drop, a saved run grows with every edit. A node with a question still waiting keeps them, so an answer survives an edit and its revert.

## One or many

- **The types decide how often a node runs.** A `Series[T]` output wired to an input of one `T` starts a loop: that node runs once per item, and so does every node downstream of it. A `Series` input collects the loop: its node is called once with every item; a `Table` input collects records the same way. An outer value, from outside the loop, is the same on every call. The node never knows. A mark on an input saying "loop over this" and a per-node switch were rejected, because the types already say it: `contract: File` runs per item, `contracts: Series[File]` runs once.
- **The type decides, never the value.** A list given to an input of one value is refused. A node that returns many values says so in its return type, `Series[...]`, so one file and ten files take the same path. Letting a caller's list start a loop was rejected: how often a node runs would depend on what a run was given, and a graph would need compiling again for each run.
- **A table is one value.** It never starts a loop by itself, and an edge can't name one of its columns (an edge can name a field of each record of a `Series`, below). A Split node takes it apart, drawn where the author wants it. Tables that started loops by themselves were rejected: how often a node ran then followed from which columns happened to be wired to it, which an author can't see in the drawing.
- **Items of two series never pair up.** Ten documents and three questions are refused as `series_not_nested` until one is returned by a node inside the other's loop (the node that returns the questions takes a document), or one is given whole to a `Series` input. Crossing them and pairing them by position were both rejected. Crossing multiplies paid calls silently, and which item goes with which is the author's to draw.
- **Collecting ends one loop, and has three rules.** A `Series` input gets the items that have a value, leaving out `None` and skipped items; when no item could have a value, its node is not called or skipped as any node would be; it waits for every item. No items is an empty list, and the node is called: "0 approved" is an answer.
- **An item limit before spending.** `max_items=` is an optional limit on the length of any one series a loop runs over, checked when the series is returned, before any call for its items. Over it the run is `stopped`, and `run.with_max_items(100).resume()` goes on under a higher one, calling nothing again.

```python
print({node_id: runs.msg for node_id, runs in graph.runs_on.items()})  # {'words': 'once', 'upper': 'per item of words.result', 'join': 'once'}


@registry.node("greet", title="Greet", description="Greets a person by name.")
def greet(name: str, greeting: str) -> str:
    return f"{greeting} {name}"


names_and_greetings = make_graph(words, words, greet, edges=[("words-1.result", "greet.name"), ("words-2.result", "greet.greeting")])
print([problem.type for problem in Graph.compile(names_and_greetings, registry).problems])  # ['series_not_nested']

try:
    graph.run(text=["a", "b"])
except InvalidCallError as e:
    print([p.type for p in e.errors()])  # ['invalid_input']
```

**Worth questioning:** a list sent to a `str` input is refused, where it could have run the node once per item. This holds because a graph then runs the same way whatever it is given, and an editor can show how often each node runs before anyone runs it.

**Worth questioning:** a table needs a Split before a node can run per record. This holds because the Split is where an author sees the run fan out, and a table that is only passed along costs nothing.

## `None` and outer values

- **A `None` keeps the node from being called for that item.** If an argument is `None` and the input's type lacks `| None`, the node isn't called there, its outputs are `None` (of branches in an `exclusive_group`, none is chosen), and `run.not_called` records the node, the item and the input. Pydantic refuses `None` for a `str`, and failing the whole run after other items' side effects went out is worse.
- **`Series[X]` leaves `None` out, `Series[X | None]` keeps it, and `Series[X] | None` is a list that may itself be `None`.** Each spelling says one thing. `Table[R] | None` reads the same way: a table that was never returned reaches the input as `None`.
- **`SKIPPED` is a branch not chosen**, not a `None`. Inside a loop it skips the nodes downstream for that item; outside a loop it skips them altogether. A skip wins over a `None`: a node with one input skipped and another `None` is skipped, whichever comes first in its signature.
- **An outer value is the same on every call of the loop.** Nothing is copied, and it starts no loop of its own: wiring in a document's name changes nothing about how often a node runs.

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

**Worth questioning:** the run ends `complete` with one message not sent. This holds because the run records which item and which input, so a host can say "Bo was not sent: no middle name" instead of failing everyone.

## `Ask`

- **`raise Ask(prompt, choices=, default=)` pauses one call.** The return type says what the answer is and how to ask: `bool` is yes or no, a `Literal` one of its values, a frozen dataclass a form. `AskBool`, `AskMulti` and a `type=` argument were rejected as a second declaration. That call and whatever reads it wait; the rest runs on.
- **A node that asks says so where it is declared**: `asks=True` on its `@node`, and a raise without it fails as `ask_not_declared`. A host learns from `describe().asks` which graphs can pause, without guessing from the code. The word is on the decorator, not in the return type: asking is something the function does, like raising, not something its value is, and a marker on the return type would be a second word beside `Asked` and would sit oddly on `-> None` or a dataclass of outputs. **Worth questioning:** the signature no longer says everything a node does.
- **A parameter can be the question** (`ok: Annotated[bool, Asked(approval)]`), filled by a function that returns a value or raises `Ask`. It is for a node whose outputs are more than the answer, an approval that hands the invoice on: one box where two were needed, and the body called once with a real `bool`. One rule covers both: a function that raises `Ask` is answered with a value of its return type. Rejected: a pre-built approval node (the person must send the invoice back, and a forged one goes on), a static `Asked[bool]` (it can't skip the question or propose a draft), `ok = ask(...)` replayed (the body runs twice, and nothing is known at load), the node called again with `ok=None` (a body that forgets the check never asks), and two-phase nodes.
- **`Ask` carries only what the run knows, typed.** How a question looks is its type's `Param` (a widget, a title, choices known when the node is written), so a hook's output types carry it too; `widget=` on `Ask` was a second home for it and went. `-> str` with a bare `Ask(prompt)` is free text. Each argument has the answer's shape: a value for one output, a mapping by output name, naming only some, for several. There is no plural `defaults=`.
- **A question is the same record as a field:** the `FieldDescription` a palette reads, with the `choices` and `default` the `Ask` set. A host draws it the way it draws an input. One JSON Schema for the whole answer was rejected: it loses a host type's id, whether a field is one or many, and exclusive groups, and has no place for a widget.
- **`run.answer(waiting, value)` or `run.answer({id: value, ...})`.** Every fault is refused at once and nothing is half-answered. Each answer is read strictly from its JSON form (`true`, never `"yes"`). A declared choice is read exactly the same way, held to the field's limits, so an enum's value or a date's ISO text is a choice. A choice's id is its value as the field's type writes it in JSON: `Decimal("1.0")` has the id `"1.0"`, and a `datetime` its ISO text, `"2026-01-02T03:04:05"`.
- **A waiting id names the call and what it was given**, `approve@1~3fa2c0d1` for a call inside a loop (`@` and an index per loop) and `tone~400b7c9a` for a node inside none; the part after `~` is a fingerprint of the node, its code and the call's arguments. An answer meant for a question that has moved is `answer_stale`.

```python
from typing import Literal

from conductor import Ask


@registry.node("tone", title="Tone", description="Asks a person which tone a text should have.", asks=True)
def tone(text: str) -> Literal["formal", "casual"]:
    raise Ask(f"Which tone for {text!r}?")


asked = Graph.compile(make_graph(tone), registry).run(text="hello")
print(asked.answer(asked.waiting[0], "casual").resume().outputs)  # {'tone.result': 'casual'}
```

**Worth questioning:** `Ask` derives from `BaseException`. Otherwise a node's `except Exception` would swallow the pause and carry on with a wrong value.

**Worth questioning:** an id without its `~` part still answers the call that waits there. A host stores the full id, which the guard covers. The bare id is what a person types about the run in front of them.

## The engine

- **A small core.** Compiling decides once, from the declared types alone, how often each node runs. A run keeps one outcome per call, keyed by node and item indexes (a tuple of numbers, one per loop). A table stays the frame it is. The parts pay off together: the core's speed comes from calling inline and from checking values only where a check can fail.
- **Nodes are called on the run's own thread by default** (`Policy(concurrency=1)`). An author asks for threads with `concurrency` or `attempt_timeout`. Threads by default would surprise code that isn't thread-safe, and calling inline is most of the core's speed.
- **Values are checked where it matters.** A node's return is read strictly against its type (`invalid_output`). An edge's value is checked again only where its source's type doesn't already guarantee it fits, decided when compiling, so a plain chain checks nothing at run time.
- **Only `ExternalError`, or a class the node's `retry_on` names, is retried.** A `NodeError` of the node's own, an output its type refused and an `Ask` are never retried, whatever `retry_on` says: repeating the call repeats them. After a failure, calls already on threads finish and are kept, so resuming repeats none.
- **A graph record can be a node, a subgraph:** `subgraph("id", title=..., description=...)(record)`, taking no policy or hook, as its inner nodes have their own. From outside it is one node, whose inputs and outputs are those of its graph compiled alone. Compiling the outer graph lays out the subgraph's inner function nodes in it, each named by its node path (`shout/upper` in a saved run), so the compiled graph and a run have function nodes only. An inner node has its calls as any node does, one per item of every loop it is inside, outside the subgraph and in it. So a `None` or a skip that arrives reaches the inner nodes wired to it, exactly as if they were drawn in the outer graph, and the subgraph node is never called, skipped or not called itself. A separate run per item of the subgraph was chosen against: a saved run of 2,000 items was 34 times the size of the same nodes drawn in one graph.

```python
from conductor import subgraph

shout = subgraph("shout", title="Shout", description="Turns a text into capital letters.")(make_graph(upper))
registry.add(shout)
shouted = Graph.compile(make_graph(shout, join, edges=[("shout.upper.result", "join.texts")]), registry)
print(list(shouted.inputs), shouted.outputs)  # ['text'] ['join.result']
```

**Worth questioning:** a host whose slow nodes forget `concurrency` runs them in series. This holds because the slow node knows it's slow, and saying so is one argument.

## Checkpoints and serving

- **A run is one JSON text.** `Run.from_json(saved, graph)` checks it against the graph and refuses another graph's saved run by fingerprint (`saved_run_of_another_graph`). Saved runs survive a store that reorders keys, such as Postgres `jsonb`, and have only the nodes that were called, so a long graph waiting at its start saves small. A saved run's shape is declared once per graph, as one pydantic schema made from the schema of each input's and each output's own type, the first time a run of the graph is written or read; `to_json` is one write through it and `Run.from_json` one read. Each node's outcomes are saved as one list per output, one entry per call in item order, `"shout": {"states": ["done", "done"], "outputs": {"result": ["A!", "B!"]}}`, so item indexes are never written (reading works them out again from the saved series' lengths) and a call the run never made can't be written. A waiting call's entry is its question, so a waiting call can't be saved without one. A saved run has one reader, its JSON text, and a damaged one is one `saved_run_unreadable` whose `ctx.where` is the dotted place pydantic names, `outcomes.shout.outputs.result.0`.
- **A `Series` of records keeps the table it was returned as.** A node that returns a table for `Series[R]` (Split does) has made no record yet, and most nodes downstream want one field: an edge from `series-split.result.email` reads the column. A node that takes the record, or the whole `Series`, receives plain records, so an author never meets a second kind of value. The table is checked by its columns when it is returned, and a saved run writes it once, as a table. Making every record up front was the whole cost of Split on a run.
- **A float that isn't finite is its text in a saved run, and its declared type reads it back.** `float` gives the float; `Any`, a `dict` and `float | str` give the text, as a date under `Any` comes back as its ISO text. An author who wants the float back declares it. Marking each such float's place in the saved run was rejected: it would be a second statement of a value's type, beside the type.
- **Checkpoints come from the stream.** `run.stream(checkpoint_every=seconds)` sends `run_checkpoint` events carrying a copy of the run. Resuming one calls only the calls it has no outcome for, so a process that dies while running repeats at most the calls that were in flight. A callback argument was rejected as a second channel beside the stream. A host that shuts down hands the run back with `run.stream(hand_back=event)`: the stream stops as for a cancel, and its last event is a checkpoint of the run still `running`, with no `run_end`, so the stored run says what its row says and another process goes on from it. Cancelling a run and handing it back are two acts; one event for both would save a run nobody cancelled as `cancelled`.
- **Events carry counts, not values.** `node_complete` says how many calls a node had, and the values stay in the run, so an event is small whatever the size of the table. The one exception is a `Problem`: `node_failed` carries one, and `run_end` carries the run's `cause`, so a run stopped at its item limit says how many items there were and a client can choose a higher limit.

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

## The pydantic seam

- **Every record is a frozen pydantic model:** `GraphRecord`, `Problem`, the events (one union on `type`), `Waiting`, and every `describe()` result. Each serves as a FastAPI response model as it is.
- **`Param` is pydantic metadata.** It implements pydantic's schema hooks, so its limits and its choices apply inside a host's own `BaseModel` as on a node's input: the choice rule has one home, pydantic's schema hook on `Param`. A field's limits live in its JSON schema, not in a separate list. On a node's input or output `Param` is the only spelling: pydantic's `Field(...)` in the hint is refused (`pydantic_field_in_hint`), so a limit is never written two ways. A pydantic type that has its limits (`PositiveInt`) is a type and is read as pydantic reads it.
- **Strict where a program produced the value, lax where a person or caller wrote it.** Answers, declared choices and node outputs are read strictly (`"12"` is never an `int`). Constants, defaults, graph inputs and edges are read laxly. A `bool` never lands on a number, nor on a union a number is in and a `bool` isn't (`int | float`), as a value or in a table's column.
- **One reading of an annotation.** Each hint is read once into one object that names the type, gives its pydantic schema, writes a value as JSON and says which types may land on it. It reads no value itself: the gate does (next section). Validation is pydantic's throughout. The one question pydantic doesn't answer is whether an output of one type may feed an input of another, so that is conductor's own walk over the two types; answering it from pydantic's JSON schema was tried and accepted edges whose values were then refused.
- **`Series[T]` and `Table[R]` are ordinary generic classes** with their own pydantic schema. So a table can be a member of a union, an item of a `Series` and a field of a host's model, and it has one JSON form wherever a host sees one, `{"columns": [{"name", "type", "values"}]}`. A value read as `Table[R]` has exactly `R`'s columns wherever it comes from: a caller, a person's answer, collected records, a node's own return; a missing column is refused and others are cut. A bare `Table` keeps every column. To a type checker a `Series[T]` is a `list[T]` and a `Table[R]` a frame.
- **A node may be written for any type.** `def split[T](value: Table[T] | list[T]) -> Series[T]` takes the type that arrives on its edge, decided when the graph is compiled; after that the node has ordinary inputs and outputs. The nodes `series-split`, `series-stack` and `control-decide` of `syv-conductor-nodes` are written this way and have no hooks. A hook stays for what a type can't say: inputs or outputs named by a constant.
- **One flat record per input or output.** A `FieldDescription` always has a `type` and a `schema`, `is_series` says whether it is a `Series`, and a table's columns are in its schema. Where it is used it is nested, not copied: an input of a node in `graph.describe()` is `{field, source}`, a graph input `{address, node_id, field}`.
- **`Policy` is a strict pydantic model.** `Policy(retries="2")` raises pydantic's own error where it is written.
- **What a person sees for a type is declared.** Conductor's own types take their id and title from one fixed table; a host's type shown to people is declared with `@host_type("id", title="Title")` above the class, checked when the class statement ends. Nothing falls back to a class's name. A type's id shows its generic arguments (`list[address]`), so a mismatch can say them, and a class inside a generic declares its id and title too. A host shows its own words for type ids in its editor. Every Python block in the docs is type-checked under mypy and pyright.

```python
from typing import Annotated

from pydantic import BaseModel

from conductor import Param


class Settings(BaseModel):
    temperature: Annotated[float, Param(ge=0, le=2)] = 1.0


print(Settings.model_json_schema()["properties"]["temperature"])  # {'default': 1.0, 'maximum': 2, 'minimum': 0, 'title': 'Temperature', 'type': 'number'}
```

**Worth questioning:** `@host_type` is a decorator, not a base class. A base class runs before the class is finished, which breaks pydantic's `Box[int]`, `@dataclass(slots=True)` and enums, and a library's class can't be given one. The decorator gives a subclass of a builtin (`class Email(str)`) the schema it lacks, on the class, so conductor still knows no host type.

**Worth questioning:** `@host_type(..., accepts=[...])` takes a list of classes whose values may be wired to the type, so a rule that isn't a list of classes can't be said. A list is data an editor can show before anything runs.

**Worth questioning:** a type checker accepts a `Table[Sale]` where a `Table[Order]` is expected. A distinct generic would refuse it, but it would make every plain `polars.DataFrame` return a type error.

## The gate: where values are checked

A value from outside is checked once, where it comes in, and never again. Each such place is a **door**: a graph record, a constant or default in it, a default a node declares, what a hook returns, a caller's graph inputs and limits, the `context=` a host passes, a person's answer, what a node returns or raises, a value an edge gives an input whose type asks more than the edge's source promises, and a saved run. A door is built once for what it reads (a graph's inputs, a node's outputs) and holds one pydantic validator for it. Called with the outside value, it returns the value as the engine keeps it, or a problem with the door's code (`invalid_input`, `invalid_output`, `invalid_answer`, ...). Inside the wall nothing is checked again: a node gets its arguments as the doors read them, and the engine trusts them.

- **The doors are one package**, `_internal/gate/`, a module per door, and its `__init__.py` lists them all. No module outside it builds a validator of a type. The saved-run door stays beside the saved run's writer, since its reader and writer are one schema.
- **A type's own rules are its schema; the rules for every type are the gate's.** What a `Series` is, a table's columns, `Param`'s limits and choices are declared with the type. That a yes or no is not a number holds for every type, so the gate puts it into each type's schema when it builds the type's validator, once per type: `True` is refused for an `int` input, an item of a `list[int]`, a field of a `TypedDict` and a cell of a table alike (`bool_is_not_a_number`).
- **A door costs a run nothing it can avoid.** A value already exactly of a plain class (`"b"` for `str`) skips pydantic, and an edge whose source's type already fits its input has no check at all.

```python
from typing import Annotated

from conductor import Param


@registry.node("times", title="Times", description="Repeats a mark a number of times.")
def times(n: Annotated[int, Param(ge=1)]) -> str:
    return "x" * n  # n is an int of at least 1 here: the inputs door read it, and nothing checks it again


counted = Graph.compile(make_graph(times), registry)
print(counted.run(n="3").outputs)  # {'times.result': 'xxx'}
try:
    counted.run(n=True)
except InvalidCallError as refused:
    print([found.type for found in refused.errors()])  # ['invalid_input']: a yes or no is not a number
```

**Worth questioning:** a door reports the first fault of a node's return, but every fault of a caller's inputs. A return is one call's outcome, one problem; a caller fixes all their inputs at once.

## The companion packages

- **`conductor-nodes`** (on PyPI `syv-conductor-nodes`)**:** the 34 standard nodes, a registry per module, and `conductor_nodes.registry` holding them all. It has no types of its own. A failure a person can act on is a `NodeError` with a code. A graph record names a node by its id, so an id changes only in a major release, with the old and the new id listed in the changelog.
- **`conductor-providers`** (on PyPI `syv-conductor-providers`)**:** `run_router(store=, registry=, caller=, may_see=)`, a FastAPI router mounted with `app.include_router(...)`. Starting, reading, following, answering, resuming, running again and cancelling are routes, by run id, served from any worker.
- **The host owns storage and login.** `RunStore` is six async methods over one row per run, and every change is a compare-and-set on a revision. A lease lets another worker resume a dead worker's run from its last checkpoint. `caller` is a FastAPI dependency whose return type is kept on the run as JSON, so any worker can rebuild the registry for that caller. Who may see a run is the host's `may_see(caller, started_by)`, required whenever there is a `caller`; `again` by another caller runs the graph on that caller's own nodes.
- **What the router promises.** An answer sent while a run is running is queued, and the write that records how it stopped either takes it or refuses it in the run's `refused`, never drops it; a second answer to a question already queued is 409. The item limit a run was started or resumed with applies to the run from then on. A cancel stays set through a takeover, so only an explicit resume restarts the run. One call of `resume()` may take 5 minutes by default (`resume_timeout`), and a timed-out run waits for a resume.
- **Proved by two real workers.** `tests/test_providers/test_serving_runs.py` runs the SQLite store from [Serving runs over HTTP](serving-runs.md) in two uvicorn processes. A killed worker's run is taken over and repeats at most 6 of 40 calls; a deploy repeats none.
- **Left out:** React and Mermaid adapters (nothing reads them), an `/execute` endpoint (a run lives in a store, by id), and a store inside the package.

**Worth questioning:** shipping no store means every host writes one. Storage is what every host already has, and the recipe's plain SQL ports to Postgres line by line.

## Where to start reading

Start at `packages/conductor/src/conductor/__init__.py`, then read `_internal/` from `base`, `types` and `field` up through `record`, `compiled` (`input_plans`, then `flat_nodes`), `run` (`outcome`, `json`, `scheduler`, `arguments`, then `type`) and `graph`. The providers' `router.py` and `store.py` come after, and `conductor-nodes` last.
