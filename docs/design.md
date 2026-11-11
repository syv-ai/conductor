# Conductor 2.0's design: a reviewer's guide

This page is a reviewer's guide to 2.0's design. It goes topic by topic: what each choice is, why it was made, and what it was chosen over. **Worth questioning** marks a choice a reviewer could fairly push back on, with the reason it holds. The code blocks run in order as one module, and the comment after each `print` is what it prints.

## What conductor is

- **A general library for running graphs of typed Python functions.** Someone writes the functions, someone else draws the graph in an editor, and the graph reaches the host as JSON. The core has no web framework, database or login. Its source uses no host's words, and `tests/test_core/test_surface.py` checks that. Fitting it to one platform would narrow the library and give the code nothing.
- **Two surfaces, two bars.** The node author's surface (`@node`, `Param`, `Series`, `Table`, `SKIPPED`, `Ask`, the errors) is general and stable, because other packages are written against it. The host's surface (load, run, answer, save, events) holds only what a real host calls. Five modules are public: `conductor`, `conductor.widgets`, `conductor.events`, `conductor.testing` and `conductor.about`, the reference. Anything in `conductor._internal` may move.

## The front door and the registry

- **A graph is data.** There is no `@graph` decorator and no placing nodes in Python. A graph built in code is the same record an editor saves. Most graphs are drawn, and two ways to define one would mean two sets of rules.
- **`@node` wraps a function and registers nothing.** It returns a frozen, callable `FunctionDefinition`, so calling it like the function is its unit test; `stored_graph(...)(record)` returns a `StoredGraphDefinition`, which has no function to call. `NodeDefinition` is either one. `@registry.node(...)` and `registry.stored_graph(...)` take the same arguments as the two, and add the node.
- **`NodeRegistry` works like a FastAPI router.** `add_subregistry` copies another registry's nodes in, so a package keeps one registry per module. One merge rule applies wherever a node enters: the registry's tags come first, then the node's; the node's own `category` wins; `extra` merges key by key. Replacing `extra` whole would drop a registry's filter key from any node with an `extra` of its own. Live, linked subregistries were not built, because nothing needs them.
- **One way each.** The decorator is for your own nodes, `add(*definitions)` for other people's, and the constructor takes no node list. `add` raises on a repeated id; `replace` and `remove` are explicit.
- **`registry.filter(predicate)` returns a registry of its own.** The predicate gets the whole definition. `extra` is the host's JSON, typed as JSON: shown by `describe()`, never read by conductor.

```python
from conductor import InvalidCallError, NodeRegistry, Series, make_graph

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

## The stored record and `make_graph`

- **A record is `{"nodes": [...], "edges": [[from, to], ...]}`**, nothing more: what a host keeps about the whole graph, such as a canvas viewport, it stores beside the record. A node is `{"id", "node", "version", "values", "defaults", "display"}`, with empty fields left out. `id` is where the node sits in this graph, and `node` is which definition it uses. One combined string would need parsing everywhere. In conductor's JSON, `node` always means the coded node and `node_id` a placed one.
- **`version` defaults to 1, and loading never upgrades.** `registry.upgrade` does, when asked.
- **`values` are fixed; `defaults` are prefilled and stay open.** There is one map per intent, chosen over a `locked` flag beside one map. A default is allowed only where a caller can reach it: on an open input of a first node. Anywhere else it is `default_on_later_node`.
- **Typed-in values and a node's `display` are JSON.** A value typed in is kept as its JSON reads back: a date as its ISO text, a float that isn't finite as `"NaN"` or `"Infinity"`. So a record always equals its own JSON, and its fingerprint survives a store. A value with no JSON form, such as a polars frame, is `invalid_value` at the call. A hand-built record holding a float that isn't finite is `invalid_record`.
- **A node's `display` is the host's.** Conductor stores it, never reads it, and leaves it out of the fingerprint. It sits on the node so it follows the node through an upgrade, a removal and an embedding.
- **There is no setting flag on a field.** A node can't know every graph it sits in, so whether an input is fixed is said per graph: a `value` in the record is the graph author's, an open first-node input the caller's.
- **Edges are a sorted list of address pairs** (`node.field`, or `node.output.field` for one field of a record). Two records that differ only in edge order are equal, and nothing a run does depends on the list's order: several edges into one input are alternatives, opposite branches of one decision, never a list to join. So a host may sort the list. Edges stored per node were rejected: reconnecting reorders either shape.
- **`make_graph(*nodes, edges=, values=, defaults=)`** follows scikit-learn's `make_pipeline`. With no `edges` the nodes form a chain: each node takes its one required open input from the one output of the node before it, or the call fails with `not_a_chain`. With `edges`, those are every edge. It writes a record and loads nothing. It raises one `InvalidCallError` listing what it can't write (a value for a node it wasn't given, a value with no JSON form, an edge that is no address, a chain it can't draw); everything else, a mistyped input or an edge between types that don't meet included, is in `registry.load(record).problems`, so loading is the one place a record's faults are found. A script that only saves the record hears nothing until it is loaded.

```python
record = make_graph(words, upper, join)
print(record.model_dump_json())  # {"nodes":[{"id":"words","node":"words"},{"id":"upper","node":"upper"},{"id":"join","node":"join"}],"edges":[["upper.result","join.texts"],["words.result","upper.text"]]}


@registry.node("half", title="Half", description="Halves a number.")
def half(n: float) -> float:
    return n / 2


typed = make_graph(half, values={"half.n": float("nan")})
print(typed.nodes[0].values)  # {'n': 'NaN'}
```

**Worth questioning:** the chain never matches by name. Matching by name silently left a same-named column unused. A chain that refuses to guess can't do that.

**Worth questioning:** a repeated node is `upper-1`, `upper-2`, not `upper`, `upper-2`. The second form silently moves `upper` to whichever node is inserted before it.

## `Graph` and `Run`

- **`registry.load(record)` never raises for a fault in the graph.** Each fault is a `Problem` (`type`, `loc`, `msg`, `ctx`) an editor can mark on the node. Only input that isn't a record at all raises (`invalid_record`). The shape is the caller's to get right; what the record says is the graph's.
- **Nobody declares a graph's inputs or outputs.** Inputs are the open inputs of its first nodes; outputs are the outputs no node reads. An open input on a first node is not a problem, just as an unfitted scikit-learn pipeline isn't broken.
- **A loaded graph keeps its own copy of its definitions**, so a later change to the registry can't break it.
- **One way in, and verbs that mirror each other.** `Graph(...)` and `Run(...)` refuse to be constructed. `graph.with_inputs` checks inputs and gives a run that hasn't started; `graph.run` is that plus `resume()`. On a run, `with_inputs` and `again` do the same with changed inputs, keeping the earlier work. `answer` returns another run and runs nothing.
- **`again` reuses by content.** A call takes an earlier result when it would receive exactly what an earlier call of its node received, under the same version, code and fields. Identical items share one earlier result. A person's answers are taken in item order, first to first and second to second, so an item inserted above keeps everyone's answer. `run.forget(*ids)` calls chosen nodes again, for a host that must recheck access. `edited.with_inputs(..., earlier=old)` lends work across an edit to the nodes the edit left alone.
- **Errors are data.** One `Problem` record is used everywhere, with stable codes listed in `PROBLEM_CODES` for a host to translate by. A message says how to fix the fault and never quotes a value. A node's own exception text goes to the log. A caller's input named twice, by name and by address, is `input_given_twice` from every verb that takes inputs.

```python
graph = registry.load(record)
print(graph.run(text="a b").outputs)  # {'join.result': 'A+B'}
```

**Worth questioning:** the code in the reuse key is the node function's own bytecode and constants, so a changed helper isn't seen. A node that depends on more than its inputs sets `Policy(reuse_when_inputs_match=False)`. Hashing all reachable code would be slow and would still miss the outside world.

**Worth questioning:** carried work that a finished node didn't take is dropped at the end of the leg, so an edit undone two runs later calls the node for that item again. Without the drop, a save grows with every edit. A node with a question still waiting keeps its carried work, so an answer survives an edit and its revert.

## One or many

- **The types decide how often a node runs.** A `Series[T]` output read by an input of one `T` runs the reader once per item, and everything that reads the reader too. A `Series` input gathers every item and runs once; a `Table` input gathers records the same way. A value made outside the items is there on every call. The node never knows. A `many` mark on inputs and a per-node switch were rejected, because the types already say it: `contract: File` runs per item, `contracts: Series[File]` runs once.
- **The type decides, never the value.** A list given to an input of one value is refused. A node that brings many says so in its return type, so one file and ten files take the same path. Letting a caller's list make the items was rejected: how often a node runs would depend on what a run was given, and a graph would need loading again for each run.
- **A table is one value.** It never makes items by itself, and a column of it is no address. A Split node takes it apart, drawn where the author wants it. Tables that made items by themselves were rejected: how often a node ran then followed from which columns happened to be wired to it, which an author can't see in the drawing.
- **Items of two Series never pair up.** Ten documents and three questions are refused as `series_not_nested` until one is made inside the other, or one is given whole. Crossing them and pairing them by position were both rejected. Crossing multiplies paid calls silently, and which item goes with which is the author's to draw.
- **A gather goes up one level, and has three rules.** It gets the items that have a value; one never made is treated as for any node; it waits for every item. No items is an empty `Series`, and the node is called: "0 approved" is an answer.
- **A ceiling before spending.** `max_items=` is an optional ceiling on any one `Series` a node runs per item of, checked when the `Series` is made, before any call on its items. Over it the run is `stopped`, and `run.with_max_items(100).resume()` goes on under a higher one, calling nothing again.

```python
print({node_id: runs.msg for node_id, runs in graph.plan.items()})  # {'words': 'once', 'upper': 'per item of words.result', 'join': 'once'}


@registry.node("greet", title="Greet", description="Greets a person by name.")
def greet(name: str, greeting: str) -> str:
    return f"{greeting} {name}"


names_and_greetings = make_graph(words, words, greet, edges=[("words-1.result", "greet.name"), ("words-2.result", "greet.greeting")])
print([problem.type for problem in registry.load(names_and_greetings).problems])  # ['series_not_nested']

try:
    graph.run(text=["a", "b"])
except InvalidCallError as e:
    print([p.type for p in e.errors()])  # ['invalid_input']
```

**Worth questioning:** a list sent to a `str` input is refused, where it could have run the node once per item. This holds because a graph then runs the same way whatever it is given, and an editor can show how often each node runs before anyone runs it.

**Worth questioning:** a table needs a Split before a node can run per record. This holds because the Split is where an author sees the run fan out, and a table that is only passed along costs nothing.

## Blanks and outer values

- **A blank keeps the node from being called for that item.** If an input's type lacks `| None`, the node isn't called there, its outputs are blank (those of an `exclusive_group` are a branch not taken, as no branch was chosen), and `run.not_called` records the node, the item and the input. Pydantic refuses `None` for a `str`, and failing the whole run after other items' side effects went out is worse.
- **`Series[X]` drops blanks, `Series[X | None]` keeps them, and `Series[X] | None` is a list that may itself be blank.** Each spelling says one thing. `Table[R] | None` reads the same way: a table never made reaches the input as `None`.
- **`SKIPPED` is a branch not taken**, not a blank. For one item it leaves a gap, and outside the items it skips the node. A skip wins over a blank: a node with one input skipped and another blank is skipped, whichever comes first in its signature.
- **A value made outside the items is there on every call.** Nothing is copied, and it adds no items of its own: wiring in a document's name changes nothing about how often a node runs.

```python
middle = {"Ida": "Marie"}


@registry.node("middle-name", title="Middle name", description="Looks up a person's middle name, if there is one.")
def middle_name(name: str) -> str | None:
    return middle.get(name)


@registry.node("send", title="Send", description="Sends a message to a contact.")
def send(middle: str) -> str:
    return f"sent {middle}"


sent = registry.load(make_graph(words, middle_name, send)).run(text="Ida Bo")
print(sent.status, sent.results["send"]["result"])  # complete ['sent Marie', None]
print([(skipped.loc, skipped.input) for skipped in sent.not_called])  # [(("send", 1), "middle")]
```

**Worth questioning:** the run ends `complete` with one message not sent. This holds because the run records which item and which input, so a host can say "Bo was not sent: middle name blank" instead of failing everyone.

## `Ask`

- **`raise Ask(prompt, choices=, default=)` pauses one call.** The return type says what the answer is and how to ask: `bool` is yes or no, a `Literal` one of its values, a frozen dataclass a form. `AskBool`, `AskMulti` and a `type=` argument were rejected as a second declaration. That call and whatever reads it wait; the rest runs on.
- **`Ask` holds only what the run knows, typed.** How a question looks is its type's `Param` (a widget, a title, choices known when the node is written), so a hook's output types carry it too; `widget=` on `Ask` was a second home for it and went. `-> str` with a bare `Ask(prompt)` is free text. Each argument has the answer's shape: a value for one output, a mapping by output name, naming only some, for several. There is no plural `defaults=`.
- **A question is the same record as a field:** the `FieldDescription` a palette reads, with the `choices` and `default` the `Ask` set. A host draws it the way it draws an input. One JSON Schema for the whole answer was rejected: it loses a host type's id, whether a field is one or many, and exclusive groups, and has no place for a widget.
- **`run.answer(unit, value)` or `run.answer({id: value, ...})`.** Every fault is refused at once and nothing is half-answered. Each answer is read strictly from its JSON form (`true`, never `"yes"`). A declared choice is read exactly the same way, held to the field's limits, so an enum's value or a date's ISO text is a choice. A choice's id is the type's JSON of the value it reads as, so `Decimal` choices show `"1"` and a `datetime` its seconds.
- **A waiting id names the place and the question**, `node@item~fingerprint`. An answer meant for a question that has moved is `answer_stale`.

```python
from typing import Literal

from conductor import Ask


@registry.node("tone", title="Tone", description="Asks a person which tone a text should have.")
def tone(text: str) -> Literal["formal", "casual"]:
    raise Ask(f"Which tone for {text!r}?")


asked = registry.load(make_graph(tone)).run(text="hello")
print(asked.answer(asked.waiting[0], "casual").resume().outputs)  # {'tone.result': 'casual'}
```

**Worth questioning:** `Ask` derives from `BaseException`. Otherwise a node's `except Exception` would swallow the pause and carry on with a wrong value.

**Worth questioning:** an id without its `~` part still answers what waits there. A host stores the full id, which the guard covers. The bare id is what a person types about the run in front of them.

## The engine

- **A small core.** Loading decides once, from the declared types alone, how often each node runs. A run keeps one ledger of calls, and an item is a tuple of numbers. A table stays the frame it is. The parts pay off together: the core's speed comes from calling inline and from checking values only where a check can fail.
- **Nodes run on the leg's own thread by default** (`Policy(concurrency=1)`). An author asks for threads with `concurrency` or `attempt_timeout`. Threads by default would surprise code that isn't thread-safe, and calling inline is most of the core's speed.
- **Values are checked where it matters.** A node's return is read strictly against its type (`invalid_output`). An edge's value is read again only where its source's type doesn't settle it, decided at load, so a plain chain checks nothing at run time.
- **Only `ExternalError`, or a class the node's `retry_on` names, is retried.** A `NodeError` of the node's own, an output its type refused and an `Ask` are never retried, whatever `retry_on` says: repeating the call repeats them. After a failure, calls already on threads finish and are kept, so resuming repeats none.
- **A stored graph is a node:** `stored_graph("id", title=..., description=...)(record)`, taking no policy or hook, as its own nodes run by theirs. From outside it is one node with the inputs and outputs it offers, which are its graph's own loaded alone, as its hooks made them. It loads once however many graphs use it, and nothing of it is copied into the graph around it: a run walks the tree those graphs make. A node inside keeps its calls as any node does, one per item of the stored graph's node and of its own Series, and is named by a path: the node and item of each stored graph around it, then the inner node. So a blank or a skip that arrives reaches the inner nodes that read it, exactly as drawn flat, and the stored graph is never called, blank or skipped itself. A run per item of the stored graph was built first and removed: a 2,000-item save was 34 times a flat graph's.

```python
from conductor import stored_graph

shout = stored_graph("shout", title="Shout", description="Turns a text into capital letters.")(make_graph(upper))
registry.add(shout)
shouted = registry.load(make_graph(shout, join, edges=[("shout.upper.result", "join.texts")]))
print(list(shouted.inputs), shouted.outputs)  # ['text'] ['join.result']
```

**Worth questioning:** a stored graph's node keeps one unit per item, what its outputs came to there, so a stored graph run per item costs about a quarter more than the same nodes drawn flat. Reading its outputs straight from the nodes inside would close that, at the price of two ways for an output to get out.

**Worth questioning:** a host whose slow nodes forget `concurrency` runs them in series. This holds because the slow node knows it's slow, and saying so is one argument.

## Checkpoints and serving

- **A run is one JSON value.** `graph.load_run(saved)` checks it against the graph and refuses another graph's save by fingerprint (`state_of_another_graph`). Saves survive a store that reorders keys, such as Postgres `jsonb`, and hold only the nodes with work done, so a long graph waiting at its start saves small. A save's shape is declared once per graph, as one pydantic schema composed from the schema of each input's and each output's own type the first time a run of the graph is saved or loaded; `to_json` is one write through it and `load_run` one read. Each node's units are columns with a slot per instance of the node, in instance order, so an instance is never written and a unit the run never made can't be. A unit waits by the question its slot holds, so a waiting unit can't be saved without one. A save has one reader, its JSON text (a mapping is written as JSON first), and a damaged save is one `state_unreadable` whose `ctx.at` is the dotted place pydantic names.
- **A `Series` of records keeps the table it was returned as.** A node that returns a table for `Series[R]` (Split does) has made no record yet, and most readers want one field: `series-split.result.email` reads the column's cell. A node that takes the record, or the whole `Series`, receives plain records, so an author never meets a second kind of value. The table is read by its columns where it enters, and a save writes it once, as a table. Making every record up front was the whole cost of Split on a run.
- **A float that isn't finite is its text in a save, and its declared type reads it back.** `float` gives the float; `Any`, a `dict` and `float | str` give the text, as a date under `Any` comes back as its ISO text. An author who wants the float back declares it. Marking each such float's place in the save was built and removed: it was a second account of a value's type, kept beside the type.
- **Checkpoints come from the stream.** `run.stream(checkpoint_every=seconds)` sends `run_checkpoint` events carrying a copy of the run. Resuming one calls only what it holds no record of, so a process that dies mid-leg repeats at most the calls that were in flight. A callback argument was rejected as a second channel beside the stream. A host that shuts down hands its leg back with `run.stream(hand_back=event)`: the leg stops as for a cancel, and its last event is a checkpoint of the run still `running`, with no `run_end`, so the stored run says what its row says and another process goes on from it. Cancelling a run and handing it back are two acts; one event for both would save a run nobody cancelled as `cancelled`.
- **Events carry counts, not values.** `node_complete` says how many calls a node came to, and the values stay in the run, so an event is small whatever the size of the table. The one exception is a `Problem`: `node_failed` carries one, and `run_end` carries the leg's `cause`, so a run stopped at its ceiling says how many items it met and a client can choose a higher one.

```python
import asyncio


async def follow() -> None:
    async for event in graph.with_inputs(text="a b c", max_items=2).stream(checkpoint_every=1.0):
        if event.type == "run_checkpoint":
            pass  # store event.run.to_json()
        if event.type == "run_end":
            print(event.status, event.cause and event.cause.type)  # stopped item_ceiling


asyncio.run(follow())
```

## The pydantic seam

- **Every record is a frozen pydantic model:** `GraphRecord`, `Problem`, the events (one union on `type`), `Waiting`, and every `describe()` result. Each serves as a FastAPI response model as it is.
- **`Param` is pydantic metadata.** It implements pydantic's schema hooks, so its limits and its choices hold inside a host's own `BaseModel` as on a node's field: the choice rule has one home, the hook. A field's limits live in its JSON schema, not in a separate list.
- **Strict where a program produced the value, lax where a person or caller typed it.** Answers, declared choices and node outputs are read strictly (`"12"` is never an `int`). Typed-in values, inputs and edges are read laxly. A `bool` never lands on a number, nor on a union a number is in and a `bool` isn't (`int | float`), as a value or in a table's column.
- **One reading of an annotation.** Each hint is read once into one object that names the type, reads a value, writes it as JSON and says which types may land on it. Validation is pydantic's throughout. The one question pydantic doesn't answer is whether an output of one type may feed an input of another, so that is conductor's own walk over the two types; answering it from pydantic's JSON schema was tried and accepted edges whose values were then refused.
- **`Series[T]` and `Table[R]` are ordinary generic classes** with their own pydantic schema. So a table stands in a union, in a `Series` and inside a host's model, and it has one JSON form wherever a host sees one, `{"columns": [{"name", "type", "values"}]}`. A value read as `Table[R]` holds exactly `R`'s columns at every door: a caller's, a person's answer, gathered records, a node's own return. To a type checker a `Series[T]` is a `list[T]` and a `Table[R]` a frame.
- **A node may be written for any type.** `def split[T](value: Table[T] | list[T]) -> Series[T]` takes the type that arrives on its edge, bound when the graph loads; after that the placed node has ordinary fields. Split, stack and the decision are written this way and have no hooks. A hook stays for what a type can't say: fields named by a setting.
- **One flat record per field.** A `FieldDescription` always has a `type` and a `schema`, `many` says one or many, and a table's columns are in its schema. Where a field stands it is held, not copied: an input of a placed node is `{field, feed}`, a graph's input `{address, node_id, field}`.
- **`Policy` is a strict pydantic model.** `Policy(retries="2")` raises pydantic's own error where it is written.
- **What a person sees for a type is declared.** Conductor's own types take their id and title from one fixed table; a host's type shown to people declares `type_id` and `type_title` as class attributes (`NamedType`). Nothing falls back to a class's name. A type's id shows its generic arguments (`list[address]`), so a mismatch can say them, and a class inside a generic declares its id and title too. A host shows its own words for type ids in its editor. Every Python block in the docs is type-checked under mypy and pyright.

```python
from typing import Annotated

from pydantic import BaseModel

from conductor import Param


class Settings(BaseModel):
    temperature: Annotated[float, Param(ge=0, le=2)] = 1.0


print(Settings.model_json_schema()["properties"]["temperature"])  # {'default': 1.0, 'maximum': 2, 'minimum': 0, 'title': 'Temperature', 'type': 'number'}
```

**Worth questioning:** a host's subclass of a builtin (`class Email(str)`) needs three lines of its own, a `__get_pydantic_core_schema__`, or it is refused as `type_not_carried`. This holds because conductor knows no host type: reading it as a `str` for the host dropped a limit on it and refused `list[Email]`.

**Worth questioning:** `type_accepts` is a list of classes, so a rule that isn't a list of classes can't be said. A list is data an editor can show before anything runs.

**Worth questioning:** a type checker accepts a `Table[Sale]` where a `Table[Order]` is expected. A distinct generic would refuse it, but it would make every plain `polars.DataFrame` return a type error.

## The companion packages

- **`conductor-nodes`:** the 34 standard nodes, a registry per module, and `conductor_nodes.registry` holding them all. It has no types of its own. A failure a person can act on is a `NodeError` with a code. Node ids never change, since stored graphs name them.
- **`conductor-providers`:** `run_router(store=, registry=, caller=, may_see=)`, a FastAPI router mounted with `app.include_router(...)`. Starting, reading, following, answering, resuming, running again and cancelling are routes, by run id, served from any worker.
- **The host owns storage and login.** `RunStore` is six async methods over one row per run, and every change is a compare-and-set on a revision. A lease lets another worker resume a dead worker's run from its last checkpoint. `caller` is a FastAPI dependency whose return type is kept on the run as JSON, so any worker can rebuild the registry for that caller. A run is visible only to whoever started it, unless the host widens `may_see`; `again` by another caller loads the run on that caller's own nodes.
- **What the router promises.** An answer sent while a leg runs is queued, and the write that ends the leg either takes it or refuses it in the run's `refused`, never drops it; only a leg that ends `waiting` takes answers; a second answer to a question already queued is 409. The ceiling a run was started or resumed with holds for every later leg. A cancel stays set through a takeover, so only an explicit resume restarts the run. One leg may run 5 minutes by default (`leg_timeout`), and a timed-out run waits for a resume.
- **Proved by two real workers.** `tests/test_providers/test_serving_runs.py` runs the SQLite store from [Serving runs over HTTP](serving-runs.md) in two uvicorn processes. A killed worker's run is taken over and repeats at most 6 of 40 calls; a deploy repeats none.
- **Left out:** React and Mermaid adapters (nothing reads them), an `/execute` endpoint (a run lives in a store, by id), and a store inside the package.

**Worth questioning:** shipping no store means every host writes one. Storage is what every host already has, and the recipe's plain SQL ports to Postgres line by line.

## Where to start reading

Start at `packages/conductor/src/conductor/__init__.py`, then read `_internal/` from `fields` up through `graph_record`, `checked_graph` (`input_plans`, then `tree`), `units`, `saved_run`, `leg`, `run` (`call_inputs`, then `type`) and `graph`. The providers' `router.py` and `store.py` come after, and `conductor-nodes` last.
