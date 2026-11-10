# Conductor 2.0's design: a reviewer's guide

This page is a reviewer's guide to 2.0's design. It goes topic by topic: what each choice is, why it was made, and what it was chosen over. **Worth questioning** marks a choice a reviewer could fairly push back on, with the reason it holds. The code blocks run in order as one module, and the comment after each `print` is what it prints.

## What conductor is

- **A general library for running graphs of typed Python functions.** Someone writes the functions, someone else draws the graph in an editor, and the graph reaches the host as JSON. The core has no web framework, database or login. Its source uses no host's words, and `tests/test_core/test_surface.py` checks that. Fitting it to one platform would narrow the library and give the code nothing.
- **Two surfaces, two bars.** The node author's surface (`@node`, `Param`, `Series`, `Table`, `SKIPPED`, `Ask`, the errors) is general and stable, because other packages are written against it. The host's surface (load, run, answer, save, events) holds only what a real host calls. Five modules are public: `conductor`, `conductor.widgets`, `conductor.events`, `conductor.testing` and `conductor.about`, the reference. Anything in `conductor._internal` may move.

## The front door and the registry

- **A graph is data.** There is no `@graph` decorator and no placing nodes in Python. A graph built in code is the same record an editor saves. Most graphs are drawn, and two ways to define one would mean two sets of rules.
- **`@node` wraps a function and registers nothing.** It returns a frozen, callable `NodeDefinition`, so calling it like the function is its unit test. `@registry.node(...)` takes the same arguments, typed as one signature, and adds the node.
- **`NodeRegistry` works like a FastAPI router.** `add_subregistry` copies another registry's nodes in, so a package keeps one registry per module. One merge rule applies wherever a node enters: the registry's tags come first, then the node's; the node's own `category` wins; `extra` merges key by key. Replacing `extra` whole would drop a registry's filter key from any node with an `extra` of its own. Live, linked subregistries were not built, because nothing needs them.
- **One way each.** The decorator is for your own nodes, `add(*definitions)` for other people's, and the constructor takes no node list. `add` raises on a repeated id; `replace` and `remove` are explicit.
- **`registry.filter(predicate)` returns a registry of its own.** The predicate gets the whole definition. `extra` is the host's JSON: checked when it is declared, shown by `describe()`, never read by conductor.

```python
from conductor import InvalidCallError, NodeRegistry, Series, make_graph

text = NodeRegistry(category="Text", extra={"tier": "free"})


@text.node("upper", tags=("case",))
def upper(text: str) -> str:
    return text.upper()


@text.node("join")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


registry = NodeRegistry(tags=("core",))
registry.add_subregistry(text)
print(registry["upper"].tags)  # ('core', 'case')
free = registry.filter(lambda d: d.extra.get("tier") == "free")
print(sorted(d.id for d in free))  # ['join', 'upper']
```

**Worth questioning:** a node that registers nothing needs a registry before anyone can find it. This holds because one definition can then go into any registry, and what a host offers is written in one place: where it builds the registry.

## The stored record and `make_graph`

- **A record is `{"nodes": [...], "edges": [[from, to], ...]}`**, nothing more: what a host keeps about the whole graph, such as a canvas viewport, it stores beside the record. A node is `{"id", "node", "version", "values", "defaults", "display"}`, with empty fields left out. `id` is where the node sits in this graph, and `node` is which definition it uses. One combined string would need parsing everywhere. In conductor's JSON, `node` always means the coded node and `node_id` a placed one.
- **`version` defaults to 1, and loading never upgrades.** `registry.upgrade` does, when asked.
- **`values` are fixed; `defaults` are prefilled and stay open.** There is one map per intent, chosen over a `locked` flag beside one map. A default is allowed only where a caller can reach it: on an open input of a first node. Anywhere else it is `default_on_later_node`.
- **Typed-in values and a node's `display` are JSON.** A value typed in is kept as its JSON reads back: a date as its ISO text, a float that isn't finite as `"NaN"` or `"Infinity"`. So a record always equals its own JSON, and its fingerprint survives a store. A value with no JSON form, such as a polars frame, is `invalid_value` at the call. A hand-built record holding a float that isn't finite is `invalid_record`.
- **A node's `display` is the host's.** Conductor stores it, never reads it, and leaves it out of the fingerprint. It sits on the node so it follows the node through an upgrade, a removal and an embedding.
- **There is no setting flag on a field.** A node can't know every graph it sits in, so whether an input is fixed is said per graph: a `value` in the record is the graph author's, an open first-node input the caller's.
- **Edges are a sorted list of address pairs** (`node.field`, or `node.output.column`). Two records that differ only in edge order are equal. Several edges into one input gather in a fixed order that never depends on the list: on rows of one table, that table's row order; across tables, the order of `nodes`, then the order the source declares its outputs. So a host may sort the list. Edges stored per node were rejected: reconnecting reorders either shape.
- **`make_graph(*nodes, edges=, values=, defaults=)`** follows scikit-learn's `make_pipeline`. With no `edges` the nodes form a chain: each node takes its one required open input from the one output of the node before it, or the call fails with `not_a_chain`. With `edges`, those are every edge. It loads what it writes, each node against the definition it was given, and raises one `InvalidCallError` listing every fault of the call, its own and loading's together. A half-drawn graph is still written.

```python
record = make_graph(upper, join)
print(record.model_dump_json())  # {"nodes":[{"id":"upper","node":"upper"},{"id":"join","node":"join"}],"edges":[["upper.result","join.texts"]]}


@registry.node("half")
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
- **`again` reuses by content.** A unit takes an earlier result when it receives exactly what an earlier unit of its node received, under the same version, code, fields and rows. Identical rows match first to first, second to second, so a row inserted above keeps everyone's answer. `run.forget(*ids)` calls chosen nodes again, for a host that must recheck access. `edited.with_inputs(..., earlier=old)` lends work across an edit to the nodes the edit left alone.
- **Errors are data.** One `Problem` record is used everywhere, with stable codes listed in `PROBLEM_CODES` for a host to translate by. A message says how to fix the fault and never quotes a value. A node's own exception text goes to the log. A caller's input named twice, by name and by address, is `input_given_twice` from every verb that takes inputs, `calls_for` included.

```python
graph = registry.load(record)
print(graph.run(text=["a", "b"]).outputs)  # {'join.result': 'A+B'}
```

**Worth questioning:** the code in the reuse key is the node function's own bytecode and constants, so a changed helper isn't seen. A node that depends on more than its inputs sets `Policy(reuse_when_inputs_match=False)`. Hashing all reachable code would be slow and would still miss the outside world.

**Worth questioning:** carried work that a finished node didn't take is dropped at the end of the leg, so an edit undone two runs later calls that row again. Without the drop, a save grows with every edit. A node with a row still waiting keeps its carried work, so an answer survives an edit and its revert.

## Rows and one-or-many

- **The unit of work is a node on a row.** A `Series` or `Table` output makes a table under its node's row, so rows form a tree. A one-value input fed a column runs once per row; a `Series` or `Table` input gathers the rows under it.
- **The value decides one or many.** A list given to a one-value input whose type takes no list is a row per item, a list of one included. A type that takes a list takes it whole. The node never knows. A `many` mark on inputs and a per-node switch were rejected, because the types already say it: `contract: File` runs per row, `contracts: Series[File]` runs once.
- **Two lists never meet uncollected.** Ten files are ten rows. Two given lists reaching one node are `lists_not_collected`; the fix is a node that collects one of them as a `Series`. Crossing the lists, pairing them by position and a per-node setting were all rejected. Crossing multiplies paid calls silently, and partitioning belongs in nodes the author draws.
- **Tables two nodes made never meet by themselves.** The author joins them on a key (`rows_not_related`). There is no hidden row id to merge on: merging on one caused every silent row trap.
- **Count before spending.** `graph.calls_for(...)` gives call counts for an editor to show before a run. `max_rows=` is an optional ceiling, checked as rows are made, before any call on them.

```python
print(graph.calls_for(text=11))  # {'upper': 11, 'join': 1}


@registry.node("greet")
def greet(name: str, greeting: str) -> str:
    return f"{greeting} {name}"


try:
    registry.load(make_graph(greet)).run(name=["Ida", "Bo"], greeting=["Hi", "Hello"])
except InvalidCallError as e:
    print([p.type for p in e.errors()])  # ['lists_not_collected']
```

**Worth questioning:** a list sent to a `str` input runs N times instead of failing. This holds because the type settles every case where a list could mean anything else.

## Blanks and copy-down

- **A blank skips that node on that row.** If an input's type lacks `| None`, the node isn't called there, its outputs are blank, and `run.not_called` records the node, row and input. Pydantic refuses `None` for a `str`, and failing the whole run after other rows' side effects went out is worse.
- **`Series[X]` drops blanks, `Series[X | None]` keeps them, and `Series[X] | None` is a list that may itself be blank.** Each spelling says one thing. `Table[R] | None` reads the same way: a table never made reaches the input as `None`.
- **`SKIPPED` is a branch not taken**, not a blank. It leaves a gap in the column, and above a node's rows it skips the node. A skip wins over a blank: a node with one input skipped and another blank is skipped, whichever comes first in its signature.
- **A value from a row above is copied down.** Letting it add rows of its own was rejected: wiring in a document's name added a row, and a document showed blank after a per-document filter.

```python
middle = {"Ida": "Marie"}


@registry.node("middle-name")
def middle_name(name: str) -> str | None:
    return middle.get(name)


@registry.node("send")
def send(middle: str) -> str:
    return f"sent {middle}"


sent = registry.load(make_graph(middle_name, send)).run(name=["Ida", "Bo"])
print(sent.status, sent.not_called)  # complete [{'node_id': 'send', 'row': (1,), 'input': 'middle'}]
```

**Worth questioning:** the run ends `complete` with a row not sent. This holds because the run records which row and which input, so a host can say "Bo's row was not sent: middle name blank" instead of failing everyone.

## `Ask`

- **`raise Ask(prompt, choices=, default=)` pauses one row.** The return type says what the answer is and how to ask: `bool` is yes or no, a `Literal` one of its values, a frozen dataclass a form. `AskBool`, `AskMulti` and a `type=` argument were rejected as a second declaration. That row and whatever reads it wait; the rest runs on.
- **`Ask` holds only what the run knows, typed.** How a question looks is its type's `Param` (a widget, a title, choices known when the node is written), so a hook's output types carry it too; `widget=` on `Ask` was a second home for it and went. `-> str` with a bare `Ask(prompt)` is free text. Each argument has the answer's shape: a value for one output, a mapping by output name, naming only some, for several. There is no plural `defaults=`.
- **A question is the same record as a field:** the `FieldDescription` a palette reads, with the `choices` and `default` the `Ask` set. A host draws it the way it draws an input. One JSON Schema for the whole answer was rejected: it loses a host type's id, a field's shape and exclusive groups, and has no place for a widget.
- **`run.answer(unit, value)` or `run.answer({id: value, ...})`.** Every fault is refused at once and nothing is half-answered. Each answer is read strictly from its JSON form (`true`, never `"yes"`). A declared choice is read exactly the same way, held to the field's limits, so an enum's value or a date's ISO text is a choice.
- **A waiting id names the place and the question**, `node@row~fingerprint`. An answer meant for a question that has moved is `answer_stale`.

```python
from typing import Literal

from conductor import Ask


@registry.node("tone")
def tone(text: str) -> Literal["formal", "casual"]:
    raise Ask(f"Which tone for {text!r}?")


asked = registry.load(make_graph(tone)).run(text="hello")
print(asked.answer(asked.waiting[0], "casual").resume().outputs)  # {'tone.result': 'casual'}
```

**Worth questioning:** `Ask` derives from `BaseException`. Otherwise a node's `except Exception` would swallow the pause and carry on with a wrong value.

**Worth questioning:** an id without its `~` part still answers what waits there. A host stores the full id, which the guard covers. The bare id is what a person types about the run in front of them.

## The engine

- **A small core.** Rows are tuples, there is one ledger of units, embedded graphs are spliced in before compiling, and tables are kept as columns. The parts pay off together: the core's speed comes from calling inline and from checking values only where a check can fail.
- **Nodes run on the leg's own thread by default** (`Policy(concurrency=1)`). An author asks for threads with `concurrency` or `attempt_timeout`. Threads by default would surprise code that isn't thread-safe, and calling inline is most of the core's speed.
- **Values are checked where it matters.** A node's return is read strictly against its type (`invalid_output`). An edge's value is read again only where its source's type doesn't settle it, decided at load, so a plain chain checks nothing at run time.
- **Only `ExternalFailure`, or a class the node's `retry_on` names, is retried.** A `NodeError` of the node's own, an output its type refused and an `Ask` are never retried, whatever `retry_on` says: repeating the call repeats them. After a failure, calls already on threads finish and are kept, so resuming repeats none.
- **A stored graph is a node:** `node("id")(record)`, taking no policy or hook, as its own nodes run by theirs. From outside it is one node with the inputs and outputs it offers, which are its graph's own loaded alone, as its hooks made them, running on each row its node runs on. Inner ids show only inside `embedded_graph_broken`.

```python
from conductor import node

shout = node("shout")(make_graph(upper))
registry.add(shout)
shouted = registry.load(make_graph(shout, join, edges=[("shout.result", "join.texts")]))
print(list(shouted.inputs), shouted.run(text=["a"]).outputs)  # ['text'] {'join.result': 'A'}
```

**Worth questioning:** a host whose slow nodes forget `concurrency` runs them in series. This holds because the slow node knows it's slow, and saying so is one argument.

## Checkpoints and serving

- **A run is one JSON value.** `graph.load_run(saved)` checks it against the graph and refuses another graph's save by fingerprint (`state_of_another_graph`). Saves survive a store that reorders keys, such as Postgres `jsonb`, and hold only the nodes with work done, so a long graph waiting at its start saves small. A save's shape is declared once, as pydantic models that `to_json` writes through and `load_run` reads through, and a save has one reader, its JSON text (a mapping is written as JSON first), so a float that isn't finite is put back before its type reads it, and a damaged save is one `state_unreadable` whose `ctx.at` is the dotted place of the fault.
- **Checkpoints come from the stream.** `run.stream(checkpoint_every=seconds)` sends `run_checkpoint` events carrying a copy of the run. Resuming one calls only what it holds no record of, so a process that dies mid-leg repeats at most the calls that were in flight. A callback argument was rejected as a second channel beside the stream.
- **Events carry counts, not values.** `node_complete` says how many rows a node did, and the values stay in the run, so an event is small whatever the size of the table. The one exception is a `Problem`: `node_failed` carries one, and `run_end` carries the leg's `cause`, so a run stopped at its ceiling says how many rows it reached and a client can choose a higher one.

```python
import asyncio


async def follow() -> None:
    async for event in graph.with_inputs(text=["a", "b", "c"], max_rows=2).stream(checkpoint_every=1.0):
        if event.type == "run_checkpoint":
            pass  # store event.run.to_json()
        if event.type == "run_end":
            print(event.status, event.cause and event.cause.type)  # stopped row_ceiling


asyncio.run(follow())
```

## The pydantic seam

- **Every record is a frozen pydantic model:** `GraphRecord`, `Problem`, the events (one union on `type`), `Waiting`, and every `describe()` result. Each serves as a FastAPI response model as it is.
- **`Param` is pydantic metadata.** It implements pydantic's schema hooks, so its limits and its choices hold inside a host's own `BaseModel` as on a node's field: the choice rule has one home, the hook. A field's limits live in its JSON schema, not in a separate list.
- **Strict where a program produced the value, lax where a person or caller typed it.** Answers, declared choices and node outputs are read strictly (`"12"` is never an `int`). Typed-in values, inputs and edges are read laxly. A `bool` never lands on a number, nor on a union a number is in and a `bool` isn't (`int | float`), as a value or in a table's column.
- **A type's id is its own.** A builtin is known by its name (`bool`, `str`); a type that needs an id and a title of its own is a class (`id`, `display_title`). A host shows its own words for type ids in its editor. `Series[T]` and `FromRun[T]` are `Annotated` aliases, and every Python block in the docs is type-checked under mypy and pyright.

```python
from typing import Annotated

from pydantic import BaseModel

from conductor import Param


class Settings(BaseModel):
    temperature: Annotated[float, Param(ge=0, le=2)] = 1.0


print(Settings.model_json_schema()["properties"]["temperature"])  # {'default': 1.0, 'maximum': 2, 'minimum': 0, 'title': 'Temperature', 'type': 'number'}
```

**Worth questioning:** a type checker accepts a `Table[Sale]` where a `Table[Order]` is expected. A distinct generic would refuse it, but it would make every plain `polars.DataFrame` return a type error.

## The companion packages

- **`conductor-nodes`:** the 29 standard nodes, a registry per module, and `conductor_nodes.registry` holding them all. It has no types of its own. A failure a person can act on is a `NodeError` with a code. Node ids never change, since stored graphs name them.
- **`conductor-providers`:** `run_router(store=, registry=, caller=)`, a FastAPI router mounted with `app.include_router(...)`. Starting, reading, following, answering, resuming, running again and cancelling are routes, by run id, served from any worker.
- **The host owns storage and login.** `RunStore` is six async methods over one row per run, and every change is a compare-and-set on a revision. A lease lets another worker resume a dead worker's run from its last checkpoint. `caller` is a FastAPI dependency whose return type is kept on the run as JSON, so any worker can rebuild the registry for that caller. A run is visible only to whoever started it, unless the host widens `may_see`; `again` by another caller loads the run on that caller's own nodes.
- **What the router promises.** An answer sent while a leg runs is queued, and the write that ends the leg either takes it or refuses it in the run's `refused`, never drops it; only a leg that ends `waiting` takes answers; a second answer to a row already queued is 409. The ceiling a run was started or resumed with holds for every later leg. A cancel stays set through a takeover, so only an explicit resume restarts the run. One leg may run 5 minutes by default (`leg_timeout`), and a timed-out run waits for a resume.
- **Proved by two real workers.** `tests/test_providers/test_serving_runs.py` runs the SQLite store from [Serving runs over HTTP](serving-runs.md) in two uvicorn processes. A killed worker's run is taken over and repeats at most 6 of 40 calls; a deploy repeats none.
- **Left out:** React and Mermaid adapters (nothing reads them), an `/execute` endpoint (a run lives in a store, by id), and a store inside the package.

**Worth questioning:** shipping no store means every host writes one. Storage is what every host already has, and the recipe's plain SQL ports to Postgres line by line.

## Where to start reading

Start at `packages/conductor/src/conductor/__init__.py`, then read `_internal/` from `fields` up through `record`, `graph`, `run` and `leg`. The providers' `router.py` and `store.py` come after, and `conductor-nodes` last.
