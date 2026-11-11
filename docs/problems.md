# Problem codes

Everything conductor reports has a stable code. A `Problem`'s `type` is one, and so is a `NodeDeclarationError`'s `code`. This page has each code: where you meet it, an example that makes it, and the fix. A host that translates messages keys on the code and fills its sentence from `ctx`. `PROBLEM_CODES` holds every code a `Problem` can have that conductor writes itself; a code not in it is a node's own, from `NodeError(code=)`.

Every example on this page runs after this block. `problems(drawing)` loads a drawing and lists the codes of what is wrong with it; `refused(call)` makes a call and lists the codes it was refused for.

```python
import asyncio
import time
from dataclasses import dataclass
from typing import Annotated, ClassVar

from conductor import (
    Ask, EndedEarly, ExternalError, FromRun, GraphRecord, InvalidCallError, NodeDeclarationError,
    NodeError, NodeRegistry, Param, Policy, Series, Upgrade, make_graph, node,
)
from conductor.testing import check_registry

registry = NodeRegistry()


@registry.node("upper", title="Uppercase", description="Turns a text into capital letters.")
def upper(text: str) -> str:
    return text.upper()


@registry.node("greet", title="Greet", description="Greets a person by name.")
def greet(name: str, greeting: str) -> str:
    return f"{greeting} {name}"


@registry.node("count", title="Count", description="Passes a number on as it is.")
def count(n: int) -> int:
    return n


@registry.node("words", title="Words", description="Gives the words of a text, one at a time.")
def words(text: str) -> Series[str]:
    return text.split()


@registry.node("join", title="Join", description="Joins the pieces into one text.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@dataclass(frozen=True)
class Route:
    short: Annotated[str | None, Param(exclusive_group="length")] = None
    long: Annotated[str | None, Param(exclusive_group="length")] = None


@registry.node("route", title="Route", description="Sends a text down the short or the long path.")
def route(text: str) -> Route:
    return Route(short=text) if len(text) < 4 else Route(long=text)


@registry.node("approve", title="Approve", description="Asks a person to approve an amount in kroner.")
def approve(amount: int) -> bool:
    raise Ask(f"Approve {amount} kr.?")


@dataclass(frozen=True)
class Caller:
    name: str


@registry.node("who", title="Who", description="Tells the name of the person running it.")
def who(caller: FromRun[Caller]) -> str:
    return caller.name


asking_who = registry.load(make_graph(who))


def problems(drawing: dict) -> list[str]:
    """The codes of what loading ``drawing`` finds."""
    return [problem.type for problem in registry.load(drawing).problems]


def refused(call) -> list[str]:
    """The codes a refused call lists."""
    try:
        call()
    except InvalidCallError as error:
        return [problem.type for problem in error.errors()]
    return []
```

## Loading a graph

In `graph.problems`, never raised. `loc` is `(node_id,)` or `(node_id, field)`.

### `alternatives_in_different_series`

> '{field}' takes alternatives that run per item of different Series, {series}, so a call for an item of one has no place for the other's value. Bring them to items of one Series first.

```python
print(problems({
    "nodes": [
        {"id": "w", "node": "words"}, {"id": "r", "node": "route"},
        {"id": "letters", "node": "words"}, {"id": "u", "node": "upper"},
    ],
    "edges": [
        ["w.result", "r.text"], ["r.short", "letters.text"],
        ["letters.result", "u.text"], ["r.long", "u.text"],
    ],
}))  # ['alternatives_in_different_series']
```

`u.text` is fed from both branches of `r`. The long branch gives one value per item of `w.result`. The short branch goes through `letters`, which makes a Series of its own, so it gives one value per item of `letters.result`.

**Fix.** Bring both branches to items of one Series first: gather the inner one in a node with a `Series` input, or feed the node from one of them.

### `compute_hook_failed`

> The node could not work out its fields from what it was given: {reason}. Check what is typed into the node and what is wired to it.

```python
def by_names(names: str) -> dict[str, type]:
    if not names:
        raise ValueError("no names")
    return {name: str for name in names.split(",")}


@registry.node("pick", title="Pick", description="Turns a list of names into one answer per name.", compute_outputs=by_names)
def pick(names: str) -> dict[str, str]:
    return {name: name for name in names.split(",")}


print(problems({"nodes": [{"id": "pick", "node": "pick", "values": {"names": ""}}]}))  # ['compute_hook_failed']
```

**Fix.** Type in a value the node's hook can work its fields out from. The hook's own exception is in the `conductor` log. A hook that raises `NodeError` gives its `code` as `ctx.error` (`failed` when it names none, `external_failed` for an `ExternalError`) and its own `ctx` whole as `ctx.error_ctx`; any other exception gives its class name.

### `cycle`

> The node is part of a cycle, so the graph cannot run; remove one of the edges that close it.

```python
print(problems({
    "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}],
    "edges": [["a.result", "b.text"], ["b.result", "a.text"]],
}))  # ['cycle', 'cycle']
```

**Fix.** Remove one of the edges that close the loop. A graph runs each node after what it reads.

### `default_on_later_node`

> '{field}' is not an open input of the graph, so a caller can't change its default; make it a value.

```python
print(problems({
    "nodes": [{"id": "a", "node": "upper"}, {"id": "g", "node": "greet", "defaults": {"greeting": "hi"}}],
    "edges": [["a.result", "g.name"]],
}))  # ['default_on_later_node']
```

**Fix.** Write it as a value (`values`), which fixes it. Only an open input of a first node is the caller's to fill, so only there does a default mean anything; an input a hook reads is never open.

### `duplicate_node_id`

> Two nodes have the id '{node_id}'; rename one.

```python
print(problems({"nodes": [{"id": "a", "node": "upper"}, {"id": "a", "node": "count"}]}))  # ['duplicate_node_id']
```

**Fix.** Give one of the nodes another id. `make_graph` picks free ids itself (`upper-1`, `upper-2`).

### `edge_names_missing_node`

> The edge from '{source}' to '{target}' names '{node_id}', which is not a node of the graph; add that node or remove the edge.

```python
print(problems({"nodes": [{"id": "a", "node": "upper"}], "edges": [["a.result", "ghost.text"]]}))  # ['edge_names_missing_node']
```

**Fix.** Add the node the edge names, or remove the edge.

### `embeds_itself`

> The node '{node}' is a graph that contains itself; embed a graph that does not contain this one.

```python
inner_record = GraphRecord.model_validate({"nodes": [{"id": "again", "node": "itself"}]})
itself = node("itself", title="Itself", description="A graph that contains itself.")(inner_record)
looping = NodeRegistry()
looping.add(itself)
broken = looping.load({"nodes": [{"id": "i", "node": "itself"}]}).problems[0]
print(broken.type, broken.loc)  # embeds_itself ('i', 'again')
```

**Fix.** A stored graph can't contain itself, directly or through another. Remove the node that leads back. The problem
sits on the node that closes the circle: a problem of a graph used as a node is that graph's own, with the id of the
node that uses it in front of its `loc`.

### `hook_needs_value`

> '{field}' must be typed in: the node works out its fields from it when the graph loads.

```python
print(problems({"nodes": [{"id": "pick", "node": "pick"}]}))  # ['hook_needs_value']
```

**Fix.** Type a value into the input the hook reads. The node's fields depend on it, so it is fixed when the graph loads and never one of the graph's inputs.

### `invalid_node_id`

> '{node_id}' cannot be a node id: '.', '/', '@' and '~' each mean something in an address or a waiting id; rename it without them.

```python
print(problems({"nodes": [{"id": "a.b", "node": "upper"}]}))  # ['invalid_node_id']
```

**Fix.** Pick an id without `.`, `/`, `@` or `~`: those characters build addresses (`a.result`) and waiting ids (`a@1~3fa2`).

### `invalid_value`

> The value typed into '{field}' is not valid: {reason}. Type a value of the field's type.

```python
print(problems({"nodes": [{"id": "c", "node": "count", "values": {"n": "many"}}]}))  # ['invalid_value']
```

**Fix.** Type in a value of the field's type. `reason` says what the type refused, without quoting the value; `errors` is pydantic's list of what it refused (`type`, `loc`, `msg`; `loc` as far as the type names it, an index or a declared field, never a key of a dict), each without the value, so a host can key on `greater_than_equal` or `bool_is_not_a_number`.

### `parameter_name_invalid`

> '{field}' cannot be a field name: '.', '/' and '@' each mean something in an address, and a name can't be empty; rename the field.

```python
def dotted(names: str) -> dict[str, type]:
    return {name: str for name in names.split(",")}


@registry.node("fields", title="Fields", description="Gives one answer per name.", compute_outputs=dotted)
def fields(names: str) -> dict[str, str]:
    return {}


print(problems({"nodes": [{"id": "f", "node": "fields", "values": {"names": "a.b"}}]}))  # ['parameter_name_invalid']
```

**Fix.** Have the hook give names without `.`, `/` or `@`, and never an empty one.

### `repeated_edge`

> The edge from '{source}' into '{field}' is drawn twice; remove one.

```python
print(problems({
    "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}],
    "edges": [["a.result", "b.text"], ["a.result", "b.text"]],
}))  # ['repeated_edge']
```

**Fix.** Remove one of the two.

### `series_inputs_never_share_an_item`

> {inputs} read branches that never both have a value for an item, so no item would reach the node. Write Series[X | None] on them to keep every item, or take the branches through one input.

```python
@registry.node("pair", title="Pair", description="Puts two lists of words side by side.")
def pair(a: Series[str], b: Series[str]) -> str:
    return f"{a} {b}"


print(problems({
    "nodes": [{"id": "w", "node": "words"}, {"id": "r", "node": "route"}, {"id": "p", "node": "pair"}],
    "edges": [["w.result", "r.text"], ["r.short", "p.a"], ["r.long", "p.b"]],
}))  # ['series_inputs_never_share_an_item']
```

A node's `Series` inputs leave out an item together when one of them has no value for it. Each word here is short or long, never both, so every item would be left out.

**Fix.** Write `Series[X | None]` on both inputs to keep every item, blanks included, or take the branches through one input.

### `series_not_nested`

> '{node_id}' reads items of {series}, and neither is made inside the other, so nothing says which item goes with which. Make one inside the other (wire '{wire_from}' into '{wire_into}', the node that makes the other), or give one of them whole to a Series input.

```python
unpaired = registry.load({
    "nodes": [{"id": "names", "node": "words"}, {"id": "greetings", "node": "words"}, {"id": "g", "node": "greet"}],
    "edges": [["names.result", "g.name"], ["greetings.result", "g.greeting"]],
}).problems[0]
print(unpaired.type, unpaired.ctx["wire_from"], unpaired.ctx["wire_into"])  # series_not_nested names.result greetings
```

`g` would run once per name and once per greeting, and nothing says which name goes with which greeting. `ctx.series` lists the two Series, `ctx.inputs` the inputs that read them, and `ctx.wire_from` and `ctx.wire_into` name an edge that would put one inside the other.

**Fix.** Make one Series inside the other: wire an item of the first into the node that makes the second, so the second is made once per item of the first, and the node runs for each pair. Or give one of them whole to a `Series` input, so it arrives as one list.

### `several_edges_into_one_input`

> More than one of the edges into '{field}', from {sources}, can hold a value on the same call. Feed it from opposite branches of one decision, or put a node between that has an input for each.

```python
print(problems({
    "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "c", "node": "upper"}],
    "edges": [["a.result", "c.text"], ["b.result", "c.text"]],
}))  # ['several_edges_into_one_input']
```

**Fix.** Feed it from opposite branches of one decision (outputs of one `exclusive_group`), or write a node with an input for each and wire them there. A `Series` input takes several edges on the same terms.

### `type_mismatch`

> '{source}' gives {got}, but '{node_id}.{field}' takes {expected}; connect an output that gives {expected}, or convert it in a node between.

```python
print(problems({
    "nodes": [{"id": "c", "node": "count"}, {"id": "u", "node": "upper"}],
    "edges": [["c.result", "u.text"]],
}))  # ['type_mismatch']
```

**Fix.** Wire an output of the type the input takes, or put a node between that turns one into the other.

Two mismatches have a sentence of their own, and `ctx.hint` says which.

A whole table or list wired into an input of one value is `split_first`:

> '{source}' is a whole {whole} and '{node_id}.{field}' takes one {expected}. To run '{node_id}' once for each record or item in it, put a Split between them and wire Split's result, or one field of it.

```python
from typing import TypedDict

import polars as pl
from conductor import Table


class Contact(TypedDict):
    name: str
    email: str


class Invoice(TypedDict):
    name: str
    amount: int


@registry.node("contacts", title="Contacts", description="Reads contacts from a text, a name and an email on each line.")
def contacts(text: str) -> Table[Contact]:
    return pl.DataFrame([dict(zip(("name", "email"), line.split(), strict=True)) for line in text.splitlines()])


whole = registry.load({
    "nodes": [{"id": "c", "node": "contacts"}, {"id": "u", "node": "upper"}],
    "edges": [["c.result", "u.text"]],
}).problems[0]
print(whole.type, whole.ctx["hint"], whole.ctx["fields"])  # type_mismatch split_first ['name', 'email']
```

**Fix.** A table or a list is one value. Put a node between that gives its records or items as a `Series` (`split` in `syv-conductor-nodes`), and wire that node's result, or one field of it (`split.result.email`). `ctx.whole` is `table` or `list`, and `ctx.fields` lists a table's columns.

A table wired into a `Table[R]` input that names a column the table lacks is `add_column`:

> '{node_id}.{field}' needs the column '{column}' ({column_type}); '{source}' has {has}. Add the column in a node between, or connect a table that has it.

```python
@registry.node("total", title="Total", description="Adds up the amounts on all invoices.")
def total(invoices: Table[Invoice]) -> int:
    return int(invoices["amount"].sum())


lacking = registry.load({
    "nodes": [{"id": "c", "node": "contacts"}, {"id": "t", "node": "total"}],
    "edges": [["c.result", "t.invoices"]],
}).problems[0]
print(lacking.type, lacking.ctx["hint"], lacking.ctx["column"])  # type_mismatch add_column amount
```

**Fix.** Wire in a table that has every column the input names, add the column in a node between, or type the input as a bare `Table`, which takes any columns. `ctx.column` and `ctx.column_type` say what is missing, and `ctx.has` lists the columns the table has.

### `unbound_on_later_node`

> Nothing feeds '{field}', and only a first node takes inputs from the caller; draw an edge into it or type a value in.

```python
print(problems({
    "nodes": [{"id": "a", "node": "upper"}, {"id": "g", "node": "greet"}],
    "edges": [["a.result", "g.name"]],
}))  # ['unbound_on_later_node']
```

**Fix.** Draw an edge into it, or type a value in. A caller fills only a first node's open inputs.

### `unknown_input`

> '{node_id}' has no input '{field}'; it has {inputs}.

```python
print(problems({"nodes": [{"id": "u", "node": "upper", "values": {"txt": "a"}}]}))  # ['unknown_input']
```

**Fix.** Use one of the inputs `ctx.inputs` lists. After a version renamed an input, `registry.upgrade` moves the value across.

### `unknown_node`

> There is no node '{node}'; the nodes are {choices}.

```python
print(problems({"nodes": [{"id": "m", "node": "send-mail"}]}))  # ['unknown_node']
```

**Fix.** Use a node the registry has (`ctx.choices`), or add the missing one to the registry the graph loads through.

### `unknown_output`

> '{source}' is not something that node gives; it gives {outputs}.

```python
print(problems({
    "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}],
    "edges": [["a.text", "b.text"]],
}))  # ['unknown_output']
```

**Fix.** Wire from one of the addresses `ctx.outputs` lists. An edge may name one field of a record an output gives, `node.output.field`; `ctx.outputs` lists those too.

A field named on a whole table or list has a sentence of its own, with `ctx.hint` set to `split_first`:

> '{source}' names a field of a whole {whole}, which has no field to wire. Split it first: put a Split after it and wire the field from Split's result.

```python
column = registry.load({
    "nodes": [{"id": "c", "node": "contacts"}, {"id": "u", "node": "upper"}],
    "edges": [["c.result.email", "u.text"]],
}).problems[0]
print(column.type, column.ctx["hint"])  # unknown_output split_first
```

**Fix.** A table is one value, so it has no field to wire. Put a node after it that gives its records as a `Series` (`split` in `syv-conductor-nodes`), and wire the field from that node's result: `split.result.email`.

### `unknown_version`

> '{node}' has no version {version}; the newest is {newest}.

```python
print(problems({"nodes": [{"id": "u", "node": "upper", "version": 9}]}))  # ['unknown_version']
```

**Fix.** Use a version the node has. `registry.upgrade(record, node_id)` moves a saved node to the newest.

### `value_and_default`

> Field '{field}' has both a value and a default; keep one.

```python
print(problems({"nodes": [{"id": "u", "node": "upper", "values": {"text": "a"}, "defaults": {"text": "b"}}]}))  # ['value_and_default']
```

**Fix.** Keep one: a value fixes the input, a default leaves it open for a caller.

### `value_and_edge`

> Field '{field}' has an edge and a typed-in value; keep one.

```python
print(problems({
    "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper", "values": {"text": "x"}}],
    "edges": [["a.result", "b.text"]],
}))  # ['value_and_edge']
```

**Fix.** Keep one: remove the edge or the typed-in value.

## A refused call

Raised as `InvalidCallError` by the call that caused it; `error.errors()` lists every one, and nothing changed.

### `answer_stale`

> The answer was given for {waiting_id!r}, and another question waits at that place by now, {current!r}; show that one and answer it.

```python
approving = registry.load({"nodes": [{"id": "approve", "node": "approve"}]})
asked = approving.run(amount=5000)
asked_again = asked.again(amount=5500)
print(refused(lambda: asked_again.answer(asked.waiting[0].id, True)))  # ['answer_stale']
```

**Fix.** Answer the question that waits there now: `ctx.current` is its id, and `run.waiting` has it.

### `cannot_move_to_version`

> '{node}' cannot move from version {version} to {target}; it has {versions}.

```python
record = make_graph(upper)
print(refused(lambda: registry.upgrade(record, "upper", to=3)))  # ['cannot_move_to_version']
```

**Fix.** Move it to a version `ctx.versions` lists, or leave `to` out for the newest.

### `from_run_missing`

> '{node_id}' needs a {needs} for '{parameter}', and from_run gives none (it gives {given}); pass from_run={{{needs}: ...}}.

```python
print(refused(lambda: asking_who.run()))  # ['from_run_missing']
```

**Fix.** Pass the value by its type: `graph.run(from_run={Caller: Caller("Ida")})`. It is never saved with the run, so pass it again on each `resume`.

### `input_given_twice`

> '{name}' is given twice, by its name and by its address '{address}'; give it once.

```python
uppering = registry.load(make_graph(upper))
print(refused(lambda: uppering.run({"upper.text": "a"}, text="b")))  # ['input_given_twice']
```

**Fix.** Give it once, by its name or by its address.

### `invalid_answer`

> That is not a valid answer: {reason}

```python
waiting = registry.load({"nodes": [{"id": "approve", "node": "approve"}]}).run(amount=5000)
print(refused(lambda: waiting.answer(waiting.waiting[0], "yes")))  # ['invalid_answer']
```

**Fix.** Answer with a value of the question's type, as JSON reads it: `true`, not `"yes"`. `ctx.question` names the question.

### `invalid_checkpoint_every`

> checkpoint_every is a number of seconds above 0; it was {checkpoint_every}.

```python
print(refused(lambda: uppering.with_inputs(text="a").stream(checkpoint_every=0)))  # ['invalid_checkpoint_every']
```

**Fix.** Give a number of seconds above 0, or leave it out for no checkpoints.

### `invalid_input`

> The value given for '{name}' is not valid: {reason}. Give a value of the type the input takes.

```python
counting = registry.load(make_graph(count))
print(refused(lambda: counting.run(n="many")))  # ['invalid_input']
print(refused(lambda: uppering.run(text=["a", "b"])))  # ['invalid_input']
```

**Fix.** Give a value of the input's type. `graph.inputs[name].json_schema` is its schema, limits included. `reason` and `errors` are as `invalid_value` has them. An input of one value takes one value: a list given there is refused, since a graph runs once per item only after a node that returns a `Series`. An open `Series` input takes a list, also for one value.

### `invalid_max_items`

> max_items is at least 1; it was {max_items}.

```python
print(refused(lambda: uppering.run(text="a", max_items=0)))  # ['invalid_max_items']
```

**Fix.** Give `max_items` of 1 or more, or leave it out for no ceiling.

### `invalid_timeout`

> timeout is a number of seconds above 0; it was {timeout}.

```python
print(refused(lambda: uppering.with_inputs(text="a").resume(timeout=-1)))  # ['invalid_timeout']
```

**Fix.** Give a number of seconds above 0, or leave it out for a leg with no deadline.

### `invalid_record`

> The record is not a graph record at '{at}': {reason}. Fix the record's JSON.

```python
print(refused(lambda: registry.load({"nodes": [{"id": "u", "type": "upper"}]})))  # ['invalid_record', 'invalid_record']: 1.x's 'type', and no 'node'
```

**Fix.** Fix the record's JSON: `ctx.at` says where it stopped being a graph record. A drawing that is well formed but unfinished loads, with its problems in `graph.problems`.

### `missing_input`

> '{name}' is required; give it in the call.

```python
print(refused(lambda: uppering.run()))  # ['missing_input']
```

**Fix.** Give it in the call. `graph.inputs` lists what the graph takes and which are required.

### `node_not_in_graph`

> '{node_id}' is not a node of this graph; it has {nodes}.

```python
done = uppering.run(text="a")
print(refused(lambda: done.forget("lower")))  # ['node_not_in_graph']
```

**Fix.** Name a node of the graph; `ctx.nodes` lists them.

### `not_a_chain`

> '{node_id}' can't follow '{before}' in a chain: a chain takes its one required open input (it has {inputs}) from the one output of the node before it (that has {outputs}); pass edges= to make_graph.

```python
print(refused(lambda: make_graph(upper, greet)))  # ['not_a_chain']
```

**Fix.** Pass `edges=[("upper.result", "greet.name")]`. `make_graph` chains nodes only when each takes one required open input from the one output before it. A chain reads the fields each node declares; a stored graph, or a node whose fields a hook works out, takes `edges=`.

### `not_offered`

> The graph has no input '{name}'; it has {inputs}.

```python
print(refused(lambda: uppering.run(text="a", txt="a")))  # ['not_offered']
```

**Fix.** Use one of the inputs `ctx.inputs` lists.

### `not_waiting`

> Nothing is waiting on an answer as {waiting_id!r} in a run that is {status}; run.waiting lists what is.

```python
print(refused(lambda: waiting.answer("upper", True)))  # ['not_waiting']
```

**Fix.** Answer an id from `run.waiting`.

### `run_not_complete`

> The run is {status}, so it has no outputs yet; run.results has what it did. {reason}

```python
print(refused(lambda: waiting.outputs))  # ['run_not_complete']
```

**Fix.** Answer what waits and `resume()`, or read `run.results`, which has what the run did so far.

### `state_of_another_graph`

> The saved run is of another graph ({saved}) than this one ({graph}); an edited graph starts a new run.

```python
print(refused(lambda: counting.load_run(uppering.run(text="a").to_json())))  # ['state_of_another_graph']
```

**Fix.** Load the save into the graph it was saved from. An edited graph starts a new run.

### `state_unreadable`

> The saved run does not fit this graph at '{at}': {reason}. Load it into the graph it was saved from.

```python
import json

tampered = json.loads(uppering.run(text="a").to_json())
tampered["units"] = {"upper": "not units"}
print(refused(lambda: uppering.load_run(tampered)))  # ['state_unreadable']
```

**Fix.** Load what `run.to_json()` wrote, unchanged. `ctx.at` says where it stopped fitting.

### `state_unwritable`

> The run can't be written as JSON at '{at}': {reason}. Give that field a type pydantic writes as JSON.

```python
@registry.node("raw", title="Raw", description="Turns a text into raw bytes.")
def raw(text: str) -> bytes:
    return b"\xff" + text.encode()


unwritable = registry.load(make_graph(raw)).run(text="a")
print(refused(lambda: unwritable.to_json()))  # ['state_unwritable']
```

**Fix.** Return a type pydantic writes as JSON. `ctx.at` names the output that couldn't be written.

## Why a node or a run stopped

In `run.state` (an `EndedEarly`'s `cause`) and on `node_failed`. `loc` starts with the node's id, and is `()` for the run as a whole.

### `execution_failed`

> The node failed; its developer finds why in the conductor log.

```python
@registry.node("divide", title="Divide", description="Divides one by a number.")
def divide(n: int) -> float:
    return 1 / n


failed = registry.load(make_graph(divide)).run(n=0)
match failed.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # execution_failed
```

**Fix.** The node let out an exception of its own. Its class is `ctx.error`; the exception and its traceback are in the `conductor` log. Fix the node, then `resume()`.

### `external_failed`

> An outside service did not answer.

```python
@registry.node("fetch", title="Fetch", description="Fetches a page from an address, and fails.", policy=Policy(retry_on=(ConnectionError,)))
def fetch(url: str) -> str:
    raise ConnectionError(url)


cut_off = registry.load(make_graph(fetch)).run(url="https://example.com")
match cut_off.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # external_failed
```

**Fix.** The outside world failed after every retry. `resume()` tries again; raise `Policy(retries=)` for more tries.

### `failed`

> The node reported a failure.

```python
@registry.node("read_pdf", title="Read PDF", description="Reads the text from a PDF file, and fails.")
def read_pdf(path: str) -> str:
    raise NodeError("The document could not be read.")


unread = registry.load(make_graph(read_pdf)).run(path="a.pdf")
match unread.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # failed
```

**Fix.** The node said why in `msg`. A node gives its own code with `NodeError(message, code=)`.

### `invalid_ask`

> The node asked a question its outputs can't be answered with: {reason}. Fix the node's Ask.

```python
@registry.node("confirm", title="Confirm", description="Asks a person to confirm.")
def confirm(text: str) -> bool:
    raise Ask("Sure?", default="maybe")


misasked = registry.load(make_graph(confirm)).run(text="a")
match misasked.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # invalid_ask
```

**Fix.** Ask with choices and a default the node's return type accepts. `ctx.reason` says which part doesn't fit.

### `invalid_edge_value`

> The value an edge brought to '{name}' is not one it takes: {reason}. Fix the node that made it, or connect an output of the type the input takes.

```python
@registry.node("letters", title="Letters", description="Counts the letters in a text.")
def letters(text: str) -> int:
    return len(text)


@registry.node("small", title="Small", description="Passes on a small number.")
def small(n: Annotated[int, Param(le=3)]) -> int:
    return n


too_big = registry.load(make_graph(letters, small)).run(text="many letters")
match too_big.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # invalid_edge_value
```

**Fix.** Make the node upstream give what the input takes, or put a node between that converts it. Loading lets through what a type alone can't settle: a limit or a choice on the input, a host type's own check, a table whose columns are known only once it is made. `ctx.name` names the input, a table's `ctx.column` its column. An edge that names a field of a record whose type declares none (a `dict`, `Any`, a record split from a table of unknown columns) is trusted when the graph loads; if the record has no such field, the reader fails here, and `ctx.field` names the field.

### `invalid_output`

> The node returned a value that is not what it declares: {reason}. Fix what the node returns.

```python
@registry.node("guess", title="Guess", description="Hands back a text where a number is promised.")
def guess(text: str) -> int:
    return text  # type: ignore[return-value]


wrong = registry.load(make_graph(guess)).run(text="12")
match wrong.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # invalid_output
```

**Fix.** Return what the node declares. `"12"` is never an `int`: outputs are read strictly. `ctx.name` names the output.

### `item_ceiling`

> {series} has {items} items, more than the {max_items} this run may work through; resume it with a larger max_items.

```python
per_word = registry.load(make_graph(words, upper))
capped = per_word.run(text="a b c", max_items=2)
match capped.state:
    case EndedEarly(cause=cause):
        print(cause.type, cause.ctx["items"])  # item_ceiling 3
```

`upper` would run once per word, and there are three. The ceiling counts only a Series some node runs per item of: one that is only gathered or passed on whole costs no calls.

**Fix.** `capped.resume(max_items=...)` with a higher ceiling carries on, calling nothing again.

### `question_unwritable`

> The node asked a person, but what it received at '{input}' has no JSON form to show them: {reason}. Give that input a type pydantic writes as JSON.

```python
@registry.node("approve-file", title="Approve file", description="Asks a person to approve sending a file.")
def approve_file(data: bytes) -> bool:
    raise Ask("Send this file?")


unshown = registry.load(make_graph(approve_file)).run(data=b"\xff")
match unshown.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # question_unwritable
```

**Fix.** Give the input a type pydantic writes as JSON, or ask in a node that receives one. A question carries what its node received, for the person answering it. `ctx.input` names the input.

### `run_cancelled`

> The run was cancelled. Resume it to carry on, or start a new run.

```python
async def cancelled() -> str:
    stop = asyncio.Event()
    stop.set()
    async for event in uppering.with_inputs(text="a").stream(cancel=stop):
        if event.type == "run_end":
            match event.run.state:
                case EndedEarly(cause=cause):
                    return cause.type
    return ""


print(asyncio.run(cancelled()))  # run_cancelled
```

**Fix.** Store the run and `resume()` it to carry on. The calls made before the cancel are kept.

### `run_timed_out`

> The run took too long and was stopped; run.resume() carries on.

```python
@registry.node("slow", title="Slow", description="Waits a moment, then hands back the text.")
def slow(text: str) -> str:
    time.sleep(0.2)
    return text


timed_out = registry.load(make_graph(slow)).run(text="a", timeout=0.05)
match timed_out.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # run_timed_out
```

**Fix.** `resume()` carries on, calling nothing again that finished. Give a longer `timeout`, or none.

### `timeout`

> The node did not answer in time; its Policy(attempt_timeout=) sets how long it may take.

```python
@registry.node("stuck", title="Stuck", description="Waits too long, then hands back the text.", policy=Policy(attempt_timeout=0.05))
def stuck(text: str) -> str:
    time.sleep(0.2)
    return text


late = registry.load(make_graph(stuck)).run(text="a")
match late.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # timeout
```

**Fix.** Raise the node's `Policy(attempt_timeout=)`, or make the call faster. The timeout is final for that call: it is not retried.

## Serving runs

In `refused` on a run `run_router` serves, by the waiting id of the answer, and as the `detail` of a 409 from `POST /runs/{run_id}/answers`.

### `run_not_waiting`

> The run is {status}, so it takes no answers; resume it first, then answer.

An answer queued while a leg ran, when that leg ended `stopped`, `timed_out`, `cancelled` or `failed`, or an answer sent to a run in one of those states. `ctx.status` is the run's status. The router starts a leg on answers only for a run that waits for them, so a run that stopped at its ceiling or its timeout doesn't start again there on an answer.

**Fix.** `POST /runs/{run_id}/resume`, then send the answer again once the run is `waiting`.

## Declaring a node

Raised as `NodeDeclarationError` when the module that declares the node is imported. `error.code` is the code. The text is English and for the developer.

### `arriving_on_compute_added_inputs`

> compute_added_inputs parameter {name} reads an arriving type, but inputs are decided before anything arrives; read it in compute_outputs

```python
def added(value: type) -> dict[str, type]:
    return {"extra": value}


try:
    @node("adds", title="Adds", description="Passes the text on as it is.", compute_added_inputs=added)
    def adds(value: str, **extra: str) -> str:
        return value
except NodeDeclarationError as error:
    print(error.code)  # arriving_on_compute_added_inputs
```

**Fix.** Read the arriving type in `compute_outputs`. The added inputs are decided before anything arrives.

### `async_run`

> a node is a plain def; the engine gives a slow one threads with Policy(concurrency=)

```python
try:
    @node("fetch_async", title="Fetch async", description="Fetches a page from an address.")
    async def fetch_async(url: str) -> str:
        return url
except NodeDeclarationError as error:
    print(error.code)  # async_run
```

**Fix.** Write a plain `def`. For a slow call, `Policy(concurrency=8)` makes several calls at once on threads.

### `bound_invalid`

> {unheld} can't apply to {Type(source).id}, as pydantic holds no such limit on it; take them off, or check the value in the node

> the limits {param.limits} don't fit {self.id}, so change one: {error}

```python
try:
    @node("short", title="Short", description="Passes the text on as it is.")
    def short(text: Annotated[str, Param(ge=0)]) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # bound_invalid
```

**Fix.** Use limits pydantic holds on the type: `ge`, `gt`, `le`, `lt` and `multiple_of` on numbers (a `Decimal` too), dates, times and durations; `min_length` and `max_length` on texts, bytes, lists, tuples, sets and dicts; `pattern` on texts. A table takes none. The same holds for a `Param` on a field of a model a node takes.

### `choices_invalid`

> the choices hold an id twice, or none; give each choice once, as a value of the field's type

> on a Literal the choices title its values, one each: {literal}; give each choice once, as a value of the field's type

> a choice has no JSON form; give each choice once, as a value of the field's type

> a choice is not a value of the field's type: {value_free_errors(refused)[0]['msg']}; give each choice once, as a value of the field's type

```python
from typing import Literal

from conductor import Choice

try:
    @node("size", title="Size", description="Hands back the chosen size.")
    def size(pick: Annotated[Literal["s", "m"], Param(choices=[Choice("s", "Small")])]) -> str:
        return pick
except NodeDeclarationError as error:
    print(error.code)  # choices_invalid
```

**Fix.** Give each choice once, each a value of the field's type. On a `Literal`, the choices title its values, one each: `[Choice("s", "Small"), Choice("m", "Medium")]`.

### `compute_added_inputs_without_var_keyword`

> a node with compute_added_inputs takes them as **kwargs; add **values to its signature

```python
def operands(count: int) -> dict[str, type]:
    return {f"value_{i}": float for i in range(count)}


try:
    @node("sum_of", title="Sum of", description="Adds up numbers.", compute_added_inputs=operands)
    def sum_of(count: int) -> float:
        return 0.0
except NodeDeclarationError as error:
    print(error.code)  # compute_added_inputs_without_var_keyword
```

**Fix.** Take the added inputs as `**values: float`.

### `default_invalid`

> the default None is a blank; write its type with | None

> the default {self.default} is not a value of {whole.id} that it writes as JSON, so give one that is: {error}

```python
try:
    @node("maybe", title="Maybe", description="Passes the text on as it is.")
    def maybe(text: str = None) -> str:  # type: ignore[assignment]
        return text
except NodeDeclarationError as error:
    print(error.code)  # default_invalid
```

**Fix.** Write `text: str | None = None` for an input that takes a blank, or give a default of the input's type that it writes as JSON: an editor shows it, and a saved graph stores it.

### `duplicate_node_id` (declaring a node)

> two nodes have the id {definition.id}: {_where_declared(held)} and {_where_declared(definition)}; ids are unique, so swap one for the other with registry.replace(...)

> make_graph was given two nodes with the id {definition.id} that aren't versions of one node; ids are unique, so give each its own id

```python
twice = NodeRegistry()
twice.add(upper)
try:
    @twice.node("upper", title="Uppercase", description="Turns a text into capital letters.")
    def upper_again(text: str) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # duplicate_node_id
```

**Fix.** Swap one for the other with `registry.replace(definition)`, or give the new node an id of its own. A newer version of the same node names the older in `replaces=`.

### `duplicate_type_id`

> type id {Type(held).id} is declared by both {_qualified_name(existing)} and {_qualified_name(held)}; one registry holds one type per id

```python
@dataclass
class Amount:
    type_id: ClassVar[str] = "money"
    type_title: ClassVar[str] = "Amount"
    kroner: int


@dataclass
class Price:
    type_id: ClassVar[str] = "money"
    type_title: ClassVar[str] = "Price"
    kroner: float


try:
    NodeRegistry(types=[Amount, Price])
except NodeDeclarationError as error:
    print(error.code)  # duplicate_type_id
```

**Fix.** Give one of the classes another `type_id`.

### `exclusive_group_on_input`

> {name}: exclusive_group is for outputs; put it on an output's Param

```python
try:
    @node("branchy", title="Branchy", description="Passes the text on as it is.")
    def branchy(text: Annotated[str, Param(exclusive_group="x")]) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # exclusive_group_on_input
```

**Fix.** Put `exclusive_group` on the outputs that are branches, the fields of the frozen dataclass the node returns.

### `from_run_misplaced`

> FromRun is a whole parameter the host always gives, name: FromRun[Session]; anywhere else, a caller would fill it

```python
class Session: ...

try:
    @node("maybe", title="Maybe", description="Passes the text on as it is.")
    def maybe(session: FromRun[Session] | None = None) -> str:
        return ""
except NodeDeclarationError as error:
    print(error.code)  # from_run_misplaced
```

**Fix.** Write the parameter as `name: FromRun[Session]`, never inside `| None`, a list or a return type. The host
gives that value on every run, so a node can trust it; anywhere else it would be an input a caller fills.

### `from_run_not_a_class`

> {name}: FromRun names the class of the value the run is given, FromRun[Session]

```python
try:
    @node("listed", title="Listed", description="Hands back an empty text.")
    def listed(names: FromRun[list[str]]) -> str:
        return ""
except NodeDeclarationError as error:
    print(error.code)  # from_run_not_a_class
```

**Fix.** Name a class, `FromRun[Session]`: the run is given its value by that class, `from_run={Session: ...}`.

### `hook_parameter_unknown`

> {kind} parameter {name}: a hook's parameters are passed by name; name each after an input of the node

> {kind} parameter {name} names no input of the node; it has {sorted(inputs)}

> {kind} parameter **{name}: a hook takes ** only as **{name}: type on compute_outputs, for the types arriving on the inputs compute_added_inputs added; annotate it type and give the node that hook, or name each parameter after an input

```python
def from_names(names: str) -> dict[str, type]:
    return {name: str for name in names.split(",")}


try:
    @node("named", title="Named", description="Gives one answer per label.", compute_outputs=from_names)
    def named(labels: str) -> dict[str, str]:
        return {}
except NodeDeclarationError as error:
    print(error.code)  # hook_parameter_unknown
```

**Fix.** Name the hook's parameters after the node's inputs, and take them by name.

### `input_without_type`

> {name}: an input needs a type hint; write one, as it is the input's interface

```python
try:
    @node("untyped", title="Untyped", description="Passes the text on as it is.")
    def untyped(text) -> str:  # type: ignore[no-untyped-def]
        return str(text)
except NodeDeclarationError as error:
    print(error.code)  # input_without_type
```

**Fix.** Give the input a type hint. The hint is the input's interface.

### `invalid_node_id` (declaring a node)

> @node takes the node's id first, a non-empty string without '.', '/', '@' or '~', which mean something in an address or a waiting id: @node('upper'); got {id}

```python
try:
    @node("a.b", title="Dotted id", description="Passes the text on as it is.")
    def dotted_id(text: str) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # invalid_node_id
```

**Fix.** Pick an id without `.`, `/`, `@` or `~`, and never an empty one. The same rule holds for an id in a record.

### `nested_series`

> a Series of Series; a field holds one level of items, so return one Series

```python
try:
    @node("nested", title="Nested", description="Wraps a text in lists.")
    def nested(text: str) -> Series[Series[str]]:
        return [[text]]
except NodeDeclarationError as error:
    print(error.code)  # nested_series
```

**Fix.** Return one `Series`. A node after it that runs once per item and returns a `Series` of its own makes the next level.

### `not_for_a_graph`

> @node({id}) on a stored graph takes no {given}: its nodes run by their own policies and fields; set them there

```python
try:
    node("wrapped", title="Wrapped", description="Turns a text into capital letters, as a saved graph.", policy=Policy(concurrency=4))(make_graph(upper))
except NodeDeclarationError as error:
    print(error.code)  # not_for_a_graph
```

**Fix.** Set the policy, or the hook, on the nodes inside the graph.

### `output_without_type`

> a node needs a return type; write one, or -> None for no outputs

```python
try:
    @node("silent", title="Silent", description="Passes the text on as it is.")
    def silent(text: str):  # type: ignore[no-untyped-def]
        return text
except NodeDeclarationError as error:
    print(error.code)  # output_without_type
```

**Fix.** Write the return type. `-> None` declares no outputs.

### `outputs_not_frozen`

> a dataclass of outputs is frozen; write @dataclass(frozen=True)

```python
@dataclass
class Parts:
    first: str
    last: str


try:
    @node("parts", title="Parts", description="Splits a name into first and last name.")
    def parts(name: str) -> Parts:
        return Parts(*name.split(" ", 1))
except NodeDeclarationError as error:
    print(error.code)  # outputs_not_frozen
```

**Fix.** Write `@dataclass(frozen=True)`. Several calls in a run can be given the same value.

### `parameter_not_by_name`

> {name}: a node's inputs are passed by name; drop the / and *args

```python
try:
    @node("positional", title="Positional", description="Passes the text on as it is.")
    def positional(text: str, /) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # parameter_not_by_name
```

**Fix.** Drop the `/` and `*args`: the engine passes every input by its name.

### `replaces_another_node`

> @node({id}) replaces {replaces}; a node replaces an earlier @node of its own id

```python
try:
    @node("lower", title="Lower", description="Turns a text into small letters.", replaces=upper)
    def lower(text: str) -> str:
        return text.lower()
except NodeDeclarationError as error:
    print(error.code)  # replaces_another_node
```

**Fix.** A version replaces an earlier `@node` of its own id. A different node gets no `replaces=`.

### `type_not_carried`

> conductor can't check or write {self.id} as JSON; use a type pydantic knows, such as a pydantic model or a dataclass, or give the class its own __get_pydantic_core_schema__

```python
class Plain:
    type_id: ClassVar[str] = "plain"
    type_title: ClassVar[str] = "Plain"


try:
    @node("plain", title="Plain", description="Hands back an empty text.")
    def plain(value: Plain) -> str:
        return ""
except NodeDeclarationError as error:
    print(error.code)  # type_not_carried
```

**Fix.** Use a type pydantic can check and write as JSON: a pydantic model, a dataclass or a builtin. A class of your own that subclasses a builtin (`class Text(str)`) says how pydantic reads it, in a `__get_pydantic_core_schema__` of its own. [Writing nodes](writing-nodes.md#a-text-or-a-number-of-your-own) has the three lines.

### `type_variable_constrained`

> {spec.name}: the type variable {variable.__name__} is constrained to a few types; give it a bound, or write the union

```python
try:
    @node("hand-on", title="Hand on", description="Hands a value on.")
    def hand_on[T: (int, str)](value: T) -> T:
        return value
except NodeDeclarationError as error:
    print(error.code)  # type_variable_constrained
```

**Fix.** Give the variable a bound (`T: int | str`), or write the union in the signature. A bound says what may arrive; the variable is then whatever does.

### `type_variable_only_in_return`

> {spec.name}: the type variable {variable.__name__} is on no input, so no edge can say what it is; put it on an input

```python
try:
    @node("make", title="Make", description="Makes an empty list.")
    def make[T](text: str) -> list[T]:
        return []
except NodeDeclarationError as error:
    print(error.code)  # type_variable_only_in_return
```

**Fix.** A type variable is what arrives on an edge, so it stands on an input. An output whose type depends on a setting, not on an edge, is typed by a `compute_outputs` hook.

### `upgrade_missing`

> the version it replaces has {sorted(unexplained)}, gone from this one; its Upgrade says where they went

```python
@node("hello", title="Hello v1", description="Says hello to a person.")
def hello_v1(name: str) -> str:
    return name


try:
    @node("hello", title="Hello", description="Says hello to a person.", replaces=hello_v1)
    def hello(first: str) -> str:
        return first
except NodeDeclarationError as error:
    print(error.code)  # upgrade_missing
```

**Fix.** Say where the gone field went: `upgrade=Upgrade(renamed_inputs={"name": "first"})`, or a `rewrite=` for an input; a gone output takes `renamed_outputs=`, since a rewrite reads only what is typed into the node.

### `upgrade_without_replaces`

> @node({id}) has an upgrade but replaces nothing; name the version it upgrades with replaces=

```python
try:
    @node("hi", title="Hi", description="Says hi to a person.", upgrade=Upgrade(renamed_inputs={"name": "first"}))
    def hi(first: str) -> str:
        return first
except NodeDeclarationError as error:
    print(error.code)  # upgrade_without_replaces
```

**Fix.** Name the version it upgrades with `replaces=`.

### `var_keyword_without_hook`

> **kwargs takes the inputs compute_added_inputs adds; the node has no such hook, so remove **kwargs or add compute_added_inputs=

```python
try:
    @node("loose", title="Loose", description="Hands back an empty text.")
    def loose(**values: str) -> str:
        return ""
except NodeDeclarationError as error:
    print(error.code)  # var_keyword_without_hook
```

**Fix.** Remove `**values`, or add the hook that adds those inputs, `compute_added_inputs=`.

### `version_not_after`

> @node({id}) is version {number}, not after the version {replaces.version} it replaces; leave version= out, or give one above {replaces.version}

```python
try:
    @node("hello", title="Hello again", description="Says hello to a person.", replaces=hello_v1, version=1)
    def hello_again(name: str) -> str:
        return name
except NodeDeclarationError as error:
    print(error.code)  # version_not_after
```

**Fix.** Leave `version=` out (one past the version replaced), or give a number above it.

### `widget_outside_param`

> a widget in Annotated goes on Param: Annotated[..., Param(widget=...)]

```python
from conductor.widgets import Textarea

try:
    @node("note", title="Note", description="Hands a note on.")
    def note(text: Annotated[str, Textarea()]) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # widget_outside_param
```

**Fix.** Put the widget on the field's `Param`: `Annotated[str, Param(widget=Textarea())]`. Other metadata in `Annotated` is pydantic's and is read as pydantic reads it; a widget there would be passed by, and the field would show no control.

## `check_registry`

Returned by `conductor.testing.check_registry`, in a test run. A host never meets these, and they aren't in `PROBLEM_CODES`.

### `changed_its_inputs`

```python
@node("sort_in_place", title="Sort", description="Sorts a list of texts.")
def sort_in_place(texts: list[str]) -> list[str]:
    texts.sort()
    return texts


checked = NodeRegistry()
checked.add(sort_in_place)
print([p.type for p in check_registry(checked, examples={"sort_in_place": [{"texts": ["b", "a"]}]})])  # ['changed_its_inputs']
```

**Fix.** Copy the value before changing it, `sorted(texts)`. Several calls in a run can be given the same value.

### `example_for_no_node`

```python
checked = NodeRegistry()
checked.add(upper)
print([p.type for p in check_registry(checked, examples={"uper": [{"text": "a"}]})])  # ['example_for_no_node']
```

**Fix.** Check the id in `examples`, or remove the example.

### `hook_failed`

```python
def no_names(names: str) -> dict[str, type]:
    raise KeyError(names)


@node("unnamed", title="Unnamed", description="Gives one answer per name.", compute_outputs=no_names)
def unnamed(names: str) -> dict[str, str]:
    return {}


checked = NodeRegistry()
checked.add(unnamed)
print([p.type for p in check_registry(checked)])  # ['hook_failed']
```

**Fix.** Make the hook work on the node's defaults and made-up values, or give an example it works on.

### `needs_example`

```python
def by_code(code: str) -> dict[str, type]:
    return {f"field_{code}": str}


@node("coded", title="Coded", description="Gives one answer per code.", compute_outputs=by_code)
def coded(code: Annotated[str, Param(pattern=r"^\d{4}$")]) -> dict[str, str]:
    return {}


checked = NodeRegistry()
checked.add(coded)
print([p.type for p in check_registry(checked)])  # ['needs_example']
```

**Fix.** Give the node an example, `examples={"coded": [{"code": "1234"}]}`.

### `pause_untested`

```python
checked = NodeRegistry()
checked.add(approve)
print([p.type for p in check_registry(checked)])  # ['pause_untested']
```

**Fix.** Give an example that makes the node ask, `examples={"approve": [{"amount": 5000}]}`.

### `run_failed`

```python
checked = NodeRegistry()
checked.add(divide)
print([p.type for p in check_registry(checked, examples={"divide": [{"n": 0}]})])  # ['run_failed']
```

**Fix.** Fix the node or the example. `ctx.cause` has the run's own problem, and the message names the exception class a body raised. A node that is known to fail goes in a test of its own marked `@pytest.mark.xfail(strict=True)`, over `registry.filter(...)`.

### `upgrade_broken`

```python
def broken_rewrite(values: dict) -> dict:
    raise KeyError("first")


@node("hello", title="Hello v2", description="Says hello to a person.", replaces=hello_v1, upgrade=Upgrade(rewrite=broken_rewrite))
def hello_v2(first: str) -> str:
    return first


checked = NodeRegistry()
checked.add(hello_v2)
print([p.type for p in check_registry(checked)])  # ['upgrade_broken']
```

**Fix.** Fix the `Upgrade`, so a node saved at the older version loads cleanly at the newer one.
