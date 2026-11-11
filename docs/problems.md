# Problem codes

Everything conductor reports has a stable code. A `Problem`'s `type` is one, and so is a `NodeDeclarationError`'s `code`. This page has each code: where you meet it, an example that makes it, and the fix. A host that translates messages keys on the code and fills its sentence from `ctx`. `PROBLEMS.codes` holds every code a `Problem` can have that conductor writes itself; a code not in it is a node's own, from `NodeError(code=)`.

Every example on this page runs after this block. `problems(drawing)` loads a drawing and lists the codes of what is wrong with it, and `told(drawing)` the sentence of each; `refused(call)` makes a call and lists the codes it was refused for, and `said(call)` the sentence of each.

```python
import asyncio
import time
from dataclasses import dataclass
from typing import Annotated, ClassVar

from conductor import (
    Ask, EndedEarly, ExternalError, FromRun, GraphRecord, InvalidCallError, NodeDeclarationError,
    NodeError, NodeRegistry, Param, Policy, Series, Upgrade, make_graph, node, stored_graph,
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
    return [problem.type for problem in registry.load(GraphRecord.model_validate(drawing)).problems]


def told(drawing: dict) -> list[str]:
    """What loading ``drawing`` tells a person, a sentence for each problem."""
    return [problem.msg for problem in registry.load(GraphRecord.model_validate(drawing)).problems]


def refused(call) -> list[str]:
    """The codes a refused call lists."""
    try:
        call()
    except InvalidCallError as error:
        return [problem.type for problem in error.errors()]
    return []


def said(call) -> list[str]:
    """What a refused call tells a person, a sentence for each problem."""
    try:
        call()
    except InvalidCallError as error:
        return [problem.msg for problem in error.errors()]
    return []
```

## Loading a graph

In `graph.problems`, never raised. `loc` is `(node_id,)` or `(node_id, field)`.

### `alternatives_in_different_series`

```python
drawing = {
    "nodes": [
        {"id": "w", "node": "words"}, {"id": "r", "node": "route"},
        {"id": "letters", "node": "words"}, {"id": "u", "node": "upper"},
    ],
    "edges": [
        ["w.result", "r.text"], ["r.short", "letters.text"],
        ["letters.result", "u.text"], ["r.long", "u.text"],
    ],
}
print(problems(drawing))  # ['alternatives_in_different_series']
print(told(drawing)[0])  # 'text' takes alternatives that run per item of different Series, w.result, letters.result, so a call for an item of one has no place for the other's value. Bring them to items of one Series first.
```

`u.text` is fed from both branches of `r`. The long branch gives one value per item of `w.result`. The short branch goes through `letters`, which makes a Series of its own, so it gives one value per item of `letters.result`.

**Fix.** Bring both branches to items of one Series first: gather the inner one in a node with a `Series` input, or feed the node from one of them.

Its `ctx` keys are in the reference: `python -m conductor.about alternatives-in-different-series`.

### `compute_hook_failed`

```python
def by_names(names: str) -> dict[str, type]:
    if not names:
        raise ValueError("no names")
    return {name: str for name in names.split(",")}


@registry.node("pick", title="Pick", description="Turns a list of names into one answer per name.", compute_outputs=by_names)
def pick(names: str) -> dict[str, str]:
    return {name: name for name in names.split(",")}


drawing = {"nodes": [{"id": "pick", "node": "pick", "values": {"names": ""}}]}
print(problems(drawing))  # ['compute_hook_failed']
print(told(drawing)[0])  # The node could not work out its fields from what it was given: no names. Check what is typed into the node and what is wired to it.
```

**Fix.** Type in a value the node's hook can work its fields out from. The hook's own exception is in the `conductor` log. A hook that raises `NodeError` gives its `code` as `ctx.error` (`failed` when it names none, `external_failed` for an `ExternalError`) and its own `ctx` whole as `ctx.error_ctx`; any other exception gives its class name.

Its `ctx` keys are in the reference: `python -m conductor.about compute-hook-failed`.

### `cycle`

```python
drawing = {
    "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}],
    "edges": [["a.result", "b.text"], ["b.result", "a.text"]],
}
print(problems(drawing))  # ['cycle', 'cycle']
print(told(drawing)[0])  # The node is part of a cycle, so the graph cannot run; remove one of the edges that close it.
```

**Fix.** Remove one of the edges that close the loop. A graph runs each node after what it reads.

Its `ctx` keys are in the reference: `python -m conductor.about cycle`.

### `default_on_later_node`

```python
drawing = {
    "nodes": [{"id": "a", "node": "upper"}, {"id": "g", "node": "greet", "defaults": {"greeting": "hi"}}],
    "edges": [["a.result", "g.name"]],
}
print(problems(drawing))  # ['default_on_later_node']
print(told(drawing)[0])  # 'greeting' is not an open input of the graph, so a caller can't change its default; make it a value.
```

**Fix.** Write it as a value (`values`), which fixes it. Only an open input of a first node is the caller's to fill, so only there does a default mean anything; an input a hook reads is never open.

Its `ctx` keys are in the reference: `python -m conductor.about default-on-later-node`.

### `duplicate_node_id`

```python
drawing = {"nodes": [{"id": "a", "node": "upper"}, {"id": "a", "node": "count"}]}
print(problems(drawing))  # ['duplicate_node_id']
print(told(drawing)[0])  # Two nodes have the id 'a'; rename one.
```

**Fix.** Give one of the nodes another id. `make_graph` picks free ids itself (`upper-1`, `upper-2`).

Its `ctx` keys are in the reference: `python -m conductor.about duplicate-node-id`.

### `edge_names_missing_node`

```python
drawing = {"nodes": [{"id": "a", "node": "upper"}], "edges": [["a.result", "ghost.text"]]}
print(problems(drawing))  # ['edge_names_missing_node']
print(told(drawing)[0])  # The edge from 'a.result' to 'ghost.text' names 'ghost', which is not a node of the graph; add that node or remove the edge.
```

**Fix.** Add the node the edge names, or remove the edge.

Its `ctx` keys are in the reference: `python -m conductor.about edge-names-missing-node`.

### `embeds_itself`

```python
inner_record = GraphRecord.model_validate({"nodes": [{"id": "again", "node": "itself"}]})
itself = stored_graph("itself", title="Itself", description="A graph that contains itself.")(inner_record)
looping = NodeRegistry()
looping.add(itself)
broken = looping.load(GraphRecord.model_validate({"nodes": [{"id": "i", "node": "itself"}]})).problems[0]
print(broken.type, broken.loc)  # embeds_itself ('i', 'again')
print(broken.msg)  # The node 'itself' is a graph that contains itself; embed a graph that does not contain this one.
```

**Fix.** A stored graph can't contain itself, directly or through another. Remove the node that leads back. The problem
sits on the node that closes the circle: a problem of a graph used as a node is that graph's own, with the id of the
node that uses it in front of its `loc`.

Its `ctx` keys are in the reference: `python -m conductor.about embeds-itself`.

### `hook_needs_value`

```python
drawing = {"nodes": [{"id": "pick", "node": "pick"}]}
print(problems(drawing))  # ['hook_needs_value']
print(told(drawing)[0])  # 'names' must be typed in: the node works out its fields from it when the graph loads.
```

**Fix.** Type a value into the input the hook reads. The node's fields depend on it, so it is fixed when the graph loads and never one of the graph's inputs.

Its `ctx` keys are in the reference: `python -m conductor.about hook-needs-value`.

### `invalid_node_id`

```python
drawing = {"nodes": [{"id": "a.b", "node": "upper"}]}
print(problems(drawing))  # ['invalid_node_id']
print(told(drawing)[0])  # 'a.b' cannot be a node id: '.', '/', '@' and '~' each mean something in an address or a waiting id; rename it without them.
```

**Fix.** Pick an id without `.`, `/`, `@` or `~`: those characters build addresses (`a.result`) and waiting ids (`a@1~3fa2`).

Its `ctx` keys are in the reference: `python -m conductor.about invalid-node-id`.

### `invalid_value`

```python
drawing = {"nodes": [{"id": "c", "node": "count", "values": {"n": "many"}}]}
print(problems(drawing))  # ['invalid_value']
print(told(drawing)[0])  # The value typed into 'n' is not valid: Input should be a valid integer, unable to parse string as an integer. Type a value of the field's type.
```

**Fix.** Type in a value of the field's type. `reason` says what the type refused, without quoting the value; `errors` is pydantic's list of what it refused (`type`, `loc`, `msg`; `loc` as far as the type names it, an index or a declared field, never a key of a dict), each without the value, so a host can key on `greater_than_equal` or `bool_is_not_a_number`.

Its `ctx` keys are in the reference: `python -m conductor.about invalid-value`.

### `parameter_name_invalid`

```python
def dotted(names: str) -> dict[str, type]:
    return {name: str for name in names.split(",")}


@registry.node("fields", title="Fields", description="Gives one answer per name.", compute_outputs=dotted)
def fields(names: str) -> dict[str, str]:
    return {}


drawing = {"nodes": [{"id": "f", "node": "fields", "values": {"names": "a.b"}}]}
print(problems(drawing))  # ['parameter_name_invalid']
print(told(drawing)[0])  # 'a.b' cannot be a field name: '.', '/' and '@' each mean something in an address, and a name can't be empty; rename the field.
```

**Fix.** Have the hook give names without `.`, `/` or `@`, and never an empty one.

Its `ctx` keys are in the reference: `python -m conductor.about parameter-name-invalid`.

### `repeated_edge`

```python
drawing = {
    "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}],
    "edges": [["a.result", "b.text"], ["a.result", "b.text"]],
}
print(problems(drawing))  # ['repeated_edge']
print(told(drawing)[0])  # The edge from 'a.result' into 'text' is drawn twice; remove one.
```

**Fix.** Remove one of the two.

Its `ctx` keys are in the reference: `python -m conductor.about repeated-edge`.

### `series_inputs_never_share_an_item`

```python
@registry.node("pair", title="Pair", description="Puts two lists of words side by side.")
def pair(a: Series[str], b: Series[str]) -> str:
    return f"{a} {b}"


drawing = {
    "nodes": [{"id": "w", "node": "words"}, {"id": "r", "node": "route"}, {"id": "p", "node": "pair"}],
    "edges": [["w.result", "r.text"], ["r.short", "p.a"], ["r.long", "p.b"]],
}
print(problems(drawing))  # ['series_inputs_never_share_an_item']
print(told(drawing)[0])  # a, b read branches that never both have a value for an item, so no item would reach the node. Write Series[X | None] on them to keep every item, or take the branches through one input.
```

A node's `Series` inputs leave out an item together when one of them has no value for it. Each word here is short or long, never both, so every item would be left out.

**Fix.** Write `Series[X | None]` on both inputs to keep every item, blanks included, or take the branches through one input.

Its `ctx` keys are in the reference: `python -m conductor.about series-inputs-never-share-an-item`.

### `series_not_nested`

```python
unpaired = registry.load(GraphRecord.model_validate({
    "nodes": [{"id": "names", "node": "words"}, {"id": "greetings", "node": "words"}, {"id": "g", "node": "greet"}],
    "edges": [["names.result", "g.name"], ["greetings.result", "g.greeting"]],
})).problems[0]
print(unpaired.type, unpaired.ctx["wire_from"], unpaired.ctx["wire_into"])  # series_not_nested names.result greetings
print(unpaired.msg)  # 'g' reads items of names.result, greetings.result, and neither is made inside the other, so nothing says which item goes with which. Make one inside the other (wire 'names.result' into 'greetings', the node that makes the other), or give one of them whole to a Series input.
```

`g` would run once per name and once per greeting, and nothing says which name goes with which greeting. `ctx.series` lists the two Series, `ctx.inputs` the inputs that read them, and `ctx.wire_from` and `ctx.wire_into` name an edge that would put one inside the other.

**Fix.** Make one Series inside the other: wire an item of the first into the node that makes the second, so the second is made once per item of the first, and the node runs for each pair. Or give one of them whole to a `Series` input, so it arrives as one list.

Its `ctx` keys are in the reference: `python -m conductor.about series-not-nested`.

### `several_edges_into_one_input`

```python
drawing = {
    "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "c", "node": "upper"}],
    "edges": [["a.result", "c.text"], ["b.result", "c.text"]],
}
print(problems(drawing))  # ['several_edges_into_one_input']
print(told(drawing)[0])  # More than one of the edges into 'text', from a.result, b.result, can hold a value on the same call. Feed it from opposite branches of one decision, or put a node between that has an input for each.
```

**Fix.** Feed it from opposite branches of one decision (outputs of one `exclusive_group`), or write a node with an input for each and wire them there. A `Series` input takes several edges on the same terms.

Its `ctx` keys are in the reference: `python -m conductor.about several-edges-into-one-input`.

### `type_mismatch`

```python
drawing = {
    "nodes": [{"id": "c", "node": "count"}, {"id": "u", "node": "upper"}],
    "edges": [["c.result", "u.text"]],
}
print(problems(drawing))  # ['type_mismatch']
print(told(drawing)[0])  # 'c.result' gives int, but 'u.text' takes str; connect an output that gives str, or convert it in a node between.
```

**Fix.** Wire an output of the type the input takes, or put a node between that turns one into the other.

Two mismatches have a sentence of their own, and `ctx.hint` says which.

A whole table or list wired into an input of one value is `split_first`:

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


whole = registry.load(GraphRecord.model_validate({
    "nodes": [{"id": "c", "node": "contacts"}, {"id": "u", "node": "upper"}],
    "edges": [["c.result", "u.text"]],
})).problems[0]
print(whole.type, whole.ctx["hint"], whole.ctx["fields"])  # type_mismatch split_first ['name', 'email']
print(whole.msg)  # 'c.result' is a whole table and 'u.text' takes one str. To run 'u' once for each record or item in it, put a Split between them and wire Split's result, or one field of it.
```

**Fix.** A table or a list is one value. Put a node between that gives its records or items as a `Series` (`series-split` in `syv-conductor-nodes`), and wire that node's result, or one field of it (`series-split.result.email`). `ctx.whole` is `table` or `list`, and `ctx.fields` lists a table's columns.

A table wired into a `Table[R]` input that names a column the table lacks is `add_column`:

```python
@registry.node("total", title="Total", description="Adds up the amounts on all invoices.")
def total(invoices: Table[Invoice]) -> int:
    return int(invoices["amount"].sum())


lacking = registry.load(GraphRecord.model_validate({
    "nodes": [{"id": "c", "node": "contacts"}, {"id": "t", "node": "total"}],
    "edges": [["c.result", "t.invoices"]],
})).problems[0]
print(lacking.type, lacking.ctx["hint"], lacking.ctx["column"])  # type_mismatch add_column amount
print(lacking.msg)  # 't.invoices' needs the column 'amount' (int); 'c.result' has name, email. Add the column in a node between, or connect a table that has it.
```

**Fix.** Wire in a table that has every column the input names, add the column in a node between, or type the input as a bare `Table`, which takes any columns. `ctx.column` and `ctx.column_type` say what is missing, and `ctx.has` lists the columns the table has.

Its `ctx` keys are in the reference: `python -m conductor.about type-mismatch`.

### `unbound_on_later_node`

```python
drawing = {
    "nodes": [{"id": "a", "node": "upper"}, {"id": "g", "node": "greet"}],
    "edges": [["a.result", "g.name"]],
}
print(problems(drawing))  # ['unbound_on_later_node']
print(told(drawing)[0])  # Nothing feeds 'greeting', and only a first node takes inputs from the caller; draw an edge into it or type a value in.
```

**Fix.** Draw an edge into it, or type a value in. A caller fills only a first node's open inputs.

Its `ctx` keys are in the reference: `python -m conductor.about unbound-on-later-node`.

### `unknown_input`

```python
drawing = {"nodes": [{"id": "u", "node": "upper", "values": {"txt": "a"}}]}
print(problems(drawing))  # ['unknown_input']
print(told(drawing)[0])  # 'u' has no input 'txt'; it has text.
```

**Fix.** Use one of the inputs `ctx.inputs` lists. After a version renamed an input, `registry.upgrade` moves the value across.

Its `ctx` keys are in the reference: `python -m conductor.about unknown-input`.

### `unknown_node`

```python
drawing = {"nodes": [{"id": "m", "node": "send-mail"}]}
print(problems(drawing))  # ['unknown_node']
print(told(drawing)[0])  # There is no node 'send-mail'; the nodes are approve, contacts, count, fields, greet, join, pair, pick, route, total, upper, who, words.
```

**Fix.** Use a node the registry has (`ctx.choices`), or add the missing one to the registry the graph loads through.

Its `ctx` keys are in the reference: `python -m conductor.about unknown-node`.

### `unknown_output`

```python
drawing = {
    "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}],
    "edges": [["a.text", "b.text"]],
}
print(problems(drawing))  # ['unknown_output']
print(told(drawing)[0])  # 'a.text' is not something that node gives; it gives a.result.
```

**Fix.** Wire from one of the addresses `ctx.outputs` lists. An edge may name one field of a record an output gives, `node.output.field`; `ctx.outputs` lists those too.

A field named on a whole table or list has a sentence of its own, with `ctx.hint` set to `split_first`:

```python
column = registry.load(GraphRecord.model_validate({
    "nodes": [{"id": "c", "node": "contacts"}, {"id": "u", "node": "upper"}],
    "edges": [["c.result.email", "u.text"]],
})).problems[0]
print(column.type, column.ctx["hint"])  # unknown_output split_first
print(column.msg)  # 'c.result.email' names a field of a whole table, which has no field to wire. Split it first: put a Split after it and wire the field from Split's result.
```

**Fix.** A table is one value, so it has no field to wire. Put a node after it that gives its records as a `Series` (`series-split` in `syv-conductor-nodes`), and wire the field from that node's result: `series-split.result.email`.

Its `ctx` keys are in the reference: `python -m conductor.about unknown-output`.

### `unknown_version`

```python
drawing = {"nodes": [{"id": "u", "node": "upper", "version": 9}]}
print(problems(drawing))  # ['unknown_version']
print(told(drawing)[0])  # 'upper' has no version 9; the newest is 1.
```

**Fix.** Use a version the node has. `registry.upgrade(record, node_id)` moves a saved node to the newest.

Its `ctx` keys are in the reference: `python -m conductor.about unknown-version`.

### `value_and_default`

```python
drawing = {"nodes": [{"id": "u", "node": "upper", "values": {"text": "a"}, "defaults": {"text": "b"}}]}
print(problems(drawing))  # ['value_and_default']
print(told(drawing)[0])  # Field 'text' has both a value and a default; keep one.
```

**Fix.** Keep one: a value fixes the input, a default leaves it open for a caller.

Its `ctx` keys are in the reference: `python -m conductor.about value-and-default`.

### `value_and_edge`

```python
drawing = {
    "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper", "values": {"text": "x"}}],
    "edges": [["a.result", "b.text"]],
}
print(problems(drawing))  # ['value_and_edge']
print(told(drawing)[0])  # Field 'text' has an edge and a typed-in value; keep one.
```

**Fix.** Keep one: remove the edge or the typed-in value.

## A refused call

Raised as `InvalidCallError` by the call that caused it; `error.errors()` lists every one, and nothing changed.

Its `ctx` keys are in the reference: `python -m conductor.about value-and-edge`.

### `answer_stale`

```python
approving = registry.load(GraphRecord.model_validate({"nodes": [{"id": "approve", "node": "approve"}]}))
asked = approving.run(amount=5000)
asked_again = asked.again(amount=5500)
print(refused(lambda: asked_again.answer(asked.waiting[0].id, True)))  # ['answer_stale']
print(said(lambda: asked_again.answer(asked.waiting[0].id, True))[0])  # The answer was given for 'approve~baeb6e2d', and another question waits at that place by now, 'approve~81aa8d1c'; show that one and answer it.
```

**Fix.** Answer the question that waits there now: `ctx.current` is its id, and `run.waiting` has it.

Its `ctx` keys are in the reference: `python -m conductor.about answer-stale`.

### `cannot_move_to_version`

```python
record = make_graph(upper)
print(refused(lambda: registry.upgrade(record, "upper", to=3)))  # ['cannot_move_to_version']
print(said(lambda: registry.upgrade(record, "upper", to=3))[0])  # 'upper' cannot move from version 1 to 3; it has 1.
```

**Fix.** Move it to a version `ctx.versions` lists, or leave `to` out for the newest.

Its `ctx` keys are in the reference: `python -m conductor.about cannot-move-to-version`.

### `from_run_missing`

```python
print(refused(lambda: asking_who.run()))  # ['from_run_missing']
print(said(lambda: asking_who.run())[0])  # 'who' needs a Caller for 'caller', and from_run gives none (it gives none); pass from_run with a value for Caller.
```

**Fix.** Pass the value by its type: `graph.run(from_run={Caller: Caller("Ida")})`. It is never saved with the run, so pass it again on each `resume`.

Its `ctx` keys are in the reference: `python -m conductor.about from-run-missing`.

### `input_given_twice`

```python
uppering = registry.load(make_graph(upper))
print(refused(lambda: uppering.run({"upper.text": "a"}, text="b")))  # ['input_given_twice']
print(said(lambda: uppering.run({"upper.text": "a"}, text="b"))[0])  # 'text' is given twice, by its name and by its address 'upper.text'; give it once.
```

**Fix.** Give it once, by its name or by its address.

Its `ctx` keys are in the reference: `python -m conductor.about input-given-twice`.

### `invalid_answer`

```python
waiting = registry.load(GraphRecord.model_validate({"nodes": [{"id": "approve", "node": "approve"}]})).run(amount=5000)
print(refused(lambda: waiting.answer(waiting.waiting[0], "yes")))  # ['invalid_answer']
print(said(lambda: waiting.answer(waiting.waiting[0], "yes"))[0])  # That is not a valid answer: Input should be a valid boolean
```

**Fix.** Answer with a value of the question's type, as JSON reads it: `true`, not `"yes"`. `ctx.question` names the question.

Its `ctx` keys are in the reference: `python -m conductor.about invalid-answer`.

### `invalid_checkpoint_every`

```python
print(refused(lambda: uppering.with_inputs(text="a").stream(checkpoint_every=0)))  # ['invalid_checkpoint_every']
print(said(lambda: uppering.with_inputs(text="a").stream(checkpoint_every=0))[0])  # checkpoint_every is a number of seconds above 0; it was 0.
```

**Fix.** Give a number of seconds above 0, or leave it out for no checkpoints.

Its `ctx` keys are in the reference: `python -m conductor.about invalid-checkpoint-every`.

### `invalid_input`

```python
counting = registry.load(make_graph(count))
print(refused(lambda: counting.run(n="many")))  # ['invalid_input']
print(said(lambda: counting.run(n="many"))[0])  # The value given for 'n' is not valid: Input should be a valid integer, unable to parse string as an integer. Give a value of the type the input takes.
print(refused(lambda: uppering.run(text=["a", "b"])))  # ['invalid_input']
```

**Fix.** Give a value of the input's type. `graph.inputs[name].json_schema` is its schema, limits included. `reason` and `errors` are as `invalid_value` has them. An input of one value takes one value: a list given there is refused, since a graph runs once per item only after a node that returns a `Series`. An open `Series` input takes a list, also for one value.

Its `ctx` keys are in the reference: `python -m conductor.about invalid-input`.

### `invalid_max_items`

```python
print(refused(lambda: uppering.run(text="a", max_items=0)))  # ['invalid_max_items']
print(said(lambda: uppering.run(text="a", max_items=0))[0])  # max_items is at least 1; it was 0.
```

**Fix.** Give `max_items` of 1 or more, or leave it out for no ceiling.

Its `ctx` keys are in the reference: `python -m conductor.about invalid-max-items`.

### `invalid_timeout`

```python
print(refused(lambda: uppering.with_inputs(text="a").resume(timeout=-1)))  # ['invalid_timeout']
print(said(lambda: uppering.with_inputs(text="a").resume(timeout=-1))[0])  # timeout is a number of seconds above 0; it was -1.
```

**Fix.** Give a number of seconds above 0, or leave it out for a leg with no deadline.

Its `ctx` keys are in the reference: `python -m conductor.about invalid-timeout`.

### `invalid_record`

```python
print(refused(lambda: registry.load('{"nodes": [{"id": "u", "type": "upper"}]}')))  # ['invalid_record', 'invalid_record']: 1.x's 'type', and no 'node'
print(said(lambda: registry.load('{"nodes": [{"id": "u", "type": "upper"}]}'))[0])  # The record is not a graph record at 'nodes.0.node': Field required. Fix the record's JSON.
```

**Fix.** Fix the record's JSON: `ctx.at` says where it stopped being a graph record. A drawing that is well formed but unfinished loads, with its problems in `graph.problems`.

Its `ctx` keys are in the reference: `python -m conductor.about invalid-record`.

### `missing_input`

```python
print(refused(lambda: uppering.run()))  # ['missing_input']
print(said(lambda: uppering.run())[0])  # 'text' is required; give it in the call.
```

**Fix.** Give it in the call. `graph.inputs` lists what the graph takes and which are required.

Its `ctx` keys are in the reference: `python -m conductor.about missing-input`.

### `node_not_in_graph`

```python
done = uppering.run(text="a")
print(refused(lambda: done.forget("lower")))  # ['node_not_in_graph']
print(said(lambda: done.forget("lower"))[0])  # 'lower' is not a node of this graph; it has upper.
```

**Fix.** Name a node of the graph; `ctx.nodes` lists them.

Its `ctx` keys are in the reference: `python -m conductor.about node-not-in-graph`.

### `not_a_chain`

```python
print(refused(lambda: make_graph(upper, greet)))  # ['not_a_chain']
print(said(lambda: make_graph(upper, greet))[0])  # 'greet' can't follow 'upper' in a chain: a chain takes its one required open input (it has name, greeting) from the one output of the node before it (that has result); pass edges= to make_graph.
```

**Fix.** Pass `edges=[("upper.result", "greet.name")]`. `make_graph` chains nodes only when each takes one required open input from the one output before it. A chain reads the fields each node declares; a stored graph, or a node whose fields a hook works out, takes `edges=`.

Its `ctx` keys are in the reference: `python -m conductor.about not-a-chain`.

### `not_offered`

```python
print(refused(lambda: uppering.run(text="a", txt="a")))  # ['not_offered']
print(said(lambda: uppering.run(text="a", txt="a"))[0])  # The graph has no input 'txt'; it has text.
```

**Fix.** Use one of the inputs `ctx.inputs` lists.

Its `ctx` keys are in the reference: `python -m conductor.about not-offered`.

### `not_waiting`

```python
print(refused(lambda: waiting.answer("upper", True)))  # ['not_waiting']
print(said(lambda: waiting.answer("upper", True))[0])  # Nothing is waiting on an answer as 'upper' in a run that is waiting; run.waiting lists what is.
```

**Fix.** Answer an id from `run.waiting`.

Its `ctx` keys are in the reference: `python -m conductor.about not-waiting`.

### `run_not_complete`

```python
print(refused(lambda: waiting.outputs))  # ['run_not_complete']
print(said(lambda: waiting.outputs)[0])  # The run is waiting, so it has no outputs yet; run.results has what it did.
```

**Fix.** Answer what waits and `resume()`, or read `run.results`, which has what the run did so far.

Its `ctx` keys are in the reference: `python -m conductor.about run-not-complete`.

### `state_of_another_graph`

```python
print(refused(lambda: counting.load_run(uppering.run(text="a").to_json())))  # ['state_of_another_graph']
print(said(lambda: counting.load_run(uppering.run(text="a").to_json()))[0])  # The saved run is of another graph (2071554324bb5138) than this one (afb5e890aac04b55); an edited graph starts a new run.
```

**Fix.** Load the save into the graph it was saved from. An edited graph starts a new run.

Its `ctx` keys are in the reference: `python -m conductor.about state-of-another-graph`.

### `state_unreadable`

```python
import json

tampered = json.loads(uppering.run(text="a").to_json())
tampered["units"] = {"upper": "not units"}
print(refused(lambda: uppering.load_run(json.dumps(tampered))))  # ['state_unreadable']
print(said(lambda: uppering.load_run(json.dumps(tampered)))[0])  # The saved run does not fit this graph at 'units.upper': Input should be an object. Load it into the graph it was saved from.
```

**Fix.** Load what `run.to_json()` wrote, unchanged. `ctx.at` says where it stopped fitting.

Its `ctx` keys are in the reference: `python -m conductor.about state-unreadable`.

### `state_unwritable`

```python
@registry.node("raw", title="Raw", description="Turns a text into raw bytes.")
def raw(text: str) -> bytes:
    return b"\xff" + text.encode()


unwritable = registry.load(make_graph(raw)).run(text="a")
print(refused(lambda: unwritable.to_json()))  # ['state_unwritable']
print(said(lambda: unwritable.to_json())[0])  # The run can't be written as JSON at 'raw.result': a value has no JSON form. Give that field a type pydantic writes as JSON.
```

**Fix.** Return a type pydantic writes as JSON. `ctx.at` names the output that couldn't be written.

## Why a node or a run stopped

In `run.state` (an `EndedEarly`'s `cause`) and on `node_failed`. `loc` starts with the node's id, and is `()` for the run as a whole.

Its `ctx` keys are in the reference: `python -m conductor.about state-unwritable`.

### `execution_failed`

```python
@registry.node("divide", title="Divide", description="Divides one by a number.")
def divide(n: int) -> float:
    return 1 / n


failed = registry.load(make_graph(divide)).run(n=0)
match failed.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # execution_failed
        print(cause.msg)  # The node failed; its developer finds why in the conductor log.
```

**Fix.** The node let out an exception of its own. Its class is `ctx.error`; the exception and its traceback are in the `conductor` log. Fix the node, then `resume()`.

Its `ctx` keys are in the reference: `python -m conductor.about execution-failed`.

### `external_failed`

```python
@registry.node("fetch", title="Fetch", description="Fetches a page from an address, and fails.", policy=Policy(retry_on=(ConnectionError,)))
def fetch(url: str) -> str:
    raise ConnectionError(url)


cut_off = registry.load(make_graph(fetch)).run(url="https://example.com")
match cut_off.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # external_failed
        print(cause.msg)  # An outside service did not answer.
```

**Fix.** The outside world failed after every retry. `resume()` tries again; raise `Policy(retries=)` for more tries.

Its `ctx` keys are in the reference: `python -m conductor.about external-failed`.

### `failed`

```python
@registry.node("read_pdf", title="Read PDF", description="Reads the text from a PDF file, and fails.")
def read_pdf(path: str) -> str:
    raise NodeError("The document could not be read.")


unread = registry.load(make_graph(read_pdf)).run(path="a.pdf")
match unread.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # failed
```

**Fix.** The node said why in `msg`. A node gives its own code with `NodeError(message, code=)`, and what a host fills its sentence from with `ctx=`, JSON values only: a `ctx` holding anything else raises pydantic's `ValidationError` where the node raises the error, which is `execution_failed`.

Without a message of your own a host falls back on the catalogue's sentence for the code, in the reference.

Its `ctx` keys are in the reference: `python -m conductor.about failed`.

### `invalid_ask`

```python
@registry.node("confirm", title="Confirm", description="Asks a person to confirm.")
def confirm(text: str) -> bool:
    raise Ask("Sure?", default="maybe")


misasked = registry.load(make_graph(confirm)).run(text="a")
match misasked.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # invalid_ask
        print(cause.msg)  # The node asked a question its outputs can't be answered with: the default for 'result' is not an answer it takes: Input should be a valid boolean. Fix the node's Ask.
```

**Fix.** Ask with choices and a default the node's return type accepts. `ctx.reason` says which part doesn't fit.

Its `ctx` keys are in the reference: `python -m conductor.about invalid-ask`.

### `invalid_edge_value`

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
        print(cause.msg)  # The value an edge brought to 'n' is not one it takes: Input should be less than or equal to 3. Fix the node that made it, or connect an output of the type the input takes.
```

**Fix.** Make the node upstream give what the input takes, or put a node between that converts it. Loading lets through what a type alone can't settle: a limit or a choice on the input, a host type's own check, a table whose columns are known only once it is made. `ctx.name` names the input, a table's `ctx.column` its column. An edge that names a field of a record whose type declares none (a `dict`, `Any`, a record split from a table of unknown columns) is trusted when the graph loads; if the record has no such field, the reader fails here, and `ctx.record_field` names the field.

Its `ctx` keys are in the reference: `python -m conductor.about invalid-edge-value`.

### `invalid_output`

```python
@registry.node("guess", title="Guess", description="Hands back a text where a number is promised.")
def guess(text: str) -> int:
    return text  # type: ignore[return-value]


wrong = registry.load(make_graph(guess)).run(text="12")
match wrong.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # invalid_output
        print(cause.msg)  # The node returned a value that is not what it declares: Input should be a valid integer. Fix what the node returns.
```

**Fix.** Return what the node declares. `"12"` is never an `int`: outputs are read strictly. `ctx.name` names the output.

Its `ctx` keys are in the reference: `python -m conductor.about invalid-output`.

### `item_ceiling`

```python
per_word = registry.load(make_graph(words, upper))
capped = per_word.run(text="a b c", max_items=2)
match capped.state:
    case EndedEarly(cause=cause):
        print(cause.type, cause.ctx["items"])  # item_ceiling 3
        print(cause.msg)  # words.result has 3 items, more than the 2 this run may work through; raise its max_items and resume it.
```

`upper` would run once per word, and there are three. The ceiling counts only a Series some node runs per item of: one that is only gathered or passed on whole costs no calls.

**Fix.** `capped.with_max_items(...).resume()` with a higher ceiling carries on, calling nothing again.

Its `ctx` keys are in the reference: `python -m conductor.about item-ceiling`.

### `question_unwritable`

```python
@registry.node("approve-file", title="Approve file", description="Asks a person to approve sending a file.")
def approve_file(data: bytes) -> bool:
    raise Ask("Send this file?")


unshown = registry.load(make_graph(approve_file)).run(data=b"\xff")
match unshown.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # question_unwritable
        print(cause.msg)  # The node asked a person, but what it received at 'data' has no JSON form to show them: Error serializing to JSON: invalid utf-8 sequence of 1 bytes from index 0. Give that input a type pydantic writes as JSON.
```

**Fix.** Give the input a type pydantic writes as JSON, or ask in a node that receives one. A question carries what its node received, for the person answering it. `ctx.input` names the input.

Its `ctx` keys are in the reference: `python -m conductor.about question-unwritable`.

### `run_cancelled`

```python
async def cancelled() -> tuple[str, str]:
    stop = asyncio.Event()
    stop.set()
    async for event in uppering.with_inputs(text="a").stream(cancel=stop):
        if event.type == "run_end":
            match event.run.state:
                case EndedEarly(cause=cause):
                    return cause.type, cause.msg
    return "", ""


print(asyncio.run(cancelled())[0])  # run_cancelled
print(asyncio.run(cancelled())[1])  # The run was cancelled. Resume it to carry on, or start a new run.
```

**Fix.** Store the run and `resume()` it to carry on. The calls made before the cancel are kept.

Its `ctx` keys are in the reference: `python -m conductor.about run-cancelled`.

### `run_timed_out`

```python
@registry.node("slow", title="Slow", description="Waits a moment, then hands back the text.")
def slow(text: str) -> str:
    time.sleep(0.2)
    return text


timed_out = registry.load(make_graph(slow)).run(text="a", timeout=0.05)
match timed_out.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # run_timed_out
        print(cause.msg)  # The run took too long and was stopped; run.resume() carries on.
```

**Fix.** `resume()` carries on, calling nothing again that finished. Give a longer `timeout`, or none.

Its `ctx` keys are in the reference: `python -m conductor.about run-timed-out`.

### `timeout`

```python
@registry.node("stuck", title="Stuck", description="Waits too long, then hands back the text.", policy=Policy(attempt_timeout=0.05))
def stuck(text: str) -> str:
    time.sleep(0.2)
    return text


late = registry.load(make_graph(stuck)).run(text="a")
match late.state:
    case EndedEarly(cause=cause):
        print(cause.type)  # timeout
        print(cause.msg)  # The node did not answer in time; its Policy(attempt_timeout=) sets how long it may take.
```

**Fix.** Raise the node's `Policy(attempt_timeout=)`, or make the call faster. The timeout is final for that call: it is not retried.

Its `ctx` keys are in the reference: `python -m conductor.about timeout`.

## Declaring a node

Raised as `NodeDeclarationError` when the module that declares the node is imported. `error.code` is the code. The text is English and for the developer. Each example prints the code, then the text.

### `arriving_on_compute_added_inputs`

```python
def added(value: type) -> dict[str, type]:
    return {"extra": value}


try:
    @node("adds", title="Adds", description="Passes the text on as it is.", compute_added_inputs=added)
    def adds(value: str, **extra: str) -> str:
        return value
except NodeDeclarationError as error:
    print(error.code)  # arriving_on_compute_added_inputs
    print(error)  # @node('adds') on adds: compute_added_inputs parameter 'value' reads an arriving type, but inputs are decided before anything arrives; read it in compute_outputs
```

**Fix.** Read the arriving type in `compute_outputs`. The added inputs are decided before anything arrives.

### `async_run`

```python
try:
    @node("fetch_async", title="Fetch async", description="Fetches a page from an address.")
    async def fetch_async(url: str) -> str:
        return url
except NodeDeclarationError as error:
    print(error.code)  # async_run
    print(error)  # @node('fetch_async') on fetch_async: a node is a plain def; the engine gives a slow one threads with Policy(concurrency=)
```

**Fix.** Write a plain `def`. For a slow call, `Policy(concurrency=8)` makes several calls at once on threads.

### `bound_invalid`

```python
try:
    @node("short", title="Short", description="Passes the text on as it is.")
    def short(text: Annotated[str, Param(ge=0)]) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # bound_invalid
    print(error)  # @node('short') on short: text: ['ge'] can't apply to str, as pydantic holds no such limit on it; take them off, or check the value in the node

try:
    @node("narrow", title="Narrow", description="Passes the number on as it is.")
    def narrow(number: Annotated[int, Param(ge="a")]) -> int:  # type: ignore[arg-type]
        return number
except NodeDeclarationError as error:
    print(str(error).splitlines()[0])  # @node('narrow') on narrow: number: the limits {'ge': 'a'} don't fit int, so change one: Error building "lax-or-strict" validator:
```

**Fix.** Use limits pydantic holds on the type: `ge`, `gt`, `le`, `lt` and `multiple_of` on numbers (a `Decimal` too), dates, times and durations; `min_length` and `max_length` on texts, bytes, lists, tuples, sets and dicts; `pattern` on texts. A table takes none. The same holds for a `Param` on a field of a model a node takes.

### `choices_invalid`

```python
from typing import Literal

from conductor import Choice

try:
    @node("size", title="Size", description="Hands back the chosen size.")
    def size(pick: Annotated[Literal["s", "m"], Param(choices=[Choice("s", "Small")])]) -> str:
        return pick
except NodeDeclarationError as error:
    print(error.code)  # choices_invalid
    print(error)  # @node('size') on size: pick: on a Literal the choices title its values, one each: ['s', 'm']; give each choice once, as a value of the field's type

try:
    @node("doubled", title="Doubled", description="Hands back the chosen size.")
    def doubled(pick: Annotated[str, Param(choices=[Choice("s", "Small"), Choice("s", "Short")])]) -> str:
        return pick
except NodeDeclarationError as error:
    print(error)  # @node('doubled') on doubled: pick: the choices hold an id twice, or none; give each choice once, as a value of the field's type

try:
    @node("number_size", title="Number size", description="Hands back the chosen size.")
    def number_size(pick: Annotated[int, Param(choices=[Choice("s", "Small")])]) -> int:
        return pick
except NodeDeclarationError as error:
    print(error)  # @node('number_size') on number_size: pick: a choice is not a value of the field's type: Input should be a valid integer; give each choice once, as a value of the field's type

try:
    @node("object_size", title="Object size", description="Hands back the chosen size.")
    def object_size(pick: Annotated[str, Param(choices=[Choice(object(), "Small")])]) -> str:
        return pick
except NodeDeclarationError as error:
    print(error)  # @node('object_size') on object_size: pick: a choice has no JSON form; give each choice once, as a value of the field's type
```

**Fix.** Give each choice once, each a value of the field's type. On a `Literal`, the choices title its values, one each: `[Choice("s", "Small"), Choice("m", "Medium")]`.

### `compute_added_inputs_without_var_keyword`

```python
def operands(count: int) -> dict[str, type]:
    return {f"value_{i}": float for i in range(count)}


try:
    @node("sum_of", title="Sum of", description="Adds up numbers.", compute_added_inputs=operands)
    def sum_of(count: int) -> float:
        return 0.0
except NodeDeclarationError as error:
    print(error.code)  # compute_added_inputs_without_var_keyword
    print(error)  # @node('sum_of') on sum_of: a node with compute_added_inputs takes them as **kwargs; add **values to its signature
```

**Fix.** Take the added inputs as `**values: float`.

### `default_invalid`

```python
try:
    @node("maybe", title="Maybe", description="Passes the text on as it is.")
    def maybe(text: str = None) -> str:  # type: ignore[assignment]
        return text
except NodeDeclarationError as error:
    print(error.code)  # default_invalid
    print(error)  # @node('maybe') on maybe: text: the default None is a blank; write its type with | None

try:
    @node("counted", title="Counted", description="Passes the number on as it is.")
    def counted(number: int = "many") -> int:  # type: ignore[assignment]
        return number
except NodeDeclarationError as error:
    print(error)  # @node('counted') on counted: number: the default 'many' is not a value of int that it writes as JSON, so give one that is: Input should be a valid integer, unable to parse string as an integer
```

**Fix.** Write `text: str | None = None` for an input that takes a blank, or give a default of the input's type that it writes as JSON: an editor shows it, and a saved graph stores it.

### `duplicate_node_id` (declaring a node)

```python
twice = NodeRegistry()
twice.add(upper)
try:
    @twice.node("upper", title="Uppercase", description="Turns a text into capital letters.")
    def upper_again(text: str) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # duplicate_node_id
    print(error)  # two nodes have the id 'upper': doc_docs_problems_md.upper and doc_docs_problems_md.upper_again; ids are unique, so swap one for the other with registry.replace(...)
```

**Fix.** For a node the registry already holds, swap one for the other with `registry.replace(newer)`, or give the new node an id of its own. The older versions of a chain are plain `@node` definitions that are never registered; only the newest is declared on the registry, with `replaces=`.

### `duplicate_type_id`

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
    print(error)  # type id 'money' is declared by both doc_docs_problems_md.Amount and doc_docs_problems_md.Price; one registry holds one type per id
```

**Fix.** Give one of the classes another `type_id`.

### `empty_title_or_description`

```python
try:
    @node("shout", title="Shout", description="")
    def shout(text: str) -> str:
        return text.upper()
except NodeDeclarationError as error:
    print(error.code)  # empty_title_or_description
    print(error)  # @node('shout') has an empty description; write the words a person reads for the node in a palette
```

**Fix.** Give the node a title and a description: they are what a person reads for it in a palette.

### `exclusive_group_on_input`

```python
try:
    @node("branchy", title="Branchy", description="Passes the text on as it is.")
    def branchy(text: Annotated[str, Param(exclusive_group="x")]) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # exclusive_group_on_input
    print(error)  # @node('branchy') on branchy: text: exclusive_group is for outputs; put it on an output's Param
```

**Fix.** Put `exclusive_group` on the outputs that are branches, the fields of the frozen dataclass the node returns.

### `from_run_misplaced`

```python
class Session: ...

try:
    @node("maybe", title="Maybe", description="Passes the text on as it is.")
    def maybe(session: FromRun[Session] | None = None) -> str:
        return ""
except NodeDeclarationError as error:
    print(error.code)  # from_run_misplaced
    print(error)  # @node('maybe') on maybe: session: FromRun is a whole parameter the host always gives, name: FromRun[Session]; anywhere else, a caller would fill it
```

**Fix.** Write the parameter as `name: FromRun[Session]`, never inside `| None`, a list or a return type. The host
gives that value on every run, so a node can trust it; anywhere else it would be an input a caller fills.

### `from_run_not_a_class`

```python
try:
    @node("listed", title="Listed", description="Hands back an empty text.")
    def listed(names: FromRun[list[str]]) -> str:
        return ""
except NodeDeclarationError as error:
    print(error.code)  # from_run_not_a_class
    print(error)  # @node('listed') on listed: names: FromRun names the class of the value the run is given, FromRun[Session]
```

**Fix.** Name a class, `FromRun[Session]`: the run is given its value by that class, `from_run={Session: ...}`.

### `hook_parameter_unknown`

```python
def from_names(names: str) -> dict[str, type]:
    return {name: str for name in names.split(",")}


try:
    @node("named", title="Named", description="Gives one answer per label.", compute_outputs=from_names)
    def named(labels: str) -> dict[str, str]:
        return {}
except NodeDeclarationError as error:
    print(error.code)  # hook_parameter_unknown
    print(error)  # @node('named') on named: compute_outputs parameter 'names' names no input of the node; it has ['labels']


def from_any(*names: str) -> dict[str, type]:
    return {}


try:
    @node("starred", title="Starred", description="Gives one answer per label.", compute_outputs=from_any)
    def starred(labels: str) -> dict[str, str]:
        return {}
except NodeDeclarationError as error:
    print(error)  # @node('starred') on starred: compute_outputs parameter 'names': a hook's parameters are passed by name; name each after an input of the node


def from_keywords(**names: str) -> dict[str, type]:
    return {}


try:
    @node("keyed", title="Keyed", description="Gives one answer per label.", compute_outputs=from_keywords)
    def keyed(labels: str) -> dict[str, str]:
        return {}
except NodeDeclarationError as error:
    print(error)  # @node('keyed') on keyed: compute_outputs parameter **names: a hook takes ** only as **names: type on compute_outputs, for the types arriving on the inputs compute_added_inputs added; annotate it type and give the node that hook, or name each parameter after an input
```

**Fix.** Name the hook's parameters after the node's inputs, and take them by name.

### `input_without_type`

```python
try:
    @node("untyped", title="Untyped", description="Passes the text on as it is.")
    def untyped(text) -> str:  # type: ignore[no-untyped-def]
        return str(text)
except NodeDeclarationError as error:
    print(error.code)  # input_without_type
    print(error)  # @node('untyped') on untyped: text: an input needs a type hint; write one, as it is the input's interface
```

**Fix.** Give the input a type hint. The hint is the input's interface.

### `invalid_node_id` (declaring a node)

```python
try:
    @node("a.b", title="Dotted id", description="Passes the text on as it is.")
    def dotted_id(text: str) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # invalid_node_id
    print(error)  # @node takes the node's id first, a non-empty string without '.', '/', '@' or '~', which mean something in an address or a waiting id: @node('upper'); got 'a.b'
```

**Fix.** Pick an id without `.`, `/`, `@` or `~`, and never an empty one. The same rule holds for an id in a record.

### `nested_series`

```python
try:
    @node("nested", title="Nested", description="Wraps a text in lists.")
    def nested(text: str) -> Series[Series[str]]:
        return [[text]]
except NodeDeclarationError as error:
    print(error.code)  # nested_series
    print(error)  # @node('nested') on nested: result: a Series of Series; a field holds one level of items, so return one Series
```

**Fix.** Return one `Series`. A node after it that runs once per item and returns a `Series` of its own makes the next level.

### `output_without_type`

```python
try:
    @node("silent", title="Silent", description="Passes the text on as it is.")
    def silent(text: str):  # type: ignore[no-untyped-def]
        return text
except NodeDeclarationError as error:
    print(error.code)  # output_without_type
    print(error)  # @node('silent') on silent: a node needs a return type; write one, or -> None for no outputs
```

**Fix.** Write the return type. `-> None` declares no outputs.

### `outputs_not_frozen`

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
    print(error)  # @node('parts') on parts: a dataclass of outputs is frozen; write @dataclass(frozen=True)
```

**Fix.** Write `@dataclass(frozen=True)`. Several calls in a run can be given the same value.

### `parameter_not_by_name`

```python
try:
    @node("positional", title="Positional", description="Passes the text on as it is.")
    def positional(text: str, /) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # parameter_not_by_name
    print(error)  # @node('positional') on positional: text: a node's inputs are passed by name; drop the / and *args
```

**Fix.** Drop the `/` and `*args`: the engine passes every input by its name.

### `replaces_another_node`

```python
try:
    @node("lower", title="Lower", description="Turns a text into small letters.", replaces=upper)
    def lower(text: str) -> str:
        return text.lower()
except NodeDeclarationError as error:
    print(error.code)  # replaces_another_node
    print(error)  # @node('lower') replaces FunctionDefinition(id='upper', version=1, title='Uppercase'); a node replaces an earlier node of its own id
```

**Fix.** A version replaces an earlier `@node` of its own id. A different node gets no `replaces=`.

### `type_not_carried`

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
    print(error)  # @node('plain') on plain: value: conductor can't check or write plain as JSON; use a type pydantic knows, such as a pydantic model or a dataclass, or give the class its own __get_pydantic_core_schema__
```

**Fix.** Use a type pydantic can check and write as JSON: a pydantic model, a dataclass or a builtin. A class of your own that subclasses a builtin (`class Text(str)`) says how pydantic reads it, in a `__get_pydantic_core_schema__` of its own. [Writing nodes](writing-nodes.md#a-text-or-a-number-of-your-own) has the three lines.

### `type_variable_constrained`

```python
try:
    @node("hand-on", title="Hand on", description="Hands a value on.")
    def hand_on[T: (int, str)](value: T) -> T:
        return value
except NodeDeclarationError as error:
    print(error.code)  # type_variable_constrained
    print(error)  # @node('hand-on') on hand_on: value: the type variable T is constrained to a few types; give it a bound, or write the union
```

**Fix.** Give the variable a bound (`T: int | str`), or write the union in the signature. A bound says what may arrive; the variable is then whatever does.

### `type_variable_only_in_return`

```python
try:
    @node("make", title="Make", description="Makes an empty list.")
    def make[T](text: str) -> list[T]:
        return []
except NodeDeclarationError as error:
    print(error.code)  # type_variable_only_in_return
    print(error)  # @node('make') on make: result: the type variable T is on no input, so no edge can say what it is; put it on an input
```

**Fix.** A type variable is what arrives on an edge, so it stands on an input. An output whose type depends on a setting, not on an edge, is typed by a `compute_outputs` hook.

### `upgrade_missing`

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
    print(error)  # @node('hello') on hello: the version it replaces has ['name'], gone from this one; its Upgrade says where they went
```

**Fix.** Say where the gone field went: `upgrade=Upgrade(renamed_inputs={"name": "first"})`, or a `rewrite=` for an input; a gone output takes `renamed_outputs=`, since a rewrite reads only what is typed into the node.

### `upgrade_without_replaces`

```python
try:
    @node("hi", title="Hi", description="Says hi to a person.", upgrade=Upgrade(renamed_inputs={"name": "first"}))
    def hi(first: str) -> str:
        return first
except NodeDeclarationError as error:
    print(error.code)  # upgrade_without_replaces
    print(error)  # @node('hi') has an upgrade but replaces nothing; name the version it upgrades with replaces=
```

**Fix.** Name the version it upgrades with `replaces=`.

### `var_keyword_without_hook`

```python
try:
    @node("loose", title="Loose", description="Hands back an empty text.")
    def loose(**values: str) -> str:
        return ""
except NodeDeclarationError as error:
    print(error.code)  # var_keyword_without_hook
    print(error)  # @node('loose') on loose: **kwargs takes the inputs compute_added_inputs adds; the node has no such hook, so remove **kwargs or add compute_added_inputs=
```

**Fix.** Remove `**values`, or add the hook that adds those inputs, `compute_added_inputs=`.

### `version_not_after`

```python
try:
    @node("hello", title="Hello again", description="Says hello to a person.", replaces=hello_v1, version=1)
    def hello_again(name: str) -> str:
        return name
except NodeDeclarationError as error:
    print(error.code)  # version_not_after
    print(error)  # @node('hello') is version 1, not after the version 1 it replaces; leave version= out, or give one above 1
```

**Fix.** Leave `version=` out (one past the version replaced), or give a number above it.

### `widget_outside_param`

```python
from conductor.widgets import Textarea

try:
    @node("note", title="Note", description="Hands a note on.")
    def note(text: Annotated[str, Textarea()]) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # widget_outside_param
    print(error)  # @node('note') on note: text: a widget in Annotated goes on Param: Annotated[..., Param(widget=...)]
```

**Fix.** Put the widget on the field's `Param`: `Annotated[str, Param(widget=Textarea())]`. Other metadata in `Annotated` is pydantic's and is read as pydantic reads it; a widget there would be passed by, and the field would show no control.

## `check_registry`

Returned by `conductor.testing.check_registry`, in a test run. A host never meets these, and they aren't in `PROBLEMS.codes`. What each one's `ctx` holds is in the reference, `python -m conductor.about conductor-testing`.

### `changed_its_inputs`

```python
@node("sort_in_place", title="Sort", description="Sorts a list of texts.")
def sort_in_place(texts: list[str]) -> list[str]:
    texts.sort()
    return texts


checked = NodeRegistry()
checked.add(sort_in_place)
print([p.type for p in check_registry(checked, examples={"sort_in_place": [{"texts": ["b", "a"]}]})])  # ['changed_its_inputs']
print([p.msg for p in check_registry(checked, examples={"sort_in_place": [{"texts": ["b", "a"]}]})][0])  # the body changed 'texts', a value it was given; several calls of a run can be given it, so copy the value before changing it
```

**Fix.** Copy the value before changing it, `sorted(texts)`. Several calls in a run can be given the same value.

### `example_for_no_node`

```python
checked = NodeRegistry()
checked.add(upper)
print([p.type for p in check_registry(checked, examples={"uper": [{"text": "a"}]})])  # ['example_for_no_node']
print([p.msg for p in check_registry(checked, examples={"uper": [{"text": "a"}]})][0])  # 'uper' names no node of the registry; check the id, or remove the example
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
print([p.msg for p in check_registry(checked)][0])  # a hook failed on the node's defaults and made-up values: The node could not work out its fields from what it was given: 'x'. Check what is typed into the node and what is wired to it.
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
print([p.msg for p in check_registry(checked)][0])  # its hooks read ['code'], and no value could be made up; give an example
```

**Fix.** Give the node an example, `examples={"coded": [{"code": "1234"}]}`.

### `run_failed`

```python
import pytest

only_divide = registry.filter(lambda definition: definition.id == "divide")
print([p.type for p in check_registry(only_divide, examples={"divide": [{"n": 0}]})])  # ['run_failed']
print([p.msg for p in check_registry(only_divide, examples={"divide": [{"n": 0}]})][0])  # the example ended failed: the node raised ZeroDivisionError


@pytest.mark.xfail(strict=True, reason="divide by zero is not handled yet")
def test_divide_by_zero_is_known():
    assert check_registry(only_divide, examples={"divide": [{"n": 0}]}) == []
```

**Fix.** Fix the node or the example. `ctx.cause` has the run's own problem, and the message names the exception class a body raised. A node that is known to fail goes in a test of its own, checked down to that node with `registry.filter` as above; `xfail(strict=True)` turns the run red once the node starts passing.

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
print([p.msg for p in check_registry(checked)][0])  # moving to version 2 raised KeyError: 'first'
```

**Fix.** Fix the `Upgrade`, so a node saved at the older version loads cleanly at the newer one.

## The runs router

Written by `conductor_providers.fastapi.run_router` itself, beside the codes of the runs it serves: the `detail` of a 404 or a 409 is a list of these, `answer_check_failed` is in a run's `refused`, and `leg_failed` is a run's `cause`. `loc` is `()`: each is about the run as a whole. They aren't in `PROBLEMS.codes`: `conductor_providers.fastapi.ROUTER_PROBLEMS` is their catalogue, and each kind in it has its `code`, its `sentence` and the type of its `ctx`, as the kinds of `PROBLEMS` have. The examples below print each kind's sentence from it.

### `no_such_run`

```python
from conductor_providers.fastapi import ROUTER_PROBLEMS

router_sentence = {kind.code: kind.sentence for kind in ROUTER_PROBLEMS}
print("no_such_run:", router_sentence["no_such_run"])  # no_such_run: There is no run by that id that the caller may see; check the id and who is calling.
```

A 404 from every route under `/runs/{run_id}`: no run has the id, or `may_see` says the caller may not see it. The two read the same, so a caller learns nothing about a run that isn't theirs.

**Fix.** Use the `id` that `POST /runs` gave, as the caller who started the run or one `may_see` lets in.

### `already_answered`

```python
print("already_answered:", router_sentence["already_answered"])  # already_answered: An answer to '{waiting_id}' already waits for the running leg to end; read the run once it has, and answer what still waits.
```

A 409 from `POST /runs/{run_id}/answers` while a leg runs: an earlier call queued an answer for the same waiting id, and nothing of this call was queued. `ctx.waiting_id` is the id; there is one problem for each id sent twice.

**Fix.** Wait for the leg to end. The queued answer is taken then, or is in `refused` with why; answer what still waits.

### `answer_check_failed`

```python
print("answer_check_failed:", router_sentence["answer_check_failed"])  # answer_check_failed: The answer queued for '{waiting_id}' could not be checked: its type's own check raised {error}; the operator finds why in the server log. Send it again once that is fixed.
```

In a run's `refused`, under the waiting id: an answer queued while a leg ran was given to the run when the leg ended, and the check of the answer's type raised something that is no refusal of the value (a `KeyError` in a host type's validator, where a `ValueError` would refuse it). `ctx.waiting_id` is the id and `ctx.error` the class of what was raised. The leg's end is written as it stands and the question still waits. The same fault on an answer sent while no leg runs is the request's own 500.

**Fix.** The fault is in the type: the traceback is in the log `conductor_providers.fastapi`. Answer again once it is fixed.

### `run_complete`

```python
print("run_complete:", router_sentence["run_complete"])  # run_complete: The run is complete, so nothing in it waits for an answer; start a new run to ask again.
```

A 409 from `POST /runs/{run_id}/answers` on a run whose status is `complete`.

**Fix.** Nothing to answer. `POST /runs/{run_id}/again` starts a new run with some inputs changed.

### `run_not_resumable`

```python
print("run_not_resumable:", router_sentence["run_not_resumable"])  # run_not_resumable: The run is {status}, and only a run whose last leg ended early is resumed; read the run to see what it needs.
```

A 409 from `POST /runs/{run_id}/resume` on a run that is `running`, `waiting` or `complete`. `ctx.status` is the run's status.

**Fix.** A `waiting` run goes on when it is answered, a `running` one is already going on, and a `complete` one is done.

### `run_changed`

```python
print("run_changed:", router_sentence["run_changed"])  # run_changed: The run changed since it was read, so nothing was done; read it again and repeat the call if it still applies.
```

A 409 from `POST /runs/{run_id}/answers` or `/resume`: between the read of the run and the write that claims it, another call (an answer, a resume, on any worker) changed the run. One of the two calls wins and the other is told so; no node is called twice.

**Fix.** `GET /runs/{run_id}`, and send the call again if the run still needs it.

### `leg_failed`

```python
print("leg_failed:", router_sentence["leg_failed"])  # leg_failed: The run's worker failed before the leg could end; its operator finds why in the server log. Resume it to carry on.
```

The `cause` of a run that reads `failed` because its worker raised outside any node in a way that would raise again and that conductor has no problem of its own for: `from_run` raised, the function that gives a caller's registry raised. A leg conductor itself refused has conductor's problem as its cause (`from_run_missing`, `state_unreadable`, `state_unwritable`). A raise from the store is not one: it fails no run, and the run is gone on with once its lease runs out. It is in `GET /runs/{run_id}` and on the leg's `run_end` event. The run is kept as its last checkpoint, and answers queued during the leg stay queued until a leg ends with a run. `waiting` lists what waits in that checkpoint, the run an answer is given to; nothing, when the run could not be loaded.

**Fix.** `POST /runs/{run_id}/resume`. The exception is in the log `conductor_providers.fastapi`, with the run's id.
