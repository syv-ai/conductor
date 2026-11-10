# Problem codes

Everything conductor reports has a stable code. A `Problem`'s `type` is one, and so is a `NodeDeclarationError`'s `code`. This page has each code: where you meet it, an example that makes it, and the fix. A host that translates messages keys on the code and fills its sentence from `ctx`. `PROBLEM_CODES` holds every code a `Problem` can have that conductor writes itself; a code not in it is a node's own, from `NodeError(code=)`.

Every example on this page runs after this block. `problems(drawing)` loads a drawing and lists the codes of what is wrong with it; `refused(call)` makes a call and lists the codes it was refused for.

```python
import asyncio
import time
from dataclasses import dataclass
from typing import Annotated

from conductor import (
    Ask, ExternalFailure, FromRun, GraphRecord, InvalidCallError, NodeDeclarationError,
    NodeError, NodeRegistry, Param, Policy, Series, Upgrade, make_graph, node,
)
from conductor.testing import check_registry

registry = NodeRegistry()


@registry.node("upper")
def upper(text: str) -> str:
    return text.upper()


@registry.node("greet")
def greet(name: str, greeting: str) -> str:
    return f"{greeting} {name}"


@registry.node("count")
def count(n: int) -> int:
    return n


@registry.node("split")
def split(text: str) -> Series[str]:
    return text.split()


@registry.node("join")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@dataclass(frozen=True)
class Route:
    short: Annotated[str | None, Param(exclusive_group="length")] = None
    long: Annotated[str | None, Param(exclusive_group="length")] = None


@registry.node("route")
def route(text: str) -> Route:
    return Route(short=text) if len(text) < 4 else Route(long=text)


@registry.node("approve")
def approve(amount: int) -> bool:
    raise Ask(f"Approve {amount} kr.?")


@dataclass(frozen=True)
class Caller:
    name: str


@registry.node("who")
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

### `alternatives_on_different_rows`

> '{field}' takes alternatives from rows of different tables, {tables}, so a row of one has no place for the other's value. Bring them onto rows of one table first.

```python
print(problems({
    "nodes": [
        {"id": "s", "node": "split"}, {"id": "r", "node": "route"},
        {"id": "words", "node": "split"}, {"id": "u", "node": "upper"},
    ],
    "edges": [
        ["s.result", "r.text"], ["r.short", "words.text"],
        ["words.result", "u.text"], ["r.long", "u.text"],
    ],
}))  # ['alternatives_on_different_rows']
```

**Fix.** Bring the branches onto rows of one table first, or feed the node from one of them.

### `columns_missing`

> '{node_id}.{field}' takes the columns {missing}, which nothing wired into it from {sources} gives; wire them in, or take a Table of any columns.

```python
from typing import TypedDict

import polars as pl
from conductor import Table


class Named(TypedDict):
    name: str


class Invoice(TypedDict):
    name: str
    amount: int


@registry.node("names")
def names(text: str) -> Table[Named]:
    return pl.DataFrame({"name": text.split()})


@registry.node("total")
def total(invoices: Table[Invoice]) -> int:
    return int(invoices["amount"].sum())


print(problems({
    "nodes": [{"id": "n", "node": "names"}, {"id": "t", "node": "total"}],
    "edges": [["n.result", "t.invoices"]],
}))  # ['columns_missing']
```

**Fix.** Wire in a table or columns that give every column the input names, or type the input as a bare `Table`, which takes any columns.

### `compute_hook_failed`

> The node could not work out its fields from what it was given: {reason}. Check what is typed into the node.

```python
def by_names(names: str) -> dict[str, type]:
    if not names:
        raise ValueError("no names")
    return {name: str for name in names.split(",")}


@registry.node("pick", compute_outputs=by_names)
def pick(names: str) -> dict[str, str]:
    return {name: name for name in names.split(",")}


print(problems({"nodes": [{"id": "pick", "node": "pick", "values": {"names": ""}}]}))  # ['compute_hook_failed']
```

**Fix.** Type in a value the node's hook can work its fields out from. The hook's own exception is in the `conductor` log.

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

**Fix.** Write it as a value (`values`), which fixes it. Only an open input of a first node is the caller's to fill, so only there does a default mean anything.

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

### `embedded_graph_broken`

> The graph '{graph}' has {count} problem(s); fix them in that graph before it can be used here.

```python
inner = node("inner")(GraphRecord.model_validate({"nodes": [{"id": "x", "node": "ghost"}]}))
with_inner = NodeRegistry()
with_inner.add_subregistry(registry)
with_inner.add(inner)
print([p.type for p in with_inner.load({"nodes": [{"id": "i", "node": "inner"}]}).problems])  # ['embedded_graph_broken']
```

**Fix.** Fix the embedded graph: `ctx` holds its own problems. Load it on its own to see them in place.

### `embeds_itself`

> The node '{node}' is a graph that contains itself; embed a graph that does not contain this one.

```python
inner_record = GraphRecord.model_validate({"nodes": [{"id": "again", "node": "itself"}]})
itself = node("itself")(inner_record)
looping = NodeRegistry()
looping.add(itself)
broken = looping.load({"nodes": [{"id": "i", "node": "itself"}]}).problems[0]
print(broken.type, [p["type"] for p in broken.ctx["problems"]])  # embedded_graph_broken ['embeds_itself']
```

**Fix.** A stored graph can't contain itself, directly or through another. Remove the node that leads back. From outside, the graph that uses it reports `embedded_graph_broken`, with this problem in its `ctx`.

### `hook_needs_value`

> '{field}' must be typed in: the node works out its fields from it when the graph loads.

```python
print(problems({"nodes": [{"id": "pick", "node": "pick"}]}))  # ['hook_needs_value']
```

**Fix.** Type a value into the input the hook reads. The node's fields depend on it, so it is needed before the graph runs.

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

**Fix.** Type in a value of the field's type. `reason` says what the type refused, without quoting the value.

### `many_not_open`

> '{field}' is not an open input of the graph, so it takes no list from the caller; take '{field}' out of many.

```python
print(problems({
    "nodes": [{"id": "a", "node": "upper"}, {"id": "g", "node": "greet", "many": ["greeting"]}],
    "edges": [["a.result", "g.name"]],
}))  # ['many_not_open', 'unbound_on_later_node']
```

**Fix.** Take the input out of `many`, or make the node a first node. Only a caller gives a list, and a caller fills only a first node's open inputs.

### `many_on_edge`

> '{field}' is fed by an edge; only an open input takes one or many values from the caller, so take '{field}' out of many.

```python
print(problems({
    "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper", "many": ["text"]}],
    "edges": [["a.result", "b.text"]],
}))  # ['many_on_edge']
```

**Fix.** Take the input out of `many`. What an edge brings decides how many values it is.

### `many_on_series`

> '{field}' already takes many values; take '{field}' out of many.

```python
print(problems({"nodes": [{"id": "j", "node": "join", "many": ["texts"]}]}))  # ['many_on_series']
```

**Fix.** Take the input out of `many`: a `Series` input already takes a list.

### `parameter_name_invalid`

> '{field}' cannot be a field name: '.', '/' and '@' each mean something in an address, and a name can't be empty; rename the field.

```python
def dotted(names: str) -> dict[str, type]:
    return {name: str for name in names.split(",")}


@registry.node("fields", compute_outputs=dotted)
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

### `rows_not_related`

> {tables} are rows of different tables; join them on a key in a node, and wire the join.

```python
print(problems({
    "nodes": [{"id": "s1", "node": "split"}, {"id": "s2", "node": "split"}, {"id": "g", "node": "greet"}],
    "edges": [["s1.result", "g.name"], ["s2.result", "g.greeting"]],
}))  # ['rows_not_related']
```

**Fix.** Rows of two tables never meet by themselves. Write a node that joins them on a key, and wire the join.

### `series_inputs_never_share_a_row`

> {inputs} read branches that never both have a value on a row, so no row would reach the node. Write Series[X | None] on them to keep every row, or take the branches through one input.

```python
@registry.node("pair")
def pair(a: Series[str], b: Series[str]) -> str:
    return f"{a} {b}"


print(problems({
    "nodes": [{"id": "s", "node": "split"}, {"id": "r", "node": "route"}, {"id": "p", "node": "pair"}],
    "edges": [["s.result", "r.text"], ["r.short", "p.a"], ["r.long", "p.b"]],
}))  # ['series_inputs_never_share_a_row']
```

**Fix.** Write `Series[X | None]` on both inputs to keep every row, blanks included, or take the branches through one input.

### `setting_not_set`

> '{field}' is a setting with no default; type a value into it.

```python
@registry.node("model")
def model(text: str, name: Annotated[str, Param(setting=True)]) -> str:
    return f"{name}: {text}"


print(problems({"nodes": [{"id": "m", "node": "model"}]}))  # ['setting_not_set']
```

**Fix.** Type a value into it. A setting is the graph author's to set, so a caller never fills it.

### `several_edges_into_one_value`

> '{field}' takes one value, and more than one of the edges into it from {sources} can hold one on a row. Feed it from opposite branches of one decision, or take them through a Series or Table input.

```python
print(problems({
    "nodes": [{"id": "a", "node": "upper"}, {"id": "b", "node": "upper"}, {"id": "c", "node": "upper"}],
    "edges": [["a.result", "c.text"], ["b.result", "c.text"]],
}))  # ['several_edges_into_one_value']
```

**Fix.** Feed it from opposite branches of one decision (outputs of one `exclusive_group`), or take the values through a `Series` input.

### `two_typed_in_lists`

> Only one list can be typed into a node, and {inputs} are all lists; type one in and feed the others through edges.

```python
print(problems({"nodes": [{
    "id": "g", "node": "greet",
    "values": {"name": ["Ida", "Bo"], "greeting": ["hi", "hello"]}, "many": ["name", "greeting"],
}]}))  # ['two_typed_in_lists']
```

**Fix.** Type one list in. Feed the other through an edge, or split the node in two.

### `type_mismatch`

> '{source}' gives {got}, but '{node_id}.{field}' takes {expected}; connect an output that gives {expected}, or convert it in a node between.

```python
print(problems({
    "nodes": [{"id": "c", "node": "count"}, {"id": "u", "node": "upper"}],
    "edges": [["c.result", "u.text"]],
}))  # ['type_mismatch']
```

**Fix.** Wire an output of the type the input takes, or put a node between that turns one into the other.

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

**Fix.** Wire from one of the addresses `ctx.outputs` lists.

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

> The question {waiting_id!r} was answered for has changed on its row; it is now {current!r}, so answer that one.

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

### `from_run_ambiguous`

> '{node_id}' needs a {needs} for '{parameter}', and from_run gives more than one ({given}); key the one to use by {needs}.

```python
@dataclass(frozen=True)
class Admin(Caller):
    pass


@dataclass(frozen=True)
class Guest(Caller):
    pass


print(refused(lambda: asking_who.run(from_run={Admin: Admin("Ida"), Guest: Guest("Bo")})))  # ['from_run_ambiguous']
```

**Fix.** Key the value to use by the class the node names: `from_run={Caller: ...}`.

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
```

**Fix.** Give a value of the input's type. `graph.inputs[name].describe()` has its schema.

### `invalid_max_rows`

> max_rows is at least 1; it was {max_rows}.

```python
print(refused(lambda: uppering.run(text="a", max_rows=0)))  # ['invalid_max_rows']
```

**Fix.** Give `max_rows` of 1 or more, or leave it out for no ceiling.

### `invalid_record`

> The record is not a graph record at '{at}': {reason}. Fix the record's JSON.

```python
print(refused(lambda: registry.load({"nodes": [{"id": "u", "type": "upper"}]})))  # ['invalid_record']
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

**Fix.** Pass `edges=[("upper.result", "greet.name")]`. `make_graph` chains nodes only when each takes one required open input from the one output before it.

### `not_offered`

> The graph has no input '{name}'; it has {inputs}.

```python
print(refused(lambda: uppering.run(txt="a")))  # ['not_offered']
```

**Fix.** Use one of the inputs `ctx.inputs` lists.

### `not_waiting`

> Nothing is waiting on an answer as {waiting_id!r}; run.waiting lists what is.

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
@registry.node("raw")
def raw(text: str) -> bytes:
    return b"\xff" + text.encode()


unwritable = registry.load(make_graph(raw)).run(text="a")
print(refused(lambda: unwritable.to_json()))  # ['state_unwritable']
```

**Fix.** Return a type pydantic writes as JSON. `ctx.at` names the output that couldn't be written.

## Why a node or a run stopped

In `run.cause` and on `node_failed`. `loc` is `(node_id, *row)`, or `()` for the run as a whole.

### `execution_failed`

> The node failed; its developer finds why in the conductor log.

```python
@registry.node("divide")
def divide(n: int) -> float:
    return 1 / n


failed = registry.load(make_graph(divide)).run(n=0)
assert failed.cause is not None
print(failed.cause.type)  # execution_failed
```

**Fix.** The node let out an exception of its own. Its class is `ctx.error`; the exception and its traceback are in the `conductor` log. Fix the node, then `resume()`.

### `external_failed`

> An outside service did not answer.

```python
@registry.node("fetch", policy=Policy(retry_on=(ConnectionError,)))
def fetch(url: str) -> str:
    raise ConnectionError(url)


cut_off = registry.load(make_graph(fetch)).run(url="https://example.com")
assert cut_off.cause is not None
print(cut_off.cause.type)  # external_failed
```

**Fix.** The outside world failed after every retry. `resume()` tries again; raise `Policy(retries=)` for more tries.

### `failed`

> The node reported a failure.

```python
@registry.node("read_pdf")
def read_pdf(path: str) -> str:
    raise NodeError("The document could not be read.")


unread = registry.load(make_graph(read_pdf)).run(path="a.pdf")
assert unread.cause is not None
print(unread.cause.type)  # failed
```

**Fix.** The node said why in `msg`. A node gives its own code with `NodeError(message, code=)`.

### `invalid_ask`

> The node asked a question its outputs can't be answered with: {reason}. Fix the node's Ask.

```python
@registry.node("confirm")
def confirm(text: str) -> bool:
    raise Ask("Sure?", default="maybe")


misasked = registry.load(make_graph(confirm)).run(text="a")
assert misasked.cause is not None
print(misasked.cause.type)  # invalid_ask
```

**Fix.** Ask with choices and a default the node's return type accepts. `ctx.reason` says which part doesn't fit.

### `invalid_output`

> The node returned a value that is not what it declares: {reason}. Fix what the node returns.

```python
@registry.node("guess")
def guess(text: str) -> int:
    return text  # type: ignore[return-value]


wrong = registry.load(make_graph(guess)).run(text="12")
assert wrong.cause is not None
print(wrong.cause.type)  # invalid_output
```

**Fix.** Return what the node declares. `"12"` is never an `int`: outputs are read strictly. `ctx.name` names the output.

### `row_ceiling`

> {table} has {rows} rows, more than the {max_rows} this run may make; resume it with a larger max_rows.

```python
splitting = registry.load(make_graph(split, upper))
capped = splitting.run(text="a b c", max_rows=2)
assert capped.cause is not None
print(capped.cause.type)  # row_ceiling
```

**Fix.** `capped.resume(max_rows=...)` with a higher ceiling carries on, calling nothing again.

### `run_cancelled`

> The run was cancelled. Resume it to carry on, or start a new run.

```python
async def cancelled() -> str:
    stop = asyncio.Event()
    stop.set()
    async for event in uppering.with_inputs(text="a").stream(cancel=stop):
        if event.type == "run_end":
            assert event.run.cause is not None
            return event.run.cause.type
    return ""


print(asyncio.run(cancelled()))  # run_cancelled
```

**Fix.** Store the run and `resume()` it to carry on. The calls made before the cancel are kept.

### `run_timed_out`

> The run took too long and was stopped; run.resume() carries on.

```python
@registry.node("slow")
def slow(text: str) -> str:
    time.sleep(0.2)
    return text


timed_out = registry.load(make_graph(slow)).run(text="a", timeout=0.05)
assert timed_out.cause is not None
print(timed_out.cause.type)  # run_timed_out
```

**Fix.** `resume()` carries on, calling nothing again that finished. Give a longer `timeout`, or none.

### `timeout`

> The node did not answer in time; its Policy(attempt_timeout=) sets how long it may take.

```python
@registry.node("stuck", policy=Policy(attempt_timeout=0.05))
def stuck(text: str) -> str:
    time.sleep(0.2)
    return text


late = registry.load(make_graph(stuck)).run(text="a")
assert late.cause is not None
print(late.cause.type)  # timeout
```

**Fix.** Raise the node's `Policy(attempt_timeout=)`, or make the call faster. The timeout is final for that row: it is not retried.

## Declaring a node

Raised as `NodeDeclarationError` when the module that declares the node is imported. `error.code` is the code. The text is English and for the developer.

### `annotated_extra_unknown`

> {name}: Annotated holds {unknown[0]}, which conductor doesn't read; a widget, a title or a limit goes on Param(...)

```python
try:
    @node("titled")
    def titled(text: Annotated[str, "Title"]) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # annotated_extra_unknown
```

**Fix.** Put a title, a widget or a limit on `Param(...)`: `Annotated[str, Param(title="Title")]`.

### `arriving_on_compute_added_inputs`

> compute_added_inputs parameter {name} reads an arriving type, but inputs are decided before anything arrives; read it in compute_outputs

```python
def added(value: type) -> dict[str, type]:
    return {"extra": value}


try:
    @node("adds", compute_added_inputs=added)
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
    @node("fetch_async")
    async def fetch_async(url: str) -> str:
        return url
except NodeDeclarationError as error:
    print(error.code)  # async_run
```

**Fix.** Write a plain `def`. For a slow call, `Policy(concurrency=8)` runs several rows at once on threads.

### `bound_invalid`

> {spec.name}: a table takes no limits

> {spec.name}: {sorted(limits)} can't apply to {getattr(base, '__name__', base)}; ge, gt, le, lt and multiple_of fit numbers and dates, min_length, max_length and pattern texts

> Param has no argument {unknown[0]}; it takes title, description, widget, exclusive_group, setting and the limits {sorted(_NUMERIC | _SIZED)}

> {spec.name}: the limits {spec.param.constraints} don't fit {getattr(base, '__name__', base)}: {error}

```python
try:
    @node("short")
    def short(text: Annotated[str, Param(ge=0)]) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # bound_invalid
```

**Fix.** Use limits that fit the type: `ge`, `gt`, `le`, `lt` and `multiple_of` for numbers and dates, `min_length`, `max_length` and `pattern` for text. A table takes none.

### `choices_declared_twice`

> {spec.name}: the choices are the Literal's values, so the widget declares none; write Dropdown()

```python
from typing import Literal

from conductor import Choice
from conductor.widgets import Dropdown

try:
    @node("size")
    def size(pick: Annotated[Literal["s", "m"], Param(widget=Dropdown(choices=(Choice("s", "Small"),)))]) -> str:
        return pick
except NodeDeclarationError as error:
    print(error.code)  # choices_declared_twice
```

**Fix.** Write `Dropdown()`: a `Literal` input's choices are its values. To give them titles, use `Choice`s on a `str` input instead.

### `compute_added_inputs_without_var_keyword`

> a node with compute_added_inputs takes them as **kwargs

```python
def operands(count: int) -> dict[str, type]:
    return {f"value_{i}": float for i in range(count)}


try:
    @node("sum_of", compute_added_inputs=operands)
    def sum_of(count: int) -> float:
        return 0.0
except NodeDeclarationError as error:
    print(error.code)  # compute_added_inputs_without_var_keyword
```

**Fix.** Take the added inputs as `**values: float`.

### `default_invalid`

> {spec.name}: the default None is a blank; write its type with | None

> {spec.name}: the default {spec.default} is not a {getattr(base, '__name__', base)}: {error}

```python
try:
    @node("maybe")
    def maybe(text: str = None) -> str:  # type: ignore[assignment]
        return text
except NodeDeclarationError as error:
    print(error.code)  # default_invalid
```

**Fix.** Write `text: str | None = None` for an input that takes a blank, or give a default of the input's type.

### `duplicate_node_id` (declaring a node)

> two nodes have the id {definition.id}: {_where_declared(held)} and {_where_declared(definition)}; ids are unique, so swap one for the other with registry.replace(...)

```python
twice = NodeRegistry()
twice.add(upper)
try:
    @twice.node("upper")
    def upper_again(text: str) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # duplicate_node_id
```

**Fix.** Swap one for the other with `registry.replace(definition)`, or give the new node an id of its own. A newer version of the same node names the older in `replaces=`.

### `duplicate_type_id`

> type id {type_id(held)} is declared by both {_qualified_name(existing)} and {_qualified_name(held)}; one registry holds one type per id

```python
class Amount(int):
    id = "money"


class Price(float):
    id = "money"


try:
    NodeRegistry(types=[Amount, Price])
except NodeDeclarationError as error:
    print(error.code)  # duplicate_type_id
```

**Fix.** Give one of the classes another `id`.

### `exclusive_group_on_input`

> {name}: exclusive_group is for outputs; put it on an output's Param

```python
try:
    @node("branchy")
    def branchy(text: Annotated[str, Param(exclusive_group="x")]) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # exclusive_group_on_input
```

**Fix.** Put `exclusive_group` on the outputs that are branches, the fields of the frozen dataclass the node returns.

### `extra_not_json`

> extra must be JSON, a mapping of str to JSON values with finite numbers; got {extra}

```python
try:
    @node("tagged", extra={"weight": float("nan")})
    def tagged(text: str) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # extra_not_json
```

**Fix.** Give `extra` as JSON: a mapping of strings to JSON values, every number finite.

### `from_run_not_a_class`

> {name}: FromRun names the class of the value the run is given, FromRun[Session]

```python
try:
    @node("listed")
    def listed(names: FromRun[list[str]]) -> str:
        return ""
except NodeDeclarationError as error:
    print(error.code)  # from_run_not_a_class
```

**Fix.** Name a class, `FromRun[Session]`: the run is given its value by that class, `from_run={Session: ...}`.

### `hook_parameter_unknown`

> {kind} parameter {name}: a hook's parameters are passed by name

> {kind} parameter {name} names no input of the node; it has {sorted(inputs)}

```python
def from_names(names: str) -> dict[str, type]:
    return {name: str for name in names.split(",")}


try:
    @node("named", compute_outputs=from_names)
    def named(labels: str) -> dict[str, str]:
        return {}
except NodeDeclarationError as error:
    print(error.code)  # hook_parameter_unknown
```

**Fix.** Name the hook's parameters after the node's inputs, and take them by name.

### `input_without_type`

> {name}: an input needs a type hint

```python
try:
    @node("untyped")
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
    @node("a.b")
    def dotted_id(text: str) -> str:
        return text
except NodeDeclarationError as error:
    print(error.code)  # invalid_node_id
```

**Fix.** Pick an id without `.`, `/`, `@` or `~`, and never an empty one. The same rule holds for an id in a record.

### `nested_series`

> {name}: a Series of Series; rows are never nested in a field, so return one Series

```python
try:
    @node("nested")
    def nested(text: str) -> Series[Series[str]]:
        return [[text]]
except NodeDeclarationError as error:
    print(error.code)  # nested_series
```

**Fix.** Return one `Series`. A node after it that runs per row makes the next level.

### `not_a_node_definition`

> {definition} is not a node; declare one with @node

```python
try:
    NodeRegistry().add(len)  # type: ignore[arg-type]
except NodeDeclarationError as error:
    print(error.code)  # not_a_node_definition
```

**Fix.** Declare the function with `@node("id")` first, and add what it returns.

### `not_a_type`

> {named} is not a type; pass the class itself

```python
try:
    NodeRegistry(types=["money"])  # type: ignore[list-item]
except NodeDeclarationError as error:
    print(error.code)  # not_a_type
```

**Fix.** Pass the class itself.

### `output_without_type`

> a node needs a return type

```python
try:
    @node("silent")
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
    @node("parts")
    def parts(name: str) -> Parts:
        return Parts(*name.split(" ", 1))
except NodeDeclarationError as error:
    print(error.code)  # outputs_not_frozen
```

**Fix.** Write `@dataclass(frozen=True)`. The rows of a run can share one value.

### `parameter_not_by_name`

> {name}: a node's inputs are passed by name

```python
try:
    @node("positional")
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
    @node("lower", replaces=upper)
    def lower(text: str) -> str:
        return text.lower()
except NodeDeclarationError as error:
    print(error.code)  # replaces_another_node
```

**Fix.** A version replaces an earlier `@node` of its own id. A different node gets no `replaces=`.

### `series_of_tables`

> {name}: a Series of tables; a table already holds many rows, so take or return one table

```python
import polars as pl

try:
    @node("frames")
    def frames(text: str) -> Series[pl.DataFrame]:
        return []
except NodeDeclarationError as error:
    print(error.code)  # series_of_tables
```

**Fix.** Return one table: it already holds many rows.

### `setting_on_output`

> {name}: a setting is an input the graph's author sets; an output is never one

```python
try:
    @node("chosen")
    def chosen(text: str) -> Annotated[str, Param(setting=True)]:
        return text
except NodeDeclarationError as error:
    print(error.code)  # setting_on_output
```

**Fix.** Take `setting=True` off the output. Only an input is a setting.

### `type_not_carried`

> {spec.name}: conductor can't check or write {getattr(base, '__name__', base)} as JSON; use a type pydantic knows, such as a pydantic model or a dataclass

```python
class Plain:
    pass


try:
    @node("plain")
    def plain(value: Plain) -> str:
        return ""
except NodeDeclarationError as error:
    print(error.code)  # type_not_carried
```

**Fix.** Use a type pydantic can check and write as JSON: a pydantic model, a dataclass, a `TypedDict` or a builtin.

### `upgrade_missing`

> the version it replaces has {sorted(unexplained)}, gone from this one; its Upgrade says where they went

```python
@node("hello")
def hello_v1(name: str) -> str:
    return name


try:
    @node("hello", replaces=hello_v1)
    def hello(first: str) -> str:
        return first
except NodeDeclarationError as error:
    print(error.code)  # upgrade_missing
```

**Fix.** Say where the gone field went: `upgrade=Upgrade(renamed_inputs={"name": "first"})`, or a `rewrite=`.

### `upgrade_without_replaces`

> @node({id}) has an upgrade but replaces nothing; name the version it upgrades with replaces=

```python
try:
    @node("hi", upgrade=Upgrade(renamed_inputs={"name": "first"}))
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
    @node("loose")
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
    @node("hello", replaces=hello_v1, version=1)
    def hello_again(name: str) -> str:
        return name
except NodeDeclarationError as error:
    print(error.code)  # version_not_after
```

**Fix.** Leave `version=` out (one past the version replaced), or give a number above it.

### `widget_not_json`

> Param's widget {widget} has no JSON form; give an instance, such as Textarea()

```python
from conductor.widgets import Textarea

try:
    @node("note")
    def note(text: Annotated[str, Param(widget=Textarea)]) -> str:  # type: ignore[arg-type]
        return text
except NodeDeclarationError as error:
    print(error.code)  # widget_not_json
```

**Fix.** Give an instance, `Textarea()`, not the class.

## `check_registry`

Returned by `conductor.testing.check_registry`, in a test run. A host never meets these, and they aren't in `PROBLEM_CODES`.

### `changed_its_inputs`

```python
@node("sort_in_place")
def sort_in_place(texts: list[str]) -> list[str]:
    texts.sort()
    return texts


checked = NodeRegistry()
checked.add(sort_in_place)
print([p.type for p in check_registry(checked, examples={"sort_in_place": [{"texts": ["b", "a"]}]})])  # ['changed_its_inputs']
```

**Fix.** Copy the value before changing it, `sorted(texts)`. The rows of a run can share one value.

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


@node("unnamed", compute_outputs=no_names)
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


@node("coded", compute_outputs=by_code)
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

**Fix.** Fix the node or the example. `ctx.cause` has the run's own problem.

### `upgrade_broken`

```python
def broken_rewrite(values: dict) -> dict:
    raise KeyError("first")


@node("hello", replaces=hello_v1, upgrade=Upgrade(rewrite=broken_rewrite))
def hello_v2(first: str) -> str:
    return first


checked = NodeRegistry()
checked.add(hello_v2)
print([p.type for p in check_registry(checked)])  # ['upgrade_broken']
```

**Fix.** Fix the `Upgrade`, so a node saved at the older version loads cleanly at the newer one.

### `waiver_not_needed`

```python
checked = NodeRegistry()
checked.add(upper)
print([p.type for p in check_registry(checked, expected_failures={"upper:run_failed": "flaky"})])  # ['waiver_not_needed']
```

**Fix.** Remove the waiver: the check it accepted passes.
