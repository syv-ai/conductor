# Writing nodes for others

This page is for you if you write nodes that other people place in graphs: a package of nodes, or the nodes your own editor offers. The [first page](https://github.com/syv-ai/conductor#readme) shows a host loading and running a graph. Here every example builds its own small graph with `make_graph`, so you can see what a node does in one.

`make_graph(a, b)` writes a graph record from Python. Each node goes under its own id, `upper-1` and `upper-2` when one is used twice. With no `edges`, the nodes form a chain: each takes its one required input from the one output of the node before it. `edges=[("a.result", "b.text")]` names the edges instead. `values={"a.text": "hi"}` types a value in, and `defaults=` gives one a caller may change. A list given to an input of one value runs the node once per item. Every key is a `"node.input"` address. A wrong address, a value of the wrong type or an edge between types that don't meet raises at the call.

The rules in full are in the reference: `python -m conductor.about`.

## A registry of your own

Keep your nodes in a registry. Its `category`, `tags` and `extra` are defaults for every node you add:

```python
from typing import Annotated

from conductor import NodeRegistry, Param, make_graph
from conductor.widgets import NumberInput, TextInput

text_nodes = NodeRegistry(category="text", tags=["acme"])


@text_nodes.node("repeat", extra={"docs_url": "https://example.com/repeat"})
def repeat(
    text: Annotated[str, Param(title="Text", widget=TextInput())],
    times: Annotated[int, Param(title="Times", ge=1, le=10, widget=NumberInput(integer_only=True))] = 2,
) -> str:
    """Repeats a text."""
    return text * times


print(repeat("ab"))  # abab
```

The host adds your registry to its own. Your nodes are copied in, with your defaults:

```python
host = NodeRegistry()
host.add_subregistry(text_nodes)

print(host["repeat"].category, host["repeat"].tags)  # text ('acme',)
```

A second node under an id the host already has raises. `extra` is your own JSON about a node. Conductor keeps it and shows it in `describe()`, and never reads it.

## What a person sees

`Annotated[..., Param(...)]` says what an editor shows for a field: a `title`, a `description` and a `widget`, the control to edit it with. Without a title, the field's name is used, in everyday words. The title of the node is its function's name, and its description is the docstring's first paragraph.

`Param` also takes pydantic's limits: `ge`, `gt`, `le`, `lt`, `multiple_of`, `min_length`, `max_length` and `pattern`. They are checked on every value someone types in or a caller gives:

```python
drawn = {"nodes": [{"id": "repeat", "node": "repeat", "values": {"times": 20}}]}  # as an editor saves it
graph = host.load(drawn)

for problem in graph.problems:
    print(problem.loc, problem.type)
# ('repeat', 'times') invalid_value
```

`make_graph(repeat, values={"repeat.times": 20})` raises that same problem at the call, as an `InvalidCallError`: from Python, the traceback points at the line you wrote.

`repeat.describe()` is the node as an editor's palette reads it, a `NodeDescription`, and `registry.describe()` is the whole palette, a `RegistryDescription`: pydantic records a host returns as they are, which `model_dump(mode="json")` writes as JSON.

### Widgets

No type has a default widget. A `str` may be one line, a text area or a dropdown, so each input names its own. An input with no widget gets no control: an edge fills it, or a value set in the graph. These ship in `conductor.widgets`: `TextInput`, `Textarea`, `Dropdown`, `NumberInput`, `Slider`, `Switch`, `DatePicker`, `FileUpload` and `ListInput`. What each is for and the options it takes are in the reference (`python -m conductor.about conductor-widgets`). A `Dropdown` takes no choices of its own: they are the field's, a `Literal`'s values or `Param(choices=[Choice("en", "English"), ...])`.

A widget is a pydantic model, and conductor never reads it: it passes it on to the editor. So a control of your own needs no conductor release. Subclass `Widget`, next to the component in your frontend that draws it:

```python
from typing import Literal

from conductor.widgets import Widget


class EntityDropdown(Widget):
    type: Literal["entity-dropdown"] = "entity-dropdown"
    entity_kind: str = ""


print(EntityDropdown(entity_kind="customer").model_dump(mode="json"))
# {'type': 'entity-dropdown', 'entity_kind': 'customer'}
```

Pass an instance, `widget=Textarea()`. A class passed bare, `widget=Textarea`, is refused when the node is declared.

## Types

Any type pydantic can check and write as JSON can be an input or an output: `str`, `int`, `float`, `bool`, `date`, a pydantic model, a dataclass, a subclass of `str`. A pydantic model is the usual way to carry a record of your own:

```python
from pydantic import BaseModel

from conductor import node


class Address(BaseModel):
    street: str
    city: str


@node("city")
def city(address: Address) -> str:
    return address.city


print(city(Address(street="Nørregade 1", city="Aarhus")))  # Aarhus
```

A class pydantic can't check, such as a plain class with only an `__init__`, is refused when the node is declared, as `type_not_carried`.


- `X | None` takes a blank. If an input's type has no `| None`, the node isn't called for a row where it is blank, and its outputs are blank there too. Rows a blank node never made are not zero rows: a `Series` or `Table` input that would gather them is blank in turn.
- `Series[X]` is many values of `X`, a plain list to your function. Wired to a column, it gets every value of it. Returned, each item is a row.
- `Table[R]` is a whole table, a `polars.DataFrame` whose columns the `TypedDict` `R` names. Each column of a table you return is an output of its own: `read.result.email`.
- `list[X]` is one value that happens to be a list.

A node typed for one value runs once per row when a column is wired to it. You never write the loop.

## Several outputs, and branches

Return a frozen dataclass for several outputs, one per field. Outputs that share an `exclusive_group` are branches: on each row one gets a value and the others get `SKIPPED`. A node fed only `SKIPPED` is skipped too:

```python
from dataclasses import dataclass

from conductor import SKIPPED, NodeRegistry


@dataclass(frozen=True)
class Routed:
    large: Annotated[int, Param(exclusive_group="size")]
    small: Annotated[int, Param(exclusive_group="size")]


registry = NodeRegistry()


@registry.node("route")
def route(amount: int) -> Routed:
    if amount >= 1000:
        return Routed(large=amount, small=SKIPPED)
    return Routed(large=SKIPPED, small=amount)


@registry.node("notify")
def notify(amount: int) -> str:
    return f"told the manager about {amount}"


graph = registry.load(make_graph(route, notify, edges=[("route.large", "notify.amount")]))
print(graph.run(amount=5000).results["notify"])  # {'result': 'told the manager about 5000'}
print(graph.run(amount=5).results["notify"])     # {'result': None}
```

A branch not taken reads as `None` in `run.results` and `run.outputs`.

## Asking a person

`raise Ask(prompt)` stops the node on that row and asks a person. Your return type says what they answer: `-> bool` asks a yes or no, and a frozen dataclass asks a form, one question per field, each field's default offered. What every type asks is in the reference (`python -m conductor.about`, "Legs and Asking").

`Ask` adds what only the run knows: `choices`, `default` and `widget`. On a node with one output, `choices` is a list:

```python
from conductor import Ask


@registry.node("assign")
def assign(ticket: str) -> str:
    raise Ask(f"Who takes {ticket}?", choices=["ida", "bo"], default="ida")


run = registry.load(make_graph(assign)).run(ticket="T-1")
print(run.waiting[0].questions[0].choices)  # [{'id': 'ida', 'title': 'ida'}, {'id': 'bo', 'title': 'bo'}]
```

A node with several outputs is answered by output name, so `choices`, `default` and `widget` go by output name too, naming only the outputs they are for. A `Choice(id, title)` gives a choice a title of its own:

```python
from conductor import Choice
from conductor.widgets import Textarea


@dataclass(frozen=True)
class Review:
    approved: bool
    owner: str
    note: str | None = None


@registry.node("review")
def review(proposal: str) -> Review:
    raise Ask(
        f"Review: {proposal}",
        choices={"owner": [Choice("ida", "Ida B."), Choice("bo", "Bo K.")]},
        widget={"note": Textarea()},
    )


run = registry.load(make_graph(review)).run(proposal="Move the server")
(waiting,) = run.waiting
print([question.name for question in waiting.questions])  # ['approved', 'owner', 'note']

run = run.answer(waiting, {"approved": True, "owner": "ida"}).resume()
print(run.outputs)  # {'review.approved': True, 'review.owner': 'ida', 'review.note': None}
```

The node isn't called again for that row: the answer is its outputs. An answer that doesn't fit its question is refused before anything runs. An `Ask` that doesn't fit your outputs, say a default your type refuses, fails the node as `invalid_ask`.

`Ask` is a `BaseException`, so an `except Exception` in your node lets it through.

## When something fails

Raise `NodeError` for a failure you can explain to a person. Its message reaches the host as written, and `code` is yours to choose:

```python
from conductor import NodeError


@registry.node("parse")
def parse(text: str) -> int:
    if not text.isdigit():
        raise NodeError("The text is not a number.", code="not_a_number")
    return int(text)


run = registry.load(make_graph(parse)).run(text="twelve")
assert run.cause is not None  # a run that failed, stopped or timed out says why
print(run.status, run.cause.type, run.cause.msg)  # failed not_a_number The text is not a number.
```

Any other exception fails the node too. Its traceback goes to the `conductor` logger, and only its class name reaches the host, so a value your node handled never leaks into an event.

Raise `ExternalFailure` when the outside world failed: a timeout, a rate limit, a server error. Only that is retried, under the node's `Policy`. A bug fails the same way every time, so it isn't retried:

```python
from conductor import ExternalFailure, Policy

attempts: list[str] = []


@registry.node("fetch", policy=Policy(retries=2, retry_delay=0.01))
def fetch(url: str) -> str:
    attempts.append(url)
    if len(attempts) < 3:
        raise ExternalFailure("The server did not answer.")
    return "<html>"


run = registry.load(make_graph(fetch)).run(url="https://example.com")
print(run.status, len(attempts))  # complete 3
```

`Policy` also sets `attempt_timeout`, `retry_on` (exception classes of your client to treat as external) and `concurrency`, how many rows are called at once on threads. By default a node is called one row at a time.

## What the host gives the run

Some values come from the host, not from the graph: who runs it, a database session, a client. Ask for them with `FromRun[T]`. The host passes them by type:

```python
from conductor import FromRun


class Caller:
    def __init__(self, name: str) -> None:
        self.name = name


@registry.node("sign")
def sign(text: str, who: FromRun[Caller]) -> str:
    return f"{text}, {who.name}"


run = registry.load(make_graph(sign)).run(text="Done", from_run={Caller: Caller("Ida")})
print(run.outputs)  # {'sign.result': 'Done, Ida'}
```

## A second version

Saved graphs name the version of each node they use. To change a node's inputs or outputs, write a new version that names the one it replaces, and say how a saved node moves on:

```python
from conductor import Upgrade, node


@node("greet")
def greet_v1(name: str) -> str:
    return f"Hello {name}"


@registry.node("greet", replaces=greet_v1, upgrade=Upgrade(renamed_inputs={"name": "person"}))
def greet(person: str, greeting: str = "Hello") -> str:
    return f"{greeting} {person}"


saved = make_graph(greet_v1, values={"greet.name": "Ida"})
print(registry.load(saved).run().outputs)  # {'greet.result': 'Hello Ida'}: it still runs version 1

moved = registry.upgrade(saved, "greet")
print(moved.nodes[0].version, moved.nodes[0].values)  # 2 {'person': 'Ida'}
```

Loading never upgrades. A host calls `registry.upgrade` and shows the author the result. `Deprecation(...)` on a version's `@node` tells an editor it is going away.

## Test the whole registry

Calling a node tests its body. `check_registry` tests what a host relies on and a call can't show: an upgrade that breaks a saved node, a body that changes the value it was given, an `Ask` no example reaches. It returns every problem at once:

```python
from conductor.testing import check_registry

problems = check_registry(registry, examples={
    "review": [{"proposal": "Move the server"}],
    "parse": [{"text": "12"}],
    "fetch": [{"url": "https://example.com"}],
    "sign": [{"text": "Done"}],
    "greet": [{"person": "Ida"}],
    "route": [{"amount": 5}],
    "assign": [{"ticket": "T-1"}],
    "notify": [{"amount": 5}],
}, from_run={Caller: Caller("Ida")})
print(problems)  # []
```

Put it in your test suite as `assert check_registry(registry, examples=...) == []`. A node's body is called only when you give it an example, since a node may send mail. `python -m conductor.about testing` lists every check.
