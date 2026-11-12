# Writing nodes for others

This page is for you if you write nodes that other people use in their graphs: a package of nodes, or the nodes your own editor offers. The [first page](https://github.com/syv-ai/conductor#readme) shows a host (the application that runs graphs) compiling and running a graph. Here every example builds its own small graph with `make_graph`, so you can see what a node does in one. The blocks run in order as one script. `run.results` is every node's outputs, by node id and then output name; `run.outputs` is the graph's outputs, those of the nodes no edge leaves, by address `node.output`.

`make_graph(a, b)` writes a graph record from Python. Each node goes under its own id, `upper-1` and `upper-2` when one is used twice. With no `edges`, the nodes form a chain: each takes its one required input from the one output of the node before it. `edges=[("a.result", "b.text")]` names the edges instead; an output of one value is named `result`. `constants={"a.text": "hi"}` gives an input a constant, a value it has on every call of every run, and `defaults=` gives one a value the caller may change. Every key is an address, `"node.input"`. `make_graph` only writes the record. It raises for what it can't write: a value for a node it wasn't given, a value with no JSON form, an edge that is no address, a chain it can't draw. Everything else is found when the record is compiled, in `Graph.compile(record, registry).problems`: an input a node doesn't have, a value of the wrong type, a default on an input that isn't a graph input (a caller gives only the inputs with no edge on the nodes no edge leads into; a required input with no edge on any other node is `unbound_on_later_node`), an edge between types that don't fit.

The rules in full are in the reference: `python -m conductor.about`.

## A registry of your own

Keep your nodes in a registry. Its `category`, `tags` and `extra` are defaults for every node you add:

```python
from typing import Annotated

from conductor import Graph, GraphRecord, NodeRegistry, Param, Run, make_graph
from conductor.widgets import NumberInput, TextInput

text_nodes = NodeRegistry(category="text", tags=["acme"])


@text_nodes.node("repeat", title="Repeat", description="Repeats a text.", extra={"docs_url": "https://example.com/repeat"})
def repeat(
    text: Annotated[str, Param(title="Text", widget=TextInput())],
    times: Annotated[int, Param(title="Times", ge=1, le=10, widget=NumberInput())] = 2,
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

`Annotated[..., Param(...)]` says what an editor shows for an input or an output: a `title`, a `description` and a `widget`, the control to edit it with. Without a title, its label is its name made readable (`case_sensitive` is "Case sensitive"). A node declares what a person reads for it, `@node("id", title=, description=)`, and both are required; the docstring is for developers and never reaches a person.

`Param` also takes pydantic's limits: `ge`, `gt`, `le`, `lt`, `multiple_of`, `min_length`, `max_length` and `pattern`. `Param` is the one way to write them on an input or an output: pydantic's own `Field(...)` in a node's hint is refused when the node is declared (`pydantic_field_in_hint`), with the `Param` to write. A type that has its limits is still a type, so `PositiveInt`, `constr(min_length=1)` and `Annotated[int, annotated_types.Gt(0)]` work as hints, and `Field(...)` stays the way to write the fields of your own pydantic model. The limits are checked on every constant and default in a graph record and on every value a caller gives:

```python
drawn = {"nodes": [{"id": "repeat", "node": "repeat", "constants": {"times": 20}}]}  # as an editor saves it
graph = Graph.compile(GraphRecord.model_validate(drawn), host)

for problem in graph.problems:
    print(problem.loc, problem.type)
# ('repeat', 'times') invalid_value
```

`make_graph(repeat, constants={"repeat.times": 20})` writes the same record, and compiling it reports the same problem: a record's faults are found in one place, `graph.problems`.

An alias may be wrapped again: with `Line = Annotated[str, Param(widget=TextInput())]`, `Annotated[Line, Param(min_length=1)]` keeps the widget and adds the limit. Stacked `Param`s merge as pydantic's stacked `Field`s do, the later one winning per argument and per limit. A `Param` on the item type of a `Series`, `Series[Annotated[int, Param(ge=0, title="Counts")]]`, is the input's: each item is held to the limit, and the title is the input's title.

`repeat.describe()` is the node as an editor's palette reads it, a `FunctionNodeDescription`, and `registry.describe()` is the whole palette, a `RegistryDescription`: pydantic records a host returns as they are, which `model_dump(mode="json")` writes as JSON.

### Widgets

No type has a default widget. A `str` may be one line, a text area or a dropdown, so each input names its own. An input with no widget gets no control: an edge gives it its value, or a constant in the graph record. These ship in `conductor.widgets`: `TextInput`, `Textarea`, `Dropdown`, `NumberInput`, `Slider`, `Switch`, `DatePicker`, `FileUpload` and `ListInput`. What each is for and the options it takes are in the reference that ships with the package: run `python -m conductor.about conductor-widgets`. A `Dropdown` takes no choices of its own: they are the input's, a `Literal`'s values or `Param(choices=[Choice("en", "English"), ...])`.

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

Pass an instance, `widget=Textarea()`: the class passed bare, `widget=Textarea`, is refused by a type checker and raises a `TypeError` where the node is declared.

## Types

Any type pydantic can check and write as JSON can be an input or an output: `str`, `int`, `float`, `bool`, `date`, a pydantic model, a dataclass. A pydantic model is the usual way to carry a record of your own:

```python
from pydantic import BaseModel

from conductor import host_type, node


@host_type("address", title="Address")
class Address(BaseModel):
    street: str
    city: str


@node("city", title="City", description="Picks the city out of an address.")
def city(address: Address) -> str:
    return address.city


print(city(Address(street="Nørregade 1", city="Aarhus")))  # Aarhus
```

A type of your own is shown to people, so you declare it with `@host_type`: the id an editor knows it by and the title a person reads, as `Address` does. Write it above the class, and above `@dataclass` when there is one; a library's class takes the same call, `host_type("ip", title="IP address")(ipaddress.IPv4Address)`. The declaration is checked when the class statement ends, so a mistake fails at import, on the class: an empty title, an id that is not one word or is one of conductor's own, or a class pydantic can't validate (`type_not_carried`), such as a plain class with only an `__init__` and no pydantic schema. A class used as an input's or output's type without a declaration fails where its node is declared, as `type_not_named`. A subclass does not inherit its parent's declaration; declare it too. A `TypedDict` is a record of plain fields and declares nothing. Conductor's own types (`str`, `int`, `date` and the rest) have ids and titles built in: `str` is "Text", `int` "Whole number".

An editor reads an input's type as an id, and the id says what is inside: a `list[Address]` input has the type `list[address]`, and a generic pydantic model `Box[int]` declared as `box` is `box[int]`. So a class of your own inside a list or a dict is declared too.

Why a decorator and not a base class: a decorator works on any class, a base class doesn't. A generic pydantic model (`Box[int]`) and `@dataclass(slots=True)` both build the class a second time, without a base class's keyword arguments; an enum must list such a base first; and a library's class can't be given a new base at all. A decorator runs after the class is built, so none of that matters.

- `X | None` takes `None`. An input receives `None` only if its type says so, with `| None` or as `Any`. Otherwise the node isn't called where the input's value is `None` (`run.not_called` lists it), and its outputs are `None` there too. If those outputs are branches (below), neither branch is chosen there.
- `Series[X]` is many values of `X`, a plain list to your function.
- `Table[R]` is a whole table, one value. The next example shows one.
- `list[X]` is one value that happens to be a list.

### A table

A `TypedDict` names a table's columns. To your function the table is a `polars.DataFrame`:

```python
import io
from typing import TypedDict

import polars as pl

from conductor import Table

tables = NodeRegistry()


class Contact(TypedDict):
    name: str
    email: str | None


@tables.node("contacts", title="Contacts", description="Reads contacts from a CSV text.")
def contacts(csv: str) -> Table[Contact]:
    return pl.read_csv(io.StringIO(csv))


@tables.node("reachable", title="Reachable", description="Keeps the contacts that have an email.")
def reachable(contacts: Table[Contact]) -> Table[Contact]:
    return contacts.filter(pl.col("email").is_not_null())


run = Graph.compile(make_graph(contacts, reachable), tables).run(csv="name,email,phone\nIda,ida@a.dk,1234\nBo,,5678\n")
print(run.outputs["reachable.result"].to_dicts())  # [{'name': 'Ida', 'email': 'ida@a.dk'}]
```

The CSV had a `phone` column, and `Contact` doesn't name one, so it was cut: a `Table[Contact]` has exactly the columns `Contact` names, wherever the table comes from. A column that is missing is refused, and a column may have a `None` only if its type says `| None`, as `email` does. A node that passes on columns it doesn't know declares a bare `Table`.

A node may also return a table as a list of records, `[{"name": "Ida", "email": None}]`. And a table is a type like any other: `Table[Contact] | None` takes `None`, and `Series[Table[Contact]]` is a table per item. Declare the table you return. An output typed `Any` says nothing about it, so a `Table` input wired to it guesses: where the output is returned once, it takes the value as a table; where it is returned once per item, it takes each value as one record and collects them into a table.

### A text or a number of your own

A subclass of `str`, `int` or another builtin needs only its declaration. A value is validated as the builtin and then made your class, so a check in its `__new__` runs:

```python
@host_type("email", title="Email")
class Email(str):
    pass


@node("domain", title="Domain", description="Gives the part of an email after the @.")
def domain(email: Annotated[Email, Param(max_length=254)]) -> str:
    return email.partition("@")[2]


print(domain.describe().inputs[0].type)  # email
```

A class that needs another way to be validated or written as JSON gives its own `__get_pydantic_core_schema__`, which `@host_type` leaves alone.

### A type that takes another

An output wires to an input of its own class, or of a parent class. A `Celsius` is no `Temperature`, so that edge is a `type_mismatch` until `Temperature` lists `Celsius` in `accepts`:

```python
from pydantic import ConfigDict


@host_type("celsius", title="Celsius")
class Celsius(BaseModel):
    degrees: float


@host_type("temperature", title="Temperature", accepts=[Celsius])
class Temperature(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # how it reads a Celsius: by its attributes
    degrees: float


@tables.node("measure", title="Measure", description="Measures the temperature.")
def measure() -> Celsius:
    return Celsius(degrees=21.5)


@tables.node("report", title="Report", description="Says how warm it is.")
def report(temperature: Temperature) -> str:
    return f"{temperature.degrees} degrees"


print(Graph.compile(make_graph(measure, report), tables).run().outputs)  # {'report.result': '21.5 degrees'}
```

`accepts` only lets the edge be drawn. Your class reads the value itself, so it has to be able to: here `from_attributes` does it.

## One or many

Return a `Series` and whatever takes one value of it runs once per item. You never write the loop:

```python
from dataclasses import dataclass

from conductor import NodeRegistry, Series

registry = NodeRegistry()


@dataclass(frozen=True)
class Letter:
    sender: str
    lines: Series[str]


@registry.node("open", title="Open", description="Reads a letter: who sent it, and its lines.")
def open_letter(text: str) -> Letter:
    sender, *lines = text.splitlines()
    return Letter(sender=sender, lines=lines)


@registry.node("tag", title="Tag", description="Puts the sender in front of a line.")
def tag(line: str, sender: str) -> str:
    return f"{sender}: {line}"


@registry.node("count", title="Count", description="Counts the texts it is given.")
def count(texts: Series[str]) -> int:
    return len(texts)


letter = make_graph(open_letter, tag, count, edges=[("open.lines", "tag.line"), ("open.sender", "tag.sender"), ("tag.result", "count.texts")])
run = Graph.compile(letter, registry).run(text="Ida\nhello\nsee you")
print(run.results["tag"]["result"])  # ['Ida: hello', 'Ida: see you']
print(run.outputs)                   # {'count.result': 2}
```

- `open` returns the lines as a `Series`, and `tag` takes one line, so `tag` runs once per line: it is inside the loop over `open.lines`. Every node downstream of `tag` would run once per line too.
- `count` takes a `Series`, so it collects the loop: it is called once, with every tagged line as a list, in order. A `Table[R]` input collects records into a table the same way.
- `open.sender` is returned once, outside the loop, so every call of `tag` gets the same sender. Such a value is an outer value.

A table never starts a loop by itself. Return `Series[R]`, with `R` a `TypedDict`, to return its records one item each; then an edge may name one field of each, written `node.output.field`, as `each.result.name` does below. You may return the records as a list or as the table they are in, which is cheaper for a large one: the nodes downstream receive the same plain records either way. The companion package `syv-conductor-nodes` (`uv add syv-conductor-nodes`) has a Split node that does this for any table or list.

A `Series` input collects only loops its own node is not inside. Below, `compare.line` takes one line, so `compare` runs once per line. Its `others` input is wired to `shout`, which runs once per line too, in the same loop: so for the line "hello", `others` gets only `shout`'s output for "hello", a list of one. To give every call all the lines, collect them first with a node that is outside the loop, such as the Collect node of `syv-conductor-nodes` (`series-collect`, which takes a `Series` and returns it as one list), and wire that list in:

```python
from conductor_nodes.series import collect

registry.add(collect)


@registry.node("shout", title="Shout", description="A line in capitals.")
def shout(line: str) -> str:
    return line.upper()


@registry.node("compare", title="Compare", description="Says how many other texts a line is compared with.")
def compare(line: str, others: Series[str]) -> str:
    return f"{line} against {len(others)}"


own = make_graph(open_letter, shout, compare, edges=[("open.lines", "shout.line"), ("open.lines", "compare.line"), ("shout.result", "compare.others")])
print(Graph.compile(own, registry).run(text="Ida\nhello\nsee you").results["compare"]["result"])  # ['hello against 1', 'see you against 1']

every = make_graph(
    open_letter, shout, collect, compare,
    edges=[("open.lines", "shout.line"), ("shout.result", "series-collect.values"), ("open.lines", "compare.line"), ("series-collect.result", "compare.others")],
)
print(Graph.compile(every, registry).run(text="Ida\nhello\nsee you").results["compare"]["result"])  # ['hello against 2', 'see you against 2']
```

- In `own`, `compare` runs once per line for its `line`, and `shout` returned one text for that same line: `others` is that one text.
- In `every`, `series-collect` takes a `Series` and nothing else, so it is outside the loop over the lines: it is called once and collects every shouted line. Its output is an outer value for `compare`, the same list on each call. `syv-conductor-nodes` also has Join, which collects texts into one text.

So a graph means the same for one letter as for many: put a node that returns a `Series` of letters in front of it, or use the whole graph as one node of another graph (a [subgraph](subgraphs.md)), and each loop in it goes on collecting only its own items.

An item whose value is `None` is left out of a `Series[X]` input, and `Series[X | None]` keeps its place as `None`. No items at all is an empty list, and your node is still called. What collecting does with branches not chosen is in the reference (`python -m conductor.about one-or-many`).

## Several outputs, and branches

Return a frozen dataclass for several outputs, one per field. Outputs that share an `exclusive_group` are branches: on each call the node returns a value for one, the chosen branch, and `SKIPPED` for the others. A node wired only to a branch that was not chosen is skipped, and so is every node downstream of it:

```python
from conductor import SKIPPED


@dataclass(frozen=True)
class Routed:
    large: Annotated[int, Param(exclusive_group="size")]
    small: Annotated[int, Param(exclusive_group="size")]


@registry.node("route", title="Route", description="Sends an amount down one of two paths.")
def route(amount: int) -> Routed:
    if amount >= 1000:
        return Routed(large=amount, small=SKIPPED)
    return Routed(large=SKIPPED, small=amount)


@registry.node("notify", title="Notify", description="Tells the manager about an amount.")
def notify(amount: int) -> str:
    return f"told the manager about {amount}"


graph = Graph.compile(make_graph(route, notify, edges=[("route.large", "notify.amount")]), registry)
print(graph.run(amount=5000).results["notify"])  # {'result': 'told the manager about 5000'}
print(graph.run(amount=5).results["notify"])     # {'result': None}
```

A skipped node's outputs read as `None` in `run.results` and `run.outputs`.

## A node for any type

A node that passes a value on doesn't care what the value is. Write its type as a type variable, and the node takes the type that arrives on its edge:

```python
@registry.node("keep", title="Keep", description="Passes a value on as it is.")
def keep[T](value: T) -> T:
    return value


graph = Graph.compile(make_graph(route, keep, notify, edges=[("route.large", "keep.value"), ("keep.result", "notify.amount")]), registry)
print(graph.describe().nodes["keep"].outputs[0].type)  # int
print(graph.run(amount=5000).outputs)                  # {'notify.result': 'told the manager about 5000'}
```

`route.large` is an `int`, so in this graph `keep` takes an `int` and returns one, and `notify.amount` may be wired to it. In a graph where a text arrives, `keep` returns a text. What the node returns is checked against the type that arrived. `None` is taken only by a hint that says so: where a `str | None` arrives, `value: T` is a `str` input and the node isn't called for a `None`, and a node that passes `None` on writes `value: T | None`.

A variable can stand inside another type. This node takes a table or a list apart, and returns the table's records or the list's items as a `Series`:

```python
from typing import cast


@registry.node("each", title="Each", description="Gives the records of a table, or the items of a list, one at a time.")
def each[T](value: Table[T] | list[T]) -> Series[T]:
    return cast("list[T]", value)  # conductor reads a table returned for a Series as its records; the cast tells the type checker


@registry.node("address-book", title="Address book", description="Reads contacts from a CSV text.")
def address_book(csv: str) -> Table[Contact]:
    return pl.read_csv(io.StringIO(csv))


@registry.node("greet-contact", title="Greet", description="Greets a person by name.")
def greet_contact(name: str) -> str:
    return f"Hello {name}"


graph = Graph.compile(make_graph(address_book, each, greet_contact, edges=[("address-book.result", "each.value"), ("each.result.name", "greet-contact.name")]), registry)
print(graph.run(csv="name,email\nIda,ida@a.dk\nBo,\n").outputs)  # {'greet-contact.result': ['Hello Ida', 'Hello Bo']}
```

A `Table[Contact]` arrived, so `T` is `Contact` and the edge may name its field, `each.result.name`. Wired to a `list[int]`, the same node returns a `Series[int]`.

A dataclass of outputs can be generic too. Each output takes the type that arrives:

```python
@dataclass(frozen=True)
class Checked[T]:
    passed: Annotated[T, Param(exclusive_group="check")]
    failed: Annotated[T, Param(exclusive_group="check")]


@registry.node("check", title="Check", description="Sends a value one way if it passed and the other if it failed.")
def check[T](value: T, ok: bool) -> Checked[T]:
    return Checked(passed=value, failed=SKIPPED) if ok else Checked(passed=SKIPPED, failed=value)
```

Only an edge says what `T` is. With no edge, a constant or a caller's value is taken as it is, as `Any`.

## Inputs and outputs that a constant decides

Some inputs or outputs can't be written in a signature, because the graph's author decides them: one output per column name they enter. A hook says what they are. It is a plain function whose parameters name the inputs it reads, and it returns the name and type of each output:

```python
def outputs_by_name(names: list[str]) -> dict[str, type]:
    return {name: str for name in names}


@registry.node("cells", title="Cells", description="Gives each cell of a CSV line under a name of its own.", compute_outputs=outputs_by_name)
def cells(line: str, names: list[str]) -> dict[str, str]:
    return dict(zip(names, line.split(",")))


graph = Graph.compile(make_graph(cells, constants={"cells.names": ["name", "city"]}), registry)
print(graph.outputs)                           # ['cells.name', 'cells.city']
print(graph.run(line="Ida,Aarhus").outputs)    # {'cells.name': 'Ida', 'cells.city': 'Aarhus'}
```

The hook is called when the graph is compiled, with the constant of `names`, so that input must be a constant in the graph record: no edge or caller can give it. The node returns its outputs as a dict by name. `compute_added_inputs=` adds inputs the same way, and the function takes them as `**values`. A hook that raises is a problem on the node, `compute_hook_failed`, with your message if you raise a `NodeError`.

Reach for a hook only for names a constant decides. A type that follows what arrives is a type variable.

## Asking a person

`raise Ask(prompt)` stops that call of the node and asks a person. Your return type says what they answer: `-> bool` asks a yes or no, and a frozen dataclass asks a form, one question per field, each field's default offered. What every type asks is in the reference (`python -m conductor.about asking`).

With no more than a prompt, `-> str` asks for a line of free text: `raise Ask(f"Why was {ticket} closed?")`. How a question looks is its type's, as for an input: `-> Annotated[str, Param(title="Reason", widget=Textarea())]`.

`Ask` adds what only the run knows: `choices` and `default`. On a node with one output, `choices` is a list:

```python
from conductor import Ask


@registry.node("assign", title="Assign", description="Asks a person who should take a ticket.")
def assign(ticket: str) -> str:
    raise Ask(f"Who takes {ticket}?", choices=["ida", "bo"], default="ida")


run = Graph.compile(make_graph(assign), registry).run(ticket="T-1")
print(run.waiting[0].questions[0].choices)  # [{'id': 'ida', 'title': 'ida'}, {'id': 'bo', 'title': 'bo'}]
```

A node with several outputs is answered by output name, so `choices` and `default` go by output name too, naming only the outputs they are for. A `Choice(id, title)` gives a choice a title of its own:

```python
from conductor import Choice
from conductor.widgets import Textarea


@dataclass(frozen=True)
class Review:
    approved: bool
    owner: str
    note: Annotated[str | None, Param(widget=Textarea())] = None


@registry.node("review", title="Review", description="Asks a person to review a proposal.")
def review(proposal: str) -> Review:
    raise Ask(f"Review: {proposal}", choices={"owner": [Choice("ida", "Ida B."), Choice("bo", "Bo K.")]})


graph = Graph.compile(make_graph(review), registry)
run = graph.run(proposal="Move the server")
stored = run.to_json()  # a waiting run is JSON: store it, and the person may answer next week

run = Run.from_json(stored, graph)  # in another process, with the same graph compiled again
(waiting,) = run.waiting
print([question.name for question in waiting.questions])  # ['approved', 'owner', 'note']

run = run.answer(waiting, {"approved": True, "owner": "ida"}).resume()
print(run.outputs)  # {'review.approved': True, 'review.owner': 'ida', 'review.note': None}
```

The node isn't called again for that call: the answer is its outputs. `note` has a default, so the answer may leave it out. An answer that doesn't fit its question is refused before anything runs. An `Ask` that doesn't fit your outputs, say a default your type refuses, fails the node as `invalid_ask`.

`Ask` is a `BaseException`, so an `except Exception` in your node lets it through.

## When something fails

Raise `NodeError` for a failure you can explain to a person. Its message reaches the host as written, and `code` is yours to choose:

```python
from conductor import EndedEarly, NodeError


@registry.node("parse", title="Parse", description="Reads a text as a whole number.")
def parse(text: str) -> int:
    if not text.isdigit():
        raise NodeError("The text is not a number.", code="not_a_number")
    return int(text)


run = Graph.compile(make_graph(parse), registry).run(text="twelve")
match run.state:  # a run that failed, stopped or timed out says why
    case EndedEarly(status=status, cause=cause):
        print(status, cause.type, cause.msg)  # failed not_a_number The text is not a number.
```

`ctx=` takes what a host needs beyond the text, as JSON values: text, numbers, `True`/`False`, `None`, lists and dicts of them. The host reads it as `cause.ctx`. A problem is saved and streamed as JSON, so a `ctx` with anything else in it (an object, a set, a tuple, a date) raises pydantic's `ValidationError` at your `raise`, and the node fails as `execution_failed`. A run's `state` is `Going(status)` while nothing ended it early, and `EndedEarly(status, cause)` when it failed, stopped, was cancelled or timed out.

Any other exception fails the node too. Its traceback goes to the `conductor` logger, and the host gets a fixed message with only the exception's class name, in `cause.ctx["error"]`, so a value your node handled never leaks into an event.

Raise `ExternalError` (a subclass of `NodeError`) when the outside world failed: a timeout, a rate limit, a server error. Only that is retried, under the node's `Policy`. A bug fails the same way every time, so it isn't retried:

```python
from conductor import ExternalError, Policy

attempts: list[str] = []


@registry.node("fetch", title="Fetch", description="Fetches a page from an address.", policy=Policy(retries=2, retry_delay=0.01))
def fetch(url: str) -> str:
    attempts.append(url)
    if len(attempts) < 3:
        raise ExternalError("The server did not answer.")
    return "<html>"


run = Graph.compile(make_graph(fetch), registry).run(url="https://example.com")
print(run.status, len(attempts))  # complete 3
```

`Policy` also sets `attempt_timeout`, `retry_on` (exception classes of your client to treat as external) and `concurrency`, how many calls of the node run at once, on threads. By default a node is called for one item at a time. A call given up at its `attempt_timeout` isn't stopped: it keeps running on its thread, the run goes on without it, and the process waits for it when it exits, so give a slow client its own timeout as well.

## A node that costs nothing

A host wants to tell a person how long a run will take before it starts: "this will make 33 calls to a language model". How many calls a node has depends on how many items the node before it returns, so the host can know it only once that node is called. Mark a node whose call costs nothing, and touches nothing outside the run, with `Policy(costs_nothing=True)`: one that hands on an upload's files, splits a table, or returns a list given to it. The host can then call those nodes first:

```python
@registry.node("files", title="Files", description="The files a person uploaded, one item each.", policy=Policy(costs_nothing=True))
def files(names: list[str]) -> Series[str]:
    return names


@registry.node("summarize", title="Summarize", description="Summarizes a file.")
def summarize(name: str) -> str:
    return f"A summary of {name}"  # yours: the call that costs something


ready = Graph.compile(make_graph(files, summarize), registry).with_inputs(names=["a.pdf", "b.pdf", "c.pdf"])
estimated = ready.resume(only_costs_nothing=True)

print(estimated.status)                       # ready
print(estimated.progress.nodes["summarize"])  # done=0 waiting=0 known=3 total=3
print(estimated.resume().progress.done)       # 4
```

`resume(only_costs_nothing=True)` is one call of `resume()` that calls only the marked nodes, with its `timeout`, its events when streamed, and its checkpoints; the run is `ready` afterwards. The next `resume()` calls the rest and makes none of those calls again. A marked node that raises `Ask` or fails then is left without an outcome, and asks or fails at the next `resume()`. A node that takes the run's context is refused the mark ([`costs_nothing_takes_context`](problems.md#costs_nothing_takes_context)): it uses the host's objects, so its call is not free.

`run.progress` is read from the run's outcomes each time, for the run and for each node: how many calls are `done` (their outcome is not waiting), how many are `waiting` for a person, how many are `known` so far, and the `total`. The `total` is `None` while it depends on a node that has not been called yet, such as a language model that finds the invoices in a document; `known` is then the least the total can be, and it only grows.

## What the host gives the run

Some things come from the host, not from the graph: who runs it, a database, a client. They are the run's context: one object of a class of yours, which the host passes for each run and which is never saved with it. Name the class on the registry, and a node that needs it takes it as its first parameter, typed `Context[...]`:

```python
from dataclasses import dataclass

from conductor import Context


@dataclass(frozen=True)
class Caller:
    name: str


@dataclass(frozen=True)
class AppContext:
    caller: Caller


signing = NodeRegistry(context_type=AppContext)


@signing.node("sign", title="Sign", description="Signs a text with the name of the person running it.")
def sign(ctx: Context[AppContext], text: str) -> str:
    return f"{text}, {ctx.caller.name}"


ida = AppContext(caller=Caller("Ida"))
print(sign(ida, "Done"))  # Done, Ida

signed = Graph.compile(make_graph(sign), signing).run(text="Done", context=ida)
print(signed.outputs)  # {'sign.result': 'Done, Ida'}
```

`ctx` is not an input: no edge reaches it and no caller fills it, and a call whose inputs are the same as before reuses its outcome whatever the context is. Pass `context=` again to each `run.resume`, `run.stream` and `run.again`. A graph compiled against `signing` is a `Graph[AppContext]`, so a type checker refuses a `context=` of another class; the run refuses one too, as `invalid_context`, and refuses a run with no context when a node takes it (`context_missing`).

A node may also name a base of the class, or a `Protocol` with the members it reads. That is how a node that doesn't know your class, such as one from a package of nodes, asks for what it needs:

```python
from typing import Protocol


class HasCaller(Protocol):
    @property
    def caller(self) -> Caller: ...


@signing.node("greet_caller", title="Greet the caller", description="Greets the person running it.")
def greet_caller(ctx: Context[HasCaller]) -> str:
    return f"Hello {ctx.caller.name}"
```

Declare a member as a read-only property, as here: a frozen dataclass can't satisfy a Protocol whose member is a plain attribute. Compiling checks every node's `Context[...]` against the registry's class: a node asking for something the class lacks, or for a context on a registry that names no class, has the problem `context_does_not_fit` at that node and parameter. A registry added with `add_subregistry` names the same class or none.

## A second version

A graph record names the version of each node it uses (1 when it says none). To change a node's inputs or outputs, write a new version that names the one it replaces, and say how a node in a saved graph record is moved to it:

```python
from conductor import Upgrade, node


@node("greet", title="Greet", description="Greets a person by name.")
def greet_v1(name: str) -> str:
    return f"Hello {name}"


@registry.node("greet", title="Greet", description="Greets a person by name.", replaces=greet_v1, upgrade=Upgrade(renamed_inputs={"name": "person"}))
def greet(person: str, greeting: str = "Hello") -> str:
    return f"{greeting} {person}"


saved = make_graph(greet_v1, constants={"greet.name": "Ida"})
print(Graph.compile(saved, registry).run().outputs)  # {'greet.result': 'Hello Ida'}: it still runs version 1

moved = registry.upgrade(saved, "greet")
print(moved.nodes[0].version, moved.nodes[0].constants)  # 2 {'person': 'Ida'}
```

Compiling never upgrades. A host calls `registry.upgrade` and shows the author the result. `@node(..., deprecation=Deprecation(header=..., description=..., alternative=...))` tells an editor a version is going away, and which node id to use instead.

## Test the whole registry

Calling a node tests its body. `check_registry` tests what a host relies on and a call can't show: an upgrade that breaks a saved node, a hook that fails, a body that changes the value it was given, an example that fails. It returns every problem at once:

```python
from conductor.testing import check_registry

problems = check_registry(registry, examples={
    "review": [{"proposal": "Move the server"}],
    "parse": [{"text": "12"}],
    "fetch": [{"url": "https://example.com"}],
    "greet": [{"person": "Ida"}],
    "route": [{"amount": 5}],
    "assign": [{"ticket": "T-1"}],
    "notify": [{"amount": 5}],
    "open": [{"text": "Ida\nhello"}],
    "tag": [{"line": "hello", "sender": "Ida"}],
    "count": [{"texts": ["hello"]}],
    "cells": [{"line": "Ida,Aarhus", "names": ["name", "city"]}],
})
print(problems)  # []
print(check_registry(signing, examples={"sign": [{"text": "Done"}]}, context=ida))  # []
```

Put it in your test suite as `assert check_registry(registry, examples=...) == []`. A node's body is called only when you give it an example, since a node may send mail. A node with no example is still checked for everything that needs no call, such as its upgrades and its hooks; a hook that reads an input whose value can't be made up, such as one with a `pattern`, needs an example (`needs_example`). `python -m conductor.about testing` lists every check.
