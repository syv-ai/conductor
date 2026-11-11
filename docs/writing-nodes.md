# Writing nodes for others

This page is for you if you write nodes that other people place in graphs: a package of nodes, or the nodes your own editor offers. The [first page](https://github.com/syv-ai/conductor#readme) shows a host loading and running a graph. Here every example builds its own small graph with `make_graph`, so you can see what a node does in one.

`make_graph(a, b)` writes a graph record from Python. Each node goes under its own id, `upper-1` and `upper-2` when one is used twice. With no `edges`, the nodes form a chain: each takes its one required input from the one output of the node before it. `edges=[("a.result", "b.text")]` names the edges instead. `values={"a.text": "hi"}` types a value in, and `defaults=` gives one a caller may change. Every key is a `"node.input"` address. `make_graph` only writes the record. It raises for what it can't write: a value for a node it wasn't given, a value with no JSON form, an edge that is no address, a chain it can't draw. An input a node doesn't have, a value of the wrong type, a default on an input no caller reaches or an edge between types that don't meet is in `registry.load(record).problems`.

The rules in full are in the reference: `python -m conductor.about`.

## A registry of your own

Keep your nodes in a registry. Its `category`, `tags` and `extra` are defaults for every node you add:

```python
from typing import Annotated

from conductor import NodeRegistry, Param, make_graph
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

`Annotated[..., Param(...)]` says what an editor shows for a field: a `title`, a `description` and a `widget`, the control to edit it with. Without a title, a field's label is its name made readable (`case_sensitive` is "Case sensitive"). A node declares what a person reads for it, `@node("id", title=, description=)`, and both are required; the docstring is for developers and never reaches a person.

`Param` also takes pydantic's limits: `ge`, `gt`, `le`, `lt`, `multiple_of`, `min_length`, `max_length` and `pattern`. They are checked on every value someone types in or a caller gives:

```python
drawn = {"nodes": [{"id": "repeat", "node": "repeat", "values": {"times": 20}}]}  # as an editor saves it
graph = host.load(drawn)

for problem in graph.problems:
    print(problem.loc, problem.type)
# ('repeat', 'times') invalid_value
```

`make_graph(repeat, values={"repeat.times": 20})` writes the same record, and loading it reports the same problem: a record's faults are found in one place, `graph.problems`.

An alias may be wrapped again: with `Line = Annotated[str, Param(widget=TextInput())]`, `Annotated[Line, Param(min_length=1)]` keeps the widget and adds the limit. Stacked `Param`s merge as pydantic's stacked `Field`s do, the later one winning per argument and per limit. A `Param` on a `Series` item, `Series[Annotated[int, Param(ge=0)]]`, is the field's.

`repeat.describe()` is the node as an editor's palette reads it, a `FunctionNodeDescription`, and `registry.describe()` is the whole palette, a `RegistryDescription`: pydantic records a host returns as they are, which `model_dump(mode="json")` writes as JSON.

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

Pass an instance, `widget=Textarea()`: a type checker refuses the class passed bare, `widget=Textarea`.

## Types

Any type pydantic can check and write as JSON can be an input or an output: `str`, `int`, `float`, `bool`, `date`, a pydantic model, a dataclass. A pydantic model is the usual way to carry a record of your own:

```python
from typing import ClassVar

from pydantic import BaseModel

from conductor import node


class Address(BaseModel):
    type_id: ClassVar[str] = "address"
    type_title: ClassVar[str] = "Address"
    street: str
    city: str


@node("city", title="City", description="Picks the city out of an address.")
def city(address: Address) -> str:
    return address.city


print(city(Address(street="Nørregade 1", city="Aarhus")))  # Aarhus
```

A type of your own is shown to people, so it declares the id an editor knows it by and the title a person reads as class attributes, `type_id` and `type_title`, as `Address` does. They are read when the node is declared, so a type missing one fails at import. A `TypedDict` is a record of plain fields and declares nothing. Conductor's own types (`str`, `int`, `date` and the rest) take theirs from conductor's table. A class pydantic can't check, such as a plain class with only an `__init__`, is refused when the node is declared, as `type_not_carried`.

An editor reads a field's type as an id, and the id says what is inside: a `list[Address]` input has the type `list[address]`. So a class of your own inside a list or a dict declares its id and title too.

- `X | None` takes a blank. An input receives a blank only if its type says so, with `| None` or as `Any`. Otherwise the node isn't called where the input is blank, and its outputs are blank there too. Outputs that share an `exclusive_group` are a branch not taken there, since no branch was chosen.
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


run = tables.load(make_graph(contacts, reachable)).run(csv="name,email,phone\nIda,ida@a.dk,1234\nBo,,5678\n")
print(run.outputs["reachable.result"].to_dicts())  # [{'name': 'Ida', 'email': 'ida@a.dk'}]
```

The CSV had a `phone` column, and `Contact` doesn't name one, so it was cut: a `Table[Contact]` holds exactly the columns `Contact` names, whoever brings the table. A column that is missing is refused, and a column holds a blank only if its type says `| None`, as `email` does. A node that passes on columns it doesn't know declares a bare `Table`.

A node may also return a table as a list of records, `[{"name": "Ida", "email": None}]`. And a table is a type like any other: `Table[Contact] | None` takes a blank, and `Series[Table[Contact]]` is a table per item. Declare the table you return: an output typed `Any` says nothing, so a `Table` input takes it as a table only where there is nothing to gather, and as one record of a table wherever it is made per item.

### A text or a number of your own

A subclass of `str`, `int` or another builtin says how pydantic reads it. Three lines do it: read a `str`, then make it an `Email`:

```python
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema


class Email(str):
    type_id: ClassVar[str] = "email"
    type_title: ClassVar[str] = "Email"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(cls, handler(str))


@node("domain", title="Domain", description="Gives the part of an email after the @.")
def domain(email: Annotated[Email, Param(max_length=254)]) -> str:
    return email.partition("@")[2]


print(domain.describe().inputs[0].type)  # email
```

Without the method, `Email` is refused as `type_not_carried`.

### A type that takes another

An output wires to an input of its own class, or of a parent class. A `Celsius` is no `Temperature`, so that edge is a `type_mismatch` until `Temperature` lists `Celsius` in `type_accepts`:

```python
from pydantic import ConfigDict


class Celsius(BaseModel):
    type_id: ClassVar[str] = "celsius"
    type_title: ClassVar[str] = "Celsius"
    degrees: float


class Temperature(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # how it reads a Celsius: by its attributes
    type_id: ClassVar[str] = "temperature"
    type_title: ClassVar[str] = "Temperature"
    type_accepts: ClassVar[tuple[type, ...]] = (Celsius,)
    degrees: float


@tables.node("measure", title="Measure", description="Measures the temperature.")
def measure() -> Celsius:
    return Celsius(degrees=21.5)


@tables.node("report", title="Report", description="Says how warm it is.")
def report(temperature: Temperature) -> str:
    return f"{temperature.degrees} degrees"


print(tables.load(make_graph(measure, report)).run().outputs)  # {'report.result': '21.5 degrees'}
```

`type_accepts` only lets the edge be drawn. Your class reads the value itself, so it has to be able to: here `from_attributes` does it.

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
run = registry.load(letter).run(text="Ida\nhello\nsee you")
print(run.results["tag"]["result"])  # ['Ida: hello', 'Ida: see you']
print(run.outputs)                   # {'count.result': 2}
```

- `open` gives many lines, and `tag` takes one, so `tag` runs once per line. Anything reading `tag` would run once per line too.
- `count` takes a `Series`, so it gathers: it runs once and gets every tagged line, in order. A `Table[R]` input gathers records into a table the same way.
- The sender was made outside the lines, so it is simply there on every call of `tag`.

A table never makes items by itself. Return `Series[R]`, with `R` a `TypedDict`, to give records one at a time; then an edge may name one field of each, `read.result.email`. You may return the records as a list or as the table they are in, which is cheaper for a large one: the nodes after it receive the same plain records either way. A host with `syv-conductor-nodes` has a Split node that does this for any table or list.

A `Series` input gets what its source made for that call. When the node itself runs per item, because another input reads one item, a source that runs per that same item made one value for the call, and the input receives that one, as a list of one. To hand every item's value to each call, gather them once outside the items and wire the list in:

```python
@registry.node("shout", title="Shout", description="A line in capitals.")
def shout(line: str) -> str:
    return line.upper()


@registry.node("all-of", title="All of", description="Every text it is given, as one list.")
def all_of(texts: Series[str]) -> Series[str]:
    return texts


@registry.node("compare", title="Compare", description="Says how many other texts a line is compared with.")
def compare(line: str, others: Series[str]) -> str:
    return f"{line} against {len(others)}"


own = make_graph(open_letter, shout, compare, edges=[("open.lines", "shout.line"), ("open.lines", "compare.line"), ("shout.result", "compare.others")])
print(registry.load(own).run(text="Ida\nhello\nsee you").results["compare"]["result"])  # ['hello against 1', 'see you against 1']

every = make_graph(
    open_letter, shout, all_of, compare,
    edges=[("open.lines", "shout.line"), ("shout.result", "all-of.texts"), ("open.lines", "compare.line"), ("all-of.result", "compare.others")],
)
print(registry.load(every).run(text="Ida\nhello\nsee you").results["compare"]["result"])  # ['hello against 2', 'see you against 2']
```

- In `own`, `compare` runs once per line for its `line`, and `shout` made one text for that line: `others` is that one text.
- In `every`, `all-of` runs once, outside the lines, and gathers every shouted line; its list is then there on each call of `compare`. With `syv-conductor-nodes`, Join gathers texts into one text and Collect gathers records into a table the same way.

So a graph that works for one letter means the same when a node in front of it makes it run per letter, and when it is stored and used as a node per letter.

A blank item is left out of a `Series[X]` input, and `Series[X | None]` keeps its place as `None`. No items at all is an empty list, and your node is still called. What a gather does with branches not taken is in the reference (`python -m conductor.about one-or-many`).

## Several outputs, and branches

Return a frozen dataclass for several outputs, one per field. Outputs that share an `exclusive_group` are branches: on each call one gets a value and the others get `SKIPPED`. A node fed only `SKIPPED` is skipped too:

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


graph = registry.load(make_graph(route, notify, edges=[("route.large", "notify.amount")]))
print(graph.run(amount=5000).results["notify"])  # {'result': 'told the manager about 5000'}
print(graph.run(amount=5).results["notify"])     # {'result': None}
```

A branch not taken reads as `None` in `run.results` and `run.outputs`.

## A node for any type

A node that passes a value on doesn't care what the value is. Write its type as a type variable, and the node takes the type that arrives on its edge:

```python
@registry.node("keep", title="Keep", description="Passes a value on as it is.")
def keep[T](value: T) -> T:
    return value


graph = registry.load(make_graph(route, keep, notify, edges=[("route.large", "keep.value"), ("keep.result", "notify.amount")]))
print(graph.describe().nodes["keep"].outputs[0].type)  # int
print(graph.run(amount=5000).outputs)                  # {'notify.result': 'told the manager about 5000'}
```

`route.large` gives an `int`, so in this graph `keep` takes an `int` and gives one, and `notify.amount` may read it. In a graph where a text arrives, `keep` gives a text. What the node returns is checked against the type that arrived. A blank is taken only by a hint that says so: where a `str | None` arrives, `value: T` is a `str` input and the node isn't called for the blank, and a node that passes blanks on writes `value: T | None`.

A variable can stand inside another type. This node takes a table or a list apart, and what it gives are the table's records or the list's items:

```python
from typing import cast


@registry.node("each", title="Each", description="Gives the records of a table, or the items of a list, one at a time.")
def each[T](value: Table[T] | list[T]) -> Series[T]:
    return cast("list[T]", value)  # a table returned for a Series is read as its records


@registry.node("address-book", title="Address book", description="Reads contacts from a CSV text.")
def address_book(csv: str) -> Table[Contact]:
    return pl.read_csv(io.StringIO(csv))


@registry.node("greet-contact", title="Greet", description="Greets a person by name.")
def greet_contact(name: str) -> str:
    return f"Hello {name}"


graph = registry.load(make_graph(address_book, each, greet_contact, edges=[("address-book.result", "each.value"), ("each.result.name", "greet-contact.name")]))
print(graph.run(csv="name,email\nIda,ida@a.dk\nBo,\n").outputs)  # {'greet-contact.result': ['Hello Ida', 'Hello Bo']}
```

A `Table[Contact]` arrived, so `T` is `Contact` and the edge may name its field, `each.result.name`. Fed a `list[int]`, the same node gives a `Series[int]`.

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

Only an edge says what `T` is. With nothing wired to it, a value typed in or given by a caller is taken as it is, as `Any`.

A blank is no different here. An input receives one only if its type says so, so where a `str | None` output is wired to `value: T`, `T` is `str` and the node isn't called for a blank. A node that passes blanks on writes `value: T | None`.

## Fields that a setting decides

Some fields can't be written in a signature, because the graph's author decides them: one output per column name they type in. A hook says what those fields are. It is a plain function whose parameters name the inputs it reads, and it returns each field's name and type:

```python
def outputs_by_name(names: list[str]) -> dict[str, type]:
    return {name: str for name in names}


@registry.node("cells", title="Cells", description="Gives each cell of a CSV line under a name of its own.", compute_outputs=outputs_by_name)
def cells(line: str, names: list[str]) -> dict[str, str]:
    return dict(zip(names, line.split(",")))


graph = registry.load(make_graph(cells, values={"cells.names": ["name", "city"]}))
print(graph.outputs)                           # ['cells.name', 'cells.city']
print(graph.run(line="Ida,Aarhus").outputs)    # {'cells.name': 'Ida', 'cells.city': 'Aarhus'}
```

The hook runs when a graph is loaded, on what is typed into `names` there, so that input is set in the graph and no edge or caller fills it. The node returns its outputs as a mapping by name. `compute_added_inputs=` adds inputs the same way, and the function takes them as `**values`. A hook that raises is a problem on the node, `compute_hook_failed`, with your message if you raise a `NodeError`.

Reach for a hook only for names a setting decides. A type that follows what arrives is a type variable.

## Asking a person

`raise Ask(prompt)` stops that call of the node and asks a person. Your return type says what they answer: `-> bool` asks a yes or no, and a frozen dataclass asks a form, one question per field, each field's default offered. What every type asks is in the reference (`python -m conductor.about`, "Legs and Asking").

With no more than a prompt, `-> str` asks for a line of free text: `raise Ask(f"Why was {ticket} closed?")`. How a question looks is its type's, as for an input: `-> Annotated[str, Param(title="Reason", widget=Textarea())]`.

`Ask` adds what only the run knows: `choices` and `default`. On a node with one output, `choices` is a list:

```python
from conductor import Ask


@registry.node("assign", title="Assign", description="Asks a person who should take a ticket.")
def assign(ticket: str) -> str:
    raise Ask(f"Who takes {ticket}?", choices=["ida", "bo"], default="ida")


run = registry.load(make_graph(assign)).run(ticket="T-1")
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


graph = registry.load(make_graph(review))
run = graph.run(proposal="Move the server")
stored = run.to_json()  # a waiting run is JSON: store it, and the person may answer next week

run = graph.load_run(stored)  # in another process, on the same graph
(waiting,) = run.waiting
print([question.name for question in waiting.questions])  # ['approved', 'owner', 'note']

run = run.answer(waiting, {"approved": True, "owner": "ida"}).resume()
print(run.outputs)  # {'review.approved': True, 'review.owner': 'ida', 'review.note': None}
```

The node isn't called again for it: the answer is its outputs. An answer that doesn't fit its question is refused before anything runs. An `Ask` that doesn't fit your outputs, say a default your type refuses, fails the node as `invalid_ask`.

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


run = registry.load(make_graph(parse)).run(text="twelve")
match run.state:  # a run that failed, stopped or timed out says why
    case EndedEarly(status=status, cause=cause):
        print(status, cause.type, cause.msg)  # failed not_a_number The text is not a number.
```

`ctx=` takes what a host needs beyond the text, as JSON values: text, numbers, `True`/`False`, `None`, lists and dicts of them. A problem is saved and streamed as JSON, so a `ctx` holding anything else (an object, a set, a tuple, a date) raises pydantic's `ValidationError` at your `raise`, and the node fails as `execution_failed`.

Any other exception fails the node too. Its traceback goes to the `conductor` logger, and only its class name reaches the host, so a value your node handled never leaks into an event.

Raise `ExternalError` when the outside world failed: a timeout, a rate limit, a server error. Only that is retried, under the node's `Policy`. A bug fails the same way every time, so it isn't retried:

```python
from conductor import ExternalError, Policy

attempts: list[str] = []


@registry.node("fetch", title="Fetch", description="Fetches a page from an address.", policy=Policy(retries=2, retry_delay=0.01))
def fetch(url: str) -> str:
    attempts.append(url)
    if len(attempts) < 3:
        raise ExternalError("The server did not answer.")
    return "<html>"


run = registry.load(make_graph(fetch)).run(url="https://example.com")
print(run.status, len(attempts))  # complete 3
```

`Policy` also sets `attempt_timeout`, `retry_on` (exception classes of your client to treat as external) and `concurrency`, how many calls run at once on threads. By default a node is called for one item at a time. A call given up at its `attempt_timeout` isn't stopped: it keeps running on its thread, the run goes on without it, and the process waits for it when it exits, so give a slow client its own timeout as well.

## What the host gives the run

Some values come from the host, not from the graph: who runs it, a database session, a client. Ask for them with `FromRun[T]`. The host passes them by type:

```python
from conductor import FromRun


class Caller:
    def __init__(self, name: str) -> None:
        self.name = name


@registry.node("sign", title="Sign", description="Signs a text with the name of the person running it.")
def sign(text: str, who: FromRun[Caller]) -> str:
    return f"{text}, {who.name}"


run = registry.load(make_graph(sign)).run(text="Done", from_run={Caller: Caller("Ida")})
print(run.outputs)  # {'sign.result': 'Done, Ida'}
```

## A second version

Saved graphs name the version of each node they use. To change a node's inputs or outputs, write a new version that names the one it replaces, and say how a saved node moves on:

```python
from conductor import Upgrade, node


@node("greet", title="Greet", description="Greets a person by name.")
def greet_v1(name: str) -> str:
    return f"Hello {name}"


@registry.node("greet", title="Greet", description="Greets a person by name.", replaces=greet_v1, upgrade=Upgrade(renamed_inputs={"name": "person"}))
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
    "open": [{"text": "Ida\nhello"}],
    "tag": [{"line": "hello", "sender": "Ida"}],
    "count": [{"texts": ["hello"]}],
    "cells": [{"line": "Ida,Aarhus", "names": ["name", "city"]}],
}, from_run={Caller: Caller("Ida")})
print(problems)  # []
```

Put it in your test suite as `assert check_registry(registry, examples=...) == []`. A node's body is called only when you give it an example, since a node may send mail. `python -m conductor.about testing` lists every check.
