# Writing nodes for others

This page is for you if you write nodes that other people put in their graphs. That might be a package of nodes, or the nodes your own editor offers. The application that runs the graphs is called the host. It might be yours, or someone else's.

You don't need to know conductor yet. We start with one node you can run in a minute, and add one idea at a time. Every example runs: the blocks on this page run in order, as one script, and the comment after a `print` is what it prints.

The page teaches by example. The full rules are in the reference that ships with the package. `python -m conductor.about` prints all of it, and `python -m conductor.about <section>` prints one part. Each section below says which part to read.

## Your first node

A node is a plain Python function with `@node` on top. Its type hints are its interface. This one takes a text and returns a text:

```python
from conductor import Graph, NodeRegistry, make_graph

registry = NodeRegistry()


@registry.node("shout", title="Shout", description="Writes a text in capitals.")
def shout(text: str) -> str:
    return text.upper() + "!"


print(shout("hello"))  # HELLO!
```

The node is still your function, so you can call it, and that is its unit test. The registry is the set of nodes that graphs may use. `"shout"` is the node's id. The `title` and `description` are what a person reads in an editor, and both are required. A docstring is for developers, and a person never sees it.

To run the node, put it in a graph:

```python
graph = Graph.compile(make_graph(shout), registry)
run = graph.run({"text": "hello"})
print(run.outputs)  # {'result': 'HELLO!'}
```

`make_graph` writes a graph from Python. Usually a person draws the graph in an editor and it reaches the host as JSON, but `make_graph` is the quick way to try a node. `Graph.compile` checks the graph against your registry, and `graph.run` runs it.

Nothing in the graph gives `text` a value, so `text` becomes an input of the graph, and you pass it to `run`. The function's one return value is an output called `result`. No other node reads it, so it is what the graph returns.

## Wiring nodes together

Give `make_graph` two nodes and it wires them in a row. Each node takes its input from the output of the node before it:

```python
@registry.node("count-letters", title="Count letters", description="Counts the letters in a text.")
def count_letters(text: str) -> int:
    return sum(char.isalpha() for char in text)


run = Graph.compile(make_graph(shout, count_letters), registry).run({"text": "hi there"})
print(run.results["shout"])  # {'result': 'HI THERE!'}
print(run.outputs)           # {'result': 7}
```

`run.outputs` is what the graph returns. `run.results` has the outputs of every node, by node id and then by output name.

You can also say exactly how a graph is wired. Each place in it is written `"node.name"`:

- `edges=[("shout.result", "count-letters.text")]` lists the edges.
- `constants={"shout.text": "hey"}` gives an input a fixed value, the same on every run.
- `defaults={"shout.text": "hey"}` makes an input a graph input, with a value the caller may change.

```python
fixed = Graph.compile(make_graph(shout, constants={"shout.text": "hey"}), registry)
print(fixed.run().outputs)  # {'result': 'HEY!'}

flexible = Graph.compile(make_graph(shout, defaults={"shout.text": "hey"}), registry)
print(flexible.run().outputs)           # {'result': 'HEY!'}
print(flexible.run({"text": "yo"}).outputs)  # {'result': 'YO!'}
```

`make_graph` only writes the graph. It raises when it can't, for example for a place that isn't written `"node.name"`. Every other fault is reported when the graph is compiled, in `graph.problems`. [What a person sees](#what-a-person-sees) shows one. The full rules of `make_graph` are in `python -m conductor.about writing-a-graph-in-python`.

## A registry of your own

When you share nodes, keep them in a registry of their own. Its `category`, `tags` and `extra` become the defaults of every node you add to it:

```python
from typing import Annotated

from conductor import Param

text_nodes = NodeRegistry(category="text", tags=["acme"])


@text_nodes.node("repeat", title="Repeat", description="Repeats a text.", extra={"docs_url": "https://example.com/repeat"})
def repeat(text: str, times: Annotated[int, Param(title="Times", ge=1, le=10)] = 2) -> str:
    return text * times


print(repeat("ab"))  # abab
```

The host adds your registry to its own, and your nodes are copied in with your defaults:

```python
host = NodeRegistry()
host.add_subregistry(text_nodes)

print(host["repeat"].category, host["repeat"].tags)  # text ('acme',)
```

If the host already has a node with one of your ids, `add_subregistry` raises. `extra` is your own JSON about a node. Conductor keeps it and shows it to the host, but never reads it. Its values are JSON values: a tuple, a date or `inf` in it is refused where the node or the registry is declared, as is any argument of `@node` or `NodeRegistry` that isn't of the type its signature says, such as `tags="text"` for `tags=["text"]` ([`invalid_declaration_argument`](problems.md#invalid_declaration_argument)).

## What a person sees

An editor shows each input with a label and a control. You describe them with `Param`, inside `Annotated`, as `times` does above:

- `title` and `description` are what a person reads. Without a title, the title is `None`: conductor never makes one up from the parameter's name.
- `widget` is the control to edit the value with. [Widgets](#widgets) below lists them.
- `ge`, `le`, `max_length`, `pattern` and pydantic's other limits say which values are allowed.

Conductor checks the limits on every value a graph sets and every value a caller passes. Here is a graph, the way an editor saves it, that sets `times` to 20:

```python
from conductor import GraphRecord

drawn = {
    "nodes": {"repeat": {"node": "repeat", "inputs": {"text": {"graph_input": "text"}, "times": {"constant": 20}}}},
    "outputs": {"result": {"node": "repeat", "output": "result"}},
}
graph = Graph.compile(GraphRecord.model_validate(drawn), host)

for problem in graph.problems:
    print(problem.loc, problem.type)
# ('repeat', 'times') invalid_value
```

Compiling doesn't raise for a fault like this. It reports it in `graph.problems`, which is where every fault in a graph is found. `make_graph(repeat, constants={"repeat.times": 20})` gives the same problem.

Write limits with `Param`, not with pydantic's `Field(...)`. Conductor refuses a `Field` in a node's type hint, and tells you the `Param` to write instead. Types that carry their own limits, such as `PositiveInt`, work as they are. `Param`s can be stacked, as when you wrap an alias again, and the later one wins. The details are in `python -m conductor.about the-node-contract` and `python -m conductor.about types`.

### Settings fixed when the graph is drawn

Some inputs are settings. The graph's author picks a value when they draw the graph, and it never comes from another node or from the caller. Mark such an input `constant_only=True`:

```python
from conductor import node


@node("cut", title="Cut", description="Keeps the first letters of a text.")
def cut(text: str, keep: Annotated[int, Param(title="Letters kept", constant_only=True)] = 3) -> str:
    return text[:keep]


print([field.constant_only for field in cut.describe().inputs])  # [False, True]
```

An editor then offers no edge for `keep`. A graph that wires an edge to it, or makes it a graph input, gets the problem `input_takes_only_a_constant` when it is compiled.

`@node` makes a node without adding it to a registry, and `registry.add(cut)` adds it later. `cut.describe()` is the node as an editor's palette shows it, and `registry.describe()` is the whole palette. Both are pydantic models, so `model_dump(mode="json")` gives you their JSON.

### Widgets

No type has a default widget. A `str` might be one line, a text box or a dropdown, so you choose one per input. An input with no widget has none in `describe()`, and its control is left to the host's editor. Many such inputs get their value from an edge, or from a constant in the graph.

These widgets ship in `conductor.widgets`: `TextInput`, `Textarea`, `Dropdown`, `NumberInput`, `Slider`, `Switch`, `DatePicker`, `FileUpload` and `ListInput`. Pass an instance, `widget=Textarea()`, not the class. A `Dropdown` has no choices of its own. It shows the input's choices, which you give with `Param(choices=[...])` on a plain type such as `str`:

```python
from conductor import Choice
from conductor.widgets import Dropdown, Textarea


@node("translate", title="Translate", description="Translates a text.")
def translate(
    text: Annotated[str, Param(title="Text", widget=Textarea())],
    language: Annotated[
        str, Param(title="Language", choices=[Choice("da", "Danish"), Choice("en", "English")], widget=Dropdown())
    ] = "en",
) -> str:
    return f"[{language}] {text}"


print([field.widget for field in translate.describe().inputs])  # [{'type': 'textarea', 'rows': 4}, {'type': 'dropdown'}]
print(translate.describe().inputs[1].choices)  # (Choice(id='da', title='Danish'), Choice(id='en', title='English'))
```

Each choice is a value, or a `Choice` to give it a title. `Choice("en", "English")` is `Choice(id="en", title="English")`: the value the input gets, then what a person reads. Without titles, write `Param(choices=["da", "en"])`. A `Literal` works too: `language: Literal["da", "en"]` gives the same choices, untitled, and the same `enum` in the input's JSON schema. A `Literal` takes no `Param(choices=)`, since that would write each value twice, so titled choices always go on the plain type.

`describe()` gives each widget as the JSON the editor gets. What each widget is for, and its settings, is in `python -m conductor.about conductor-widgets`.

A widget is a plain pydantic model, and conductor passes it on to the editor without reading it. So a control of your own needs no new conductor release. Subclass `Widget`, next to the component in your frontend that draws it:

```python
from typing import Literal

from conductor.widgets import Widget


class EntityDropdown(Widget):
    type: Literal["entity-dropdown"] = "entity-dropdown"
    entity_kind: str = ""


print(EntityDropdown(entity_kind="customer").model_dump(mode="json"))
# {'type': 'entity-dropdown', 'entity_kind': 'customer'}
```

The widget is sent to the editor as JSON, so each of its fields needs a type pydantic can write as JSON. A field it can't write, such as a client object, raises an error as soon as the class is defined.

## Types

An input or an output can have any type pydantic can check and write as JSON: `str`, `int`, `float`, `bool`, `date`, `bytes`, a pydantic model, a dataclass. JSON has no bytes, so `bytes` cross JSON as URL-safe base64 text: a caller sends a file's contents as `"__4="`, and a saved run keeps them that way. Decode it with `base64.urlsafe_b64decode` in Python or `Buffer.from(text, "base64url")` in Node, since `base64.b64decode` silently drops `-` and `_` and `atob` throws on them. Conductor reads standard and URL-safe base64, with or without padding. A pydantic model with a `bytes` field builds its `model_config` from `conductor.JSON_CONFIG`, as `ConfigDict(**JSON_CONFIG, ...)`. A pydantic model is the usual way to pass a value with fields of your own. Declare it with `@host_type`, which gives it an id that editors know it by and a title that a person reads:

```python
from pydantic import BaseModel

from conductor import host_type


@host_type("address", title="Address")
class Address(BaseModel):
    street: str
    city: str


@registry.node("city", title="City", description="Picks the city out of an address.")
def city(address: Address) -> str:
    return address.city


print(city(Address(street="Nørregade 1", city="Aarhus")))  # Aarhus
print(city.describe().inputs[0].type)                       # address
```

Conductor checks the declaration as soon as the class is defined, so a mistake fails at import. If you forget `@host_type`, the node that uses the class fails where it is declared, with `type_not_named`, and the message gives a line to paste, such as `@host_type("invoice-line", title="Invoice line")` for `InvoiceLine`. A class inside a list or a dict needs the declaration too, and so does a subclass. Put `@host_type` above `@dataclass` when there is one. For a library's class that you can't decorate, call it: `host_type("http-status", title="HTTP status")(http.HTTPStatus)`.

Conductor's own types already have titles: `str` is "Text" and `int` is "Whole number". Pydantic's own types and the standard library's classes pydantic checks, such as `HttpUrl`, `EmailStr`, `AwareDatetime`, `pathlib.Path` and `ipaddress.IPv4Address`, are conductor's own too, and need no declaration; a library's class, such as one from pydantic-extra-types, does. To give one of them an id of your own, declare a subclass of it, such as `@host_type("web", title="Web")` above `class Web(HttpUrl)`; `@host_type` on conductor's own class is refused. More on types, and why `@host_type` is a decorator and not a base class, is in `python -m conductor.about types`.

A few types mean something special:

- `X | None` takes `None`. An input gets `None` only if its type says so. Otherwise the node isn't called when that input would be `None`, its outputs are `None` too, and `run.none_input_calls` lists the call.
- `Series[X]` is many values of `X`. To your function it is a plain list. [One or many](#one-or-many) shows what it does.
- `Table[R]` is a whole table, as one value. The next example shows one.
- `list[X]` is one value that happens to be a list.

### A table

A `TypedDict` names a table's columns, and needs no `@host_type`. To your function, the table is a `polars.DataFrame`:

```python
import io
from typing import TypedDict

import polars as pl

from conductor import Table


class Contact(TypedDict):
    name: str
    email: str | None


@registry.node("contacts", title="Contacts", description="Reads contacts from a CSV text.")
def contacts(csv: str) -> Table[Contact]:
    return pl.read_csv(io.StringIO(csv))


@registry.node("reachable", title="Reachable", description="Keeps the contacts that have an email.")
def reachable(contacts: Table[Contact]) -> Table[Contact]:
    return contacts.filter(pl.col("email").is_not_null())


run = Graph.compile(make_graph(contacts, reachable), registry).run({"csv": "name,email,phone\nIda,ida@a.dk,1234\nBo,,5678\n"})
print(run.outputs["result"].to_dicts())  # [{'name': 'Ida', 'email': 'ida@a.dk'}]
```

The CSV had a `phone` column, which `Contact` doesn't name, so it was dropped. A `Table[Contact]` always has exactly the columns `Contact` names. A missing column is refused, and a column may hold `None` only if its type says `| None`, as `email` does.

A node that passes on columns it doesn't know declares a bare `Table`. A node may also return a table as a list of rows, such as `[{"name": "Ida", "email": None}]`. The rest is in `python -m conductor.about types`.

### A text or a number of your own

A subclass of `str`, `int` or another builtin needs only `@host_type`:

```python
@host_type("email", title="Email")
class Email(str):
    pass


@node("domain", title="Domain", description="Gives the part of an email after the @.")
def domain(email: Annotated[Email, Param(max_length=254)]) -> str:
    return email.partition("@")[2]


print(domain(Email("ida@a.dk")))        # a.dk
print(domain.describe().inputs[0].type)  # email
```

A value is checked as the builtin and then turned into your class, so a check you write in `__new__` runs. A class that needs another way to be checked or written as JSON can define `__get_pydantic_core_schema__`, and `@host_type` leaves that alone. That schema must take a value of your class itself, not only its JSON form, since what a node returns is checked by it.

### A type that takes another

An edge joins an output to an input of the same class, or of a parent class. A `Celsius` is not a `Temperature`, so an edge from one to the other is refused, with `type_mismatch`. Adding `accepts=[Celsius]` to `Temperature` allows it:

```python
from pydantic import ConfigDict


@host_type("celsius", title="Celsius")
class Celsius(BaseModel):
    degrees: float


@host_type("temperature", title="Temperature", accepts=[Celsius])
class Temperature(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # lets it read a Celsius by its attributes
    degrees: float


@registry.node("measure", title="Measure", description="Measures the temperature.")
def measure() -> Celsius:
    return Celsius(degrees=21.5)


@registry.node("report", title="Report", description="Says how warm it is.")
def report(temperature: Temperature) -> str:
    return f"{temperature.degrees} degrees"


print(Graph.compile(make_graph(measure, report), registry).run().outputs)  # {'result': '21.5 degrees'}
```

`accepts` only allows the edge. Your class still has to read the value itself, and here `from_attributes` does that.

## One or many

Nodes often work on many things: the lines of a letter, the rows of a file. You don't write the loop. Return a `Series`, and a node that takes one value of it runs once per item:

```python
from dataclasses import dataclass

from conductor import Series


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
run = Graph.compile(letter, registry).run({"text": "Ida\nhello\nsee you"})
print(run.results["tag"]["result"])  # ['Ida: hello', 'Ida: see you']
print(run.outputs)                   # {'result': 2}
```

`open` returns a frozen dataclass, which gives it two outputs, `sender` and `lines`. [Several outputs](#several-outputs-and-branches) says more about that. Here is what happened:

- `open.lines` is a `Series`, and `tag` takes one line, so `tag` ran once per line. Every node after `tag` would run once per line too.
- `count` takes a `Series`, so it collects the lines. It ran once, with all the tagged lines as a list, in order.
- `open.sender` is one value, not one per line, so every call of `tag` got the same sender.

A table never starts a loop by itself. To go through a table one row at a time, return `Series[R]`, with `R` a `TypedDict`. [A node for any type](#a-node-for-any-type) has a node that does this for any table, and `syv-conductor-nodes` ships one, Split.

### A Series input collects only the loops it is outside

`count` collected the lines because it sits outside the loop over them. A node that is inside the loop can't collect it. Below, `compare` takes one `line`, so it runs once per line. Its `others` input is wired to `shout`, which also runs once per line, in the same loop. So each call of `compare` could only get the shouted version of its own line: one value, never a list. Compiling refuses it, with `type_mismatch_series_given_one_value`. If `compare` needs only its own line's shout, declare `others: str`.

To give every call all the lines, collect them first, in a node that sits outside the loop, and wire that list in. The Collect node of the companion package `syv-conductor-nodes` does exactly that:

```python
from conductor_nodes.series import collect

registry.add(collect)


@registry.node("compare", title="Compare", description="Says how many texts a line is compared with.")
def compare(line: str, others: Series[str]) -> str:
    return f"{line} against {len(others)}"


own = make_graph(open_letter, shout, compare, edges=[("open.lines", "shout.text"), ("open.lines", "compare.line"), ("shout.result", "compare.others")])
print([problem.type for problem in Graph.compile(own, registry).problems])  # ['type_mismatch_series_given_one_value']

every = make_graph(
    open_letter, shout, collect, compare,
    edges=[("open.lines", "shout.text"), ("shout.result", "series-collect.values"), ("open.lines", "compare.line"), ("series-collect.result", "compare.others")],
)
print(Graph.compile(every, registry).run({"text": "Ida\nhello\nsee you"}).results["compare"]["result"])  # ['hello against 2', 'see you against 2']
```

`series-collect` takes nothing but a `Series`, so it is outside the loop. It runs once and returns every shouted line as one list, and each call of `compare` gets that same list.

This is what lets a graph mean the same for one letter as for many. Put a node in front that returns a `Series` of letters, or use the whole graph inside another graph (a [subgraph](subgraphs.md)), and each loop in it still collects only its own items.

A subgraph puts every node inside it inside its node's loops. So a `Series` input inside a subgraph wired to a node the subgraph calls once gets one value per call, and is refused: when the subgraph's own record is compiled, and wherever the subgraph is used.

An item that is `None` is left out of a `Series[X]` input, and `Series[X | None]` keeps it. With no items at all, the input is an empty list and your node is still called. The full rules for loops and collecting are in `python -m conductor.about one-or-many`.

## Several outputs, and branches

Return a dataclass to give a node several outputs, one per field. An output whose default is `SKIPPED` is a branch, a way the value can go. On each call the node gives a value to at most one of its branches, and the others stay `SKIPPED`. A node wired to a skipped branch is skipped, and so is every node after it:

```python
from conductor import SKIPPED


@dataclass(frozen=True)
class Routed:
    large: int = SKIPPED
    small: int = SKIPPED


@registry.node("route", title="Route", description="Sends an amount down one of two paths.")
def route(amount: int) -> Routed:
    if amount >= 1000:
        return Routed(large=amount)
    return Routed(small=amount)


@registry.node("notify", title="Notify", description="Tells the manager about an amount.")
def notify(amount: int) -> str:
    return f"told the manager about {amount}"


graph = Graph.compile(make_graph(route, notify, edges=[("route.large", "notify.amount")]), registry)
print(graph.run({"amount": 5000}).results["notify"])  # {'result': 'told the manager about 5000'}
print(graph.run({"amount": 5}).results["notify"])     # {'result': None}
```

For 5, `route` gave `small` the value, so `notify` was skipped. A skipped node's outputs show as `None` in `run.results` and `run.outputs`.

`graph.outputs` is `notify`'s `result` alone. `make_graph` makes graph outputs only of the nodes no edge leaves, and an edge leaves `route`, so `route.small` is no output of the graph, though nothing reads it. To return it, name it: `outputs={"small": "route.small"}`.

Since a call never gives two branches of one node a value, both may be wired into one input. That input gets the value of whichever branch has it.

`SKIPPED` means a way not taken, and `None` means no value. So an output that may be missing beside other outputs, like the phone number an extractor didn't find next to the email it did, isn't a branch. Type it `str | None` and return `None`. A branch returned as `None` was still taken, though: beside another branch with a value, it counts as a second one. Two branches that both get a value fail the call with `invalid_output`, and so does `SKIPPED` on an output that isn't a branch. A node has one set of branches: two independent ways to go are two nodes.

Only a dataclass with no `@host_type` lists a node's outputs. A dataclass you declare with `@host_type` is a type, like `Address` above, so a node that returns one has one output, `result`, and passes the whole value on. An edge can still read one field of it:

```python
@host_type("coordinate", title="Coordinate")
@dataclass(frozen=True)
class Coordinate:
    lat: float
    lon: float


@registry.node("locate", title="Locate", description="Finds where a city is.")
def locate(city: str) -> Coordinate:
    return Coordinate(lat=56.16, lon=10.2)


@registry.node("hemisphere", title="Hemisphere", description="Says which half of the world a latitude is in.")
def hemisphere(lat: float) -> str:
    return "north" if lat >= 0 else "south"


print([output.name for output in locate.describe().outputs])  # ['result']
graph = Graph.compile(make_graph(locate, hemisphere, edges=[("locate.result.lat", "hemisphere.lat")]), registry)
print(graph.run({"city": "Aarhus"}).outputs)  # {'result': 'north'}
```

A declared class is one value, so its fields aren't branches. `SKIPPED` inside a returned value has no JSON form: the run can't be saved (`run_unwritable`), and an edge that reads that field fails the node it feeds (`invalid_edge_value`).

## A node for any type

Some nodes don't care what type a value is. They pass it on, or take it apart. Write the type as a type variable, and the node takes whatever type arrives on its edge:

```python
@registry.node("keep", title="Keep", description="Passes a value on as it is.")
def keep[T](value: T) -> T:
    return value


graph = Graph.compile(make_graph(route, keep, notify, edges=[("route.large", "keep.value"), ("keep.result", "notify.amount")]), registry)
print(graph.describe().nodes["keep"].outputs[0].type)  # int
print(graph.run({"amount": 5000}).outputs)                  # {'result': 'told the manager about 5000'}
```

`route.large` is an `int`, so in this graph `keep` takes an `int` and returns one. That is why `notify.amount` may be wired to it. In a graph where a text arrives, `keep` returns a text. Conductor checks what the node returns against the type that arrived.

A type variable can sit inside another type. This node takes a table or a list, and returns its rows or items as a `Series`:

```python
from typing import cast


@registry.node("each", title="Each", description="Gives the rows of a table, or the items of a list, one at a time.")
def each[T](value: Table[T] | list[T]) -> Series[T]:
    return cast("list[T]", value)  # conductor reads a table returned for a Series as its rows; the cast is for the type checker


graph = Graph.compile(make_graph(contacts, each, shout, edges=[("contacts.result", "each.value"), ("each.result.name", "shout.text")]), registry)
print(graph.run({"csv": "name,email\nIda,ida@a.dk\nBo,\n"}).outputs)  # {'result': ['IDA!', 'BO!']}
```

A `Table[Contact]` arrived, so `T` is `Contact`. That is why the edge may name one field of each row: `each.result.name` is the `name` of each contact. Wired to a `list[int]`, the same node returns a `Series[int]`.

A dataclass of outputs can take a type variable too:

```python
@dataclass(frozen=True)
class Checked[T]:
    passed: T = SKIPPED
    failed: T = SKIPPED


@registry.node("check", title="Check", description="Sends a value one way if it passed and the other way if it failed.")
def check[T](value: T, ok: bool) -> Checked[T]:
    return Checked(passed=value) if ok else Checked(failed=value)


print(Graph.compile(make_graph(check), registry).run({"value": "late", "ok": False}).outputs)  # {'failed': 'late', 'passed': None}
```

Only an edge decides what `T` is. Here no edge reaches `value`, so the caller's value is taken as it is. The rules are in `python -m conductor.about a-node-for-any-type`.

## Inputs and outputs that a constant decides

Some inputs or outputs can't be written in a signature, because the graph's author decides them. Say the author types in column names, and the node should have one output per name. A hook says what those outputs are. It is a plain function: its parameters name the inputs it reads, and it returns the name and type of each output:

```python
def outputs_by_name(names: list[str]) -> dict[str, type]:
    return {name: str for name in names}


@registry.node("cells", title="Cells", description="Gives each cell of a CSV line under a name of its own.", compute_outputs=outputs_by_name)
def cells(line: str, names: list[str]) -> dict[str, str]:
    return dict(zip(names, line.split(",")))


graph = Graph.compile(
    make_graph(cells, constants={"cells.names": ["name", "city"]}, outputs={"name": "cells.name", "city": "cells.city"}),
    registry,
)
print(list(graph.outputs))                   # ['city', 'name']
print(graph.run({"line": "Ida,Aarhus"}).outputs)  # {'city': 'Aarhus', 'name': 'Ida'}
```

The hook is called when the graph is compiled, with the constant given to `names`. So `names` must be a constant: no edge or caller can give it. The node returns its outputs as a dict by name, and its return type says so: `@registry.node` refuses `-> str` on it (`outputs_not_a_mapping`).

The outputs are known only once the graph is compiled, so `make_graph` can't find them by itself. You name the ones the graph returns with `outputs=`, and without it `make_graph` raises `hook_outputs_not_named`. `compute_added_inputs=` adds inputs the same way, and the function takes them as `**values`. A hook that raises doesn't stop compiling: the node gets the problem `compute_hook_failed` with your message if you raised a `NodeError`, or `compute_hook_raised` for any other exception.

Whether a field needs a hook or a type variable is in `python -m conductor.about a-node-for-any-type`, and how a hook works is in `python -m conductor.about field-hooks`.

## Asking a person

Some nodes need a person to decide. `raise Ask(prompt)` stops that call of the node and asks. `asks=True` on the `@node` says the node may ask, so a host knows before any run which graphs can pause for a person. Without it, `@node` refuses a function whose own body raises `Ask` ([`ask_not_declared_in_source`](problems.md#ask_not_declared_in_source)), and an `Ask` raised from a function the node calls fails the call ([`ask_not_declared`](problems.md#ask_not_declared)).

Your return type says what the person answers. `-> str` asks for a line of text, `-> bool` for a yes or no, `-> Annotated[str, Param(choices=["ida", "bo"])]` for one of those values, `-> Color` for one of an enum's members (an enum declared with `@host_type`, as any class of your own), and a dataclass for a form, with one question per field. The type is the node's contract, so fixed choices go on it, and `describe()` shows them before any run. `Ask` carries what only the run knows: choices that come from the run's data, and a proposed `default`.

```python
from conductor import Ask


@registry.node("assign", title="Assign", description="Asks a person who should take a ticket.", asks=True)
def assign(ticket: str) -> Annotated[str, Param(choices=["ida", "bo"])]:
    raise Ask(f"Who takes {ticket}?", default="ida")


run = Graph.compile(make_graph(assign), registry).run({"ticket": "T-1"})
print(run.status)                           # waiting
print(run.waiting[0].questions[0].choices)  # (Choice(id='ida', title='ida'), Choice(id='bo', title='bo'))
```

When the choices are data, such as a team that changes from run to run, they come in on an input, and the `Ask` offers them:

```python
@registry.node("pick", title="Pick", description="Asks a person which member of a team takes a ticket.", asks=True)
def pick(ticket: str, team: list[str]) -> str:
    raise Ask(f"Who takes {ticket}?", choices=team)


run = Graph.compile(make_graph(pick), registry).run({"ticket": "T-1", "team": ["ida", "al"]})
print(run.waiting[0].questions[0].choices)  # (Choice(id='ida', title='ida'), Choice(id='al', title='al'))
```

A form works the same way, with `Param(choices=)` on a field. Since a form has several questions, an `Ask`'s `choices` and `default` are given by field name, for only the fields they are for. A waiting run can be saved as JSON and answered much later, in another process:

```python
from conductor import Choice, Run


@dataclass(frozen=True)
class Review:
    approved: bool
    owner: Annotated[str, Param(choices=[Choice("ida", "Ida B."), Choice("bo", "Bo K.")])]
    note: Annotated[str | None, Param(widget=Textarea())] = None


@registry.node("review", title="Review", description="Asks a person to review a proposal.", asks=True)
def review(proposal: str) -> Review:
    raise Ask(f"Review: {proposal}")


graph = Graph.compile(make_graph(review), registry)
run = graph.run({"proposal": "Move the server"})
stored = run.to_json()  # store this, and the person may answer next week

run = Run.from_json(stored, graph)  # later, with the same graph compiled again
(waiting,) = run.waiting
print([question.name for question in waiting.questions])  # ['approved', 'owner', 'note']

run = run.with_answers(waiting, {"approved": True, "owner": "ida"}).resume()  # keep the run it returns
print(run.outputs)  # {'approved': True, 'note': None, 'owner': 'ida'}
```

The answer becomes the node's outputs, and the node isn't called again for that call. `note` has a default, so the answer could leave it out. An answer that doesn't fit its questions is refused before anything runs, with every fault at once: one [`invalid_answer`](problems.md#invalid_answer) per wrong question, whose `ctx.errors` has pydantic's codes (`bool_type`, `not_a_choice`, `extra_forbidden`):

```python
from conductor import InvalidCallError

try:
    run = Run.from_json(stored, graph)
    run.with_answers(run.waiting[0], {"approved": "yes", "owner": "al", "colour": "red"})
except InvalidCallError as refused:
    for problem in refused.problems:
        print(problem.msg)
    # That isn't a valid answer to 'approved': Input should be a valid boolean.
    # That isn't a valid answer to 'owner': the value is not "ida" or "bo".
    # That isn't a valid answer to 'colour': Extra inputs are not permitted.
```

An `Ask` whose choices or default don't fit fails the node the same way, with one [`invalid_ask`](problems.md#invalid_ask) that lists every fault.

`Ask` is a `BaseException`, not an `Exception`, so an `except Exception` in your node doesn't catch it by mistake. What each return type asks, and the rules for answers, are in `python -m conductor.about asking-a-person`.

### Using the answer in another node

The answer is always the outputs of the node that asked. When another node needs it, give the question a node of its own and wire its output on. The asking node doesn't have to ask every time. Here it approves small bills by itself:

```python
class Bill(TypedDict):
    number: str
    amount: int


@registry.node("approval", title="Approval", description="Approves a small bill, and asks a person about a large one.", asks=True)
def approval(bill: Bill) -> bool:
    if bill["amount"] <= 500:
        return True
    raise Ask(f"Pay {bill['number']}, {bill['amount']} kr.?")


@registry.node("pay", title="Pay", description="Pays a bill that is approved, and holds the rest.")
def pay(bill: Bill, ok: bool) -> str:
    return f"{'paid' if ok else 'held'} {bill['number']}"


approve_and_pay = make_graph(approval, pay, edges=[("approval.result", "pay.ok")], inputs={"bill": ["approval.bill", "pay.bill"]})
graph = Graph.compile(approve_and_pay, registry)
print(graph.run({"bill": {"number": "B-1", "amount": 100}}).outputs)  # {'result': 'paid B-1'}

run = graph.run({"bill": {"number": "B-2", "amount": 900}})
print([(question.name, question.type) for question in run.waiting[0].questions])  # [('result', 'bool')]
print(run.with_answers(run.waiting[0], False).resume().outputs)  # {'result': 'held B-2'}
```

`inputs={"bill": [...]}` gives both nodes one graph input, `bill`. `pay` is called once, with `ok` a real `bool`, so `pay(bill, ok=True)` is its unit test.

To offer the two nodes as one entry in a palette, turn the graph into a node: `registry.subgraph("approve_and_pay", title="Approve and pay", description="...")(approve_and_pay)`. The question then waits on `approval`, inside the subgraph. [A graph inside a graph](subgraphs.md) shows how.

## When something fails

Raise `NodeError` for a failure you can explain to a person. The host gets your message as you wrote it, and you choose the `code`:

```python
from conductor import NodeError


@registry.node("parse", title="Parse", description="Reads a text as a whole number.")
def parse(text: str) -> int:
    if not text.isdigit():
        raise NodeError("The text is not a number.", code="not_a_number")
    return int(text)


run = Graph.compile(make_graph(parse), registry).run({"text": "twelve"})
if run.cause is not None:  # a run that failed, stopped or timed out says why
    print(run.status, run.cause.type, run.cause.msg)  # failed not_a_number The text is not a number.
```

`run.cause` is the problem that ended the run, and `None` when nothing went wrong. For anything the host needs beyond the message, pass `ctx=`, a dict of plain JSON values: text, numbers, `True` and `False`, `None`, and lists and dicts of them. The host reads it as `run.cause.ctx`.

Any other exception fails the node too. Its traceback goes to the `conductor` logger, and the host gets a fixed message with only the exception's class name, in `cause.ctx["error"]`. That way a value your node handled never leaks out to the host.

### Retries

Raise `ExternalError`, a kind of `NodeError`, when something outside failed: a timeout, a rate limit, a server error. Only that is retried, as often as the node's `Policy` says. A bug fails the same way every time, so it is never retried:

```python
from conductor import ExternalError, Policy

attempts: list[str] = []


@registry.node("fetch", title="Fetch", description="Fetches a page from an address.", policy=Policy(retries=2, retry_delay=0.01))
def fetch(url: str) -> str:
    attempts.append(url)
    if len(attempts) < 3:
        raise ExternalError("The server did not answer.")
    return "<html>"


run = Graph.compile(make_graph(fetch), registry).run({"url": "https://example.com"})
print(run.status, len(attempts))  # complete 3
```

The call failed twice and worked the third time. `Policy` has a few more settings:

- `retry_on` lists exception classes of your client to retry as well, as if they were an `ExternalError`.
- `attempt_timeout` gives up on a call that takes too long. The call isn't stopped, though: it keeps running on its thread, and the run goes on without it. So give a slow client a timeout of its own too.
- `concurrency` is how many calls of the node may run at once, across the whole process: every graph that uses the node, and every run going on. `Policy(concurrency=4)` keeps a service's quota however many runs a server has open. Don't set it on a node whose function runs a graph that uses the same node again, since the inner run would wait for ever.

By default a node has no such limit, so a server with ten runs open may call it ten times at once. The rules are in `python -m conductor.about versions-and-policies` and `python -m conductor.about retries-and-timeouts`.

## A node that costs nothing

A host often wants to tell a person what a run will cost before it starts, such as "this will make 3 calls to a language model". But how many calls a node gets depends on how many items the node before it returns. The host only knows that once that node has run.

So mark a node whose call is free, and touches nothing outside the run, with `Policy(costs_nothing=True)`. Think of a node that hands on an upload's files, or splits a table. The host can then run just those nodes first:

```python
@registry.node("files", title="Files", description="The files a person uploaded, one item each.", policy=Policy(costs_nothing=True))
def files(names: list[str]) -> Series[str]:
    return names


@registry.node("summarize", title="Summarize", description="Summarizes a file.")
def summarize(name: str) -> str:
    return f"A summary of {name}"  # the call that costs something


ready = Graph.compile(make_graph(files, summarize), registry).with_inputs({"names": ["a.pdf", "b.pdf", "c.pdf"]})
estimated = ready.resume(only_costs_nothing=True)

print(estimated.status)                       # ready
print(estimated.progress.nodes["summarize"])  # finished=0 waiting=0 skipped=0 none_input=0 known=3 total=3
print(estimated.resume().progress.finished)   # 4
```

`graph.with_inputs` makes a run without starting it. `resume(only_costs_nothing=True)` calls only the free nodes, and `run.progress` then says `summarize` will be called 3 times. The next `resume()` calls the rest, and doesn't call `files` again.

A node that takes the run's context (the next section) can't be marked: it uses the host's objects, so its call isn't free ([`costs_nothing_takes_context`](problems.md#costs_nothing_takes_context)). What `run.progress` counts is in `python -m conductor.about running`.

## What the host gives the run

Some things come from the host, not from the graph: who is running it, a database, a client. They are the run's context. The host passes one object of its own class to each run, and it is never saved with the run. Name the class on the registry, and a node that needs it takes it as its first parameter, typed `Context[...]`:

```python
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

signed = Graph.compile(make_graph(sign), signing).run({"text": "Done"}, context=ida)
print(signed.outputs)  # {'result': 'Done, Ida'}
```

`ctx` is not an input: no edge reaches it, and no caller fills it in. So a call whose inputs are the same as in an earlier run reuses that outcome, whatever the context is. So `signed.again` with Bo's context gives back `'Done, Ida'`, the outcome made for Ida. A node whose result depends on the context, as `sign`'s does, takes that value as an input instead, or says `Policy(reuse_when_inputs_match=False)`, and is then called on every run. The host passes `context=` again each time it goes on with a run (`run.resume`, `run.stream` and `run.again`). A type checker refuses a context of another class, and so does the run, with `invalid_context`.

A node from a package of nodes doesn't know the host's class. It names a `Protocol` with just the members it reads instead:

```python
from typing import Protocol


class HasCaller(Protocol):
    @property
    def caller(self) -> Caller: ...


@signing.node("greet_caller", title="Greet the caller", description="Greets the person running it.")
def greet_caller(ctx: Context[HasCaller]) -> str:
    return f"Hello {ctx.caller.name}"


print(Graph.compile(make_graph(greet_caller), signing).run(context=ida).outputs)  # {'result': 'Hello Ida'}
```

Write each member as a read-only property, as here: a frozen dataclass can't satisfy a `Protocol` whose member is a plain attribute. Compiling checks each node's `Context[...]` against the registry's class. A node that asks for a member the class lacks gets the problem `context_does_not_fit_missing_member`. The rules are in `python -m conductor.about context`.

## A second version

Each node in a graph record says which version of the node it uses. To change a node's inputs or outputs, write a new version that names the one it replaces, and say how a saved graph moves to it:

```python
from conductor import Upgrade


@node("greet", title="Greet", description="Greets a person by name.")
def greet_v1(name: str) -> str:
    return f"Hello {name}"


@registry.node("greet", title="Greet", description="Greets a person by name.", replaces=greet_v1, upgrade=Upgrade(renamed_inputs={"name": "person"}))
def greet(person: str, greeting: str = "Hello") -> str:
    return f"{greeting} {person}"


saved = make_graph(greet_v1, constants={"greet.name": "Ida"})
print(Graph.compile(saved, registry).run().outputs)  # {'result': 'Hello Ida'}: it still runs version 1

moved = registry.upgrade(saved, "greet")
print(moved.nodes["greet"].version, moved.nodes["greet"].inputs)  # 2 {'person': InputEntry(constant='Ida')}
```

Only the newest version goes on the registry. A saved graph keeps running the version it names, because compiling never upgrades. The host calls `registry.upgrade` and shows the graph's author the result. To tell an editor that a version is going away, and which node to use instead, pass `deprecation=Deprecation(...)`. The rules are in `python -m conductor.about versions-and-policies`.

A new version is also how a run made from a previous one stops reusing a node's results after you change something its function reads from outside, such as a template or a helper. Reuse sees only the function's own source (`python -m conductor.about running-again`).

## Test the whole registry

Calling a node tests its body. `check_registry` tests what a call can't show: an upgrade that breaks a saved graph, a hook that fails, a body that changes the value it was given, an example that fails. It returns every problem at once:

```python
from conductor.testing import check_registry

problems = check_registry(registry, examples={
    "shout": [{"text": "hello"}],
    "open": [{"text": "Ida\nhello"}],
    "tag": [{"line": "hello", "sender": "Ida"}],
    "count": [{"texts": ["hello"]}],
    "route": [{"amount": 5}],
    "cells": [{"line": "Ida,Aarhus", "names": ["name", "city"]}],
    "assign": [{"ticket": "T-1"}],
    "review": [{"proposal": "Move the server"}],
    "parse": [{"text": "12"}],
    "fetch": [{"url": "https://example.com"}],
    "greet": [{"person": "Ida"}],
})
print(problems)  # []
print(check_registry(signing, examples={"sign": [{"text": "Done"}]}, context=ida))  # []
```

Put it in your test suite as `assert check_registry(registry, examples=...) == []`. A node's body is called only when you give it an example, since a node might send an email. A node with no example is still checked for everything that needs no call, such as its upgrades and its hooks. `python -m conductor.about conductor-testing` lists every check.
