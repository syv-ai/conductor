# Problems

Everything conductor reports has a stable code. A `Problem`'s `type` is one, and so is a `NodeDeclarationError`'s `code`. This page has every code: when you get it, an example that makes it, and how to fix it. A `Problem` has four fields, as a pydantic error has: `type` (the code), `loc` (where it is), `msg` (the message, for a person) and `ctx` (the values the message is made from). A host that writes its own messages translates by `type` and fills its sentence from `ctx`. `PROBLEMS.codes` is every code conductor's core writes; a code not in it is a node's own, from `NodeError(code=)`.

This page is printed from the code: each kind of problem from the function that declares it (`python -m conductor.about problems` prints the page). Every example runs after this setup, and a test checks that each one makes the problem it is listed under:

```python
import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, TypedDict

import polars as pl
from conductor import (
    Ask, Context, EndedEarly, Graph, GraphRecord, InvalidCallError, NodeDeclarationError, NodeError, NodeRegistry,
    Param, Policy, Problem, Run, Series, Table, Upgrade, host_type, make_graph, node, subgraph,
)
from conductor.testing import check_registry

registry = NodeRegistry()


@registry.node("upper", title="Uppercase", description="Turns a text into capital letters.")
def upper(text: str) -> str:
    return text.upper()


@registry.node("greet", title="Greet", description="Greets a person by name.")
def greet(name: str, greeting: str) -> str:
    return f"{greeting} {name}"


@registry.node("count", title="Count", description="Returns the number it is given.")
def count(n: int) -> int:
    return n


@registry.node("words", title="Words", description="Returns the words of a text, one item each.")
def words(text: str) -> Series[str]:
    return text.split()


@registry.node("join", title="Join", description="Joins texts into one.")
def join(texts: Series[str]) -> str:
    return "+".join(texts)


@dataclass(frozen=True)
class Route:
    short: Annotated[str | None, Param(exclusive_group="length")] = None
    long: Annotated[str | None, Param(exclusive_group="length")] = None


@registry.node("route", title="Route", description="Sends a text down the short or the long branch.")
def route(text: str) -> Route:
    return Route(short=text) if len(text) < 4 else Route(long=text)


@registry.node("approve", title="Approve", description="Asks a person to approve an amount in kroner.")
def approve(amount: int) -> bool:
    raise Ask(f"Approve {amount} kr.?")


@dataclass(frozen=True)
class Caller:
    name: str


@registry.node("who", title="Who", description="Returns the name of the person running it.")
def who(caller: Context[Caller]) -> str:
    return caller.name


def by_names(names: str) -> dict[str, type]:
    if not names:
        raise ValueError("no names")
    return {name: str for name in names.split(",")}


@registry.node("pick", title="Pick", description="Returns one output per name.", compute_outputs=by_names)
def pick(names: str) -> dict[str, str]:
    return {name: name for name in names.split(",")}


class Contact(TypedDict):
    name: str
    email: str


@registry.node("contacts", title="Contacts", description="Reads contacts from a text, a name and an email on each line.")
def contacts(text: str) -> Table[Contact]:
    return pl.DataFrame([dict(zip(("name", "email"), line.split(), strict=True)) for line in text.splitlines()])


def problems(record: dict) -> list[Problem]:
    """The problems of a graph record, compiled with ``registry``."""
    return Graph.compile(GraphRecord.model_validate(record), registry).problems


def refused(call: Callable[[], object]) -> list[Problem]:
    """The problems a call is refused for: it raises ``InvalidCallError``, which lists them."""
    try:
        call()
    except InvalidCallError as error:
        return error.errors()
    raise AssertionError("the call was not refused")


def why_it_stopped(run: Run) -> Problem:
    """The problem a run ended early on."""
    match run.state:
        case EndedEarly(cause=cause):
            return cause
    raise AssertionError(f"the run is {run.status}")
```

<!-- problems: conductor.PROBLEMS -->

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
    print(str(error).splitlines()[0])  # @node('narrow') on narrow: number: the limits {'ge': 'a'} don't fit int, so change one: Error building "json-or-python" validator:
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
    print(error)  # two nodes have the id 'upper': doc_problems.upper and doc_problems.upper_again; ids are unique, so swap one for the other with registry.replace(...)
```

**Fix.** For a node the registry already holds, swap one for the other with `registry.replace(newer)`, or give the new node an id of its own. The older versions of a chain are plain `@node` definitions that are never registered; only the newest is declared on the registry, with `replaces=`.

### `duplicate_type_id`

```python
@host_type("money", title="Amount")
@dataclass
class Amount:
    kroner: int


@host_type("money", title="Price")
@dataclass
class Price:
    kroner: float


try:
    NodeRegistry(types=[Amount, Price])
except NodeDeclarationError as error:
    print(error.code)  # duplicate_type_id
    print(error)  # type id 'money' is declared by both doc_problems.Amount and doc_problems.Price; one registry holds one type per id
```

**Fix.** Give one of the classes another id in its `@host_type`.

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

### `context_misplaced`

```python
class Session: ...

try:
    @node("maybe", title="Maybe", description="Passes the text on as it is.")
    def maybe(session: Context[Session] | None = None) -> str:
        return ""
except NodeDeclarationError as error:
    print(error.code)  # context_misplaced
    print(error)  # @node('maybe') on maybe: session: Context is a whole parameter the host always gives, name: Context[Session]; anywhere else, a caller would fill it
```

**Fix.** Write the parameter as `name: Context[Session]`, never inside `| None`, a list or a return type. The host
gives that value on every run, so a node can trust it; anywhere else it would be an input a caller fills.

### `context_not_a_class`

```python
try:
    @node("listed", title="Listed", description="Hands back an empty text.")
    def listed(names: Context[list[str]]) -> str:
        return ""
except NodeDeclarationError as error:
    print(error.code)  # context_not_a_class
    print(error)  # @node('listed') on listed: names: Context names the class of the value the run is given, Context[Session]
```

**Fix.** Name a class, `Context[Session]`: the run is given its value by that class, `context={Session: ...}`.

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

### `pydantic_field_in_hint`

```python
from pydantic import Field

try:
    @node("repeat", title="Repeat", description="Repeats a text.")
    def repeat(text: str, times: Annotated[int, Field(ge=0)] = 1) -> str:
        return text * times
except NodeDeclarationError as error:
    print(error.code)  # pydantic_field_in_hint
    print(error)  # @node('repeat') on repeat: times: pydantic's Field(...) in the hint; an input or output takes its title, description and limits from Param, so write Annotated[int, Param(ge=0)] where you had Annotated[int, Field(ge=0)]
```

**Fix.** Write the title, the description and the limits on `Param`, which takes them by the same names: `Annotated[int, Param(ge=0)]`. There is one way to write them on an input or an output, so `Field(...)` is refused wherever it is written in the hint: on the parameter, on the return type, on an item (`Series[Annotated[int, Field(ge=0)]]`, `list[Annotated[int, Field(ge=0)]]`) and on a column of a table's row type. A type that has its limits is still a type: `PositiveInt`, `constr(min_length=1)` and `Annotated[int, annotated_types.Gt(0)]` are read as pydantic reads them, and `Field(...)` on the fields of your own model or dataclass is that class's.

A `Field(discriminator="kind")` on a union is refused with a message of its own, which shows pydantic's `Discriminator` to write: `Annotated[Cat | Dog, Discriminator("kind")]`.

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

### `type_declaration_invalid`

```python
try:
    @host_type("str", title="My text")
    class MyText(str):
        pass
except NodeDeclarationError as error:
    print(error.code)  # type_declaration_invalid
    print(str(error).partition(";")[0])  # @host_type('str'): id: Value error, 'str' is the id of one of conductor's own types
```

**Fix.** Give the type an id that is one word (letters, digits, `_`, `.` or `-`) and is not the id of one of conductor's own types, a title that is not empty, and only classes in `accepts`.

### `type_not_carried`

```python
try:
    @host_type("plain", title="Plain")
    class Plain:
        def __init__(self, path: str) -> None:
            self.path = path
except NodeDeclarationError as error:
    print(error.code)  # type_not_carried
    print(error)  # @host_type('plain') on Plain: pydantic can't validate it or write it as JSON; make it a pydantic model or a dataclass (with @host_type above @dataclass), or give it its own __get_pydantic_core_schema__
```

**Fix.** Use a type pydantic can validate and write as JSON: a pydantic model, a dataclass or a subclass of a builtin (`class Text(str)`), with `@host_type` above `@dataclass`. A class that needs its own way gives its own `__get_pydantic_core_schema__`.

### `type_not_named`

```python
@dataclass
class Unnamed:
    kroner: int


try:
    @node("unnamed-type", title="Unnamed type", description="Hands back an empty text.")
    def unnamed_type(values: list[Unnamed]) -> str:
        return ""
except NodeDeclarationError as error:
    print(error.code)  # type_not_named
    print(str(error).partition(" is the type")[2])  #  of a field and is not declared; write @host_type("...", title="...") above the class
```

**Fix.** Write `@host_type("...", title="...")` above the class. A subclass does not inherit its parent's declaration.

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

**Fix.** Put the widget on the field's `Param`: `Annotated[str, Param(widget=Textarea())]`. Other metadata in `Annotated` is pydantic's and is read as pydantic reads it, except `Field(...)` ([`pydantic_field_in_hint`](#pydantic_field_in_hint)); a widget there would be passed by, and the field would show no control.

## `check_registry`

Returned by `conductor.testing.check_registry`, in a test run. A host never meets these, and they are not in `PROBLEMS.codes`. `loc` is the node's id and its version.

<!-- problems: conductor.testing.problems.REGISTRY_CHECK_PROBLEMS -->
