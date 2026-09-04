---
name: add-node
description: Use when adding a new conductor node — any class extending NodeDefinition, or when the user asks how to expose a function to a flow. Covers the typed run signature, widget selection, the type vocabulary, multi-output records, branching, versions with a Policy, and the roster hooks. Triggers on phrases like "add a node", "register a node", "new flow node", "expose X as a node".
---

# Adding a conductor node

Use this skill when the user wants to create a new node in a project that depends on the [`syv-conductor`](https://pypi.org/project/syv-conductor/) library.

## First — pull the packaged library reference

Conductor ships its own reference text inside the wheel. **Run these before writing code** so your advice matches the installed version, not stale training data:

```bash
python -m conductor.about sections          # list section slugs
python -m conductor.about                   # the whole reference
```

Programmatic equivalent: `from conductor.about import get_content, list_sections, get_section`.

## Core pattern

A node is a class. It declares its identity and what a palette shows, and implements `run`, whose typed signature **is** its interface — validation, execution and the palette all read that one declaration.

```python
from typing import Annotated
from conductor import NodeDefinition, NodeRegistry, Result
from conductor.widgets import Number as NumberWidget, Text as TextWidget
from myapp.types import Number, Text          # the host's own DTypes

class Greet(NodeDefinition):
    id = "greet"                              # the registry id; a placement stores it
    title = "Greet"
    description = "Produces a greeting."
    category = "text"                         # where the palette files it

    def run(
        self,
        name: Annotated[Text, TextWidget(title="Name")],
        times: Annotated[Number, NumberWidget(title="Times", integer_only=True)] = Number(1),
    ) -> Annotated[Text, Result(title="Greeting")]:
        return Text("hello " + (name + " ") * int(times))

registry = NodeRegistry()     # usually one per host, populated at import
registry.register(Greet)
```

Rules:

- `id`, `title`, `description` and `category` are required; the class is checked the moment it is defined.
- Every parameter a cable can reach declares a `DType` (or `Any`) **and** a widget inside `Annotated[...]`. There is no default widget for any type. A default value makes the input optional.
- Parameter order is UI order.
- The return annotation declares the output. A `DType` with a `Result` is one output named `result`.
- Return a value of the declared type — `Text(...)`, never a bare `str` — because a value arrives downstream as the type the wire carried.
- `title` and `description` on the widget are the field's; `show_handle=False` closes an input to cables (it may then declare any pydantic-validatable type).

## The type vocabulary

Every wire value has a `DType`. Conductor declares none — the host does, once, and every node imports them:

```python
from conductor import DType

class Text(DType, str):
    id = "text"
    title = "Text"

class Number(DType, float):
    id = "number"
    title = "Number"
```

`Series[X]` (from `conductor`) is the one collection: a parameter declared `Series[Text]` receives the whole series; a series output is returned as a plain list. Use `Any` only for a value the node routes without reading — an `Any` output requires a `compute_outputs` override. The standard library's vocabulary (`conductor_nodes.types`: `Text`, `Number`, `Flag`, `Json`) is fine for a notebook; a host declares its own.

## Multi-output

Several outputs are a frozen dataclass whose fields are the outputs; `run` returns an instance. The field names are the output names — nothing is positional:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Parts:
    head: Annotated[Text, Result(title="Head")]
    tail: Annotated[Text, Result(title="Tail")]

class Split(NodeDefinition):
    id = "split"
    title = "Split"
    description = "Splits a string."
    category = "text"

    def run(
        self,
        s: Annotated[Text, TextWidget(title="Input")],
        sep: Annotated[Text, TextWidget(title="Separator")] = Text(","),
    ) -> Parts:
        head, _, tail = s.partition(sep)
        return Parts(head=Text(head), tail=Text(tail))
```

## Branching

A node that takes one of two branches returns `SKIPPED` on the other; whatever is wired to that output does not run. Outputs that are exclusive alternatives share a `choice`, so an editor knows exactly one arrives:

```python
from conductor import SKIPPED

@dataclass(frozen=True)
class Branches:
    if_true: Annotated[Text, Result(title="If true", choice="when")]
    if_false: Annotated[Text, Result(title="If false", choice="when")]
```

There is no role, flag or marker on the class: the engine acts on the value.

## Versions, retries, timeout

Several versions live in one class; the current one is the method named `run`. Retries and timeout belong to the version, on its `Policy`. Raise `NodeConnectionError` (from `conductor.errors`) to mark a failure as transient:

```python
from conductor import Policy, deprecated, upgrade, version
from conductor.errors import NodeConnectionError

class Fetch(NodeDefinition):
    id = "fetch"
    title = "Fetch"
    description = "Fetch a URL."
    category = "http"

    @version(1)
    @deprecated(header="Use version 2", migration="`url` is now `address`.")
    def run_v1(self, url: Annotated[Text, TextWidget(title="URL")]) -> Annotated[Text, Result(title="Body")]:
        return self.run(address=url)

    @version(2, policy=Policy(retries=3, delay=0.5, timeout=10.0))
    def run(self, address: Annotated[Text, TextWidget(title="Address")]) -> Annotated[Text, Result(title="Body")]:
        try:
            return Text(_http_get(address))
        except TimeoutError as e:
            raise NodeConnectionError(str(e)) from e

    @upgrade(1, 2)
    def _rename(values: dict) -> dict:
        return {"address": values["url"]}
```

`NodeValidationError` is never retried. A registered node numbers its versions from 1 with no holes.

## Roster hooks

When the inputs or outputs of one *placement* depend on its configuration (a mode dropdown that adds a field, a sheet whose header row names the outputs), override `compute_inputs(self, declared, values)` or `compute_outputs(self, declared, values, arriving)` and return the `Input` / `Output` tuple that placement has. A `run` whose outputs are computed declares `-> Mapping[str, Any]` and returns a dict naming exactly them. This is the only home for placement-specific shape.

## Where to register

Most projects keep one module-level `registry = NodeRegistry()` and register classes in the modules that define them; `discover_nodes("myapp.nodes", registry)` (from `conductor.registry.discovery`) imports a package so those registrations run. Pull in the standard library with `conductor_nodes.register_all(registry, categories=[...])`. Ids are unique across a registry — two classes under one id raise.

## Checklist before shipping a node

- [ ] `id`, `title`, `description`, `category` declared; the id is unique in the target registry.
- [ ] Every handle parameter has a `DType` (or `Any`) and a widget; every return has a `Result` (or is a record of them).
- [ ] `run` returns the declared dtypes, never bare builtins.
- [ ] External calls raise `NodeConnectionError` and the version's `Policy` sets `retries`.
- [ ] Tests cover: happy path, invalid input, retryable failure, each output of a record.

## When your advice diverges from the installed version

The library is the source of truth. If a user reports behavior that contradicts this skill, run `python -m conductor.about` first and trust the output. Skill docs are a shortcut; the packaged reference is authoritative.
