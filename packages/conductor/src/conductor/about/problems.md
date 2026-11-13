# Problems

Everything conductor reports has a stable code. A `Problem`'s `type` is one, and so is a `NodeDeclarationError`'s `code`. This page has every code: when you get it, an example that makes it, and how to fix it. A `Problem` has four fields, as a pydantic error has: `type` (the code), `loc` (where it is), `msg` (the message, for a person) and `ctx` (the values the message is made from). A host that writes its own messages translates by `type` and fills its sentence from `ctx`. `PROBLEMS.codes` is every code conductor's core writes; a code not in it is a node's own, from `NodeError(code=)`.

This page is printed from the code: each kind of problem from the function that declares it (`python -m conductor.about problems` prints the page). Every example runs after this setup, and a test checks that each one makes the problem it is listed under:

```python
import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Protocol, TypedDict

import polars as pl
from conductor import (
    Ask, Asked, Context, EndedEarly, Graph, GraphRecord, InvalidCallError, NodeDeclarationError, NodeError, NodeRegistry,
    Param, Policy, Problem, Run, SKIPPED, Series, Table, Upgrade, host_type, make_graph, node, subgraph,
)
from conductor.testing import check_registry


@dataclass(frozen=True)
class Caller:
    name: str


@dataclass(frozen=True)
class HostContext:
    caller: Caller


registry = NodeRegistry(context_type=HostContext)


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
    short: Annotated[str, Param(exclusive_group="length")]
    long: Annotated[str, Param(exclusive_group="length")]


@registry.node("route", title="Route", description="Sends a text down the short or the long branch.")
def route(text: str) -> Route:
    return Route(short=text, long=SKIPPED) if len(text) < 4 else Route(short=SKIPPED, long=text)


@registry.node("approve", title="Approve", description="Asks a person to approve an amount in kroner.", asks=True)
def approve(amount: int) -> bool:
    raise Ask(f"Approve {amount} kr.?")


@registry.node("who", title="Who", description="Returns the name of the person running it.")
def who(ctx: Context[HostContext]) -> str:
    return ctx.caller.name


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


def problems(record: dict | GraphRecord, nodes: NodeRegistry = registry) -> list[Problem]:
    """The problems of a graph record, compiled with ``nodes``."""
    return Graph.compile(GraphRecord.model_validate(record), nodes).problems


def refused(call: Callable[[], object]) -> list[Problem]:
    """The problems a call is refused for: it raises ``InvalidCallError``, which lists them."""
    try:
        call()
    except InvalidCallError as error:
        return error.errors()
    raise AssertionError("the call was not refused")


def declaration_refused(declare: Callable[[], object]) -> Problem:
    """The problem a declaration is refused for: it raises ``NodeDeclarationError``, which carries it."""
    try:
        declare()
    except NodeDeclarationError as error:
        return error.problem
    raise AssertionError("the declaration was not refused")


def why_it_stopped(run: Run) -> Problem:
    """The problem a run ended early on."""
    match run.state:
        case EndedEarly(cause=cause):
            return cause
    raise AssertionError(f"the run is {run.status}")
```

<!-- problems: conductor.PROBLEMS -->

<!-- problems: conductor.DECLARATION_PROBLEMS -->

## `check_registry`

Returned by `conductor.testing.check_registry`, in a test run. A host never meets these, and they are not in `PROBLEMS.codes`. `loc` is the node's id and its version.

<!-- problems: conductor.testing.problems.REGISTRY_CHECK_PROBLEMS -->
