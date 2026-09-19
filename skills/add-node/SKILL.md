---
name: add-node
description: Declares a conductor node as a NodeDefinition subclass whose typed run signature is its interface. Use when adding or changing a node in a project that depends on syv-conductor, when a class extends NodeDefinition, when choosing a widget or a DType for a parameter, when a node needs several outputs, a branch, a version, retries, a value the run supplies, or a person's answer, or on "add a node", "register a node", "expose X as a node".
---

# Adding a conductor node

## Overview

A node is a class. Its typed `run` signature **is** its interface: validation, execution and the palette all read that one declaration, so nothing is declared twice. For placing nodes in a graph and running it, use the **create-graph** skill.

## First, read the installed reference

Conductor ships its reference in the wheel. Read it before writing code, so the code matches the installed version and not what you remember:

```bash
python -m conductor.about sections     # the section slugs
python -m conductor.about node         # one section, by prefix: node, versions, field, types, rows, legs
```

## Core pattern

```python
from typing import Annotated

from conductor import NodeDefinition, NodeRegistry, Param, Result
from conductor.widgets import NumberWidget, TextWidget
from myapp.types import Number, Text          # the host's own DTypes


class Greet(NodeDefinition):
    id = "greet"                              # the registry id; a graph pins it with a version
    title = "Greet"
    description = "Produces a greeting."
    category = "text"                         # where the palette files it

    def run(
        self,
        name: Annotated[Text, Param(title="Name", widget=TextWidget())],
        times: Annotated[Number, Param(title="Times", widget=NumberWidget(integer_only=True))] = Number(1),
    ) -> Annotated[Text, Result(title="Greeting")]:
        return Text(" ".join(f"hello {name}" for _ in range(int(times))))


registry = NodeRegistry()
registry.register(Greet)
```

## Rules the class is checked against when it is defined

- `id`, `title`, `description` and `category` are required.
- Every parameter an edge can reach is `Annotated[DType, Widget(title=...)]`, or `Any` for a value the node routes without reading. There is no default widget for any type. A default value makes the input optional.
- `show_handle=False` on the widget closes an input to edges; it may then declare any pydantic-validatable type.
- The return annotation declares the outputs. `Annotated[X, Result(title=...)]` is one output named `result`.
- `run` is a plain function: `async def run` is refused, since the engine calls `run` in a worker thread.
- Return the declared type, `Text(...)` and not a bare `str`.
- Every call gets a fresh instance: keep nothing on `self` between calls.

## Quick reference

| The node needs… | Write | Details |
|---|---|---|
| Its own value types | `class Text(DType, str): id = "text"; title = "Text"`, once per host | REFERENCE.md → Types |
| Several outputs | a frozen dataclass of `Annotated[X, Result(...)]` fields as the return type | REFERENCE.md → Outputs |
| A whole list at once | a `Series[X]` parameter with `Param(title=...)` alone (an edge fills it) or with `widget=List()` | REFERENCE.md → Series |
| To run once per item | nothing: a series on a scalar input iterates the node | create-graph |
| A branch | return `SKIPPED` on the output not taken; share a `choice` | REFERENCE.md → Branching |
| A person's answer | `-> X \| Asks`, and return `Asks(questions=(Input(...),))` | REFERENCE.md → Asking |
| A clock, a client, the caller | `Annotated[T, FromRun()]`; the host passes `execute(from_run={T: value})` | REFERENCE.md → FromRun |
| Retries or a timeout | `@version(1, policy=Policy(retries=3, delay=0.5, timeout=10))` | REFERENCE.md → Versions |
| A second version | `@version(n)` on every version, `run` included; `@upgrade(a, b)` | REFERENCE.md → Versions |
| Fields that depend on configuration | override `compute_inputs` / `compute_outputs` | REFERENCE.md → Field hooks |
| Every connected name as an input | `def run(self, **inputs: Single)` | REFERENCE.md → Open interfaces |

## Where to register

One module-level `registry = NodeRegistry()` per host, with each module registering the classes it defines; `discover_nodes("myapp.nodes", registry)` (`conductor.registry.discovery`) imports a package so those registrations run. The standard nodes: `conductor_nodes.registry(categories=["text", "math"])` builds a registry of them, and `conductor_nodes.register_all(registry)` adds them to one you have. Two classes under one id raise. Printed, `registry` shows each node's declaration, and a record shows only the fields not at their default.

## Common mistakes

| Mistake | Fix |
|---|---|
| A parameter typed `str` or `Text` with no widget | `Annotated[Text, Param(title="Text", widget=Textarea())]`; the class raises `TypeError` otherwise |
| Returning `"done"` from a node declared `-> Annotated[Text, …]` | `return Text("done")` |
| `@version(1)` on `run_v1` and a plain `def run` beside it | mark `run` too: `@version(2)` |
| Retrying with a loop inside `run` | put `retries` on the version's `Policy`; raise `ExternalFailure` for a transient failure, or name the client's classes in `retry_on` |
| A mode flag that changes what `run` returns | declare the outputs, or compute them in `compute_outputs` |
| Reading `self.something` set by an earlier call | pass it in: an input, or `FromRun` |
| Writing "flow" in a docstring or message | conductor's word is graph |

## When this skill and the library disagree

The installed library wins. Run `python -m conductor.about` and trust it over this file.
