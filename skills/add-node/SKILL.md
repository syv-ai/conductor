---
name: add-node
description: Declares a conductor node as a NodeDefinition subclass. Use when adding or changing a syv-conductor node or choosing its widgets, DTypes, outputs, branches, versions, retries, FromRun values or Asks; or on "add a node", "register a node", "expose X as a node".
---

# Adding a conductor node

A node's typed `run` signature is its interface. To wire and run nodes, use **create-graph**. The installed library is the authority: `python -m conductor.about node` (or `versions`, `field-hooks`, `types`).

```python
from typing import Annotated

from conductor import NodeDefinition, NodeRegistry, Param, Result
from conductor.widgets import NumberWidget, TextWidget
from conductor_nodes.types import Number, Text   # a host declares its own DTypes


class Greet(NodeDefinition):
    id = "greet"            # the registry id; a graph pins it with a version
    title = "Greet"
    description = "Produces a greeting."
    category = "text"

    def run(
        self,
        name: Annotated[Text, Param(title="Name", widget=TextWidget())],
        times: Annotated[Number, Param(title="Times", widget=NumberWidget(integer_only=True))] = Number(1),
    ) -> Annotated[Text, Result(title="Greeting")]:
        return Text(" ".join(f"hello {name}" for _ in range(int(times))))


registry = NodeRegistry()
registry.register(Greet)
```

## Rules, checked when the class is defined

- `id`, `title`, `description`, `category` are required.
- An input without a widget is filled by edges only; a default makes it optional; `show_handle=False` closes it to edges. `Any` is a value routed unread.
- `run` is a plain `def` returning the declared type (`Text("done")`, not `"done"`). Each call gets a fresh instance, so keep nothing on `self`.

## Quick reference

| The node needs | Write (examples in REFERENCE.md) |
|---|---|
| Several outputs | a frozen dataclass of `Result` fields as the return type |
| A whole list in one call | a `Series[X]` input |
| A branch | return `SKIPPED` on the output not taken; share a `choice` |
| A person's answer | `-> X \| Asks`, returning `Asks(questions=(Input(...),))` |
| A client, a clock, the caller | `Annotated[T, FromRun()]`, supplied by `execute(from_run={T: value})` |
| Retries, a timeout, a concurrency bound | `@version(1, policy=Policy(retries=3, timeout=10, concurrency=4))` |
| A second version | `@version(n)` on every version, `run` included, and `@upgrade(n - 1, n)` |
| Fields that depend on configuration | override `compute_inputs` / `compute_outputs` |
| Every connected name as an input | `def run(self, **inputs: Single)` |

One `NodeRegistry` per host; two classes under one id raise. `conductor_nodes.registry()` holds the standard nodes.

## Common mistakes

| Mistake | Fix |
|---|---|
| `@version(1)` on `run_v1` beside a plain `def run` | mark `run` too: `@version(2)` |
| A retry loop inside `run` | `Policy(retries=...)`; raise `ExternalFailure` or set `retry_on` |
| "flow" in a docstring or message | conductor's word is graph |
