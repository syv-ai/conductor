---
name: add-node
description: Declares a conductor node as a function under @node. Use when adding or changing a syv-conductor node or choosing its widgets, DTypes, outputs, branches, versions, retries, FromRun values or Asks; or on "add a node", "register a node", "expose X as a node".
---

# Adding a conductor node

A node is a function; its typed signature is its interface. To wire and run nodes, use **create-graph**. The installed library is the authority: `python -m conductor.about node` (or `versions`, `field-hooks`, `types`).

```python
from typing import Annotated

from conductor import NodeRegistry, Param, node
from conductor.widgets import NumberWidget, TextWidget



@node("greet", category="text")         # the registry id; a graph pins it with a version
def greet(
    name: Annotated[str, Param(title="Name", widget=TextWidget())],
    times: Annotated[float, Param(title="Times", widget=NumberWidget(integer_only=True))] = float(1),
) -> str:
    """Produces a greeting."""
    return " ".join(f"hello {name}" for _ in range(int(times)))


assert greet("Ida") == "hello Ida"    # a node is called like its function
registry = NodeRegistry([greet])
```

## Rules, checked by `@node`

- Only the id is required. The title defaults to the function's name, the description to its docstring.
- An input without a widget is filled by edges only; a default makes it optional; `connectable=False` closes it to edges, and a type no edge can carry (`list[X]`) closes it by itself. `Any` is a value routed unread.
- The function is a plain `def` returning a value of its declared type.

## Quick reference

| The node needs | Write (examples in REFERENCE.md) |
|---|---|
| Several outputs | a frozen dataclass of `Result` fields as the return type |
| A whole list in one call | a `Series[X]` input |
| A branch | return `SKIPPED` on the output not taken; share a `choice` |
| A blank (an empty cell) | `email: str \| None`; otherwise the node is not called for that row and its outputs are blank there |
| A person's answer | `-> X \| AskPerson`, returning `AskPerson(defaults=..., widgets=...)`; the declared outputs are the questions |
| A client, a clock, the caller | `FromRun[T]`, supplied by `compiled.run(from_run={T: value})` |
| Retries, a timeout, a concurrency bound | `@node("x", policy=Policy(retries=3, timeout=10, concurrency=4))` |
| A second version | `older=[OlderVersion(run_v1, upgrade=Upgrade(renamed_inputs={...}))]` on `@node`, oldest first |
| Fields that depend on configuration | `compute_inputs=` / `compute_outputs=`: plain functions whose parameters name the inputs they read; `Arriving` gives the type on an edge |
| Every connected name as an input | `**inputs: AsItArrives`, or `**values: Number` to run per row |

One `NodeRegistry` per host, built whole; two definitions under one id raise. `conductor_nodes.registry()` holds the standard nodes.

## Common mistakes

| Mistake | Fix |
|---|---|
| A retry loop inside the node | `Policy(retries=...)`; raise `ExternalFailure` or set `retry_on` |
| "flow" in a docstring or message | conductor's word is graph |
