---
name: add-node
description: Declares a conductor node as a function under @node. Use when adding or changing a syv-conductor node or choosing its widgets, types, outputs, branches, versions, retries, FromRun values or questions to a person; or on "add a node", "register a node", "expose X as a node".
---

# Adding a conductor node

A node is a function; its type hints are its interface, so `def greet(name: str) -> str` under `@node("greet")` is already a node. `Annotated` adds what a person sees in an editor. To wire and run nodes, use **create-graph**. The installed library is the authority: `python -m conductor.about node` (or `versions`, `field-hooks`, `types`).

```python
from typing import Annotated

from conductor import NodeRegistry, Param
from conductor.widgets import NumberWidget, TextWidget

registry = NodeRegistry(category="text")   # its nodes' category, unless one sets its own


@registry.node("greet")                 # the id; a graph pins it with a version
def greet(
    name: Annotated[str, Param(title="Name", widget=TextWidget())],
    times: Annotated[int, Param(title="Times", widget=NumberWidget(integer_only=True))] = 1,
) -> str:
    """Produces a greeting."""
    return " ".join(f"hello {name}" for _ in range(times))


assert greet("Ida") == "hello Ida"    # a node is called like its function
assert registry["greet"] is greet
```

## Rules, checked by `@node`

- Only the id is required. The title defaults to the function's name, the description to its docstring.
- An input without a widget gets no control in an editor; an edge fills it, or a value set in code. A default makes it optional; `connectable=False` closes it to edges, and a type no edge can carry (`list[X]`) closes it by itself. `Any` is a value routed unread.
- The function is a plain `def` returning a value of its declared type.

## Quick reference

| The node needs | Write (examples in REFERENCE.md) |
|---|---|
| Several outputs | a frozen dataclass of `Result` fields as the return type |
| A whole list in one call | a `Series[X]` input |
| A whole table in or out | `Table[Contact]` (a `TypedDict` names the columns) or bare `Table`; it is a `polars.DataFrame` |
| A branch | return `SKIPPED` on the output not taken; share an `exclusive_group` |
| A blank (an empty cell) | `email: str \| None`; otherwise the node is not called for that row and its outputs are blank there |
| A person's answer | `-> X \| AskPerson`, returning `AskPerson(defaults=..., widgets=...)`; the declared outputs are the questions |
| A client, a clock, the caller | `FromRun[T]`, supplied by `compiled.run(from_run={T: value})` |
| Retries, a timeout, a concurrency bound | `@node("x", policy=Policy(retries=3, attempt_timeout=10, concurrency=4))` |
| A second version | `replaces=node("x")(run_v1), upgrade=Upgrade(renamed_inputs={...})` on `@node` |
| Fields that depend on configuration | `compute_added_inputs=` / `compute_outputs=`: plain functions whose parameters name the inputs they read; `ArrivingType` gives the type on an edge |
| Every connected name as an input | `**inputs: AsItArrives`, or `**values: float` to run per row |

`@registry.node` takes `@node`'s arguments and adds the node; plain `@node` registers nothing, and `registry.add(node_a, node_b)` adds such nodes later. A module keeps a registry of its own (`NodeRegistry(category=, tags=, extra=)` set its nodes' defaults) and the host joins them with `registry.add_subregistry(module_registry)`. Two definitions under one id raise; `replace` and `remove` change one, and `registry.filter(predicate)` is a narrower registry. `extra=` is your own JSON about a node, which conductor keeps and never reads.

## Common mistakes

| Mistake | Fix |
|---|---|
| A retry loop inside the node | `Policy(retries=...)`; raise `ExternalFailure` or set `retry_on` |
| "flow" in a docstring or message | conductor's word is graph |
| Dropping `__row` from a table that gets merged with its source | keep the column; otherwise the node fails with `row_ids_lost` |
