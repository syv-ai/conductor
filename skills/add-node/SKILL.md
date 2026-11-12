---
name: add-node
description: Declares a conductor node as a function under @node. Use when adding or changing a syv-conductor node or choosing its widgets, types, outputs, branches, versions, retries, Context values or questions to a person; or on "add a node", "register a node", "expose X as a node".
---

# Adding a conductor node

A node is a function; its type hints are its interface, so `def greet(name: str) -> str` under `@node("greet", title="Greet", description="Greets a person.")` is already a node. `Annotated` adds what a person sees in an editor. To wire and run nodes, use **create-graph**. The installed library is the authority: `python -m conductor.about node` (or `versions`, `field-hooks`, `types`).

```python
from typing import Annotated

from conductor import NodeRegistry, Param
from conductor.widgets import NumberInput, TextInput

registry = NodeRegistry(category="text")   # its nodes' category, unless one sets its own


@registry.node("greet", title="Greet", description="Greets a person by name.")               # the id; a graph pins it with a version
def greet(
    name: Annotated[str, Param(title="Name", widget=TextInput())],
    times: Annotated[int, Param(title="Times", widget=NumberInput())] = 1,
) -> str:
    """Produces a greeting."""
    return " ".join(f"hello {name}" for _ in range(times))


assert greet("Ida") == "hello Ida"    # a node is called like its function
assert registry["greet"] is greet
```

## Rules, checked by `@node`

- The id, `title=` and `description=` are required: they are what a person reads in the palette, written for someone who is not a developer. The docstring is for developers and never shown. A field's label is its name made readable unless `Param(title=)` says otherwise.
- An input without a widget gets no control in an editor; an edge fills it, or a value set in code. A default makes it optional. `Param(ge=0, max_length=31, ...)` holds a value to pydantic's limits, and a limit that can't apply to the type is refused at `@node`. `Any` is a value routed unread.
- The function is a plain `def` returning a value of its declared type. A mistake raises `NodeDeclarationError` with a `code`.

## Quick reference

| The node needs | Write (examples in REFERENCE.md) |
|---|---|
| Several outputs | a frozen dataclass as the return type, `Annotated[str, Param(title=...)]` on a field to say more |
| To run once per item | return `Series[X]`; what takes one `X` from it runs once per item, and so does everything after |
| Every item in one call | a `Series[X]` input: it gathers them, in order |
| A whole table in or out | `Table[Contact]` (a `TypedDict` names the columns, and the table holds exactly those) or bare `Table`; it is a `polars.DataFrame`, one value, and may be returned as a list of records. A `Table` input fed records gathers them |
| A branch | return `SKIPPED` on the output not taken; share an `exclusive_group` |
| A blank (an empty cell) | `email: str \| None`; otherwise the node is not called for that item and its outputs are blank there |
| A person's answer | `raise Ask(prompt, choices=..., default=...)`; the return type says what the answer is and how to ask, its `Param` how it looks (`-> str` free text, `bool` yes or no, a frozen dataclass a form; every type in `python -m conductor.about legs-and-asking`). Each has the answer's shape: on several outputs it goes by output name; `Choice(id, title)` titles a choice |
| A client, a clock, the caller | `Context[T]`, supplied by `run.resume(context={T: value})` |
| Retries, a timeout, a concurrency bound | `@node("x", title="X", description="...", policy=Policy(retries=3, attempt_timeout=10, concurrency=4))` |
| A second version | `replaces=node("x", title="X", description="...")(run_v1), upgrade=Upgrade(renamed_inputs={...})` on `@node` |
| A type that follows what arrives | a type variable: `def keep[T](value: T) -> T`, `def split[T](value: Table[T] \| list[T]) -> Series[T]`, a generic frozen dataclass of outputs. `T` is the type on the edge, bound when the graph loads |
| Fields named by a setting | `compute_added_inputs=` / `compute_outputs=`: plain functions whose parameters name the inputs they read and that return `{name: type hint}`. Added inputs arrive on `**values` |
| A text or number class of your own | `@host_type("email", title="Email")` above `class Email(str)` |

`@registry.node` takes `@node`'s arguments and adds the node; plain `@node` registers nothing, and `registry.add(node_a, node_b)` adds such nodes later. A module keeps a registry of its own (`NodeRegistry(category=, tags=, extra=)` set its nodes' defaults) and the host joins them with `registry.add_subregistry(module_registry)`. Two definitions under one id raise; `replace` and `remove` change one, and `registry.filter(predicate)` is a narrower registry. `extra=` is your own JSON about a node, which conductor keeps and never reads.

## Testing a registry

`upper("a")` tests one node. `check_registry` tests what a declaration can't show, for a whole registry; `python -m conductor.about testing` lists its checks.

```python
from conductor.testing import check_registry

assert check_registry(registry, examples={"greet": [{"name": "Ida"}]}) == []
```

A body is called only for a node given an example (a node may send mail), once per example, through the engine. `("greet", 1)` keys an older version's examples, and `context={T: value}` gives the examples' `Context` values, as `graph.run` takes them. A known failure goes in a test of its own marked `pytest.mark.xfail(strict=True)`, over `registry.filter(...)` (`docs/problems.md`, under `run_failed`, shows one).

## Common mistakes

| Mistake | Fix |
|---|---|
| A retry loop inside the node | `Policy(retries=...)`; raise `ExternalError` or set `retry_on` |
| "flow" in a docstring or message | conductor's word is graph |
| Catching `Exception` around an `Ask` | `Ask` is a `BaseException`, so `except Exception` lets it through; don't catch `BaseException` |
| `Ask(default=value)` on a node of several outputs | `default={"field": value}`, by output name as the answer is; an `Ask` that doesn't fit the outputs fails the node as `invalid_ask` |
