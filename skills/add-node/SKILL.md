---
name: add-node
description: Writes a conductor node, a Python function under @node. Use when adding or changing a syv-conductor node, or choosing its widgets, types, outputs, branches, versions, retries, Context values or questions to a person; or on "add a node", "register a node", "expose X as a node".
---

# Adding a conductor node

A node is a plain Python function. Its type hints are its interface: each parameter is an input, and the return type is its output. So `def greet(name: str) -> str` under `@node("greet", title="Greet", description="Greets a person.")` is already a node. `Annotated[..., Param(...)]` adds what a person sees in an editor.

To wire nodes into a graph and run it, use the **create-graph** skill. The installed library is the authority on every rule: `python -m conductor.about node` (or `versions`, `field-hooks`, `types`).

```python
from typing import Annotated

from conductor import NodeRegistry, Param

registry = NodeRegistry(category="text")   # its nodes' category, unless one sets its own


@registry.node("greet", title="Greet", description="Greets a person by name.")               # the id; a graph pins it with a version
def greet(
    name: Annotated[str, Param(title="Name")],
    times: Annotated[int, Param(title="Times")] = 1,
) -> str:
    """Produces a greeting."""
    return " ".join(f"hello {name}" for _ in range(times))


assert greet("Ida") == "hello Ida"    # a node is called like its function
assert registry["greet"] is greet
```

## What `@node` checks

`@node` checks the declaration when the module is imported. A mistake raises `NodeDeclarationError`. Its `problems` holds one problem, whose `type` is the code.

**Id, title and description.**
- The id, `title=` and `description=` are all required.
- The title and description are what a person reads in the palette. Write them for someone who isn't a developer.
- The docstring is for developers. It's never shown.
- A field's title is its `Param(title=)`. Without one it's `None`: nothing is guessed from the parameter's name.

**Inputs.**
- Give an input a widget only where it says more than its type, such as `Param(widget=Textarea())` for a long text. A host's editor picks the control for a `str`, a number, a `bool` or a list from the type, so `TextInput()`, `NumberInput()`, `Switch()` and `ListInput()` there add nothing.
- A default makes an input optional, and the palette shows it as `default` (`null` means the value `None`). An input without a default is `required` and has no `default` key.
- `Param(constant_only=True)` makes an input a setting that's fixed when the graph is drawn. An edge never wires it, and a caller never gives it.
- `Param(ge=0, max_length=31, ...)` sets pydantic's limits on a value. `@node` refuses a limit that can't apply to the type.
- Never write pydantic's `Field(...)` in a node's hint. It's refused (`pydantic_field_in_hint`). Use `Param` instead.
- `Any` means a value passed on without being read.

**Outputs.**
- Only an output can default to `SKIPPED` or have an `exclusive_group`. That holds in a hook too.

**The function.**
- It's a plain `def` (not `async def`) that returns a value of its declared type.

## Quick reference

| The node needs | Write (examples in REFERENCE.md) |
|---|---|
| Several outputs | a frozen dataclass as the return type. Put `Annotated[str, Param(title=...)]` on a field to say more |
| To run once per item | return `Series[X]`. A node that takes one `X` from it runs once per item, and so does everything after it |
| Every item in one call | a `Series[X]` input. It collects the items, in order |
| A whole table in or out | `Table[Contact]` (a `TypedDict` names the columns, and the table has exactly those) or a bare `Table`. It's a `polars.DataFrame` and counts as one value. The node may return it as a list of records. A `Table` input wired to records collects them |
| A branch | return `SKIPPED` on the output not taken, and put the outputs in one `exclusive_group` |
| To take `None` | `email: str \| None`. Otherwise the node isn't called for that item, and its outputs are `None` there |
| A person's answer | `raise Ask(prompt, choices=..., default=...)`, and `asks=True` on the `@node`. The answer becomes the node's outputs. See "Asking a person" below |
| A client, a clock, the caller | the run's context: `def sign(ctx: Context[AppContext], text: str)`. Name the class once with `NodeRegistry(context_type=AppContext)`, and pass the object as `run.resume(context=AppContext(...))` |
| Retries, a timeout, a concurrency limit | `@node("x", title="X", description="...", policy=Policy(retries=3, attempt_timeout=10, concurrency=4))` |
| A second version | `replaces=node("x", title="X", description="...")(run_v1), upgrade=Upgrade(renamed_inputs={...})` on `@node` |
| A type that follows what arrives | a type variable: `def keep[T](value: T) -> T`, `def split[T](value: Table[T] \| list[T]) -> Series[T]`, or a generic frozen dataclass of outputs. `T` becomes the type on the edge when the graph is compiled |
| Inputs or outputs named by a constant | `compute_added_inputs=` / `compute_outputs=`: plain functions whose parameters name the inputs they read, and that return `{name: type hint}`. Added inputs arrive in `**values` |
| A text or number class of your own | `@host_type("email", title="Email")` above `class Email(str)` |

### Asking a person

- The return type of the function that raises `Ask` says what the answer is and how to ask: `-> str` is free text, `-> bool` a yes or no, and a frozen dataclass a form. Every type is in `python -m conductor.about asking-a-person`.
- The output's `Param` says how the question looks.
- `choices=` and `default=` have the answer's shape. On a node with several outputs, they go by output name. `Choice(id, title)` gives a choice a title.
- A node that needs the answer reads it through an edge. For example, `approval(invoice) -> bool` asks, and is wired into `pay(invoice, ok: bool)`. To offer the two as one palette entry, put them in a `subgraph`.

## Registries

- `@registry.node` takes the same arguments as `@node`, and adds the node to the registry.
- Plain `@node` registers nothing. Add such nodes later with `registry.add(node_a, node_b)`.
- A module keeps a registry of its own, and the host joins them with `registry.add_subregistry(module_registry)`. `NodeRegistry(category=, tags=, extra=)` sets defaults for its nodes.
- Two definitions under one id raise. `replace` and `remove` change one. `registry.filter(predicate)` gives a narrower registry.
- `extra=` is your own JSON about a node. Conductor keeps it and never reads it.

## Testing a registry

Calling `greet("Ida")` tests one node. `check_registry` tests what a declaration alone can't show, for a whole registry. `python -m conductor.about testing` lists its checks.

```python
from conductor.testing import check_registry

assert check_registry(registry, examples={"greet": [{"name": "Ida"}]}) == []
```

- A node's body is only called for a node you give an example (a node may send mail). It's called once per example, through the engine.
- `("greet", 1)` as a key gives examples for an older version.
- `context=AppContext(...)` gives the examples the run's context, the same way `graph.run` takes it.
- A known failure goes in a test of its own, over `registry.filter(...)`, marked `pytest.mark.xfail(strict=True)`. `python -m conductor.about run-failed` shows one.

## Common mistakes

| Mistake | Fix |
|---|---|
| A retry loop inside the node | use `Policy(retries=...)`, and raise `ExternalError` or set `retry_on` |
| "flow" in a docstring or message | conductor's word is graph |
| Catching `Exception` around an `Ask` | `Ask` is a `BaseException`, so `except Exception` lets it through. Don't catch `BaseException` |
| `Ask(default=value)` on a node with several outputs | write `default={"field": value}`, by output name, as the answer is. An `Ask` that doesn't fit the outputs fails the node as `invalid_ask` |
