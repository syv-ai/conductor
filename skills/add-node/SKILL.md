---
name: add-node
description: Use when adding a new conductor node — any Python function being decorated with @registry.node, any new class extending BaseNode, or when the user asks how to expose a function to a flow. Covers widget selection, multi-output, validation, retries, and versioning. Triggers on phrases like "add a node", "register a node", "new flow node", "expose X as a node".
---

# Adding a conductor node

Use this skill when the user wants to create a new node in a project that depends on the [`conductor`](https://pypi.org/project/conductor/) library.

## First — pull the authoritative library reference

Conductor ships its own reference text inside the wheel. **Run these before writing code** so your advice matches the installed version, not stale training data:

```bash
python -m conductor.about sections          # list section slugs
python -m conductor.about registration      # decorator + metadata
python -m conductor.about widgets           # widget catalog + type→widget defaults
python -m conductor.about data-flow         # edges, shared refs, static data, defaults
python -m conductor.about retry             # node-level and global retries
python -m conductor.about errors            # error hierarchy + what to raise
```

Use `python -m conductor.about` (no args) for the full text when section names are uncertain. For class-based nodes: `python -m conductor.about class-based-nodes`.

Programmatic equivalent: `from conductor.about import get_content, list_sections, get_section`.

## Core pattern

A node is a plain Python function decorated with `@registry.node(...)`. Type hints with `Annotated[T, Widget(...)]` drive validation **and** frontend rendering — they are the single source of truth.

```python
from typing import Annotated
from conductor import NodeRegistry
from conductor.widgets import Text, Number, Output

registry = NodeRegistry()  # usually a module-level singleton in the host project

@registry.node(
    "greet",
    version=1,
    name="Greet",
    description="Produces a greeting.",
)
def greet(
    name: Annotated[str, Text(label="Name")],
    times: Annotated[int, Number(label="Times", integer_only=True)] = 1,
) -> Annotated[str, Output(label="Greeting")]:
    return "hello " + (name + " ") * times
```

Rules:

- `base_id` must be unique per version. Versioned as `greet@1`, `greet@2`.
- First positional arg is the first input; order matches UI order.
- Default values become the default on the frontend widget.
- Return annotation describes the output. Use a tuple return for multi-output (see below).

## Picking a widget

Every `Annotated[T, Widget]` is explicit. When a parameter has no widget, conductor infers a default from the Python type:

| Type | Default widget |
|---|---|
| `str` | `Text` |
| `int` | `Number(integer_only=True)` |
| `float` | `Number` |
| `bool` | `Checkbox` |
| `Date` | `DatePicker` |
| `list[T]` | `List(item_widget=default(T))` |
| `dict` | `SchemaBuilder` |
| `Base64Str`, `NamedFile`, `MultiNamedFile` | `FileUpload` |

Prefer an explicit `Annotated` when you want a `label`, `description`, `choices`, or other UI config. Leave it off for quick-and-dirty internal nodes.

Full widget catalog: `python -m conductor.about widgets` or read `docs/widgets.md` in the conductor repo.

## Multi-output

Return a tuple **and** annotate each output:

```python
from conductor.widgets import Output

@registry.node("split", version=1, name="Split", description="Splits a string.")
def split(
    s: Annotated[str, Text(label="Input")],
    sep: Annotated[str, Text(label="Separator")] = ",",
) -> tuple[
    Annotated[str, Output(label="Head")],
    Annotated[str, Output(label="Tail")],
]:
    head, _, tail = s.partition(sep)
    return head, tail
```

Tuple returns are normalized to `{"output_1": head, "output_2": tail}`. Single returns are `{"result": value}`.

Returning a pydantic `BaseModel` or a `dict` spreads the fields as individual outputs (`DICT_SPREAD`).

## Retries

Add `max_retries` / `retry_delay` on the decorator for transient failures. Raise `NodeConnectionError` (from `conductor.errors`) to signal a retryable failure from node code:

```python
from conductor.errors import NodeConnectionError

@registry.node(
    "fetch",
    version=1,
    name="Fetch",
    description="Fetch a URL.",
    max_retries=3,
    retry_delay=0.5,  # delay doubles each attempt
)
def fetch(url: Annotated[str, Text(label="URL")]) -> Annotated[str, Output(label="Body")]:
    try:
        return _http_get(url)
    except TimeoutError as e:
        raise NodeConnectionError(str(e)) from e
```

`NodeValidationError` is never retried. `HumanInputRequired` short-circuits retries to pause the flow.

## Human-in-the-loop

Pause a flow to collect input from a user:

```python
from conductor.errors import HumanInputRequired

@registry.node("approve", version=1, name="Approve", description="Ask for approval.")
def approve(text: Annotated[str, Text(label="Text")]) -> Annotated[bool, Output(label="Approved")]:
    raise HumanInputRequired(
        prompt=f"Approve this text? {text}",
        schema={"type": "object", "properties": {"approved": {"type": "boolean"}}},
    )
    # On resume, the node is re-invoked with the response available via
    # FlowStore or by reading conductor.about about section "hitl".
```

Full HITL contract: `python -m conductor.about hitl`.

## Class-based nodes

When a node needs persistent state, custom dispatch, or dynamic I/O, extend `BaseNode` and register via `registry.register_class(MyNode)`. This is a minority case — check `python -m conductor.about class-based-nodes` before reaching for it.

## Shared references

Declared at graph-build time on the `GraphNode`, not on the decorated function. See the create-flow skill or `python -m conductor.about shared-references`.

## Where to register

Most projects keep one module-level `registry = NodeRegistry()` and import it from each file that defines nodes. Compose with pre-built libraries via `registry.merge(other_registry)`:

```python
import conductor_nodes
registry.merge(conductor_nodes.get_default_registry())
```

## Checklist before shipping a node

- [ ] Unique `base_id@version` within the target registry.
- [ ] Every parameter and every return has a type annotation.
- [ ] Explicit widgets where UI needs a label/choices/constraint; implicit where a default suffices.
- [ ] External calls wrapped to raise `NodeConnectionError` and have `max_retries` set.
- [ ] Invalid input raises `NodeValidationError` (or let pydantic do it automatically).
- [ ] Tests cover: happy path, invalid input, retryable failure, multi-output ordering.

## When your advice diverges from the installed version

The library is the source of truth. If a user reports behavior that contradicts this skill, run `python -m conductor.about` first and trust the output. Skill docs are a shortcut; the packaged reference is authoritative.
