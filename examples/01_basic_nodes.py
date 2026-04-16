"""Example 1: Creating and registering basic nodes.

Shows how to create function-based nodes with different widget types,
single and multi-output nodes, and optional parameters.
"""

from typing import Annotated

from flowengine import NodeRegistry
from flowengine.widgets import (
    Checkbox,
    Dropdown,
    Output,
    Range,
    Text,
    Textarea,
)

# Create a fresh registry (each project gets its own)
registry = NodeRegistry()


# ---------------------------------------------------------------------------
# Simple single-input, single-output node
# ---------------------------------------------------------------------------


@registry.node("greet", version=1, name="Greeting", description="Generates a greeting")
def greet(
    name: Annotated[str, Text(label="Name", description="Who to greet")],
) -> Annotated[str, Output(label="Greeting")]:
    return f"Hello, {name}!"


# ---------------------------------------------------------------------------
# Node with multiple inputs and a dropdown
# ---------------------------------------------------------------------------


@registry.node(
    "format-name",
    version=1,
    name="Format Name",
    description="Formats a full name",
)
def format_name(
    first: Annotated[str, Text(label="First name")],
    last: Annotated[str, Text(label="Last name")],
    style: Annotated[
        str,
        Dropdown(label="Style", choices=["First Last", "Last, First", "LAST First"]),
    ] = "First Last",
) -> Annotated[str, Output(label="Full name")]:
    match style:
        case "First Last":
            return f"{first} {last}"
        case "Last, First":
            return f"{last}, {first}"
        case "LAST First":
            return f"{last.upper()} {first}"
        case _:
            return f"{first} {last}"


# ---------------------------------------------------------------------------
# Node with a range slider and checkbox
# ---------------------------------------------------------------------------


@registry.node(
    "truncate",
    version=1,
    name="Truncate",
    description="Truncates text to a max length",
)
def truncate(
    text: Annotated[str, Textarea(label="Text", rows=3)],
    max_length: Annotated[
        int,
        Range(label="Max length", min_val=1, max_val=500, step=1),
    ] = 100,
    add_ellipsis: Annotated[bool, Checkbox(label="Add ...")] = True,
) -> Annotated[str, Output(label="Truncated")]:
    if len(text) <= max_length:
        return text
    truncated = text[:max_length]
    return f"{truncated}..." if add_ellipsis else truncated


# ---------------------------------------------------------------------------
# Multi-output node (returns a tuple)
# ---------------------------------------------------------------------------


@registry.node(
    "split-text",
    version=1,
    name="Split Text",
    description="Splits text into first line and rest",
)
def split_text(
    text: Annotated[str, Textarea(label="Text", rows=4)],
) -> tuple[
    Annotated[str, Output(label="First line")],
    Annotated[str, Output(label="Remaining")],
]:
    lines = text.split("\n", 1)
    first = lines[0]
    rest = lines[1] if len(lines) > 1 else ""
    return first, rest


# ---------------------------------------------------------------------------
# Node with optional input (has default value)
# ---------------------------------------------------------------------------


@registry.node(
    "prefix",
    version=1,
    name="Add Prefix",
    description="Adds a prefix to text",
)
def prefix(
    text: Annotated[str, Text(label="Text")],
    prefix: Annotated[str, Text(label="Prefix")] = "> ",
) -> Annotated[str, Output(label="Result")]:
    return f"{prefix}{text}"


# ---------------------------------------------------------------------------
# Verify registration
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Registered {len(registry.all())} nodes:\n")
    for node in registry.all():
        inputs = ", ".join(f"{i.name}: {i.type_str}" for i in node.inputs)
        outputs = ", ".join(f"{o.name}: {o.type_str}" for o in node.outputs)
        print(f"  {node.id}: {node.name}")
        print(f"    inputs:  ({inputs})")
        print(f"    outputs: ({outputs})")
        print()

    # Test one node directly (they're plain functions!)
    result = greet("World")
    print(f'greet("World") = "{result}"')
