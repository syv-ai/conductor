"""Demo nodes for the flow engine playground.

The set is deliberately small — enough to exercise every UI feature
(static data, edges, shared references, for-each compound regions) without
drowning in categories.
"""

from __future__ import annotations

import functools
import operator
import re
from typing import Annotated

from conductor.registry import NodeRegistry
from conductor.types import NodeCategory
from conductor.widgets import (
    Checkbox,
    ConnectionList,
    Dropdown,
    Output,
    Range,
    Text,
    Textarea,
)

registry = NodeRegistry()


# ---------------------------------------------------------------------------
# Text primitives
# ---------------------------------------------------------------------------


@registry.node("text", version=1, name="Text", description="Static text value")
def text_node(
    value: Annotated[str, Textarea(label="Text", description="Any string", rows=3)] = "",
) -> Annotated[str, Output(label="Text")]:
    return value


@registry.node("uppercase", version=1, name="Uppercase", description="Convert text to uppercase")
def uppercase_node(
    text: Annotated[str, Text(label="Text")],
) -> Annotated[str, Output(label="Result")]:
    return text.upper()


@registry.node(
    "template",
    version=1,
    name="Template",
    description="Substitute {input} in a template string",
)
def template_node(
    template: Annotated[str, Textarea(label="Template", rows=3)] = "Result: {input}",
    input: Annotated[str, Text(label="Input")] = "",
) -> Annotated[str, Output(label="Result")]:
    return template.replace("{input}", str(input))


@registry.node("combine", version=1, name="Combine", description="Join two strings")
def combine_node(
    a: Annotated[str, Text(label="A")],
    b: Annotated[str, Text(label="B")],
    separator: Annotated[str, Text(label="Separator")] = " ",
) -> Annotated[str, Output(label="Result")]:
    return f"{a}{separator}{b}"


# ---------------------------------------------------------------------------
# Regex (new)
# ---------------------------------------------------------------------------


@registry.node(
    "regex",
    version=1,
    name="Regex",
    description="Match a regex against text; returns the first match, its groups, and a boolean",
)
def regex_node(
    text: Annotated[str, Textarea(label="Text", rows=3)] = "",
    pattern: Annotated[str, Text(label="Pattern", description="Python regex")] = r"\w+",
    ignore_case: Annotated[bool, Checkbox(label="Ignore case")] = False,
) -> tuple[
    Annotated[str, Output(label="Match")],
    Annotated[list[str], Output(label="Groups")],
    Annotated[bool, Output(label="Matched?")],
]:
    flags = re.IGNORECASE if ignore_case else 0
    m = re.search(pattern, text, flags=flags)
    if not m:
        return ("", [], False)
    return (m.group(0), list(m.groups()), True)


# ---------------------------------------------------------------------------
# List creation (for for-each)
# ---------------------------------------------------------------------------


@registry.node(
    "make-list",
    version=1,
    name="Make List",
    description="Split text into a list — feed into For Each (Start)",
)
def make_list_node(
    text: Annotated[
        str,
        Textarea(
            label="Lines",
            description="One item per line, or custom separator below",
            rows=4,
        ),
    ] = "",
    separator: Annotated[
        str,
        Text(label="Separator", description="Leave as '\\n' for one-per-line"),
    ] = r"\n",
    trim: Annotated[bool, Checkbox(label="Trim whitespace")] = True,
) -> Annotated[list[str], Output(label="Items")]:
    sep = separator.encode().decode("unicode_escape") if separator else "\n"
    items = text.split(sep)
    if trim:
        items = [s.strip() for s in items]
    return [s for s in items if s]


# ---------------------------------------------------------------------------
# Numbers + math
# ---------------------------------------------------------------------------


@registry.node("number", version=1, name="Number", description="Static numeric value")
def number_node(
    value: Annotated[
        float,
        Range(label="Value", min_val=-1000, max_val=1000, step=0.1),
    ] = 0,
) -> Annotated[float, Output(label="Number")]:
    return value


@registry.node(
    "math",
    version=1,
    name="Math",
    description="Arithmetic on N numbers (connect a ConnectionList)",
)
def math_node(
    numbers: Annotated[
        dict[str, float],
        ConnectionList(label="Numbers", description="Connect number inputs"),
    ],
    operation: Annotated[
        str,
        Dropdown(
            label="Operation",
            choices=["Add", "Subtract", "Multiply", "Divide", "Min", "Max", "Average"],
        ),
    ] = "Add",
) -> Annotated[float, Output(label="Result")]:
    vals = [float(v) for v in (numbers.values() if isinstance(numbers, dict) else numbers)]
    if not vals:
        return 0.0
    match operation:
        case "Add":
            return functools.reduce(operator.add, vals)
        case "Subtract":
            return functools.reduce(operator.sub, vals)
        case "Multiply":
            return functools.reduce(operator.mul, vals)
        case "Divide":
            return functools.reduce(lambda a, b: a / b if b else float("inf"), vals)
        case "Min":
            return min(vals)
        case "Max":
            return max(vals)
        case "Average":
            return sum(vals) / len(vals)
        case _:
            raise ValueError(f"Unknown operation: {operation}")


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------


@registry.node(
    "if-else",
    version=1,
    name="If / Else",
    description="Route a value to True/False based on a condition",
    category=NodeCategory.CONTROL,
)
def if_else_node(
    value: Annotated[str, Text(label="Value")],
    condition: Annotated[
        str,
        Dropdown(label="Condition", choices=["Is not empty", "Is empty", "Equals", "Contains"]),
    ] = "Is not empty",
    compare_to: Annotated[str, Text(label="Compare to")] = "",
) -> tuple[
    Annotated[str, Output(label="True")],
    Annotated[str, Output(label="False")],
]:
    from conductor._sentinel import SKIPPED

    match condition:
        case "Is not empty":
            truthy = bool(value and value.strip())
        case "Is empty":
            truthy = not (value and value.strip())
        case "Equals":
            truthy = value == compare_to
        case "Contains":
            truthy = compare_to in value
        case _:
            truthy = False

    return (value, SKIPPED) if truthy else (SKIPPED, value)


@registry.node(
    "for-each-start",
    version=1,
    name="For Each (Start)",
    description="Iterate over a list of items",
    category=NodeCategory.CONTROL,
)
def for_each_start(
    items: Annotated[list[str], ConnectionList(label="Items")],
    execution_mode: Annotated[
        str,
        Dropdown(label="Mode", choices=["Sequential", "Parallel"]),
    ] = "Sequential",
) -> tuple[
    Annotated[str, Output(label="Item")],
    Annotated[int, Output(label="Index")],
]:
    raise NotImplementedError("Handled by compound node")


@registry.node(
    "for-each-end",
    version=1,
    name="For Each (End)",
    description="Collect loop iteration results",
    category=NodeCategory.CONTROL,
)
def for_each_end(
    item: Annotated[str, Text(label="Item")],
) -> Annotated[list[str], Output(label="Collected")]:
    raise NotImplementedError("Handled by compound node")
