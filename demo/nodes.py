"""Demo nodes for the flow engine playground."""

from typing import Annotated

from conductor.registry import NodeRegistry
from conductor.types import NodeCategory
from conductor.widgets import (
    ConnectionList,
    Dropdown,
    Output,
    Range,
    Text,
    Textarea,
)

registry = NodeRegistry()


# ---------------------------------------------------------------------------
# IO Nodes
# ---------------------------------------------------------------------------


@registry.node(
    "text",
    version=1,
    name="Text",
    description="Outputs a static text value",
)
def text_node(
    text: Annotated[str, Textarea(label="Text", description="Enter text", rows=3)] = "",
) -> Annotated[str, Output(label="Text")]:
    return text


@registry.node(
    "number",
    version=1,
    name="Number",
    description="Outputs a numeric value",
)
def number_node(
    value: Annotated[float, Range(label="Value", description="Enter a number", min_val=-1000, max_val=1000, step=0.1)] = 0,
) -> Annotated[float, Output(label="Number")]:
    return value


@registry.node(
    "math",
    version=2,
    name="Math",
    description="Performs arithmetic on N numbers",
)
def math_node(
    numbers: Annotated[dict[str, float], ConnectionList(label="Numbers", description="Connect number inputs")],
    operation: Annotated[
        str,
        Dropdown(
            label="Operation",
            description="Arithmetic operation",
            choices=["Add", "Subtract", "Multiply", "Divide", "Min", "Max", "Average"],
        ),
    ] = "Add",
) -> Annotated[float, Output(label="Result")]:
    if isinstance(numbers, dict):
        vals = [float(v) for v in numbers.values()]
    elif isinstance(numbers, list):
        vals = [float(v) for v in numbers]
    else:
        vals = [float(numbers)]

    if not vals:
        return 0.0

    import functools
    import operator

    match operation:
        case "Add":
            return functools.reduce(operator.add, vals)
        case "Subtract":
            return functools.reduce(operator.sub, vals)
        case "Multiply":
            return functools.reduce(operator.mul, vals)
        case "Divide":
            return functools.reduce(lambda a, b: a / b if b != 0 else float("inf"), vals)
        case "Min":
            return min(vals)
        case "Max":
            return max(vals)
        case "Average":
            return sum(vals) / len(vals)
        case _:
            raise ValueError(f"Unknown operation: {operation}")


@registry.node(
    "uppercase",
    version=1,
    name="Uppercase",
    description="Converts text to uppercase",
)
def uppercase_node(
    text: Annotated[str, Text(label="Text", description="Text to uppercase")],
) -> Annotated[str, Output(label="Result")]:
    return text.upper()


@registry.node(
    "template",
    version=1,
    name="Template",
    description="Simple string template with {placeholders}",
)
def template_node(
    template: Annotated[str, Textarea(label="Template", description="Use {input} for substitution", rows=3)] = "Result: {input}",
    input: Annotated[str, Text(label="Input", description="Value to insert")] = "",
) -> Annotated[str, Output(label="Result")]:
    return template.replace("{input}", str(input))


@registry.node(
    "combine",
    version=1,
    name="Combine",
    description="Joins two values with a separator",
)
def combine_node(
    a: Annotated[str, Text(label="A", description="First value")],
    b: Annotated[str, Text(label="B", description="Second value")],
    separator: Annotated[str, Text(label="Separator", description="Join separator")] = " ",
) -> Annotated[str, Output(label="Result")]:
    return f"{a}{separator}{b}"


@registry.node(
    "summarizer",
    version=1,
    name="Summarizer (LLM)",
    description="Summarizes text using an LLM (placeholder)",
)
def summarizer_node(
    text: Annotated[str, Textarea(label="Text", description="Text to summarize", rows=4)],
    max_length: Annotated[int, Range(label="Max words", description="Maximum summary length in words", min_val=10, max_val=500, step=10)] = 50,
) -> Annotated[str, Output(label="Summary")]:
    word_count = len(text.split())
    return (
        f"[LLM Summary - Not implemented yet]\n"
        f"Input: {word_count} words\n"
        f"Requested max: {max_length} words\n"
        f"Preview: {text[:100]}{'...' if len(text) > 100 else ''}"
    )


# ---------------------------------------------------------------------------
# Control Nodes
# ---------------------------------------------------------------------------


@registry.node(
    "conditional",
    version=1,
    name="If / Else",
    description="Routes data based on a condition",
    category=NodeCategory.CONTROL,
)
def conditional_node(
    value: Annotated[str, Text(label="Value", description="Value to check")],
    condition: Annotated[
        str,
        Dropdown(
            label="Condition",
            choices=["Is not empty", "Is empty", "Equals", "Contains"],
        ),
    ] = "Is not empty",
    compare_to: Annotated[str, Text(label="Compare to", description="Comparison value")] = "",
) -> tuple[
    Annotated[str, Output(label="True")],
    Annotated[str, Output(label="False")],
]:
    from conductor._sentinel import SKIPPED

    match condition:
        case "Is not empty":
            result = bool(value and value.strip())
        case "Is empty":
            result = not (value and value.strip())
        case "Equals":
            result = value == compare_to
        case "Contains":
            result = compare_to in value
        case _:
            result = False

    if result:
        return (value, SKIPPED)
    else:
        return (SKIPPED, value)


@registry.node(
    "for-each-start",
    version=1,
    name="For Each (Start)",
    description="Iterates over a list of items",
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
    description="Collects loop iteration results",
    category=NodeCategory.CONTROL,
)
def for_each_end(
    item: Annotated[str, Text(label="Item")],
) -> Annotated[list[str], Output(label="Collected")]:
    raise NotImplementedError("Handled by compound node")
