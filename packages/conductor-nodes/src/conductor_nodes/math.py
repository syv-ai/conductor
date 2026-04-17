"""Arithmetic nodes (``math-add``, ``math-subtract``, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from conductor.widgets import Output, Range, Text

if TYPE_CHECKING:
    from conductor import NodeRegistry


def register(registry: "NodeRegistry") -> None:
    """Register every math node on the supplied registry."""

    @registry.node("math-add", version=1, name="Add", description="a + b")
    def add(
        a: Annotated[float, Text(label="A")],
        b: Annotated[float, Text(label="B")],
    ) -> Annotated[float, Output(label="Sum")]:
        return a + b

    @registry.node("math-subtract", version=1, name="Subtract", description="a - b")
    def subtract(
        a: Annotated[float, Text(label="A")],
        b: Annotated[float, Text(label="B")],
    ) -> Annotated[float, Output(label="Difference")]:
        return a - b

    @registry.node("math-multiply", version=1, name="Multiply", description="a * b")
    def multiply(
        a: Annotated[float, Text(label="A")],
        b: Annotated[float, Text(label="B")],
    ) -> Annotated[float, Output(label="Product")]:
        return a * b

    @registry.node(
        "math-divide", version=1, name="Divide",
        description="a / b — raises on b == 0",
    )
    def divide(
        a: Annotated[float, Text(label="A")],
        b: Annotated[float, Text(label="B")],
    ) -> Annotated[float, Output(label="Quotient")]:
        if b == 0:
            raise ValueError("Division by zero")
        return a / b

    @registry.node("math-modulo", version=1, name="Modulo", description="a % b")
    def modulo(
        a: Annotated[float, Text(label="A")],
        b: Annotated[float, Text(label="B")],
    ) -> Annotated[float, Output(label="Remainder")]:
        if b == 0:
            raise ValueError("Modulo by zero")
        return a % b

    @registry.node(
        "math-round", version=1, name="Round",
        description="Rounds to the given number of decimals",
    )
    def round_(
        value: Annotated[float, Text(label="Value")],
        decimals: Annotated[int, Range(label="Decimals", min_val=0, max_val=10, step=1)] = 0,
    ) -> Annotated[float, Output(label="Rounded")]:
        return round(value, decimals)

    @registry.node(
        "math-min", version=1, name="Min",
        description="Minimum of a list of numbers",
    )
    def min_(
        values: Annotated[list[float], Text(label="Values")],
    ) -> Annotated[float, Output(label="Min")]:
        if not values:
            raise ValueError("min requires at least one value")
        return min(values)

    @registry.node(
        "math-max", version=1, name="Max",
        description="Maximum of a list of numbers",
    )
    def max_(
        values: Annotated[list[float], Text(label="Values")],
    ) -> Annotated[float, Output(label="Max")]:
        if not values:
            raise ValueError("max requires at least one value")
        return max(values)

    @registry.node("math-abs", version=1, name="Absolute", description="|value|")
    def abs_(
        value: Annotated[float, Text(label="Value")],
    ) -> Annotated[float, Output(label="Absolute")]:
        return abs(value)
