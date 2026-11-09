"""Arithmetic nodes (``math-add``, ``math-subtract``, etc.)."""

from __future__ import annotations

from typing import Annotated

from conductor import Param, Result, Series
from conductor.widgets import ListWidget, NumberWidget, Range

from conductor_nodes.types import stdlib_node


@stdlib_node("math-add", title="Add", description="a + b", category="math")
def add(
    a: Annotated[float, Param(widget=NumberWidget())],
    b: Annotated[float, Param(widget=NumberWidget())],
) -> Annotated[float, Result(title="Sum")]:
    return float(a + b)


@stdlib_node("math-subtract", title="Subtract", description="a - b", category="math")
def subtract(
    a: Annotated[float, Param(widget=NumberWidget())],
    b: Annotated[float, Param(widget=NumberWidget())],
) -> Annotated[float, Result(title="Difference")]:
    return float(a - b)


@stdlib_node("math-multiply", title="Multiply", description="a * b", category="math")
def multiply(
    a: Annotated[float, Param(widget=NumberWidget())],
    b: Annotated[float, Param(widget=NumberWidget())],
) -> Annotated[float, Result(title="Product")]:
    return float(a * b)


@stdlib_node("math-divide", title="Divide", description="a / b — fails on b == 0", category="math")
def divide(
    a: Annotated[float, Param(widget=NumberWidget())],
    b: Annotated[float, Param(widget=NumberWidget())],
) -> Annotated[float, Result(title="Quotient")]:
    if b == 0:
        raise ValueError("Division by zero")
    return float(a / b)


@stdlib_node("math-modulo", title="Modulo", description="a % b", category="math")
def modulo(
    a: Annotated[float, Param(widget=NumberWidget())],
    b: Annotated[float, Param(widget=NumberWidget())],
) -> Annotated[float, Result(title="Remainder")]:
    if b == 0:
        raise ValueError("Modulo by zero")
    return float(a % b)


@stdlib_node("math-round", title="Round", description="Rounds to the given number of decimals", category="math")
def round_node(
    value: Annotated[float, Param(widget=NumberWidget())],
    decimals: Annotated[
        float, Param(ge=0, le=10, widget=Range(step=1))
    ] = 0.0,
) -> Annotated[float, Result(title="Rounded")]:
    return float(round(value, int(decimals)))


@stdlib_node("math-min", title="Min", description="Minimum of a series of numbers", category="math")
def min_node(
    values: Annotated[
        Series[float], Param(widget=ListWidget())
    ],
) -> Annotated[float, Result(title="Min")]:
    if not len(values):
        raise ValueError("min requires at least one value")
    return float(min(values))


@stdlib_node("math-max", title="Max", description="Maximum of a series of numbers", category="math")
def max_node(
    values: Annotated[
        Series[float], Param(widget=ListWidget())
    ],
) -> Annotated[float, Result(title="Max")]:
    if not len(values):
        raise ValueError("max requires at least one value")
    return float(max(values))


@stdlib_node("math-abs", title="Absolute", description="|value|", category="math")
def absolute(value: Annotated[float, Param(widget=NumberWidget())]
) -> Annotated[float, Result(title="Absolute")]:
    return float(abs(value))


#: Every node this module defines, in palette order.
NODES = (add, subtract, multiply, divide, modulo, round_node, min_node, max_node, absolute)
