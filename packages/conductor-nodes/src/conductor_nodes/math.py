"""Arithmetic nodes: ``math-add``, ``math-divide``, ``math-min`` and the rest."""

import builtins
from typing import Annotated

from conductor import NodeError, NodeRegistry, Param, Series
from conductor.widgets import ListInput, NumberInput

from conductor_nodes._inputs import Number

registry = NodeRegistry(category="math")

Numbers = Annotated[Series[float], Param(widget=ListInput())]


@registry.node("math-add")
def add(a: Number, b: Number) -> Annotated[float, Param(title="Sum")]:
    """a + b"""
    return a + b


@registry.node("math-subtract")
def subtract(a: Number, b: Number) -> Annotated[float, Param(title="Difference")]:
    """a - b"""
    return a - b


@registry.node("math-multiply")
def multiply(a: Number, b: Number) -> Annotated[float, Param(title="Product")]:
    """a * b"""
    return a * b


@registry.node("math-divide")
def divide(a: Number, b: Number) -> Annotated[float, Param(title="Quotient")]:
    """a / b; fails when b is 0."""
    if b == 0:
        raise NodeError("Division by zero.", code="division_by_zero")
    return a / b


@registry.node("math-modulo")
def modulo(a: Number, b: Number) -> Annotated[float, Param(title="Remainder")]:
    """a % b; fails when b is 0."""
    if b == 0:
        raise NodeError("Modulo by zero.", code="division_by_zero")
    return a % b


@registry.node("math-round", title="Round")
def round_to(
    value: Number,
    decimals: Annotated[int, Param(ge=0, le=10, widget=NumberInput(integer_only=True))] = 0,
) -> Annotated[float, Param(title="Rounded")]:
    """Rounds to the given number of decimals."""
    return builtins.round(value, decimals)


@registry.node("math-min", title="Min")
def smallest(values: Numbers) -> Annotated[float, Param(title="Min")]:
    """The smallest of the numbers; fails when there are none."""
    if not values:
        raise NodeError("There are no numbers.", code="no_numbers")
    return builtins.min(values)


@registry.node("math-max", title="Max")
def largest(values: Numbers) -> Annotated[float, Param(title="Max")]:
    """The largest of the numbers; fails when there are none."""
    if not values:
        raise NodeError("There are no numbers.", code="no_numbers")
    return builtins.max(values)


@registry.node("math-abs", title="Absolute")
def absolute(value: Number) -> Annotated[float, Param(title="Absolute")]:
    """|value|"""
    return builtins.abs(value)
