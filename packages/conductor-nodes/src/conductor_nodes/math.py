"""Arithmetic nodes: ``math-add``, ``math-divide``, ``math-min`` and the rest."""

import math
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated

from conductor import NodeError, NodeRegistry, Param, Series
from conductor.widgets import ListInput, NumberInput

from conductor_nodes._inputs import Number

registry = NodeRegistry(category="math")

Numbers = Annotated[Series[float], Param(widget=ListInput())]


@registry.node("math-add", title="Add", description="Adds two numbers together.")
def add(a: Number, b: Number) -> Annotated[float, Param(title="Sum")]:
    """a + b"""
    return a + b


@registry.node("math-subtract", title="Subtract", description="Takes the second number away from the first.")
def subtract(a: Number, b: Number) -> Annotated[float, Param(title="Difference")]:
    """a - b"""
    return a - b


@registry.node("math-multiply", title="Multiply", description="Multiplies two numbers.")
def multiply(a: Number, b: Number) -> Annotated[float, Param(title="Product")]:
    """a * b"""
    return a * b


@registry.node("math-divide", title="Divide", description="Divides the first number by the second. Dividing by zero is an error.")
def divide(a: Number, b: Number) -> Annotated[float, Param(title="Quotient")]:
    """a / b; fails when b is 0."""
    if b == 0:
        raise NodeError("Cannot divide by 0. Give a second number other than 0.", code="division_by_zero")
    return a / b


@registry.node("math-modulo", title="Remainder", description="Gives what is left over when the first number is divided by the second. Dividing by zero is an error.")
def modulo(a: Number, b: Number) -> Annotated[float, Param(title="Remainder")]:
    """a % b; fails when b is 0."""
    if b == 0:
        raise NodeError("Cannot divide by 0. Give a second number other than 0.", code="division_by_zero")
    return a % b


@registry.node("math-round", title="Round", description="Rounds a number to the number of decimals you choose, with halves rounded away from zero.")
def round_to(
    value: Number,
    decimals: Annotated[int, Param(ge=0, le=10, widget=NumberInput())] = 0,
) -> Annotated[float, Param(title="Rounded")]:
    """Rounds to the given number of decimals, a half away from zero: 2.5 to 3, -2.5 to -3, 0.125 to 0.13 at 2.

    The number is rounded as it is written, so 0.125 is a half though the float nearest it is a little less.
    """
    if not math.isfinite(value) or abs(value) >= 2**52:  # a float this large has no fraction to round
        return value
    return float(Decimal(str(value)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


@registry.node("math-min", title="Smallest", description="Finds the smallest of the numbers. With no numbers it is an error.")
def smallest(values: Numbers) -> Annotated[float, Param(title="Min")]:
    """The smallest of the numbers; fails when there are none."""
    if not values:
        raise NodeError("There are no numbers. Give at least one number.", code="no_numbers")
    return min(values)


@registry.node("math-max", title="Largest", description="Finds the largest of the numbers. With no numbers it is an error.")
def largest(values: Numbers) -> Annotated[float, Param(title="Max")]:
    """The largest of the numbers; fails when there are none."""
    if not values:
        raise NodeError("There are no numbers. Give at least one number.", code="no_numbers")
    return max(values)


@registry.node("math-abs", title="Absolute value", description="Gives the number without its minus sign.")
def absolute(value: Number) -> Annotated[float, Param(title="Absolute")]:
    """|value|"""
    return abs(value)
