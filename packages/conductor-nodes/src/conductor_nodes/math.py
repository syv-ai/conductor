"""Arithmetic nodes (``math-add``, ``math-subtract``, etc.)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from conductor.returns import Result
from conductor.series import Series
from conductor.widgets import List, Range
from conductor.widgets import Number as NumberWidget

from conductor_nodes.types import Number, StdlibNode

if TYPE_CHECKING:
    from conductor import NodeRegistry


class Add(StdlibNode):
    id = "math-add"
    title = "Add"
    description = "a + b"
    category = "math"

    def run(
        self,
        a: Annotated[Number, NumberWidget(title="A")],
        b: Annotated[Number, NumberWidget(title="B")],
    ) -> Annotated[Number, Result(title="Sum")]:
        return Number(a + b)


class Subtract(StdlibNode):
    id = "math-subtract"
    title = "Subtract"
    description = "a - b"
    category = "math"

    def run(
        self,
        a: Annotated[Number, NumberWidget(title="A")],
        b: Annotated[Number, NumberWidget(title="B")],
    ) -> Annotated[Number, Result(title="Difference")]:
        return Number(a - b)


class Multiply(StdlibNode):
    id = "math-multiply"
    title = "Multiply"
    description = "a * b"
    category = "math"

    def run(
        self,
        a: Annotated[Number, NumberWidget(title="A")],
        b: Annotated[Number, NumberWidget(title="B")],
    ) -> Annotated[Number, Result(title="Product")]:
        return Number(a * b)


class Divide(StdlibNode):
    id = "math-divide"
    title = "Divide"
    description = "a / b — fails on b == 0"
    category = "math"

    def run(
        self,
        a: Annotated[Number, NumberWidget(title="A")],
        b: Annotated[Number, NumberWidget(title="B")],
    ) -> Annotated[Number, Result(title="Quotient")]:
        if b == 0:
            raise ValueError("Division by zero")
        return Number(a / b)


class Modulo(StdlibNode):
    id = "math-modulo"
    title = "Modulo"
    description = "a % b"
    category = "math"

    def run(
        self,
        a: Annotated[Number, NumberWidget(title="A")],
        b: Annotated[Number, NumberWidget(title="B")],
    ) -> Annotated[Number, Result(title="Remainder")]:
        if b == 0:
            raise ValueError("Modulo by zero")
        return Number(a % b)


class Round(StdlibNode):
    id = "math-round"
    title = "Round"
    description = "Rounds to the given number of decimals"
    category = "math"

    def run(
        self,
        value: Annotated[Number, NumberWidget(title="Value")],
        decimals: Annotated[
            Number, Range(title="Decimals", min_val=0, max_val=10, step=1)
        ] = Number(0),
    ) -> Annotated[Number, Result(title="Rounded")]:
        return Number(round(value, int(decimals)))


class Min(StdlibNode):
    id = "math-min"
    title = "Min"
    description = "Minimum of a series of numbers"
    category = "math"

    def run(
        self,
        values: Annotated[
            Series[Number], List(title="Values")
        ],
    ) -> Annotated[Number, Result(title="Min")]:
        if not len(values):
            raise ValueError("min requires at least one value")
        return Number(min(values))


class Max(StdlibNode):
    id = "math-max"
    title = "Max"
    description = "Maximum of a series of numbers"
    category = "math"

    def run(
        self,
        values: Annotated[
            Series[Number], List(title="Values")
        ],
    ) -> Annotated[Number, Result(title="Max")]:
        if not len(values):
            raise ValueError("max requires at least one value")
        return Number(max(values))


class Absolute(StdlibNode):
    id = "math-abs"
    title = "Absolute"
    description = "|value|"
    category = "math"

    def run(
        self, value: Annotated[Number, NumberWidget(title="Value")]
    ) -> Annotated[Number, Result(title="Absolute")]:
        return Number(abs(value))


NODES = (Add, Subtract, Multiply, Divide, Modulo, Round, Min, Max, Absolute)


def register(registry: "NodeRegistry") -> None:
    """Register every math node on the supplied registry."""
    for node_cls in NODES:
        registry.register(node_cls)
