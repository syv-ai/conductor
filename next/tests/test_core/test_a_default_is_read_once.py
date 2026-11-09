"""What an input may hold is decided in one place, and a default is held to it where it is written."""

from typing import Annotated

import pytest
from conductor import (
    CompiledGraph,
    DType,
    Graph,
    Input,
    NodeDeclarationError,
    NodeRegistry,
    Param,
    Series,
    node,
)


class Whole(DType, int):
    id = "default-test-whole"


def test_a_default_its_type_cannot_read_fails_where_it_is_written():
    def bad(x: int = "abc") -> int:  # type: ignore[assignment]
        return x

    with pytest.raises(NodeDeclarationError) as raised:
        node("bad")(bad)
    assert raised.value.code == "default_invalid" and "'x'" in str(raised.value)


def test_a_default_outside_its_own_bounds_fails_where_it_is_written():
    def bad(x: Annotated[int, Param(ge=5)] = 1) -> int:
        return x

    with pytest.raises(NodeDeclarationError) as raised:
        node("bad")(bad)
    assert raised.value.code == "default_invalid"


def test_a_series_default_is_held_to_its_length():
    with pytest.raises(NodeDeclarationError):
        Input("xs", Series[str], default=[], min_length=1)


def _compiled_with_a_series_or_nothing(**bindings):
    def s(xs: Series[str] | None = None) -> str:
        return repr(xs)

    made = node("s")(s)
    return CompiledGraph.from_graph(Graph(nodes=[made.graph_node("a", **bindings)]), NodeRegistry([made]))


