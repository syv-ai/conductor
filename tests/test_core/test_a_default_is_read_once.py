"""What an input may hold is decided in one place, and a default is held to it where it is written."""

from typing import Annotated

import pytest
from conductor import (
    CompiledGraph,
    DType,
    From,
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


def test_a_default_is_kept_as_its_type_read_it():
    assert isinstance(Input("n", Whole, default=3).default, Whole)
    seen = []

    def take(n: Whole = 3) -> int:  # type: ignore[assignment]
        seen.append(n)
        return n

    made = node("take")(take)
    assert made.versions[1].interface.inputs[0].default.__class__ is Whole
    compiled = CompiledGraph.from_graph(Graph(nodes=[made.graph_node("t")]), NodeRegistry([made]))
    assert compiled.run_sync().status == "complete" and seen[0].__class__ is Whole


def _compiled_with_a_series_or_nothing(**bindings):
    def s(xs: Series[str] | None = None) -> str:
        return repr(xs)

    made = node("s")(s)
    return CompiledGraph.from_graph(Graph(nodes=[made.graph_node("a", **bindings)]), NodeRegistry([made]))


def test_a_series_that_may_be_nothing_reads_an_explicit_none_like_leaving_it_out():
    set_to_none, left_out = _compiled_with_a_series_or_nothing(xs=None), _compiled_with_a_series_or_nothing()
    assert set_to_none.problems == left_out.problems == ()
    assert set_to_none.outputs(set_to_none.run_sync()) == left_out.outputs(left_out.run_sync()) == {"a.result": "None"}


def test_a_static_is_read_once_at_compile_and_not_again_on_each_call():
    reads = []

    class Counted(DType, str):
        id = "default-test-counted"

        @classmethod
        def __get_pydantic_core_schema__(cls, source, handler):
            from pydantic_core import core_schema

            def read(value):
                reads.append(value)
                return value

            return core_schema.no_info_plain_validator_function(read)

    def fan() -> Series[int]:
        return [1, 2, 3]

    def use(row: int, fixed: Counted) -> str:
        return f"{row}{fixed}"

    made_fan, made_use = node("fan")(fan), node("use")(use)
    compiled = CompiledGraph.from_graph(Graph(nodes=[made_fan.graph_node("f"), made_use.graph_node("u", row=From("f.result"), fixed="x")]), NodeRegistry([made_fan, made_use]))
    assert not compiled.problems
    reads.clear()
    assert compiled.run_sync().status == "complete"
    assert reads == []
