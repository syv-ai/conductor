"""Conductor's three walks over a type's pydantic schema reach the same places.

A type's schema is walked to add the rule that a yes or no is not a number, to refuse a set a node declares, and to
save a value that holds a host's model so a saved run reads it back. Each walk asks one rule which parts of a
schema are the schemas inside it. These tests put a leaf at each place a schema keeps one, and show that every walk
reaches it: a ``set[int]`` there is refused when the node is declared, ``True`` for an ``int`` there is refused when
the graph runs, and a model whose float is ``inf`` there is saved and read back.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict, Union

import pytest
from conductor import (
    Graph,
    InvalidCallError,
    NodeDeclarationError,
    NodeRegistry,
    Run,
    host_type,
    make_graph,
    node,
)
from pydantic import BaseModel, ConfigDict, Discriminator, GetPydanticSchema, WrapValidator
from pydantic_core import core_schema


@host_type("walks-reading", title="Reading")
class Reading(BaseModel):
    """A host's model, written by its own config, so its ``inf`` is saved only if the codec's walk finds it."""

    model_config = ConfigDict(frozen=True)
    value: float


class IntChain(TypedDict):
    leaf: int
    next: IntChain | None


class SetChain(TypedDict):
    leaf: set[int]
    next: SetChain | None


class ReadingChain(TypedDict):
    leaf: Reading
    next: ReadingChain | None


#: The TypedDict that refers to itself, so pydantic writes it as ``definitions`` around a ``definition-ref``.
_CHAINS: dict[Any, type] = {int: IntChain, set[int]: SetChain, Reading: ReadingChain}


def _tagged(leaf: Any) -> Any:
    a = TypedDict("A", {"kind": Literal["a"], "leaf": leaf})  # type: ignore[misc]
    b = TypedDict("B", {"kind": Literal["b"]})  # type: ignore[misc]
    return Annotated[Union[a, b], Discriminator("kind")]  # noqa: UP007


def _wrapped(value: Any, handler: Callable[[Any], Any]) -> Any:
    return handler(value)


def _chained(leaf: Any) -> Any:
    """``leaf`` checked as the last step of a ``chain`` schema, after a step that takes any value."""
    return Annotated[
        leaf,
        GetPydanticSchema(
            lambda source, handler: core_schema.chain_schema(
                [core_schema.any_schema(), handler(source)]
            )
        ),
    ]


#: Each place a schema keeps a schema, as the hint that puts ``leaf`` there and the value that puts a leaf's value
#: there.
SHAPES: dict[str, tuple[Callable[[Any], Any], Callable[[Any], Any]]] = {
    "list": (lambda leaf: list[leaf], lambda v: [v]),
    "dict": (lambda leaf: dict[str, leaf], lambda v: {"a": v}),
    "tuple": (lambda leaf: tuple[str, leaf], lambda v: ("a", v)),
    "variadic tuple": (lambda leaf: tuple[leaf, ...], lambda v: (v,)),
    "nullable": (lambda leaf: leaf | None, lambda v: v),
    "union": (lambda leaf: Union[leaf, str], lambda v: v),  # noqa: UP007
    "function-wrap": (lambda leaf: Annotated[leaf, WrapValidator(_wrapped)], lambda v: v),
    "chain": (_chained, lambda v: v),
    "typed-dict": (lambda leaf: TypedDict("Box", {"leaf": leaf}), lambda v: {"leaf": v}),  # type: ignore[misc]
    "tagged-union": (_tagged, lambda v: {"kind": "a", "leaf": v}),
    "definitions": (
        lambda leaf: _CHAINS[leaf],
        lambda v: {"leaf": v, "next": {"leaf": v, "next": None}},
    ),
}


def _graph(function: Callable[..., Any], **hints: Any) -> Graph[None]:
    """Declares ``function`` as a node with ``hints`` as its annotations, and compiles a graph of it alone."""
    function.__annotations__ = hints
    definition = node("take", title="Take", description="For a test.")(function)
    registry = NodeRegistry()
    registry.add(definition)
    return Graph.compile(make_graph(definition), registry)


def _taking(hint: Any) -> Graph[None]:
    """A graph of one node that takes a graph input ``value`` of type ``hint`` and returns it."""

    def take(value: Any) -> Any:
        return value

    return _graph(take, value=hint, **{"return": hint})


@pytest.mark.parametrize("shape", SHAPES)
def test_a_set_at_each_place_in_a_schema_is_refused(shape: str) -> None:
    """The declaration check reaches every place a schema keeps a schema, so a set there is refused."""
    hint, _ = SHAPES[shape]
    with pytest.raises(NodeDeclarationError) as raised:
        _taking(hint(set[int]))
    assert [problem.type for problem in raised.value.problems] == ["type_not_supported_set"]


@pytest.mark.parametrize("shape", SHAPES)
def test_true_at_each_place_in_a_schema_is_refused_for_an_int(shape: str) -> None:
    """The walk that adds conductor's rules reaches the same places, so ``True`` there is refused for an ``int``."""
    hint, value = SHAPES[shape]
    graph = _taking(hint(int))
    assert graph.run({"value": value(1)}).outputs["result"] == value(1)
    with pytest.raises(InvalidCallError) as refused:
        graph.run({"value": value(True)})
    (problem,) = refused.value.problems
    assert "bool_is_not_a_number" in [error["type"] for error in problem.ctx["errors"]]


@pytest.mark.parametrize("shape", SHAPES)
def test_a_model_at_each_place_in_a_schema_is_saved_and_read_back(shape: str) -> None:
    """The codec's walk reaches the same places, so a model's ``inf`` there is saved as text and read back."""
    hint, value = SHAPES[shape]
    returned = value(Reading(value=math.inf))

    def make() -> Any:
        return returned

    graph = _graph(make, **{"return": hint(Reading)})
    run = graph.run({})
    assert Run.from_json(run.to_json(), graph).outputs == run.outputs == {"result": returned}


def test_a_set_in_a_hosts_model_or_dataclass_is_refused() -> None:
    """The declaration check enters a host's own class too, since its values are saved and hashed like any other."""

    @host_type("walks-tagged-model", title="Tagged model")
    class TaggedModel(BaseModel):
        model_config = ConfigDict(frozen=True)
        tags: set[int]

    @host_type("walks-tagged-dataclass", title="Tagged dataclass")
    @dataclass(frozen=True)
    class TaggedDataclass:
        tags: set[int]

    for hint, where in (
        (TaggedModel, "TaggedModel.tags"),
        (TaggedDataclass, "TaggedDataclass.tags"),
    ):
        with pytest.raises(NodeDeclarationError) as raised:
            _taking(hint)
        (problem,) = raised.value.problems
        assert (problem.type, problem.ctx) == ("type_not_supported_set", {"type": where})
