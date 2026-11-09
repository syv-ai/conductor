"""The programs a new developer writes first, through the three doors."""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import TypedDict

import polars as pl
import pytest
from conductor import (
    DType,
    From,
    GraphNode,
    NodeDeclarationError,
    NodeRegistry,
    Series,
    Static,
    Table,
    node,
)
from pydantic import BaseModel, ValidationError


class Text(DType, str):
    id = "text"


@node("boom")
def _boom() -> Text:
    raise ValueError("kaboom: the secret")


def test_a_text_built_on_str_can_call_str_title():
    assert Text("hello world").title() == "Hello World"


# -- plain Python types travel on edges ------------------------------------------------


class Person(BaseModel):
    name: str


class Email(DType, str):
    id = "email"


@node("hello")
def _hello(name: str) -> str:
    return f"hello {name}"


@node("shout")
def _shout(text: str, times: int = 1) -> str:
    return text.upper() + "!" * times


@node("lengths")
def _lengths(words: Series[str]) -> float:
    return float(sum(len(word) for word in words))


@node("words")
def _words(text: str) -> Series[str]:
    return text.split()


@node("flag")
def _flag(on: bool) -> bool:
    return not on


@node("next-day")
def _next_day(day: date) -> date:
    return day + timedelta(days=1)


@node("person")
def _person(name: str) -> Person:
    return Person(name=name)


@node("name-of")
def _name_of(person: Person) -> str:
    return person.name


@node("email")
def _email() -> Email:
    return Email("ida@x")


@node("wants-email")
def _wants_email(address: Email) -> str:
    return address


@node("count")
def _count() -> int:
    return 3


@node("half")
def _half(value: float) -> float:
    return value / 2


PLAIN = NodeRegistry([_hello, _shout, _lengths, _words, _flag, _next_day, _person, _name_of, _email, _wants_email, _count, _half])


def _n(id: str, type: str, **bindings) -> GraphNode:
    return GraphNode(id=id, type=type, version=1, bindings=bindings)


def test_a_table_declares_plain_columns():
    class Contact(TypedDict):
        name: str
        age: int

    assert [(n, d.id) for n, d in Table[Contact].columns] == [("name", "str"), ("age", "int")]


def test_a_type_no_edge_can_carry_closes_the_input_and_keeps_out_of_the_vocabulary():
    @node("tags")
    def tags(names: list[str] = []) -> str:  # noqa: B006
        return ",".join(names)

    registry = NodeRegistry([tags])
    assert "setting" not in registry.types
    (names,) = tags.versions[1].interface.inputs
    assert names.connectable is False and names.dtype.describe() == {"id": "setting", "of": "list"}


# -- a table is a frame ----------------------------------------------------------------


class Contact(TypedDict):
    name: str
    age: int


@node("people")
def _people() -> Table[Contact]:
    return pl.DataFrame({"name": ["Ida", "Bo"], "age": [40, 30]})


@node("oldest")
def _oldest(people: Table[Contact]) -> Table[Contact]:
    assert isinstance(people, pl.DataFrame) and len(people) == 2
    return people.sort("age", descending=True)


@node("headcount")
def _headcount(table: Table) -> int:
    return table.height


def test_table_is_only_a_type():
    with pytest.raises(TypeError, match="polars.DataFrame"):
        Table({"a": [1]})  # type: ignore[call-arg]


# -- an embedded graph has its own interface -------------------------------------------


@node("trim")
def _trim(text: str) -> str:
    return text.strip()


@node("up")
def _up(text: str) -> str:
    return text.upper()


# -- a node placed in a graph, typed -----------------------------------------------------


def test_graph_node_is_a_plain_graph_node_at_the_current_version():
    placed = _shout.graph_node("b", text=From("a.result"), times=2)
    assert placed == GraphNode(id="b", type="shout", version=1, bindings={"text": From("a.result"), "times": Static(2)})


def test_graph_node_binds_positional_arguments_to_their_parameters():
    assert _shout.graph_node("b", From("a.result"), 3).bindings == {"text": From("a.result"), "times": Static(3)}


def test_graph_node_refuses_a_name_the_node_doesnt_have_where_it_is_written():
    with pytest.raises(TypeError, match="txt"):
        _shout.graph_node("b", txt="x")  # type: ignore[call-arg]


# -- errors say what to do ---------------------------------------------------------------


def test_a_binding_that_is_neither_from_nor_static_is_one_error():
    with pytest.raises(ValidationError) as raised:
        GraphNode(id="a", type="t", version=1, bindings={"text": "hi"})
    (error,) = raised.value.errors()
    assert error["type"] == "not_a_binding"


class Clock:
    pass


@pytest.mark.parametrize("written", ["", 5])
def test_a_node_id_must_be_a_string_with_something_in_it(written):
    with pytest.raises(NodeDeclarationError, match="id"):
        node(written)


def test_a_declaration_error_names_the_function():
    with pytest.raises(NodeDeclarationError, match=r"@node\('x'\) on bad"):
        @node("x")
        def bad(a):  # no annotation
            return a


# -- a question's title -------------------------------------------------------------------


# -- a float that is not finite ------------------------------------------------------------


NOT_FINITE: dict[str, float] = {}


@node("not-finite")
def _not_finite() -> float:
    return NOT_FINITE["value"]


@node("not-finite-many")
def _not_finite_many() -> Series[float]:
    return [1.5, NOT_FINITE["value"]]


@node("not-finite-per-row")
def _not_finite_per_row(text: str) -> float:
    return NOT_FINITE["value"] if text == "b" else 1.5


@node("passes-on")
def _passes_on(value: float) -> float:
    return value


@node("int-of-not-finite")
def _int_of_not_finite() -> int:
    return NOT_FINITE["value"]  # type: ignore[return-value]


_FLOATS = NodeRegistry([_words, _not_finite, _not_finite_many, _not_finite_per_row, _passes_on, _half, _int_of_not_finite])


def _same_float(got: float, wanted: float) -> bool:
    return isinstance(got, float) and (math.isnan(got) if math.isnan(wanted) else got == wanted)


