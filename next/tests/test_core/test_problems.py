"""What is wrong with a graph, as data."""

from typing import Annotated

import pytest
from conductor import (
    DType,
    From,
    NodeRegistry,
    Output,
    Param,
    Problem,
    Result,
    Series,
    node,
)
from conductor.widgets import Textarea
from pydantic import ValidationError


def test_a_problem_names_a_code_a_message_and_a_place():
    p = Problem(type="unknown_input", msg="The node has no such field.", loc=("letter", "template"))

    assert p.type == "unknown_input"
    assert p.loc[0] == "letter"
    assert p.loc[1] == "template"


def test_a_problem_about_a_whole_node_names_no_field():
    p = Problem(type="cycle", msg="The graph loops back on itself.", loc=("letter",))

    assert p.loc[0] == "letter"
    assert p.loc[1:] == ()


def test_a_problem_is_frozen():
    p = Problem(type="c", msg="m", loc=("n",))

    with pytest.raises(Exception):
        p.type = "other"


def test_problems_compare_by_value():
    a = Problem(type="c", msg="m", loc=("n",))
    b = Problem(type="c", msg="m", loc=("n",))

    assert a == b


def test_details_carry_what_the_message_names():
    """A host that translates by code needs the values, not the sentence."""
    p = Problem(type="unknown_ref_node", msg="Field 'text' is connected to 'a', which is not in the graph.", loc=("b", "text"), ctx={"source_node": "a"})

    assert p.ctx == {"source_node": "a"}
    assert Problem(type="c", msg="m", loc=("n",)).ctx == {}


class Txt(DType, str):
    id = "problems-test-txt"
    display_title = "Text"


class Nonempty(DType, str):
    """A type with a constructor rule: a value is one only when it has text."""

    id = "problems-test-nonempty"
    display_title = "Non-empty text"

    def __new__(cls, value=""):
        if not value:
            raise ValueError("The field must be filled in.")
        return super().__new__(cls, value)


Out = Annotated[Txt, Result(title="Result")]


@node("echo", title="Echo", description="d", category="test")
def echo(
    x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt(""),
    y: Annotated[Txt, Param(title="Y", widget=Textarea())] = Txt(""),
) -> Out:
    return Txt(x + y)


@node("script", title="Python", description="d", category="test")
def script(code: Annotated[Txt, Param(title="Code", connectable=False, widget=Textarea())] = Txt("")) -> Out:
    return code


@node("needs", title="Needs", description="d", category="test")
def needs(x: Annotated[Txt, Param(title="X", widget=Textarea())]) -> Out:
    return x


@node("join", title="Join", description="d", category="test")
def join_node(texts: Annotated[Series[Txt], Param(title="Texts")] = ()) -> Out:
    return Txt("\n".join(texts))


@node("picky", title="Picky", description="d", category="test")
def picky(x: Annotated[Nonempty, Param(title="X", widget=Textarea())] = Nonempty("x")) -> Out:
    """Its constraint is its type's: `x` must be a `Nonempty`."""
    return Txt(x)


def twice_outputs():
    return (
        Output("x", Txt, title="X"),
        Output("x", Txt, title="X again"),
        Output("value", Txt, title="Value"),
    )

@node("twice", title="Twice", description="d", category="test", compute_outputs=twice_outputs)
def twice(value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt("")) -> Out:
    """An interface naming one field twice: one name on two outputs, and one on both sides."""
    return value


def _registry():
    registry = NodeRegistry()
    for node_cls in (echo, script, needs, join_node, picky, twice):
        registry = registry.extended_with([node_cls])
    return registry


# --- a value's own rule is its type's ---------------------------------------------


# --- a field name is unique within a node -------------------------------------------


# --- the stored bindings, validated -----------------------------------------------


# --- the codes compile emits are declared once ------------------------------------


def test_every_code_is_declared_once_in_the_catalogue():
    """``PROBLEM_CODES`` is what a host translating by code has to cover: the keys of
    the one table every emitting site reads. A site naming a code the table
    lacks raises where it is written, so nothing can be emitted undeclared."""
    from conductor import PROBLEM_CODES
    from conductor._internal.errors import PROBLEM_CATALOGUE, make_problem

    assert PROBLEM_CODES == frozenset(PROBLEM_CATALOGUE)
    with pytest.raises(KeyError):
        make_problem("no_such_code", "n")


def test_a_slot_the_details_do_not_fill_raises_unless_it_is_optional():
    """A template typo or a forgotten keyword is a programming error, caught
    where the problem is built; only the type's own sentence may be absent."""
    from conductor._internal.errors import make_problem

    with pytest.raises(KeyError):
        make_problem("unknown_ref_node", "b", "text")  # no source_node
    assert make_problem("invalid_static", "b", "text", input="1", expected="text").msg == "The value 1 in 'text' cannot be read as text."


def test_a_problem_is_formatted_from_its_details_and_keeps_them():
    from conductor._internal.errors import make_problem

    p = make_problem("unknown_ref_node", "b", "text", source_node="a")
    assert p.msg == "Field 'text' is connected to 'a', which is not in the graph."
    assert p.ctx == {"source_node": "a"}
    assert p.fatal is True
    assert make_problem("unknown_locked_field", "b", "old").fatal is False


# --- what compile refuses ----------------------------------------------------------------


def test_an_edge_with_no_refs_is_refused_where_it_is_written():
    """``From()`` is not a binding: the model refuses it before compile sees it."""

    with pytest.raises(ValidationError):
        From()


