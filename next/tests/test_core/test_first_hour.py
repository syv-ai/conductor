"""The programs a new developer writes first, through the three doors."""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from typing import TypedDict

import polars as pl
import pytest
from conductor import (
    CompiledGraph,
    DType,
    From,
    Graph,
    GraphNode,
    InvalidCallError,
    NodeDeclarationError,
    NodeDefinition,
    NodeRegistry,
    Ref,
    RunState,
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


def _plain(*nodes: GraphNode) -> CompiledGraph:
    compiled = CompiledGraph.from_graph(Graph(nodes=nodes), PLAIN)
    return compiled


def _n(id: str, type: str, **bindings) -> GraphNode:
    return GraphNode(id=id, type=type, version=1, bindings=bindings)


def test_a_dtype_built_on_str_lands_on_a_str_input_and_not_the_other_way():
    assert _plain(_n("e", "email"), _n("h", "hello", name=From("e.result"))).is_runnable
    raised = _plain(_n("h", "hello", name=Static("x")), _n("w", "wants-email", address=From("h.result")))
    assert [p.type for p in raised.problems] == ["type_mismatch"]


def test_an_int_lands_on_a_float_as_a_type_checker_says():
    assert _plain(_n("c", "count"), _n("h", "half", value=From("c.result"))).is_runnable


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


def _clean() -> tuple[NodeRegistry, NodeDefinition]:
    base = NodeRegistry([_trim, _up])
    inner = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="t", type="trim", version=1), GraphNode(id="u", type="up", version=1, bindings={"text": From("t.result")})]), base,
    )
    clean = NodeDefinition.from_compiled_graphs("clean-text", versions=inner)
    return base.extended_with([clean]), clean


def test_renaming_a_node_inside_breaks_no_caller():
    def graph_with_inner_id(inner_id: str) -> CompiledGraph:
        base = NodeRegistry([_trim])
        inner = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id=inner_id, type="trim", version=1)]), base)
        wrapped = NodeDefinition.from_compiled_graphs("wrapped", versions=inner)
        return CompiledGraph.from_graph(
            Graph(nodes=[GraphNode(id="w", type="wrapped", version=1, bindings={"text": Static(" x ")})]), base.extended_with([wrapped]),
        )

    assert graph_with_inner_id("before").is_runnable and graph_with_inner_id("after-rename").is_runnable


def test_an_inner_address_never_wires_into_or_out_of_an_embedded_graph():
    registry, _ = _clean()
    binding = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="c", type="clean-text", version=1, bindings={"t.text": Static("x")})]), registry)
    assert [(p.type, p.loc[1]) for p in binding.problems if p.type == "stale_binding"] == [("stale_binding", "t.text")]

    edge = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="c", type="clean-text", version=1, bindings={"text": Static("x")}),
            GraphNode(id="v", type="up", version=1, bindings={"text": From("c.u.result")}),
        ]),
        registry,
    )
    assert [p.type for p in edge.problems] == ["unknown_ref_output"]

    slash = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="c", type="clean-text", version=1, bindings={"text": Static("x")}),
            GraphNode(id="v", type="up", version=1, bindings={"text": From("c/u.result")}),
        ]),
        registry,
    )
    assert [p.type for p in slash.problems] == ["unknown_ref_node"]


def test_a_clash_is_named_by_address_and_only_then():
    @node("join-two")
    def _join_two(a: str, b: str) -> str:
        return a + b

    base = NodeRegistry([_trim, _join_two])
    inner = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="x", type="trim", version=1), GraphNode(id="y", type="trim", version=1)]), base,
    )
    clashing = NodeDefinition.from_compiled_graphs("two", versions=inner)
    described = clashing.describe().versions[1]
    assert [i.name for i in described.inputs] == ["x.text", "y.text"]
    assert [o.name for o in described.outputs] == ["x.result", "y.result"]


# -- a node placed in a graph, typed -----------------------------------------------------


def test_graph_node_is_a_plain_graph_node_at_the_current_version():
    placed = _shout.graph_node("b", text=From("a.result"), times=2)
    assert placed == GraphNode(id="b", type="shout", version=1, bindings={"text": From("a.result"), "times": Static(2)})


def test_graph_node_binds_positional_arguments_to_their_parameters():
    assert _shout.graph_node("b", From("a.result"), 3).bindings == {"text": From("a.result"), "times": Static(3)}


def test_graph_node_refuses_a_name_the_node_doesnt_have_where_it_is_written():
    with pytest.raises(TypeError, match="txt"):
        _shout.graph_node("b", txt="x")  # type: ignore[call-arg]


def test_graph_node_keeps_a_graph_it_was_given_on_a_node_that_is_a_graph():
    registry, clean = _clean()
    placed = clean.graph_node("c", text="  x ")
    assert placed.bindings == {"text": Static("  x ")}
    compiled = CompiledGraph.from_graph(Graph(nodes=[placed]), registry)
    assert compiled.is_runnable, compiled.problems


# -- errors say what to do ---------------------------------------------------------------


def test_an_edge_to_a_missing_output_says_what_it_has():
    compiled = _plain(_n("a", "hello", name=Static("x")), _n("b", "shout", text=From("a.reslt")))
    (found,) = compiled.problems
    assert found.type == "unknown_ref_output" and found.ctx["outputs"] == "result" and "it has result" in found.msg


def test_a_bad_typed_in_value_says_what_it_was_and_what_it_should_be():
    compiled = _plain(_n("a", "shout", text=Static("x"), times=Static("many")))
    (found,) = compiled.problems
    assert found.type == "invalid_static" and found.ctx["input"] == "'many'" and found.ctx["expected"] == "int"
    assert "'many'" in found.msg and "int" in found.msg


def test_a_typo_on_an_input_is_one_problem():
    compiled = _plain(_n("a", "shout", txt=Static("x")))
    assert [p.type for p in compiled.problems] == ["stale_binding"]


def test_each_missing_input_beside_a_typo_is_still_reported():
    compiled = _plain(_n("a", "hello", txt=Static("x")), _n("b", "shout"))
    assert [p.type for p in compiled.problems] == ["stale_binding", "unbound_required"]


def test_a_binding_that_is_neither_from_nor_static_is_one_error():
    with pytest.raises(ValidationError) as raised:
        GraphNode(id="a", type="t", version=1, bindings={"text": "hi"})
    (error,) = raised.value.errors()
    assert error["type"] == "not_a_binding"


class Clock:
    pass


def test_with_inputs_raises_for_a_bad_value_and_a_typo_together():
    compiled = _plain(_n("a", "shout", text=Static("x")), _n("b", "flag", on=Static(True)))
    with pytest.raises(InvalidCallError) as raised:
        compiled.with_inputs(times="many", txt="y")
    assert sorted((p.type, p.loc) for p in raised.value.errors()) == [("invalid_static", ("inputs", "times")), ("not_offered", ("inputs", "txt"))]


@pytest.mark.parametrize("written", ["", 5])
def test_a_node_id_must_be_a_string_with_something_in_it(written):
    with pytest.raises(NodeDeclarationError, match="id"):
        node(written)


def test_a_declaration_error_names_the_function():
    with pytest.raises(NodeDeclarationError, match=r"@node\('x'\) on bad"):
        @node("x")
        def bad(a):  # no annotation
            return a


def test_a_field_is_read_by_its_address_as_a_string_too():
    compiled = _plain(_n("a", "hello", name=Static("x")))
    assert compiled.field("a.result") is compiled.field(Ref("a.result"))


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


def _through_json(state: RunState) -> RunState:
    written = state.model_dump_json()
    json.loads(written, parse_constant=lambda bad: pytest.fail(f"{bad} is not JSON"))
    return RunState.model_validate_json(written)


