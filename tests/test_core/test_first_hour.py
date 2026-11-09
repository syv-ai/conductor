"""The programs a new developer writes first, through the three doors."""

from __future__ import annotations

import json
import logging
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
    NodeDefinition,
    NodeRegistry,
    RunState,
    Series,
    Static,
    Table,
    node,
)
from pydantic import BaseModel


class Text(DType, str):
    id = "text"


@node("boom")
def _boom() -> Text:
    raise ValueError("kaboom: the secret")


def test_a_nodes_exception_reaches_the_log_with_its_traceback_and_stays_off_the_state(caplog):
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="a", type="boom", version=1)]), NodeRegistry([_boom]))

    with caplog.at_level(logging.ERROR, logger="conductor"):
        state = compiled.run_sync()

    assert state.status == "failed"
    (logged,) = [record for record in caplog.records if record.name == "conductor"]
    assert logged.exc_info is not None and logged.exc_info[0] is ValueError
    assert "'a'" in logged.getMessage()
    assert state.failure.ctx == {"exception": "ValueError"}
    assert "kaboom" not in json.dumps(state.model_dump(mode="json"))


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


def test_a_str_input_and_output_run_with_no_type_declared_and_the_values_stay_plain():
    compiled = _plain(_n("a", "hello", name=Static("Ida")), _n("b", "shout", text=From("a.result"), times=Static(2)))
    assert compiled.is_runnable, compiled.problems
    value = compiled.results(compiled.run_sync())["b"]["result"]
    assert value == "HELLO IDA!!" and type(value) is str


def test_plain_types_run_per_row_and_reduce_like_any_other():
    compiled = _plain(_n("w", "words", text=Static("a bb")), _n("s", "shout", text=From("w.result")), _n("l", "lengths", words=From("s.result")))
    results = compiled.results(compiled.run_sync())
    assert list(results["s"]["result"]) == ["A!", "BB!"] and results["l"]["result"] == 5.0


@pytest.mark.parametrize(
    ("node_type", "bound", "expected"),
    [
        ("flag", {"on": Static(True)}, False),
        ("next-day", {"day": Static(date(2026, 1, 1))}, date(2026, 1, 2)),
    ],
)
def test_each_plain_type_carries_its_own_value(node_type, bound, expected):
    compiled = _plain(_n("n", node_type, **bound))
    assert compiled.results(compiled.run_sync())["n"]["result"] == expected


def test_a_pydantic_model_travels_on_an_edge_and_through_json():
    compiled = _plain(_n("p", "person", name=Static("Ida")), _n("o", "name-of", person=From("p.result")))
    state = compiled.run_sync()
    assert isinstance(compiled.results(state)["p"]["result"], Person)
    back = RunState.model_validate_json(state.model_dump_json())
    assert compiled.results(back)["o"]["result"] == "Ida"


def test_a_dtype_built_on_str_lands_on_a_str_input_and_not_the_other_way():
    assert _plain(_n("e", "email"), _n("h", "hello", name=From("e.result"))).is_runnable
    refused = _plain(_n("h", "hello", name=Static("x")), _n("w", "wants-email", address=From("h.result")))
    assert [p.type for p in refused.problems] == ["type_mismatch"]


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


def test_a_table_parameter_is_a_frame_and_a_node_returns_a_frame():
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[_n("p", "people"), _n("o", "oldest", people=From("p.result")), _n("h", "headcount", table=From("o.result"))]),
        NodeRegistry([_people, _oldest, _headcount]),
    )
    results = compiled.results(compiled.run_sync())
    assert isinstance(results["o"]["result"], pl.DataFrame) and results["o"]["result"]["name"].to_list() == ["Ida", "Bo"]
    assert results["h"]["result"] == 2


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


def test_an_embedded_graph_is_wired_by_its_own_names_and_titled_like_a_node():
    registry, clean = _clean()
    assert clean.title == "Clean text"
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="c", type="clean-text", version=1, bindings={"text": Static("  hi ")}),
            GraphNode(id="v", type="up", version=1, bindings={"text": From("c.result")}),
        ]),
        registry,
    )
    assert compiled.is_runnable, compiled.problems
    state = compiled.run_sync()
    assert compiled.results(state)["c/u"]["result"] == "HI"          # a run is read by the inner address
    assert compiled.outputs(state) == {"v.result": "HI"}


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
