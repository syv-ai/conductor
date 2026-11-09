"""The programs a new developer writes first, through the three doors."""

from __future__ import annotations

import json
import logging
import math
from datetime import date, timedelta
from typing import TypedDict

import polars as pl
import pytest
from conductor import (
    AskPerson,
    CompiledGraph,
    DType,
    From,
    FromRun,
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


def test_a_nodes_exception_reaches_the_log_with_its_traceback_and_stays_off_the_state(caplog):
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="a", type="boom", version=1)]), NodeRegistry([_boom]))

    with caplog.at_level(logging.ERROR, logger="conductor"):
        state = compiled.run_sync()

    assert state.status == "failed"
    (logged,) = [record for record in caplog.records if record.name == "conductor"]
    assert logged.exc_info is not None and logged.exc_info[0] is ValueError
    assert "'a'" in logged.getMessage()
    assert state.cause.ctx == {"exception": "ValueError"}
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
    assert results["s"]["result"]["values"] == ["A!", "BB!"] and results["l"]["result"] == 5.0


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
    return base.with_nodes([clean]), clean


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
            Graph(nodes=[GraphNode(id="w", type="wrapped", version=1, bindings={"text": Static(" x ")})]), base.with_nodes([wrapped]),
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
    assert [i.name for i in described.interface.inputs] == ["x.text", "y.text"]
    assert [o.name for o in described.interface.outputs] == ["x.result", "y.result"]


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


def test_a_column_into_a_one_value_input_stays_legal():
    @node("rows")
    def rows() -> Table[Contact]:
        return pl.DataFrame({"name": ["a", "b"], "age": [1, 2]})

    compiled = CompiledGraph.from_graph(
        Graph(nodes=[rows.graph_node("r"), _shout.graph_node("s", text=From("r.result.name"))]), NodeRegistry([rows, _shout]),
    )
    assert compiled.results(compiled.run_sync())["s"]["result"]["values"] == ["A!", "B!"]


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


def test_from_run_missing_raises_at_the_call():
    @node("timed")
    def timed(clock: FromRun[Clock]) -> str:
        return "now"

    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="t", type="timed", version=1)]), NodeRegistry([timed]))
    with pytest.raises(InvalidCallError) as raised:
        compiled.run_sync()
    (error,) = raised.value.errors()
    assert error.type == "from_run_missing" and error.ctx["node"] == "t" and error.ctx["needs"] == "Clock"


def test_a_from_run_key_that_is_no_type_is_refused_and_a_type_no_node_needs_is_fine():
    compiled = _plain(_n("a", "hello", name=Static("x")))
    with pytest.raises(InvalidCallError) as raised:
        compiled.run_sync(from_run={"x": 1})
    assert [p.type for p in raised.value.errors()] == ["from_run_not_a_type"]
    assert compiled.run_sync(from_run={int: 1}).status == "complete"


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


def test_using_a_stream_without_async_with_says_how():
    compiled = _plain(_n("a", "hello", name=Static("x")))
    with pytest.raises(TypeError, match="async with"):
        aiter(compiled.stream())


def test_a_field_is_read_by_its_address_as_a_string_too():
    compiled = _plain(_n("a", "hello", name=Static("x")))
    assert compiled.field("a.result") is compiled.field(Ref("a.result"))


def test_a_state_that_is_no_state_is_refused_at_the_call():
    compiled = _plain(_n("a", "hello", name=Static("x")))
    with pytest.raises(InvalidCallError):
        compiled.run_sync({})  # type: ignore[arg-type]


# -- a question's title -------------------------------------------------------------------


def test_a_question_falls_back_to_the_nodes_title_not_result():
    @node("ask", title="Approve the invoice")
    def ask() -> str | AskPerson:
        return AskPerson()

    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="a", type="ask", version=1)]), NodeRegistry([ask]))
    (waiting,) = compiled.run_sync().waiting
    assert waiting.questions[0].title == "Approve the invoice"


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


@pytest.mark.parametrize(("value", "spelled"), [(math.nan, "NaN"), (math.inf, "Infinity"), (-math.inf, "-Infinity")])
def test_a_float_output_that_is_not_finite_survives_json_as_the_float(value, spelled):
    NOT_FINITE["value"] = value
    compiled = CompiledGraph.from_graph(Graph(nodes=[_n("n", "not-finite"), _n("p", "passes-on", value=From("n.result"))]), _FLOATS)

    state = compiled.run_sync()

    assert state.status == "complete"
    assert state.model_dump(mode="json")["nodes"]["n"]["outputs"]["result"]["values"] == [spelled]
    back = _through_json(state)
    assert _same_float(compiled.results(back)["n"]["result"], value)
    assert _same_float(compiled.results(compiled.run_sync(back.without_nodes("p")))["p"]["result"], value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_a_series_of_floats_keeps_one_that_is_not_finite_through_json(value):
    NOT_FINITE["value"] = value
    compiled = CompiledGraph.from_graph(Graph(nodes=[_n("n", "not-finite-many")]), _FLOATS)

    back = compiled.results(_through_json(compiled.run_sync()))["n"]["result"]["values"]

    assert back[0] == 1.5 and _same_float(back[1], value)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_an_output_per_row_keeps_a_float_that_is_not_finite_through_json(value):
    NOT_FINITE["value"] = value
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[_n("w", "words", text=Static("a b")), _n("n", "not-finite-per-row", text=From("w.result"))]), _FLOATS,
    )

    back = compiled.results(_through_json(compiled.run_sync()))["n"]["result"]["values"]

    assert back[0] == 1.5 and _same_float(back[1], value)


def test_a_not_finite_float_is_refused_on_an_int_output_so_none_reaches_a_float_input_that_way():
    NOT_FINITE["value"] = math.nan
    compiled = CompiledGraph.from_graph(Graph(nodes=[_n("i", "int-of-not-finite"), _n("h", "half", value=From("i.result"))]), _FLOATS)

    assert compiled.is_runnable, compiled.problems
    state = compiled.run_sync()

    assert state.status == "failed" and state.cause.type == "invalid_output"
