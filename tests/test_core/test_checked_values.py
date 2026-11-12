"""What a node returns is checked against what it declares, and so is a value an edge brings that its source's type doesn't settle.

A value that isn't what the node declares fails the node that returned it as
``invalid_output``, or the node it reached as ``invalid_edge_value``, on that item,
quoting nothing of the value. A value already of its declared class is taken
as it is; anything else is read strictly by the type, so a model may be
returned as a dict, but ``"12"`` is never an ``int`` nor ``12.0`` an ``int`` column.
"""

import asyncio
import datetime
import json
import math
from dataclasses import dataclass
from threading import Event
from typing import Annotated, Any, Literal, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    Param,
    Policy,
    Run,
    Series,
    Table,
    host_type,
    make_graph,
    node,
)
from pydantic import BaseModel, Discriminator, Field, field_validator
from pydantic_core import core_schema

from tests.test_core.nodes import registry_of

SECRET = "cpr-1234567890"


def _returning(returned: Any, return_type: Any) -> Any:
    """The run of a graph of one node, ``g``, returning ``returned`` from ``-> return_type``."""

    def give() -> return_type:  # type: ignore[valid-type]
        return returned

    made = node("give", title="Give", description="For a test.")(give)
    return Graph.compile(GraphRecord(nodes=[NodeRecord(id="g", node="give")]), registry_of(made)).run()


def _feeding(returned: Any, return_type: Any, sink: Any) -> Any:
    """The run of ``a`` returning ``returned`` into ``sink``'s one input, as node ``b``."""

    def give() -> return_type:  # type: ignore[valid-type]
        return returned

    made = node("give", title="Give", description="For a test.")(give)
    (name,) = sink.inputs
    return Graph.compile(GraphRecord(
        nodes=[NodeRecord(id="a", node="give"), NodeRecord(id="b", node=sink.id)], edges=[("a.result", f"b.{name}")],
    ), registry_of(made, sink)).run()


# -- what a node returns ----------------------------------------------------------


@pytest.mark.parametrize(("returned", "return_type"), [(42, str), ("abc", Series[str]), (None, str), (float("nan"), int), ("x", None)],
                         ids=["an-int-for-a-str", "a-str-for-a-series", "none-without-the-hint", "nan-for-an-int", "a-value-from-no-outputs"])
def test_a_return_that_is_not_the_declared_type_fails_the_node_that_returned_it(returned, return_type):
    """``-> str`` returning ``42`` fails at ``g``, never downstream; ``"abc"`` for a ``Series`` is never three rows."""
    run = _returning(returned, return_type)

    assert (run.status, run.state.cause.type, run.state.cause.loc) == ("failed", "invalid_output", ("g",))
    assert run.results["g"] == ({"result": None} if return_type is not None else {})


def test_returning_none_needs_the_hint():
    assert _returning(None, str | None).status == "complete"
    assert _returning(None, str).state.cause.type == "invalid_output"


@pytest.mark.parametrize(("returned", "return_type"), [([1], list[int] | None), ("a", Literal["a", "b"] | None)],
                         ids=["a-list", "a-literal"])
def test_a_blank_taking_output_of_a_type_that_is_not_a_class_takes_its_value(returned, return_type):
    """``-> list[int] | None`` returning ``[1]`` is read by its type as ``-> list[int]`` is, not held to the class ``NoneType``."""
    run = _returning(returned, return_type)

    assert run.status == "complete"
    assert run.results["g"]["result"] == returned


def test_a_value_read_as_its_type_is_kept_as_the_type():
    """A model returned as a dict, an int for a float, a str for a ``class Text(str)``: each lands as its declared type."""

    @host_type("Point", title="Point")
    class Point(BaseModel):
        x: int

    @host_type("Text", title="Text")
    class Text(str):
        pass

    assert _returning({"x": 1}, Point).results["g"]["result"] == Point(x=1)
    assert type(_returning(3, float).results["g"]["result"]) is float
    assert type(_returning("a", Text).results["g"]["result"]) is Text


class Shape(dict):  # type: ignore[type-arg]
    pass


@host_type("Square", title="Square")
class Square(Shape):
    """A type whose own check lets any ``Shape`` through as it is: in memory it accepts what it would never build."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        return core_schema.no_info_plain_validator_function(lambda value: value)


def test_a_type_whose_own_check_does_not_give_the_type_fails_the_node_naming_the_type():
    run = _returning(Shape(side=2), Square)

    assert (run.status, run.state.cause.type, run.state.cause.loc) == ("failed", "invalid_output", ("g",))
    assert "Square" in run.state.cause.ctx["reason"]


@dataclass(frozen=True)
class Pair:
    name: str
    count: int


def test_a_node_returning_something_other_than_its_dataclass_fails_naming_the_class():
    run = _returning(("a", 1), Pair)

    assert (run.state.cause.type, run.state.cause.loc) == ("invalid_output", ("g",))
    assert "Pair" in run.state.cause.ctx["reason"] and "tuple" in run.state.cause.ctx["reason"]


@dataclass(frozen=True)
class OnlyResult:
    result: str


def test_a_dataclass_whose_one_field_is_named_result_is_spread_like_any_dataclass_of_outputs():
    assert _returning(OnlyResult("hi"), OnlyResult).results["g"] == {"result": "hi"}

    run = _returning("hi", OnlyResult)

    assert (run.state.cause.type, run.state.cause.loc) == ("invalid_output", ("g",)) and "OnlyResult" in run.state.cause.ctx["reason"]


def test_a_unit_whose_second_output_is_invalid_writes_none_of_its_outputs():
    run = _returning(Pair("ok", "many"), Pair)  # type: ignore[arg-type]

    assert (run.state.cause.type, run.state.cause.ctx["name"]) == ("invalid_output", "count")
    assert run.results["g"] == {"name": None, "count": None}
    assert "g" not in json.loads(run.to_json())["outcomes"]


def _computed(returned: Any) -> Any:
    @node("computed", title="Computed", description="For a test.", compute_outputs=lambda: {"name": str, "age": int})
    def computed() -> Any:
        return returned

    return Graph.compile(GraphRecord(nodes=[NodeRecord(id="c", node="computed")]), registry_of(computed)).run()


def test_a_node_with_computed_outputs_returns_exactly_them_as_a_mapping():
    assert _computed({"name": "Ida", "age": 3}).results["c"] == {"name": "Ida", "age": 3}
    for returned, named in (({"name": "Ida"}, "age"), ({"name": "Ida", "age": 3, "more": 1}, "more")):
        run = _computed(returned)
        assert (run.state.cause.type, run.state.cause.ctx["name"]) == ("invalid_output", named)
    assert _computed(["Ida", 3]).state.cause.type == "invalid_output"


def test_a_series_with_a_skipped_item_is_checked_around_the_gap():
    run = _returning([1, SKIPPED, 2], Series[int])

    assert run.status == "complete" and run.results["g"]["result"] == [1, None, 2]
    assert _returning([1, SKIPPED, "2"], Series[int]).state.cause.type == "invalid_output"


class Line(TypedDict):
    item: str
    amount: int


def test_a_fraction_in_an_int_column_fails_the_node_rather_than_being_cut_off():
    run = _returning(pl.DataFrame({"item": ["a"], "amount": [12.75]}), Table[Line])

    assert (run.state.cause.type, run.state.cause.loc, run.state.cause.ctx["name"], run.state.cause.ctx["column"]) == ("invalid_output", ("g",), "result", "amount")


def test_a_float_column_for_an_int_one_fails_even_when_whole():
    """Outputs are read strictly: ``12.0`` is a float, so an ``int`` column refuses it, as ``-> int`` refuses ``"12"``."""
    assert _returning(pl.DataFrame({"item": ["a"], "amount": [12.0]}), Table[Line]).state.cause.type == "invalid_output"


def test_a_column_holds_a_blank_only_where_its_type_takes_one():
    """A type means what it says: a blank in an ``int`` column is refused where the table arrives, naming the column;
    a column that may be blank is declared ``int | None``."""

    class Loose(TypedDict):
        item: str
        amount: int | None

    with_a_blank = pl.DataFrame({"item": ["a", "b"], "amount": [12, None]})
    refused = _returning(with_a_blank, Table[Line]).state.cause
    assert (refused.type, refused.ctx["column"]) == ("invalid_output", "amount")
    kept = _returning(with_a_blank, Table[Loose]).results["g"]["result"]
    assert kept.to_dicts() == [{"item": "a", "amount": 12}, {"item": "b", "amount": None}]


def test_a_column_of_yes_or_no_is_never_an_int_column_returned_brought_or_given():
    """A yes or no never lands on a number, though Python counts ``True`` as 1: a column is held to it as a value is."""
    yes_no = pl.DataFrame({"item": ["a", "b"], "amount": [True, False]})
    assert (_returning(yes_no, Table[Line]).state.cause.type, _returning(yes_no, Table[Line]).state.cause.ctx["column"]) == ("invalid_output", "amount")

    @node("total", title="Total", description="For a test.")
    def total(lines: Table[Line]) -> int:
        return int(lines["amount"].sum())

    brought = _feeding(yes_no, Table, total)
    assert (brought.state.cause.type, brought.state.cause.loc, brought.state.cause.ctx["column"]) == ("invalid_edge_value", ("b",), "amount")
    with pytest.raises(InvalidCallError) as given:
        Graph.compile(GraphRecord(nodes=[NodeRecord(id="t", node="total")]), registry_of(total)).run(lines=yes_no)
    assert [p.type for p in given.value.errors()] == ["invalid_input"]


class Graded(TypedDict):
    level: Literal[1, 2]
    score: int | str


@pytest.mark.parametrize("column", ["level", "score"])
def test_a_column_refuses_a_yes_or_no_wherever_a_field_of_its_type_does(column):
    """A column of a ``Literal`` of numbers or of a union holding a number and no ``bool`` refuses ``True``, as a field
    of that type does: Python counts ``True`` as 1, and both the field and the column read one rule."""

    @node("count_rows", title="Count rows", description="For a test.")
    def count_rows(rows: Table[Graded]) -> int:
        return rows.height

    given = pl.DataFrame({"level": [1], "score": [3]}).with_columns(pl.lit(True).alias(column))
    with pytest.raises(InvalidCallError) as refused:
        Graph.compile(GraphRecord(nodes=[NodeRecord(id="c", node="count_rows")]), registry_of(count_rows)).run(rows=given)
    assert [(p.type, p.ctx.get("column")) for p in refused.value.errors()] == [("invalid_input", column)]


@pytest.mark.parametrize(("returned", "return_type"), [("no", bool), ("12", int), ("2026-01-02", datetime.date), (12.0, int)],
                         ids=["a-word-for-a-bool", "text-for-an-int", "text-for-a-date", "a-float-for-an-int"])
def test_an_output_is_read_strictly_never_from_text_that_spells_it(returned, return_type):
    assert _returning(returned, return_type).state.cause.type == "invalid_output"


def test_a_node_may_return_a_table_as_records_and_anything_else_fails_it():
    assert _returning([{"item": "a", "amount": 1}], Table[Line]).results["g"]["result"].to_dicts() == [{"item": "a", "amount": 1}]
    assert _returning("item,amount", Table[Line]).state.cause.type == "invalid_output"


# -- what an edge brings -------------------------------------------------------------------


@host_type("PickyId", title="Picky id", accepts=[object])
class PickyId(str):
    """A type whose own check quotes what it refuses, which is ordinary Python."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        def check(value: Any) -> Any:
            if isinstance(value, PickyId):
                return value
            raise ValueError(f"not a valid id: {value}")

        return core_schema.no_info_plain_validator_function(check)



def test_a_value_an_edge_brings_that_the_input_refuses_fails_the_node_it_reached_quoting_nothing():
    @node("sink", title="Sink", description="For a test.")
    def sink(identifier: PickyId) -> str:
        return identifier

    run = _feeding(SECRET, str, sink)

    assert (run.status, run.state.cause.type, run.state.cause.loc, run.state.cause.ctx["name"]) == ("failed", "invalid_edge_value", ("b",), "identifier")
    assert SECRET not in run.state.cause.model_dump_json() and run.state.cause.ctx["reason"]
    assert run.results["a"] == {"result": SECRET}


def test_a_value_on_an_edge_outside_its_bounds_fails_the_node_for_that_item():
    @node("numbers", title="Numbers", description="For a test.")
    def numbers() -> Series[int]:
        return [5, 1, 7]

    @node("at_least_five", title="At least five", description="For a test.")
    def at_least_five(n: Annotated[int, Param(ge=5)]) -> int:
        return n

    run = Graph.compile(GraphRecord(
        nodes=[NodeRecord(id="a", node="numbers"), NodeRecord(id="b", node="at_least_five")], edges=[("a.result", "b.n")],
    ), registry_of(numbers, at_least_five)).run()

    assert (run.state.cause.type, run.state.cause.loc) == ("invalid_edge_value", ("b", 1))
    assert [(e["type"], e["msg"]) for e in run.state.cause.ctx["errors"]] == [("greater_than_equal", "Input should be greater than or equal to 5")]


def test_an_int_arriving_on_a_float_input_arrives_a_float():
    @node("half", title="Half", description="For a test.")
    def half(x: float) -> str:
        return type(x).__name__

    assert _feeding(3, int, half).results["b"] == {"result": "float"}


class Contacts(TypedDict):
    name: str
    email: str


def test_a_table_input_given_a_table_without_a_column_it_declares_fails_naming_the_column():
    @node("needs_email", title="Needs email", description="For a test.")
    def needs_email(contacts: Table[Contacts]) -> int:
        return contacts.height

    run = _feeding(pl.DataFrame({"name": ["Ida"]}), Table, needs_email)

    assert (run.state.cause.type, run.state.cause.loc, run.state.cause.ctx["name"], run.state.cause.ctx["column"]) == ("invalid_edge_value", ("b",), "contacts", "email")


def test_a_callers_records_are_read_cell_by_cell_as_they_came_not_as_polars_made_them():
    @node("count_contacts", title="Count contacts", description="Counts the rows of a contacts table.")
    def count_contacts(contacts: Table[Contacts]) -> int:
        return contacts.height

    graph = Graph.compile(make_graph(count_contacts), registry_of(count_contacts))
    with pytest.raises(InvalidCallError) as refused:
        graph.run(contacts=[{"name": "Ida", "email": "i@x"}, {"name": 5, "email": "b@x"}])  # polars would make the 5 a text
    assert refused.value.errors()[0].ctx["column"] == "name"


def test_a_table_whose_column_is_of_a_type_its_input_wont_take_is_a_type_mismatch_when_the_graph_loads():
    """Two tables are compared by their columns where the graph loads: a text column is never an ``int`` column."""
    class Amounts(TypedDict):
        result: int

    class Texts(TypedDict):
        result: str

    @node("texts", title="Texts", description="For a test.")
    def texts() -> Table[Texts]:
        return pl.DataFrame({"result": ["1", "two"]})

    @node("total", title="Total", description="For a test.")
    def total(rows: Table[Amounts]) -> int:
        return int(rows["result"].sum())

    graph = Graph.compile(GraphRecord(
        nodes=[NodeRecord(id="a", node="texts"), NodeRecord(id="b", node="total")], edges=[("a.result", "b.rows")],
    ), registry_of(texts, total))

    [problem] = graph.problems
    assert (problem.type, problem.loc, problem.ctx["got"], problem.ctx["expected"]) == ("type_mismatch", ("b", "rows"), "Table[result: str]", "Table[result: int]")


def test_a_column_of_a_table_of_unknown_columns_its_input_wont_take_fails_that_node_as_what_the_edge_brought():
    class Amounts(TypedDict):
        result: int

    @node("total", title="Total", description="For a test.")
    def total(rows: Table[Amounts]) -> int:
        return int(rows["result"].sum())

    run = _feeding(pl.DataFrame({"result": ["1", "two"]}), Table, total)

    assert (run.state.cause.type, run.state.cause.loc, run.state.cause.ctx["column"]) == ("invalid_edge_value", ("b",), "result")


# -- a failure says why, and nothing of the value ------------------------------------------


class Cat(BaseModel):
    kind: Literal["cat"]


class Dog(BaseModel):
    kind: Literal["dog"]


@host_type("Pet", title="Pet")
class Pet(BaseModel):
    pet: Annotated[Cat | Dog, Discriminator("kind")]


@host_type("Asserting", title="Asserting")
class Asserting(BaseModel):
    code: str

    @field_validator("code")
    @classmethod
    def refuse(cls, value: str) -> str:
        assert value == "ok", f"bad code {value}"
        return value


@host_type("ByPerson", title="By person")
class ByPerson(BaseModel):
    counts: dict[str, int]


@host_type("Bounded", title="Bounded")
class Bounded(BaseModel):
    number: int = Field(ge=5)


@pytest.mark.parametrize(("returned", "return_type"), [
    (SECRET, PickyId),
    ({"pet": {"kind": SECRET}}, Pet),
    ({"code": SECRET}, Asserting),
    ({"counts": {SECRET: "many"}}, ByPerson),
], ids=["own-validator", "a-tag-pydantic-quotes", "an-assert", "a-key-of-a-dict"])
def test_what_a_node_returned_is_not_in_the_failure_of_its_output(returned, return_type):
    run = _returning(returned, return_type)

    assert run.state.cause.type == "invalid_output"
    assert SECRET not in run.state.cause.model_dump_json() and SECRET not in run.to_json()


def test_the_failure_of_an_output_says_which_check_refused_and_the_limit_it_broke():
    cause = _returning({"number": 1}, Bounded).state.cause

    assert [(error["type"], error["msg"]) for error in cause.ctx["errors"]] == [("greater_than_equal", "Input should be greater than or equal to 5")]


def test_a_record_field_is_named_in_the_failure_of_an_output():
    cause = _returning({"counts": {SECRET: "many"}}, ByPerson).state.cause

    assert [tuple(error["loc"]) for error in cause.ctx["errors"]] == [("counts",)]


def test_a_key_of_a_dict_given_is_never_in_the_errors_and_a_declared_field_or_an_index_is():
    @node("tally", title="Tally", description="For a test.")
    def tally(scores: dict[str, int], people: list[ByPerson]) -> str:
        return "ok"

    graph = Graph.compile(GraphRecord(nodes=[NodeRecord(id="t", node="tally")]), registry_of(tally))

    for given, loc in (({"scores": {SECRET: "x"}, "people": []}, ()), ({"scores": {}, "people": [{"counts": {}}, {"counts": {SECRET: "x"}}]}, (1, "counts"))):
        with pytest.raises(InvalidCallError) as refused:
            graph.run(given)
        (problem,) = refused.value.errors()
        assert SECRET not in problem.model_dump_json() and [tuple(e["loc"]) for e in problem.ctx["errors"]] == [loc]


# -- where the check runs ----------------------------------------------------------------------


@host_type("_SlowOutput", title="Slow output")
class _SlowOutput(str):
    """An output type whose check waits until the other node's completion has been streamed."""

    check_began = Event()
    other_reported = Event()

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        def check(value: Any) -> Any:
            cls.check_began.set()
            assert cls.other_reported.wait(timeout=10)
            return cls(value)

        return core_schema.no_info_plain_validator_function(check)


def test_a_slow_output_check_runs_on_its_nodes_thread_and_holds_up_no_other_unit():
    """The other node finishes only after the check began, and the check passes only once the stream reported it."""

    @node("slow", title="Slow", description="For a test.", policy=Policy(concurrency=2))
    def slow() -> _SlowOutput:
        return "x"  # type: ignore[return-value]  # not yet the type, so its check runs

    @node("other", title="Other", description="For a test.", policy=Policy(concurrency=2))
    def other() -> int:
        assert _SlowOutput.check_began.wait(timeout=10)
        return 1

    graph = Graph.compile(GraphRecord(nodes=[NodeRecord(id="s", node="slow"), NodeRecord(id="o", node="other")]), registry_of(slow, other))

    async def go() -> str:
        async for event in graph.with_inputs().stream():
            if event.type == "node_complete" and event.loc[0] == "o":
                _SlowOutput.other_reported.set()
            if event.type == "run_end":
                return event.status
        raise AssertionError("no run_end")

    assert asyncio.run(go()) == "complete"


def test_a_value_already_of_its_type_is_kept_as_returned():
    """The fast path: an exact ``float`` that isn't finite stays the float it was."""
    assert math.isinf(_returning(math.inf, float).results["g"]["result"])


@pytest.mark.parametrize("return_type", [Any, int | Any, Any | None])
@pytest.mark.parametrize("returned", [1.5, "s", {"k": [1]}, None, 3])
def test_a_node_typed_any_returns_anything_and_a_node_taking_any_receives_it(return_type: Any, returned: Any) -> None:
    @node("take_any", title="Take any", description="For a test.")
    def take_any(value: Any) -> Any:
        return value

    run = _feeding(returned, return_type, take_any)
    assert run.status == "complete"
    assert run.results["b"]["result"] == returned


def test_a_yes_or_no_returned_for_a_literal_of_numbers_is_refused_as_for_a_number():
    for return_type in (int, float, Literal[1, 2]):
        run = _returning(True, return_type)
        assert (run.status, run.state.cause.type) == ("failed", "invalid_output"), return_type


@node("anything-in", title="Anything in", description="For a test.")
def _anything_in(x: Any) -> Any:
    return x


@node("all-of", title="All of", description="For a test.")
def _all_of(values: Series[Any]) -> int:
    return len(values)


def _giving(returned: Any, return_type: Any) -> Any:
    """A node ``give`` returning ``returned`` from ``-> return_type``."""

    def give() -> return_type:  # type: ignore[valid-type]
        return returned

    return node("give", title="Give", description="For a test.")(give)


def _fails_with(graph: Any) -> tuple[Any, ...]:
    """How a run of ``graph`` ended: its status, and its cause's code, place, input and field."""
    run = graph.run()
    cause = run.state.cause
    return run.status, cause.type, cause.loc, cause.ctx["name"], cause.ctx["record_field"]


@pytest.mark.parametrize(("declared", "returned"), [(dict, {"other": 1}), (Any, 5), (Any, {"other": 1})])
def test_a_field_named_on_an_edge_that_the_value_at_run_time_lacks_fails_the_reader_naming_the_field(declared: Any, returned: Any):
    """A record whose type declares no fields is wired by field on trust, so a missing one is the reader's ``invalid_edge_value``."""
    give = _giving(returned, declared)
    graph = Graph.compile(make_graph(give, _anything_in, edges=[("give.result.name", "anything-in.x")]), registry_of(give, _anything_in))
    assert graph.is_runnable
    assert _fails_with(graph) == ("failed", "invalid_edge_value", ("anything-in",), "x", "name")
    assert SECRET not in graph.run().state.cause.msg and "no field 'name'" in graph.run().state.cause.msg


def test_a_field_an_item_or_a_column_of_unknown_columns_lacks_fails_the_reader_at_that_item_and_a_gather_whole():
    items = _giving([{"name": "a"}, {"other": "b"}], Series[dict])
    per_item = Graph.compile(make_graph(items, _anything_in, edges=[("give.result.name", "anything-in.x")]), registry_of(items, _anything_in))
    assert _fails_with(per_item) == ("failed", "invalid_edge_value", ("anything-in", 1), "x", "name")
    gathered = Graph.compile(make_graph(items, _all_of, edges=[("give.result.name", "all-of.values")]), registry_of(items, _all_of))
    assert _fails_with(gathered) == ("failed", "invalid_edge_value", ("all-of",), "values", "name")
    by_column = _giving(pl.DataFrame({"other": ["a", "b"]}), Series[Any])
    column_read = Graph.compile(make_graph(by_column, _anything_in, edges=[("give.result.name", "anything-in.x")]), registry_of(by_column, _anything_in))
    assert _fails_with(column_read) == ("failed", "invalid_edge_value", ("anything-in", 0), "x", "name")
    column_gathered = Graph.compile(make_graph(by_column, _all_of, edges=[("give.result.name", "all-of.values")]), registry_of(by_column, _all_of))
    assert _fails_with(column_gathered) == ("failed", "invalid_edge_value", ("all-of",), "values", "name")


@node("words", title="Words", description="For a test.")
def _words(text: str) -> Series[str]:
    return text.split()


@node("with-it", title="With it", description="For a test.")
def _with_it(word: str, x: Any) -> str:
    return word


@node("with-them", title="With them", description="For a test.")
def _with_them(word: str, values: Series[Any]) -> str:
    return word


@pytest.mark.parametrize(
    ("returned", "declared", "reader", "input_name"),
    [({"other": 1}, dict, _with_it, "x"), ([{"name": "a"}, {"other": "b"}], Series[dict], _with_them, "values"), (pl.DataFrame({"other": ["a"]}), Series[Any], _with_them, "values")],
)
def test_a_run_that_failed_on_a_field_a_record_lacks_shows_its_results_with_a_blank_for_each_item_never_called(
    returned: Any, declared: Any, reader: Any, input_name: str,
):
    """The failure is the run's cause; a host that reads the run afterwards is shown a blank where the field was read."""
    give = _giving(returned, declared)
    edges = [("words.result", f"{reader.id}.word"), ("give.result.name", f"{reader.id}.{input_name}"), (f"{reader.id}.result", "all-of.values")]
    graph = Graph.compile(make_graph(_words, give, reader, _all_of, edges=edges), registry_of(_words, give, reader, _all_of))
    run = graph.run({"words.text": "a b"})
    assert (run.status, run.state.cause.type, run.state.cause.ctx["record_field"]) == ("failed", "invalid_edge_value", "name")
    for read in (run, Run.from_json(run.to_json(), graph)):
        assert read.results["words"] == {"result": ["a", "b"]}
        assert read.results[reader.id] == {"result": [None, None]} and read.results["all-of"] == {"result": None}
