"""What a node returns is checked against what it declares, and so is a value an edge brings that its source's type doesn't settle.

A value that isn't what the node declares fails the node that returned it as
``invalid_output``, or the node it reached as ``invalid_input``, on that row,
quoting nothing of the value. A value already of its declared class is taken
as it is; anything else is read strictly by the type, so a model may be
returned as a dict, but ``"12"`` is never an ``int`` nor ``12.0`` an ``int`` column.
"""

import asyncio
import datetime
import json
import math
import threading
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

import polars as pl
import pytest
from conductor import (
    SKIPPED,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    Param,
    Policy,
    Series,
    Table,
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

    made = node("give")(give)
    return registry_of(made).load(GraphRecord(nodes=[NodeRecord(id="g", node="give")])).run()


def _feeding(returned: Any, return_type: Any, sink: Any) -> Any:
    """The run of ``a`` returning ``returned`` into ``sink``'s one input, as node ``b``."""

    def give() -> return_type:  # type: ignore[valid-type]
        return returned

    made = node("give")(give)
    (name,) = sink.inputs
    return registry_of(made, sink).load(GraphRecord(
        nodes=[NodeRecord(id="a", node="give"), NodeRecord(id="b", node=sink.id)], edges=[("a.result", f"b.{name}")],
    )).run()


# -- what a node returns ----------------------------------------------------------


@pytest.mark.parametrize(("returned", "return_type"), [(42, str), ("abc", Series[str]), (None, str), (float("nan"), int), ("x", None)],
                         ids=["an-int-for-a-str", "a-str-for-a-series", "none-without-the-hint", "nan-for-an-int", "a-value-from-no-outputs"])
def test_a_return_that_is_not_the_declared_type_fails_the_node_that_returned_it(returned, return_type):
    """``-> str`` returning ``42`` fails at ``g``, never downstream; ``"abc"`` for a ``Series`` is never three rows."""
    run = _returning(returned, return_type)

    assert (run.status, run.cause.type, run.cause.loc) == ("failed", "invalid_output", ("g",))
    assert run.results["g"] == ({"result": None} if return_type is not None else {})


def test_returning_none_needs_the_hint():
    assert _returning(None, str | None).status == "complete"
    assert _returning(None, str).cause.type == "invalid_output"


def test_a_value_read_as_its_type_is_kept_as_the_type():
    """A model returned as a dict, an int for a float, a str for a ``class Text(str)``: each lands as its declared type."""

    class Point(BaseModel):
        x: int

    class Text(str):
        pass

    assert _returning({"x": 1}, Point).results["g"]["result"] == Point(x=1)
    assert type(_returning(3, float).results["g"]["result"]) is float
    assert type(_returning("a", Text).results["g"]["result"]) is Text


class Shape(dict):  # type: ignore[type-arg]
    pass


class Square(Shape):
    """A type whose own check lets any ``Shape`` through as it is: in memory it accepts what it would never build."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        return core_schema.no_info_plain_validator_function(lambda value: value)


def test_a_type_whose_own_check_does_not_give_the_type_fails_the_node_naming_the_type():
    run = _returning(Shape(side=2), Square)

    assert (run.status, run.cause.type, run.cause.loc) == ("failed", "invalid_output", ("g",))
    assert "Square" in run.cause.ctx["reason"]


@dataclass(frozen=True)
class Pair:
    name: str
    count: int


def test_a_node_returning_something_other_than_its_dataclass_fails_naming_the_class():
    run = _returning(("a", 1), Pair)

    assert (run.cause.type, run.cause.loc) == ("invalid_output", ("g",))
    assert "Pair" in run.cause.ctx["reason"] and "tuple" in run.cause.ctx["reason"]


@dataclass(frozen=True)
class OnlyResult:
    result: str


def test_a_dataclass_whose_one_field_is_named_result_is_spread_like_any_dataclass_of_outputs():
    assert _returning(OnlyResult("hi"), OnlyResult).results["g"] == {"result": "hi"}

    run = _returning("hi", OnlyResult)

    assert (run.cause.type, run.cause.loc) == ("invalid_output", ("g",)) and "OnlyResult" in run.cause.ctx["reason"]


def test_a_unit_whose_second_output_is_invalid_writes_none_of_its_outputs():
    run = _returning(Pair("ok", "many"), Pair)  # type: ignore[arg-type]

    assert (run.cause.type, run.cause.ctx["name"]) == ("invalid_output", "count")
    assert run.results["g"] == {"name": None, "count": None}
    assert json.loads(run.to_json())["units"]["g"]["states"] == []


def _computed(returned: Any) -> Any:
    @node("computed", compute_outputs=lambda: {"name": str, "age": int})
    def computed() -> Any:
        return returned

    return registry_of(computed).load(GraphRecord(nodes=[NodeRecord(id="c", node="computed")])).run()


def test_a_node_with_computed_outputs_returns_exactly_them_as_a_mapping():
    assert _computed({"name": "Ida", "age": 3}).results["c"] == {"name": "Ida", "age": 3}
    for returned, named in (({"name": "Ida"}, "age"), ({"name": "Ida", "age": 3, "more": 1}, "more")):
        run = _computed(returned)
        assert (run.cause.type, run.cause.ctx["name"]) == ("invalid_output", named)
    assert _computed(["Ida", 3]).cause.type == "invalid_output"


def test_a_series_with_a_skipped_item_is_checked_around_the_gap():
    run = _returning([1, SKIPPED, 2], Series[int])

    assert run.status == "complete" and run.results["g"]["result"] == [1, None, 2]
    assert _returning([1, SKIPPED, "2"], Series[int]).cause.type == "invalid_output"


class Line(TypedDict):
    item: str
    amount: int


def test_a_fraction_in_an_int_column_fails_the_node_rather_than_being_cut_off():
    run = _returning(pl.DataFrame({"item": ["a"], "amount": [12.75]}), Table[Line])

    assert (run.cause.type, run.cause.loc, run.cause.ctx["name"], run.cause.ctx["column"]) == ("invalid_output", ("g",), "result", "amount")


def test_a_float_column_for_an_int_one_fails_even_when_whole_and_an_int_column_with_blanks_is_kept():
    """Outputs are read strictly: ``12.0`` is a float, so an ``int`` column refuses it, as ``-> int`` refuses ``"12"``."""
    assert _returning(pl.DataFrame({"item": ["a"], "amount": [12.0]}), Table[Line]).cause.type == "invalid_output"
    table = _returning(pl.DataFrame({"item": ["a", "b"], "amount": [12, None]}), Table[Line]).results["g"]["result"]
    assert table.to_dicts() == [{"item": "a", "amount": 12}, {"item": "b", "amount": None}]


@pytest.mark.parametrize(("returned", "return_type"), [("no", bool), ("12", int), ("2026-01-02", datetime.date), (12.0, int)],
                         ids=["a-word-for-a-bool", "text-for-an-int", "text-for-a-date", "a-float-for-an-int"])
def test_an_output_is_read_strictly_never_from_text_that_spells_it(returned, return_type):
    assert _returning(returned, return_type).cause.type == "invalid_output"


def test_a_table_that_is_not_a_frame_fails_its_node():
    assert _returning([{"item": "a", "amount": 1}], Table[Line]).cause.type == "invalid_output"


# -- what an edge brings -------------------------------------------------------------------


class PickyId(str):
    """A type whose own check quotes what it refuses, which is ordinary Python."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        def check(value: Any) -> Any:
            if isinstance(value, PickyId):
                return value
            raise ValueError(f"not a valid id: {value}")

        return core_schema.no_info_plain_validator_function(check)

    @classmethod
    def accepts(cls, source: Any) -> bool:
        return True


def test_a_value_an_edge_brings_that_the_input_refuses_fails_the_node_it_reached_quoting_nothing():
    @node("sink")
    def sink(identifier: PickyId) -> str:
        return identifier

    run = _feeding(SECRET, str, sink)

    assert (run.status, run.cause.type, run.cause.loc, run.cause.ctx["name"]) == ("failed", "invalid_input", ("b",), "identifier")
    assert SECRET not in run.cause.model_dump_json() and run.cause.ctx["reason"]
    assert run.results["a"] == {"result": SECRET}


def test_a_value_on_an_edge_outside_its_bounds_fails_the_node_for_that_row():
    @node("numbers")
    def numbers() -> Series[int]:
        return [5, 1, 7]

    @node("at_least_five")
    def at_least_five(n: Annotated[int, Param(ge=5)]) -> int:
        return n

    run = registry_of(numbers, at_least_five).load(GraphRecord(
        nodes=[NodeRecord(id="a", node="numbers"), NodeRecord(id="b", node="at_least_five")], edges=[("a.result", "b.n")],
    )).run()

    assert (run.cause.type, run.cause.loc) == ("invalid_input", ("b", 1))
    assert [(e["type"], e["msg"]) for e in run.cause.ctx["errors"]] == [("greater_than_equal", "Input should be greater than or equal to 5")]


def test_an_int_arriving_on_a_float_input_arrives_a_float():
    @node("half")
    def half(x: float) -> str:
        return type(x).__name__

    assert _feeding(3, int, half).results["b"] == {"result": "float"}


class Contacts(TypedDict):
    name: str
    email: str


def test_a_table_input_given_a_table_without_a_column_it_declares_fails_naming_the_column():
    @node("needs_email")
    def needs_email(contacts: Table[Contacts]) -> int:
        return contacts.height

    run = _feeding(pl.DataFrame({"name": ["Ida"]}), Table, needs_email)

    assert (run.cause.type, run.cause.loc, run.cause.ctx["name"], run.cause.ctx["column"]) == ("invalid_input", ("b",), "contacts", "email")


def test_a_combined_table_whose_column_wont_read_as_the_input_fails_that_node():
    class Amounts(TypedDict):
        result: int

    @node("texts")
    def texts() -> Series[str]:
        return ["1", "two"]

    @node("total")
    def total(rows: Table[Amounts]) -> int:
        return int(rows["result"].sum())

    graph = registry_of(texts, total).load(GraphRecord(
        nodes=[NodeRecord(id="a", node="texts"), NodeRecord(id="b", node="total")], edges=[("a.result", "b.rows")],
    ))
    assert graph.problems == []  # a str column may hold numbers written as text, so loading lets it through

    run = graph.run()

    assert (run.cause.type, run.cause.loc, run.cause.ctx["column"]) == ("invalid_input", ("b",), "result")


# -- a failure says why, and nothing of the value ------------------------------------------


class Cat(BaseModel):
    kind: Literal["cat"]


class Dog(BaseModel):
    kind: Literal["dog"]


class Pet(BaseModel):
    pet: Annotated[Cat | Dog, Discriminator("kind")]


class Asserting(BaseModel):
    code: str

    @field_validator("code")
    @classmethod
    def refuse(cls, value: str) -> str:
        assert value == "ok", f"bad code {value}"
        return value


class ByPerson(BaseModel):
    counts: dict[str, int]


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

    assert run.cause.type == "invalid_output"
    assert SECRET not in run.cause.model_dump_json() and SECRET not in run.to_json()


def test_the_failure_of_an_output_says_which_check_refused_and_the_limit_it_broke():
    cause = _returning({"number": 1}, Bounded).cause

    assert [(error["type"], error["msg"]) for error in cause.ctx["errors"]] == [("greater_than_equal", "Input should be greater than or equal to 5")]


def test_a_record_field_is_named_in_the_failure_of_an_output():
    cause = _returning({"counts": {SECRET: "many"}}, ByPerson).cause

    assert [tuple(error["loc"]) for error in cause.ctx["errors"]] == [("counts",)]


def test_a_key_of_a_dict_given_is_never_in_the_errors_and_a_declared_field_or_an_index_is():
    @node("tally")
    def tally(scores: dict[str, int], people: list[ByPerson]) -> str:
        return "ok"

    graph = registry_of(tally).load(GraphRecord(nodes=[NodeRecord(id="t", node="tally")]))

    for given, loc in (({"scores": {SECRET: "x"}, "people": []}, ()), ({"scores": {}, "people": [{"counts": {}}, {"counts": {SECRET: "x"}}]}, (1, "counts"))):
        with pytest.raises(InvalidCallError) as refused:
            graph.run(given)
        (problem,) = refused.value.errors()
        assert SECRET not in problem.model_dump_json() and [tuple(e["loc"]) for e in problem.ctx["errors"]] == [loc]


# -- where the check runs ----------------------------------------------------------------------


class _SlowOutput(str):
    """An output type whose check waits until the other node's completion has been streamed."""

    check_began = threading.Event()
    other_reported = threading.Event()

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: Any) -> Any:
        def check(value: Any) -> Any:
            cls.check_began.set()
            assert cls.other_reported.wait(timeout=10)
            return cls(value)

        return core_schema.no_info_plain_validator_function(check)


def test_a_slow_output_check_runs_on_its_nodes_thread_and_holds_up_no_other_unit():
    """The other node finishes only after the check began, and the check passes only once the stream reported it."""

    @node("slow", policy=Policy(concurrency=2))
    def slow() -> _SlowOutput:
        return "x"  # type: ignore[return-value]  # not yet the type, so its check runs

    @node("other", policy=Policy(concurrency=2))
    def other() -> int:
        assert _SlowOutput.check_began.wait(timeout=10)
        return 1

    graph = registry_of(slow, other).load(GraphRecord(nodes=[NodeRecord(id="s", node="slow"), NodeRecord(id="o", node="other")]))

    async def go() -> str:
        async for event in graph.with_inputs().stream():
            if event.type == "node_complete" and event.node_id == "o":
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
    @node("take_any")
    def take_any(value: Any) -> Any:
        return value

    run = _feeding(returned, return_type, take_any)
    assert run.status == "complete"
    assert run.results["b"]["result"] == returned


def test_a_yes_or_no_returned_for_a_literal_of_numbers_is_refused_as_for_a_number():
    for return_type in (int, float, Literal[1, 2]):
        run = _returning(True, return_type)
        assert (run.status, run.cause.type) == ("failed", "invalid_output"), return_type
