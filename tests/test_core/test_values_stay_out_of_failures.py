"""A value a run handles never reaches a failure: a host streams failures to a browser, and the value may be somebody's data."""

import json
from typing import Annotated, Literal

import pytest
from conductor import (
    CompiledGraph,
    DType,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    RunState,
    node,
)
from helpers import registry_of
from pydantic import BaseModel, Discriminator, Field, field_validator
from pydantic_core import core_schema

SECRET = "cpr-1234567890"


class PickyId(DType, str):
    """A type whose validator quotes what it refuses, which is ordinary Python."""

    id = "values-test-picky"

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        def check(value):
            if isinstance(value, PickyId):
                return value
            raise ValueError(f"not a valid id: {value}")

        return core_schema.no_info_plain_validator_function(check)

    @classmethod
    def accepts(cls, source):
        return True


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
    def refuse(cls, value):
        assert value == "ok", f"bad code {value}"
        return value


class ByPerson(BaseModel):
    counts: dict[str, int]


class Bounded(BaseModel):
    number: int = Field(ge=5)


def _failed_run(returned, return_type):
    def give() -> return_type:
        return returned

    made = node("give")(give)
    compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="g", node=made.id),
    ]), registry_of(made))
    state = compiled.run_sync()
    assert state.status == "failed"
    return state.cause


@pytest.mark.parametrize(("returned", "return_type"), [
    (SECRET, PickyId),
    ({"pet": {"kind": SECRET}}, Pet),
    ({"code": SECRET}, Asserting),
    ({"counts": {SECRET: "many"}}, ByPerson),
], ids=["own-validator", "a-tag-pydantic-quotes", "an-assert", "a-key-of-a-dict"])
def test_what_a_node_returned_is_not_in_the_failure_of_its_output(returned, return_type):
    failure = _failed_run(returned, return_type)
    assert failure.type == "invalid_output"
    assert SECRET not in failure.model_dump_json()


def test_the_failure_of_an_output_still_says_which_check_refused_and_a_limit_it_broke():
    failure = _failed_run({"number": 1}, Bounded)
    assert [(error["type"], error["msg"]) for error in failure.ctx["errors"]] == [("greater_than_equal", "Input should be greater than or equal to 5")]


def test_a_record_field_is_named_in_the_failure_of_an_output():
    failure = _failed_run({"counts": {SECRET: "many"}}, ByPerson)
    assert [tuple(error["loc"]) for error in failure.ctx["errors"]] == [("counts",)]


def test_what_a_node_received_is_not_in_the_failure_of_its_input():
    def source() -> str:
        return SECRET

    def sink(identifier: PickyId) -> str:
        return identifier

    made_source, made_sink = node("source")(source), node("sink")(sink)
    compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="a", node=made_source.id),
        NodeRecord(id="b", node=made_sink.id),
    ], edges=[("a.result", "b.identifier")]), registry_of(made_source, made_sink))
    state = compiled.run_sync()
    assert (state.status, state.cause.type) == ("failed", "invalid_input")
    assert SECRET not in state.cause.model_dump_json()
    assert "Identifier" in state.cause.ctx["reason"]


def test_a_stored_value_is_not_in_the_refusal_to_read_a_state_back():
    def give() -> PickyId:
        return PickyId("fine")

    made = node("give")(give)
    compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="g", node=made.id),
    ]), registry_of(made))
    stored = json.loads(compiled.run_sync().model_dump_json())
    stored["nodes"]["g"]["outputs"]["result"]["values"] = [SECRET]
    with pytest.raises(InvalidCallError) as refused:
        compiled.run_sync(RunState.model_validate(stored))
    assert [problem.type for problem in refused.value.errors()] == ["state_unreadable"]
    assert SECRET not in json.dumps([problem.model_dump(mode="json") for problem in refused.value.errors()])
