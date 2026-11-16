"""Tests that no problem ever quotes the value it was made for.

A host may send problems straight to a browser, and a value can be private. So a value that is refused never shows
up in a problem's ``msg``, its ``ctx`` or its text. That holds for a graph input, a constant, a value on an edge, a
value a node returns, a person's answer and a value in a saved run.

Pydantic quotes a value in several ways: in its own ``input``, in a validator's message, in an ``assert``, in a
union's tag, in a key of a dict and in the message of a host's own pydantic error, even one that gives the type of
one of pydantic's own errors, and in a ``column`` in its ``ctx``. Each case below makes a value that pydantic would quote in one of those ways. It
feeds that value into each place a value enters and can be refused.
"""

from dataclasses import dataclass
from typing import Annotated, Any, Literal

import pytest
from conductor import (
    Ask,
    Graph,
    GraphRecord,
    InvalidCallError,
    Problem,
    Run,
    host_type,
    make_graph,
    node,
)
from pydantic import BaseModel, Discriminator, field_validator
from pydantic_core import PydanticCustomError, to_json

from tests.test_core.nodes import registry_of

SECRET = "cpr-0102031234"


class _Cat(BaseModel):
    kind: Literal["cat"]


class _Dog(BaseModel):
    kind: Literal["dog"]


@host_type("QuotedPet", title="Pet")
class _Pet(BaseModel):
    pet: Annotated[_Cat | _Dog, Discriminator("kind")]


@host_type("QuotedCode", title="Code")
class _Code(BaseModel):
    code: str

    @field_validator("code")
    @classmethod
    def refuse(cls, value: str) -> str:
        if value != "ok":
            raise ValueError(f"no code {value}")
        return value


@host_type("QuotedAsserted", title="Asserted")
class _Asserted(BaseModel):
    code: str

    @field_validator("code")
    @classmethod
    def refuse(cls, value: str) -> str:
        assert value == "ok", f"bad code {value}"
        return value


@host_type("QuotedByContext", title="Quoted by context")
class _QuotedByContext(BaseModel):
    code: str

    @field_validator("code")
    @classmethod
    def refuse(cls, value: str) -> str:
        if value != "ok":
            raise PydanticCustomError("not_allowed", "{value} is not allowed", {"value": value})
        return value


@host_type("QuotedByText", title="Quoted by text")
class _QuotedByText(BaseModel):
    code: str

    @field_validator("code")
    @classmethod
    def refuse(cls, value: str) -> str:
        if value != "ok":
            raise PydanticCustomError("not_allowed", f"{value} is not allowed")
        return value


@host_type("QuotedUnderPydanticsType", title="Quoted under pydantic's type")
class _QuotedUnderPydanticsType(BaseModel):
    code: str

    @field_validator("code")
    @classmethod
    def refuse(cls, value: str) -> str:
        if value != "ok":
            raise PydanticCustomError("value_error", f"{value} is not allowed")
        return value


@host_type("QuotedUnderALimitsType", title="Quoted under a limit's type")
class _QuotedUnderALimitsType(BaseModel):
    code: str

    @field_validator("code")
    @classmethod
    def refuse(cls, value: str) -> str:
        if value != "ok":
            raise PydanticCustomError("string_pattern_mismatch", f"{value} doesn't match")
        return value


@host_type("QuotedAsAColumn", title="Quoted as a column")
class _QuotedAsAColumn(BaseModel):
    code: str

    @field_validator("code")
    @classmethod
    def refuse(cls, value: str) -> str:
        if value != "ok":
            raise PydanticCustomError("bad_code", "bad code", {"column": value})
        return value


@host_type("QuotedCounts", title="Counts")
class _Counts(BaseModel):
    counts: dict[str, int]


#: Each type, with a value it accepts and a value it refuses. The refused value carries ``SECRET`` in one of the ways
#: pydantic quotes a value.
TYPES = {
    "pydantic-input": (int, 1, SECRET),
    "a-validator-message": (_Code, {"code": "ok"}, {"code": SECRET}),
    "an-assert": (_Asserted, {"code": "ok"}, {"code": SECRET}),
    "a-union-tag": (_Pet, {"pet": {"kind": "cat"}}, {"pet": {"kind": SECRET}}),
    "a-dict-key": (_Counts, {"counts": {"a": 1}}, {"counts": {SECRET: "many"}}),
    "a-custom-error-context": (_QuotedByContext, {"code": "ok"}, {"code": SECRET}),
    "a-custom-error-text": (_QuotedByText, {"code": "ok"}, {"code": SECRET}),
    "a-custom-error-under-pydantics-type": (
        _QuotedUnderPydanticsType,
        {"code": "ok"},
        {"code": SECRET},
    ),
    "a-custom-error-under-a-limits-type": (
        _QuotedUnderALimitsType,
        {"code": "ok"},
        {"code": SECRET},
    ),
    "a-custom-errors-column": (_QuotedAsAColumn, {"code": "ok"}, {"code": SECRET}),
}


def _refused(call: Any) -> list[Problem]:
    with pytest.raises(InvalidCallError) as raised:
        call()
    assert SECRET not in str(raised.value)
    return raised.value.problems


def _graph_input(hint: Any, good: Any, bad: Any) -> list[Problem]:
    @node("take", title="Take", description="For a test.")
    def take(x: hint) -> str:  # type: ignore[valid-type]
        return "ok"

    return _refused(lambda: Graph.compile(make_graph(take), registry_of(take)).run({"x": bad}))


def _constant(hint: Any, good: Any, bad: Any) -> list[Problem]:
    @node("take", title="Take", description="For a test.")
    def take(x: hint) -> str:  # type: ignore[valid-type]
        return "ok"

    return Graph.compile(make_graph(take, constants={"take.x": bad}), registry_of(take)).problems


def _edge(hint: Any, good: Any, bad: Any) -> list[Problem]:
    @node("give", title="Give", description="For a test.")
    def give() -> Any:
        return bad

    @node("take", title="Take", description="For a test.")
    def take(x: hint) -> str:  # type: ignore[valid-type]
        return "ok"

    record = GraphRecord.model_validate(
        {
            "nodes": {
                "a": {"node": "give"},
                "b": {
                    "node": "take",
                    "inputs": {"x": {"sources": [{"node": "a", "output": "result"}]}},
                },
            },
            "outputs": {"result": {"node": "b", "output": "result"}},
        }
    )
    return [Graph.compile(record, registry_of(give, take)).run().cause]


def _giving(definition: Any) -> list[Problem]:
    return [
        Graph.compile(
            GraphRecord.model_validate({"nodes": {"g": {"node": definition.id}}}),
            registry_of(definition),
        )
        .run()
        .cause
    ]


def _one_output(hint: Any, good: Any, bad: Any) -> list[Problem]:
    def give() -> hint:  # type: ignore[valid-type]
        return bad

    return _giving(node("give", title="Give", description="For a test.")(give))


def _several_outputs(hint: Any, good: Any, bad: Any) -> list[Problem]:
    @dataclass(frozen=True)
    class Both:
        first: hint  # type: ignore[valid-type]
        second: int

    def give() -> Both:
        return Both(bad, 1)

    return _giving(node("give", title="Give", description="For a test.")(give))


def _answer(hint: Any, good: Any, bad: Any) -> list[Problem]:
    def ask(text: str) -> hint:  # type: ignore[valid-type]
        raise Ask("Which?")

    run = Graph.compile(
        make_graph(
            definition := node("ask", title="Ask", description="For a test.", asks=True)(ask)
        ),
        registry_of(definition),
    ).run({"text": "a"})
    return _refused(lambda: run.answer(run.waiting[0], bad))


def _saved_output(hint: Any, good: Any, bad: Any) -> list[Problem]:
    def give() -> hint:  # type: ignore[valid-type]
        return good

    definition = node("give", title="Give", description="For a test.")(give)
    graph = Graph.compile(
        GraphRecord.model_validate(
            {
                "nodes": {"g": {"node": "give"}},
                "outputs": {"result": {"node": "g", "output": "result"}},
            }
        ),
        registry_of(definition),
    )
    saved = graph.run().to_json().replace(f'"result":[{_json(good)}]', f'"result":[{_json(bad)}]')
    assert _json(bad) in saved
    return _refused(lambda: Run.from_json(saved, graph))


def _saved_input(hint: Any, good: Any, bad: Any) -> list[Problem]:
    @node("take", title="Take", description="For a test.")
    def take(x: hint) -> str:  # type: ignore[valid-type]
        return "ok"

    graph = Graph.compile(make_graph(take), registry_of(take))
    saved = graph.run({"x": good}).to_json().replace(f'"x":{_json(good)}', f'"x":{_json(bad)}')
    assert _json(bad) in saved
    return _refused(lambda: Run.from_json(saved, graph))


def _json(value: Any) -> str:
    return to_json(value).decode()


#: Each place a value enters, with the code of the problem a refused value gets there. Each function is given a type,
#: a value it accepts and a value it refuses, and feeds the refused one in.
WHERE_VALUES_ENTER = {
    "a-graph-input": (_graph_input, "invalid_input"),
    "a-constant": (_constant, "invalid_value"),
    "an-edge": (_edge, "invalid_edge_value"),
    "an-output": (_one_output, "invalid_output"),
    "one-of-several-outputs": (_several_outputs, "invalid_output"),
    "an-answer": (_answer, "invalid_answer"),
    "a-saved-output": (_saved_output, "saved_run_unreadable"),
    "a-saved-input": (_saved_input, "saved_run_unreadable"),
}


@pytest.mark.parametrize("way", sorted(TYPES))
@pytest.mark.parametrize("where", sorted(WHERE_VALUES_ENTER))
def test_a_refused_value_is_in_no_problem(where: str, way: str) -> None:
    feed, code = WHERE_VALUES_ENTER[where]
    problems = feed(*TYPES[way])

    assert [problem.type for problem in problems] == [code]
    for problem in problems:
        assert (
            SECRET not in problem.msg
            and SECRET not in problem.model_dump_json()
            and SECRET not in str(problem)
        ), (where, way, problem)


def test_a_value_in_a_graph_record_is_in_no_problem() -> None:
    @node("take", title="Take", description="For a test.")
    def take(x: dict[str, Any]) -> str:
        return "ok"

    for constant in ({SECRET: {1, 2}}, {"a": {SECRET: object()}}):
        for problem in _refused(lambda: make_graph(take, constants={"take.x": constant})):
            assert SECRET not in problem.model_dump_json() and SECRET not in str(problem)
    record = (
        f'{{"nodes": [{{"id": "t", "node": "take", "extra": {{"{SECRET}": 1}}}}], "edges": []}}'
    )
    for problem in _refused(lambda: Graph.compile(record, registry_of(take))):
        assert SECRET not in problem.model_dump_json() and SECRET not in str(problem)


@pytest.mark.parametrize(
    "record",
    [
        f'{{"nodes": {{"t": {{"node": "take", "inputs": {{"x": {{"constant": {{"{SECRET}": 1, "{SECRET}": 2}}}}}}}}}}}}',
        {"nodes": {"t": {"node": "take", "inputs": {"x": {"constant": {SECRET: {1, 2}}}}}}},
        {
            "nodes": {
                "t": {
                    "node": "take",
                    "inputs": {"x": {"graph_input": "x", "default": {SECRET: float("nan")}}},
                }
            }
        },
        {"nodes": {"t": {"node": "take", "display": {SECRET: {1}}}}},
    ],
    ids=["a-key-written-twice", "a-constant", "a-default", "display"],
)
def test_a_key_inside_a_value_of_a_record_is_in_no_problem(record: Any) -> None:
    """A record's fault inside a constant, a default or ``display`` stops at that value, and never quotes its keys."""

    @node("take", title="Take", description="For a test.")
    def take(x: dict[str, Any]) -> str:
        return "ok"

    [problem] = _refused(lambda: Graph.compile(record, registry_of(take)))
    assert problem.type == "invalid_record"
    assert SECRET not in problem.model_dump_json() and SECRET not in str(problem)
