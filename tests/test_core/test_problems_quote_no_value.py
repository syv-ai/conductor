"""No problem quotes the value it was made for.

A host may stream problems to a browser, so a value a graph is given, returns, saves or is answered with never
appears in a problem's ``msg``, its ``ctx`` or its text. Each case below feeds one value that pydantic would quote,
in one of the ways it quotes one (its own ``input``, a validator's message, an ``assert``, a union's tag, a key of a
dict), into each place a value enters and can be refused.
"""

from dataclasses import dataclass
from typing import Annotated, Any, Literal

import pytest
from conductor import (
    Ask,
    Asked,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeRecord,
    Problem,
    Run,
    host_type,
    make_graph,
    node,
)
from pydantic import BaseModel, Discriminator, field_validator
from pydantic_core import to_json

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


@host_type("QuotedCounts", title="Counts")
class _Counts(BaseModel):
    counts: dict[str, int]


#: Each type with a value of it, and a value it refuses that carries ``SECRET`` in one way pydantic quotes a value.
TYPES = {
    "pydantic-input": (int, 1, SECRET),
    "a-validator-message": (_Code, {"code": "ok"}, {"code": SECRET}),
    "an-assert": (_Asserted, {"code": "ok"}, {"code": SECRET}),
    "a-union-tag": (_Pet, {"pet": {"kind": "cat"}}, {"pet": {"kind": SECRET}}),
    "a-dict-key": (_Counts, {"counts": {"a": 1}}, {"counts": {SECRET: "many"}}),
}


def _refused(call: Any) -> list[Problem]:
    with pytest.raises(InvalidCallError) as raised:
        call()
    assert SECRET not in str(raised.value)
    return raised.value.errors()


def _graph_input(hint: Any, good: Any, bad: Any) -> list[Problem]:
    @node("take", title="Take", description="For a test.")
    def take(x: hint) -> str:  # type: ignore[valid-type]
        return "ok"

    return _refused(lambda: Graph.compile(make_graph(take), registry_of(take)).run(x=bad))


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

    record = GraphRecord(nodes=[NodeRecord(id="a", node="give"), NodeRecord(id="b", node="take")], edges=[("a.result", "b.x")])
    return [Graph.compile(record, registry_of(give, take)).run().state.cause]


def _giving(definition: Any) -> list[Problem]:
    return [Graph.compile(GraphRecord(nodes=[NodeRecord(id="g", node=definition.id)]), registry_of(definition)).run().state.cause]


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


def _asked_value(hint: Any, good: Any, bad: Any) -> list[Problem]:
    def ask(text: str) -> hint:  # type: ignore[valid-type]
        return bad

    @node("take", title="Take", description="For a test.")
    def take(text: str, x: Annotated[hint, Asked(ask)]) -> str:  # type: ignore[valid-type]
        return "ok"

    return [Graph.compile(make_graph(take), registry_of(take)).run(text="a").state.cause]


def _answer(hint: Any, good: Any, bad: Any) -> list[Problem]:
    def ask(text: str) -> hint:  # type: ignore[valid-type]
        raise Ask("Which?")

    run = Graph.compile(make_graph(definition := node("ask", title="Ask", description="For a test.", asks=True)(ask)), registry_of(definition)).run(text="a")
    return _refused(lambda: run.answer(run.waiting[0], bad))


def _saved_output(hint: Any, good: Any, bad: Any) -> list[Problem]:
    def give() -> hint:  # type: ignore[valid-type]
        return good

    definition = node("give", title="Give", description="For a test.")(give)
    graph = Graph.compile(GraphRecord(nodes=[NodeRecord(id="g", node="give")]), registry_of(definition))
    saved = graph.run().to_json().replace(f'"result":[{_json(good)}]', f'"result":[{_json(bad)}]')
    assert _json(bad) in saved
    return _refused(lambda: Run.from_json(saved, graph))


def _saved_input(hint: Any, good: Any, bad: Any) -> list[Problem]:
    @node("take", title="Take", description="For a test.")
    def take(x: hint) -> str:  # type: ignore[valid-type]
        return "ok"

    graph = Graph.compile(make_graph(take), registry_of(take))
    saved = graph.run(x=good).to_json().replace(f'"take.x":{_json(good)}', f'"take.x":{_json(bad)}')
    assert _json(bad) in saved
    return _refused(lambda: Run.from_json(saved, graph))


def _json(value: Any) -> str:
    return to_json(value).decode()


#: Each place a value enters, as a function that feeds it a type, a value of it and a value it refuses, with the
#: code of the problem it refuses with.
DOORS = {
    "a-graph-input": (_graph_input, "invalid_input"),
    "a-constant": (_constant, "invalid_value"),
    "an-edge": (_edge, "invalid_edge_value"),
    "an-output": (_one_output, "invalid_output"),
    "one-of-several-outputs": (_several_outputs, "invalid_output"),
    "an-asked-value": (_asked_value, "invalid_asked_value"),
    "an-answer": (_answer, "invalid_answer"),
    "a-saved-output": (_saved_output, "saved_run_unreadable"),
    "a-saved-input": (_saved_input, "saved_run_unreadable"),
}

@pytest.mark.parametrize("way", sorted(TYPES))
@pytest.mark.parametrize("door", sorted(DOORS))
def test_a_refused_value_is_in_no_problem(door: str, way: str) -> None:
    feed, code = DOORS[door]
    problems = feed(*TYPES[way])

    assert [problem.type for problem in problems] == [code]
    for problem in problems:
        assert SECRET not in problem.msg and SECRET not in problem.model_dump_json() and SECRET not in str(problem), (door, way, problem)


def test_a_value_in_a_graph_record_is_in_no_problem() -> None:
    @node("take", title="Take", description="For a test.")
    def take(x: dict[str, Any]) -> str:
        return "ok"

    for constant in ({SECRET: {1, 2}}, {"a": {SECRET: object()}}):
        for problem in _refused(lambda: make_graph(take, constants={"take.x": constant})):
            assert SECRET not in problem.model_dump_json() and SECRET not in str(problem)
    record = f'{{"nodes": [{{"id": "t", "node": "take", "extra": {{"{SECRET}": 1}}}}], "edges": []}}'
    for problem in _refused(lambda: Graph.compile(record, registry_of(take))):
        assert SECRET not in problem.model_dump_json() and SECRET not in str(problem)
