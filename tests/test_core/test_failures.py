"""Two failure families, and what people are told.

A node's failure is internal unless the outside world caused it. An
``ExternalFailure`` — raised by the node itself, or a foreign exception the
version's ``Policy(retry_on=...)`` names — retries under the policy; any
other failure runs once. A cause carries a generic message per code unless
the node wrote one for people; a foreign exception's text goes to the
log and never reaches an event.
"""

import json
from typing import Annotated

import pytest
from conductor import (
    CompiledGraph,
    DType,
    ExternalFailure,
    GraphRecord,
    NodeDefinition,
    NodeError,
    NodeRecord,
    Param,
    Policy,
    Result,
    Series,
    node,
)
from conductor._internal.execution.call_node import CallFailed, call_node
from conductor.widgets import Textarea
from helpers import events_of, registry_of


class Txt(DType, str):
    id = "failures-test-text"
    display_title = "Text"


In = Annotated[Txt, Param(title="In", widget=Textarea())]
Out = Annotated[Txt, Result(title="Out")]


def _compiled(node_cls: type[NodeDefinition], *more: type[NodeDefinition]) -> CompiledGraph:
    """One node of ``node_cls`` as ``n1``, its text typed in."""
    reg = registry_of(node_cls, *more)
    return CompiledGraph.from_graph(
        GraphRecord(nodes=[
            NodeRecord(id="n1", node=node_cls.id, values={"text": "x"}),
        ]), reg
    )


def _error(events: list[dict]) -> dict:
    return next(e for e in events if e.type == "node_failed")


# -- the families -----------------------------------------------------------------


def test_a_foreign_exception_runs_once_and_is_execution_failed():
    """``{}['missing']`` under ``Policy(retries=3)`` is a bug, not weather: one
    call, no retry, and the event says only that the node failed."""
    calls: list[int] = []

    @node("buggy", title="Buggy", description="d", category="test", policy=Policy(retries=3, retry_delay=0.01))
    def buggy(text: In = Txt("")) -> Out:
        calls.append(1)
        return {}["missing"]

    events = events_of(_compiled(buggy))

    assert len(calls) == 1
    assert [e.type for e in events if e.type == "node_retry"] == []
    error = _error(events)
    assert error.cause.type == "execution_failed"
    assert error.cause.msg == "The node failed."
    assert "missing" not in json.dumps(error.cause.model_dump(mode="json"))


def test_a_foreign_exception_named_in_retry_on_is_external_and_retries():
    calls: list[int] = []

    @node("flaky", title="Flaky", description="d", category="test", policy=Policy(retries=2, retry_delay=0.01, retry_on=(ConnectionError,)))
    def flaky(text: In = Txt("")) -> Out:
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError("socket closed")
        return Txt("ok")

    compiled = _compiled(flaky)
    events = events_of(compiled)

    assert len(calls) == 3
    retries = [e for e in events if e.type == "node_retry"]
    assert [(e.attempt, e.retries, e.node_id) for e in retries] == [(1, 2, "n1"), (2, 2, "n1")]
    assert retries[0].cause.msg == "An outside service did not answer."
    assert events[-1].state.status == "complete"
    assert compiled.results(events[-1].state)["n1"]["result"] == "ok"


def test_an_exhausted_external_failure_is_external_failed():
    calls: list[int] = []

    @node("down", title="Down", description="d", category="test", policy=Policy(retries=2, retry_delay=0.01, retry_on=(ConnectionError,)))
    def down(text: In = Txt("")) -> Out:
        calls.append(1)
        raise ConnectionError("socket closed")

    compiled = _compiled(down)
    events = events_of(compiled)

    assert len(calls) == 3
    error = _error(events)
    assert error.cause.type == "external_failed"
    assert error.cause.msg == "An outside service did not answer."
    assert "socket" not in error.cause.msg
    assert compiled.run_sync().status == "failed"


def test_an_external_failure_the_node_raised_retries_and_keeps_its_message():
    """A node that knows the outside world failed says so with ``ExternalFailure``;
    the message it wrote is what people read, and the code is the family's."""
    calls: list[int] = []

    @node("bank", title="Bank", description="d", category="test", policy=Policy(retries=1, retry_delay=0.01))
    def bank(text: In = Txt("")) -> Out:
        calls.append(1)
        raise ExternalFailure("Nationalbanken svarede ikke.")

    error = _error(events_of(_compiled(bank)))

    assert len(calls) == 2
    assert error.cause.type == "external_failed"
    assert error.cause.msg == "Nationalbanken svarede ikke."


def test_a_node_error_the_node_raised_runs_once_and_keeps_its_message():
    calls: list[int] = []

    @node("reader", title="Reader", description="d", category="test", policy=Policy(retries=3, retry_delay=0.01))
    def reader(text: In = Txt("")) -> Out:
        calls.append(1)
        if len(calls) == 1:
            raise NodeError("Dokumentet kunne ikke læses.")
        raise NodeError(
            "unreadable", code="unreadable"
        )

    error = _error(events_of(_compiled(reader)))

    assert len(calls) == 1
    assert (error.cause.type, error.cause.msg) == ("failed", "Dokumentet kunne ikke læses.")


def test_a_nodes_own_cause_streams_as_written():
    @node("reader", title="Reader", description="d", category="test")
    def reader(text: In = Txt("")) -> Out:
        raise NodeError(
            "unreadable", code="unreadable"
        )

    error = _error(events_of(_compiled(reader)))

    assert (error.cause.type, error.cause.msg) == ("unreadable", "unreadable")


def test_a_validation_error_the_node_raised_is_never_retried():
    calls: list[int] = []

    @node("picky", title="Picky", description="d", category="test", policy=Policy(retries=3, retry_delay=0.01))
    def picky(text: In = Txt("")) -> Out:
        calls.append(1)
        raise NodeError("intentionally invalid input", code="invalid_input")

    assert _compiled(picky).run_sync().status == "failed"

    assert len(calls) == 1


# -- the foreign exception's text -----------------------------------------------


def test_a_foreign_exceptions_text_is_on_no_problem():
    """The failure carries the generic line and the exception's class name; its own text is in the log only."""

    @node("buggy", title="Buggy", description="d", category="test")
    def buggy(text: In = Txt("")) -> Out:
        raise KeyError("secret-token")

    outcome = call_node(_compiled(buggy).node("n1"), (), {"text": "x"}, {})

    assert isinstance(outcome, CallFailed) and not outcome.retryable
    assert (outcome.problem.type, outcome.problem.msg, outcome.problem.ctx) == ("execution_failed", "The node failed.", {"exception": "KeyError"})
    assert "secret-token" not in outcome.problem.model_dump_json()



def test_node_error_has_no_retryable_flag():
    """The family is the class: an ``ExternalFailure`` retries, nothing else does."""
    assert not hasattr(NodeError, "retryable")
    assert issubclass(ExternalFailure, NodeError)


# -- the policy's bounds ------------------------------------------------------------


@pytest.mark.parametrize("raised", [{"concurrency": 0}, {"retries": -1}, {"delay": -0.5}, {"timeout": 0}])
def test_a_policy_out_of_bounds_is_refused(raised):
    with pytest.raises(ValueError):
        Policy(**raised)


def test_a_policy_keeps_retry_on_out_of_what_a_palette_reads():
    """Which exceptions a node retries on is the engine's business; a palette
    dumps the policy to JSON, and exception classes have no JSON form."""
    policy = Policy(retries=2, retry_on=(ConnectionError, TimeoutError))

    assert policy.retry_on == (ConnectionError, TimeoutError)
    assert policy.model_dump(mode="json") == {"retries": 2, "retry_delay": 1.0, "attempt_timeout": None, "concurrency": 8, "reuse_when_inputs_match": True}
    assert "retry_on" not in Policy.model_json_schema()["properties"]


# -- retries and rows ------------------------------------------------------------------


def test_a_failed_row_is_retried_alone():
    """A node that runs per row retries the row that failed; the rows that
    completed are not run again."""
    calls: list[str] = []

    @node("split", title="Split", description="d", category="test")
    def split_node(text: In = Txt("")) -> Annotated[Series[Txt], Result(title="Parts")]:
        return [Txt(part) for part in text.split(",")]

    @node("flaky-on-b", title="Flaky on b", description="d", category="test", policy=Policy(retries=2, retry_delay=0.01))
    def flaky_on_b(text: In = Txt("")) -> Out:
        calls.append(str(text))
        if text == "b" and calls.count("b") == 1:
            raise ExternalFailure("transient")
        return Txt(text.upper())

    reg = registry_of(split_node, flaky_on_b)
    compiled = CompiledGraph.from_graph(
        GraphRecord(nodes=[
            NodeRecord(id="split", node="split", values={"text": "a,b,c"}),
            NodeRecord(id="rows", node="flaky-on-b"),
        ], edges=[("split.result", "rows.text")]),
        reg,
    )

    results = compiled.results(compiled.run_sync())

    assert results["rows"]["result"]["values"] == ["A", "B", "C"]
    assert sorted(calls) == ["a", "b", "b", "c"]


def test_a_flaky_node_in_one_branch_retries_while_the_other_branch_completes():
    calls = {"a": 0, "b": 0}

    @node("flaky-a", title="Flaky A", description="d", category="test", policy=Policy(retries=2, retry_delay=0.01))
    def flaky_a(text: In = Txt("")) -> Out:
        calls["a"] += 1
        if calls["a"] == 1:
            raise ExternalFailure("first try")
        return Txt(f"A:{text}")

    @node("fast-b", title="Fast B", description="d", category="test")
    def fast_b(text: In = Txt("")) -> Out:
        calls["b"] += 1
        return Txt(f"B:{text}")

    @node("join", title="Join", description="d", category="test")
    def join(a: Annotated[Txt, Param(title="A", widget=Textarea())] = Txt(""), b: Annotated[Txt, Param(title="B", widget=Textarea())] = Txt("")) -> Out:
        return Txt(f"{a}+{b}")

    reg = registry_of(flaky_a, fast_b, join)
    compiled = CompiledGraph.from_graph(
        GraphRecord(nodes=[
            NodeRecord(id="n1", node="flaky-a", values={"text": "x"}),
            NodeRecord(id="n2", node="fast-b", values={"text": "y"}),
            NodeRecord(id="n3", node="join"),
        ], edges=[("n1.result", "n3.a"), ("n2.result", "n3.b")]),
        reg,
    )

    results = compiled.results(compiled.run_sync())

    assert results["n3"]["result"] == "A:x+B:y"
    assert calls == {"a": 2, "b": 1}


class Shape(DType, dict):
    id = "failures-shape"
    display_title = "Shape"


class Square(Shape):
    """A narrower type whose validator lets any ``Shape`` through as it is —
    the hole a host type can have: in memory it accepts what it would never
    build from JSON."""

    id = "failures-square"
    display_title = "Square"

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        from pydantic_core import core_schema

        return core_schema.no_info_plain_validator_function(
            lambda value: value if isinstance(value, Shape) else cls(value)
        )


def test_a_type_whose_validator_does_not_return_the_type_fails_loud():
    """An output is read as the type its field declares when the node
    returns it. A type whose validator hands back something else is a bug
    in the type, not in the node, and it fails the run naming the type,
    rather than leaving a value on the field that its type would not build.
    The error is the engine's kind, ``engine_error``: the node did nothing wrong."""

    @node("draws", title="Draws", description="d", category="test")
    def draws() -> Annotated[Square, Result(title="Square")]:
        return Shape(side=2)

    registry = registry_of(draws)
    compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
        NodeRecord(id="d", node="draws"),
    ]), registry)

    ending = compiled.run_sync()

    assert ending.status == "failed" and ending.cause.loc[0] == "d"
    assert ending.cause.type == "engine_error" and ending.cause.ctx == {"exception": "TypeError"}
    assert compiled.results(ending) == {}
