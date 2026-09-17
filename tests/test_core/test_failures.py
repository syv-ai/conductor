"""Two failure families, and what people are told.

A node's failure is internal unless the outside world caused it. An
``ExternalFailure`` — raised by the node itself, or a foreign exception the
version's ``Policy(retry_on=...)`` names — retries under the policy; any
other failure runs once. A cause carries a generic message per code unless
the node wrote one for people; a foreign exception's text stays on the
exception's ``original`` and never reaches an event.
"""

import asyncio
import json
from typing import Annotated

import pytest
from conductor import CompiledGraph, GraphNode, NodeRegistry
from conductor.dtype import DType
from conductor.errors import (
    MESSAGES,
    ErrorCause,
    ExternalFailure,
    GraphExecutionError,
    NodeError,
    NodeExecutionError,
    NodeValidationError,
)
from conductor.execution.engine import _Leg, execute, execute_sync
from conductor.graph.binding import Edges, Static
from conductor.graph.model import Graph
from conductor.node import NodeDefinition, Policy, version
from conductor.ref import Ref
from conductor.returns import Result
from conductor.series import Series
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "failures-test-text"
    title = "Text"


In = Annotated[Txt, Textarea(title="In")]
Out = Annotated[Txt, Result(title="Out")]


def _compiled(node_cls: type[NodeDefinition], *more: type[NodeDefinition]) -> CompiledGraph:
    """One node of ``node_cls`` as ``n1``, its text typed in."""
    reg = NodeRegistry()
    for cls in (node_cls, *more):
        reg.register(cls)
    return CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="n1", type=node_cls.id, version=1, bindings={"text": Static(value="x")})]), reg
    )


def _events(compiled: CompiledGraph) -> list[dict]:
    async def run() -> list[dict]:
        return [event async for event in execute(compiled)]

    return asyncio.run(run())


def _error(events: list[dict]) -> dict:
    return next(e for e in events if e["type"] == "node_error")


# -- the families -----------------------------------------------------------------


def test_a_foreign_exception_runs_once_and_is_execution_failed():
    """``{}['missing']`` under ``Policy(retries=3)`` is a bug, not weather: one
    call, no retry, and the event says only that the node failed."""
    calls: list[int] = []

    class Buggy(NodeDefinition):
        id = "buggy"
        title = "Buggy"
        description = "d"
        category = "test"

        @version(1, policy=Policy(retries=3, delay=0.01))
        def run(self, text: In = Txt("")) -> Out:
            calls.append(1)
            return {}["missing"]

    events = _events(_compiled(Buggy))

    assert len(calls) == 1
    assert [e["type"] for e in events if e["type"] == "node_retry"] == []
    error = _error(events)
    assert error["cause"].code == "execution_failed"
    assert error["cause"].message == "The node failed."
    assert error["error"] == "The node failed."
    assert "missing" not in json.dumps(error["cause"].model_dump(mode="json"))


def test_a_foreign_exception_named_in_retry_on_is_external_and_retries():
    calls: list[int] = []

    class Flaky(NodeDefinition):
        id = "flaky"
        title = "Flaky"
        description = "d"
        category = "test"

        @version(1, policy=Policy(retries=2, delay=0.01, retry_on=(ConnectionError,)))
        def run(self, text: In = Txt("")) -> Out:
            calls.append(1)
            if len(calls) < 3:
                raise ConnectionError("socket closed")
            return Txt("ok")

    events = _events(_compiled(Flaky))

    assert len(calls) == 3
    retries = [e for e in events if e["type"] == "node_retry"]
    assert [(e["attempt"], e["retries"], e["node_id"]) for e in retries] == [(1, 2, "n1"), (2, 2, "n1")]
    assert retries[0]["error"] == "An outside service did not answer."
    assert events[-1]["type"] == "graph_complete"
    assert events[-1]["results"]["n1"]["result"] == "ok"


def test_an_exhausted_external_failure_is_external_failed():
    calls: list[int] = []

    class Down(NodeDefinition):
        id = "down"
        title = "Down"
        description = "d"
        category = "test"

        @version(1, policy=Policy(retries=2, delay=0.01, retry_on=(ConnectionError,)))
        def run(self, text: In = Txt("")) -> Out:
            calls.append(1)
            raise ConnectionError("socket closed")

    compiled = _compiled(Down)
    events = _events(compiled)

    assert len(calls) == 3
    error = _error(events)
    assert error["cause"].code == "external_failed"
    assert error["cause"].message == "An outside service did not answer."
    assert "socket" not in error["error"]
    with pytest.raises(GraphExecutionError):
        execute_sync(compiled)


def test_an_external_failure_the_node_raised_retries_and_keeps_its_message():
    """A node that knows the outside world failed says so with ``ExternalFailure``;
    the message it wrote is what people read, and the code is the family's."""
    calls: list[int] = []

    class Bank(NodeDefinition):
        id = "bank"
        title = "Bank"
        description = "d"
        category = "test"

        @version(1, policy=Policy(retries=1, delay=0.01))
        def run(self, text: In = Txt("")) -> Out:
            calls.append(1)
            raise ExternalFailure("Nationalbanken svarede ikke.")

    error = _error(_events(_compiled(Bank)))

    assert len(calls) == 2
    assert error["cause"].code == "external_failed"
    assert error["cause"].message == "Nationalbanken svarede ikke."


def test_a_node_error_the_node_raised_runs_once_and_keeps_its_message():
    calls: list[int] = []

    class Reader(NodeDefinition):
        id = "reader"
        title = "Reader"
        description = "d"
        category = "test"

        @version(1, policy=Policy(retries=3, delay=0.01))
        def run(self, text: In = Txt("")) -> Out:
            calls.append(1)
            if len(calls) == 1:
                raise NodeError("Dokumentet kunne ikke læses.")
            raise NodeExecutionError(
                "unreadable", cause=ErrorCause(code="unreadable", message="Dokumentet er beskadiget.")
            )

    error = _error(_events(_compiled(Reader)))

    assert len(calls) == 1
    assert (error["cause"].code, error["cause"].message) == ("failed", "Dokumentet kunne ikke læses.")


def test_a_nodes_own_cause_streams_as_written():
    class Reader(NodeDefinition):
        id = "reader"
        title = "Reader"
        description = "d"
        category = "test"

        def run(self, text: In = Txt("")) -> Out:
            raise NodeExecutionError(
                "unreadable", cause=ErrorCause(code="unreadable", message="Dokumentet er beskadiget.")
            )

    error = _error(_events(_compiled(Reader)))

    assert (error["cause"].code, error["cause"].message) == ("unreadable", "Dokumentet er beskadiget.")


def test_a_validation_error_the_node_raised_is_never_retried():
    calls: list[int] = []

    class Picky(NodeDefinition):
        id = "picky"
        title = "Picky"
        description = "d"
        category = "test"

        @version(1, policy=Policy(retries=3, delay=0.01))
        def run(self, text: In = Txt("")) -> Out:
            calls.append(1)
            raise NodeValidationError("intentionally invalid input")

    with pytest.raises(GraphExecutionError):
        execute_sync(_compiled(Picky))

    assert len(calls) == 1


# -- the foreign exception's text -----------------------------------------------


def test_a_foreign_exceptions_text_is_on_original_and_nowhere_else():
    """The wrapped exception keeps the foreign one on ``original`` (and as its
    ``__cause__``) for a log; its own text and its cause are the generic line."""

    class Buggy(NodeDefinition):
        id = "buggy"
        title = "Buggy"
        description = "d"
        category = "test"

        def run(self, text: In = Txt("")) -> Out:
            raise KeyError("secret-token")

    compiled = _compiled(Buggy)
    leg = _Leg(compiled, record=None, from_run={}, timeout=None, cancel=asyncio.Event())

    with pytest.raises(NodeExecutionError) as caught:
        leg._call(("n1", None), {"text": "x"})

    wrapped = caught.value
    assert isinstance(wrapped.original, KeyError)
    assert wrapped.__cause__ is wrapped.original
    assert str(wrapped) == "The node failed."
    assert wrapped.cause is not None and wrapped.cause.message == "The node failed."
    assert "secret-token" not in str(wrapped) + json.dumps(wrapped.cause.model_dump(mode="json"))


def test_the_generic_messages_cover_the_engines_own_codes():
    """``MESSAGES`` says what people read for each code the engine writes
    itself; ``failed`` is the node's own message and has no generic line."""
    assert set(MESSAGES) == {"engine_error", "execution_failed", "external_failed", "invalid_input", "timeout"}


def test_node_error_has_no_retryable_flag():
    """The family is the class: an ``ExternalFailure`` retries, nothing else does."""
    assert not hasattr(NodeError, "retryable")
    assert issubclass(ExternalFailure, NodeError)


# -- the policy's bounds ------------------------------------------------------------


@pytest.mark.parametrize("refused", [{"concurrency": 0}, {"retries": -1}, {"delay": -0.5}, {"timeout": 0}])
def test_a_policy_out_of_bounds_is_refused(refused):
    with pytest.raises(ValueError):
        Policy(**refused)


def test_a_policy_keeps_retry_on_out_of_what_a_palette_reads():
    """Which exceptions a node retries on is the engine's business; a palette
    dumps the policy to JSON, and exception classes have no JSON form."""
    policy = Policy(retries=2, retry_on=(ConnectionError, TimeoutError))

    assert policy.retry_on == (ConnectionError, TimeoutError)
    assert policy.model_dump(mode="json") == {"retries": 2, "delay": 1.0, "timeout": None, "concurrency": 8}
    assert "retry_on" not in Policy.model_json_schema()["properties"]


# -- retries and rows ------------------------------------------------------------------


def test_a_failed_row_is_retried_alone():
    """A node that runs per row retries the row that failed; the rows that
    completed are not run again."""
    calls: list[str] = []

    class Split(NodeDefinition):
        id = "split"
        title = "Split"
        description = "d"
        category = "test"

        def run(self, text: In = Txt("")) -> Annotated[Series[Txt], Result(title="Parts")]:
            return [Txt(part) for part in text.split(",")]

    class FlakyOnB(NodeDefinition):
        id = "flaky-on-b"
        title = "Flaky on b"
        description = "d"
        category = "test"

        @version(1, policy=Policy(retries=2, delay=0.01))
        def run(self, text: In = Txt("")) -> Out:
            calls.append(str(text))
            if text == "b" and calls.count("b") == 1:
                raise ExternalFailure("transient")
            return Txt(text.upper())

    reg = NodeRegistry()
    reg.register(Split)
    reg.register(FlakyOnB)
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="split", type="split", version=1, bindings={"text": Static(value="a,b,c")}),
            GraphNode(id="rows", type="flaky-on-b", version=1, bindings={"text": Edges(refs=(Ref("split", "result"),))}),
        ]),
        reg,
    )

    results = execute_sync(compiled)

    assert list(results["rows"]["result"]) == ["A", "B", "C"]
    assert sorted(calls) == ["a", "b", "b", "c"]


def test_a_flaky_node_in_one_branch_retries_while_the_other_branch_completes():
    calls = {"a": 0, "b": 0}

    class FlakyA(NodeDefinition):
        id = "flaky-a"
        title = "Flaky A"
        description = "d"
        category = "test"

        @version(1, policy=Policy(retries=2, delay=0.01))
        def run(self, text: In = Txt("")) -> Out:
            calls["a"] += 1
            if calls["a"] == 1:
                raise ExternalFailure("first try")
            return Txt(f"A:{text}")

    class FastB(NodeDefinition):
        id = "fast-b"
        title = "Fast B"
        description = "d"
        category = "test"

        def run(self, text: In = Txt("")) -> Out:
            calls["b"] += 1
            return Txt(f"B:{text}")

    class Join(NodeDefinition):
        id = "join"
        title = "Join"
        description = "d"
        category = "test"

        def run(self, a: Annotated[Txt, Textarea(title="A")] = Txt(""), b: Annotated[Txt, Textarea(title="B")] = Txt("")) -> Out:
            return Txt(f"{a}+{b}")

    reg = NodeRegistry()
    for cls in (FlakyA, FastB, Join):
        reg.register(cls)
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[
            GraphNode(id="n1", type="flaky-a", version=1, bindings={"text": Static(value="x")}),
            GraphNode(id="n2", type="fast-b", version=1, bindings={"text": Static(value="y")}),
            GraphNode(id="n3", type="join", version=1, bindings={"a": Edges(refs=(Ref("n1", "result"),)), "b": Edges(refs=(Ref("n2", "result"),))}),
        ]),
        reg,
    )

    results = execute_sync(compiled)

    assert results["n3"]["result"] == "A:x+B:y"
    assert calls == {"a": 2, "b": 1}
