"""Cross-cutting engine scenarios: a deciding node's branches meeting failure, retry policies, and skip propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from conductor import (
    SKIPPED,
    CompiledGraph,
    DType,
    ExternalFailure,
    GraphRecord,
    NodeDefinition,
    NodeError,
    NodeRecord,
    NodeRegistry,
    Param,
    Policy,
    Result,
    node,
)
from conductor.widgets import NumberWidget, Textarea
from helpers import registry_of, stream_of


class Txt(DType, str):
    id = "integration-scenarios-test-text"
    display_title = "Text"


class Num(DType, float):
    id = "integration-scenarios-test-number"
    display_title = "Number"


Out = Annotated[Txt, Result(title="Out")]
# ---------------------------------------------------------------------------
# Shared nodes
# ---------------------------------------------------------------------------


@node("echo", title="Echo", description="Returns its text", category="test")
def echo(text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
    return text


@node("record", title="Record", description="Returns its label", category="test")
def record(label: Annotated[Txt, Param(title="Label", widget=Textarea())] = Txt("hit")) -> Out:
    return label


@node("always-fail", title="Always fail", description="Fails with the given reason", category="test")
def always_fail(
    reason: Annotated[Txt, Param(title="Why", widget=Textarea())] = Txt("boom"),
    number: Annotated[Num, Param(title="Number", widget=NumberWidget())] = Num(0),
) -> Out:
    raise NodeError(reason)


@node("tally", title="Tally", description="Returns its label once a number arrives", category="test")
def tally(
    number: Annotated[Num, Param(title="Number", widget=NumberWidget())] = Num(0),
    label: Annotated[Txt, Param(title="Label", widget=Textarea())] = Txt("hit"),
) -> Out:
    """Hangs off one branch of a ``Decide``: receives its number, returns its label."""
    return label


@dataclass(frozen=True)
class Branches:
    """What ``Decide.run`` returns: the value on one output, ``SKIPPED`` on the other."""

    high: Annotated[Num, Result(title="High", exclusive_group="branch")]
    low: Annotated[Num, Result(title="Low", exclusive_group="branch")]


@node("decide", title="Decide", description="Routes a number to `high` or `low` around a threshold", category="test")
def decide(
    value: Annotated[Num, Param(title="Value", widget=NumberWidget())] = Num(0),
    threshold: Annotated[Num, Param(title="Threshold", widget=NumberWidget())] = Num(50),
) -> Branches:
    if value > threshold:
        return Branches(high=value, low=SKIPPED)
    return Branches(high=SKIPPED, low=value)


def _registry(*extra: type[NodeDefinition]) -> NodeRegistry:
    return registry_of(echo, record, always_fail, decide, tally, *extra)


async def _events(compiled) -> list[dict]:
    """Every event of one run, including those after a ``graph_error``."""
    return [event async for event in stream_of(compiled)]


def _kinds(events: list[dict]) -> list[str]:
    return [e.type for e in events]


def _completed(events: list[dict], node_id: str) -> bool:
    return any(
        e.get("type") == "node_complete" and e.get("node_id") == node_id for e in events
    )


# ===========================================================================
# Decision combinations
# ===========================================================================


class TestDecisionCombinations:
    """A deciding node's ``SKIPPED`` branch intersecting with failure."""

    def test_decision_branch_failure_does_not_affect_other_branch(self):
        """The branch not taken holds a failing node that never runs."""
        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id="d", node="decide", values={"value": 100}),
            NodeRecord(id="a", node="tally", values={"label": "A"}),
            NodeRecord(id="b", node="always-fail"),
        ], edges=[('d.high', "a.number"), ('d.low', "b.number")]), _registry())
        r = compiled.results(compiled.run_sync())
        # A ran; B was skipped so it never failed
        assert r["a"]["result"] == "A"
        assert "b" not in r

# ===========================================================================
# Compensation scenarios
# ===========================================================================


# ===========================================================================
# Retry
# ===========================================================================


class TestRetry:
    """A version's ``Policy`` across failures."""

    def test_node_level_retry_recovers(self):
        """A node that fails once and succeeds on retry succeeds overall."""
        calls = 0

        @node("flaky", title="Flaky", description="Fails once, then succeeds", category="test", policy=Policy(retries=3, retry_delay=0.01))
        def flaky() -> Out:
            nonlocal calls
            calls += 1
            if calls < 2:
                raise ExternalFailure("transient")
            return Txt("ok")

        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id="n1", node="flaky"),
        ]), _registry(flaky))
        r = compiled.results(compiled.run_sync())
        assert r["n1"]["result"] == "ok"
        assert calls == 2  # one failure + one success

# ===========================================================================
# Full-circle scenarios
# ===========================================================================


# ===========================================================================
# Edge cases
# ===========================================================================


@dataclass(frozen=True)
class Match:
    match: Annotated[Txt, Result(title="Match", exclusive_group="equality")]
    other: Annotated[Txt, Result(title="Other", exclusive_group="equality")]

@node("route", title="Route", description="Routes a text by whether it equals `expect`", category="test")
def route(
    text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt(""),
    expect: Annotated[Txt, Param(title="Expect", widget=Textarea())] = Txt("data"),
) -> Match:
    if text == expect:
        return Match(match=text, other=SKIPPED)
    return Match(match=SKIPPED, other=text)


class TestEdgeCases:
    """Shapes that once caught bugs."""

    def test_decision_routes_only_the_taken_branch(self):
        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id="d", node="decide", values={"value": 100}),
            NodeRecord(id="taken", node="tally", values={"label": "TAKEN"}),
            NodeRecord(id="other", node="tally", values={"label": "OTHER"}),
        ], edges=[('d.high', "taken.number"), ('d.low', "other.number")]), _registry())
        r = compiled.results(compiled.run_sync())
        assert r["taken"]["result"] == "TAKEN"
        assert "other" not in r

    def test_skip_propagates_through_decision_else_branch(self):
        """A deciding node fed by an edge routes the taken branch; the else branch is skipped."""

        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id="source", node="echo", values={"text": "data"}),
            NodeRecord(id="d", node="route"),
            NodeRecord(id="taken", node="echo"),
            NodeRecord(id="else_b", node="echo"),
        ], edges=[('source.result', "d.text"), ('d.match', "taken.text"), ('d.other', "else_b.text")]), _registry(route))
        r = compiled.results(compiled.run_sync())
        assert r["taken"]["result"] == "data"
        assert "else_b" not in r
