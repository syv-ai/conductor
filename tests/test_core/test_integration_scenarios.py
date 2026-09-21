"""Cross-cutting engine scenarios: a deciding node's branches meeting failure, retry policies, and skip propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from conductor import CompiledGraph, GraphNode, NodeRegistry, Param, run_sync
from conductor._sentinel import SKIPPED
from conductor.dtype import DType
from conductor.errors import ExternalFailure, NodeExecutionError
from conductor.execution.engine import execute
from conductor.graph.binding import Edges, Static
from conductor.graph.model import Graph
from conductor.metadata import Result
from conductor.node import NodeDefinition, Policy, version
from conductor.ref import Ref
from conductor.widgets import NumberWidget, Textarea


class Txt(DType, str):
    id = "integration-scenarios-test-text"
    title = "Text"


class Num(DType, float):
    id = "integration-scenarios-test-number"
    title = "Number"


Out = Annotated[Txt, Result(title="Out")]
# ---------------------------------------------------------------------------
# Shared nodes
# ---------------------------------------------------------------------------


class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "Returns its text"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt("")) -> Out:
        return text


class Record(NodeDefinition):
    id = "record"
    title = "Record"
    description = "Returns its label"
    category = "test"

    def run(self, label: Annotated[Txt, Param(title="Label", widget=Textarea())] = Txt("hit")) -> Out:
        return label


class AlwaysFail(NodeDefinition):
    id = "always-fail"
    title = "Always fail"
    description = "Fails with the given reason"
    category = "test"

    def run(
        self,
        reason: Annotated[Txt, Param(title="Why", widget=Textarea())] = Txt("boom"),
        number: Annotated[Num, Param(title="Number", widget=NumberWidget())] = Num(0),
    ) -> Out:
        raise NodeExecutionError(reason, node_id="always_fail")


class Tally(NodeDefinition):
    """Hangs off one branch of a ``Decide``: receives its number, returns its label."""

    id = "tally"
    title = "Tally"
    description = "Returns its label once a number arrives"
    category = "test"

    def run(
        self,
        number: Annotated[Num, Param(title="Number", widget=NumberWidget())] = Num(0),
        label: Annotated[Txt, Param(title="Label", widget=Textarea())] = Txt("hit"),
    ) -> Out:
        return label


@dataclass(frozen=True)
class Branches:
    """What ``Decide.run`` returns: the value on one output, ``SKIPPED`` on the other."""

    high: Annotated[Num, Result(title="High", choice="branch")]
    low: Annotated[Num, Result(title="Low", choice="branch")]


class Decide(NodeDefinition):
    id = "decide"
    title = "Decide"
    description = "Routes a number to `high` or `low` around a threshold"
    category = "test"

    def run(
        self,
        value: Annotated[Num, Param(title="Value", widget=NumberWidget())] = Num(0),
        threshold: Annotated[Num, Param(title="Threshold", widget=NumberWidget())] = Num(50),
    ) -> Branches:
        if value > threshold:
            return Branches(high=value, low=SKIPPED)
        return Branches(high=SKIPPED, low=value)


def _registry(*extra: type[NodeDefinition]) -> NodeRegistry:
    reg = NodeRegistry()
    for node_cls in (Echo, Record, AlwaysFail, Decide, Tally, *extra):
        reg.register(node_cls)
    return reg


async def _events(compiled) -> list[dict]:
    """Every event of one run, including those after a ``graph_error``."""
    return [event async for event in execute(compiled)]


def _kinds(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


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
        compiled = CompiledGraph.from_graph(Graph(nodes=[
                GraphNode(id="d", type="decide", version=1, bindings={"value": Static(value=100)}),
                GraphNode(id="a", type="tally", version=1, bindings={"label": Static(value="A"), "number": Edges(refs=(Ref('d', 'high'),))}),
                GraphNode(id="b", type="always-fail", version=1, bindings={"number": Edges(refs=(Ref('d', 'low'),))}),
            ]), _registry())
        r = run_sync(compiled)["results"]
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

        class Flaky(NodeDefinition):
            id = "flaky"
            title = "Flaky"
            description = "Fails once, then succeeds"
            category = "test"

            @version(1, policy=Policy(retries=3, delay=0.01))
            def run(self) -> Out:
                nonlocal calls
                calls += 1
                if calls < 2:
                    raise ExternalFailure("transient", node_id="flaky")
                return Txt("ok")

        compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n1", type="flaky", version=1)]), _registry(Flaky))
        r = run_sync(compiled)["results"]
        assert r["n1"]["result"] == "ok"
        assert calls == 2  # one failure + one success

# ===========================================================================
# Full-circle scenarios
# ===========================================================================


# ===========================================================================
# Edge-case regressions
# ===========================================================================


@dataclass(frozen=True)
class Match:
    match: Annotated[Txt, Result(title="Match", choice="equality")]
    other: Annotated[Txt, Result(title="Other", choice="equality")]

class Route(NodeDefinition):
    id = "route"
    title = "Route"
    description = "Routes a text by whether it equals `expect`"
    category = "test"

    def run(
        self,
        text: Annotated[Txt, Param(title="Text", widget=Textarea())] = Txt(""),
        expect: Annotated[Txt, Param(title="Expect", widget=Textarea())] = Txt("data"),
    ) -> Match:
        if text == expect:
            return Match(match=text, other=SKIPPED)
        return Match(match=SKIPPED, other=text)


class TestEdgeCases:
    """Shapes that once caught bugs."""

    def test_decision_routes_only_the_taken_branch(self):
        compiled = CompiledGraph.from_graph(Graph(nodes=[
                GraphNode(id="d", type="decide", version=1, bindings={"value": Static(value=100)}),
                GraphNode(id="taken", type="tally", version=1, bindings={"label": Static(value="TAKEN"), "number": Edges(refs=(Ref('d', 'high'),))}),
                GraphNode(id="other", type="tally", version=1, bindings={"label": Static(value="OTHER"), "number": Edges(refs=(Ref('d', 'low'),))}),
            ]), _registry())
        r = run_sync(compiled)["results"]
        assert r["taken"]["result"] == "TAKEN"
        assert "other" not in r

    def test_skip_propagates_through_decision_else_branch(self):
        """A deciding node fed by an edge routes the taken branch; the else branch is skipped."""

        compiled = CompiledGraph.from_graph(Graph(nodes=[
                GraphNode(id="source", type="echo", version=1, bindings={"text": Static(value="data")}),
                GraphNode(id="d", type="route", version=1, bindings={"text": Edges(refs=(Ref('source', 'result'),))}),
                GraphNode(id="taken", type="echo", version=1, bindings={"text": Edges(refs=(Ref('d', 'match'),))}),
                GraphNode(id="else_b", type="echo", version=1, bindings={"text": Edges(refs=(Ref('d', 'other'),))}),
            ]), _registry(Route))
        r = run_sync(compiled)["results"]
        assert r["taken"]["result"] == "data"
        assert "else_b" not in r
