"""Cross-cutting engine scenarios: a deciding node's branches meeting failure, retry policies, and skip propagation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from conductor import (
    GraphNode,
    NodeRegistry,
    compile,
    execute,
    execute_sync,
)
from conductor._sentinel import SKIPPED
from conductor.dtype import DType
from conductor.errors import NodeExecutionError
from conductor.graph.binding import Sources, Static
from conductor.graph.model import Flow
from conductor.node import NodeDefinition, Policy, version
from conductor.ref import Ref
from conductor.returns import Result
from conductor.widgets import Number as NumberWidget
from conductor.widgets import Textarea


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

    def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("")) -> Out:
        return text


class Record(NodeDefinition):
    id = "record"
    title = "Record"
    description = "Returns its label"
    category = "test"

    def run(self, label: Annotated[Txt, Textarea(title="Label")] = Txt("hit")) -> Out:
        return label


class AlwaysFail(NodeDefinition):
    id = "always-fail"
    title = "Always fail"
    description = "Fails with the given reason"
    category = "test"

    def run(self, reason: Annotated[Txt, Textarea(title="Why")] = Txt("boom")) -> Out:
        raise NodeExecutionError(reason, node_id="always_fail")


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
        value: Annotated[Num, NumberWidget(title="Value")] = Num(0),
        threshold: Annotated[Num, NumberWidget(title="Threshold")] = Num(50),
    ) -> Branches:
        if value > threshold:
            return Branches(high=value, low=SKIPPED)
        return Branches(high=SKIPPED, low=value)


def _registry(*extra: type[NodeDefinition]) -> NodeRegistry:
    reg = NodeRegistry()
    for node_cls in (Echo, Record, AlwaysFail, Decide, *extra):
        reg.register(node_cls)
    return reg


async def _events(compiled) -> list[dict]:
    """Every event of one run, including those after a ``flow_error``."""
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
        compiled = compile(Flow(nodes=[
                GraphNode("d", "decide", 1, bindings={"value": Static(value=100)}),
                GraphNode("a", "record", 1, bindings={"label": Static(value="A"), "_": Sources(refs=(Ref('d', 'high'),))}),
                GraphNode("b", "always-fail", 1, bindings={"_": Sources(refs=(Ref('d', 'low'),))}),
            ]), _registry())
        r = execute_sync(compiled)
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
                    raise NodeExecutionError("transient", node_id="flaky")
                return Txt("ok")

        compiled = compile(Flow(nodes=[GraphNode("n1", "flaky", 1)]), _registry(Flaky))
        r = execute_sync(compiled)
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
        text: Annotated[Txt, Textarea(title="Text")] = Txt(""),
        expect: Annotated[Txt, Textarea(title="Expect")] = Txt("data"),
    ) -> Match:
        if text == expect:
            return Match(match=text, other=SKIPPED)
        return Match(match=SKIPPED, other=text)


class TestEdgeCases:
    """Shapes that once caught bugs."""

    def test_decision_routes_only_the_taken_branch(self):
        compiled = compile(Flow(nodes=[
                GraphNode("d", "decide", 1, bindings={"value": Static(value=100)}),
                GraphNode("taken", "echo", 1, bindings={"text": Static(value="TAKEN"), "_": Sources(refs=(Ref('d', 'high'),))}),
                GraphNode("other", "echo", 1, bindings={"text": Static(value="OTHER"), "_": Sources(refs=(Ref('d', 'low'),))}),
            ]), _registry())
        r = execute_sync(compiled)
        assert "taken" in r
        assert "other" not in r

    def test_skip_propagates_through_decision_else_branch(self):
        """A deciding node fed by a wire routes the taken branch; the else branch is skipped."""

        compiled = compile(Flow(nodes=[
                GraphNode("source", "echo", 1, bindings={"text": Static(value="data")}),
                GraphNode("d", "route", 1, bindings={"text": Sources(refs=(Ref('source', 'result'),))}),
                GraphNode("taken", "echo", 1, bindings={"text": Static(value="TAKEN"), "_": Sources(refs=(Ref('d', 'match'),))}),
                GraphNode("else_b", "echo", 1, bindings={"text": Static(value="ELSE"), "_": Sources(refs=(Ref('d', 'other'),))}),
            ]), _registry(Route))
        r = execute_sync(compiled)
        assert r["taken"]["result"] == "TAKEN"
        assert "else_b" not in r
