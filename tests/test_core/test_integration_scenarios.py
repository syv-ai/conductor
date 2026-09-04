"""Scenarios that combine several features of one run.

* ``TestDecisionCombinations`` — a deciding node routes by ``SKIPPED``;
  the branch taken fails and is compensated, the branch not taken never runs
* ``TestCompensationScenarios`` — cascade ordering, a failing compensation,
  what a compensation node receives, the ``on_error`` policies
* ``TestRetry`` — a version's ``Policy`` recovers a flaky node; exhausted
  retries fall through to compensation
* ``TestFullCircle`` — a saga and a parallel ``on_error="continue"``
* ``TestEdgeCases`` — shapes that once caught bugs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any

import pytest
from conductor import (
    Flow,
    FlowDependency,
    GraphEdge,
    GraphNode,
    NodeRegistry,
    compile,
    execute,
    execute_sync,
)
from conductor._sentinel import SKIPPED
from conductor.dtype import DType
from conductor.errors import FlowExecutionError, NodeExecutionError
from conductor.node import NodeDefinition, Policy, version
from conductor.returns import Result
from conductor.widgets import ConnectionList, Textarea
from conductor.widgets import Number as NumberWidget


class Txt(DType, str):
    id = "integration-scenarios-test-text"
    title = "Text"


class Num(DType, float):
    id = "integration-scenarios-test-number"
    title = "Number"


Out = Annotated[Txt, Result(title="Out")]
#: The compensated node's id, handed to a compensation node by the engine.
Target = Annotated[Txt, Textarea(title="Target node")]

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


class Compensate(NodeDefinition):
    id = "compensate"
    title = "Compensate"
    description = "Undoes its target"
    category = "test"

    def run(self, target_node_id: Target = Txt("")) -> Out:
        return Txt(f"undone:{target_node_id}")


class CompensateFailing(NodeDefinition):
    id = "compensate-failing"
    title = "Compensate, failing"
    description = "A compensation that itself fails"
    category = "test"

    def run(self, target_node_id: Target = Txt("")) -> Out:
        raise NodeExecutionError("compensation failed", node_id="compensate_failing")


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
    for node_cls in (Echo, Record, AlwaysFail, Compensate, CompensateFailing, Decide, *extra):
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
    """A deciding node's ``SKIPPED`` branch intersecting with failure and compensation."""

    def test_decision_branch_failure_does_not_affect_other_branch(self):
        """The branch not taken holds a failing node that never runs."""
        compiled = compile(
            nodes=[
                GraphNode("d", "decide", 1, {"value": 100}),
                GraphNode("a", "record", 1, {"label": "A"}),
                GraphNode("b", "always-fail", 1, {}),
            ],
            edges=[
                GraphEdge("e1", "d", "a", "high", "_"),
                GraphEdge("e2", "d", "b", "low", "_"),
            ],
            registry=_registry(),
        )
        r = execute_sync(compiled)
        # A ran; B was skipped so it never failed
        assert r["a"]["result"] == "A"
        assert "b" not in r

    async def test_decision_drives_taken_branch_that_fails_and_compensates(self):
        """Branch taken -> branch fails -> its compensation runs."""
        compiled = compile(
            nodes=[
                GraphNode("d", "decide", 1, {"value": 100}),
                GraphNode("a", "record", 1, {"label": "A-record"}, compensation="undo_a"),
                GraphNode("undo_a", "compensate", 1, {}),
                GraphNode("failure", "always-fail", 1, {}),
                GraphNode("b", "record", 1, {"label": "B-record"}),
            ],
            edges=[
                GraphEdge("e1", "d", "a", "high", "_"),
                GraphEdge("e2", "d", "b", "low", "_"),
                GraphEdge("e3", "a", "failure", "result", "_"),
            ],
            registry=_registry(),
        )
        events = await _events(compiled)
        kinds = _kinds(events)
        assert "flow_error" in kinds
        assert "compensation_start" in kinds
        # A's compensation (undo_a) ran; B never ran
        assert not _completed(events, "b")


# ===========================================================================
# Compensation scenarios
# ===========================================================================


class TestCompensationScenarios:
    """Saga cascades, failure modes, ``on_error`` policies."""

    async def test_cascade_runs_in_reverse_topological_order(self):
        """n1 -> n2 -> n3 complete, n4 fails; compensation runs c3, c2, c1."""
        compiled = compile(
            nodes=[
                GraphNode("n1", "record", 1, {"label": "one"}, compensation="c1"),
                GraphNode("n2", "record", 1, {"label": "two"}, compensation="c2"),
                GraphNode("n3", "record", 1, {"label": "three"}, compensation="c3"),
                GraphNode("n4", "always-fail", 1, {}),
                GraphNode("c1", "compensate", 1, {}),
                GraphNode("c2", "compensate", 1, {}),
                GraphNode("c3", "compensate", 1, {}),
            ],
            edges=[
                GraphEdge("e1", "n1", "n2", "result", "_"),
                GraphEdge("e2", "n2", "n3", "result", "_"),
                GraphEdge("e3", "n3", "n4", "result", "_"),
            ],
            registry=_registry(),
        )
        events = await _events(compiled)
        starts = [e for e in events if e.get("type") == "compensation_start"]
        # n3 compensates first, then n2, then n1 (reverse completed order)
        assert [e["node_id"] for e in starts] == ["n3", "n2", "n1"]

    async def test_compensation_failure_does_not_halt_cascade(self):
        """Best-effort: a failed compensation still lets the others run."""
        compiled = compile(
            nodes=[
                GraphNode("n1", "record", 1, {"label": "one"}, compensation="c_ok"),
                GraphNode("n2", "record", 1, {"label": "two"}, compensation="c_bad"),
                GraphNode("n3", "record", 1, {"label": "three"}, compensation="c_ok2"),
                GraphNode("n4", "always-fail", 1, {}),
                GraphNode("c_ok", "compensate", 1, {}),
                GraphNode("c_bad", "compensate-failing", 1, {}),
                GraphNode("c_ok2", "compensate", 1, {}),
            ],
            edges=[
                GraphEdge("e1", "n1", "n2", "result", "_"),
                GraphEdge("e2", "n2", "n3", "result", "_"),
                GraphEdge("e3", "n3", "n4", "result", "_"),
            ],
            registry=_registry(),
        )
        events = await _events(compiled)
        assert "compensation_failed" in _kinds(events)
        # But n1's good compensation still ran
        assert any(
            e.get("type") == "compensation_complete" and e.get("node_id") == "n1"
            for e in events
        )

    def test_on_error_continue_skips_compensation_and_propagates_null(self):
        """A continue policy turns the failure into a null result; downstream runs."""
        compiled = compile(
            nodes=[
                GraphNode("n1", "always-fail", 1, {}, on_error="continue"),
                GraphNode("n2", "record", 1, {"label": "after"}),
            ],
            edges=[GraphEdge("e1", "n1", "n2", "result", "_")],
            registry=_registry(),
        )
        r = execute_sync(compiled)
        # n2 ran despite n1's failure
        assert r["n2"]["result"] == "after"

    async def test_on_error_compensate_triggers_cascade_immediately(self):
        """``on_error="compensate"`` runs the cascade for the failing node too."""
        compiled = compile(
            nodes=[
                GraphNode("n1", "record", 1, {"label": "first"}, compensation="c1"),
                GraphNode("n2", "always-fail", 1, {}, on_error="compensate"),
                GraphNode("c1", "compensate", 1, {}),
            ],
            edges=[GraphEdge("e1", "n1", "n2", "result", "_")],
            registry=_registry(),
        )
        assert "compensation_start" in _kinds(await _events(compiled))

    def test_compensation_receives_original_inputs_and_output(self):
        """The compensation node can read its target's inputs and output."""
        seen: dict[str, object] = {}

        class Inspect(NodeDefinition):
            id = "inspect"
            title = "Inspect"
            description = "Records what the engine hands a compensation node"
            category = "test"

            def run(
                self,
                target_node_id: Target = Txt(""),
                original_inputs: Annotated[Any, ConnectionList(title="Original inputs")] = None,
                original_output: Annotated[Any, ConnectionList(title="Original output")] = None,
            ) -> Out:
                seen["target"] = target_node_id
                seen["inputs"] = original_inputs
                seen["output"] = original_output
                return Txt("ok")

        compiled = compile(
            nodes=[
                GraphNode("work", "echo", 1, {"text": "hello"}, compensation="inspector"),
                GraphNode("inspector", "inspect", 1, {}),
                GraphNode("fail", "always-fail", 1, {}),
            ],
            edges=[GraphEdge("e1", "work", "fail", "result", "_")],
            registry=_registry(Inspect),
        )
        with pytest.raises(FlowExecutionError):
            execute_sync(compiled)
        assert seen["target"] == "work"
        assert seen["inputs"]["text"] == "hello"
        assert seen["output"]["result"] == "hello"


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

        compiled = compile(
            nodes=[GraphNode("n1", "flaky", 1, {})], edges=[], registry=_registry(Flaky)
        )
        r = execute_sync(compiled)
        assert r["n1"]["result"] == "ok"
        assert calls == 2  # one failure + one success

    async def test_retry_exhausts_then_compensates(self):
        """A node retries ``retries`` times, finally fails, and compensation runs."""

        class Stubborn(NodeDefinition):
            id = "stubborn"
            title = "Stubborn"
            description = "Always fails, but retries"
            category = "test"

            @version(1, policy=Policy(retries=2, delay=0.01))
            def run(self) -> Out:
                raise NodeExecutionError("still broken", node_id="stubborn")

        compiled = compile(
            nodes=[
                GraphNode("setup", "record", 1, {"label": "setup"}, compensation="undo_setup"),
                GraphNode("work", "stubborn", 1, {}),
                GraphNode("undo_setup", "compensate", 1, {}),
            ],
            edges=[GraphEdge("e1", "setup", "work", "result", "_")],
            registry=_registry(Stubborn),
        )
        events = await _events(compiled)
        kinds = _kinds(events)
        # Retries emitted
        assert kinds.count("node_retry") == 2
        # Final compensation of setup ran
        assert "compensation_start" in kinds


# ===========================================================================
# Full-circle scenarios
# ===========================================================================


class TestFullCircle:
    """Shapes combining several features."""

    def test_order_fulfillment_saga(self):
        """Classic saga: charge -> save -> notify. Any failure rolls back.

        Exercises: compensation cascade, retry policy, a compensation
        reading its target's output, flow dependencies.
        """
        log: list[str] = []

        class ChargeCard(NodeDefinition):
            id = "charge-card"
            title = "Charge"
            description = "Charges the given amount"
            category = "test"

            @version(1, policy=Policy(retries=2, delay=0.01))
            def run(
                self, amount: Annotated[Num, NumberWidget(title="Amount")] = Num(100)
            ) -> Annotated[Txt, Result(title="Charge id")]:
                log.append(f"charge:{int(amount)}")
                return Txt("ch_001")

        class RefundCard(NodeDefinition):
            id = "refund-card"
            title = "Refund"
            description = "Refunds the charge it compensates"
            category = "test"

            def run(
                self,
                target_node_id: Target = Txt(""),
                original_output: Annotated[Any, ConnectionList(title="Original output")] = None,
            ) -> Annotated[Txt, Result(title="Refund id")]:
                charge_id = (original_output or {}).get("result", "?")
                log.append(f"refund:{charge_id}")
                return Txt("rf_001")

        class SaveOrder(NodeDefinition):
            id = "save-order"
            title = "Save"
            description = "Saves the order; the database is down"
            category = "test"

            def run(self) -> Annotated[Txt, Result(title="Order id")]:
                log.append("save")
                raise NodeExecutionError("db crash", node_id="save_order")

        class Notify(NodeDefinition):
            id = "notify"
            title = "Notify"
            description = "Sends the confirmation"
            category = "test"

            def run(self) -> Annotated[Txt, Result(title="Message id")]:
                log.append("notify")
                return Txt("msg_001")

        flow = Flow(
            nodes=[
                GraphNode("charge", "charge-card", 1, {"amount": 200}, compensation="refund"),
                GraphNode("save", "save-order", 1, {}),
                GraphNode("notify", "notify", 1, {}),
                GraphNode("refund", "refund-card", 1, {}),
            ],
            edges=[
                GraphEdge("e1", "charge", "save", "result", "_"),
                GraphEdge("e2", "save", "notify", "result", "_"),
            ],
            dependencies=(
                FlowDependency(id="stripe", kind="api"),
                FlowDependency(id="orders_db", kind="db"),
            ),
        )
        compiled = compile(flow=flow, registry=_registry(ChargeCard, RefundCard, SaveOrder, Notify))

        with pytest.raises(FlowExecutionError):
            execute_sync(compiled)

        # charge ran; save failed; refund rolled it back
        assert log[0] == "charge:200"
        assert "save" in log
        assert log[-1] == "refund:ch_001"
        # notify never ran
        assert "notify" not in log

    def test_continue_on_error_preserves_parallel_work(self):
        """Two parallel branches: one fails with ``on_error="continue"``, the other finishes."""
        compiled = compile(
            nodes=[
                GraphNode("start", "record", 1, {"label": "start"}),
                GraphNode("branch_a", "always-fail", 1, {}, on_error="continue"),
                GraphNode("branch_b", "record", 1, {"label": "B"}),
                GraphNode("join", "record", 1, {"label": "join"}),
            ],
            edges=[
                GraphEdge("e1", "start", "branch_a", "result", "_"),
                GraphEdge("e2", "start", "branch_b", "result", "_"),
                GraphEdge("e3", "branch_a", "join", "result", "_a"),
                GraphEdge("e4", "branch_b", "join", "result", "_b"),
            ],
            registry=_registry(),
        )
        r = execute_sync(compiled)
        # join ran; branch_a's failure did not propagate
        assert r["join"]["result"] == "join"
        assert r["branch_b"]["result"] == "B"


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
        compiled = compile(
            nodes=[
                GraphNode("d", "decide", 1, {"value": 100}),
                GraphNode("taken", "echo", 1, {"text": "TAKEN"}),
                GraphNode("other", "echo", 1, {"text": "OTHER"}),
            ],
            edges=[
                GraphEdge("e1", "d", "taken", "high", "_"),
                GraphEdge("e2", "d", "other", "low", "_"),
            ],
            registry=_registry(),
        )
        r = execute_sync(compiled)
        assert "taken" in r
        assert "other" not in r

    async def test_compensation_node_does_not_run_if_target_never_completed(self):
        """If setup failed before completing, its compensation does not run."""
        compiled = compile(
            nodes=[
                GraphNode("setup", "always-fail", 1, {}, compensation="undo"),
                GraphNode("undo", "compensate", 1, {}),
            ],
            edges=[],
            registry=_registry(),
        )
        kinds = _kinds(await _events(compiled))
        # setup never completed -> nothing to compensate
        assert "compensation_start" not in kinds

    def test_skip_propagates_through_decision_else_branch(self):
        """A deciding node fed by a wire routes the taken branch; the else branch is skipped."""

        compiled = compile(
            nodes=[
                GraphNode("source", "echo", 1, {"text": "data"}),
                GraphNode("d", "route", 1, {}),
                GraphNode("taken", "echo", 1, {"text": "TAKEN"}),
                GraphNode("else_b", "echo", 1, {"text": "ELSE"}),
            ],
            edges=[
                GraphEdge("e0", "source", "d", "result", "text"),
                GraphEdge("e1", "d", "taken", "match", "_"),
                GraphEdge("e2", "d", "else_b", "other", "_"),
            ],
            registry=_registry(Route),
        )
        r = execute_sync(compiled)
        assert r["taken"]["result"] == "TAKEN"
        assert "else_b" not in r
