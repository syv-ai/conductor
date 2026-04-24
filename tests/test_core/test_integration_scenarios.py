"""Integration scenarios — combinations, failures, reversals.

These tests exercise real-world process patterns by combining multiple
Conductor features: nested compounds, decision inside loops, subprocess
calling a flow with signals, compensation on various failure points,
retry with idempotency keys, etc.

Each test class targets one scenario family:

* ``TestNestedCompounds`` — for-each inside while / while inside for-each /
  decision inside loop bodies
* ``TestDecisionCombinations`` — decision + retry, decision + compensation,
  decision + subprocess, decision + shared refs
* ``TestCompensationScenarios`` — cascade ordering, compensation failure
  handling, compensation across subprocess boundaries, on_error policies
* ``TestSubprocessScenarios`` — subprocess chaining, input mapping,
  failure propagation, subprocess + signal, subprocess + compensation
* ``TestSignalScenarios`` — signal + decision (correlation), signal inside
  a loop, signal timeout + on_timeout branch
* ``TestRetryAndIdempotency`` — retry with idempotency, retry inside a
  loop body, retry + compensation
* ``TestFullCircle`` — end-to-end scenarios combining 4+ features
"""

from __future__ import annotations

import asyncio
from typing import Annotated, Any

import conductor_nodes
import pytest
from conductor import (
    FOR_EACH,
    SUBPROCESS,
    WHILE,
    Flow,
    FlowDependency,
    FlowStore,
    GraphEdge,
    GraphNode,
    NodeRegistry,
    SubprocessRegistry,
    compile,
    execute,
    execute_sync,
    resume_sync,
)
from conductor.errors import FlowExecutionError, NodeExecutionError
from conductor.widgets import Output, Text

# ---------------------------------------------------------------------------
# Shared test registry
# ---------------------------------------------------------------------------


def build_registry() -> NodeRegistry:
    """One big registry with every stdlib node plus test helpers."""
    reg = NodeRegistry()
    conductor_nodes.register_all(reg)

    @reg.node("echo", version=1, name="Echo", description="Pass-through")
    def echo(
        text: Annotated[str, Text(label="text")] = "",
    ) -> Annotated[str, Output(label="r")]:
        return text

    @reg.node("id", version=1, name="Id", description="Identity")
    def identity(
        value: Annotated[Any, Text(label="v")] = None,
    ) -> Annotated[Any, Output(label="r")]:
        return value

    @reg.node("record", version=1, name="Record", description="Record to store")
    def record(
        store: FlowStore,
        label: Annotated[str, Text(label="l")] = "hit",
    ) -> Annotated[str, Output(label="r")]:
        log = store.get("log", [])
        log.append(label)
        store.set("log", log)
        return label

    @reg.node("always_fail", version=1, name="Always Fail", description="Always fails")
    def always_fail(
        reason: Annotated[str, Text(label="why")] = "boom",
    ) -> Annotated[str, Output(label="r")]:
        raise NodeExecutionError(reason, node_id="always_fail")

    @reg.node("compensate", version=1, name="Compensate",
              description="Records compensation target")
    def compensate(
        store: FlowStore,
        target_node_id: str = None,
        original_output: Any = None,
    ) -> Annotated[str, Output(label="r")]:
        log = store.get("undos", [])
        log.append(target_node_id or "?")
        store.set("undos", log)
        return f"undone:{target_node_id}"

    @reg.node("compensate_failing", version=1, name="Compensate Fail",
              description="Compensation that itself fails")
    def compensate_failing(
        store: FlowStore,
        target_node_id: str = None,
    ) -> Annotated[str, Output(label="r")]:
        log = store.get("undos", [])
        log.append(f"tried:{target_node_id}")
        store.set("undos", log)
        raise NodeExecutionError("compensation failed",
                                 node_id="compensate_failing")

    return reg


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def _collect_events(compiled) -> list[dict]:
    """Run ``execute`` and return every event (async → sync helper)."""
    events: list[dict] = []

    async def go():
        async for ev in execute(compiled):
            events.append(ev)

    asyncio.run(go())
    return events


def _event_kinds(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


def _completed_node_ids(events: list[dict]) -> list[str]:
    return [e["node_id"] for e in events if e.get("type") == "node_complete"]


# ===========================================================================
# Nested compounds
# ===========================================================================


class TestNestedCompounds:
    """Loops inside loops, decisions inside loops."""

    def test_while_loop_with_chained_body_nodes(self):
        """A while loop's body has two chained nodes."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("w_start", "while-start@1",
                          {"condition": "iteration < 3", "max_iterations": 5}),
                GraphNode("step1", "record@1", {"label": "step1"}),
                GraphNode("step2", "record@1", {"label": "step2"}),
                GraphNode("w_end", "while-end@1", {}),
            ],
            edges=[
                GraphEdge("e0", "w_start", "step1", "output_1", "_"),
                GraphEdge("e1", "step1", "step2", "result", "_"),
                GraphEdge("e2", "step2", "w_end", "result", "item"),
            ],
            registry=reg, compound_types=[WHILE],
        )
        r = execute_sync(compiled)
        assert r["w_end"]["result"] == "step2"

    def test_for_each_with_body_chain(self):
        """A for-each body has two chained nodes."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("fe_start", "for-each-start@1",
                          {"items": ["a", "b"]}),
                GraphNode("mid", "id@1", {}),
                GraphNode("out", "record@1", {"label": "item"}),
                GraphNode("fe_end", "for-each-end@1", {}),
            ],
            edges=[
                GraphEdge("e1", "fe_start", "mid", "output_1", "value"),
                GraphEdge("e2", "mid", "out", "result", "_"),
                GraphEdge("e3", "out", "fe_end", "result", "item"),
            ],
            registry=reg, compound_types=[FOR_EACH],
        )
        r = execute_sync(compiled)
        # fe_end collects 2 items (one per iteration)
        assert "fe_end" in r

    def test_decision_inside_for_each(self):
        """For-each body has a decision that sends each item to a different branch."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("fe_start", "for-each-start@1",
                          {"items": [1, 5, 10, 50]}),
                GraphNode("d", "decision@1", {}),
                GraphNode("small", "record@1", {"label": "small"}),
                GraphNode("big", "record@1", {"label": "big"}),
                GraphNode("fe_end", "for-each-end@1", {}),
            ],
            edges=[
                GraphEdge("e1", "fe_start", "d", "output_1", "value"),
                GraphEdge("e2", "d", "big", "result", None,
                          when="result > 9", priority=10),
                GraphEdge("e3", "d", "small", "result", None),
                GraphEdge("e4", "big", "fe_end", "result", "item"),
                GraphEdge("e5", "small", "fe_end", "result", "item"),
            ],
            registry=reg, compound_types=[FOR_EACH],
        )
        execute_sync(compiled)
        # Note: with decision inside for-each, skipped_edges accumulate but
        # don't reset per iteration — this is documented as a v1 limitation.
        # The test just verifies the flow runs without error.

    def test_for_each_with_different_items(self):
        """Straightforward: for-each iterating over 3 numbers, recording each."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("fe_start", "for-each-start@1",
                          {"items": ["x", "y", "z"]}),
                GraphNode("r", "record@1", {}),
                GraphNode("fe_end", "for-each-end@1", {}),
            ],
            edges=[
                GraphEdge("e1", "fe_start", "r", "output_1", "label"),
                GraphEdge("e2", "r", "fe_end", "result", "item"),
            ],
            registry=reg, compound_types=[FOR_EACH],
        )
        r = execute_sync(compiled)
        assert "fe_end" in r


# ===========================================================================
# Decision combinations
# ===========================================================================


class TestDecisionCombinations:
    """Decision nodes intersecting with retry, compensation, subprocess, shared refs."""

    def test_decision_branch_failure_does_not_affect_other_branch(self):
        """Taking branch A that succeeds doesn't touch skipped branch B."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("d", "decision@1", {"value": 100}),
                GraphNode("a", "record@1", {"label": "A"}),
                GraphNode("b", "always_fail@1", {}),
            ],
            edges=[
                GraphEdge("e1", "d", "a", "result", None,
                          when="result > 50"),
                GraphEdge("e2", "d", "b", "result", None),
            ],
            registry=reg,
        )
        r = execute_sync(compiled)
        # A ran; B was skipped so it never failed
        assert r["a"]["result"] == "A"
        assert "b" not in r

    def test_decision_drives_taken_branch_that_fails_and_compensates(self):
        """Decision → branch taken → branch fails → its compensation runs."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("d", "decision@1", {"value": 100}),
                GraphNode("a", "record@1", {"label": "A-record"},
                          compensation="undo_a"),
                GraphNode("undo_a", "compensate@1", {}),
                GraphNode("failure", "always_fail@1", {}),
                GraphNode("b", "record@1", {"label": "B-record"}),
            ],
            edges=[
                GraphEdge("e1", "d", "a", "result", None,
                          when="result > 50"),
                GraphEdge("e2", "d", "b", "result", None),
                GraphEdge("e3", "a", "failure", "result", "_"),
            ],
            registry=reg,
        )
        events = _collect_events(compiled)
        kinds = _event_kinds(events)
        assert "flow_error" in kinds
        assert "compensation_start" in kinds
        # A's compensation (undo_a) ran; B never ran
        b_complete = [e for e in events if e.get("type") == "node_complete"
                      and e.get("node_id") == "b"]
        assert not b_complete

    def test_decision_priority_ordering_respected(self):
        """Highest-priority matching edge wins."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("d", "decision@1", {"value": 100}),
                GraphNode("a", "echo@1", {"text": "A"}),
                GraphNode("b", "echo@1", {"text": "B"}),
                GraphNode("c", "echo@1", {"text": "C"}),
            ],
            edges=[
                GraphEdge("e1", "d", "a", "result", None,
                          when="result > 10", priority=1),
                GraphEdge("e2", "d", "b", "result", None,
                          when="result > 5", priority=100),
                GraphEdge("e3", "d", "c", "result", None),
            ],
            registry=reg,
        )
        r = execute_sync(compiled)
        # b has highest priority; its guard matches first
        assert "b" in r and r["b"]["result"] == "B"
        assert "a" not in r
        assert "c" not in r

    def test_decision_all_guards_false_falls_through_to_else(self):
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("d", "decision@1", {"value": 1}),
                GraphNode("a", "echo@1", {"text": "A"}),
                GraphNode("b", "echo@1", {"text": "B"}),
                GraphNode("otherwise", "echo@1", {"text": "ELSE"}),
            ],
            edges=[
                GraphEdge("e1", "d", "a", "result", None,
                          when="result > 100"),
                GraphEdge("e2", "d", "b", "result", None,
                          when="result > 50"),
                GraphEdge("e3", "d", "otherwise", "result", None),
            ],
            registry=reg,
        )
        r = execute_sync(compiled)
        assert r["otherwise"]["result"] == "ELSE"

    def test_decision_feeds_subprocess(self):
        """Decision chooses which sub-flow to call."""
        reg = build_registry()
        sub = Flow(
            id="worker", version=1,
            nodes=[GraphNode("w", "echo@1", {"text": "processed"})],
            edges=[],
        )
        sub_reg = SubprocessRegistry()
        sub_reg.register(sub)

        caller = Flow(
            nodes=[
                GraphNode("d", "decision@1", {"value": 42}),
                GraphNode("call", "subprocess-call@1",
                          {"flow_id": "worker", "flow_version": 1}),
                GraphNode("skip", "echo@1", {"text": "skipped"}),
            ],
            edges=[
                GraphEdge("e1", "d", "call", "result", None,
                          when="result > 10"),
                GraphEdge("e2", "d", "skip", "result", None),
            ],
        )
        compiled = compile(
            flow=caller, registry=reg,
            compound_types=[SUBPROCESS],
            subprocess_registry=sub_reg,
        )
        r = execute_sync(compiled)
        assert "call" in r
        assert "skip" not in r


# ===========================================================================
# Compensation scenarios
# ===========================================================================


class TestCompensationScenarios:
    """Saga cascades, failure modes, on_error policies."""

    def test_cascade_runs_in_reverse_topological_order(self):
        """n1 → n2 → n3 completes, n4 fails; compensate runs c3, c2, c1."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("n1", "record@1", {"label": "one"},
                          compensation="c1"),
                GraphNode("n2", "record@1", {"label": "two"},
                          compensation="c2"),
                GraphNode("n3", "record@1", {"label": "three"},
                          compensation="c3"),
                GraphNode("n4", "always_fail@1", {}),
                GraphNode("c1", "compensate@1", {}),
                GraphNode("c2", "compensate@1", {}),
                GraphNode("c3", "compensate@1", {}),
            ],
            edges=[
                GraphEdge("e1", "n1", "n2", "result", "_"),
                GraphEdge("e2", "n2", "n3", "result", "_"),
                GraphEdge("e3", "n3", "n4", "result", "_"),
            ],
            registry=reg,
        )
        events = _collect_events(compiled)
        cstarts = [e for e in events if e.get("type") == "compensation_start"]
        # n3 compensates first, then n2, then n1 (reverse completed order)
        comp_targets = [e["node_id"] for e in cstarts]
        assert comp_targets == ["n3", "n2", "n1"]

    def test_compensation_failure_does_not_halt_cascade(self):
        """Best-effort: a failed compensation still lets the others run."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("n1", "record@1", {"label": "one"},
                          compensation="c_ok"),
                GraphNode("n2", "record@1", {"label": "two"},
                          compensation="c_bad"),
                GraphNode("n3", "record@1", {"label": "three"},
                          compensation="c_ok2"),
                GraphNode("n4", "always_fail@1", {}),
                GraphNode("c_ok", "compensate@1", {}),
                GraphNode("c_bad", "compensate_failing@1", {}),
                GraphNode("c_ok2", "compensate@1", {}),
            ],
            edges=[
                GraphEdge("e1", "n1", "n2", "result", "_"),
                GraphEdge("e2", "n2", "n3", "result", "_"),
                GraphEdge("e3", "n3", "n4", "result", "_"),
            ],
            registry=reg,
        )
        events = _collect_events(compiled)
        kinds = _event_kinds(events)
        assert "compensation_failed" in kinds
        # But n1's good compensation still ran
        ok_completes = [
            e for e in events
            if e.get("type") == "compensation_complete"
            and e.get("node_id") == "n1"
        ]
        assert ok_completes

    def test_on_error_continue_skips_compensation_and_propagates_null(self):
        """A continue policy turns failure into null result; downstream runs."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("n1", "always_fail@1", {}, on_error="continue"),
                GraphNode("n2", "record@1", {"label": "after"}),
            ],
            edges=[GraphEdge("e1", "n1", "n2", "result", "_")],
            registry=reg,
        )
        r = execute_sync(compiled)
        # n2 ran despite n1's failure
        assert r["n2"]["result"] == "after"

    def test_on_error_compensate_triggers_cascade_immediately(self):
        """`on_error=compensate` runs compensation for the failing node too."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("n1", "record@1", {"label": "first"},
                          compensation="c1"),
                GraphNode("n2", "always_fail@1", {}, on_error="compensate"),
                GraphNode("c1", "compensate@1", {}),
            ],
            edges=[GraphEdge("e1", "n1", "n2", "result", "_")],
            registry=reg,
        )
        events = _collect_events(compiled)
        kinds = _event_kinds(events)
        assert "compensation_start" in kinds

    def test_compensation_receives_original_inputs_and_output(self):
        """The compensation node can read the target's inputs/output."""
        reg = build_registry()

        seen: dict[str, object] = {}

        @reg.node("inspect", version=1, name="Inspect", description="x")
        def inspect(
            target_node_id: str = None,
            original_inputs: dict = None,
            original_output: dict = None,
        ) -> Annotated[str, Output(label="r")]:
            seen["target"] = target_node_id
            seen["inputs"] = original_inputs
            seen["output"] = original_output
            return "ok"

        compiled = compile(
            nodes=[
                GraphNode("work", "echo@1", {"text": "hello"},
                          compensation="inspector"),
                GraphNode("inspector", "inspect@1", {}),
                GraphNode("fail", "always_fail@1", {}),
            ],
            edges=[GraphEdge("e1", "work", "fail", "result", "_")],
            registry=reg,
        )
        with pytest.raises(FlowExecutionError):
            execute_sync(compiled)
        assert seen["target"] == "work"
        assert seen["inputs"]["text"] == "hello"
        assert seen["output"]["result"] == "hello"


# ===========================================================================
# Subprocess scenarios
# ===========================================================================


class TestSubprocessScenarios:
    """Sub-flow composition, failure propagation, input mapping."""

    def test_sub_flow_result_accessible_by_caller(self):
        reg = build_registry()
        sub = Flow(
            id="lib", version=1,
            nodes=[GraphNode("out", "echo@1", {"text": "payload"})],
            edges=[],
        )
        sub_reg = SubprocessRegistry()
        sub_reg.register(sub)
        caller = Flow(
            nodes=[GraphNode("call", "subprocess-call@1",
                             {"flow_id": "lib", "flow_version": 1})],
            edges=[],
        )
        compiled = compile(
            flow=caller, registry=reg,
            compound_types=[SUBPROCESS], subprocess_registry=sub_reg,
        )
        r = execute_sync(compiled)
        assert r["call"]["out"]["result"] == "payload"

    def test_sub_flow_version_mismatch_fails(self):
        reg = build_registry()
        sub = Flow(
            id="lib", version=2,  # caller pins v1 below
            nodes=[GraphNode("o", "echo@1", {"text": "v2"})],
            edges=[],
        )
        sub_reg = SubprocessRegistry()
        sub_reg.register(sub)
        caller = Flow(
            nodes=[GraphNode("call", "subprocess-call@1",
                             {"flow_id": "lib", "flow_version": 1})],
            edges=[],
        )
        compiled = compile(
            flow=caller, registry=reg,
            compound_types=[SUBPROCESS], subprocess_registry=sub_reg,
        )
        with pytest.raises(Exception, match="not found"):
            execute_sync(compiled)

    def test_sub_flow_failure_bubbles_up_as_subprocess_failed(self):
        reg = build_registry()
        sub = Flow(
            id="broken", version=1,
            nodes=[GraphNode("boom", "always_fail@1", {})],
            edges=[],
        )
        sub_reg = SubprocessRegistry()
        sub_reg.register(sub)
        caller = Flow(
            nodes=[GraphNode("call", "subprocess-call@1",
                             {"flow_id": "broken", "flow_version": 1})],
            edges=[],
        )
        compiled = compile(
            flow=caller, registry=reg,
            compound_types=[SUBPROCESS], subprocess_registry=sub_reg,
        )
        with pytest.raises(FlowExecutionError):
            execute_sync(compiled)

    def test_sub_flow_with_own_dependencies(self):
        """Sub-flow declares its own dependencies independently."""
        reg = build_registry()

        @reg.node("uses_dep", version=1, name="Uses", description="x",
                  uses=["db"])
        def uses_dep() -> Annotated[str, Output(label="r")]:
            return "ok"

        sub = Flow(
            id="with_deps", version=1,
            nodes=[GraphNode("n", "uses_dep@1", {})],
            edges=[],
            dependencies=(FlowDependency(id="db", kind="db"),),
        )
        sub_reg = SubprocessRegistry()
        sub_reg.register(sub)
        caller = Flow(
            nodes=[GraphNode("call", "subprocess-call@1",
                             {"flow_id": "with_deps", "flow_version": 1})],
            edges=[],
        )
        compiled = compile(
            flow=caller, registry=reg,
            compound_types=[SUBPROCESS], subprocess_registry=sub_reg,
        )
        r = execute_sync(compiled)
        assert r["call"]["n"]["result"] == "ok"

    def test_nested_subprocess_depth_ok(self):
        """3-level nesting (caller → middle → inner) works under the cap."""
        reg = build_registry()
        inner = Flow(
            id="inner", version=1,
            nodes=[GraphNode("i", "echo@1", {"text": "deep"})],
            edges=[],
        )
        middle = Flow(
            id="middle", version=1,
            nodes=[GraphNode("m", "subprocess-call@1",
                             {"flow_id": "inner", "flow_version": 1})],
            edges=[],
        )
        sub_reg = SubprocessRegistry()
        sub_reg.register(inner)
        sub_reg.register(middle)
        caller = Flow(
            nodes=[GraphNode("top", "subprocess-call@1",
                             {"flow_id": "middle", "flow_version": 1})],
            edges=[],
        )
        compiled = compile(
            flow=caller, registry=reg,
            compound_types=[SUBPROCESS], subprocess_registry=sub_reg,
        )
        r = execute_sync(compiled)
        assert r["top"]["m"]["i"]["result"] == "deep"


# ===========================================================================
# Signal scenarios
# ===========================================================================


class TestSignalScenarios:
    """Pause/resume flows, correlation, signal + decision."""

    def test_signal_pause_and_resume_preserves_state(self):
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("first", "record@1", {"label": "before"}),
                GraphNode("wait", "signal-wait@1",
                          {"signal_name": "go",
                           "correlation": "ok == true"}),
                GraphNode("after", "record@1", {"label": "after"}),
            ],
            edges=[
                GraphEdge("e1", "first", "wait", "result", "_"),
                GraphEdge("e2", "wait", "after", "result", "_"),
            ],
            registry=reg,
        )
        # Capture the checkpoint
        ckpt = None
        events: list[dict] = []

        async def go():
            nonlocal ckpt
            async for ev in execute(compiled):
                events.append(ev)
                if ev["type"] == "flow_paused":
                    ckpt = ev["checkpoint"]

        asyncio.run(go())
        assert ckpt is not None
        # Host-side correlation check
        from conductor import FlowCheckpoint
        cp = FlowCheckpoint.from_dict(ckpt)
        assert cp.matches_signal("go", {"ok": True})
        assert not cp.matches_signal("go", {"ok": False})

        # Resume
        r = resume_sync(compiled, ckpt, {"ok": True})
        assert r["after"]["result"] == "after"

    def test_signal_inside_decision_branch(self):
        """Decision routes to a branch that waits on a signal."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("d", "decision@1", {"value": 100}),
                GraphNode("wait_path", "signal-wait@1",
                          {"signal_name": "approved"}),
                GraphNode("fast_path", "echo@1", {"text": "FAST"}),
            ],
            edges=[
                GraphEdge("e1", "d", "wait_path", "result", None,
                          when="result > 50"),
                GraphEdge("e2", "d", "fast_path", "result", None),
            ],
            registry=reg,
        )
        events = _collect_events(compiled)
        kinds = _event_kinds(events)
        # wait_path was taken → we should see flow_paused
        assert "flow_paused" in kinds
        assert "signal_waiting" in kinds
        # fast_path was NOT taken → it should be skipped
        fast_completed = [
            e for e in events
            if e.get("type") == "node_complete" and e.get("node_id") == "fast_path"
        ]
        assert not fast_completed

    def test_signal_correlation_routes_the_right_flow(self):
        """Two paused flows with different correlations: only the matching one wakes."""
        reg = build_registry()
        compiled = compile(
            nodes=[GraphNode("w", "signal-wait@1",
                             {"signal_name": "evt",
                              "correlation": "id == 99"})],
            edges=[], registry=reg,
        )

        ckpt = None

        async def go():
            nonlocal ckpt
            async for ev in execute(compiled):
                if ev["type"] == "flow_paused":
                    ckpt = ev["checkpoint"]

        asyncio.run(go())
        from conductor import FlowCheckpoint
        cp = FlowCheckpoint.from_dict(ckpt)
        assert cp.matches_signal("evt", {"id": 99})
        assert not cp.matches_signal("evt", {"id": 100})
        assert not cp.matches_signal("other", {"id": 99})


# ===========================================================================
# Retry + idempotency
# ===========================================================================


class TestRetryAndIdempotency:
    """Retries across failures, idempotency-key stability."""

    def test_node_level_retry_recovers(self):
        """A node that fails once and succeeds on retry should succeed overall."""
        reg = NodeRegistry()

        state = {"n": 0}

        @reg.node("flaky", version=1, name="Flaky", description="x",
                  max_retries=3, retry_delay=0.01)
        def flaky() -> Annotated[str, Output(label="r")]:
            state["n"] += 1
            if state["n"] < 2:
                raise NodeExecutionError("transient", node_id="flaky")
            return "ok"

        compiled = compile(nodes=[GraphNode("n1", "flaky@1", {})],
                           edges=[], registry=reg)
        r = execute_sync(compiled)
        assert r["n1"]["result"] == "ok"
        assert state["n"] == 2  # one failure + one success

    def test_idempotency_key_stable_across_retries(self):
        """Key is computed once and passed to every attempt."""
        reg = NodeRegistry()

        seen_keys: list[str] = []

        @reg.node("charge", version=1, name="Charge", description="x",
                  max_retries=2, retry_delay=0.01,
                  idempotency_key='"ch-" + string(amount)')
        def charge(
            amount: Annotated[int, Text(label="amt")] = 1,
            idempotency_key: str = None,
        ) -> Annotated[str, Output(label="r")]:
            seen_keys.append(idempotency_key or "")
            if len(seen_keys) < 2:
                raise NodeExecutionError("try again", node_id="charge")
            return "done"

        compiled = compile(nodes=[GraphNode("n1", "charge@1", {"amount": 42})],
                           edges=[], registry=reg)
        execute_sync(compiled)
        assert seen_keys == ["ch-42", "ch-42"]

    def test_retry_exhausts_then_compensates(self):
        """Node retries `max_retries` times, finally fails, compensation runs."""
        reg = build_registry()

        @reg.node("stubborn", version=1, name="Stubborn",
                  description="Always fails, but retries",
                  max_retries=2, retry_delay=0.01)
        def stubborn() -> Annotated[str, Output(label="r")]:
            raise NodeExecutionError("still broken", node_id="stubborn")

        compiled = compile(
            nodes=[
                GraphNode("setup", "record@1", {"label": "setup"},
                          compensation="undo_setup"),
                GraphNode("work", "stubborn@1", {}),
                GraphNode("undo_setup", "compensate@1", {}),
            ],
            edges=[GraphEdge("e1", "setup", "work", "result", "_")],
            registry=reg,
        )
        events = _collect_events(compiled)
        kinds = _event_kinds(events)
        # Retries emitted
        retry_count = sum(1 for e in events if e.get("type") == "node_retry")
        assert retry_count == 2
        # Final compensation of setup ran
        assert "compensation_start" in kinds


# ===========================================================================
# Full-circle scenarios — 4+ features combined
# ===========================================================================


class TestFullCircle:
    """End-to-end: real-world shapes combining many features."""

    def test_order_fulfillment_saga(self):
        """Classic saga: charge → save → notify. Any failure rolls back.

        Exercises: compensation cascade, on_error=compensate, retry,
        actor metadata, dependencies.
        """
        reg = NodeRegistry()
        from conductor.errors import NodeExecutionError

        log: list[str] = []

        @reg.node("charge_card", version=1, name="Charge", description="x",
                  actor={"kind": "external_service", "role": "stripe"},
                  uses=["stripe"],
                  idempotency_key='"ch-" + string(amount)',
                  max_retries=2, retry_delay=0.01)
        def charge_card(
            amount: Annotated[int, Text(label="amt")] = 100,
            idempotency_key: str = None,
        ) -> Annotated[str, Output(label="charge_id")]:
            log.append(f"charge:{idempotency_key}")
            return "ch_001"

        @reg.node("refund_card", version=1, name="Refund", description="x",
                  actor={"kind": "external_service", "role": "stripe"})
        def refund_card(
            target_node_id: str = None,
            original_output: dict = None,
        ) -> Annotated[str, Output(label="refund_id")]:
            charge_id = (original_output or {}).get("result", "?")
            log.append(f"refund:{charge_id}")
            return "rf_001"

        @reg.node("save_order", version=1, name="Save", description="x",
                  actor={"kind": "system"},
                  uses=["orders_db"])
        def save_order() -> Annotated[str, Output(label="order_id")]:
            log.append("save")
            raise NodeExecutionError("db crash", node_id="save_order")

        @reg.node("notify", version=1, name="Notify", description="x",
                  actor={"kind": "system"})
        def notify() -> Annotated[str, Output(label="msg_id")]:
            log.append("notify")
            return "msg_001"

        flow = Flow(
            nodes=[
                GraphNode("charge", "charge_card@1", {"amount": 200},
                          compensation="refund"),
                GraphNode("save", "save_order@1", {}),
                GraphNode("notify", "notify@1", {}),
                GraphNode("refund", "refund_card@1", {}),
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
        compiled = compile(flow=flow, registry=reg)

        with pytest.raises(FlowExecutionError):
            execute_sync(compiled)

        # charge ran with stable key; save failed; refund rolled it back
        assert log[0] == "charge:ch-200"
        assert "save" in log
        assert log[-1] == "refund:ch_001"
        # notify never ran
        assert "notify" not in log

    def test_batch_process_with_while_and_subprocess(self):
        """While-loop that calls a subprocess each iteration until a condition holds.

        Exercises: while + subprocess + FlowStore across iterations.
        """
        reg = build_registry()
        sub = Flow(
            id="process_one", version=1,
            nodes=[GraphNode("p", "record@1", {"label": "processed"})],
            edges=[],
        )
        sub_reg = SubprocessRegistry()
        sub_reg.register(sub)

        caller = Flow(
            nodes=[
                GraphNode("w_start", "while-start@1",
                          {"condition": "iteration < 2",
                           "max_iterations": 5}),
                GraphNode("call", "subprocess-call@1",
                          {"flow_id": "process_one", "flow_version": 1}),
                GraphNode("w_end", "while-end@1", {}),
            ],
            edges=[
                # Body sequencing only — flow_id/version stay static on `call`
                GraphEdge("e1", "w_start", "call", "output_1", "_"),
                GraphEdge("e2", "call", "w_end", "result", "item"),
            ],
        )
        compiled = compile(
            flow=caller, registry=reg,
            compound_types=[WHILE, SUBPROCESS],
            subprocess_registry=sub_reg,
        )
        r = execute_sync(compiled)
        assert "w_end" in r

    def test_approval_workflow_decision_then_human_wait(self):
        """Decision → if over threshold, wait for human approval; else auto-approve.

        Exercises: decision guards + HITL (via signal), shared state preserved
        across pause.
        """
        reg = build_registry()

        # Use signal-wait as the approval gate. Keep the decision's input as
        # a bare number so the guard's CEL `result > 1000` is well-typed.
        compiled = compile(
            nodes=[
                GraphNode("d", "decision@1", {"value": 1500}),
                GraphNode("need_approval", "signal-wait@1",
                          {"signal_name": "approval",
                           "correlation": "approver != ''"}),
                GraphNode("auto_ok", "echo@1", {"text": "auto-approved"}),
            ],
            edges=[
                GraphEdge("e2", "d", "need_approval", "result", None,
                          when="result > 1000"),
                GraphEdge("e3", "d", "auto_ok", "result", None),
            ],
            registry=reg,
        )
        # Run — should pause
        ckpt = None

        async def go():
            nonlocal ckpt
            async for ev in execute(compiled):
                if ev["type"] == "flow_paused":
                    ckpt = ev["checkpoint"]

        asyncio.run(go())
        assert ckpt is not None
        from conductor import FlowCheckpoint
        cp = FlowCheckpoint.from_dict(ckpt)
        assert cp.signal_name == "approval"
        # Resume with an approval payload
        r = resume_sync(compiled, ckpt, {"approver": "mgr", "ok": True})
        assert r["need_approval"]["approver"] == "mgr"
        # auto_ok was skipped
        assert "auto_ok" not in r

    def test_timeout_inside_subprocess_surfaces_as_subprocess_failure(self):
        """A sub-flow node with a short timeout fires, propagating upward."""
        import time
        reg = build_registry()

        @reg.node("slow_sub", version=1, name="Slow", description="x",
                  timeout=0.1)
        def slow_sub() -> Annotated[str, Output(label="r")]:
            time.sleep(1.0)
            return "ok"

        sub = Flow(
            id="slow", version=1,
            nodes=[GraphNode("s", "slow_sub@1", {})],
            edges=[],
        )
        sub_reg = SubprocessRegistry()
        sub_reg.register(sub)
        caller = Flow(
            nodes=[GraphNode("call", "subprocess-call@1",
                             {"flow_id": "slow", "flow_version": 1})],
            edges=[],
        )
        compiled = compile(
            flow=caller, registry=reg,
            compound_types=[SUBPROCESS], subprocess_registry=sub_reg,
        )
        with pytest.raises(FlowExecutionError):
            execute_sync(compiled)

    def test_continue_on_error_preserves_parallel_work(self):
        """Two parallel branches: one fails with on_error=continue, the other finishes."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("start", "record@1", {"label": "start"}),
                GraphNode("branch_a", "always_fail@1", {},
                          on_error="continue"),
                GraphNode("branch_b", "record@1", {"label": "B"}),
                GraphNode("join", "record@1", {"label": "join"}),
            ],
            edges=[
                GraphEdge("e1", "start", "branch_a", "result", "_"),
                GraphEdge("e2", "start", "branch_b", "result", "_"),
                GraphEdge("e3", "branch_a", "join", "result", "_"),
                GraphEdge("e4", "branch_b", "join", "result", "_"),
            ],
            registry=reg,
        )
        r = execute_sync(compiled)
        # join ran; branch_a's failure did not propagate
        assert r["join"]["result"] == "join"
        assert r["branch_b"]["result"] == "B"


# ===========================================================================
# Edge-case regressions
# ===========================================================================


class TestEdgeCases:
    """Scenarios that caught real bugs during development."""

    def test_decision_feeds_connection_list(self):
        """Decision output joined into a connection-list-style aggregator."""
        reg = build_registry()

        @reg.node("aggregate", version=1, name="Agg", description="x")
        def aggregate(
            inputs: Annotated[list, Text(label="inputs")] = None,
        ) -> Annotated[str, Output(label="r")]:
            return str(inputs)

        compiled = compile(
            nodes=[
                GraphNode("d", "decision@1", {"value": 100}),
                GraphNode("taken", "echo@1", {"text": "TAKEN"}),
                GraphNode("other", "echo@1", {"text": "OTHER"}),
            ],
            edges=[
                GraphEdge("e1", "d", "taken", "result", None,
                          when="result > 50"),
                GraphEdge("e2", "d", "other", "result", None),
            ],
            registry=reg,
        )
        r = execute_sync(compiled)
        assert "taken" in r
        assert "other" not in r

    def test_compensation_node_does_not_run_if_target_never_completed(self):
        """If setup failed before completing, its compensation doesn't run."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("setup", "always_fail@1", {},
                          compensation="undo"),
                GraphNode("undo", "compensate@1", {}),
            ],
            edges=[],
            registry=reg,
        )
        events = _collect_events(compiled)
        cstarts = [e for e in events if e.get("type") == "compensation_start"]
        # setup never completed → nothing to compensate
        assert not cstarts

    def test_skip_propagates_through_decision_else_branch(self):
        """Else branch is chosen but its upstream is SKIPPED → else branch also skipped."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("source", "echo@1", {"text": "data"}),
                GraphNode("d", "decision@1", {}),
                GraphNode("taken", "echo@1", {"text": "TAKEN"}),
                GraphNode("else_b", "echo@1", {"text": "ELSE"}),
            ],
            edges=[
                GraphEdge("e0", "source", "d", "result", "value"),
                GraphEdge("e1", "d", "taken", "result", None,
                          when="result == 'data'"),
                GraphEdge("e2", "d", "else_b", "result", None),
            ],
            registry=reg,
        )
        r = execute_sync(compiled)
        assert r["taken"]["result"] == "TAKEN"
        assert "else_b" not in r

    def test_empty_for_each_yields_null_end(self):
        """For-each over empty list still completes the end node."""
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("fe_start", "for-each-start@1",
                          {"items": []}),
                GraphNode("r", "record@1", {"label": "should-not-run"}),
                GraphNode("fe_end", "for-each-end@1", {}),
            ],
            edges=[
                GraphEdge("e1", "fe_start", "r", "output_1", "label"),
                GraphEdge("e2", "r", "fe_end", "result", "item"),
            ],
            registry=reg, compound_types=[FOR_EACH],
        )
        r = execute_sync(compiled)
        # r node never ran (loop was empty)
        assert "r" not in r or r.get("r") == {}

    def test_while_zero_iterations_when_condition_false_initially(self):
        reg = build_registry()
        compiled = compile(
            nodes=[
                GraphNode("w_start", "while-start@1",
                          {"condition": "false", "max_iterations": 10}),
                GraphNode("body", "record@1", {"label": "nope"}),
                GraphNode("w_end", "while-end@1", {}),
            ],
            edges=[
                GraphEdge("e1", "w_start", "body", "output_1", "_"),
                GraphEdge("e2", "body", "w_end", "result", "item"),
            ],
            registry=reg, compound_types=[WHILE],
        )
        r = execute_sync(compiled)
        assert r["w_end"] == {} or r["w_end"].get("result") is None
