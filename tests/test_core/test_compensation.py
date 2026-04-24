"""Tests for compensation / saga support."""

from __future__ import annotations

import asyncio
from typing import Annotated

import pytest
from conductor import (
    CompilationError,
    FlowExecutionError,
    GraphEdge,
    GraphNode,
    NodeRegistry,
    compile,
    execute,
    execute_sync,
)
from conductor.errors import NodeExecutionError
from conductor.widgets import Output


def _registry(log: list[str]) -> NodeRegistry:
    reg = NodeRegistry()

    @reg.node("charge", version=1, name="Charge", description="x")
    def charge() -> Annotated[str, Output(label="cid")]:
        log.append("charge")
        return "ch_1"

    @reg.node("refund", version=1, name="Refund", description="x")
    def refund(
        target_node_id: str = None,
        original_inputs: dict = None,
        original_output: dict = None,
    ) -> Annotated[str, Output(label="rid")]:
        log.append(f"refund:{target_node_id}")
        return "rf_1"

    @reg.node("save_ok", version=1, name="Save Ok", description="x")
    def save_ok() -> Annotated[str, Output(label="sid")]:
        log.append("save_ok")
        return "ok"

    @reg.node("save_fail", version=1, name="Save Fail", description="x")
    def save_fail() -> Annotated[str, Output(label="sid")]:
        log.append("save_fail")
        raise NodeExecutionError("db down", node_id="save_fail")

    @reg.node("unsave", version=1, name="Unsave", description="x")
    def unsave(
        target_node_id: str = None,
    ) -> Annotated[str, Output(label="uid")]:
        log.append(f"unsave:{target_node_id}")
        return "undone"

    return reg


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_missing_compensation_target_raises() -> None:
    reg = _registry([])
    with pytest.raises(CompilationError, match="no such node"):
        compile(
            nodes=[GraphNode("n1", "charge@1", {}, compensation="ghost")],
            edges=[], registry=reg,
        )


def test_self_compensation_raises() -> None:
    reg = _registry([])
    with pytest.raises(CompilationError, match="own compensation"):
        compile(
            nodes=[GraphNode("n1", "charge@1", {}, compensation="n1")],
            edges=[], registry=reg,
        )


def test_invalid_on_error_policy() -> None:
    reg = _registry([])
    with pytest.raises(CompilationError, match="invalid on_error"):
        compile(
            nodes=[GraphNode("n1", "charge@1", {}, on_error="bogus")],
            edges=[], registry=reg,
        )


# ---------------------------------------------------------------------------
# Runtime cascade
# ---------------------------------------------------------------------------


def test_cascade_runs_in_reverse() -> None:
    log: list[str] = []
    reg = _registry(log)
    compiled = compile(
        nodes=[
            GraphNode("n1", "charge@1", {}, compensation="c1"),
            GraphNode("c1", "refund@1", {}),
            GraphNode("n2", "save_fail@1", {}),
        ],
        edges=[GraphEdge("e1", "n1", "n2", "result", "_")],
        registry=reg,
    )
    with pytest.raises(FlowExecutionError):
        execute_sync(compiled)
    # charge ran, save_fail ran, refund ran in compensation for n1
    assert "charge" in log
    assert "save_fail" in log
    assert "refund:n1" in log


def test_compensation_nodes_not_run_as_regular_nodes() -> None:
    """A node that is `compensation=` target should not run on the happy path."""
    log: list[str] = []
    reg = _registry(log)
    compiled = compile(
        nodes=[
            GraphNode("n1", "charge@1", {}, compensation="c1"),
            GraphNode("c1", "refund@1", {}),
        ],
        edges=[],
        registry=reg,
    )
    r = execute_sync(compiled)
    assert log == ["charge"]
    assert "n1" in r
    assert "c1" not in r


def test_on_error_continue_skips_compensation() -> None:
    log: list[str] = []
    reg = _registry(log)

    @reg.node("x", version=1, name="X", description="x")
    def x() -> Annotated[str, Output(label="r")]:
        raise NodeExecutionError("oops", node_id="x")

    compiled = compile(
        nodes=[
            GraphNode("n1", "x@1", {}, on_error="continue"),
            GraphNode("n2", "charge@1", {}),
        ],
        edges=[GraphEdge("e1", "n1", "n2", "result", "_")],
        registry=reg,
    )
    execute_sync(compiled)
    # n2 still ran even though n1 failed
    assert "charge" in log


def test_cascade_continues_when_a_compensation_fails() -> None:
    """A compensation that itself raises must not abort the cascade.

    The cascade is documented as best-effort: failing compensations should
    emit a `compensation_failed` event but subsequent compensations still run.
    """
    log: list[str] = []
    reg = _registry(log)

    @reg.node("bad_refund", version=1, name="Bad Refund", description="x")
    def bad_refund(
        target_node_id: str = None,
        original_inputs: dict = None,
        original_output: dict = None,
    ) -> Annotated[str, Output(label="r")]:
        log.append(f"bad_refund:{target_node_id}")
        raise NodeExecutionError("refund gateway down", node_id="bad_refund")

    compiled = compile(
        nodes=[
            # n1 completes, its compensation (bad) will fail.
            GraphNode("n1", "charge@1", {}, compensation="bad"),
            GraphNode("bad", "bad_refund@1", {}),
            # n2 completes, its compensation (good) must still run.
            GraphNode("n2", "charge@1", {}, compensation="good"),
            GraphNode("good", "refund@1", {}),
            # n3 fails, triggering the cascade. Cascade runs in reverse
            # completed order, so n2's compensation runs, then n1's.
            GraphNode("n3", "save_fail@1", {}),
        ],
        edges=[
            GraphEdge("e1", "n1", "n2", "result", "_"),
            GraphEdge("e2", "n2", "n3", "result", "_"),
        ],
        registry=reg,
    )

    async def go():
        evs = []
        try:
            async for ev in execute(compiled):
                evs.append(ev)
        except Exception:
            pass
        return evs

    evs = asyncio.run(go())
    kinds = [e["type"] for e in evs]

    # The failing compensation must emit compensation_failed but not halt the cascade.
    assert "compensation_failed" in kinds
    # The good compensation still ran after the bad one failed.
    assert "refund:n2" in log
    # Both compensation attempts happened.
    assert "bad_refund:n1" in log


def test_cascade_events_emitted() -> None:
    log: list[str] = []
    reg = _registry(log)
    compiled = compile(
        nodes=[
            GraphNode("n1", "charge@1", {}, compensation="c1"),
            GraphNode("c1", "refund@1", {}),
            GraphNode("n2", "save_fail@1", {}),
        ],
        edges=[GraphEdge("e1", "n1", "n2", "result", "_")],
        registry=reg,
    )

    async def go():
        evs = []
        async for ev in execute(compiled):
            evs.append(ev)
        return evs

    evs = asyncio.run(go())
    kinds = [e["type"] for e in evs]
    assert "compensation_start" in kinds
    assert "compensation_complete" in kinds
