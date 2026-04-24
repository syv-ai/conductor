"""Tests for the signal / event node."""

from __future__ import annotations

import asyncio
from typing import Annotated

from conductor import (
    GraphNode,
    NodeRegistry,
    SignalRequired,
    compile,
    execute,
    resume_sync,
)
from conductor.widgets import Output, Text


def _registry() -> NodeRegistry:
    reg = NodeRegistry()

    @reg.node(
        "signal-wait", version=1, name="Wait", description="x",
        is_signal=True,
    )
    def signal_wait(
        signal_name: Annotated[str, Text(label="n")] = "",
        correlation: Annotated[str, Text(label="c")] = "",
        timeout_seconds: Annotated[float, Text(label="t")] = 0,
    ) -> Annotated[object, Output(label="p")]:
        raise SignalRequired(
            signal_name or "default",
            correlation=correlation or None,
            timeout_seconds=timeout_seconds or None,
        )

    return reg


def test_signal_pauses_and_resumes() -> None:
    reg = _registry()
    compiled = compile(
        nodes=[GraphNode("s1", "signal-wait@1",
                         {"signal_name": "payment", "correlation": "x == 1"})],
        edges=[], registry=reg,
    )

    async def go():
        ckpt = None
        events = []
        async for ev in execute(compiled):
            events.append(ev)
            if ev["type"] == "flow_paused":
                ckpt = ev["checkpoint"]
        return events, ckpt

    events, ckpt = asyncio.run(go())
    assert any(ev["type"] == "signal_waiting" for ev in events)
    assert any(ev["type"] == "flow_paused" for ev in events)
    assert ckpt["signal_name"] == "payment"
    assert ckpt["correlation"] == "x == 1"

    resumed = resume_sync(compiled, ckpt, {"ok": True})
    assert resumed["s1"]["ok"] is True


def test_checkpoint_matches_signal() -> None:
    """`FlowCheckpoint.matches_signal` is the host-side correlation helper."""
    from conductor import FlowCheckpoint

    cp = FlowCheckpoint(
        completed_node_ids=[],
        waiting_node_id="s1",
        waiting_node_type="signal-wait@1",
        results={},
        store_data={},
        context={},
        prompt="waiting",
        input_schema=None,
        execution_index=-1,
        skipped_edges=[],
        signal_name="payment_received",
        correlation="invoice_id == 42",
        signal_timeout_seconds=None,
    )

    # Wrong name → no match
    assert cp.matches_signal("other", {"invoice_id": 42}) is False
    # Right name + matching correlation
    assert cp.matches_signal("payment_received", {"invoice_id": 42}) is True
    # Right name + non-matching correlation
    assert cp.matches_signal("payment_received", {"invoice_id": 7}) is False
    # Bogus correlation → treated as "no match"
    cp2 = FlowCheckpoint(
        completed_node_ids=[], waiting_node_id="s",
        waiting_node_type="signal-wait@1", results={}, store_data={},
        context={}, prompt="", input_schema=None, execution_index=-1,
        skipped_edges=[], signal_name="x", correlation=">",
    )
    assert cp2.matches_signal("x", {}) is False
    # No correlation → always matches by name
    cp3 = FlowCheckpoint(
        completed_node_ids=[], waiting_node_id="s",
        waiting_node_type="signal-wait@1", results={}, store_data={},
        context={}, prompt="", input_schema=None, execution_index=-1,
        skipped_edges=[], signal_name="anything",
    )
    assert cp3.matches_signal("anything", None) is True


def test_signal_timeout_surfaced() -> None:
    reg = _registry()
    compiled = compile(
        nodes=[GraphNode("s1", "signal-wait@1",
                         {"signal_name": "x", "timeout_seconds": 10.0})],
        edges=[], registry=reg,
    )

    async def go():
        async for ev in execute(compiled):
            if ev["type"] == "signal_waiting":
                return ev
        return None

    ev = asyncio.run(go())
    assert ev is not None
    assert ev["timeout_seconds"] == 10.0
