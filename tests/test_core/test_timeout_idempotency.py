"""Tests for per-node timeout and idempotency_key."""

from __future__ import annotations

import time
from typing import Annotated

import pytest
from conductor import (
    CompilationError,
    FlowExecutionError,
    GraphNode,
    NodeRegistry,
    compile,
    execute_sync,
)
from conductor.widgets import Output, Text

# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------


def test_per_node_timeout_triggers() -> None:
    reg = NodeRegistry()

    @reg.node("slow", version=1, name="Slow", description="x", timeout=0.2)
    def slow() -> Annotated[str, Output(label="r")]:
        time.sleep(1.0)
        return "done"

    compiled = compile(
        nodes=[GraphNode("n1", "slow@1", {})], edges=[], registry=reg,
    )
    with pytest.raises(FlowExecutionError):
        execute_sync(compiled)


def test_iso8601_timeout_parses() -> None:
    reg = NodeRegistry()

    @reg.node("s", version=1, name="S", description="x", timeout="PT0.1S")
    def s() -> Annotated[str, Output(label="r")]:
        time.sleep(0.5)
        return "ok"

    compiled = compile(
        nodes=[GraphNode("n1", "s@1", {})], edges=[], registry=reg,
    )
    with pytest.raises(FlowExecutionError):
        execute_sync(compiled)


def test_shorthand_duration_parses() -> None:
    reg = NodeRegistry()

    @reg.node("s", version=1, name="S", description="x", timeout="200ms")
    def s() -> Annotated[str, Output(label="r")]:
        time.sleep(1.0)
        return "ok"

    compiled = compile(
        nodes=[GraphNode("n1", "s@1", {})], edges=[], registry=reg,
    )
    with pytest.raises(FlowExecutionError):
        execute_sync(compiled)


def test_invalid_timeout_raises() -> None:
    reg = NodeRegistry()
    with pytest.raises(ValueError):

        @reg.node("s", version=1, name="S", description="x", timeout=-1)
        def s() -> Annotated[str, Output(label="r")]:
            return "ok"


# ---------------------------------------------------------------------------
# Idempotency key
# ---------------------------------------------------------------------------


def test_idempotency_key_evaluated() -> None:
    reg = NodeRegistry()

    @reg.node(
        "charge", version=1, name="Charge", description="x",
        idempotency_key='"charge-" + string(amount)',
    )
    def charge(
        amount: Annotated[int, Text(label="amt")] = 100,
        idempotency_key: str = None,
    ) -> Annotated[str, Output(label="r")]:
        return f"key={idempotency_key}"

    compiled = compile(
        nodes=[GraphNode("n1", "charge@1", {"amount": 42})],
        edges=[], registry=reg,
    )
    r = execute_sync(compiled)
    assert r["n1"]["result"] == "key=charge-42"


def test_idempotency_key_stable_across_retries() -> None:
    reg = NodeRegistry()
    seen: list[str] = []

    @reg.node(
        "flaky", version=1, name="F", description="x",
        idempotency_key='"op-" + string(amount)',
        max_retries=2, retry_delay=0.01,
    )
    def flaky(
        amount: Annotated[int, Text(label="amt")] = 1,
        idempotency_key: str = None,
    ) -> Annotated[str, Output(label="r")]:
        seen.append(idempotency_key or "")
        if len(seen) < 2:
            from conductor.errors import NodeExecutionError
            raise NodeExecutionError("try again", node_id="flaky")
        return "ok"

    compiled = compile(
        nodes=[GraphNode("n1", "flaky@1", {"amount": 7})],
        edges=[], registry=reg,
    )
    execute_sync(compiled)
    assert seen == ["op-7", "op-7"]


def test_invalid_idempotency_cel_at_compile() -> None:
    reg = NodeRegistry()

    @reg.node(
        "c", version=1, name="C", description="x",
        idempotency_key="not valid >>",
    )
    def c() -> Annotated[str, Output(label="r")]:
        return "ok"

    with pytest.raises(CompilationError):
        compile(
            nodes=[GraphNode("n1", "c@1", {})], edges=[], registry=reg,
        )


def test_idempotency_key_in_node_start_event() -> None:
    import asyncio

    from conductor import execute

    reg = NodeRegistry()

    @reg.node(
        "n", version=1, name="N", description="x",
        idempotency_key='"ik-" + string(v)',
    )
    def n(v: Annotated[int, Text(label="v")] = 1) -> Annotated[str, Output(label="r")]:
        return "ok"

    compiled = compile(
        nodes=[GraphNode("n1", "n@1", {"v": 99})],
        edges=[], registry=reg,
    )

    async def go():
        events = []
        async for ev in execute(compiled):
            events.append(ev)
        return events

    events = asyncio.run(go())
    start = [e for e in events if e.get("type") == "node_start"][0]
    assert start["idempotency_key"] == "ik-99"
