"""Eager parallel scheduling and node-level retry."""

import time
from typing import Annotated

import pytest
from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor.errors import FlowExecutionException
from conductor.execution.engine import execute, execute_sync
from conductor.execution.retry import RetryConfig
from conductor.widgets import Output, Text

# ---------------------------------------------------------------------------
# Eager scheduling: independent branches run concurrently
# ---------------------------------------------------------------------------


class TestEagerScheduling:
    def test_independent_branches_run_in_parallel(self):
        """Two independent branches should overlap, not run sequentially."""
        reg = NodeRegistry()

        @reg.node("slow", version=1, name="Slow", description="Sleeps 0.3s")
        def slow(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            time.sleep(0.3)
            return text.upper()

        # Two independent branches: A->C and B->D, then C+D->E
        #   A(0.3s) -> C(0.3s) ──┐
        #                         ├──> E
        #   B(0.3s) -> D(0.3s) ──┘
        compiled = compile(
            nodes=[
                GraphNode("a", "slow@1", {"text": "hello"}),
                GraphNode("b", "slow@1", {"text": "world"}),
                GraphNode("c", "slow@1", None),
                GraphNode("d", "slow@1", None),
                GraphNode("e", "slow@1", None),
            ],
            edges=[
                GraphEdge("e1", "a", "c", "result", "text"),
                GraphEdge("e2", "b", "d", "result", "text"),
                GraphEdge("e3", "c", "e", "result", "text"),
                GraphEdge("e4", "d", "e", "result", "text"),
            ],
            registry=reg,
        )

        start = time.monotonic()
        results = execute_sync(compiled)
        elapsed = time.monotonic() - start

        # Sequential would be 5 * 0.3 = 1.5s
        # Eager: A+B parallel (0.3s), C+D parallel (0.3s), E (0.3s) = ~0.9s
        assert elapsed < 1.3, f"Took {elapsed:.2f}s — branches should run in parallel"
        assert "HELLO" in str(results["e"]["result"]) or "WORLD" in str(results["e"]["result"])

    def test_linear_chain_still_works(self):
        """A->B->C should execute correctly (no parallelism possible)."""
        reg = NodeRegistry()

        @reg.node("echo", version=1, name="Echo", description="Echo")
        def echo(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            return text

        @reg.node("upper", version=1, name="Upper", description="Upper")
        def upper(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            return text.upper()

        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "hello"}),
                GraphNode("n2", "upper@1", None),
                GraphNode("n3", "echo@1", None),
            ],
            edges=[
                GraphEdge("e1", "n1", "n2", "result", "text"),
                GraphEdge("e2", "n2", "n3", "result", "text"),
            ],
            registry=reg,
        )

        results = execute_sync(compiled)
        assert results["n3"]["result"] == "HELLO"

    async def test_events_emitted_for_parallel_nodes(self):
        """Both parallel nodes should emit start/complete events."""
        reg = NodeRegistry()

        @reg.node("echo", version=1, name="Echo", description="Echo")
        def echo(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            return text

        compiled = compile(
            nodes=[
                GraphNode("a", "echo@1", {"text": "x"}),
                GraphNode("b", "echo@1", {"text": "y"}),
            ],
            edges=[],
            registry=reg,
        )

        events = []
        async for event in execute(compiled):
            events.append(event)

        types = [e["type"] for e in events]
        assert types.count("node_start") == 2
        assert types.count("node_complete") == 2
        assert "flow_complete" in types

    def test_single_node_works(self):
        """Edge case: single node with no edges."""
        reg = NodeRegistry()

        @reg.node("echo", version=1, name="Echo", description="Echo")
        def echo(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            return text

        compiled = compile(
            nodes=[GraphNode("n1", "echo@1", {"text": "hi"})],
            edges=[],
            registry=reg,
        )

        results = execute_sync(compiled)
        assert results["n1"]["result"] == "hi"


# ---------------------------------------------------------------------------
# Retry: node-level and global
# ---------------------------------------------------------------------------


class TestRetry:
    def test_global_retry_retries_on_failure(self):
        """Global RetryConfig retries failing nodes."""
        reg = NodeRegistry()
        call_count = 0

        @reg.node("flaky", version=1, name="Flaky", description="Fails twice then succeeds")
        def flaky(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError(f"Attempt {call_count} failed")
            return f"ok:{text}"

        compiled = compile(
            nodes=[GraphNode("n1", "flaky@1", {"text": "hello"})],
            edges=[],
            registry=reg,
        )

        results = execute_sync(compiled, retry=RetryConfig(max_retries=3, delay=0.05))
        assert results["n1"]["result"] == "ok:hello"
        assert call_count == 3

    def test_global_retry_exhausted_raises(self):
        """When retries are exhausted, the flow fails."""
        reg = NodeRegistry()

        @reg.node("always-fail", version=1, name="Fail", description="Always fails")
        def always_fail(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            raise RuntimeError("nope")

        compiled = compile(
            nodes=[GraphNode("n1", "always-fail@1", {"text": "hello"})],
            edges=[],
            registry=reg,
        )

        with pytest.raises(FlowExecutionException):
            execute_sync(compiled, retry=RetryConfig(max_retries=2, delay=0.01))

    def test_node_level_retry_overrides_global(self):
        """Node-level max_retries takes precedence over global."""
        reg = NodeRegistry()
        call_count = 0

        @reg.node("flaky", version=1, name="Flaky", description="Node-level retry",
                  max_retries=3, retry_delay=0.01)
        def flaky(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("not yet")
            return "done"

        compiled = compile(
            nodes=[GraphNode("n1", "flaky@1", {"text": "x"})],
            edges=[],
            registry=reg,
        )

        # Global says no retry, but node says 3 — node wins
        results = execute_sync(compiled, retry=RetryConfig(max_retries=0))
        assert results["n1"]["result"] == "done"
        assert call_count == 3

    def test_no_retry_by_default(self):
        """Without RetryConfig, failures are immediate."""
        reg = NodeRegistry()

        @reg.node("fail", version=1, name="Fail", description="Fails")
        def fail(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            raise RuntimeError("boom")

        compiled = compile(
            nodes=[GraphNode("n1", "fail@1", {"text": "x"})],
            edges=[],
            registry=reg,
        )

        with pytest.raises(FlowExecutionException):
            execute_sync(compiled)

    def test_validation_errors_not_retried(self):
        """Pydantic validation errors should never be retried."""
        reg = NodeRegistry()
        call_count = 0

        @reg.node("typed", version=1, name="Typed", description="Needs int",
                  max_retries=5, retry_delay=0.01)
        def typed(num: Annotated[int, Text(label="Num")]) -> Annotated[int, Output(label="Out")]:
            nonlocal call_count
            call_count += 1
            return num

        compiled = compile(
            nodes=[GraphNode("n1", "typed@1", {"num": "not-a-number"})],
            edges=[],
            registry=reg,
        )

        with pytest.raises(FlowExecutionException):
            execute_sync(compiled)

        # Should only have been called 0 times (validation fails before execution)
        assert call_count == 0

    async def test_retry_emits_events(self):
        """Retries emit node_retry events."""
        reg = NodeRegistry()
        call_count = 0

        @reg.node("flaky", version=1, name="Flaky", description="Fails once")
        def flaky(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("first try fails")
            return "ok"

        compiled = compile(
            nodes=[GraphNode("n1", "flaky@1", {"text": "x"})],
            edges=[],
            registry=reg,
        )

        events = []
        async for event in execute(compiled, retry=RetryConfig(max_retries=2, delay=0.01)):
            events.append(event)

        types = [e["type"] for e in events]
        assert "node_retry" in types
        assert "node_complete" in types
        assert "flow_complete" in types

        retry_event = next(e for e in events if e["type"] == "node_retry")
        assert retry_event["attempt"] == 1
        assert retry_event["node_id"] == "n1"


# ---------------------------------------------------------------------------
# Retry + parallel interaction
# ---------------------------------------------------------------------------


class TestRetryWithParallel:
    def test_flaky_node_in_parallel_branch(self):
        """A flaky node in one branch retries while the other branch completes."""
        reg = NodeRegistry()
        call_counts: dict[str, int] = {"a": 0, "b": 0}

        @reg.node("flaky-a", version=1, name="Flaky A", description="Fails once",
                  max_retries=2, retry_delay=0.01)
        def flaky_a(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            call_counts["a"] += 1
            if call_counts["a"] == 1:
                raise RuntimeError("first try")
            return f"A:{text}"

        @reg.node("fast-b", version=1, name="Fast B", description="Always works")
        def fast_b(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            call_counts["b"] += 1
            return f"B:{text}"

        @reg.node("join", version=1, name="Join", description="Joins")
        def join(
            a: Annotated[str, Text(label="A")],
            b: Annotated[str, Text(label="B")],
        ) -> Annotated[str, Output(label="Out")]:
            return f"{a}+{b}"

        compiled = compile(
            nodes=[
                GraphNode("n1", "flaky-a@1", {"text": "x"}),
                GraphNode("n2", "fast-b@1", {"text": "y"}),
                GraphNode("n3", "join@1", None),
            ],
            edges=[
                GraphEdge("e1", "n1", "n3", "result", "a"),
                GraphEdge("e2", "n2", "n3", "result", "b"),
            ],
            registry=reg,
        )

        results = execute_sync(compiled)
        assert results["n3"]["result"] == "A:x+B:y"
        assert call_counts["a"] == 2  # retried once
        assert call_counts["b"] == 1  # no retry needed
