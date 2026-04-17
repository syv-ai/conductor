"""Phase 1: Execution engine — the core end-to-end tests."""

from typing import Annotated

import pytest

from conductor.graph.model import GraphNode, GraphEdge
from conductor.graph.compiler import compile
from conductor.execution.engine import execute, execute_sync, collect
from conductor.execution.events import (
    NodeStartEvent,
    NodeCompleteEvent,
    FlowCompleteEvent,
)
from conductor.execution.results import normalize_result
from conductor.widgets import Text, Output
from conductor.errors import FlowExecutionException


# ---------------------------------------------------------------------------
# Fixtures: register nodes for test use
# ---------------------------------------------------------------------------

@pytest.fixture
def three_node_registry(registry):
    """Registry with echo, upper, and combine nodes."""

    @registry.node("echo", version=1, name="Echo", description="Returns input")
    def echo(
        text: Annotated[str, Text(label="Input")],
    ) -> Annotated[str, Output(label="Output")]:
        return text

    @registry.node("upper", version=1, name="Upper", description="Uppercases")
    def upper(
        text: Annotated[str, Text(label="Input")],
    ) -> Annotated[str, Output(label="Output")]:
        return text.upper()

    @registry.node("combine", version=1, name="Combine", description="Joins two strings")
    def combine(
        a: Annotated[str, Text(label="A")],
        b: Annotated[str, Text(label="B")],
    ) -> Annotated[str, Output(label="Output")]:
        return f"{a} {b}"

    return registry


# ---------------------------------------------------------------------------
# Result normalization
# ---------------------------------------------------------------------------

class TestNormalizeResult:
    def test_single_value_wrapped(self):
        assert normalize_result("hello") == {"result": "hello"}

    def test_dict_passthrough(self):
        assert normalize_result({"result": "hello"}) == {"result": "hello"}

    def test_tuple_creates_multi_output(self):
        result = normalize_result(("a", "b"))
        assert result == {"output_1": "a", "output_2": "b"}

    def test_none_result(self):
        assert normalize_result(None) == {"result": None}


# ---------------------------------------------------------------------------
# Streaming execution
# ---------------------------------------------------------------------------

class TestStreamingExecution:
    async def test_linear_chain_events(self, three_node_registry):
        """echo -> upper should produce start/complete events for each node."""
        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "hello"}),
                GraphNode("n2", "upper@1", None),
            ],
            edges=[GraphEdge("e1", "n1", "n2", "result", "text")],
            registry=three_node_registry,
        )

        events = []
        async for event in execute(compiled):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "node_start" in event_types
        assert "node_complete" in event_types
        assert "flow_complete" in event_types

    async def test_linear_chain_results(self, three_node_registry):
        """echo('hello') -> upper -> 'HELLO'."""
        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "hello"}),
                GraphNode("n2", "upper@1", None),
            ],
            edges=[GraphEdge("e1", "n1", "n2", "result", "text")],
            registry=three_node_registry,
        )

        results = await collect(execute(compiled))
        assert results["n2"]["result"] == "HELLO"

    async def test_diamond_execution(self, three_node_registry):
        """
        echo('hello') -> upper  -> combine
        echo('hello') -> echo2  -> combine
        """
        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "hello"}),
                GraphNode("n2", "upper@1", None),
                GraphNode("n3", "echo@1", None),
                GraphNode("n4", "combine@1", None),
            ],
            edges=[
                GraphEdge("e1", "n1", "n2", "result", "text"),
                GraphEdge("e2", "n1", "n3", "result", "text"),
                GraphEdge("e3", "n2", "n4", "result", "a"),
                GraphEdge("e4", "n3", "n4", "result", "b"),
            ],
            registry=three_node_registry,
        )

        results = await collect(execute(compiled))
        assert results["n4"]["result"] == "HELLO hello"


# ---------------------------------------------------------------------------
# Sync execution
# ---------------------------------------------------------------------------

class TestSyncExecution:
    def test_execute_sync_linear(self, three_node_registry):
        """Blocking API: echo -> upper."""
        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "world"}),
                GraphNode("n2", "upper@1", None),
            ],
            edges=[GraphEdge("e1", "n1", "n2", "result", "text")],
            registry=three_node_registry,
        )

        results = execute_sync(compiled)
        assert results["n2"]["result"] == "WORLD"

    def test_single_node_no_edges(self, three_node_registry):
        """A single node with static data, no edges."""
        compiled = compile(
            nodes=[GraphNode("n1", "echo@1", {"text": "standalone"})],
            edges=[],
            registry=three_node_registry,
        )

        results = execute_sync(compiled)
        assert results["n1"]["result"] == "standalone"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    async def test_cached_results_used(self, three_node_registry):
        """Passing cache skips execution and uses cached value."""
        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "hello"}),
                GraphNode("n2", "upper@1", None),
            ],
            edges=[GraphEdge("e1", "n1", "n2", "result", "text")],
            registry=three_node_registry,
        )

        results = await collect(execute(
            compiled,
            cache={"n1": {"result": "cached_value"}},
        ))
        # n2 should uppercase the cached value, not "hello"
        assert results["n2"]["result"] == "CACHED_VALUE"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    async def test_node_execution_error_yields_flow_error(self, registry):
        @registry.node("fail", version=1, name="Fail", description="Always fails")
        def fail(
            text: Annotated[str, Text(label="Input")],
        ) -> Annotated[str, Output(label="Output")]:
            raise RuntimeError("boom")

        compiled = compile(
            nodes=[GraphNode("n1", "fail@1", {"text": "hello"})],
            edges=[],
            registry=registry,
        )

        events = []
        async for event in execute(compiled):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "node_error" in event_types
        assert "flow_error" in event_types

    def test_execute_sync_raises_on_error(self, registry):
        @registry.node("fail", version=1, name="Fail", description="Always fails")
        def fail(
            text: Annotated[str, Text(label="Input")],
        ) -> Annotated[str, Output(label="Output")]:
            raise RuntimeError("boom")

        compiled = compile(
            nodes=[GraphNode("n1", "fail@1", {"text": "hello"})],
            edges=[],
            registry=registry,
        )

        with pytest.raises(FlowExecutionException):
            execute_sync(compiled)


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    async def test_timeout_produces_event(self, registry):
        import time

        @registry.node("slow", version=1, name="Slow", description="Sleeps")
        def slow(
            text: Annotated[str, Text(label="Input")],
        ) -> Annotated[str, Output(label="Output")]:
            time.sleep(2)
            return text

        compiled = compile(
            nodes=[GraphNode("n1", "slow@1", {"text": "hello"})],
            edges=[],
            registry=registry,
        )

        events = []
        async for event in execute(compiled, timeout_seconds=1):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "flow_timeout" in event_types


# ---------------------------------------------------------------------------
# Skip propagation
# ---------------------------------------------------------------------------

class TestSkipPropagation:
    async def test_skipped_input_skips_downstream(self, registry):
        """If all inputs to a node are SKIPPED, the node itself is skipped."""
        from conductor._sentinel import SKIPPED

        @registry.node("passthrough", version=1, name="Pass", description="Passes through")
        def passthrough(
            text: Annotated[str, Text(label="Input")],
        ) -> Annotated[str, Output(label="Output")]:
            return text

        @registry.node("conditional", version=1, name="Cond", description="Returns SKIPPED on one branch")
        def conditional(
            text: Annotated[str, Text(label="Input")],
        ) -> tuple[
            Annotated[str, Output(label="True branch")],
            Annotated[str, Output(label="False branch")],
        ]:
            return (text, SKIPPED)

        compiled = compile(
            nodes=[
                GraphNode("n1", "conditional@1", {"text": "hello"}),
                GraphNode("n2", "passthrough@1", None),  # connected to false branch
            ],
            edges=[GraphEdge("e1", "n1", "n2", "output_2", "text")],
            registry=registry,
        )

        events = []
        async for event in execute(compiled):
            events.append(event)

        # n2 should be skipped because its only input is SKIPPED
        skipped_events = [e for e in events if e["type"] == "node_skipped"]
        assert any(e["node_id"] == "n2" for e in skipped_events)
