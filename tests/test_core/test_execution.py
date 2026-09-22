"""The execution engine end to end: results, streaming, caching, errors, timeout, skips."""

import time
from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import SKIPPED, Param, run, run_sync
from conductor.dtype import DType
from conductor.execution.engine import execute
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.metadata import Result
from conductor.node import NodeDefinition
from conductor.ref import Ref
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "execution-test-text"
    title = "Text"


Out = Annotated[Txt, Result(title="Out")]


class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "Returns input"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Out:
        return text


class Upper(NodeDefinition):
    id = "upper"
    title = "Upper"
    description = "Uppercases"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Out:
        return Txt(text.upper())


class Combine(NodeDefinition):
    id = "combine"
    title = "Combine"
    description = "Joins two strings"
    category = "test"

    def run(
        self,
        a: Annotated[Txt, Param(title="A", widget=Textarea())],
        b: Annotated[Txt, Param(title="B", widget=Textarea())],
    ) -> Out:
        return Txt(f"{a} {b}")


class Fail(NodeDefinition):
    id = "fail"
    title = "Fail"
    description = "Always fails"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Out:
        raise RuntimeError("boom")


@pytest.fixture
def three_node_registry(registry):
    """Registry with echo, upper, and combine nodes."""
    registry.register(Echo)
    registry.register(Upper)
    registry.register(Combine)
    return registry


class Slow(NodeDefinition):
    id = "slow"
    title = "Slow"
    description = "Sleeps 0.3s, then uppercases"
    category = "test"

    def run(self, text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Out:
        time.sleep(0.3)
        return Txt(text.upper())


# ---------------------------------------------------------------------------
# Streaming execution
# ---------------------------------------------------------------------------

class TestStreamingExecution:
    def test_independent_branches_run_in_parallel(self, registry):
        """Two independent branches overlap instead of running one after the other."""
        # A(0.3s) -> C(0.3s) --+
        #                      +--> E
        # B(0.3s) -> D(0.3s) --+
        registry.register(Slow)
        registry.register(Combine)
        compiled = CompiledGraph.from_graph(Graph(nodes=[
            GraphNode(id="a", type="slow", version=1, bindings={"text": Static("hello")}),
            GraphNode(id="b", type="slow", version=1, bindings={"text": Static("world")}),
            GraphNode(id="c", type="slow", version=1, bindings={"text": From(Ref("a", "result"))}),
            GraphNode(id="d", type="slow", version=1, bindings={"text": From(Ref("b", "result"))}),
            GraphNode(id="e", type="combine", version=1, bindings={"a": From(Ref("c", "result")), "b": From(Ref("d", "result"))}),
        ]), registry)

        start = time.monotonic()
        results = run_sync(compiled)["results"]
        elapsed = time.monotonic() - start

        # Sequential would be 5 * 0.3 = 1.5s; eager is A+B, C+D, E = ~0.9s.
        assert elapsed < 1.3, f"Took {elapsed:.2f}s — branches should run in parallel"
        assert results["e"]["result"] == "HELLO WORLD"

    async def test_linear_chain_events(self, three_node_registry):
        """echo -> upper should produce start/complete events for each node."""
        compiled = CompiledGraph.from_graph(Graph(nodes=[
                GraphNode(id="n1", type="echo", version=1, bindings={"text": Static("hello")}),
                GraphNode(id="n2", type="upper", version=1, bindings={"text": From(Ref('n1', 'result'))}),
            ]), three_node_registry)

        events = []
        async for event in execute(compiled):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "node_start" in event_types
        assert "node_complete" in event_types
        assert "graph_complete" in event_types

    async def test_linear_chain_results(self, three_node_registry):
        """echo('hello') -> upper -> 'HELLO'."""
        compiled = CompiledGraph.from_graph(Graph(nodes=[
                GraphNode(id="n1", type="echo", version=1, bindings={"text": Static("hello")}),
                GraphNode(id="n2", type="upper", version=1, bindings={"text": From(Ref('n1', 'result'))}),
            ]), three_node_registry)

        results = (await run(compiled))["results"]
        assert results["n2"]["result"] == "HELLO"

    async def test_diamond_execution(self, three_node_registry):
        """
        echo('hello') -> upper  -> combine
        echo('hello') -> echo2  -> combine
        """
        compiled = CompiledGraph.from_graph(Graph(nodes=[
                GraphNode(id="n1", type="echo", version=1, bindings={"text": Static("hello")}),
                GraphNode(id="n2", type="upper", version=1, bindings={"text": From(Ref('n1', 'result'))}),
                GraphNode(id="n3", type="echo", version=1, bindings={"text": From(Ref('n1', 'result'))}),
                GraphNode(id="n4", type="combine", version=1, bindings={"a": From(Ref('n2', 'result')), "b": From(Ref('n3', 'result'))}),
            ]), three_node_registry)

        results = (await run(compiled))["results"]
        assert results["n4"]["result"] == "HELLO hello"


# ---------------------------------------------------------------------------
# Sync execution
# ---------------------------------------------------------------------------

class TestSyncExecution:
    def test_run_sync_linear(self, three_node_registry):
        """Blocking API: echo -> upper."""
        compiled = CompiledGraph.from_graph(Graph(nodes=[
                GraphNode(id="n1", type="echo", version=1, bindings={"text": Static("world")}),
                GraphNode(id="n2", type="upper", version=1, bindings={"text": From(Ref('n1', 'result'))}),
            ]), three_node_registry)

        results = run_sync(compiled)["results"]
        assert results["n2"]["result"] == "WORLD"

    def test_single_node_no_edges(self, three_node_registry):
        """A single node with static data, no edges."""
        compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n1", type="echo", version=1, bindings={"text": Static("standalone")})]), three_node_registry)

        results = run_sync(compiled)["results"]
        assert results["n1"]["result"] == "standalone"


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestCaching:
    async def test_cached_results_used(self, three_node_registry):
        """Passing cache skips execution and uses cached value."""
        compiled = CompiledGraph.from_graph(Graph(nodes=[
                GraphNode(id="n1", type="echo", version=1, bindings={"text": Static("hello")}),
                GraphNode(id="n2", type="upper", version=1, bindings={"text": From(Ref('n1', 'result'))}),
            ]), three_node_registry)

        results = (await run(
            compiled,
            cache={"n1": {"result": "cached_value"}},
        ))["results"]
        # n2 should uppercase the cached value, not "hello"
        assert results["n2"]["result"] == "CACHED_VALUE"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    async def test_a_node_error_ends_the_graph_in_error(self, registry):
        registry.register(Fail)
        compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n1", type="fail", version=1, bindings={"text": Static("hello")})]), registry)

        events = []
        async for event in execute(compiled):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "node_error" in event_types
        assert "graph_error" in event_types

    def test_run_sync_returns_the_error_ending(self, registry):
        registry.register(Fail)
        compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n1", type="fail", version=1, bindings={"text": Static("hello")})]), registry)

        assert run_sync(compiled)["type"] == "graph_error"


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    async def test_timeout_produces_event(self, registry):
        class Slow(NodeDefinition):
            id = "slow"
            title = "Slow"
            description = "Sleeps"
            category = "test"

            def run(self, text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Out:
                time.sleep(2)
                return text

        registry.register(Slow)
        compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n1", type="slow", version=1, bindings={"text": Static("hello")})]), registry)

        events = []
        async for event in execute(compiled, timeout=1):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "graph_timeout" in event_types


# ---------------------------------------------------------------------------
# Skip propagation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Branches:
    """Two exclusive outputs: the one not taken holds ``SKIPPED``."""

    taken: Annotated[Txt, Result(title="Taken", choice="branch")]
    not_taken: Annotated[Txt, Result(title="Not taken", choice="branch")]


class TestSkipPropagation:
    async def test_skipped_input_skips_downstream(self, registry):
        """If all inputs to a node are SKIPPED, the node itself is skipped."""

        class Conditional(NodeDefinition):
            id = "conditional"
            title = "Cond"
            description = "Returns SKIPPED on one branch"
            category = "test"

            def run(self, text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Branches:
                return Branches(taken=text, not_taken=SKIPPED)

        registry.register(Echo)
        registry.register(Conditional)
        compiled = CompiledGraph.from_graph(Graph(nodes=[
                GraphNode(id="n1", type="conditional", version=1, bindings={"text": Static("hello")}),
                GraphNode(id="n2", type="echo", version=1, bindings={"text": From(Ref('n1', 'not_taken'))}),  # connected to the branch not taken
            ]), registry)

        events = []
        async for event in execute(compiled):
            events.append(event)

        # n2 should be skipped because its only input is SKIPPED
        skipped_events = [e for e in events if e["type"] == "node_skipped"]
        assert any(e["node_id"] == "n2" for e in skipped_events)


# ---------------------------------------------------------------------------
# Stray data keys: a saved graph may carry keys that are not parameters of
# the node (host metadata). The engine validates a call with extra="ignore",
# so every node drops them rather than failing on an unexpected keyword.
# ---------------------------------------------------------------------------

