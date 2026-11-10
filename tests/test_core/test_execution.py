"""The execution engine end to end: results, streaming, caching, errors, timeout, skips."""

import time
from dataclasses import dataclass
from typing import Annotated

import pytest
from conductor import SKIPPED, CompiledGraph, DType, GraphRecord, NodeRecord, Param, Result, node
from conductor.widgets import Textarea
from helpers import stream_of, with_added


class Txt(DType, str):
    id = "execution-test-text"
    display_title = "Text"


Out = Annotated[Txt, Result(title="Out")]


@node("echo", title="Echo", description="Returns input", category="test")
def echo(text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Out:
    return text


@node("upper", title="Upper", description="Uppercases", category="test")
def upper_node(text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Out:
    return Txt(text.upper())


@node("combine", title="Combine", description="Joins two strings", category="test")
def combine(
    a: Annotated[Txt, Param(title="A", widget=Textarea())],
    b: Annotated[Txt, Param(title="B", widget=Textarea())],
) -> Out:
    return Txt(f"{a} {b}")


@node("fail", title="Fail", description="Always fails", category="test")
def fail(text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Out:
    raise RuntimeError("boom")


@pytest.fixture
def three_node_registry(registry):
    """Registry with echo, upper, and combine nodes."""
    registry = with_added(registry, echo)
    registry = with_added(registry, upper_node)
    registry = with_added(registry, combine)
    return registry


@node("slow", title="Slow", description="Sleeps 0.3s, then uppercases", category="test")
def slow(text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Out:
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
        registry = with_added(registry, slow)
        registry = with_added(registry, combine)
        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id="a", node="slow", values={"text": "hello"}),
            NodeRecord(id="b", node="slow", values={"text": "world"}),
            NodeRecord(id="c", node="slow"),
            NodeRecord(id="d", node="slow"),
            NodeRecord(id="e", node="combine"),
        ], edges=[("a.result", "c.text"), ("b.result", "d.text"), ("c.result", "e.a"), ("d.result", "e.b")]), registry)

        start = time.monotonic()
        results = compiled.results(compiled.run_sync())
        elapsed = time.monotonic() - start

        # Sequential would be 5 * 0.3 = 1.5s; eager is A+B, C+D, E = ~0.9s.
        assert elapsed < 1.3, f"Took {elapsed:.2f}s — branches should run in parallel"
        assert results["e"]["result"] == "HELLO WORLD"

    async def test_linear_chain_events(self, three_node_registry):
        """echo -> upper should produce start/complete events for each node."""
        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id="n1", node="echo", values={"text": "hello"}),
            NodeRecord(id="n2", node="upper"),
        ], edges=[('n1.result', "n2.text")]), three_node_registry)

        events = []
        async for event in stream_of(compiled):
            events.append(event)

        event_types = [e.type for e in events]
        assert "node_start" in event_types
        assert "node_complete" in event_types
        assert "run_end" in event_types

    async def test_linear_chain_results(self, three_node_registry):
        """echo('hello') -> upper -> 'HELLO'."""
        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id="n1", node="echo", values={"text": "hello"}),
            NodeRecord(id="n2", node="upper"),
        ], edges=[('n1.result', "n2.text")]), three_node_registry)

        results = compiled.results((await compiled.run()))
        assert results["n2"]["result"] == "HELLO"

    async def test_diamond_execution(self, three_node_registry):
        """
        echo('hello') -> upper  -> combine
        echo('hello') -> echo2  -> combine
        """
        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id="n1", node="echo", values={"text": "hello"}),
            NodeRecord(id="n2", node="upper"),
            NodeRecord(id="n3", node="echo"),
            NodeRecord(id="n4", node="combine"),
        ], edges=[('n1.result', "n2.text"), ('n1.result', "n3.text"), ('n2.result', "n4.a"), ('n3.result', "n4.b")]), three_node_registry)

        results = compiled.results((await compiled.run()))
        assert results["n4"]["result"] == "HELLO hello"


# ---------------------------------------------------------------------------
# Sync execution
# ---------------------------------------------------------------------------

class TestSyncExecution:
    def test_run_sync_linear(self, three_node_registry):
        """Blocking API: echo -> upper."""
        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id="n1", node="echo", values={"text": "world"}),
            NodeRecord(id="n2", node="upper"),
        ], edges=[('n1.result', "n2.text")]), three_node_registry)

        results = compiled.results(compiled.run_sync())
        assert results["n2"]["result"] == "WORLD"

    def test_single_node_no_edges(self, three_node_registry):
        """A single node with static data, no edges."""
        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id="n1", node="echo", values={"text": "standalone"}),
        ]), three_node_registry)

        results = compiled.results(compiled.run_sync())
        assert results["n1"]["result"] == "standalone"


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    async def test_a_node_error_ends_the_graph_in_error(self, registry):
        registry = with_added(registry, fail)
        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id="n1", node="fail", values={"text": "hello"}),
        ]), registry)

        events = []
        async for event in stream_of(compiled):
            events.append(event)

        event_types = [e.type for e in events]
        assert "node_failed" in event_types
        assert "node_failed" in event_types

    def test_run_sync_returns_the_error_ending(self, registry):
        registry = with_added(registry, fail)
        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id="n1", node="fail", values={"text": "hello"}),
        ]), registry)

        assert compiled.run_sync().status == "failed"


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

class TestTimeout:
    async def test_timeout_produces_event(self, registry):
        @node("slow", title="Slow", description="Sleeps", category="test")
        def slow(text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Out:
            time.sleep(2)
            return text

        registry = with_added(registry, slow)
        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id="n1", node="slow", values={"text": "hello"}),
        ]), registry)

        events = []
        async for event in stream_of(compiled, timeout=1):
            events.append(event)

        event_types = [e.type for e in events]
        assert "run_end" in event_types


# ---------------------------------------------------------------------------
# Skip propagation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Branches:
    """Two exclusive outputs: the one not taken holds ``SKIPPED``."""

    taken: Annotated[Txt, Result(title="Taken", exclusive_group="branch")]
    not_taken: Annotated[Txt, Result(title="Not taken", exclusive_group="branch")]


class TestSkipPropagation:
    async def test_skipped_input_skips_downstream(self, registry):
        """If all inputs to a node are SKIPPED, the node itself is skipped."""

        @node("conditional", title="Cond", description="Returns SKIPPED on one branch", category="test")
        def conditional(text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Branches:
            return Branches(taken=text, not_taken=SKIPPED)

        registry = with_added(registry, echo)
        registry = with_added(registry, conditional)
        compiled = CompiledGraph.from_graph(GraphRecord(nodes=[
            NodeRecord(id="n1", node="conditional", values={"text": "hello"}),
            NodeRecord(id="n2", node="echo"),
        ], edges=[('n1.not_taken', "n2.text")]), registry)

        events = []
        async for event in stream_of(compiled):
            events.append(event)

        # n2 should be skipped because its only input is SKIPPED
        skipped_events = [e for e in events if e.type == "node_skipped"]
        assert any(e.node_id == "n2" for e in skipped_events)


# ---------------------------------------------------------------------------
# Stray data keys: a saved graph may carry keys that are not parameters of
# the node (host metadata). The engine validates a call with extra="ignore",
# so every node drops them rather than failing on an unexpected keyword.
# ---------------------------------------------------------------------------
