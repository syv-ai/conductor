"""Phase 3: Compound nodes — for-each loop execution."""

from typing import Annotated

import pytest
from conductor.compound.for_each import FOR_EACH
from conductor.execution.engine import execute, execute_sync
from conductor.graph.compiler import compile
from conductor.graph.model import GraphEdge, GraphNode
from conductor.widgets import ConnectionList, Output, Text


@pytest.fixture
def loop_registry(registry):
    """Registry with nodes needed for loop testing."""

    @registry.node("upper", version=1, name="Upper", description="Uppercases")
    def upper(
        text: Annotated[str, Text(label="Input")],
    ) -> Annotated[str, Output(label="Output")]:
        return text.upper()

    @registry.node(
        "for-each-start", version=1, name="For Each Start",
        description="Start of for-each loop",
    )
    def for_each_start(
        items: Annotated[list[str], ConnectionList(label="Items")],
    ) -> tuple[
        Annotated[str, Output(label="Item")],
        Annotated[int, Output(label="Index")],
    ]:
        # This function body is NOT called directly — ForEachNode handles iteration.
        # The signature defines the outputs available inside the loop body.
        raise NotImplementedError("Handled by compound node")

    @registry.node(
        "for-each-end", version=1, name="For Each End",
        description="End of for-each loop",
    )
    def for_each_end(
        item: Annotated[str, Text(label="Item")],
    ) -> Annotated[list[str], Output(label="Collected")]:
        raise NotImplementedError("Handled by compound node")

    return registry


class TestForEachSequential:
    def test_sequential_loop(self, loop_registry):
        """
        for-each-start([a, b, c]) -> upper -> for-each-end
        Should produce ["A", "B", "C"]
        """
        nodes = [
            GraphNode("start", "for-each-start@1", {"items": ["a", "b", "c"]}),
            GraphNode("body", "upper@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "body", "output_1", "text"),  # item -> upper
            GraphEdge("e2", "body", "end", "result", "item"),      # upper result -> end
        ]

        compiled = compile(
            nodes=nodes,
            edges=edges,
            registry=loop_registry,
            compound_types=[FOR_EACH],
        )

        results = execute_sync(compiled)
        # The end node should collect all iteration results
        end_result = results["end"]["result"]
        assert isinstance(end_result, list)
        assert set(end_result) == {"A", "B", "C"}


class TestForEachParallel:
    def test_parallel_loop(self, loop_registry):
        """Same as sequential but with parallel execution mode."""
        nodes = [
            GraphNode("start", "for-each-start@1", {
                "items": ["x", "y", "z"],
                "execution_mode": "Parallel",
            }),
            GraphNode("body", "upper@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "body", "output_1", "text"),
            GraphEdge("e2", "body", "end", "result", "item"),
        ]

        compiled = compile(
            nodes=nodes,
            edges=edges,
            registry=loop_registry,
            compound_types=[FOR_EACH],
        )

        results = execute_sync(compiled)
        end_result = results["end"]["result"]
        assert isinstance(end_result, list)
        assert set(end_result) == {"X", "Y", "Z"}


class TestForEachEvents:
    async def test_loop_emits_progress_events(self, loop_registry):
        nodes = [
            GraphNode("start", "for-each-start@1", {"items": ["a", "b"]}),
            GraphNode("body", "upper@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "body", "output_1", "text"),
            GraphEdge("e2", "body", "end", "result", "item"),
        ]

        compiled = compile(
            nodes=nodes,
            edges=edges,
            registry=loop_registry,
            compound_types=[FOR_EACH],
        )

        events = []
        async for event in execute(compiled):
            events.append(event)

        event_types = [e["type"] for e in events]
        assert "node_progress" in event_types
        assert "flow_complete" in event_types


class TestForEachEmpty:
    def test_empty_items_produces_empty_result(self, loop_registry):
        nodes = [
            GraphNode("start", "for-each-start@1", {"items": []}),
            GraphNode("body", "upper@1", None),
            GraphNode("end", "for-each-end@1", None),
        ]
        edges = [
            GraphEdge("e1", "start", "body", "output_1", "text"),
            GraphEdge("e2", "body", "end", "result", "item"),
        ]

        compiled = compile(
            nodes=nodes,
            edges=edges,
            registry=loop_registry,
            compound_types=[FOR_EACH],
        )

        results = execute_sync(compiled)
        end_result = results["end"]["result"]
        assert isinstance(end_result, list)
        assert len(end_result) == 0
