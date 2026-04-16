"""Phase 1: Graph model, topological sort, cycle detection, compilation."""

import pytest

from flowengine.graph.model import GraphNode, GraphEdge
from flowengine.graph.topology import topological_sort
from flowengine.graph.compiler import compile
from flowengine.errors import CycleDetectionError, CompilationError


class TestGraphModel:
    def test_graph_node_is_frozen(self):
        node = GraphNode(id="n1", type="echo@1", data={"text": "hello"})
        assert node.id == "n1"
        assert node.type == "echo@1"
        assert node.data == {"text": "hello"}
        with pytest.raises(AttributeError):
            node.id = "n2"

    def test_graph_edge_is_frozen(self):
        edge = GraphEdge(id="e1", source="n1", target="n2", source_handle="result", target_handle="text")
        assert edge.source == "n1"
        assert edge.target == "n2"
        with pytest.raises(AttributeError):
            edge.source = "n3"

    def test_graph_node_data_optional(self):
        node = GraphNode(id="n1", type="echo@1", data=None)
        assert node.data is None


class TestTopologicalSort:
    def test_linear_chain(self):
        """A -> B -> C should produce [A, B, C]."""
        nodes = [
            GraphNode("a", "t", None),
            GraphNode("b", "t", None),
            GraphNode("c", "t", None),
        ]
        edges = [
            GraphEdge("e1", "a", "b", "result", "input"),
            GraphEdge("e2", "b", "c", "result", "input"),
        ]
        order = topological_sort(nodes, edges)
        assert order.index("a") < order.index("b") < order.index("c")

    def test_diamond_graph(self):
        """
        A -> B -> D
        A -> C -> D
        """
        nodes = [GraphNode(x, "t", None) for x in ["a", "b", "c", "d"]]
        edges = [
            GraphEdge("e1", "a", "b", "r", "i"),
            GraphEdge("e2", "a", "c", "r", "i"),
            GraphEdge("e3", "b", "d", "r", "i"),
            GraphEdge("e4", "c", "d", "r", "i"),
        ]
        order = topological_sort(nodes, edges)
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_single_node(self):
        nodes = [GraphNode("a", "t", None)]
        order = topological_sort(nodes, [])
        assert order == ["a"]

    def test_disconnected_nodes(self):
        nodes = [GraphNode(x, "t", None) for x in ["a", "b", "c"]]
        order = topological_sort(nodes, [])
        assert set(order) == {"a", "b", "c"}

    def test_cycle_detected(self):
        nodes = [GraphNode(x, "t", None) for x in ["a", "b"]]
        edges = [
            GraphEdge("e1", "a", "b", "r", "i"),
            GraphEdge("e2", "b", "a", "r", "i"),
        ]
        with pytest.raises(CycleDetectionError):
            topological_sort(nodes, edges)

    def test_self_loop_detected(self):
        nodes = [GraphNode("a", "t", None)]
        edges = [GraphEdge("e1", "a", "a", "r", "i")]
        with pytest.raises(CycleDetectionError):
            topological_sort(nodes, edges)


class TestCompile:
    def test_compile_returns_compiled_graph(self, registry):
        from typing import Annotated
        from flowengine.widgets import Text, Output

        @registry.node("echo", version=1, name="Echo", description="Echo")
        def echo(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            return text

        nodes = [
            GraphNode("n1", "echo@1", {"text": "hello"}),
            GraphNode("n2", "echo@1", None),
        ]
        edges = [GraphEdge("e1", "n1", "n2", "result", "text")]

        compiled = compile(nodes=nodes, edges=edges, registry=registry)
        assert compiled is not None
        assert "n1" in compiled.execution_order
        assert "n2" in compiled.execution_order
        assert compiled.execution_order.index("n1") < compiled.execution_order.index("n2")

    def test_compile_unknown_node_type_raises(self, registry):
        nodes = [GraphNode("n1", "nonexistent@1", None)]
        with pytest.raises(CompilationError):
            compile(nodes=nodes, edges=[], registry=registry)

    def test_compile_invalid_edge_raises(self, registry):
        from typing import Annotated
        from flowengine.widgets import Text, Output

        @registry.node("echo", version=1, name="Echo", description="Echo")
        def echo(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            return text

        nodes = [GraphNode("n1", "echo@1", None)]
        edges = [GraphEdge("e1", "n1", "n_missing", "result", "text")]

        with pytest.raises(CompilationError):
            compile(nodes=nodes, edges=edges, registry=registry)

    def test_compile_cycle_raises(self, registry):
        from typing import Annotated
        from flowengine.widgets import Text, Output

        @registry.node("echo", version=1, name="Echo", description="Echo")
        def echo(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            return text

        nodes = [
            GraphNode("n1", "echo@1", None),
            GraphNode("n2", "echo@1", None),
        ]
        edges = [
            GraphEdge("e1", "n1", "n2", "result", "text"),
            GraphEdge("e2", "n2", "n1", "result", "text"),
        ]
        with pytest.raises((CycleDetectionError, CompilationError)):
            compile(nodes=nodes, edges=edges, registry=registry)
