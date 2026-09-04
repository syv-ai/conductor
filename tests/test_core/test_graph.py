"""Graph model, topological sort, cycle detection, compilation."""

from typing import Annotated

import pytest
from conductor.dtype import DType
from conductor.errors import CompilationError, CycleDetectionError
from conductor.graph.binding import Static
from conductor.graph.compiler import compile
from conductor.graph.model import GraphEdge, GraphNode
from conductor.graph.topology import topological_sort
from conductor.node import NodeDefinition
from conductor.returns import Result
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "graph-test-text"
    title = "Text"


Out = Annotated[Txt, Result(title="Out")]


class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "Echo"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="In")]) -> Out:
        return text


class TestGraphModel:
    def test_graph_node_is_frozen(self):
        node = GraphNode(id="n1", type="echo", version=1, bindings={"text": Static(value="hello")})
        assert node.id == "n1"
        assert node.type == "echo"
        assert node.version == 1
        assert node.data == {"text": "hello"}
        with pytest.raises(AttributeError):
            node.id = "n2"

    def test_graph_edge_is_frozen(self):
        edge = GraphEdge(id="e1", source="n1", target="n2", source_handle="result", target_handle="text")
        assert edge.source == "n1"
        assert edge.target == "n2"
        with pytest.raises(AttributeError):
            edge.source = "n3"

    def test_a_node_with_no_bindings_has_no_data(self):
        node = GraphNode(id="n1", type="echo", version=1)
        assert node.data == {}


class TestTopologicalSort:
    def test_linear_chain(self):
        """A -> B -> C should produce [A, B, C]."""
        nodes = [
            GraphNode("a", "t", 1),
            GraphNode("b", "t", 1),
            GraphNode("c", "t", 1),
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
        nodes = [GraphNode(x, "t", 1) for x in ["a", "b", "c", "d"]]
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
        nodes = [GraphNode("a", "t", 1)]
        order = topological_sort(nodes, [])
        assert order == ["a"]

    def test_disconnected_nodes(self):
        nodes = [GraphNode(x, "t", 1) for x in ["a", "b", "c"]]
        order = topological_sort(nodes, [])
        assert set(order) == {"a", "b", "c"}

    def test_cycle_detected(self):
        nodes = [GraphNode(x, "t", 1) for x in ["a", "b"]]
        edges = [
            GraphEdge("e1", "a", "b", "r", "i"),
            GraphEdge("e2", "b", "a", "r", "i"),
        ]
        with pytest.raises(CycleDetectionError):
            topological_sort(nodes, edges)

    def test_self_loop_detected(self):
        nodes = [GraphNode("a", "t", 1)]
        edges = [GraphEdge("e1", "a", "a", "r", "i")]
        with pytest.raises(CycleDetectionError):
            topological_sort(nodes, edges)


class TestCompile:
    def test_compile_returns_compiled_graph(self, registry):
        registry.register(Echo)
        nodes = [
            GraphNode("n1", "echo", 1, bindings={"text": Static(value="hello")}),
            GraphNode("n2", "echo", 1),
        ]
        edges = [GraphEdge("e1", "n1", "n2", "result", "text")]

        compiled = compile(nodes=nodes, edges=edges, registry=registry)
        assert compiled is not None
        assert "n1" in compiled.execution_order
        assert "n2" in compiled.execution_order
        assert compiled.execution_order.index("n1") < compiled.execution_order.index("n2")

    def test_compile_unknown_node_type_raises(self, registry):
        nodes = [GraphNode("n1", "nonexistent", 1)]
        with pytest.raises(CompilationError):
            compile(nodes=nodes, edges=[], registry=registry)

    def test_compile_invalid_edge_raises(self, registry):
        registry.register(Echo)
        nodes = [GraphNode("n1", "echo", 1)]
        edges = [GraphEdge("e1", "n1", "n_missing", "result", "text")]

        with pytest.raises(CompilationError):
            compile(nodes=nodes, edges=edges, registry=registry)

    def test_compile_cycle_raises(self, registry):
        registry.register(Echo)
        nodes = [
            GraphNode("n1", "echo", 1),
            GraphNode("n2", "echo", 1),
        ]
        edges = [
            GraphEdge("e1", "n1", "n2", "result", "text"),
            GraphEdge("e2", "n2", "n1", "result", "text"),
        ]
        with pytest.raises((CycleDetectionError, CompilationError)):
            compile(nodes=nodes, edges=edges, registry=registry)
