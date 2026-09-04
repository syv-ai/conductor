"""Graph model, topological sort, cycle detection, compilation."""

from typing import Annotated

import pytest
from conductor.dtype import DType
from conductor.errors import CompilationError, CycleDetectionError
from conductor.graph.binding import Sources, Static
from conductor.graph.compiler import compile
from conductor.graph.model import Flow, GraphNode
from conductor.graph.topology import topological_sort
from conductor.node import NodeDefinition
from conductor.ref import Ref
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

    def test_a_node_with_no_bindings_has_no_data(self):
        node = GraphNode(id="n1", type="echo", version=1)
        assert node.data == {}


class TestTopologicalSort:
    def test_linear_chain(self):
        order = topological_sort({"a": frozenset(), "b": frozenset({"a"}), "c": frozenset({"b"})})
        assert order.index("a") < order.index("b") < order.index("c")

    def test_diamond_graph(self):
        """
        A -> B -> D
        A -> C -> D
        """
        order = topological_sort({
            "a": frozenset(), "b": frozenset({"a"}), "c": frozenset({"a"}), "d": frozenset({"b", "c"}),
        })
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_single_node(self):
        assert topological_sort({"a": frozenset()}) == ["a"]

    def test_disconnected_nodes(self):
        order = topological_sort({x: frozenset() for x in ["a", "b", "c"]})
        assert set(order) == {"a", "b", "c"}

    def test_cycle_detected(self):
        with pytest.raises(CycleDetectionError):
            topological_sort({"a": frozenset({"b"}), "b": frozenset({"a"})})

    def test_self_loop_detected(self):
        with pytest.raises(CycleDetectionError):
            topological_sort({"a": frozenset({"a"})})


class TestCompile:
    def test_compile_returns_compiled_graph(self, registry):
        registry.register(Echo)
        nodes = [
            GraphNode("n1", "echo", 1, bindings={"text": Static(value="hello")}),
            GraphNode("n2", "echo", 1, bindings={"text": Sources(refs=(Ref('n1', 'result'),))}),
        ]

        compiled = compile(Flow(nodes=nodes), registry)
        assert compiled is not None
        assert "n1" in compiled.execution_order
        assert "n2" in compiled.execution_order
        assert compiled.execution_order.index("n1") < compiled.execution_order.index("n2")

    def test_compile_unknown_node_type_raises(self, registry):
        nodes = [GraphNode("n1", "nonexistent", 1)]
        with pytest.raises(CompilationError):
            compile(Flow(nodes=nodes), registry)

    def test_compile_wire_from_a_missing_node_raises(self, registry):
        registry.register(Echo)
        nodes = [GraphNode("n1", "echo", 1, bindings={"text": Sources(refs=(Ref("n_missing", "result"),))})]

        with pytest.raises(CompilationError):
            compile(Flow(nodes=nodes), registry)

    def test_compile_cycle_raises(self, registry):
        registry.register(Echo)
        nodes = [
            GraphNode("n1", "echo", 1, bindings={"text": Sources(refs=(Ref('n2', 'result'),))}),
            GraphNode("n2", "echo", 1, bindings={"text": Sources(refs=(Ref('n1', 'result'),))}),
        ]
        with pytest.raises((CycleDetectionError, CompilationError)):
            compile(Flow(nodes=nodes), registry)
