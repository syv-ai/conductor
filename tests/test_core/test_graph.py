"""Graph model and the order over the dependency map."""

from typing import Annotated

import pytest
from conductor.dtype import DType
from conductor.graph.binding import Edges, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.graph.topology import order_of
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


class TestOrderOf:
    def test_linear_chain(self):
        order, cyclic = order_of({"a": frozenset(), "b": frozenset({"a"}), "c": frozenset({"b"})})
        assert order.index("a") < order.index("b") < order.index("c")
        assert cyclic == frozenset()

    def test_diamond_graph(self):
        """
        A -> B -> D
        A -> C -> D
        """
        order, _ = order_of({
            "a": frozenset(), "b": frozenset({"a"}), "c": frozenset({"a"}), "d": frozenset({"b", "c"}),
        })
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_single_node(self):
        assert order_of({"a": frozenset()}) == (("a",), frozenset())

    def test_disconnected_nodes(self):
        order, _ = order_of({x: frozenset() for x in ["a", "b", "c"]})
        assert set(order) == {"a", "b", "c"}

    def test_a_cycle_is_returned_not_raised(self):
        """A cycle is a state an editor can be in; compile anchors a problem on each node in it."""
        order, cyclic = order_of({"a": frozenset({"b"}), "b": frozenset({"a"}), "c": frozenset()})
        assert order == ("c",)
        assert cyclic == frozenset({"a", "b"})

    def test_self_loop_is_a_cycle(self):
        assert order_of({"a": frozenset({"a"})}) == ((), frozenset({"a"}))


class TestCompile:
    def test_compile_returns_compiled_graph(self, registry):
        registry.register(Echo)
        nodes = [
            GraphNode("n1", "echo", 1, bindings={"text": Static(value="hello")}),
            GraphNode("n2", "echo", 1, bindings={"text": Edges(refs=(Ref('n1', 'result'),))}),
        ]

        compiled = CompiledGraph.from_graph(Graph(nodes=nodes), registry)
        assert compiled.is_runnable, compiled.problems_for()
        assert compiled.execution_order() == ("n1", "n2")

    def test_compile_unknown_node_type_is_a_problem(self, registry):
        nodes = [GraphNode("n1", "nonexistent", 1)]
        compiled = CompiledGraph.from_graph(Graph(nodes=nodes), registry)
        assert [p.code for p in compiled.problems_for()] == ["unknown_node_type"]
        assert not compiled.is_runnable

    def test_compile_edge_from_a_missing_node_is_a_problem(self, registry):
        registry.register(Echo)
        nodes = [GraphNode("n1", "echo", 1, bindings={"text": Edges(refs=(Ref("n_missing", "result"),))})]

        compiled = CompiledGraph.from_graph(Graph(nodes=nodes), registry)
        assert [p.code for p in compiled.problems_for()] == ["unknown_ref_node"]

    def test_compile_cycle_is_a_problem_on_each_node_in_it(self, registry):
        registry.register(Echo)
        nodes = [
            GraphNode("n1", "echo", 1, bindings={"text": Edges(refs=(Ref('n2', 'result'),))}),
            GraphNode("n2", "echo", 1, bindings={"text": Edges(refs=(Ref('n1', 'result'),))}),
        ]
        compiled = CompiledGraph.from_graph(Graph(nodes=nodes), registry)
        assert [(p.code, p.node_id) for p in compiled.problems_for()] == [("cycle", "n1"), ("cycle", "n2")]
        assert not compiled.is_runnable
