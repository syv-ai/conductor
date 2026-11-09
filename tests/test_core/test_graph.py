"""Graph model and the order over the dependency map."""

from typing import Annotated

import pytest
from conductor import CompiledGraph, DType, From, Graph, GraphNode, Param, Result, Static, node
from conductor._internal.graph.topology import topological_order
from conductor.widgets import Textarea
from pydantic import ValidationError


class Txt(DType, str):
    id = "graph-test-text"
    display_title = "Text"


Out = Annotated[Txt, Result(title="Out")]


@node("echo", title="Echo", description="Echo", category="test")
def echo(text: Annotated[Txt, Param(title="In", widget=Textarea())]) -> Out:
    return text


class TestGraphModel:
    def test_graph_node_is_frozen(self):
        graph_node = GraphNode(id="n1", type="echo", version=1, bindings={"text": Static("hello")})
        assert graph_node.id == "n1"
        assert graph_node.type == "echo"
        assert graph_node.version == 1
        with pytest.raises(ValidationError, match="frozen"):
            graph_node.id = "n2"

class TestOrderOf:
    def test_linear_chain(self):
        order, cyclic = topological_order({"a": frozenset(), "b": frozenset({"a"}), "c": frozenset({"b"})})
        assert order.index("a") < order.index("b") < order.index("c")
        assert cyclic == frozenset()

    def test_diamond_graph(self):
        """
        A -> B -> D
        A -> C -> D
        """
        order, _ = topological_order({
            "a": frozenset(), "b": frozenset({"a"}), "c": frozenset({"a"}), "d": frozenset({"b", "c"}),
        })
        assert order.index("a") < order.index("b")
        assert order.index("a") < order.index("c")
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_single_node(self):
        assert topological_order({"a": frozenset()}) == (("a",), frozenset())

    def test_disconnected_nodes(self):
        order, _ = topological_order({x: frozenset() for x in ["a", "b", "c"]})
        assert set(order) == {"a", "b", "c"}

    def test_a_cycle_is_returned_not_raised(self):
        """A cycle is a state an editor can be in; compile anchors a problem on each node in it."""
        order, cyclic = topological_order({"a": frozenset({"b"}), "b": frozenset({"a"}), "c": frozenset()})
        assert order == ("c",)
        assert cyclic == frozenset({"a", "b"})

    def test_self_loop_is_a_cycle(self):
        assert topological_order({"a": frozenset({"a"})}) == ((), frozenset({"a"}))


class TestCompile:
    def test_compile_returns_compiled_graph(self, registry):
        registry = registry.with_nodes([echo])
        nodes = [
            GraphNode(id="n1", type="echo", version=1, bindings={"text": Static("hello")}),
            GraphNode(id="n2", type="echo", version=1, bindings={"text": From('n1.result')}),
        ]

        compiled = CompiledGraph.from_graph(Graph(nodes=nodes), registry)
        assert compiled.is_runnable, compiled.problems
        assert compiled.execution_order == ("n1", "n2")

    def test_compile_unknown_node_type_is_a_problem(self, registry):
        nodes = [GraphNode(id="n1", type="nonexistent", version=1)]
        compiled = CompiledGraph.from_graph(Graph(nodes=nodes), registry)
        assert [p.type for p in compiled.problems] == ["unknown_node_type"]
        assert not compiled.is_runnable

    def test_compile_edge_from_a_missing_node_is_a_problem(self, registry):
        registry = registry.with_nodes([echo])
        nodes = [GraphNode(id="n1", type="echo", version=1, bindings={"text": From("n_missing.result")})]

        compiled = CompiledGraph.from_graph(Graph(nodes=nodes), registry)
        assert [p.type for p in compiled.problems] == ["unknown_ref_node"]

    def test_compile_cycle_is_a_problem_on_each_node_in_it(self, registry):
        registry = registry.with_nodes([echo])
        nodes = [
            GraphNode(id="n1", type="echo", version=1, bindings={"text": From('n2.result')}),
            GraphNode(id="n2", type="echo", version=1, bindings={"text": From('n1.result')}),
        ]
        compiled = CompiledGraph.from_graph(Graph(nodes=nodes), registry)
        assert [(p.type, p.loc[0]) for p in compiled.problems] == [("cycle", "n1"), ("cycle", "n2")]
        assert not compiled.is_runnable
