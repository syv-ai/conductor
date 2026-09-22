"""The surface a newcomer meets first: what a mistake says, what a record prints, and what is refused early.

One small test per rule. Each states what a person trips on and what they
see instead: a dead binding that fails loud, a stored graph that refuses a
misspelled key, a repr that reads back as a constructor, an error whose
text names what went wrong, a declaration refused where it is written.
"""

from collections.abc import Mapping
from typing import Annotated, Any

from conductor import NodeRegistry
from conductor.dtype import DType
from conductor.graph.binding import Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.model import Graph, GraphNode
from conductor.metadata import Input, Param, Result
from conductor.node import NodeDefinition
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "surface-test-txt"
    title = "Text"


Out = Annotated[Txt, Result(title="Result")]


class Scale(NodeDefinition):
    id = "scale"
    title = "Scale"
    description = "d"
    category = "test"

    def run(
        self,
        value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt(""),
        factor: Annotated[Txt, Param(title="Factor", widget=Textarea())] = Txt("2"),
    ) -> Out:
        return Txt(value * int(factor))


class Shaped(NodeDefinition):
    """Its inputs depend on a mode, so a binding can outlive its field."""

    id = "shaped"
    title = "Shaped"
    description = "d"
    category = "test"

    def run(self, mode: Annotated[Txt, Param(title="Mode", widget=Textarea())] = Txt("a")) -> Out:
        return mode

    def compute_inputs(self, declared: tuple[Input, ...], values: Mapping[str, Any]) -> tuple[Input, ...]:
        return declared


def _registry() -> NodeRegistry:
    registry = NodeRegistry()
    registry.register(Scale)
    registry.register(Shaped)
    return registry


def _compiled(*nodes: GraphNode) -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=list(nodes)), _registry())


# --- a dead binding (F4) -------------------------------------------------------------


def test_a_misspelled_binding_on_a_node_whose_inputs_are_its_signature_is_fatal():
    """'facter' for 'factor' used to run with the default and a quiet note."""
    compiled = _compiled(GraphNode(id="s", type="scale", version=1, bindings={"facter": Static(value="3")}))

    (problem,) = compiled.problems
    assert (problem.code, problem.fatal, problem.field) == ("stale_binding", True, "facter")
    assert not compiled.is_runnable


def test_the_dead_binding_names_the_inputs_the_node_has():
    compiled = _compiled(GraphNode(id="s", type="scale", version=1, bindings={"facter": Static(value="3")}))

    assert compiled.problems[0].message == "Field 'facter' is not an input of the node; it has factor, value."


def test_a_node_whose_inputs_come_and_go_keeps_the_binding_as_a_note():
    """A node with ``compute_inputs`` really can lose a field when its mode
    changes, so the binding it held is history, not a typo."""
    compiled = _compiled(GraphNode(id="h", type="shaped", version=1, bindings={"gone": Static(value="x")}))

    (problem,) = compiled.problems
    assert (problem.code, problem.fatal) == ("stale_binding", False)
    assert compiled.is_runnable
