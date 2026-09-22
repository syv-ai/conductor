"""The surface a newcomer meets first: what a mistake says, what a record prints, and what is refused early.

One small test per rule. Each states what a person trips on and what they
see instead: a dead binding that fails loud, a stored graph that refuses a
misspelled key, a repr that reads back as a constructor, an error whose
text names what went wrong, a declaration refused where it is written.
"""

from collections.abc import Mapping
from typing import Annotated, Any

import pytest
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


# --- a stored record refuses a key it does not have (F1) ----------------------------


def test_a_stored_graph_with_a_misspelled_key_is_refused_naming_it():
    """A stored graph is a trust boundary: 'bindngs' used to load as a node
    with no bindings at all."""
    from pydantic import ValidationError

    text = "nodes:\n  - id: s\n    type: scale\n    version: 1\n    bindngs: {}\n"
    with pytest.raises(ValidationError, match="bindngs"):
        Graph.from_yaml(text)


def test_a_widget_refuses_a_key_it_does_not_have():
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="row"):
        Textarea(row=5)


def test_display_is_the_canvas_own_and_takes_any_key():
    node = GraphNode(id="s", type="scale", version=1, display={"position": {"x": 1}, "colour": "red"})
    assert node.display["colour"] == "red"


# --- a repr is the call that makes the object (F3) -----------------------------------


def test_a_series_prints_as_its_constructor():
    from conductor.series import Index, Series

    assert repr(Series[Txt](Index("lines"), [Txt("a")])) == "Series[Txt](Index('lines'), ['a'])"
    sparse = Series[Txt](Index("lines"), [Txt("b")], rows=[(1,)])
    assert repr(sparse) == "Series[Txt](Index('lines'), ['b'], rows=[(1,)])"
    assert repr(Index("b", parent=Index("a"))) == "Index('b', parent=Index('a'))"


def test_a_node_version_prints_its_run_by_name():
    assert repr(Scale.versions[1]).startswith("NodeVersion(run=Scale.run, ")


def test_a_compiled_graph_prints_a_one_line_summary():
    compiled = _compiled(GraphNode(id="s", type="scale", version=1))
    assert repr(compiled) == "CompiledGraph(nodes=('s',), is_runnable=True, problems=0)"


def test_a_registry_prints_a_call_its_constructor_accepts():
    registry = NodeRegistry(nodes=(Scale, Shaped))

    assert registry.nodes == (Scale, Shaped)
    assert repr(registry).startswith("NodeRegistry(nodes=(\n    Scale(id='scale'")
