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
    compiled = _compiled(GraphNode(id="s", type="scale", version=1, bindings={"facter": Static("3")}))

    (problem,) = compiled.problems
    assert (problem.code, problem.fatal, problem.field) == ("stale_binding", True, "facter")
    assert not compiled.is_runnable


def test_the_dead_binding_names_the_inputs_the_node_has():
    compiled = _compiled(GraphNode(id="s", type="scale", version=1, bindings={"facter": Static("3")}))

    assert compiled.problems[0].message == "Field 'facter' is not an input of the node; it has factor, value."


def test_a_node_whose_inputs_come_and_go_keeps_the_binding_as_a_note():
    """A node with ``compute_inputs`` really can lose a field when its mode
    changes, so the binding it held is history, not a typo."""
    compiled = _compiled(GraphNode(id="h", type="shaped", version=1, bindings={"gone": Static("x")}))

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


# --- an error's text names what went wrong (F5) ---------------------------------------


def test_a_compilation_error_lists_its_fatal_problems():
    """It used to say only 'the graph cannot run'; the problems were on an attribute."""
    from conductor import run_sync
    from conductor.errors import CompilationError

    compiled = _compiled(GraphNode(id="s", type="scale", version=1, bindings={"facter": Static("3")}))
    with pytest.raises(CompilationError) as raised:
        run_sync(compiled)

    assert str(raised.value) == (
        "the graph cannot run:\n"
        "  s.facter — stale_binding: Field 'facter' is not an input of the node; it has factor, value."
    )


# --- a declaration mistake is refused where it is written (F6) ------------------------


def _declare(**body):
    return type("Bad", (NodeDefinition,), {"id": "bad", "title": "Bad", "description": "d", "category": "test", **body})


def _run(annotations: dict, **kinds):
    """A ``run`` with the given parameter annotations, built from source so a
    parameter kind (``*args``, ``**inputs``) can be spelled."""
    params = ", ".join(kinds.get(name, name) + f": A_{name}" for name in annotations)
    scope = {f"A_{name}": annotation for name, annotation in annotations.items()} | {"Out": Out, "Txt": Txt}
    exec(f"def run(self, {params}) -> Out:\n    return Txt('')", scope)
    return scope["run"]


def test_a_misspelled_run_is_refused_naming_what_the_class_does_define():
    with pytest.raises(TypeError, match=r"Bad declares an id but no run.*rnu"):
        _declare(rnu=lambda self: None)


def test_a_bare_series_parameter_is_refused():
    from conductor.series import Series

    with pytest.raises(TypeError, match=r"'rows'.*Series\[X\]"):
        _declare(run=_run({"rows": Annotated[Series, Param(title="Rows")]}))


def test_an_open_interface_typed_as_one_series_is_refused():
    from conductor.series import Series

    with pytest.raises(TypeError, match=r"\*\*inputs: Series\[Txt\].*\*\*inputs: Series"):
        _declare(run=_run({"inputs": Series[Txt]}, inputs="**inputs"))


def test_an_underscore_parameter_is_refused():
    with pytest.raises(TypeError, match="'_hidden'"):
        _declare(run=_run({"_hidden": Annotated[Txt, Param(title="H", widget=Textarea())]}))


@pytest.mark.parametrize("name", ["schema", "copy", "json", "model_config"])
def test_a_parameter_named_like_a_pydantic_model_attribute_validates_without_a_warning(name):
    """A call is validated through a pydantic model with one field per input,
    so an input named ``schema`` used to shadow ``BaseModel.schema`` and warn.
    The model names its fields itself and takes each input by its alias."""
    import warnings

    from conductor import run_sync

    node = _declare(run=_run({name: Annotated[Txt, Param(title="X", widget=Textarea())]}))
    registry = NodeRegistry(nodes=(node,))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n", type="bad", version=1, bindings={name: Static("x")})]), registry)
        assert compiled.node("n").validate({name: Txt("x")}) == {name: "x"}
    assert run_sync(compiled)["type"] == "graph_complete"


def test_args_stay_refused():
    with pytest.raises(TypeError, match=r"\*args"):
        _declare(run=_run({"args": Txt}, args="*args"))


# --- a frozen graph's node list is frozen too (F9) -----------------------------------


def test_a_graphs_nodes_are_a_tuple():
    graph = Graph(nodes=[GraphNode(id="s", type="scale", version=1)])

    assert isinstance(graph.nodes, tuple)
    with pytest.raises(AttributeError):
        graph.nodes.append(GraphNode(id="t", type="scale", version=1))


# --- one refusal, raised (F12) --------------------------------------------------------


def test_refuses_is_a_conductor_error_at_the_root():
    import conductor
    from conductor.errors import ConductorError, Refuses

    assert conductor.Refuses is Refuses and issubclass(Refuses, ConductorError)


def test_a_type_refuses_to_be_read_whole_by_raising_the_same_refusal():
    from conductor.errors import Refuses

    class Loose(DType):
        id = "surface-loose"
        title = "Loose"

        @classmethod
        def refuses_whole(cls) -> None:
            raise Refuses("columns_unknown", "Nobody said the columns.")

    with pytest.raises(Refuses) as refused:
        Loose.refuses_whole()
    assert (refused.value.code, refused.value.message) == ("columns_unknown", "Nobody said the columns.")
    assert Txt.refuses_whole() is None


# --- a declared type is a type (F13) -------------------------------------------------


@pytest.mark.parametrize("bogus", [42, "text", {"id": "text"}])
def test_a_field_refuses_a_dtype_that_is_not_a_type(bogus):
    """``Output(dtype=42)`` used to be accepted and dump ``dtype: null``, and
    validating a dump set the dtype to a dict."""
    from conductor.metadata import Output
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="dtype"):
        Output(name="result", dtype=bogus, title="R")


@pytest.mark.parametrize("declared", [Txt, Any, list[str], tuple[str, ...]])
def test_a_field_takes_a_dtype_any_or_a_static_type(declared):
    from conductor.metadata import Input

    assert Input(name="x", dtype=declared, title="X", show_handle=False).dtype == declared


# --- the registry is a container, and a compiled graph's facts are properties (F10) ---


def test_a_registry_is_a_container_of_its_nodes_by_id():
    registry = _registry()

    assert "scale" in registry and "nope" not in registry
    assert len(registry) == 2
    assert registry["scale"] is Scale
    assert list(registry) == ["scale", "shaped"]


def test_an_unknown_id_is_a_key_error_that_lists_the_ids():
    with pytest.raises(KeyError, match=r"'scal'.*scale, shaped"):
        _registry()["scal"]


def test_the_order_and_the_decisions_are_properties():
    compiled = _compiled(GraphNode(id="s", type="scale", version=1))

    assert compiled.execution_order == ("s",)
    assert compiled.decisions == {}


# --- a binding reads as it is written (the From rename) --------------------------------


def test_from_and_static_construct_positionally_and_store_as_before():
    from conductor.graph.binding import From
    from conductor.ref import Ref

    edge = From("a.result", Ref("b", "result"))
    assert edge.refs == (Ref("a", "result"), Ref("b", "result"))
    assert edge.model_dump(mode="json") == {"refs": ["a.result", "b.result"]}
    assert Static(200).model_dump() == {"value": 200}
    assert GraphNode.model_validate(
        {"id": "n", "type": "scale", "version": 1, "bindings": {"value": {"refs": ["a.result"]}, "factor": {"value": "3"}}}
    ).bindings == {"value": From("a.result"), "factor": Static("3")}


def test_from_refuses_an_empty_edge():
    from conductor.graph.binding import From
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        From()
