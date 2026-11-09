"""The surface a newcomer meets first: what a mistake says, what a record prints, and what is refused early.

One small test per rule. Each states what a person trips on and what they
see instead: a dead binding that fails loud, a stored graph that refuses a
misspelled key, a repr that reads back as a constructor, an error whose
text names what went wrong, a declaration refused where it is written.
"""

from collections.abc import Mapping
from typing import Annotated, Any, NewType

import pytest
from conductor import (
    CompiledGraph,
    DType,
    Graph,
    GraphNode,
    Input,
    NodeRegistry,
    Param,
    Result,
    Static,
    node,
)
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "surface-test-txt"
    display_title = "Text"


Out = Annotated[Txt, Result(title="Result")]


@node("scale", title="Scale", description="d", category="test")
def scale(
    value: Annotated[Txt, Param(title="Value", widget=Textarea())] = Txt(""),
    factor: Annotated[Txt, Param(title="Factor", widget=Textarea())] = Txt("2"),
) -> Out:
    return Txt(value * int(factor))


def shaped_inputs():
    return ()

@node("shaped", title="Shaped", description="d", category="test", compute_added_inputs=shaped_inputs)
def shaped(mode: Annotated[Txt, Param(title="Mode", widget=Textarea())] = Txt("a"), **more: Txt) -> Out:
    """Its inputs depend on a mode, so a binding can outlive its field."""
    return mode


def _registry() -> NodeRegistry:
    registry = NodeRegistry([scale, shaped])
    return registry


def _compiled(*nodes: GraphNode) -> CompiledGraph:
    return CompiledGraph.from_graph(Graph(nodes=list(nodes)), _registry())


# --- a dead binding ------------------------------------------------------------------


def test_a_misspelled_binding_on_a_node_whose_inputs_are_its_signature_is_fatal():
    """'facter' for 'factor' used to run with the default and a quiet note."""
    compiled = _compiled(GraphNode(id="s", type="scale", version=1, bindings={"facter": Static("3")}))

    (make_problem,) = compiled.problems
    assert (make_problem.type, make_problem.fatal, make_problem.loc[1]) == ("stale_binding", True, "facter")
    assert not compiled.is_runnable


def test_the_dead_binding_names_the_inputs_the_node_has():
    compiled = _compiled(GraphNode(id="s", type="scale", version=1, bindings={"facter": Static("3")}))

    assert compiled.problems[0].msg == "Field 'facter' is not an input of the node; it has factor, value."


def test_a_node_whose_inputs_come_and_go_keeps_the_binding_as_a_note():
    """A node with ``compute_added_inputs`` really can lose a field when its mode
    changes, so the binding it held is left over, not a typo."""
    compiled = _compiled(GraphNode(id="h", type="shaped", version=1, bindings={"gone": Static("x")}))

    (make_problem,) = compiled.problems
    assert (make_problem.type, make_problem.fatal) == ("unused_binding", False)
    assert compiled.is_runnable


# --- a stored record refuses a key it does not have ---------------------------------


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
    graph_node = GraphNode(id="s", type="scale", version=1, display={"position": {"x": 1}, "colour": "red"})
    assert graph_node.display["colour"] == "red"


# --- a repr is the call that makes the object ----------------------------------------


def test_an_index_prints_as_its_constructor():
    from conductor._internal.index import Index

    assert repr(Index("b", parent=Index("a"))) == "Index('b', parent=Index('a'))"


def test_a_node_version_prints_its_run_by_name():
    assert repr(scale.versions[1]).startswith("NodeVersion(body=scale, ")


def test_a_compiled_graph_prints_a_one_line_summary():
    compiled = _compiled(GraphNode(id="s", type="scale", version=1))
    assert repr(compiled) == "CompiledGraph(nodes=('s',), is_runnable=True, problems=0)"


def test_a_registry_prints_a_call_its_constructor_accepts():
    registry = NodeRegistry((scale, shaped))

    assert registry.nodes == (scale, shaped)
    assert repr(registry).startswith("NodeRegistry(nodes=(\n    NodeDefinition(id='scale'")


# --- an error's text names what went wrong --------------------------------------------


def test_a_compilation_error_lists_its_fatal_problems():
    """The message lists the problems, not only 'the graph cannot run'."""
    from conductor import InvalidCallError

    compiled = _compiled(GraphNode(id="s", type="scale", version=1, bindings={"facter": Static("3")}))
    with pytest.raises(InvalidCallError) as raised:
        compiled.run_sync()

    assert str(raised.value) == (
        "1 problem(s):\n"
        "  s.facter — stale_binding: Field 'facter' is not an input of the node; it has factor, value."
    )


# --- a declaration mistake is refused where it is written -----------------------------


def _declare(*, run):
    return node("bad", title="Bad", description="d", category="test")(run)


def _run(annotations: dict, **kinds):
    """A ``run`` with the given parameter annotations, built from source so a
    parameter kind (``*args``, ``**inputs``) can be spelled."""
    params = ", ".join(kinds.get(name, name) + f": A_{name}" for name in annotations)
    scope = {f"A_{name}": annotation for name, annotation in annotations.items()} | {"Out": Out, "Txt": Txt}
    exec(f"def run({params}) -> Out:\n    return Txt('')", scope)
    return scope["run"]


def test_a_bare_series_parameter_is_a_series_of_anything():

    from conductor import Series

    declared = _declare(run=_run({"rows": Annotated[Series, Param(title="Rows")]}))
    assert declared.versions[1].interface.inputs[0].dtype.id == "any" and declared.versions[1].interface.inputs[0].many


def test_a_type_no_edge_can_carry_on_an_open_interface_is_refused():
    with pytest.raises(TypeError, match=r"\*\*inputs"):
        _declare(run=_run({"inputs": list[str]}, inputs="**inputs"))


def test_an_underscore_parameter_is_refused():
    with pytest.raises(TypeError, match="'_hidden'"):
        _declare(run=_run({"_hidden": Annotated[Txt, Param(title="H", widget=Textarea())]}))


@pytest.mark.parametrize("name", ["schema", "copy", "json", "model_config"])
def test_a_parameter_named_like_a_pydantic_model_attribute_validates_without_a_warning(name):
    """``BaseModel`` has ``schema``, ``copy``, ``json`` and ``model_config``;
    a call's check keys its inputs by name in a dict, so an input may carry
    any of them, without a warning."""
    import warnings


    graph_node = _declare(run=_run({name: Annotated[Txt, Param(title="X", widget=Textarea())]}))
    registry = NodeRegistry((graph_node,))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n", type="bad", version=1, bindings={name: Static("x")})]), registry)
        assert compiled.node("n")._validate_call_inputs({name: Txt("x")}) == {name: "x"}
    assert compiled.run_sync().status == "complete"


def test_an_aliased_input_and_an_input_named_like_its_field_both_arrive():
    """``json`` and ``json_`` are two inputs, and neither value may drop the other."""
    text = Annotated[Txt, Param(title="X", widget=Textarea())]
    graph_node = _declare(run=_run({"json_": text, "json": text}))
    registry = NodeRegistry((graph_node,))
    compiled = CompiledGraph.from_graph(
        Graph(nodes=[GraphNode(id="n", type="bad", version=1, bindings={"json_": Static("a"), "json": Static("b")})]),
        registry,
    )

    assert compiled.node("n")._validate_call_inputs({"json_": Txt("a"), "json": Txt("b")}) == {"json_": "a", "json": "b"}


def test_args_stay_refused():
    with pytest.raises(TypeError, match=r"\*args"):
        _declare(run=_run({"args": Txt}, args="*args"))


# --- a frozen graph's node list is frozen too ----------------------------------------


def test_a_graphs_nodes_are_a_tuple():
    graph = Graph(nodes=[GraphNode(id="s", type="scale", version=1)])

    assert isinstance(graph.nodes, tuple)
    with pytest.raises(AttributeError):
        graph.nodes.append(GraphNode(id="t", type="scale", version=1))


# --- one refusal, raised -------------------------------------------------------------


def test_compile_problem_is_a_conductor_error_at_the_root():
    import conductor
    from conductor import ComputeHookError
    from conductor._internal.errors import ConductorError

    assert conductor.ComputeHookError is ComputeHookError and issubclass(ComputeHookError, ConductorError)


# --- a declared type is a type -------------------------------------------------------


@pytest.mark.parametrize("bogus", [42, "text", {"id": "text"}])
def test_a_field_refuses_a_dtype_that_is_not_a_type(bogus):
    """A value is not a type: ``Output(dtype=42)`` would dump ``dtype: null``,
    and a dumped ``{"id": ...}`` read back would set the dtype to a dict."""
    from conductor import Output
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="dtype"):
        Output("result", bogus, title="R")


Mode = NewType("Mode", str)
type Modes = list[str]


def test_a_field_takes_a_dtype_or_any():

    assert Input("x", Txt, title="X").dtype is Txt
    assert Input("x", Any, title="X").dtype.id == "any"


# --- the registry is a container, and a compiled graph's facts are properties ---------


def test_a_registry_is_a_container_of_its_nodes_by_id():
    registry = _registry()

    assert "scale" in registry and "nope" not in registry
    assert len(registry) == 2
    assert registry["scale"] is scale
    assert list(registry) == ["scale", "shaped"]


def test_a_registry_is_a_container_and_not_a_mapping():
    """It answers ``in``, ``[]``, ``len`` and iteration over its ids, and
    nothing more: no ``get`` returning ``None`` for a typo, no ``items``."""
    from collections.abc import Container

    registry = _registry()

    assert isinstance(registry, Container) and not isinstance(registry, Mapping)
    assert not hasattr(registry, "get") and not hasattr(registry, "items")


def test_an_unknown_id_is_a_key_error_that_lists_the_ids():
    with pytest.raises(KeyError, match=r"'scal'.*scale, shaped"):
        _registry()["scal"]


def test_the_order_and_the_decisions_are_properties():
    compiled = _compiled(GraphNode(id="s", type="scale", version=1))

    assert compiled.execution_order == ("s",)
    assert compiled.decisions == {}


# --- a binding reads as it is written -------------------------------------------------


def test_from_and_static_construct_positionally_and_store_as_before():
    from conductor import From, Ref

    edge = From("a.result", Ref("b", "result"))
    assert edge.refs == (Ref("a", "result"), Ref("b", "result"))
    assert edge.model_dump(mode="json") == {"refs": ["a.result", "b.result"]}
    assert Static(200).model_dump() == {"value": 200}
    assert GraphNode.model_validate(
        {"id": "n", "type": "scale", "version": 1, "bindings": {"value": {"refs": ["a.result"]}, "factor": {"value": "3"}}}
    ).bindings == {"value": From("a.result"), "factor": Static("3")}


def test_from_refuses_an_empty_edge():
    from conductor import From
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        From()
