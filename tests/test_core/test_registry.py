"""``NodeRegistry`` — filing a node class under its id, looking it up, and
what its versions declare.

A registry holds classes: ``register(Cls)`` files the class under
``Cls.id``, ``get(id)`` gives it back, and every fact about a version's
fields is read off ``Cls.versions[n].interface``.
"""

from typing import Annotated, Any

import pytest
from conductor import (
    DType,
    NodeDeclarationError,
    NodeRegistry,
    Param,
    Result,
    Upgrade,
    node,
)
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "registry-test-text"
    display_title = "Text"


Out = Annotated[Txt, Result(title="Output")]


def echo_v1(text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Out:
    return text

@node("echo", title="Echo", description="Returns input unchanged", category="test", replaces=node("echo")(echo_v1), upgrade=Upgrade())
def echo(
    text: Annotated[Txt, Param(title="Input", widget=Textarea())],
    prefix: Annotated[Txt, Param(title="Prefix", widget=Textarea())] = Txt(""),
) -> Out:
    return Txt(f"{prefix}{text}")


@node("upper", title="Uppercase", description="Uppercases text", category="test")
def upper_node(text: Annotated[Txt, Param(title="Input", widget=Textarea())]) -> Out:
    return Txt(text.upper())


@pytest.fixture
def populated_registry() -> NodeRegistry:
    reg = NodeRegistry([echo, upper_node])
    return reg


def test_register_files_the_class_under_its_id(registry):
    registry = registry.with_nodes([echo])

    assert registry["echo"] is echo
    assert registry["echo"].title == "Echo"


def test_a_second_class_under_the_same_id_is_refused(registry):
    @node("dup", title="Dup", description="First", category="test")
    def first(x: Annotated[Txt, Param(title="X", widget=Textarea())]) -> Out:
        return x

    @node("dup", title="Dup2", description="Second", category="test")
    def second(x: Annotated[Txt, Param(title="X", widget=Textarea())]) -> Out:
        return x

    with pytest.raises(NodeDeclarationError, match="declared twice"):
        NodeRegistry([first, second])


def test_get_returns_none_for_unknown(populated_registry):
    assert "nonexistent" not in populated_registry


def test_inputs_extracted_from_signature(populated_registry):
    iface = populated_registry["echo"].versions[1].interface
    assert len(iface.inputs) == 1
    assert iface.inputs[0].name == "text"
    assert iface.inputs[0].title == "Input"


def test_outputs_extracted_from_return_type(populated_registry):
    iface = populated_registry["echo"].versions[1].interface
    assert [o.name for o in iface.outputs] == ["result"]
    assert iface.outputs[0].title == "Output"


def test_multi_input_node(populated_registry):
    iface = populated_registry["echo"].versions[2].interface
    assert [i.name for i in iface.inputs] == ["text", "prefix"]


def test_default_values_captured(populated_registry):
    iface = populated_registry["echo"].versions[2].interface
    text, prefix = iface.inputs
    assert text.required is True
    assert prefix.default == ""
    assert prefix.required is False


def test_var_keyword_is_not_an_input():
    """``**values`` is where the inputs a ``compute_added_inputs`` hook adds arrive;
    the hook declares them, so the signature walk skips the parameter."""

    @node("varkw", title="VarKw", description="Dynamic", category="test")
    def var_kw(text: Annotated[Txt, Param(title="Text", widget=Textarea())], **values: Any) -> Out:
        return text

    assert [i.name for i in var_kw.versions[1].interface.inputs] == ["text"]


def test_var_positional_is_refused():
    """A ``*args`` parameter has no name an edge could land on."""
    with pytest.raises(TypeError, match=r"\*args names none"):

        @node("varpos", title="VarPos", description="Dynamic", category="test")
        def var_pos(text: Annotated[Txt, Param(title="Text", widget=Textarea())], *args: Any) -> Out:
            return text


def test_a_node_that_is_only_var_keyword_declares_no_inputs():
    @node("allkw", title="AllKw", description="Dynamic", category="test")
    def all_kw(**values: Any) -> Out:
        return Txt("x")

    assert all_kw.versions[1].interface.inputs == ()


def _node(node_id):
    @node(node_id, title=node_id, description="d", category="test")
    def made(x: Annotated[Txt, Param(title="X", widget=Textarea())] = Txt("")) -> Annotated[Txt, Result(title="R")]:
        return x

    return made


def test_extending_leaves_the_original_alone():
    """A per-run registry must not mutate the process-wide catalog."""
    base = NodeRegistry()
    base = base.with_nodes([_node("static")])
    loaded = _node("loaded")

    extended = base.with_nodes([loaded])

    assert "loaded" in extended
    assert "static" in extended
    assert "loaded" not in base


def test_a_registered_type_cannot_be_shadowed():
    """A statically registered node means what it says, whatever a host loads."""
    static = _node("shared-id")
    base = NodeRegistry([static])

    extended = base.with_nodes([_node("other")])

    assert extended["shared-id"] is static


def test_extending_with_nothing_is_the_same_registry_contents():
    base = NodeRegistry()
    base = base.with_nodes([_node("only")])

    assert base.with_nodes([])["only"] is base["only"]


def test_a_registry_lists_its_nodes_as_the_classes_in_registration_order():
    registry = NodeRegistry()
    greet, truncate = _node("greet"), _node("truncate")
    registry = registry.with_nodes([greet])
    registry = registry.with_nodes([truncate])

    assert registry.nodes == (greet, truncate)
    assert [cls.id for cls in registry.nodes] == ["greet", "truncate"]
