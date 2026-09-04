"""``NodeRegistry`` — filing a node class under its id, looking it up, and
what its versions declare.

A registry holds classes: ``register(Cls)`` files the class under
``Cls.id``, ``get(id)`` gives it back, and every fact about a version's
fields is read off ``Cls.versions[n].interface``.
"""

from typing import Annotated, Any

import pytest
from conductor.dtype import DType
from conductor.node import NodeDefinition, version
from conductor.registry import NodeRegistry
from conductor.returns import Result
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "registry-test-text"
    title = "Text"


Out = Annotated[Txt, Result(title="Output")]


class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "Returns input unchanged"
    category = "test"

    @version(1)
    def run_v1(self, text: Annotated[Txt, Textarea(title="Input")]) -> Out:
        return text

    @version(2)
    def run(
        self,
        text: Annotated[Txt, Textarea(title="Input")],
        prefix: Annotated[Txt, Textarea(title="Prefix")] = Txt(""),
    ) -> Out:
        return Txt(f"{prefix}{text}")


class Upper(NodeDefinition):
    id = "upper"
    title = "Uppercase"
    description = "Uppercases text"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="Input")]) -> Out:
        return Txt(text.upper())


@pytest.fixture
def populated_registry() -> NodeRegistry:
    reg = NodeRegistry()
    reg.register(Echo)
    reg.register(Upper)
    return reg


def test_register_files_the_class_under_its_id(registry):
    registry.register(Echo)

    assert registry.get("echo") is Echo
    assert registry.get("echo").title == "Echo"


def test_a_second_class_under_the_same_id_is_refused(registry):
    class First(NodeDefinition):
        id = "dup"
        title = "Dup"
        description = "First"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")]) -> Out:
            return x

    class Second(NodeDefinition):
        id = "dup"
        title = "Dup2"
        description = "Second"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")]) -> Out:
            return x

    registry.register(First)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(Second)


def test_get_returns_none_for_unknown(populated_registry):
    assert populated_registry.get("nonexistent") is None


def test_inputs_extracted_from_signature(populated_registry):
    iface = populated_registry.get("echo").versions[1].interface
    assert len(iface.inputs) == 1
    assert iface.inputs[0].name == "text"
    assert iface.inputs[0].title == "Input"


def test_outputs_extracted_from_return_type(populated_registry):
    iface = populated_registry.get("echo").versions[1].interface
    assert [o.name for o in iface.outputs] == ["result"]
    assert iface.outputs[0].title == "Output"


def test_multi_input_node(populated_registry):
    iface = populated_registry.get("echo").versions[2].interface
    assert [i.name for i in iface.inputs] == ["text", "prefix"]


def test_default_values_captured(populated_registry):
    iface = populated_registry.get("echo").versions[2].interface
    text, prefix = iface.inputs
    assert text.optional is False
    assert prefix.default == ""
    assert prefix.optional is True


def test_var_keyword_is_not_an_input():
    """``**values`` is where the inputs a ``compute_inputs`` hook adds arrive;
    the hook declares them, so the signature walk skips the parameter."""

    class VarKw(NodeDefinition):
        id = "varkw"
        title = "VarKw"
        description = "Dynamic"
        category = "test"

        def run(self, text: Annotated[Txt, Textarea(title="Text")], **values: Any) -> Out:
            return text

    assert [i.name for i in VarKw.versions[1].interface.inputs] == ["text"]


def test_var_positional_is_refused():
    """A ``*args`` parameter has no widget and no name a wire could land on."""
    with pytest.raises(TypeError, match="declares no widget"):

        class VarPos(NodeDefinition):
            id = "varpos"
            title = "VarPos"
            description = "Dynamic"
            category = "test"

            def run(self, text: Annotated[Txt, Textarea(title="Text")], *args: Any) -> Out:
                return text


def test_a_node_that_is_only_var_keyword_declares_no_inputs():
    class AllKw(NodeDefinition):
        id = "allkw"
        title = "AllKw"
        description = "Dynamic"
        category = "test"

        def run(self, **values: Any) -> Out:
            return Txt("x")

    assert AllKw.versions[1].interface.inputs == ()


def _node(node_id):
    class Made(NodeDefinition):
        id = node_id
        title = node_id
        description = "d"
        category = "test"

        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Annotated[Txt, Result(title="R")]:
            return x

    return Made


def test_extending_leaves_the_original_alone():
    """A per-run registry must not mutate the process-wide catalog."""
    base = NodeRegistry()
    base.register(_node("static"))
    loaded = _node("loaded")

    extended = base.extended_with({"loaded": loaded})

    assert extended.contains("loaded")
    assert extended.contains("static")
    assert not base.contains("loaded")


def test_a_registered_type_cannot_be_shadowed():
    """A statically registered node means what it says, whatever a host loads."""
    static = _node("shared-id")
    base = NodeRegistry()
    base.register(static)

    extended = base.extended_with({"shared-id": _node("other")})

    assert extended.get("shared-id") is static


def test_a_loaded_definition_need_not_number_from_one():
    """An embedded flow is one FlowVersion, loaded because a graph pinned it;
    its versions are {3} and nothing is missing. `register()` refuses that;
    `extended_with` does not, because the catalog rule is the catalog's."""
    from conductor.node import version

    class Loaded(NodeDefinition):
        id = "loaded-3"
        title = "Loaded"
        description = "d"
        category = "test"

        @version(3)
        def run(self, x: Annotated[Txt, Textarea(title="X")] = Txt("")) -> Annotated[Txt, Result(title="R")]:
            return x

    extended = NodeRegistry().extended_with({"loaded-3": Loaded})

    assert extended.get("loaded-3").versions.keys() == {3}


def test_extending_with_nothing_is_the_same_registry_contents():
    base = NodeRegistry()
    base.register(_node("only"))

    assert base.extended_with({}).get("only") is base.get("only")


def test_conductor_names_no_loading_seam():
    """Loading needs a database, and conductor has no idea what one is."""
    import conductor
    import conductor.graph.compiler as compiler
    import conductor.registry as registry_pkg

    for gone in ("resolve", "RegistryView", "DefinitionSource", "ExtensionResolver"):
        assert not hasattr(conductor, gone), gone
        assert not hasattr(registry_pkg, gone), gone
        assert not hasattr(compiler, gone), gone
