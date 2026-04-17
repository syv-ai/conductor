"""Phase 1: Node registry — registration, lookup, versioning."""

from typing import Annotated

import pytest
from conductor.registry import NodeRegistry
from conductor.widgets import Output, Text


@pytest.fixture
def populated_registry():
    """Registry with a few nodes registered."""
    reg = NodeRegistry()

    @reg.node("echo", version=1, name="Echo", description="Returns input unchanged")
    def echo(
        text: Annotated[str, Text(label="Input")],
    ) -> Annotated[str, Output(label="Output")]:
        return text

    @reg.node("echo", version=2, name="Echo v2", description="Echo with prefix")
    def echo_v2(
        text: Annotated[str, Text(label="Input")],
        prefix: Annotated[str, Text(label="Prefix")] = "",
    ) -> Annotated[str, Output(label="Output")]:
        return f"{prefix}{text}"

    @reg.node("upper", version=1, name="Uppercase", description="Uppercases text")
    def upper(
        text: Annotated[str, Text(label="Input")],
    ) -> Annotated[str, Output(label="Output")]:
        return text.upper()

    return reg


class TestNodeRegistration:
    def test_register_via_decorator(self, registry):
        @registry.node("greet", version=1, name="Greet", description="Says hello")
        def greet(
            name: Annotated[str, Text(label="Name")],
        ) -> Annotated[str, Output(label="Greeting")]:
            return f"Hello {name}"

        node_def = registry.get("greet@1")
        assert node_def is not None
        assert node_def.name == "Greet"

    def test_decorator_returns_original_function(self, registry):
        """The decorated function is still callable as-is (raw function stored)."""

        @registry.node("greet", version=1, name="Greet", description="Says hello")
        def greet(
            name: Annotated[str, Text(label="Name")],
        ) -> Annotated[str, Output(label="Greeting")]:
            return f"Hello {name}"

        assert greet("World") == "Hello World"

    def test_duplicate_registration_raises(self, registry):
        @registry.node("dup", version=1, name="Dup", description="First")
        def first(x: Annotated[str, Text(label="X")]) -> Annotated[str, Output(label="Y")]:
            return x

        with pytest.raises(ValueError, match="already registered"):

            @registry.node("dup", version=1, name="Dup2", description="Second")
            def second(x: Annotated[str, Text(label="X")]) -> Annotated[str, Output(label="Y")]:
                return x


class TestNodeLookup:
    def test_get_by_full_id(self, populated_registry):
        node = populated_registry.get("echo@1")
        assert node is not None
        assert node.id == "echo@1"
        assert node.base_id == "echo"
        assert node.version == 1

    def test_get_returns_none_for_unknown(self, populated_registry):
        assert populated_registry.get("nonexistent@1") is None

    def test_get_latest(self, populated_registry):
        latest = populated_registry.get_latest("echo")
        assert latest is not None
        assert latest.version == 2
        assert latest.id == "echo@2"

    def test_get_latest_returns_none_for_unknown(self, populated_registry):
        assert populated_registry.get_latest("nonexistent") is None

    def test_is_deprecated(self, populated_registry):
        assert populated_registry.is_deprecated("echo@1") is True
        assert populated_registry.is_deprecated("echo@2") is False
        assert populated_registry.is_deprecated("upper@1") is False

    def test_all_returns_every_version(self, populated_registry):
        all_nodes = populated_registry.all()
        ids = {n.id for n in all_nodes}
        assert "echo@1" in ids
        assert "echo@2" in ids
        assert "upper@1" in ids

    def test_all_current_returns_only_latest(self, populated_registry):
        current = populated_registry.all_current()
        ids = {n.id for n in current}
        assert "echo@2" in ids
        assert "upper@1" in ids
        assert "echo@1" not in ids


class TestNodeDefinitionMetadata:
    def test_inputs_extracted_from_signature(self, populated_registry):
        node = populated_registry.get("echo@1")
        assert len(node.inputs) == 1
        assert node.inputs[0].name == "text"
        assert node.inputs[0].label == "Input"

    def test_outputs_extracted_from_return_type(self, populated_registry):
        node = populated_registry.get("echo@1")
        assert len(node.outputs) >= 1
        assert node.outputs[0].label == "Output"

    def test_multi_input_node(self, populated_registry):
        node = populated_registry.get("echo@2")
        assert len(node.inputs) == 2
        names = [inp.name for inp in node.inputs]
        assert "text" in names
        assert "prefix" in names

    def test_default_values_captured(self, populated_registry):
        node = populated_registry.get("echo@2")
        prefix_input = next(inp for inp in node.inputs if inp.name == "prefix")
        assert prefix_input.default == ""
        assert prefix_input.optional is True

    def test_node_definition_is_frozen(self, populated_registry):
        node = populated_registry.get("echo@1")
        with pytest.raises(AttributeError):
            node.name = "Changed"

    def test_raw_function_stored(self, populated_registry):
        """NodeDefinition stores the original unwrapped function."""
        node = populated_registry.get("echo@1")
        assert node.func is not None
        assert node.func("hello") == "hello"

    def test_validation_model_created(self, populated_registry):
        """A Pydantic model is auto-generated for input validation."""
        node = populated_registry.get("echo@1")
        assert node.validation_model is not None
        validated = node.validation_model(text="hello")
        assert validated.model_dump() == {"text": "hello"}


class TestMultiOutputNode:
    def test_tuple_return_creates_multi_outputs(self, registry):
        @registry.node("split", version=1, name="Split", description="Splits text")
        def split(
            text: Annotated[str, Text(label="Input")],
        ) -> tuple[
            Annotated[str, Output(label="First half")],
            Annotated[str, Output(label="Second half")],
        ]:
            mid = len(text) // 2
            return text[:mid], text[mid:]

        node = registry.get("split@1")
        assert len(node.outputs) == 2
        assert node.outputs[0].label == "First half"
        assert node.outputs[1].label == "Second half"
