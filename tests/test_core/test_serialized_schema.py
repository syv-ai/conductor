"""The serializers conform to their declared pydantic models."""

from conductor.registry.schema import (
    serialize_input,
    serialize_input_model,
    serialize_output,
    serialize_output_model,
)
from conductor.registry.serialized import SerializedInput, SerializedOutput

# Reuse the widget-instance factory from test_registry_serialization.
from test_core.test_registry_serialization import _all_widget_instances


def _metadata_for(widget):
    """Build an InputMetadata carrying the widget's config, as the
    registration decorator does (registry/__init__.py)."""
    from conductor.metadata import InputMetadata

    schema = widget.to_schema()
    return InputMetadata(
        name="felt",
        type_str="str",
        label=schema["label"],
        description=schema["description"],
        widget=schema["widget"],
        disable_handle=schema["disable_handle"],
        widget_config={
            k: v
            for k, v in schema.items()
            if k not in ("widget", "label", "description", "disable_handle")
        },
    )


def test_every_widget_serialization_validates() -> None:
    for widget in _all_widget_instances():
        meta = _metadata_for(widget)
        model = SerializedInput.model_validate(serialize_input(meta))
        assert model.name == "felt"
        # The typed accessor and the dict route agree exactly.
        assert serialize_input_model(meta) == model


def test_dict_and_model_routes_are_byte_identical() -> None:
    for widget in _all_widget_instances():
        meta = _metadata_for(widget)
        assert (
            serialize_input_model(meta).model_dump(exclude_none=True, exclude_defaults=True).keys()
            <= serialize_input(meta).keys()
        )


def test_serialized_output_covers_output_serializer() -> None:
    from conductor.metadata import OutputMetadata

    out = OutputMetadata(
        name="result", type_str="str", label="Resultat", download=True, filename="x.txt"
    )
    model = SerializedOutput.model_validate(serialize_output(out))
    assert model.download is True
    assert serialize_output_model(out) == model


def test_serialized_input_fields_pin_widget_schema_keys() -> None:
    """The model's widget-config fields ARE ``WIDGET_SCHEMA_KEYS``, exactly.

    ``SerializedInput`` is the typed authority for the wire; this pin keeps
    it in lockstep with the hand-declared constant (they live in different
    modules to stay import-cycle-free), so a widget that grows a new key
    can't land in one place and drift from the other. Combined with
    ``test_widget_schema_keys_covers_every_emitted_key`` (widgets ↔ constant),
    the chain widget.to_schema → WIDGET_SCHEMA_KEYS → SerializedInput is closed.
    """
    from conductor.registry.serialized import _BASE_INPUT_KEYS
    from conductor.widgets import WIDGET_SCHEMA_KEYS

    model_widget_keys = frozenset(SerializedInput.model_fields) - _BASE_INPUT_KEYS
    assert model_widget_keys == WIDGET_SCHEMA_KEYS


def test_serialize_node_model_validates_without_registry() -> None:
    """A host can serialize an ad-hoc definition without a registry."""
    from conductor.metadata import InputMetadata, OutputMetadata
    from conductor.registry.definition import NodeDefinition
    from conductor.registry.schema import serialize_node, serialize_node_model
    from conductor.registry.serialized import SerializedNode
    from conductor.types import NodeCategory

    nd = NodeDefinition(
        id="demo@1",
        base_id="demo",
        version=1,
        name="Demo",
        description="A demo node.",
        tags=("Flow",),
        category=NodeCategory("flow"),
        inputs=(InputMetadata(name="a", type_str="str", label="A"),),
        outputs=(OutputMetadata(name="result", type_str="str", label="R"),),
    )
    model = serialize_node_model(nd)
    assert isinstance(model, SerializedNode)
    # Registry-less: reported as the latest, non-deprecated.
    assert model.deprecated is False
    assert model.latest_version == 1
    assert model.category == "flow"
    assert model.inputs[0].name == "a"
    # The typed accessor and the dict route agree.
    assert model == SerializedNode.model_validate(serialize_node(nd))


def test_input_metadata_derives_expects_list_from_type() -> None:
    """``expects_list`` is filled from a ``list[...]`` type when not given."""
    from conductor.metadata import InputMetadata

    assert InputMetadata(name="xs", type_str="list[str]", label="Xs").expects_list is True
    assert InputMetadata(name="x", type_str="str", label="X").expects_list is False
    # An explicit value is never overridden.
    assert (
        InputMetadata(name="x", type_str="str", label="X", expects_list=True).expects_list is True
    )
