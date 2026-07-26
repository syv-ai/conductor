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
            serialize_input_model(meta).model_dump(exclude_none=True, exclude_defaults=True)
            .keys()
            <= serialize_input(meta).keys()
        )


def test_serialized_output_covers_output_serializer() -> None:
    from conductor.metadata import OutputMetadata

    out = OutputMetadata(name="result", type_str="str", label="Resultat", download=True, filename="x.txt")
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
