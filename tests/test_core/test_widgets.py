"""Phase 1: Widget ABC and concrete widget classes."""

from typing import Annotated, get_args

from conductor.types import WidgetType
from conductor.widgets import (
    Checkbox,
    ConnectionList,
    Dropdown,
    FileUpload,
    Output,
    Range,
    Text,
    Textarea,
    Widget,
)


class TestWidgetABC:
    def test_widget_is_abstract(self):
        """Widget cannot be instantiated directly."""
        import abc

        assert abc.ABC in Widget.__mro__

    def test_widget_has_label(self):
        w = Text(label="Name")
        assert w.label == "Name"

    def test_widget_has_description(self):
        w = Text(label="Name", description="Enter your name")
        assert w.description == "Enter your name"

    def test_widget_has_disable_handle(self):
        w = Text(label="Name", disable_handle=True)
        assert w.disable_handle is True

    def test_widget_to_schema_returns_dict(self):
        w = Text(label="Name")
        schema = w.to_schema()
        assert isinstance(schema, dict)
        assert schema["label"] == "Name"
        assert schema["widget"] == WidgetType.TEXT


class TestConcreteWidgets:
    def test_text_widget(self):
        w = Text(label="Input")
        schema = w.to_schema()
        assert schema["widget"] == WidgetType.TEXT

    def test_textarea_widget(self):
        w = Textarea(label="Prompt", rows=4)
        schema = w.to_schema()
        assert schema["widget"] == WidgetType.TEXTAREA
        assert schema["rows"] == 4

    def test_dropdown_widget(self):
        w = Dropdown(label="Model", choices=["gpt-4", "gpt-3.5"])
        schema = w.to_schema()
        assert schema["widget"] == WidgetType.DROPDOWN
        assert schema["choices"] == ["gpt-4", "gpt-3.5"]

    def test_range_widget(self):
        w = Range(label="Temperature", min_val=0.0, max_val=2.0, step=0.1)
        schema = w.to_schema()
        assert schema["widget"] == WidgetType.RANGE

    def test_checkbox_widget(self):
        w = Checkbox(label="Verbose")
        schema = w.to_schema()
        assert schema["widget"] == WidgetType.CHECKBOX

    def test_file_upload_widget(self):
        w = FileUpload(label="Document")
        schema = w.to_schema()
        assert schema["widget"] == WidgetType.FILE

    def test_output_widget(self):
        w = Output(label="Result")
        schema = w.to_schema()
        assert schema["widget"] == WidgetType.OUTPUT

    def test_connection_list_widget(self):
        w = ConnectionList(label="Items")
        schema = w.to_schema()
        assert schema["widget"] == WidgetType.CONNECTION_LIST


class TestWidgetInAnnotated:
    """Widgets work correctly inside Annotated type hints."""

    def test_widget_extractable_from_annotated(self):
        hint = Annotated[str, Text(label="Name")]
        args = get_args(hint)
        assert args[0] is str
        assert isinstance(args[1], Text)
        assert args[1].label == "Name"

    def test_output_extractable_from_annotated(self):
        hint = Annotated[str, Output(label="Result")]
        args = get_args(hint)
        assert isinstance(args[1], Output)


class TestSchemaKeyAlignment:
    """Schema keys must align with the host frontend contract.

    The Python attribute names stay stable for in-Python ergonomics, but
    the serialised dict keys are the host-facing contract — they must
    match what the AKA frontend reads. Drift here costs an adapter on
    every payload boundary forever.
    """

    def test_range_emits_range_min_and_range_max_keys(self):
        from conductor.widgets import Range

        schema = Range(label="Temperature", min_val=0.0, max_val=2.0, step=0.1).to_schema()
        assert schema["range_min"] == 0.0
        assert schema["range_max"] == 2.0
        # Legacy keys must NOT leak into the schema.
        assert "min_val" not in schema
        assert "max_val" not in schema

    def test_entity_dropdown_emits_entity_type_key(self):
        from conductor.widgets import EntityDropdown

        schema = EntityDropdown(
            label="User", entity_kind="user", multiple=False,
        ).to_schema()
        assert schema["entity_type"] == "user"
        # Legacy key must NOT leak into the schema.
        assert "entity_kind" not in schema


class TestConnectionInputMetadata:
    """``connection_input`` is a base-class field exposed in the schema.

    Widgets that drive variable autocomplete (TemplateTextarea, the
    future IfElseBuilder) declare which other input on the same node
    feeds them via this field. Without it, AKA's frontend has to guess.
    """

    def test_connection_input_serialized(self):
        from conductor.widgets import Textarea

        schema = Textarea(
            label="Body", connection_input="inputs",
        ).to_schema()
        assert schema["connection_input"] == "inputs"

    def test_connection_input_omitted_when_unset(self):
        """The default ``None`` is not serialised — keeps payloads tidy."""
        from conductor.widgets import Textarea

        schema = Textarea(label="Body").to_schema()
        assert "connection_input" not in schema
