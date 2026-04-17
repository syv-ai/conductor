"""Tests for the extended widget set and the type → default widget dispatch.

Covers:
- Each new widget class (``Number``, ``Switch``, ``DatePicker``,
  ``Multiselect``, ``List``, ``SchemaBuilder``, ``CodeEditor``,
  ``TemplateTextarea``, ``EntityDropdown``, ``IfElseBuilder``).
- The default-widget mapping from Python types when no ``Widget`` is
  attached to the parameter.
- Explicit widgets on an ``Annotated[...]`` always override the default.
"""

from __future__ import annotations

from typing import Annotated

from conductor import NodeRegistry
from conductor.types import Base64Str, Date, NamedFile, WidgetType
from conductor.widgets import (
    CodeEditor,
    DatePicker,
    Dropdown,
    EntityDropdown,
    IfElseBuilder,
    List,
    Multiselect,
    Number,
    Output,
    Range,
    SchemaBuilder,
    Switch,
    TemplateTextarea,
    Text,
    Textarea,
)

# =============================================================================
# New widget classes — shape of to_schema()
# =============================================================================


class TestNumberWidget:
    def test_defaults_are_omitted_from_schema_when_none(self):
        s = Number(label="n").to_schema()
        assert s["widget"] == WidgetType.NUMBER.value
        assert "min_val" not in s
        assert "max_val" not in s
        assert "step" not in s
        assert s["integer_only"] is False

    def test_bounds_and_integer_flag_surface(self):
        s = Number(label="n", min_val=0, max_val=10, step=1, integer_only=True).to_schema()
        assert s["min_val"] == 0
        assert s["max_val"] == 10
        assert s["step"] == 1
        assert s["integer_only"] is True


class TestSwitchWidget:
    def test_switch_emits_correct_widget_type(self):
        s = Switch(label="x").to_schema()
        assert s["widget"] == WidgetType.SWITCH.value


class TestDatePickerWidget:
    def test_date_range_optional(self):
        s = DatePicker(label="d").to_schema()
        assert s["widget"] == WidgetType.DATEPICKER.value
        assert "min_date" not in s
        assert "max_date" not in s

    def test_date_range_surfaces(self):
        s = DatePicker(label="d", min_date="2024-01-01", max_date="2024-12-31").to_schema()
        assert s["min_date"] == "2024-01-01"
        assert s["max_date"] == "2024-12-31"


class TestMultiselectWidget:
    def test_choices_and_bounds_surface(self):
        s = Multiselect(
            label="tags", choices=["a", "b", "c"], min_selected=1, max_selected=2,
        ).to_schema()
        assert s["widget"] == WidgetType.MULTISELECT.value
        assert s["choices"] == ["a", "b", "c"]
        assert s["min_selected"] == 1
        assert s["max_selected"] == 2


class TestListWidget:
    def test_emits_item_widget_null_when_unspecified(self):
        s = List(label="xs").to_schema()
        assert s["widget"] == WidgetType.LIST.value
        assert s["item_widget"] is None

    def test_item_widget_nested_into_schema(self):
        s = List(label="xs", item_widget=Text(label="item"), max_items=5).to_schema()
        assert s["item_widget"]["widget"] == WidgetType.TEXT.value
        assert s["item_widget"]["label"] == "item"
        assert s["max_items"] == 5


class TestSchemaBuilderWidget:
    def test_allow_additional_defaults_to_true(self):
        s = SchemaBuilder(label="config").to_schema()
        assert s["widget"] == WidgetType.SCHEMA_BUILDER.value
        assert s["allow_additional"] is True
        assert "schema" not in s

    def test_schema_surface(self):
        s = SchemaBuilder(
            label="config",
            schema={"type": "object", "properties": {"x": {"type": "string"}}},
            allow_additional=False,
        ).to_schema()
        assert s["schema"]["type"] == "object"
        assert s["allow_additional"] is False


class TestCodeEditorWidget:
    def test_language_surface(self):
        s = CodeEditor(label="c", language="sql").to_schema()
        assert s["widget"] == WidgetType.CODE_EDITOR.value
        assert s["language"] == "sql"

    def test_language_defaults_to_python(self):
        s = CodeEditor(label="c").to_schema()
        assert s["language"] == "python"


class TestTemplateTextareaWidget:
    def test_variables_default_empty(self):
        s = TemplateTextarea(label="t").to_schema()
        assert s["widget"] == WidgetType.TEMPLATE_TEXTAREA.value
        assert s["variables"] == []
        assert s["rows"] == 4

    def test_variables_surface(self):
        s = TemplateTextarea(label="t", variables=["name", "date"]).to_schema()
        assert s["variables"] == ["name", "date"]


class TestEntityDropdownWidget:
    def test_entity_kind_surface(self):
        s = EntityDropdown(label="user", entity_kind="user", multiple=True).to_schema()
        assert s["widget"] == WidgetType.ENTITY_DROPDOWN.value
        assert s["entity_kind"] == "user"
        assert s["multiple"] is True


class TestIfElseBuilderWidget:
    def test_variables_surface(self):
        s = IfElseBuilder(label="cond", variables=["a", "b"]).to_schema()
        assert s["widget"] == WidgetType.IF_ELSE_BUILDER.value
        assert s["variables"] == ["a", "b"]


# =============================================================================
# Type → default widget dispatch
# =============================================================================


class TestDefaultWidgetDispatch:
    """When a parameter has no Widget on its ``Annotated[...]`` (or no
    ``Annotated`` at all), the registry should infer a sensible default."""

    def _first_input(self, func, registry: NodeRegistry | None = None):
        reg = registry or NodeRegistry()
        decorated = reg.node("n", version=1, name="N", description="desc")(func)
        assert decorated is func
        node_def = reg.get("n@1")
        assert node_def is not None
        return node_def.inputs[0]

    def test_plain_str_defaults_to_text(self):
        def f(x: str) -> Annotated[str, Output(label="o")]:
            return x
        inp = self._first_input(f)
        assert inp.widget == WidgetType.TEXT

    def test_plain_int_defaults_to_number_integer_only(self):
        def f(x: int) -> Annotated[str, Output(label="o")]:
            return str(x)
        inp = self._first_input(f)
        assert inp.widget == WidgetType.NUMBER
        assert inp.widget_config["integer_only"] is True

    def test_plain_float_defaults_to_number_non_integer(self):
        def f(x: float) -> Annotated[str, Output(label="o")]:
            return str(x)
        inp = self._first_input(f)
        assert inp.widget == WidgetType.NUMBER
        assert inp.widget_config["integer_only"] is False

    def test_plain_bool_defaults_to_checkbox(self):
        def f(x: bool) -> Annotated[str, Output(label="o")]:
            return str(x)
        inp = self._first_input(f)
        assert inp.widget == WidgetType.CHECKBOX

    def test_date_alias_defaults_to_datepicker(self):
        def f(x: Date) -> Annotated[str, Output(label="o")]:
            return x
        inp = self._first_input(f)
        assert inp.widget == WidgetType.DATEPICKER

    def test_base64str_defaults_to_fileupload(self):
        def f(x: Base64Str) -> Annotated[str, Output(label="o")]:
            return x
        inp = self._first_input(f)
        assert inp.widget == WidgetType.FILE

    def test_namedfile_defaults_to_fileupload(self):
        def f(x: NamedFile) -> Annotated[str, Output(label="o")]:
            return "ok"
        inp = self._first_input(f)
        assert inp.widget == WidgetType.FILE

    def test_list_str_defaults_to_list_with_inner_text(self):
        def f(xs: list[str]) -> Annotated[str, Output(label="o")]:
            return ",".join(xs)
        inp = self._first_input(f)
        assert inp.widget == WidgetType.LIST
        assert inp.widget_config["item_widget"]["widget"] == WidgetType.TEXT.value

    def test_list_int_defaults_to_list_with_inner_number(self):
        def f(xs: list[int]) -> Annotated[str, Output(label="o")]:
            return str(sum(xs))
        inp = self._first_input(f)
        assert inp.widget == WidgetType.LIST
        assert inp.widget_config["item_widget"]["widget"] == WidgetType.NUMBER.value
        assert inp.widget_config["item_widget"]["integer_only"] is True

    def test_bare_list_defaults_to_list_without_inner_widget(self):
        def f(xs: list) -> Annotated[str, Output(label="o")]:  # noqa: E501 — bare list
            return str(xs)
        inp = self._first_input(f)
        assert inp.widget == WidgetType.LIST
        assert inp.widget_config["item_widget"] is None

    def test_dict_defaults_to_schema_builder(self):
        def f(cfg: dict) -> Annotated[str, Output(label="o")]:
            return str(cfg)
        inp = self._first_input(f)
        assert inp.widget == WidgetType.SCHEMA_BUILDER

    def test_typed_dict_defaults_to_schema_builder(self):
        def f(cfg: dict[str, int]) -> Annotated[str, Output(label="o")]:
            return str(cfg)
        inp = self._first_input(f)
        assert inp.widget == WidgetType.SCHEMA_BUILDER

    def test_annotated_without_widget_still_gets_default(self):
        """An ``Annotated[str, "description only"]`` should still default to Text."""
        def f(x: Annotated[str, "a non-widget annotation"]) -> Annotated[str, Output(label="o")]:
            return x
        inp = self._first_input(f)
        assert inp.widget == WidgetType.TEXT

    def test_label_uses_param_name_when_no_widget_given(self):
        def f(my_field: str) -> Annotated[str, Output(label="o")]:
            return my_field
        inp = self._first_input(f)
        assert inp.label == "my_field"


class TestExplicitWidgetOverridesDefault:
    """Explicit ``Annotated[T, Widget(...)]`` always wins over the default."""

    def test_textarea_on_str_overrides_text_default(self):
        reg = NodeRegistry()

        @reg.node("n", version=1, name="N", description="d")
        def f(x: Annotated[str, Textarea(label="Content", rows=8)]) -> Annotated[str, Output(label="o")]:
            return x

        inp = reg.get("n@1").inputs[0]
        assert inp.widget == WidgetType.TEXTAREA
        assert inp.widget_config["rows"] == 8

    def test_range_on_int_overrides_number_default(self):
        reg = NodeRegistry()

        @reg.node("n", version=1, name="N", description="d")
        def f(x: Annotated[int, Range(label="Pick", min_val=0, max_val=10)]) -> Annotated[str, Output(label="o")]:
            return str(x)

        inp = reg.get("n@1").inputs[0]
        assert inp.widget == WidgetType.RANGE

    def test_dropdown_on_str_overrides_text_default(self):
        reg = NodeRegistry()

        @reg.node("n", version=1, name="N", description="d")
        def f(x: Annotated[str, Dropdown(label="Choose", choices=["a", "b"])]) -> Annotated[str, Output(label="o")]:
            return x

        inp = reg.get("n@1").inputs[0]
        assert inp.widget == WidgetType.DROPDOWN
        assert inp.widget_config["choices"] == ["a", "b"]

    def test_multiselect_on_list_str_overrides_list_default(self):
        reg = NodeRegistry()

        @reg.node("n", version=1, name="N", description="d")
        def f(
            xs: Annotated[list[str], Multiselect(label="Pick", choices=["a", "b", "c"])],
        ) -> Annotated[str, Output(label="o")]:
            return ",".join(xs)

        inp = reg.get("n@1").inputs[0]
        assert inp.widget == WidgetType.MULTISELECT
        assert inp.widget_config["choices"] == ["a", "b", "c"]


# =============================================================================
# End-to-end through the registry — execute a node that uses only defaults
# =============================================================================


class TestRegistryUsesDefaultsAtExecutionTime:
    def test_node_without_explicit_widgets_still_runs(self):
        """A node with only bare type hints (no Widget annotations) should
        still compile, execute, and pass its arguments through normally."""
        from conductor import GraphNode, compile
        from conductor.execution.engine import execute_sync

        reg = NodeRegistry()

        @reg.node("sum-ints", version=1, name="Sum", description="Adds two ints")
        def sum_ints(a: int, b: int) -> Annotated[int, Output(label="Total")]:
            return a + b

        compiled = compile(
            nodes=[GraphNode("n", "sum-ints@1", {"a": 3, "b": 4})],
            edges=[],
            registry=reg,
        )
        results = execute_sync(compiled)
        assert results["n"]["result"] == 7

        # And both params have sensible default widgets
        node_def = reg.get("sum-ints@1")
        assert {i.widget for i in node_def.inputs} == {WidgetType.NUMBER}
