"""Widget ABC and concrete widget classes for node parameter UI rendering."""

from abc import ABC
from dataclasses import dataclass, field
from typing import Any

from conductor.types import WidgetType

__all__ = [
    "Widget",
    "WidgetType",
    # Core single-value widgets
    "Text",
    "Textarea",
    "Dropdown",
    "DependentDropdown",
    "Range",
    "Checkbox",
    "FileUpload",
    "ConnectionList",
    "Output",
    "Number",
    "Switch",
    "DatePicker",
    "Multiselect",
    "List",
    "SchemaBuilder",
    "CodeEditor",
    "TemplateTextarea",
    "EntityDropdown",
    "IfElseBuilder",
    # Tabular-data primitives
    "TableSource",
    "ConditionBuilder",
    "Tags",
    "ColumnSelect",
    "TableInput",
]


@dataclass
class Widget(ABC):
    """Base widget class for node parameters.

    Widgets define how node parameters are rendered in the frontend UI
    and provide validation constraints for Pydantic.

    ``connection_input`` names another input on the same node whose
    incoming edges should be scanned for variables this widget can
    reference. ``TemplateTextarea`` and the future ``IfElseBuilder``
    use it to declare which other input drives variable autocomplete.
    """

    label: str
    description: str | None = None
    disable_handle: bool = False
    hidden_when: dict[str, list[str]] | None = None
    advanced: bool = False
    connection_input: str | None = None

    @property
    def widget_type(self) -> WidgetType:
        """Return the WidgetType enum for this widget."""
        raise NotImplementedError

    def to_schema(self) -> dict[str, Any]:
        """Convert to a JSON dict for frontend rendering."""
        schema: dict[str, Any] = {
            "widget": self.widget_type.value if isinstance(self.widget_type, WidgetType) else self.widget_type,
            "label": self.label,
            "description": self.description,
            "disable_handle": self.disable_handle,
        }
        if self.hidden_when is not None:
            schema["hidden_when"] = self.hidden_when
        if self.advanced:
            schema["advanced"] = True
        if self.connection_input is not None:
            schema["connection_input"] = self.connection_input
        return schema


@dataclass
class Text(Widget):
    """Single-line text input."""

    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.TEXT

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        if self.min_length is not None:
            schema["min_length"] = self.min_length
        if self.max_length is not None:
            schema["max_length"] = self.max_length
        if self.pattern is not None:
            schema["pattern"] = self.pattern
        return schema


@dataclass
class Textarea(Widget):
    """Multi-line text input."""

    min_length: int | None = None
    max_length: int | None = None
    rows: int = 4

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.TEXTAREA

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        schema["rows"] = self.rows
        if self.min_length is not None:
            schema["min_length"] = self.min_length
        if self.max_length is not None:
            schema["max_length"] = self.max_length
        return schema


@dataclass
class Dropdown(Widget):
    """Dropdown / select input."""

    disable_handle: bool = True
    choices: list[str] = field(default_factory=list)

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.DROPDOWN

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        schema["choices"] = self.choices
        return schema


@dataclass
class DependentDropdown(Widget):
    """Dropdown whose choices depend on another field's value."""

    disable_handle: bool = True
    depends_on: str = ""
    choices_map: dict[str, list[str]] = field(default_factory=dict)

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.DEPENDENT_DROPDOWN

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        schema["depends_on"] = self.depends_on
        schema["choices_map"] = self.choices_map
        return schema


@dataclass
class Range(Widget):
    """Numeric slider / range input.

    The Python attributes are ``min_val`` / ``max_val`` for parity with
    :class:`Number`, but the serialised schema uses ``range_min`` /
    ``range_max`` so frontends can distinguish slider bounds from a
    free-input numeric field's bounds without sniffing the widget type.
    """

    disable_handle: bool = True
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.RANGE

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        if self.min_val is not None:
            schema["range_min"] = self.min_val
        if self.max_val is not None:
            schema["range_max"] = self.max_val
        if self.step is not None:
            schema["step"] = self.step
        return schema


@dataclass
class Checkbox(Widget):
    """Boolean checkbox."""

    disable_handle: bool = True

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.CHECKBOX


@dataclass
class FileUpload(Widget):
    """File upload widget (returns base64 encoded content)."""

    disable_handle: bool = True
    accept: str | None = None
    max_size_mb: float | None = None
    multiple: bool = False

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.FILE

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        if self.accept is not None:
            schema["accept"] = self.accept
        if self.max_size_mb is not None:
            schema["max_size_mb"] = self.max_size_mb
        schema["multiple"] = self.multiple
        return schema


@dataclass
class ConnectionList(Widget):
    """Widget for accepting multiple connections from other nodes."""

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.CONNECTION_LIST


@dataclass
class Output(Widget):
    """Output widget for node return values."""

    download: bool = False
    filename: str | None = None

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.OUTPUT

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        schema["download"] = self.download
        if self.filename is not None:
            schema["filename"] = self.filename
        return schema


# ---------------------------------------------------------------------------
# Extended widgets — one concrete class per WidgetType enum value so the
# frontend can render every widget by reading the registry, without any
# backend-side guesswork. Each widget's ``to_schema()`` is complete.
# ---------------------------------------------------------------------------


@dataclass
class Number(Widget):
    """Free-input numeric field. Prefer ``Range`` for slider-style.

    Distinct from ``Range`` because a text-input numeric field and a
    slider are different UX — one is for arbitrary values, the other for
    bounded, picked-from-a-range values.
    """

    disable_handle: bool = True
    min_val: float | None = None
    max_val: float | None = None
    step: float | None = None
    integer_only: bool = False

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.NUMBER

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        if self.min_val is not None:
            schema["min_val"] = self.min_val
        if self.max_val is not None:
            schema["max_val"] = self.max_val
        if self.step is not None:
            schema["step"] = self.step
        schema["integer_only"] = self.integer_only
        return schema


@dataclass
class Switch(Widget):
    """Boolean toggle switch. Semantic sibling of ``Checkbox``.

    Frontends that render toggles differently from checkboxes pick
    ``Switch`` vs ``Checkbox`` based on the widget type; the value is a
    plain bool either way.
    """

    disable_handle: bool = True

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.SWITCH


@dataclass
class DatePicker(Widget):
    """Calendar / date input. Values are ISO-8601 date strings ("YYYY-MM-DD")."""

    disable_handle: bool = True
    min_date: str | None = None
    max_date: str | None = None

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.DATEPICKER

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        if self.min_date is not None:
            schema["min_date"] = self.min_date
        if self.max_date is not None:
            schema["max_date"] = self.max_date
        return schema


@dataclass
class Multiselect(Widget):
    """Pick zero or more values from a fixed set."""

    disable_handle: bool = True
    choices: list[str] = field(default_factory=list)
    min_selected: int | None = None
    max_selected: int | None = None

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.MULTISELECT

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        schema["choices"] = self.choices
        if self.min_selected is not None:
            schema["min_selected"] = self.min_selected
        if self.max_selected is not None:
            schema["max_selected"] = self.max_selected
        return schema


@dataclass
class List(Widget):
    """User-authored array. Each item is edited with ``item_widget``.

    Distinct from ``ConnectionList``: ``List`` is for values the user
    types into the UI; ``ConnectionList`` is for values aggregated from
    multiple upstream edges.
    """

    item_widget: Widget | None = None
    min_items: int | None = None
    max_items: int | None = None

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.LIST

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        # If no item widget is given the frontend falls back to a text
        # input; we still emit ``item_widget: null`` so the absence is
        # explicit rather than missing.
        schema["item_widget"] = (
            self.item_widget.to_schema() if self.item_widget is not None else None
        )
        if self.min_items is not None:
            schema["min_items"] = self.min_items
        if self.max_items is not None:
            schema["max_items"] = self.max_items
        return schema


@dataclass
class SchemaBuilder(Widget):
    """Structured dict / object editor. Values are plain dicts.

    ``schema`` is an optional JSON-Schema-ish dict that hints at the
    expected keys and types. ``allow_additional`` controls whether the
    user may add keys beyond what the schema declares.
    """

    disable_handle: bool = True
    schema: dict[str, Any] | None = None
    allow_additional: bool = True

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.SCHEMA_BUILDER

    def to_schema(self) -> dict[str, Any]:
        out = super().to_schema()
        if self.schema is not None:
            out["schema"] = self.schema
        out["allow_additional"] = self.allow_additional
        return out


@dataclass
class CodeEditor(Widget):
    """Syntax-highlighted code editor. Value is a plain string."""

    language: str = "python"
    min_length: int | None = None
    max_length: int | None = None

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.CODE_EDITOR

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        schema["language"] = self.language
        if self.min_length is not None:
            schema["min_length"] = self.min_length
        if self.max_length is not None:
            schema["max_length"] = self.max_length
        return schema


@dataclass
class TemplateTextarea(Widget):
    """Textarea that is aware of a fixed set of interpolation variables.

    ``variables`` lists the names the user may reference. The frontend
    can render autocomplete, highlighting, or inline validation. The
    value is still a plain string at runtime; interpolation is the
    node author's responsibility.
    """

    rows: int = 4
    variables: list[str] = field(default_factory=list)

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.TEMPLATE_TEXTAREA

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        schema["rows"] = self.rows
        schema["variables"] = self.variables
        return schema


@dataclass
class EntityDropdown(Widget):
    """Dropdown whose choices are loaded by the host at render time.

    ``entity_kind`` names the resource the frontend should query
    (``"user"``, ``"project"``, …). The mapping from entity_kind to a
    concrete data source is host-application concern — this widget only
    declares the intent.

    The Python attribute is ``entity_kind`` but the serialised schema
    key is ``entity_type`` for parity with the AKA frontend contract.
    """

    disable_handle: bool = True
    entity_kind: str = ""
    multiple: bool = False

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.ENTITY_DROPDOWN

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        schema["entity_type"] = self.entity_kind
        schema["multiple"] = self.multiple
        return schema


@dataclass
class IfElseBuilder(Widget):
    """Conditional expression editor. Value shape is host-defined.

    A minimal skeleton so the frontend can offer a branch-builder UI.
    The concrete expression language lives in the host application.
    """

    disable_handle: bool = True
    variables: list[str] = field(default_factory=list)

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.IF_ELSE_BUILDER

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        schema["variables"] = self.variables
        return schema


# ---------------------------------------------------------------------------
# Tabular-data widgets — composite UI primitives shared across nodes that
# operate on table-shaped data. Like ``EntityDropdown`` and
# ``IfElseBuilder``, these are skeleton widgets: declared metadata only,
# with no Python-side execution semantics. The host application owns the
# concrete renderer and the persisted value shape.
# ---------------------------------------------------------------------------


@dataclass
class TableSource(Widget):
    """Composite tri-mode source picker for tabular data.

    Hosts typically render upload / library / manual modes and collapse
    to an upstream-source display when an edge is wired. Mode resolution
    is host-side; this widget declares no widget-specific config.
    """

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.TABLE_SOURCE


@dataclass
class ConditionBuilder(Widget):
    """Structured ``column / operator / value`` filter editor.

    Rows are evaluated left-to-right with logic chips between them.
    ``connection_input`` (inherited) declares which input's incoming
    edges feed the column / variable autocomplete.
    """

    disable_handle: bool = True

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.CONDITION_BUILDER


@dataclass
class Tags(Widget):
    """Chip-list input. The persisted value is ``list[str]``."""

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.TAGS


@dataclass
class ColumnSelect(Widget):
    """Dropdown whose choices come from a sibling input's column headers.

    ``depends_on`` names the input on the same node whose data shape
    drives the column choices.
    """

    disable_handle: bool = True
    depends_on: str = ""

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.COLUMN_SELECT

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        schema["depends_on"] = self.depends_on
        return schema


@dataclass
class TableInput(Widget):
    """Inline spreadsheet editor. Value is ``{columns: [...], rows: [...]}``."""

    min_rows: int = 1
    min_columns: int = 1

    @property
    def widget_type(self) -> WidgetType:
        return WidgetType.TABLE_INPUT

    def to_schema(self) -> dict[str, Any]:
        schema = super().to_schema()
        schema["min_rows"] = self.min_rows
        schema["min_columns"] = self.min_columns
        return schema
