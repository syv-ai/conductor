"""Widget ABC and concrete widget classes for node parameter UI rendering."""

from abc import ABC
from dataclasses import dataclass, field
from typing import Any

from conductor.types import WidgetType


@dataclass
class Widget(ABC):
    """Base widget class for node parameters.

    Widgets define how node parameters are rendered in the frontend UI
    and provide validation constraints for Pydantic.
    """

    label: str
    description: str | None = None
    disable_handle: bool = False
    hidden_when: dict[str, list[str]] | None = None
    advanced: bool = False

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
    """Numeric slider / range input."""

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
            schema["min_val"] = self.min_val
        if self.max_val is not None:
            schema["max_val"] = self.max_val
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
