"""Pre-computed node metadata (frozen dataclasses)."""

from dataclasses import dataclass, field
from typing import Any

from conductor.types import WidgetType


@dataclass(frozen=True)
class InputMetadata:
    """Pre-computed metadata for a single node input parameter."""

    name: str
    type_str: str
    label: str
    description: str | None = None
    widget: WidgetType = WidgetType.TEXT
    default: Any = None
    optional: bool = False
    expects_list: bool = False
    uses_connection_list: bool = False
    disable_handle: bool = False
    widget_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OutputMetadata:
    """Pre-computed metadata for a single node output."""

    name: str
    type_str: str
    label: str
    description: str | None = None
    optional: bool = False
    download: bool = False
    filename: str | None = None
