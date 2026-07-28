"""Pre-computed node metadata (frozen dataclasses)."""

from dataclasses import dataclass, field
from typing import Any

from conductor.types import WidgetType

__all__ = [
    "InputMetadata",
    "OutputMetadata",
]


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

    def __post_init__(self) -> None:
        # A ``list[...]`` input fans several upstream sources into a list. Derive
        # ``expects_list`` from the type so metadata built by hand carries the same
        # flag the registry derives at registration, rather than defaulting to
        # False and silently string-joining a fan-in. Only fills the default; an
        # explicit True is left untouched.
        if not self.expects_list and self.type_str.startswith("list["):
            object.__setattr__(self, "expects_list", True)


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
