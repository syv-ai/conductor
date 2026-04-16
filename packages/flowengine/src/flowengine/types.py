"""Core types, enums, and constants for flowengine."""

from enum import Enum
from typing import Any, NewType, TypeAlias, TypedDict


class WidgetType(str, Enum):
    """All widget types for node parameters. Maps 1:1 to frontend components."""

    TEXT = "text"
    TEXTAREA = "textarea"
    DROPDOWN = "dropdown"
    DEPENDENT_DROPDOWN = "dependent-dropdown"
    RANGE = "range"
    CHECKBOX = "checkbox"
    FILE = "file"
    SCHEMA_BUILDER = "schema-builder"
    DATEPICKER = "datepicker"
    NUMBER = "number"
    SWITCH = "switch"
    CONNECTION_LIST = "connection-list"
    TEMPLATE_TEXTAREA = "template-textarea"
    IF_ELSE_BUILDER = "if-else-builder"
    MULTISELECT = "multiselect"
    ENTITY_DROPDOWN = "entity-dropdown"
    CODE_EDITOR = "code-editor"
    OUTPUT = "output"


class ResultFormat(str, Enum):
    """How node results are wrapped in the container format."""

    SINGLE = "single"
    MULTI = "multi"
    DICT_SPREAD = "dict"


class NodeCategory(str, Enum):
    """Node category — metadata for frontend styling and palette grouping."""

    IO = "io"
    CONTROL = "control"


RESULT_KEY: str = "result"
OUTPUT_PREFIX: str = "output_"
NodeResult: TypeAlias = dict[str, Any]


# =============================================================================
# Custom type aliases for nodes
#
# These are NewType/TypedDict aliases that:
# - At runtime, behave as their base type (str, dict, list)
# - In the registry schema, surface as distinct type strings for the frontend
#   (e.g., "base64str", "namedfile") so it can pick the right widget/rendering
# - Are fully extensible — host apps can define their own NewType aliases
#
# To create a custom type:
#   MyType = NewType("MyType", str)
#   # In the registry JSON, this becomes type_str="mytype"
#   # The frontend matches on "mytype" to render a custom widget
# =============================================================================

# Base64-encoded string (typically for file uploads)
Base64Str = NewType("Base64Str", str)

# ISO 8601 date string (YYYY-MM-DD)
Date = NewType("Date", str)


class NamedFile(TypedDict):
    """A file with its content and original filename."""

    content: str   # Base64-encoded file content
    filename: str  # Original filename (e.g., "document.pdf")


MultiNamedFile = NewType("MultiNamedFile", list[NamedFile])
