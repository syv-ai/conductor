"""Core types, enums, and constants for conductor."""

from typing import Any, NewType, TypeAlias, TypedDict

# `NodeCategory` lives in its own module so it can carry registration
# decorators without creating a circular import with `conductor.registry`.
# Re-exported here to keep the historical `from conductor.types import
# NodeCategory` import path working.
from conductor.category import NodeCategory  # noqa: F401  (re-export)

__all__ = [
    "NodeCategory",
    "NodeResult",
    "Base64Str",
    "Date",
    "NamedFile",
    "MultiNamedFile",
    "RESULT_KEY",
    "OUTPUT_PREFIX",
]



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
