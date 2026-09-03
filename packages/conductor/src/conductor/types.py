"""Core types, enums, and constants for conductor."""

from typing import Any, NewType, TypeAlias, TypedDict

__all__ = [
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
# NewType/TypedDict aliases that behave as their base type (str, dict,
# list) at runtime. The old registry surfaced them as distinct type strings;
# a node now declares a ``DType``. The module goes with the old engine.
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
