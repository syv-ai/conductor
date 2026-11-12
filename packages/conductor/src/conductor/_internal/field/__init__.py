"""``Field``, one input or output of a node, and ``FieldDescription``, the same as JSON for an editor."""

from .describe import FieldDescription
from .type import Field

__all__ = ["Field", "FieldDescription"]
