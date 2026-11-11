"""``Field``, one input or output of a node, and ``FieldDescription``, the same as JSON for an editor."""

from conductor._internal.field.describe import FieldDescription
from conductor._internal.field.type import Field

__all__ = ["Field", "FieldDescription"]
