"""``Field``, one input or output of a node, and ``FieldDescription``, the same as JSON for an editor.

``type.py`` is ``Field``; ``describe.py`` is ``FieldDescription``; ``exclusive_group.py`` the rule of an exclusive
group of outputs, exactly one of them with a value on a call; ``validator.py`` reads a default a node declares by its
field's type, and again by the type an edge binds its type variable to.
"""

from .describe import FieldDescription
from .exclusive_group import exclusive_groups, groups_without_one_value
from .type import Field
from .validator import BoundDefaultValidator, DeclaredInputValidator, DeclaredOutputValidator

__all__ = [
    "BoundDefaultValidator",
    "DeclaredInputValidator",
    "DeclaredOutputValidator",
    "Field",
    "FieldDescription",
    "exclusive_groups",
    "groups_without_one_value",
]
