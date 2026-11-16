"""The inputs and outputs of a node.

A node is a Python function. Each of its parameters is an input, and what it returns is one or more outputs. For
``shout(line: str, mark: str = "!") -> str``, the inputs are ``line`` and ``mark``, and the one output is
``result``. Conductor calls each of these a ``Field``: a name, a type hint and maybe a default.

The package has four parts:

- ``Field`` (``type.py``) is one input or output, built from its hint.
- ``FieldDescription`` (``describe.py``) is the same field as JSON, for an editor to show.
- ``exclusive_group.py`` has the rule for outputs that are alternatives: exactly one of them gets a value on a call.
- ``validator.py`` checks the default a node gives a field, when the node is declared and again when a graph is
  compiled.
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
