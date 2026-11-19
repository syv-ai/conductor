"""The inputs and outputs of a node.

A node is a Python function. Each of its parameters is an input, and what it returns is one or more outputs. For
``shout(line: str, mark: str = "!") -> str``, the inputs are ``line`` and ``mark``, and the one output is
``result``. Conductor calls each of these a ``Field``: a name, a type hint and maybe a default.

The package has four parts:

- ``Field`` (``type.py``) is one input or output, built from its hint.
- ``FieldDescription`` (``describe.py``) is the same field as JSON, for an editor to show.
- ``outputs.py`` reads a value for each output of a node, by name: what a node returns, a person's answer, and an
  ``Ask``'s default all fill the outputs by its rules, the rule for branches among them.
- ``validator.py`` checks the default a node gives a field, when the node is declared and again when a graph is
  compiled.
"""

from .describe import FieldDescription
from .outputs import outputs_schema
from .type import Field
from .validator import BoundDefaultValidator, DeclaredInputValidator, DeclaredOutputValidator

__all__ = [
    "BoundDefaultValidator",
    "DeclaredInputValidator",
    "DeclaredOutputValidator",
    "Field",
    "FieldDescription",
    "outputs_schema",
]
