"""The inputs and outputs of a node.

A node is a Python function. Each of its parameters is an input, and what it returns is one or more outputs. For
``shout(line: str, mark: str = "!") -> str``, the inputs are ``line`` and ``mark``, and the one output is
``result``. Conductor calls each of these a ``Field``: a name, a type hint and maybe a default.

The package has four parts:

- ``Field`` (``type.py``) is one input or output, built from its hint.
- ``FieldDescription`` (``describe.py``) is the same field as JSON, for an editor to show.
- ``branch.py`` has the rule for branches, the outputs whose default is ``SKIPPED``: a call gives a value to at most
  one of them.
- ``validator.py`` checks the default a node gives a field, when the node is declared and again when a graph is
  compiled.
"""

from .branch import check_branches
from .describe import FieldDescription
from .type import Field
from .validator import BoundDefaultValidator, DeclaredInputValidator, DeclaredOutputValidator

__all__ = [
    "BoundDefaultValidator",
    "DeclaredInputValidator",
    "DeclaredOutputValidator",
    "Field",
    "FieldDescription",
    "check_branches",
]
