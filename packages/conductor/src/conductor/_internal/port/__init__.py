"""The inputs and outputs of a node.

A node is a Python function. Each of its parameters is an input, and what it returns is one or more outputs. For
``shout(line: str, mark: str = "!") -> str``, the inputs are ``line`` and ``mark``, and the one output is
``result``. Conductor calls each of these a port: a name, a type hint and maybe a default.

The package has four parts:

- ``Port`` (``type.py``) is one input or output, built from its hint.
- ``PortDescription`` (``describe.py``) is the same port as JSON, for an editor to show.
- ``outputs.py`` reads a value for each output of a node, by name: what a node returns, a person's answer, and an
  ``Ask``'s default all fill the outputs by its rules, the rules for branches and for ``SKIPPED`` among them, and
  ``ReturnShape`` says how a node's function returns them.
- ``validator.py`` checks the default a node gives a port, when the node is declared and again when a graph is
  compiled.
"""

from .describe import PortDescription
from .outputs import ReturnShape, answer_by_output, outputs_schema, without_skipped
from .type import Port
from .validator import BoundDefaultValidator, DeclaredInputValidator, DeclaredOutputValidator

__all__ = [
    "BoundDefaultValidator",
    "DeclaredInputValidator",
    "DeclaredOutputValidator",
    "Port",
    "PortDescription",
    "ReturnShape",
    "answer_by_output",
    "outputs_schema",
    "without_skipped",
]
