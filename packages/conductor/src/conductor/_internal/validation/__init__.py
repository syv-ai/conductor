"""Checking the values that come into conductor from outside.

Values reach the engine from many places: a caller's graph inputs, a constant in a graph record, what a node
returns, a person's answer. Each is checked once, where it comes in. After that, nothing inside the engine checks
it again.

Every check is a ``Validator`` (``type.py``), and each package whose values come from outside has its own. When
a validator refuses a value, whoever called it decides what happens to the problems. A public method lets them
reach its own caller, compiling keeps them in ``graph.problems``, and a run keeps the problem as the outcome of the
call.

Validators check values with the pydantic validators ``rules.py`` gives, each built from a type's own check, which
adds the rules that hold for every type, such as "a yes or no is not a number". Outside a ``validator.py``, nothing in the engine builds one.
``conductor.testing`` does, to make up values for a node's inputs.

A few values are checked elsewhere. A saved run is checked in ``run/json/``, where one schema both reads and writes
it. A ``NodeError``'s ``ctx`` is checked by ``NodeError`` itself, where a node makes one. And a type's own
declarations are checked where the type is declared: ``Param`` checks its choices against the type it limits
(``types/param.py``), and ``@host_type`` checks its arguments.
"""

from .rules import (
    exact_classes,
    output_schema_of,
    schema_of,
    schema_output_validator_of,
    schema_validator_of,
)
from .type import Validator

__all__ = [
    "Validator",
    "exact_classes",
    "output_schema_of",
    "schema_of",
    "schema_output_validator_of",
    "schema_validator_of",
]
