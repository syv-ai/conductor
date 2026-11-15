"""Validation: the rules every validator reads values by.

A value from outside is checked once, where it comes in, and nothing inside checks it again. Each package whose
values come from outside reads them in its ``validator.py``: a graph record in ``record/``, a default a node declares
in ``field/``, what a node's function or hook returns in ``node/``, a caller's graph inputs, the run's limits and a
record's constants in ``graph/``, the ``context=`` in ``context/``, an ``Ask`` and a person's answer in ``ask/``, a
value an edge gives in ``run/``. Each validator there is a ``Validator`` (``type.py``), built with what it reads a
value by and called with the value alone. A saved run is read in ``run/json/``, whose reader and writer are one schema, and a
``NodeError``'s ``ctx`` by ``NodeError``, where a node makes one. A validator returns the value as the engine keeps
it, or raises ``InvalidCallError``, whose problems carry the validator's code (a default a node declares raises
``NodeDeclarationError``: it is a fault in the node's code, met at import). Its caller decides what becomes of the
problems: a public method lets them reach its caller; compiling keeps them in ``graph.problems``, and a run keeps
the problem as the outcome of the call it belongs to.

Every validator reads values with the pydantic validators of ``rules.py``, built there from a type's schema with the
rules every type is held to (a yes or no is not a number); no module of the engine outside a ``validator.py``
builds one (``conductor.testing`` does, to make up values a node's inputs take).

A type's own declarations are read where the type is declared: ``Param`` reads its choices by the type it limits
(``types/param.py``), and ``@host_type`` its arguments.
"""

from .rules import (
    exact_classes,
    numbers_refuse_bools,
    output_schema_of,
    schema_of,
    schema_output_validator_of,
    schema_validator_of,
)
from .type import Validator

__all__ = [
    "Validator",
    "exact_classes",
    "numbers_refuse_bools",
    "output_schema_of",
    "schema_of",
    "schema_output_validator_of",
    "schema_validator_of",
]
