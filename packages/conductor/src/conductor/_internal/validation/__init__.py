"""Checking the values that come into conductor from outside.

Values reach the engine from many places: a caller's graph inputs, a constant in a graph record, what a node
returns, a person's answer. Each is checked once, where it comes in. After that, nothing inside the engine checks
it again.

Every check is a ``Validator`` (``type.py``), and each package whose values come from outside has its own. When
a validator refuses a value, whoever called it decides what happens to the problems. A public method lets them
reach its own caller, compiling keeps them in ``graph.problems``, and a run keeps the problem as the outcome of the
call.

Validators check values with the checks a ``Type`` owns: ``argument_validator`` for an argument and
``output_validator`` for an output, each built once from the type's schema with the rules that hold for every type,
such as "a yes or no is not a number" (``types/rules.py``). Outside a ``validator.py``, only compiling asks one,
of a ``Literal``'s values (``Type.takes_value``), and ``conductor.testing`` does, to make up values for a node's
inputs.

A few values are checked elsewhere. A saved run is checked in ``run/json/``, where one schema both reads and writes
it. A ``NodeError``'s ``ctx`` is checked by ``NodeError`` itself, where a node makes one. And a type's own
declarations are checked where the type is declared: ``Param`` checks its choices against the type it limits
(``types/param.py``), and ``@host_type`` checks its arguments.

What validators share besides ``Validator`` is in ``json.py``: the one strict reader of a value that enters a
graph, a caller's input, a person's answer or a record's constant or default (``read_as_json``).
"""

from .json import json_data, read_as_json, set_has_no_order
from .type import Validator

__all__ = [
    "Validator",
    "json_data",
    "read_as_json",
    "set_has_no_order",
]
