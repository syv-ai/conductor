"""Validation: every place a value from outside enters the engine, and the validator that checks it there.

A value is checked once, where it comes in. Inside, nothing checks it again. A validator takes the value from outside
and returns it as the engine keeps it::

    outputs = compiled_node.return_validator(returned)   # {"result": "B!"} for shout returning "B!"

or raises ``InvalidCallError``, whose problems carry the validator's code (a default a node declares raises
``NodeDeclarationError``: it is a fault in the node's code, met at import). A validator is a class when it is built
once for what it reads (a node's outputs, a graph's inputs), and a function when it reads one value by a type. Its
caller decides what becomes of the problems: a public method lets them reach its caller; compiling keeps them in
``graph.problems``, and a run keeps the problem as the outcome of the call it belongs to.
Every validator reads values with the pydantic validators of ``rules.py``, built there from a type's schema with the
rules every type is held to (a yes or no is not a number); no module outside this package builds one.

Where a value enters, with the module that validates it (the problems page has the codes each refuses with):

    where a value enters                                        module
    a graph record, from JSON or from what make_graph and       record
        registry.upgrade write, and the values make_graph
        writes into one
    a default a node declares, and the same default read        declared
        again by the type an edge binds its type variable to
    a constant or default written in a record                   constants
    what a compute_added_inputs or compute_outputs hook         node/validator.py
        returns
    a caller's graph inputs, max_items, timeout and             inputs
        checkpoint_every
    a person's answer to a node's outputs                       answer
    what a node's function returns                              node/validator.py
    a value an edge gives an input that demands more than       edge
        the edge's source declares
    what a node raises: an Ask's choices and default            raised
    what a node raises: a NodeError's ctx                       problem/errors.py, which stays there: NodeError
                                                                reads it when the node makes one
    the context= a host passes to start or resume a run         context
    a saved run                                                 run/json/, which stays there: its reader and its
                                                                writer are one schema, with the rules of rules.py

A type's own declarations are read where the type is declared: ``Param`` reads its choices by the type it limits
(``types/param.py``), and ``@host_type`` its arguments.
"""

from .answer import AnswerValidator
from .constants import read_constant
from .context import ContextValidator
from .declared import bound_field, declared_input, declared_output
from .edge import EdgeValidator, demands_more
from .exclusive_group import exclusive_groups, groups_without_one_value
from .inputs import InputsValidator, read_limits
from .raised import AskValidator, undeclared_ask
from .record import read_json, read_record, record_values
from .rules import (
    exact_classes,
    numbers_refuse_bools,
    output_schema_of,
    schema_output_validator_of,
    schema_validator_of,
)

__all__ = [
    "AnswerValidator",
    "AskValidator",
    "ContextValidator",
    "EdgeValidator",
    "InputsValidator",
    "bound_field",
    "declared_input",
    "declared_output",
    "demands_more",
    "exact_classes",
    "exclusive_groups",
    "groups_without_one_value",
    "numbers_refuse_bools",
    "output_schema_of",
    "read_constant",
    "read_json",
    "read_limits",
    "read_record",
    "record_values",
    "schema_output_validator_of",
    "schema_validator_of",
    "undeclared_ask",
]
