"""The gate: every place a value from outside enters the engine, called a door.

A value is checked once, at the door it comes in by. Inside, nothing checks it again. Each door is built once for
what it reads (a node's outputs, a graph's inputs) and then called with the value from outside::

    outputs = compiled_node.return_door(returned)   # {"result": "B!"} for shout returning "B!"

It returns the value as the engine keeps it, or raises ``InvalidCallError``, whose problems carry the door's code. A
public method lets it reach its caller; compiling keeps its problems in ``graph.problems``, and a run keeps the
problem as the outcome of the call it belongs to.
Every door reads values with the validators of ``rules.py``, built there from a type's schema with the rules every
type is held to (a yes or no is not a number); no module outside the gate builds a validator of a type.

The doors, with the module each is in:

    door                                                    module, then its problem codes
    a graph record, from JSON or from what make_graph and   record
        registry.upgrade write, and the values make_graph   invalid_record, invalid_value
        writes into one
    a default a node declares, and the same default read    declared
        again by the type an edge binds its type variable   NodeDeclarationError default_invalid,
        to                                                  invalid_value (bound_default)
    a constant or default written in a record               constants
                                                            invalid_value
    what a compute_added_inputs or compute_outputs hook     hook
        returns                                             compute_hook_failed
    a caller's graph inputs, max_items, timeout and         inputs
        checkpoint_every                                    not_offered, missing_input,
                                                            invalid_input, invalid_input_make_a_series,
                                                            invalid_limit
    a person's answer to a node's outputs                   answer
                                                            invalid_answer
    what a node's function returns                          returned
                                                            invalid_output
    a value an edge gives an input that demands more        edge
        than the edge's source declares                     invalid_edge_value
    what a node raises: an Ask's choices and default        raised
                                                            invalid_ask
    what a node raises: a NodeError's ctx                   problem/errors.py, which stays there: NodeError
                                                            reads it when the node makes one
                                                            (pydantic's ValidationError at the node's raise)
    the context= a host passes to start or resume a run     context
                                                            invalid_context, context_missing
    a saved run                                             run/json/, which stays there: its reader and its
                                                            writer are one schema, with the rules of rules.py
                                                            saved_run_unreadable, saved_run_of_another_graph

A type's own declarations are read where the type is declared: ``Param`` reads its choices by the type it limits
(``types/param.py``), and ``@host_type`` its arguments.
"""

from .answer import AnswerDoor
from .constants import read_constant
from .context import ContextDoor
from .declared import bound_field, declared_field
from .edge import EdgeDoor, demands_more
from .hook import read_hook_fields
from .inputs import InputsDoor, read_limits
from .raised import AskDoor, undeclared_ask
from .record import read_json, read_record, record_values
from .returned import ReturnDoor
from .rules import numbers_refuse_bools, output_validator_of, validator_of

__all__ = [
    "AnswerDoor", "AskDoor", "ContextDoor", "EdgeDoor", "InputsDoor", "ReturnDoor", "bound_field", "declared_field",
    "demands_more", "numbers_refuse_bools", "output_validator_of", "read_constant", "read_hook_fields", "read_json", "read_limits", "read_record", "record_values",
    "undeclared_ask", "validator_of",
]
