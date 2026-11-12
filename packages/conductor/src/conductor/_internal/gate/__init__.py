"""The gate: every place a value from outside enters the engine, called a door, and the one exception a door raises.

A value is checked once, at the door it comes in by. Inside, nothing checks it again. Each door is built once for
what it reads (a node's outputs, a graph's inputs) and then called with the value from outside::

    outputs = compiled_node.return_door(returned)   # {"result": "B!"} for shout returning "B!"

It returns the value as the engine keeps it, or raises ``RefusedError``, whose problems carry the door's code.

The doors, with the module each is in. The ones marked "not yet" still check by hand where the value arrives,
in the file named; they move here one by one.

    door                                                    module, then its problem codes
    a graph record, from JSON, a dict or make_graph         record (not yet: graph_record.py, make_graph.py)
                                                            invalid_record, invalid_value, invalid_node_id
    a constant or default written in a record               constants (not yet: compiled/compiler.py)
                                                            invalid_value
    what a compute_added_inputs or compute_outputs hook     hook (not yet: compiled/compiler.py)
        returns                                             parameter_name_invalid, compute_hook_failed
    a caller's graph inputs, max_items, timeout and         inputs
        checkpoint_every                                    not_offered, input_given_twice, missing_input,
                                                            invalid_input, invalid_input_make_a_series,
                                                            invalid_max_items, invalid_timeout,
                                                            invalid_checkpoint_every
    a person's answer                                       answer
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
    a saved run                                             run/json/read.py, which stays there: its reader
                                                            and its writer are one schema
                                                            saved_run_unreadable, saved_run_of_another_graph

A node author's declarations (a default, ``Param``'s choices, a host type's check) are read when the node is
declared, by the type they are declared on.
"""

from conductor._internal.gate.answer import AnswerDoor
from conductor._internal.gate.context import ContextDoor
from conductor._internal.gate.edge import EdgeDoor, demands_more
from conductor._internal.gate.inputs import InputsDoor, check_limits
from conductor._internal.gate.raised import AskDoor
from conductor._internal.gate.refused import RefusedError
from conductor._internal.gate.returned import ReturnDoor

__all__ = ["AnswerDoor", "AskDoor", "ContextDoor", "EdgeDoor", "InputsDoor", "RefusedError", "ReturnDoor", "check_limits", "demands_more"]
