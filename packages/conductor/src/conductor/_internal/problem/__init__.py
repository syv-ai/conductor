"""Problems: how conductor tells a person that something is wrong.

Whatever goes wrong, conductor reports it the same way, as a ``Problem``. A problem has a code such as
``unknown_input``, a place, a message for a person, and the values the message was made from. So a host handles one
kind of record, whether the fault is in a graph, in a call or in a run.

Each kind of problem is a small function that writes its message, and calling it makes the problem. The kinds live
in one module for each section of the problems page:

- ``compile_problems``: faults in a graph, found when it is compiled.
- ``call_problems``: a call that conductor refuses.
- ``run_problems``: why a run stopped.
- ``declaration_problems``: faults in how a node or a type is declared.

Code that reports a problem imports the module and calls the kind, like ``compile_problems.cycle(node_id)``.

The exceptions conductor raises live here too, in ``errors.py``, which says who meets which one and when. Every code,
with an example of each, is on the problems page: ``python -m conductor.about problems``.
"""

from . import call_problems, compile_problems, declaration_problems, run_problems
from .catalogue import PROBLEMS
from .declaration_problems import DECLARATION_PROBLEMS
from .errors import (
    OWN_MESSAGE,
    ExternalError,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    argument_errors,
    author_code_problem,
    conductor_error,
    is_conductor_error,
    no_json_form,
    set_has_no_order,
    shape_error,
    value_free_errors,
)
from .type import Problem, ProblemCatalogue, ProblemKind

__all__ = [
    "DECLARATION_PROBLEMS",
    "ExternalError",
    "InvalidCallError",
    "NodeDeclarationError",
    "NodeError",
    "OWN_MESSAGE",
    "PROBLEMS",
    "Problem",
    "ProblemCatalogue",
    "ProblemKind",
    "call_problems",
    "compile_problems",
    "declaration_problems",
    "run_problems",
    "author_code_problem",
    "conductor_error",
    "is_conductor_error",
    "no_json_form",
    "set_has_no_order",
    "argument_errors",
    "shape_error",
    "value_free_errors",
]
