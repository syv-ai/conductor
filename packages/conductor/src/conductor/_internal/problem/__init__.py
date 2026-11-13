"""A problem, the kinds of problem conductor's core reports, and the exceptions.

Code outside this package imports from here. ``type.py`` is ``Problem`` and how a package declares the kinds of
problem it reports (``ProblemCatalogue``). The core's kinds are one function each, in one module per section of the
problems page: ``compile_problems.py``, ``call_problems.py`` and ``run_problems.py``; a module that reports one
imports the module that declares it and calls ``compile_problems.cycle(node_id)``. ``catalogue.py`` joins them as
``PROBLEMS``. ``declarations.py`` is the kinds of a wrong declaration, which a ``NodeDeclarationError`` carries; a
module that raises one imports it as ``declarations as declaration``. ``errors.py`` is the exceptions.
"""

from . import call_problems, compile_problems, declarations, run_problems
from .catalogue import PROBLEMS
from .declarations import DECLARATION_PROBLEMS
from .errors import (
    ExternalError,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    value_free_errors,
)
from .type import Problem, ProblemCatalogue, ProblemKind

__all__ = [
    "DECLARATION_PROBLEMS",
    "ExternalError",
    "InvalidCallError",
    "NodeDeclarationError",
    "NodeError",
    "PROBLEMS",
    "Problem",
    "ProblemCatalogue",
    "ProblemKind",
    "call_problems",
    "compile_problems",
    "declarations",
    "run_problems",
    "value_free_errors",
]
