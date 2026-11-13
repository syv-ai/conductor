"""A problem, the kinds of problem conductor's core reports, and the exceptions.

Code outside this package imports from here. ``type.py`` is ``Problem`` and how a package declares the kinds of
problem it reports (``ProblemCatalogue``). ``catalogue.py`` is the core's kinds, one function each, declared in one
module per section (``compiling.py``, ``refused.py``, ``stopped.py``); a module that reports one imports it as
``from conductor._internal.problem import catalogue as problem`` and calls ``problem.cycle(node_id)``. ``declarations.py`` is the kinds of a wrong declaration, which a
``NodeDeclarationError`` carries; a module that raises one imports it as ``declarations as declaration``.
``errors.py`` is the exceptions.
"""

from . import catalogue, declarations
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
    "catalogue",
    "declarations",
    "value_free_errors",
]
