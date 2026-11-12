"""A problem, the kinds of problem conductor's core reports, and the exceptions.

Code outside this package imports from here. ``type.py`` is ``Problem`` and how a package declares the kinds of
problem it reports (``ProblemCatalogue``). ``catalogue.py`` is the core's kinds, one function each; a module that
reports one imports it as ``from conductor._internal.problem import catalogue as problem`` and calls
``problem.cycle(node_id)``. ``declarations.py`` is the kinds of a wrong declaration, which a
``NodeDeclarationError`` carries; a module that raises one imports it as ``declarations as declaration``.
``errors.py`` is the exceptions.
"""

from conductor._internal.problem import catalogue, declarations
from conductor._internal.problem.catalogue import PROBLEMS
from conductor._internal.problem.declarations import DECLARATION_PROBLEMS
from conductor._internal.problem.errors import (
    ExternalError,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    value_free_errors,
)
from conductor._internal.problem.type import Problem, ProblemCatalogue, ProblemKind

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
