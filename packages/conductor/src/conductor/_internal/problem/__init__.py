"""A problem, the kinds of it conductor writes, and the exceptions that carry one.

Code outside this package imports from here only. ``type.py`` has ``Problem`` and how a package declares its kinds,
``catalogue.py`` the core's kinds, one function per code (other modules import it as
``from conductor._internal.problem import catalogue as problem`` and call a kind as ``problem.cycle(...)``), and
``errors.py`` the exceptions.
"""

from conductor._internal.problem import catalogue
from conductor._internal.problem.catalogue import PROBLEMS
from conductor._internal.problem.errors import (
    CallRejectedError,
    DeclarationCode,
    ExternalError,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    value_free_errors,
)
from conductor._internal.problem.type import Problem, ProblemCatalogue, ProblemKind

__all__ = [
    "DeclarationCode",
    "ExternalError",
    "InvalidCallError",
    "NodeDeclarationError",
    "NodeError",
    "PROBLEMS",
    "Problem",
    "ProblemCatalogue",
    "ProblemKind",
    "CallRejectedError",
    "catalogue",
    "value_free_errors",
]
