"""A problem, the kinds of it conductor writes, and the exceptions that carry one.

Code outside this package imports from here only. ``type.py`` has ``Problem`` and how a package declares its kinds,
``catalogue.py`` the core's kinds and ``kit_catalogue.py`` those ``check_registry`` returns, one function per code;
``unconverted.py`` the codes still written from a table of sentences, and ``errors.py`` the exceptions.
"""

from conductor._internal.problem.catalogue import (
    PROBLEMS,
    cannot_move_to_version,
    from_run_missing,
    invalid_record,
    not_a_chain,
)
from conductor._internal.problem.errors import (
    DeclarationCode,
    ExternalError,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    value_free_errors,
)
from conductor._internal.problem.kit_catalogue import (
    KIT_PROBLEMS,
    changed_its_inputs,
    example_for_no_node,
    example_not_loaded,
    example_refused,
    hook_failed,
    hook_failed_on_an_example,
    needs_example,
    run_failed,
    upgrade_broken,
    upgrade_raised,
)
from conductor._internal.problem.type import Problem, ProblemCatalogue, ProblemKind
from conductor._internal.problem.unconverted import (
    PROBLEM_CATALOGUE,
    PROBLEM_CODES,
    CtxValue,
    make_problem,
)

__all__ = [
    "KIT_PROBLEMS",
    "PROBLEMS",
    "PROBLEM_CATALOGUE",
    "PROBLEM_CODES",
    "CtxValue",
    "DeclarationCode",
    "ExternalError",
    "InvalidCallError",
    "NodeDeclarationError",
    "NodeError",
    "Problem",
    "ProblemCatalogue",
    "ProblemKind",
    "cannot_move_to_version",
    "changed_its_inputs",
    "example_for_no_node",
    "example_not_loaded",
    "example_refused",
    "from_run_missing",
    "hook_failed",
    "hook_failed_on_an_example",
    "invalid_record",
    "make_problem",
    "needs_example",
    "not_a_chain",
    "run_failed",
    "upgrade_broken",
    "upgrade_raised",
    "value_free_errors",
]
