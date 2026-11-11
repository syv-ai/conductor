"""The problems the runs router writes itself, one function per code.

Code outside this package imports from here only, and calls a kind as ``router_problem.run_changed()``.
"""

from conductor_providers.fastapi.router_problem.catalogue import (
    ROUTER_PROBLEMS,
    already_answered,
    leg_failed,
    no_such_run,
    run_changed,
    run_complete,
    run_not_resumable,
)

__all__ = [
    "ROUTER_PROBLEMS",
    "already_answered",
    "leg_failed",
    "no_such_run",
    "run_changed",
    "run_complete",
    "run_not_resumable",
]
