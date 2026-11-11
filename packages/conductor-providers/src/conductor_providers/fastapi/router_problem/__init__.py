"""The problems the runs router writes itself, one function per code, and the exceptions that carry them.

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
from conductor_providers.fastapi.router_problem.errors import NoSuchRunError, RefusedError

__all__ = [
    "ROUTER_PROBLEMS",
    "NoSuchRunError",
    "RefusedError",
    "already_answered",
    "leg_failed",
    "no_such_run",
    "run_changed",
    "run_complete",
    "run_not_resumable",
]
