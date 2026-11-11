"""The problems ``check_registry`` returns, one function per code.

Code outside this package imports from here only, and calls a kind as ``kit_problem.run_failed(...)``.
"""

from conductor._internal.kit_problem.catalogue import (
    REGISTRY_CHECK_PROBLEMS,
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

__all__ = [
    "REGISTRY_CHECK_PROBLEMS",
    "changed_its_inputs",
    "example_for_no_node",
    "example_not_loaded",
    "example_refused",
    "hook_failed",
    "hook_failed_on_an_example",
    "needs_example",
    "run_failed",
    "upgrade_broken",
    "upgrade_raised",
]
