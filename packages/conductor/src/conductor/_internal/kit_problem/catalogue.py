"""The problems ``check_registry`` returns, one function each.

``@KIT_PROBLEMS.problem()`` makes the function under it a kind of problem: its name is the code, its keyword-only
parameters are the keys of ``ctx``, and what it returns is the English. Calling it builds the ``Problem``,
``kit_problem.needs_example("upper", 1, inputs=["text"])``, so a call that leaves a key out or gives one the wrong
type is a type error where it is written. ``ProblemCatalogue.problem`` has the rules in full.

Each is at ``(node_id, version)``, the version of a node it is about. A developer reads them in a test run; a host
never meets them.
"""

from __future__ import annotations

from typing import Any

from conductor._internal.problem import ProblemCatalogue

#: The kinds of problem ``check_registry`` returns.
KIT_PROBLEMS = ProblemCatalogue()


@KIT_PROBLEMS.problem()
def example_for_no_node(*loc: str | int, named: str) -> str:
    """An example is given for a node, or a version of one, that the registry doesn't have.

    Args:
        loc: The node id the example names, and the version when it names one.
        named: What the example is keyed by, as it was written.
    """
    return f"{named} names no node of the registry; check the id, or remove the example"


@KIT_PROBLEMS.problem()
def needs_example(*loc: str | int, inputs: list[str]) -> str:
    """A hook reads a required input no value can be made up for, so the hooks can't be tried without an example.

    Args:
        loc: The node and its version.
        inputs: The inputs no value could be made up for.
    """
    return f"its hooks read {inputs}, and no value could be made up; give an example"


@KIT_PROBLEMS.problem()
def hook_failed(*loc: str | int, reason: str, cause: list[str], problems: list[dict[str, Any]]) -> str:
    """A ``compute_...`` hook failed on the node's defaults and a made-up value for each required input it reads.

    Args:
        loc: The node and its version.
        reason: The sentence of the problem loading found.
        cause: The code of that problem.
        problems: That problem, as JSON.
    """
    return f"a hook failed on the node's defaults and made-up values: {reason}"


@KIT_PROBLEMS.problem("hook_failed")
def hook_failed_on_an_example(*loc: str | int, reason: str, cause: list[str], problems: list[dict[str, Any]]) -> str:
    """An example doesn't load, and every fault loading found is a hook's.

    Args:
        loc: The node and its version.
        reason: The sentence of each problem loading found.
        cause: The code of each.
        problems: Those problems, as JSON.
    """
    return f"the example doesn't load: {reason}"


@KIT_PROBLEMS.problem("upgrade_broken")
def upgrade_raised(*loc: str | int, newest: int, error: str, reason: str) -> str:
    """Moving a node placed at an older version to the newest raised.

    Args:
        loc: The node and the older version.
        newest: The version it was moved to.
        error: The class name of what was raised.
        reason: Its text.
    """
    return f"moving to version {newest} raised {error}: {reason}"


@KIT_PROBLEMS.problem()
def upgrade_broken(*loc: str | int, newest: int, reason: str, cause: list[str], problems: list[dict[str, Any]]) -> str:
    """A node placed at an older version and moved to the newest doesn't load cleanly.

    Args:
        loc: The node and the older version.
        newest: The version it was moved to.
        reason: The code and sentence of each problem loading found.
        cause: The code of each.
        problems: Those problems, as JSON.
    """
    return f"after moving to version {newest}: {reason}"


@KIT_PROBLEMS.problem()
def changed_its_inputs(*loc: str | int, input: str) -> str:  # noqa: A002  the ctx key a developer reads
    """A node's body changed a value it was given.

    Args:
        loc: The node and its version.
        input: The input whose value was changed.
    """
    return f"the body changed {input!r}, a value it was given; several calls of a run can be given it, so copy the value before changing it"


@KIT_PROBLEMS.problem("run_failed")
def example_not_loaded(*loc: str | int, reason: str, cause: list[str], problems: list[dict[str, Any]]) -> str:
    """An example doesn't load, for a fault that is not a hook's.

    Args:
        loc: The node and its version.
        reason: The sentence of each problem loading found.
        cause: The code of each.
        problems: Those problems, as JSON.
    """
    return f"the example doesn't load: {reason}"


@KIT_PROBLEMS.problem("run_failed")
def example_refused(*loc: str | int, reason: str, cause: list[str], problems: list[dict[str, Any]]) -> str:
    """An example loads, and running it is refused.

    Args:
        loc: The node and its version.
        reason: The sentence of each problem the call was refused for.
        cause: The code of each.
        problems: Those problems, as JSON.
    """
    return f"the example was refused: {reason}"


@KIT_PROBLEMS.problem()
def run_failed(*loc: str | int, status: str, reason: str, cause: list[str], problems: list[dict[str, Any]]) -> str:
    """An example ran and ended early: it failed, stopped, was cancelled or timed out.

    Args:
        loc: The node and its version.
        status: How the run ended.
        reason: The sentence of the problem it ended on; for a body's own exception, which class it raised.
        cause: The code of that problem.
        problems: That problem, as JSON.
    """
    return f"the example ended {status}: {reason}"
