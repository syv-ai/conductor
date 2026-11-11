"""The problems the runs router writes itself, beside those of the runs it serves, one function each.

``ROUTER_PROBLEMS.problem`` says how a function's signature is read. Each is about a run as a whole, so its ``loc``
is ``()``. Five are the ``detail`` of a 404 or a 409; ``leg_failed`` is the ``cause`` of a run.
"""

from __future__ import annotations

from conductor import ProblemCatalogue, RunStatus

#: The kinds of problem the runs router writes itself.
ROUTER_PROBLEMS = ProblemCatalogue()


@ROUTER_PROBLEMS.problem()
def no_such_run() -> str:
    """No run has the id, or the caller may not see the one that has; the two read the same."""
    return "There is no run by that id that the caller may see; check the id and who is calling."


@ROUTER_PROBLEMS.problem()
def already_answered(*, waiting_id: str) -> str:
    """An answer sent while a leg runs is for a waiting id that an earlier call queued an answer for.

    Args:
        waiting_id: The waiting id answered twice.
    """
    return f"An answer to '{waiting_id}' already waits for the running leg to end; read the run once it has, and answer what still waits."


@ROUTER_PROBLEMS.problem()
def run_complete() -> str:
    """An answer is sent to a run that is complete."""
    return "The run is complete, so nothing in it waits for an answer; start a new run to ask again."


@ROUTER_PROBLEMS.problem()
def run_not_resumable(*, status: RunStatus) -> str:
    """A resume is asked of a run whose last leg did not end early.

    Args:
        status: The run's status.
    """
    return f"The run is {status}, and only a run whose last leg ended early is resumed; read the run to see what it needs."


@ROUTER_PROBLEMS.problem()
def run_changed() -> str:
    """Another call claimed the run between this call's read of it and its write."""
    return "The run changed since it was read, so nothing was done; read it again and repeat the call if it still applies."


@ROUTER_PROBLEMS.problem()
def leg_failed() -> str:
    """The run's worker raised outside any node, so the leg ended with no run."""
    return "The run's worker failed before the leg could end; its operator finds why in the server log. Resume it to carry on."
