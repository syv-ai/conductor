"""The four kinds of problem the worker reports itself, in ``SERVING_PROBLEMS``, and the HTTP status of each.

Everything else a run or a call can go wrong with is conductor's own, in ``conductor.PROBLEMS``. The exceptions are
a request body of the wrong shape and a ``Last-Event-ID`` that isn't a whole number, which FastAPI refuses with its
own errors. A kind is a function that writes the message, declared the way conductor declares its own. Calling it
makes the ``Problem``::

    run_not_resumable(status="waiting")
    # Problem(type="run_not_resumable", loc=(), msg="The run is waiting, ...", ctx={"status": "waiting"})

Each one is about a run as a whole, so its ``loc`` is ``()``. ``STATUS`` sits beside the codes, so a new code gets
its status in the same place.
"""

from __future__ import annotations

from conductor import ProblemCatalogue, RunStatus

#: The kinds of problem the worker reports itself.
SERVING_PROBLEMS = ProblemCatalogue()


@SERVING_PROBLEMS.problem()
def no_such_run() -> str:
    """No run has the id, or the run filter doesn't let this request see the run."""
    return "There is no run by that id that you may see; check the id."


@SERVING_PROBLEMS.problem()
def run_not_resumable(*, status: RunStatus) -> str:
    """A resume was asked of a run that didn't end early.

    Args:
        status: The run's status.
    """
    return f"The run is {status}, and only a run that ended early can be resumed; read the run to see what it needs."


@SERVING_PROBLEMS.problem()
def previous_run_unloadable(*, run_id: str, reason: str) -> str:
    """The run named in ``previous`` can't be loaded on the new run's nodes, so its work can't be reused.

    Args:
        run_id: The previous run's id.
        reason: Why it can't be loaded, in conductor's words.
    """
    return (
        f"The previous run '{run_id}' can't be loaded on these nodes, so its work can't be reused: "
        f"{reason.rstrip('.')}. Start without it."
    )


@SERVING_PROBLEMS.problem()
def raised_outside_a_node(*, error: str) -> str:
    """Something raised outside any node while the worker resumed the run, or checked an answer or a decline queued
    while it ran.

    It is the cause of a run that failed, or a queued answer's or decline's entry in the run's ``refused``. The server
    log has the traceback.

    Args:
        error: The exception's class name.
    """
    return (
        f"The worker hit {error} outside any node while it resumed the run or checked an answer or a decline; the "
        "server log says why. Resume the run, or send the answer or decline again, once that is fixed."
    )


#: The HTTP status a route answers with when the worker refuses a call, by the code of the first problem. Every other
#: code, conductor's own among them, is a 422.
STATUS: dict[str, int] = {"no_such_run": 404, "run_not_resumable": 409}
