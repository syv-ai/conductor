"""The three kinds of problem the worker reports itself, in ``SERVING_PROBLEMS``, and the HTTP status of each.

Everything else a run or a call can go wrong with is conductor's own, in ``conductor.PROBLEMS``. The exceptions are
a request body of the wrong shape and a ``Last-Event-ID`` that isn't a whole number, which FastAPI refuses with its
own errors. A kind is a function that writes the message, declared the way conductor declares its own. Calling it
makes the ``Problem``::

    run_not_resumable(status="waiting")
    # Problem(type="run_not_resumable", loc=(), msg="The run is waiting, ...", ctx={"status": "waiting"})

Each one is about a run as a whole, so its ``loc`` is ``()``, but for ``no_such_run`` about the ``previous`` a new
run names, whose ``loc`` is ``("previous",)``. ``STATUS`` sits beside the codes, so a new code gets
its status in the same place.
"""

from __future__ import annotations

from conductor import ProblemCatalogue, RunStatus

#: The kinds of problem the worker reports itself.
SERVING_PROBLEMS = ProblemCatalogue()


@SERVING_PROBLEMS.problem
def no_such_run(*loc: str | int) -> str:
    """No run has the id, or the run filter doesn't let this request see the run.

    Args:
        loc: ``("previous",)`` when the id is the ``previous`` a new run was asked to reuse, so a client can tell it
            from a route's own run; else empty, for the run in the route's path.
    """
    return "There is no run by that id that you may see; check the id."


#: The statuses of a run that a resume goes on from: the ways a resume call ends a run early. ``Worker.resume``
#: refuses ``waiting``, ``complete`` and ``declined`` as ``run_not_resumable``, whose message lists these. A
#: ``running`` run is taken or left as it is, but it is refused a new item limit, since it hasn't stopped.
RESUMABLE: tuple[RunStatus, ...] = ("failed", "stopped", "cancelled", "timed_out")


@SERVING_PROBLEMS.problem
def run_not_resumable(*, status: RunStatus) -> str:
    """A resume was asked of a run that didn't end early or that a person's decline ended for good, or a new item
    limit was given to a run that hasn't stopped.

    Args:
        status: The run's status.
    """
    if status == "running":
        return (
            "The run is running, and a new item limit is taken only by a run that has stopped; give it again once "
            "the run stops at its limit."
        )
    resumable = f"{', '.join(RESUMABLE[:-1])} or {RESUMABLE[-1]}"
    return (
        f"The run is {status}, and only a run that is {resumable} can be resumed; read the run to see what it "
        "needs."
    )


@SERVING_PROBLEMS.problem
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
