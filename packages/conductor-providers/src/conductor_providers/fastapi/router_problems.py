"""The kinds of problem the runs router and its ``Worker`` report themselves, one function each, in ``ROUTER_PROBLEMS``,
and the HTTP status of each that a route answers with.

A kind is a function that returns the message; the decorator makes it a kind, which is called to make its ``Problem``,
as the kinds of ``conductor.PROBLEMS`` are. ``ProblemCatalogue.problem`` says how the function's signature is read::

    already_answered(waiting_id="approve")
    # Problem(type="already_answered", loc=(), msg="An answer to 'approve' is already queued ...", ctx={"waiting_id": "approve"})

Each is about a run as a whole, so its ``loc`` is ``()``. Six are raised by the ``Worker`` in an ``InvalidCallError``,
as conductor's own refusals are (``STATUS`` gives the status of each), ``answer_check_failed`` is in a run's
``refused``, and ``resume_failed`` is a run's ``cause``. The problems of the runs
themselves are conductor's, in ``conductor.PROBLEMS``. Each kind's docstring is its entry in the docs' page on serving
runs.
"""

from __future__ import annotations

from conductor import ProblemCatalogue, RunStatus

#: The kinds of problem the runs router writes itself.
ROUTER_PROBLEMS = ProblemCatalogue()


@ROUTER_PROBLEMS.problem()
def no_such_run() -> str:
    """No run has the id, or the caller may not see the run that has it.

    Raised by ``Worker.read`` (a 404 from every route under ``/runs/{run_id}``). The router
    raises it too for a run ``may_see`` turns the caller away from, so the two cases read the same and a caller learns
    nothing about a run that is not theirs. Use the id ``Worker.begin`` returned, as the caller who started the run
    or one ``may_see`` lets in.
    """
    return "There is no run by that id that the caller may see; check the id and who is calling."


@ROUTER_PROBLEMS.problem()
def already_answered(*, waiting_id: str) -> str:
    """An answer sent while the run is running is for a waiting id that an earlier call already sent an answer for.

    Raised by ``Worker.answer`` (a 409 from ``POST /runs/{run_id}/answers``); nothing of the call
    was queued. There is one problem for each waiting
    id sent twice. Wait until the run stops running: the queued answer is given to the run then, or is in its
    ``refused`` with the reason. Then answer what still waits.

    Args:
        waiting_id: The waiting id answered twice.
    """
    return (
        f"An answer to '{waiting_id}' is already queued for when the run stops running; "
        "read the run once it has stopped, and answer what still waits."
    )


@ROUTER_PROBLEMS.problem()
def answer_check_failed(*, waiting_id: str, error: str) -> str:
    """An answer queued while the run was running could not be checked when the run stopped: the check of the
    answer's type raised an exception that is not a refusal of the value.

    In a run's ``refused``, under the waiting id. The fault is in the host's type: a ``KeyError`` in its validator,
    say, where a ``ValueError`` would refuse the value. The run's status is written all the same, and the question
    still waits. The traceback is in the log ``conductor_providers.fastapi``; answer again once the type is fixed. The
    same fault on an answer sent while the run is not running raises from ``Worker.answer`` (a 500 of that request).

    Args:
        waiting_id: The waiting id the answer was for.
        error: The class name of the exception the check raised.
    """
    return (
        f"The answer queued for '{waiting_id}' could not be checked: its type's own check raised {error}; "
        "the operator finds why in the server log. Send it again once that is fixed."
    )


@ROUTER_PROBLEMS.problem()
def run_complete() -> str:
    """An answer is sent to a run that is complete.

    Raised by ``Worker.answer`` (a 409 from ``POST /runs/{run_id}/answers``). Nothing waits;
    ``Worker.again`` (``POST /runs/{run_id}/again``) starts a new run with some inputs changed.
    """
    return (
        "The run is complete, so nothing in it waits for an answer; start a new run to ask again."
    )


@ROUTER_PROBLEMS.problem()
def run_not_resumable(*, status: RunStatus) -> str:
    """A resume is asked of a run that did not end early.

    Raised by ``Worker.resume`` (a 409 from ``POST /runs/{run_id}/resume``) on a run that is
    ``running``, ``waiting`` or ``complete``. A waiting run
    goes on when it is answered, a running one is already going on, and a complete one is done.

    Args:
        status: The run's status.
    """
    return f"The run is {status}, and only a run that ended early can be resumed; read the run to see what it needs."


@ROUTER_PROBLEMS.problem()
def run_changed() -> str:
    """Another call changed the run between this call's read of it and its write.

    Raised by ``Worker.answer`` or ``Worker.resume`` (a 409 from ``POST /runs/{run_id}/answers``
    or ``/resume``). Of two calls at once (an answer, a resume, on any worker), one wins and the other gets this; no
    node is called twice. Read the run again (``Worker.read``, ``GET /runs/{run_id}``), and repeat the call if the
    run still needs it.
    """
    return "The run changed since it was read, so nothing was done; read it again and repeat the call if it still applies."


@ROUTER_PROBLEMS.problem()
def previous_run_unloadable(*, run_id: str, reason: str) -> str:
    """A run is begun from the work of a previous run, which can't be loaded on the new run's caller's nodes.

    Raised as ``InvalidCallError`` by ``Worker.begin(..., previous=)`` once the new graph itself compiled: the
    previous run's graph uses a node or a version the caller's registry no longer has, or its saved run no longer
    reads back. Nothing is stored. ``ctx.reason`` is what conductor said of the previous run. The previous run only
    lends work, so start without ``previous``: every node is called again.

    Args:
        run_id: The previous run's id.
        reason: Why it can't be loaded: the messages of conductor's problems with it.
    """
    return f"The previous run '{run_id}' can't be loaded on the caller's nodes, so its work can't be reused: {reason.rstrip('.')}. Start without it."


@ROUTER_PROBLEMS.problem()
def resume_failed() -> str:
    """The worker raised an exception outside any node while it resumed the run, so ``resume()`` returned no run.

    The ``cause`` of a run whose status is ``failed``: the worker's ``context`` function raised, or its ``nodes_for``
    function, which returns a caller's registry, raised. A run conductor itself refused to resume has conductor's problem as its cause
    (``context_missing``, ``invalid_context``, ``saved_run_unreadable``, ``run_unwritable``). An exception from the store fails no run: the
    run is resumed again once its lease runs out. The run is kept as its last checkpoint, and answers queued meanwhile
    stay queued until it is resumed and stops again. The exception is in the log ``conductor_providers.fastapi``, with
    the run's id. Fix the cause, then resume the run (``Worker.resume``, ``POST /runs/{run_id}/resume``).
    """
    return "The run's worker failed while it resumed the run; its operator finds why in the server log. Resume it to carry on."


#: The HTTP status a route answers with when the worker refuses a call, by the code of its first problem; any code not
#: here, conductor's own among them, is a 422.
STATUS: dict[str, int] = {
    "no_such_run": 404,
    "already_answered": 409,
    "run_complete": 409,
    "run_not_resumable": 409,
    "run_changed": 409,
}
