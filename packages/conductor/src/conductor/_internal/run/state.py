"""The status of a run: whether it is ready, waiting, complete, or ended early.

Most of a run's status comes from its outcomes. Take the README's graph. Before any call, the run is ``ready``.
After ``graph.run({"text": "a\\nb\\nc"})`` every call has an outcome, so it is ``complete``. If a call is waiting for a
person, the run is ``waiting``. Once that call is answered, the run is ``ready`` again, or ``complete`` if nothing
is left to call.

The rest comes from how the last ``resume()`` ended, which the run stores beside its outcomes. A resume can end
early in four ways: a call failed (``failed``), a series had more items than the limit (``stopped``), the stream
was cancelled (``cancelled``), or the timeout passed (``timed_out``). A copy of the run made while a resume is
still going is ``running``. That copy only exists in memory, and a saved run never says ``running``.
``run.decline`` stores how it left the run in the same place: ``failed``, as if a resume had failed the call.

An answer changes some of this. A failed call or a series over the limit is still there after an answer, so the
run stays ``failed`` or ``stopped`` until a ``resume()`` tries again. A cancel or a timeout belonged to that one
resume, so once a call is answered, the status comes from the outcomes again.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from pydantic import ConfigDict

from conductor._internal.problem import Problem, run_problems

from .progress import call_count, is_complete

if TYPE_CHECKING:
    from .type import Run

#: The status of a run, as ``run.status`` gives it. ``ready``, ``waiting`` and ``complete`` come from its outcomes.
#: ``running`` is a copy made while a ``resume()`` is still going. The other four say how the last ``resume()``
#: ended the run early.
RunStatus = Literal[
    "ready", "running", "waiting", "complete", "failed", "stopped", "cancelled", "timed_out"
]


@dataclass(frozen=True, slots=True)
class EndedEarly:
    """How ``resume()`` ended a run before it was complete, and why.

    The next ``resume()`` goes on from where this one stopped.

    Attributes:
        status: ``failed`` when a node raised or a question was declined, ``stopped`` when a series had more items than ``max_items``,
            ``cancelled`` when the stream's ``cancel`` was set, and ``timed_out`` when the ``timeout`` passed.
        cause: The problem that ended the run, with its ``loc`` and code.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    status: Literal["failed", "stopped", "cancelled", "timed_out"]
    cause: Problem


#: How the last ``resume()`` left a run. ``"running"`` for a copy made while it was still going, such as a
#: checkpoint or the run a stream hands back. An ``EndedEarly`` when it ended early. ``None`` when it made every
#: call that was ready.
LastResume = EndedEarly | Literal["running"] | None


def status_of(run: Run[Any]) -> tuple[RunStatus, Problem | None]:
    """Returns the status of ``run``, and the problem that ended it early if it did.

    The checks go in this order. A copy made while a resume is still going is ``running``. A run whose every
    function node is complete is ``complete``, however the last resume ended. Then a run the last resume ended
    early has that status and its cause. Otherwise the run is ``waiting`` when a call waits for a person, and
    ``ready`` when none does.
    """
    last_resume, outcomes = run._last_resume, run._outcomes
    if last_resume == "running":
        return "running", None
    counts = [call_count(outcomes, node) for node in run._blueprint.function_nodes.values()]
    if all(is_complete(count) for count in counts):
        return "complete", None
    if last_resume is not None:
        return last_resume.status, last_resume.cause
    if any(count.waiting for count in counts):
        return "waiting", None
    return "ready", None


def kept_by_an_answer(last_resume: LastResume) -> LastResume:
    """Returns what a run still says about its last ``resume()`` once a call is answered.

    A failed call and a series over the item limit stay until a ``resume()`` tries again, so they are kept. A
    cancel, a timeout or a resume still going belonged to that resume, and the answer comes after it, so they are
    dropped.
    """
    return (
        last_resume
        if isinstance(last_resume, EndedEarly) and last_resume.status in ("failed", "stopped")
        else None
    )


def deadline(timeout: float | None) -> Callable[[], EndedEarly | None] | None:
    """Returns a check that says when ``timeout`` seconds have passed, or ``None`` when there is no timeout.

    The clock starts when this is called. The scheduler calls the check before each call. It returns
    ``EndedEarly("timed_out", ...)`` once the time is up, and ``None`` before that.
    """
    if timeout is None:
        return None
    end = time.monotonic() + timeout
    return lambda: (
        EndedEarly("timed_out", run_problems.run_timed_out(timeout=timeout))
        if time.monotonic() > end
        else None
    )
