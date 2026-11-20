"""The status of a run: whether it is ready, waiting, complete, or ended early.

Most of a run's status comes from its outcomes. Take the README's graph. Before any call, the run is ``ready``.
After ``graph.run({"text": "a\\nb\\nc"})`` every call has an outcome, so it is ``complete``. If a call is waiting for a
person, the run is ``waiting``. Once that call is answered, the run is ``ready`` again, or ``complete`` if nothing
is left to call.

The rest comes from how the last ``resume()`` ended, which the run stores beside its outcomes. A resume can end
early in four ways: a call failed (``failed``), a series had more items than the limit (``stopped``), the stream
was cancelled (``cancelled``), or the timeout passed (``timed_out``). The next ``resume()`` goes on from there. A
copy of the run made while a resume is still going is ``running``. That copy only exists in memory, and a saved run
never says ``running``.

``run.with_decline`` stores how it left the run in the same place: ``declined``. A person said no, so the run has ended
for good. Nothing more is called, and nothing waits.

An answer changes some of how a resume ended. A failed call or a series over the limit is still there after an
answer, so the run stays ``failed`` or ``stopped`` until a ``resume()`` tries again. A cancel or a timeout belonged
to that one resume, so once a call is answered, the status comes from the outcomes again.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from conductor._internal.base import ConductorModel, ItemIndexes, NodePath
from conductor._internal.problem import Problem, run_problems

from .outcome import Unanswered
from .outcomes import RunOutcomes
from .progress import call_count, is_complete

#: The status of a run, as ``run.status`` gives it. ``ready``, ``waiting`` and ``complete`` come from its outcomes.
#: ``running`` is a copy made while a ``resume()`` is still going. ``declined`` is a run a person ended with
#: ``run.with_decline``. The other four say how the last ``resume()`` ended the run early.
RunStatus = Literal[
    "ready",
    "running",
    "waiting",
    "complete",
    "failed",
    "stopped",
    "cancelled",
    "timed_out",
    "declined",
]


@dataclass(frozen=True)
class Declined:
    """A call a person declined to answer: where it is, what it asked, and the reason they gave.

    ``run.with_decline`` keeps it with the run, so ``run.asked`` can list the decline, and a saved run writes it. Say
    ``approve`` waited for its second line and the person declined it as "Over budget". It is
    ``Declined(("approve",), (1,), Unanswered("Approve b?", fingerprint=None), "Over budget")``.

    Attributes:
        node_path: The node that asked, by its path (``("review", "approve")`` inside a subgraph).
        item_indexes: Which call of the node it was.
        question: What the call's ``Ask`` said. It has no fingerprint, since nothing reuses a declined call.
        reason: The reason the person gave, or ``None`` when they gave none.
    """

    node_path: NodePath
    item_indexes: ItemIndexes
    question: Unanswered
    reason: str | None = None


class EndedEarly(ConductorModel):
    """How a run ended before it was complete, and why.

    A ``resume()`` ends a run early as ``failed``, ``stopped``, ``cancelled`` or ``timed_out``, and the next
    ``resume()`` goes on from where it stopped. ``run.with_decline`` ends it as ``declined``, and that is final.

    A decline also keeps the call declined, in ``declined``. The call itself has no outcome any more, so it waits on
    nothing, and ``again`` calls it afresh.

    Attributes:
        status: ``failed`` when a node raised, ``stopped`` when a series had more items than ``max_items``,
            ``cancelled`` when the stream's ``cancel`` was set, ``timed_out`` when the ``timeout`` passed, and
            ``declined`` when a person declined to answer a question.
        cause: The problem that ended the run, with its ``loc`` and code.
        declined: For a ``declined`` run, the call declined. ``None`` for any other status.
    """

    status: Literal["failed", "stopped", "cancelled", "timed_out", "declined"]
    cause: Problem
    declined: Declined | None = None


#: How the last ``resume()`` left a run. ``"running"`` for a copy made while it was still going, such as a
#: checkpoint or the run a stream hands back. An ``EndedEarly`` when it ended early, or when ``run.with_decline`` ended
#: it. ``None`` when it made every call that was ready.
LastResume = EndedEarly | Literal["running"] | None


def status_of(outcomes: RunOutcomes, last_resume: LastResume) -> tuple[RunStatus, Problem | None]:
    """Returns the status of a run with these outcomes, and the problem that ended it early if it did.

    ``last_resume`` is how the run's last ``resume()`` left it.

    The checks go in this order. A copy made while a resume is still going is ``running``. A run whose every
    function node is complete is ``complete``, however the last resume ended. Then a run the last resume ended
    early has that status and its cause. Otherwise the run is ``waiting`` when a call waits for a person, and
    ``ready`` when none does.
    """
    if last_resume == "running":
        return "running", None
    counts = [call_count(outcomes, node) for node in outcomes.blueprint.function_nodes.values()]
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
    ``EndedEarly(status="timed_out", ...)`` once the time is up, and ``None`` before that.
    """
    if timeout is None:
        return None
    end = time.monotonic() + timeout
    return lambda: (
        EndedEarly(status="timed_out", cause=run_problems.run_timed_out(timeout=timeout))
        if time.monotonic() > end
        else None
    )
