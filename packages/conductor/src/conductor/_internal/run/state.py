"""The status of a run: whether it is ready, waiting, complete, or ended early.

Most of a run's status comes from its outcomes. Take the README's graph. Before any call, the run is ``ready``.
After ``graph.run({"text": "a\\nb\\nc"})`` every call has an outcome, so it is ``complete``. If a call is waiting for a
person, the run is ``waiting``. Once that call is answered, the run is ``ready`` again, or ``complete`` if nothing
is left to call.

The rest comes from how the last ``resume()`` ended, which the run stores beside its outcomes. A resume can end
early in four ways: a call failed (``failed``), a series had more items than the limit (``stopped``), the host
cancelled it (``cancelled``), or the timeout passed (``timed_out``). The next ``resume()`` goes on from there. A
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
from asyncio import Event as AsyncioEvent
from collections.abc import Callable
from dataclasses import dataclass
from threading import Event as ThreadEvent
from typing import ClassVar, Literal, Self

from pydantic import ConfigDict, model_validator

from conductor._internal.base import ConductorModel, ItemIndexes, NodePath
from conductor._internal.problem import Problem, conductor_error, run_problems

from .blueprint import outcome_loc
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

    # A saved run reads it strictly, as it is written: an item index is a whole number, never "1" or true, and a
    # key it doesn't declare is refused.
    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(strict=True, extra="forbid")

    node_path: NodePath
    item_indexes: ItemIndexes
    question: Unanswered
    reason: str | None = None


#: The one problem that ends a run early with each of these statuses: a series over the item limit stops it, a cancel
#: cancels it, and the timeout times it out. A failed run has no such code: it ends with the problem its call failed
#: with, and a node's ``NodeError`` may give that any code.
_CAUSE_OF_STATUS = {
    "stopped": run_problems.item_limit.code,
    "cancelled": run_problems.run_cancelled.code,
    "timed_out": run_problems.run_timed_out.code,
}


class EndedEarly(ConductorModel):
    """How a run ended before it was complete, and why.

    A ``resume()`` ends a run early as ``failed``, ``stopped``, ``cancelled`` or ``timed_out``, and the next
    ``resume()`` goes on from where it stopped. ``run.with_decline`` ends it as ``declined``, and that is final.

    A decline keeps the call declined, in ``declined``, and no ``cause``: its ``answer_declined`` is made from the
    call whenever the run's cause is asked for (``cause_of``), so the decline has one source. The call itself has no
    outcome any more, so it waits on nothing, and ``again`` calls it afresh.

    Any other status keeps its ``cause`` and no declined call. A stopped, cancelled or timed-out run keeps the one
    problem that ends a run that way (``_CAUSE_OF_STATUS``). A failed run keeps the problem of the call that failed,
    so its cause is at a call: its ``loc`` isn't empty, as a cancel's and a timeout's is. pydantic refuses any other
    ``EndedEarly`` when it is made, so a saved run that holds one is refused where it is read
    (``saved_run_unreadable``, at ``ended_early``).

    Attributes:
        status: ``failed`` when a node raised, ``stopped`` when a series had more items than ``max_items``,
            ``cancelled`` when the run's ``cancel`` (given to ``resume``, ``graph.run`` or ``stream``) was set, ``timed_out`` when the ``timeout`` passed, and
            ``declined`` when a person declined to answer a question.
        cause: The problem that ended the run, with its ``loc`` and code. ``None`` for a ``declined`` run.
        declined: For a ``declined`` run, the call declined. ``None`` for any other status.
    """

    status: Literal["failed", "stopped", "cancelled", "timed_out", "declined"]
    cause: Problem | None = None
    declined: Declined | None = None

    @model_validator(mode="after")
    def _one_a_run_writes(self) -> Self:
        """Refuses an early end no run writes, as the class says.

        Raises:
            PydanticCustomError: ``saved_lists_disagree``, since only a saved run can give one: its message names no
                value.
        """
        if (self.status == "declined") != (self.declined is not None):
            wrong = "a run keeps a declined call when, and only when, it is declined"
        elif (self.cause is None) != (self.declined is not None):
            wrong = "a declined run keeps no cause, and any other run that ended early keeps one"
        elif self.cause is not None and (
            self.cause.type != code
            if (code := _CAUSE_OF_STATUS.get(self.status)) is not None
            else not self.cause.loc
        ):
            wrong = f"a {self.status} run can't have this cause"
        else:
            return self
        raise conductor_error("saved_lists_disagree", wrong)


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
        return last_resume.status, cause_of(last_resume, outcomes)
    if any(count.waiting for count in counts):
        return "waiting", None
    return "ready", None


def cause_of(ended_early: EndedEarly, outcomes: RunOutcomes) -> Problem:
    """Returns the problem that ended a run early.

    A decline's is ``answer_declined`` at the call declined, with the reason given, as ``run.with_decline`` makes it.
    Any other is the ``cause`` it keeps.

    Raises:
        RuntimeError: A run that ended early other than by a decline keeps no cause. ``EndedEarly`` refuses one, so
            that's a bug.
    """
    if (declined := ended_early.declined) is not None:
        node = outcomes.blueprint.function_nodes[declined.node_path]
        return run_problems.answer_declined(
            *outcome_loc(node, declined.item_indexes), reason=declined.reason
        )
    if ended_early.cause is None:
        raise RuntimeError(f"a run that is {ended_early.status} keeps no cause")
    return ended_early.cause


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


def stop_check(
    timeout: float | None,
    *cancels: AsyncioEvent | ThreadEvent | None,
    hand_back: AsyncioEvent | None = None,
) -> Callable[[], LastResume] | None:
    """Returns the check the scheduler asks before each call: whether the run must stop now, and how it ends.

    It says ``cancelled`` once any of ``cancels`` is set, then ``"running"`` once ``hand_back`` is set, so a cancel
    set as well wins, then ``timed_out`` once ``timeout`` seconds have passed since this was called. Before that it
    says ``None``. With no event and no timeout there is nothing to check, so it returns ``None`` itself.
    """
    events = [event for event in cancels if event is not None]
    timed_out = _deadline(timeout)
    if not events and hand_back is None and timed_out is None:
        return None

    def check() -> LastResume:
        if any(event.is_set() for event in events):
            return EndedEarly(status="cancelled", cause=run_problems.run_cancelled())
        if hand_back is not None and hand_back.is_set():
            return "running"  # the scheduler stops, and another process goes on with the run
        return timed_out() if timed_out is not None else None

    return check


def _deadline(timeout: float | None) -> Callable[[], EndedEarly | None] | None:
    """Returns a check that says when ``timeout`` seconds have passed, or ``None`` when there is no timeout.

    The clock starts when this is called. It returns ``EndedEarly(status="timed_out", ...)`` once the time is up, and
    ``None`` before that.
    """
    if timeout is None:
        return None
    end = time.monotonic() + timeout
    return lambda: (
        EndedEarly(status="timed_out", cause=run_problems.run_timed_out(timeout=timeout))
        if time.monotonic() > end
        else None
    )
