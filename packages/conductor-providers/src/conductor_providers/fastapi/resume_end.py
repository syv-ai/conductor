"""How one call of ``resume()`` on a stored run ended, and what the run's row in the store is changed to.

A worker loads the run from its row and resumes it (as ``run.stream()``, which is ``run.resume()`` with its events).
That call ends in one of three ways, each a class here: ``Ended`` with the run it returned, ``Raised`` when it
raised, ``HandedBack`` when the worker shut down before it ended. Each has ``row_after(state)``, which takes the
row's state as last read and returns the state and columns to write.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from conductor import InvalidCallError, Problem, Run
from conductor.events import RunEnd

from conductor_providers.fastapi.problems import raised_outside_a_node
from conductor_providers.fastapi.stored_run import Outcome, RunChanges, StoredState

log = logging.getLogger("conductor_providers.fastapi")


@dataclass(frozen=True)
class RowAfterResume:
    """What to write to a run's row after one call of ``resume()`` ended.

    Attributes:
        state: The row's new state. Its ``is_running`` says whether the run is to be resumed again at once.
        changes: The new values of the row's ``run`` and ``outcome`` columns; a column that stays as it is, is left out.
    """

    state: StoredState
    changes: RunChanges


class ResumeEnd(ABC):
    """How one call of ``resume()`` on a stored run ended.

    Attributes:
        run_end: The call's ``run_end`` event to store, as JSON; ``None`` when the worker shut down before the run ended.
    """

    run_end: str | None

    @abstractmethod
    def row_after(self, state: StoredState) -> RowAfterResume:
        """What to write to the row after a resume call that ended this way.

        Args:
            state: The row's state as last read.

        Returns:
            The row's state and the columns that change with it.
        """


@dataclass(frozen=True)
class Ended(ResumeEnd):
    """A resume call that returned a run: complete, waiting, or ended early by the run's own account.

    Attributes:
        run: The run the call returned.
        saved: That run, as JSON.
        outcome: The row's ``outcome`` to write, as JSON.
        run_end: The call's ``run_end`` event, as JSON.
    """

    run: Run
    saved: str
    outcome: str
    run_end: str

    def row_after(self, state: StoredState) -> RowAfterResume:
        """What to write to the row, after the run has been given the answers queued while it was running.

        The queued answers are given to the run in the order they were sent, one waiting id at a time, with
        ``run.answer``, which takes one whatever the run's status. So a wrong answer is refused alone. An answer the
        run refuses goes to ``refused`` with its problems, and one it takes clears what ``refused`` had for its
        question: ``refused`` ends up saying what became of the last answer to each question. A second answer to a
        question the first one answered is refused by conductor as ``not_waiting``. An answer whose check raises
        something other than a refusal (a fault in a host's type) is logged and refused as
        ``raised_outside_a_node``, so one such answer doesn't keep the end from being written.

        If the run takes any answer, the row's state is ``running`` again and its ``run`` is the answered run, so the
        worker resumes it at once. A cancel that was asked but did not end this resume call stays asked then, and
        the next resume call ends before it calls any node. Otherwise the cancel is cleared.
        """
        run, taken, refused = self.run, False, dict(state.refused)
        for answers in state.queued_answers:
            for waiting_id, value in answers.items():
                try:
                    run = run.answer(waiting_id, value)
                except InvalidCallError as error:
                    refused[waiting_id] = error.problems
                except Exception as error:
                    log.exception("the answer queued for %s could not be checked", waiting_id)
                    refused[waiting_id] = [raised_outside_a_node(error=type(error).__name__)]
                else:
                    taken = True
                    refused.pop(waiting_id, None)
        unheard = state.cancel_asked and taken and self.run.status != "cancelled"
        stopped = state.running() if taken else state.stopped(self.run)
        after = stopped.model_copy(
            update={"queued_answers": [], "refused": refused, "cancel_asked": unheard}
        )
        return RowAfterResume(
            after, {"run": run.to_json() if taken else self.saved, "outcome": self.outcome}
        )


@dataclass(frozen=True)
class Raised(ResumeEnd):
    """A resume call that raised outside a node, or that conductor refused: it returned no run to store.

    Attributes:
        cause: Why the run failed.
        outcome: The row's ``outcome`` to write, as JSON.
        run_end: The ``run_end`` event to store for the call, as JSON.
    """

    cause: Problem
    outcome: str
    run_end: str

    @classmethod
    def because(cls, cause: Problem, held: Outcome) -> Raised:
        """The end of a resume call that raised: a ``run_end`` that says ``failed``, with no run, since the call returned none.

        Args:
            cause: Why the run failed.
            held: What waits in the run as the row has it, its last checkpoint (the run an answer is given to),
                and its progress there. ``Outcome.empty()``, for a run that could not be loaded.

        Returns:
            The end, with its ``run_end`` saying ``failed`` and the cause.
        """
        return cls(
            cause,
            held.model_dump_json(),
            RunEnd(status="failed", cause=cause).model_dump_json(),
        )

    @classmethod
    def refused(cls, refusal: InvalidCallError, held: Outcome) -> Raised:
        """The end of a resume call that conductor refused: the run can't be resumed, or can't be read or written.

        Trying again would be refused again, so the run fails with conductor's own problem as its cause. A cause
        is one problem: of several, it is the first.

        Args:
            refusal: What conductor raised.
            held: The outcome of the run as the row has it.

        Returns:
            The end, its cause the first of the refusal's problems.
        """
        return cls.because(refusal.problems[0], held)

    def row_after(self, state: StoredState) -> RowAfterResume:
        """What to write to the row: the run failed. The row keeps its last checkpoint as ``run``, its ``outcome``
        lists what waits in that checkpoint and its progress there, and the answers queued while the run was
        running stay queued until the run is resumed and stops again."""
        after = state.failed(self.cause).model_copy(update={"cancel_asked": False})
        return RowAfterResume(after, {"outcome": self.outcome})


@dataclass(frozen=True)
class HandedBack(ResumeEnd):
    """A resume call its worker stopped in order to shut down: the run did not end, and the row has its last checkpoint."""

    run_end: None = None

    def row_after(self, state: StoredState) -> RowAfterResume:
        """What to write to the row: nothing. Its state, a cancel asked meanwhile included, is what the worker
        that takes the run goes on from."""
        return RowAfterResume(state, {})


def end_of(run_end: RunEnd, held: Outcome) -> ResumeEnd:
    """The end of a resume call whose stream ended with a ``run_end``.

    A run that can't be written as JSON (``run_unwritable``: a value in it has no JSON form) can't be stored
    however often it is tried, so the call is treated as one that raised, with that problem as the cause.

    Args:
        run_end: The stream's last event, with the run the call returned.
        held: The outcome of the run as the row has it, its last checkpoint, which stays when the run can't be
            written.

    Returns:
        The end with that run: the run as JSON and its outcome (``Outcome.of_stopped``). Or ``Raised``, for a run that
        can't be written.
    """
    run = run_end.run
    try:
        return Ended(
            run, run.to_json(), Outcome.of_stopped(run).model_dump_json(), run_end.model_dump_json()
        )
    except InvalidCallError as unwritable:
        return Raised.refused(unwritable, held)
