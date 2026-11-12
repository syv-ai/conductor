"""How a leg of a stored run ended, and what its row holds once that is written.

A leg ends in one of three ways: with a run, having raised, or handed back by a worker that shuts down. Each is a
record that says, from the row's state as last read, what the row holds after it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

from conductor import EndedEarly, InvalidCallError, Problem, Run, Waiting
from conductor.events import RunEnd
from pydantic_core import from_json

from conductor_providers.fastapi import router_problem
from conductor_providers.fastapi.stored_run import RUNNING, Outcome, RunChanges, StoredState

log = logging.getLogger("conductor_providers.fastapi")


@dataclass(frozen=True)
class RowAfterALeg:
    """What a row holds once the end of a leg is written.

    Attributes:
        state: The row's state; a next leg is wanted when it says so.
        changes: The row's ``run`` and ``outcome``, each where it changes.
    """

    state: StoredState
    changes: RunChanges


class LegEnd(ABC):
    """How a leg ended.

    Attributes:
        run_end: The leg's ``run_end`` event to store, as JSON; ``None`` for a leg handed back, whose run did not end.
    """

    run_end: str | None

    @abstractmethod
    def row_after(self, state: StoredState) -> RowAfterALeg:
        """What a row holds after this end.

        Args:
            state: The row's state as last read.

        Returns:
            The row's state and the columns that change with it.
        """


@dataclass(frozen=True)
class Ended(LegEnd):
    """A leg that ended with a run.

    Attributes:
        run: The run the leg ended with.
        saved: That run, as JSON.
        outcome: The row's ``outcome`` after the leg, as JSON.
        run_end: The leg's ``run_end`` event, as JSON.
    """

    run: Run
    saved: str
    outcome: str
    run_end: str

    def row_after(self, state: StoredState) -> RowAfterALeg:
        """What a row holds after this end, once the run has been given the answers queued during the leg.

        Each queued answer is given to the run on its own, as ``run.answer`` takes one whatever the run's status:
        one the run refuses stays in ``refused`` with its problems, and the ones it takes make the run the next
        leg starts from. An answer whose check raises what is no refusal (a fault in a host's type) is logged and
        refused as ``answer_check_failed``, so one such answer doesn't cost the leg its end. A cancel that was
        asked and that this leg did not end on is kept for that next leg, which then ends before it calls
        anything; one the leg ended on, or that no leg is left to hear, is done with.
        """
        run, taken, refused = self.run, [], {}
        for waiting_id, value in state.queued_answers.items():
            try:
                run = run.answer(waiting_id, value)
                taken.append(waiting_id)
            except InvalidCallError as error:
                refused[waiting_id] = error.errors()
            except Exception as error:
                log.exception("the answer queued for %s could not be checked", waiting_id)
                refused[waiting_id] = [router_problem.answer_check_failed(waiting_id=waiting_id, error=type(error).__name__)]
        settled = state.answered(taken)
        unheard = state.cancel_asked and bool(taken) and self.run.status != "cancelled"
        after = settled.model_copy(update={
            "run_state": RUNNING if taken else self.run.state, "queued_answers": {}, "refused": {**settled.refused, **refused}, "cancel_asked": unheard,
        })
        return RowAfterALeg(after, {"run": run.to_json() if taken else self.saved, "outcome": self.outcome})


@dataclass(frozen=True)
class Raised(LegEnd):
    """A leg that raised outside a node, or that conductor refused: it ended with no run to store.

    Attributes:
        cause: Why the run failed.
        outcome: The row's ``outcome`` after the leg, as JSON.
        run_end: The leg's ``run_end`` event, as JSON.
    """

    cause: Problem
    outcome: str
    run_end: str

    @classmethod
    def because(cls, cause: Problem, waiting: list[Waiting]) -> Raised:
        """The end of a leg that raised.

        The ``run_end`` is made without validation: a ``RunEnd`` requires the run it ended with, which it never
        writes, and a leg that raised has none.

        Args:
            cause: Why the run failed.
            waiting: What waits in the run as the row holds it, its last checkpoint: the run an answer is given
                to. Nothing, for a run that could not be loaded.

        Returns:
            The end, with its ``run_end`` saying ``failed`` and the cause.
        """
        outcome = Outcome(waiting=waiting, outputs=None)
        return cls(cause, outcome.model_dump_json(), RunEnd.model_construct(status="failed", cause=cause).model_dump_json())

    @classmethod
    def refused(cls, refusal: InvalidCallError, waiting: list[Waiting]) -> Raised:
        """The end of a leg that conductor refused: the run can't start its leg, or can't be read or written.

        Trying again would be refused again, so the run fails with conductor's own problem as its cause. A cause
        is one problem: of several, it is the first.

        Args:
            refusal: What conductor raised.
            waiting: What waits in the run as the row holds it.

        Returns:
            The end, its cause the first of the refusal's problems.
        """
        return cls.because(refusal.errors()[0], waiting)

    def row_after(self, state: StoredState) -> RowAfterALeg:
        """What a row holds after this end: the run failed, and the row keeps its last checkpoint, lists what
        waits in it, and keeps the answers queued during the leg for the end of the next leg to deal with."""
        after = state.model_copy(update={"run_state": EndedEarly("failed", self.cause), "cancel_asked": False})
        return RowAfterALeg(after, {"outcome": self.outcome})


@dataclass(frozen=True)
class HandedBack(LegEnd):
    """A leg its worker stopped to shut down: the run did not end, and the row holds it as the leg left it."""

    run_end: None = None

    def row_after(self, state: StoredState) -> RowAfterALeg:
        """What a row holds after this end: what it held, a cancel asked meanwhile included, for the worker that
        takes the run to go on from."""
        return RowAfterALeg(state, {})


def end_of(run_end: RunEnd, stored: list[Waiting]) -> LegEnd:
    """The end of a leg whose stream ended with a ``run_end``.

    A run that can't be written as JSON (``run_unwritable``: a value in it has no JSON form) can't be stored
    however often it is tried, so its leg ended as one that raised, with that problem as the cause.

    Args:
        run_end: The leg's last event, with the run the leg ended with.
        stored: What waits in the run as the row holds it, its last checkpoint, which stays when the run can't
            be written.

    Returns:
        The end with that run: the run as JSON, what waits in it and, once complete, its outputs, each written by
        its type as the run writes it. Or the end of a leg that raised, for a run that can't be written.
    """
    run = run_end.run
    try:
        outcome = Outcome(waiting=run.waiting, outputs=from_json(run.outputs_to_json()) if run.status == "complete" else None)
        return Ended(run, run.to_json(), outcome.model_dump_json(), run_end.model_dump_json())
    except InvalidCallError as unwritable:
        return Raised.refused(unwritable, stored)
