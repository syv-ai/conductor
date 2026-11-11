"""How a leg of a stored run ended, and what its row holds once that is written.

A leg ends in one of three ways: with a run, having raised, or handed back by a worker that shuts down. Each is a
record that says, from the row's state as last read, what the row holds after it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from conductor import EndedEarly, InvalidCallError, Problem, Run
from conductor.events import RunEnd
from pydantic_core import from_json

from conductor_providers.fastapi.stored_run import RUNNING, Outcome, RunChanges, StoredState


@dataclass(frozen=True)
class RowAfterALeg:
    """What a row holds once the end of a leg is written.

    Attributes:
        state: The row's state.
        changes: The row's ``run`` and ``outcome``, each where it changes.
        run_end: The leg's ``run_end`` event to store, as JSON; ``None`` for a leg handed back, whose run did not end.
        leg_wanted: Whether a next leg is wanted: the leg was handed back, or the run took an answer queued during it.
    """

    state: StoredState
    changes: RunChanges
    run_end: str | None
    leg_wanted: bool


class LegEnd(ABC):
    """How a leg ended."""

    @abstractmethod
    def row_after(self, state: StoredState) -> RowAfterALeg:
        """What a row holds after this end.

        Args:
            state: The row's state as last read.

        Returns:
            The row's state and the columns that change with it, the ``run_end`` to store, and whether a next leg
            is wanted.
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
        leg starts from.
        """
        run, refused, taken = self.run, dict(state.refused), False
        for waiting_id, value in state.queued_answers.items():
            try:
                run, taken = run.answer(waiting_id, value), True
            except InvalidCallError as error:
                refused[waiting_id] = error.errors()
        after = state.model_copy(update={"run_state": RUNNING if taken else self.run.state, "queued_answers": {}, "refused": refused, "cancel_asked": False})
        return RowAfterALeg(after, {"run": run.to_json() if taken else self.saved, "outcome": self.outcome}, self.run_end, leg_wanted=taken)


@dataclass(frozen=True)
class Raised(LegEnd):
    """A leg that raised outside a node, or whose run can't be written: it ended with no run to store.

    Attributes:
        cause: Why the run failed.
    """

    cause: Problem

    def row_after(self, state: StoredState) -> RowAfterALeg:
        """What a row holds after this end: the run failed with nothing waiting and no outputs, and the row keeps
        its last checkpoint and the answers queued during the leg, for the end of the next leg to deal with.

        The ``run_end`` is made without validation: a ``RunEnd`` requires the run it ended with, which it never
        writes, and a leg that raised has none.
        """
        after = state.model_copy(update={"run_state": EndedEarly("failed", self.cause), "cancel_asked": False})
        run_end = RunEnd.model_construct(status="failed", cause=self.cause).model_dump_json()
        return RowAfterALeg(after, {"outcome": Outcome.empty().model_dump_json()}, run_end, leg_wanted=False)


class HandedBack(LegEnd):
    """A leg its worker stopped to shut down: the run did not end, and the row holds it as the leg left it."""

    def row_after(self, state: StoredState) -> RowAfterALeg:
        """What a row holds after this end: what it held, a cancel asked meanwhile included, for the worker that
        takes the run to go on from."""
        return RowAfterALeg(state, {}, run_end=None, leg_wanted=True)


def end_of(run_end: RunEnd) -> LegEnd:
    """The end of a leg whose stream ended with a ``run_end``.

    A run that can't be written as JSON (``state_unwritable``: a value in it has no JSON form) can't be stored
    however often it is tried, so its leg ended as one that raised, with that problem as the cause.

    Args:
        run_end: The leg's last event, with the run the leg ended with.

    Returns:
        The end with that run: the run as JSON, what waits in it and, once complete, its outputs, each written by
        its type as the run writes it. Or the end of a leg that raised, for a run that can't be written.
    """
    run = run_end.run
    try:
        outcome = Outcome(waiting=run.waiting, outputs=from_json(run.outputs_json()) if run.status == "complete" else None)
        return Ended(run, run.to_json(), outcome.model_dump_json(), run_end.model_dump_json())
    except InvalidCallError as unwritable:
        return Raised(unwritable.errors()[0])
