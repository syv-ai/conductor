"""A run as a store keeps it: the row, the columns an update changes, and the two texts the router writes in it.

``state`` and ``outcome`` are JSON, each read and written by its own model here. A store only keeps them.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, TypedDict

from conductor import EndedEarly, Going, Problem, RunState, RunStatus, Waiting
from pydantic import BaseModel


@dataclass(frozen=True)
class StoredRun:
    """One run as the store keeps it.

    A store acts on ``id``, ``rev``, ``owner`` and ``lease_until``. ``status`` and ``caller`` are columns for the
    host's own queries. The other four are text the router writes and reads, which a store only keeps.

    Attributes:
        id: The run's id.
        graph: The ``GraphRecord`` the run is of, as JSON; never changed.
        caller: The caller who started it, as JSON, so any worker can build the run's ``from_run`` again; never changed.
        run: ``run.to_json()``: the last checkpoint while a leg runs, the whole run once it ends.
        status: ``running`` while a leg is wanted, else how the last leg left the run. It is for the host's own
            queries: the router writes it in the same ``update`` as ``state``, from it, and reads only ``state``.
        state: How the run stands and what was asked of its running leg, as JSON: its status and why it ended
            early, the answers queued and refused, a cancel that was asked. Small whatever the run holds.
        outcome: What the last leg's end left for a reader, as JSON: what waits for a person and, once the run is
            complete, its outputs. Written when a leg ends, so reading a run doesn't load it.
        rev: Moved one on by every ``update``.
        owner: The worker running a leg of it.
        lease_until: Until when the owner's lease holds, as seconds since the epoch; a time already past for a run
            handed back for any worker to go on with; ``None`` when no leg is wanted.
    """

    id: str
    graph: str
    caller: str
    run: str
    status: RunStatus
    state: str
    outcome: str
    rev: int
    owner: str | None
    lease_until: float | None


class RunChanges(TypedDict, total=False):
    """The columns one ``update`` changes: any of those of a ``StoredRun`` that change after it is created."""

    run: str
    status: RunStatus
    state: str
    outcome: str
    owner: str | None
    lease_until: float | None


#: The state of a run a leg is wanted for.
RUNNING = Going("running")


class Outcome(BaseModel):
    """What a leg's end leaves for a reader, so that reading a run doesn't load it: a row's ``outcome``, as JSON.

    Attributes:
        waiting: What waits for a person.
        outputs: The graph's outputs by address once the run is complete; ``None`` until then.
    """

    waiting: list[Waiting]
    outputs: dict[str, Any] | None

    @classmethod
    def empty(cls) -> Outcome:
        """The outcome of a run no leg has ended: nothing waits, no outputs."""
        return cls(waiting=[], outputs=None)


class StoredState(BaseModel):
    """How a run stands and what was asked of its running leg: a row's ``state``, as JSON.

    It is small whatever the run holds, so a route that queues an answer or asks a cancel reads and writes it
    without touching the run's ``outcome``.

    Attributes:
        run_state: ``running`` while a leg is wanted; else how the last leg left the run, as the run itself says
            it, with its cause when it ended early.
        queued_answers: Answers that came while a leg ran, by waiting id; the run gets them when the leg ends.
        refused: Queued answers the run refused when a leg ended, by waiting id, each with its problems.
        cancel_asked: Whether someone asked the running leg to stop. The end of the leg that stopped for it clears
            it; one that came too late for its leg is kept for the next leg, when a queued answer starts one.
    """

    run_state: RunState
    queued_answers: dict[str, Any]
    refused: dict[str, list[Problem]]
    cancel_asked: bool

    @classmethod
    def started(cls) -> StoredState:
        """The state of a run that has just started: a leg is wanted, and nothing is queued, refused or asked."""
        return cls(run_state=RUNNING, queued_answers={}, refused={}, cancel_asked=False)

    @property
    def leg_wanted(self) -> bool:
        """Whether a leg is wanted for the run: a worker runs one, or the run waits for a worker to take it."""
        return self.run_state == RUNNING

    def answered(self, waiting_ids: Collection[str]) -> StoredState:
        """This state once answers to some questions are taken: nothing is refused or queued for them any more.

        Args:
            waiting_ids: The waiting ids whose answers the run took.

        Returns:
            The state without them in ``refused`` and ``queued_answers``.
        """
        left = {
            name: {waiting_id: held for waiting_id, held in column.items() if waiting_id not in waiting_ids}
            for name, column in (("queued_answers", self.queued_answers), ("refused", self.refused))
        }
        return self.model_copy(update=left)

    @property
    def cause(self) -> Problem | None:
        """Why the run's last leg ended early; ``None`` for a run whose last leg didn't, which is not resumed."""
        return self.run_state.cause if isinstance(self.run_state, EndedEarly) else None

    def columns(self) -> RunChanges:
        """The row's ``status`` and ``state`` for this state.

        Nothing else writes either column, so each write of ``state`` writes the ``status`` it holds in the same
        statement. ``status`` is for the host's own queries: the worker and the routes read ``state`` and never
        ``status``.

        Returns:
            The two columns, to write in one ``update``.
        """
        return {"status": self.run_state.status, "state": self.model_dump_json()}


@dataclass(frozen=True)
class RunAsRead:
    """A run as read from the store: its row, with the row's state read from its JSON once.

    Attributes:
        row: The row.
        state: The row's ``state``.
    """

    row: StoredRun
    state: StoredState

    @classmethod
    def of(cls, row: StoredRun) -> RunAsRead:
        """A row with its state read.

        Args:
            row: The row, as the store gave it.

        Returns:
            The row and its state.
        """
        return cls(row, StoredState.model_validate_json(row.state))
