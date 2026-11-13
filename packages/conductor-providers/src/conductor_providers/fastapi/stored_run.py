"""A run as a store keeps it: the row, the columns an update changes, and the two texts the router writes in it.

``state`` and ``outcome`` are JSON, each read and written by its own model here. A store only keeps them.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, TypedDict

from conductor import Problem, Run, RunProgress, RunStatus, Waiting
from pydantic import BaseModel
from pydantic_core import from_json


@dataclass(frozen=True)
class StoredRun:
    """One run as the store keeps it.

    A store acts on ``id``, ``rev``, ``owner`` and ``lease_until``. ``status`` and ``caller`` are columns for the
    host's own queries. The other four are text the router writes and reads, which a store only keeps.

    ``graph``, ``caller``, ``run``, ``state`` and ``outcome`` are JSON, and so is an event's body. A store may keep
    them in any form that reads back as the same JSON value, such as a Postgres ``jsonb`` column, which orders keys and
    spaces its own way: the router and conductor parse them, and never compare or hash the text itself.

    Attributes:
        id: The run's id.
        graph: The ``GraphRecord`` the run is of, as JSON; never changed.
        caller: The caller who started it, as JSON, so any worker can build the run's ``context`` again; never changed.
        run: ``run.to_json()``: the last checkpoint while the run is running, the whole run once it stops.
        status: ``running`` while a worker resumes the run or is to, else how the run was left. It is for the host's own
            queries: the router writes it in the same ``update`` as ``state``, from it, and reads only ``state``.
        state: The run's status and what was asked of it while it was running, as JSON: its status and why it
            ended early, the answers queued and refused, a cancel that was asked. Small however large the run is.
        outcome: What a reader gets without loading the run, as JSON: what waits for a person, how far the run is
            and, once the run is complete, its outputs. Written each time the run stops running, and its progress
            with each checkpoint.
        rev: Moved one on by every ``update``.
        owner: The worker resuming it.
        lease_until: Until when the owner's lease holds, as seconds since the epoch; a time already past for a run
            handed back for any worker to go on with; ``None`` when the run is not running.
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


class Outcome(BaseModel):
    """What a reader gets of a run without loading it, written each time the run stops running: a row's ``outcome``, as JSON.

    Attributes:
        waiting: What waits for a person.
        outputs: The graph's outputs by name once the run is complete; ``None`` until then.
        progress: ``run.progress``: how many calls of each node have an outcome, and how many there will be. Written
            also with each checkpoint, so it is as of the last checkpoint while the run is running. ``None`` for a
            run whose row could not be read as a run.
    """

    waiting: list[Waiting]
    outputs: dict[str, Any] | None
    progress: RunProgress | None

    @classmethod
    def empty(cls) -> Outcome:
        """The outcome of a run whose row could not be read as a run: nothing waits, no outputs, no progress."""
        return cls(waiting=[], outputs=None, progress=None)

    @classmethod
    def so_far(cls, run: Run[Any]) -> Outcome:
        """The outcome of a run that has not stopped, or ended without outputs: what waits in it and its progress.

        Args:
            run: The run.

        Returns:
            Its outcome, with no outputs.
        """
        return cls(waiting=run.waiting, outputs=None, progress=run.progress)

    @classmethod
    def of_stopped(cls, run: Run[Any]) -> Outcome:
        """The outcome of a run that stopped: what waits in it, its progress and, once it is complete, its outputs,
        each written by its type as the run writes it.

        Args:
            run: The run.

        Returns:
            Its outcome.

        Raises:
            InvalidCallError: An output can't be written as JSON (``run_unwritable``).
        """
        outputs = from_json(run.outputs_to_json()) if run.status == "complete" else None
        return cls(waiting=run.waiting, outputs=outputs, progress=run.progress)


class StoredState(BaseModel):
    """A run's status and what was asked of it while it was running: a row's ``state``, as JSON.

    Example: ``{"status": "running", "cause": null, "queued_answers": {"approve@1~3fa2c0d1": true}, "refused": {},
    "cancel_asked": false}`` for a running run that one answer was sent to.

    It is small however large the run is, so a route that queues an answer or asks a cancel reads and writes it
    without touching the row's ``run`` or ``outcome``.

    ``status`` and ``cause`` are an index of the row's run: ``stopped(run)`` writes them from the run each time a
    resume call stores the run it ended with, so ``GET /runs/{id}`` and the routes that refuse by status read them
    without loading a run that can be megabytes. Two values are the store's own, which no saved run can hold:
    ``running`` (``running()``: a worker resumes the run or is to, a fact of the lease, and a save never says
    ``running``) and ``failed`` with a cause of the router's (``failed(cause)``: a resume call raised outside
    conductor and returned no run, so the row keeps its last checkpoint).

    Attributes:
        status: ``running`` while a worker resumes the run or is to; else ``run.status`` as the run was stored, or
            ``failed`` for a resume call that raised.
        cause: ``run.cause`` as the run was stored, or why a resume call raised; ``None`` for a running run and for
            one that did not end early, which can't be resumed.
        queued_answers: Answers that came while the run was running, by waiting id; the run is given them when it
            stops.
        refused: Queued answers the run refused when it stopped, by waiting id, each with its problems.
        cancel_asked: Whether someone asked the running run to stop. It is cleared when the run stops for it. A
            cancel that came after the last node call is kept when a queued answer makes the run running again, and
            stops it then.
    """

    status: RunStatus
    cause: Problem | None
    queued_answers: dict[str, Any]
    refused: dict[str, list[Problem]]
    cancel_asked: bool

    @classmethod
    def started(cls) -> StoredState:
        """The state of a run that has just started: it is running, and nothing is queued, refused or asked."""
        return cls(status="running", cause=None, queued_answers={}, refused={}, cancel_asked=False)

    @property
    def is_running(self) -> bool:
        """Whether the run's status is ``running``: a worker resumes it, or it waits for a worker to take it."""
        return self.status == "running"

    def running(self) -> StoredState:
        """This state with the run running again, for a worker to resume: its status ``running``, and no cause."""
        return self.model_copy(update={"status": "running", "cause": None})

    def stopped(self, run: Run[Any]) -> StoredState:
        """This state once a resume call stored ``run``, the run it ended with: its status and cause are the run's.

        Args:
            run: The run as the row now stores it.

        Returns:
            The state with ``run.status`` and ``run.cause``.
        """
        return self.model_copy(update={"status": run.status, "cause": run.cause})

    def failed(self, cause: Problem) -> StoredState:
        """This state once a resume call raised outside conductor and returned no run: ``failed``, with ``cause``.

        Args:
            cause: Why the resume call failed, a problem of the router's or the one conductor refused it with.

        Returns:
            The state, ``failed`` with ``cause``.
        """
        return self.model_copy(update={"status": "failed", "cause": cause})

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

    def columns(self) -> RunChanges:
        """The row's ``status`` and ``state`` for this state.

        Nothing else writes either column, so each write of ``state`` writes the ``status`` it holds in the same
        statement. ``status`` is for the host's own queries: the worker and the routes read ``state`` and never
        ``status``.

        Returns:
            The two columns, to write in one ``update``.
        """
        return {"status": self.status, "state": self.model_dump_json()}


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
