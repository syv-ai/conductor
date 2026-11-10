"""``RunStore``: where the runs router keeps runs and their events, written by the host.

A stored run is one row of plain values (text, numbers, a flag), so any
database holds it. Every change is a compare-and-set on ``rev``: ``update``
writes only while the row is still at the ``rev`` the writer read, and moves it
one on. That one rule is what lets several workers act on one run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

from conductor import RunStatus


@dataclass(frozen=True)
class StoredRun:
    """One run as the store keeps it. The JSON fields are text the router writes and reads; a store only keeps them."""

    id: str
    #: The ``GraphRecord`` the run is of, as JSON.
    graph: str
    #: The caller who started it, as JSON, so any worker can build the run's ``from_run`` again.
    caller: str
    #: ``run.to_json()``: the last checkpoint while a leg runs, the whole run once it ends.
    run: str
    status: RunStatus
    #: Moved one on by every ``update``.
    rev: int = 0
    #: The rows waiting for a person as of the last leg's end, as JSON, so reading a run doesn't load it.
    waiting: str = "[]"
    #: The outputs as JSON once the run is complete, a table as its records.
    outputs: str | None = None
    #: The worker running a leg of it, and until when its lease holds.
    owner: str | None = None
    lease_until: float = 0.0
    #: Set when someone asked a running leg to stop.
    cancel: bool = False
    #: Answers that came while a leg ran, by waiting id, as JSON; applied when the leg ends.
    pending: str = "{}"
    #: Queued answers that were refused, by waiting id, as JSON lists of problems.
    refused: str = "{}"
    #: Why the last leg ended as it did (``failed``, ``stopped``, ``cancelled``, ``timed_out``), as a problem's JSON.
    cause: str | None = None


class RunChanges(TypedDict, total=False):
    """The fields one ``update`` changes."""

    run: str
    status: RunStatus
    waiting: str
    outputs: str | None
    owner: str | None
    lease_until: float
    cancel: bool
    pending: str
    refused: str
    cause: str | None


class RunStore(Protocol):
    """What the runs router needs of a database. Each method is one statement, and none holds a lock across calls."""

    async def create(self, run: StoredRun) -> None:
        """Insert a new run."""

    async def get(self, run_id: str) -> StoredRun | None:
        """The run, or ``None`` when there is none."""

    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        """Write ``changes`` and move ``rev`` one on, only if the run is still at ``rev``; whether it was."""
        ...

    async def expired(self, now: float) -> list[StoredRun]:
        """The runs whose status is ``running`` and whose ``lease_until`` is before ``now``: their worker died or handed them back."""
        ...

    async def add_event(self, run_id: str, owner: str, body: str) -> int | None:
        """Append an event, numbered one past the run's last, only while ``owner`` holds the run; its number, or ``None``."""

    async def events(self, run_id: str, after: int) -> list[tuple[int, str]]:
        """The run's events numbered above ``after``, in order, as ``(number, body)``."""
        ...
