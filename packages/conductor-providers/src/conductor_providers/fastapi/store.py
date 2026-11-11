"""``RunStore``: where the runs router keeps runs and their events, written by the host.

A stored run is one row of text and numbers, so any database holds it. Every
change is a compare-and-set on ``rev``: ``update`` writes only while the row is
still at the ``rev`` the writer read, and moves it one on. That one rule is
what lets several workers act on one run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypedDict

from conductor import RunStatus


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
    rev: int = 0
    owner: str | None = None
    lease_until: float | None = None


class RunChanges(TypedDict, total=False):
    """The columns one ``update`` changes: any of those of a ``StoredRun`` that change after it is created."""

    run: str
    status: RunStatus
    state: str
    outcome: str
    owner: str | None
    lease_until: float | None


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
        """The runs whose ``lease_until`` is before ``now``: their worker died or handed them back. A run that wants no
        leg has no ``lease_until``, so it is never one of them."""
        ...

    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        """Append an event, numbered one past the run's last, only while ``owner`` holds the run; whether it was appended."""
        ...

    async def events(self, run_id: str, after: int) -> list[tuple[int, str]]:
        """The run's events numbered above ``after``, in order, as ``(number, body)``."""
        ...
