"""``RunStore``: where the runs router keeps runs and their events, written by the host.

A stored run is one row of text and numbers, so any database holds it. Every
change is a compare-and-set on ``rev``: ``update`` writes only while the row is
still at the ``rev`` the writer read, and moves it one on. That one rule is
what lets several workers act on one run.
"""

from __future__ import annotations

from typing import Protocol

from conductor_providers.fastapi.stored_run import RunChanges, StoredRun


class RunStore(Protocol):
    """What the runs router needs of a database. Each method is one statement, and none holds a lock across calls.

    A host never removes a run whose ``status`` is ``running``: its worker and its followers read it again and
    take it to be there. Removing runs that stopped running, such as old ones, is the host's own.
    """

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
