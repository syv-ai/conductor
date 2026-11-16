"""``RunStore``: where the worker keeps runs and their events, written by the host, and the row it keeps.

A stored run is one row of text and numbers, ``StoredRun``, so any database holds it. Every
change is a compare-and-set on ``rev``: ``update`` writes only while the row is
still at the ``rev`` the writer read, and moves it one on. That one rule is
what lets several workers act on one run.

``GuardedStore`` wraps a host's store so that anything it raises comes out as
one exception, ``StoreFailedError``, which names the method. The worker and
``check_run_store`` call a store only through it.
"""

from __future__ import annotations

from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Protocol, TypedDict

from conductor import RunStatus


@dataclass(frozen=True)
class StoredRun:
    """One run as the store keeps it.

    A store acts on ``id``, ``rev``, ``owner`` and ``lease_until``. ``status`` and ``metadata`` are columns for the
    host's own queries. The other four are text the router writes and reads, which a store only keeps. ``id``,
    ``graph`` and ``metadata`` are written once, by ``create``, and no ``update`` changes them.

    ``graph``, ``metadata``, ``run``, ``state`` and ``outcome`` are JSON, and so is an event's body. A store may keep
    them in any form that reads back as the same JSON value, such as a Postgres ``jsonb`` column, which orders keys and
    spaces its own way: the router and conductor parse them, and never compare or hash the text itself.

    Attributes:
        id: The run's id.
        graph: The ``GraphRecord`` the run is of, as JSON.
        metadata: The run's metadata, as JSON: what the host keeps on the run, such as who started it. Any worker
            builds the run's registry and ``context`` again from it.
        run: ``run.to_json()``: the last checkpoint while the run is running, the whole run once it stops.
        status: ``running`` while a worker resumes the run or is to, else how the run was left. It is for the host's own
            queries: the router writes it in the same ``update`` as ``state``, from it, and reads only ``state``.
        state: The run's status and what was asked of it while it was running, as JSON: its status and why it
            ended early, the answers queued and refused, a cancel that was asked. Small however large the run is.
        outcome: What a reader gets without loading the run, as JSON: what waits for a person, how far the run is
            and, once the run is complete, its outputs. Written each time the run stops running, and its progress
            with each checkpoint.
        rev: Moved one on by every ``update``.
        owner: The worker resuming it. A run that stops with no resume call, such as a waiting run that a person
            declines, has an owner too while its ``run_end`` is stored: an id made for that one write, which stores
            the event and then clears the owner. ``None`` otherwise.
        lease_until: Until when the owner's lease holds, as seconds since the epoch; a time already past for a run
            handed back for any worker to go on with. A run that isn't running has one only while it has an owner, so
            a worker stores its ``run_end`` if that write gave up. ``None`` otherwise.
    """

    id: str
    graph: str
    metadata: str
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


class RunStore(Protocol):
    """What the runs router needs of a database. Each method is one statement, and none holds a lock across calls.

    A host never removes a run whose ``status`` is ``running``: its worker and its followers read it again and
    take it to be there. Removing runs that stopped running, such as old ones, is the host's own.

    A method raises only when the database fails: the router takes an exception for an outage and waits for the
    run's lease to run out. ``conductor_providers.fastapi.testing.check_run_store(store)`` checks a store against
    everything written here.
    """

    async def create(self, run: StoredRun) -> None:
        """Insert a new run.

        Args:
            run: The run, whole.
        """

    async def get(self, run_id: str) -> StoredRun | None:
        """Read a run.

        Args:
            run_id: The run's id.

        Returns:
            The run as last written, also by another worker a moment before; ``None`` when there is none by that id.
        """
        ...

    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        """Write some columns of a run and move its ``rev`` one on, only if the run is still at ``rev``.

        Args:
            run_id: The run's id.
            rev: The ``rev`` the writer read.
            changes: The columns to write.

        Returns:
            Whether it was written: the run was still at ``rev``.
        """
        ...

    async def expired(self, now: float) -> list[StoredRun]:
        """The runs whose worker died or handed them back.

        Args:
            now: The time, as seconds since the epoch.

        Returns:
            The runs whose ``lease_until`` is before ``now``, whatever their ``status``. A run that isn't running
            has a ``lease_until`` only while its ``run_end`` is stored, and a sweep then stores it if that write gave
            up.
        """
        ...

    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        """Append an event to a run, numbered one past the run's last (the first is ``0``), only while ``owner`` is the run's ``owner``.

        Only the owner is checked, never the ``status``: a run that stopped with no resume call has its ``run_end``
        added by its owner while its ``status`` already says how it stopped.

        Checking the owner and appending are one statement: a worker whose lease ran out may still add events while
        another worker takes the run over, and its events must not land among the new owner's.

        Args:
            run_id: The run's id.
            owner: The worker that appends it.
            body: The event, as JSON.

        Returns:
            Whether it was appended: ``owner`` is the run's ``owner``.
        """
        ...

    async def events(self, run_id: str, after: int) -> list[tuple[int, str]]:
        """Read a run's events.

        Args:
            run_id: The run's id.
            after: The number of the last event the reader has; ``-1`` for none.

        Returns:
            The events numbered above ``after``, in order, each as ``(number, body)``.
        """
        ...


class StoreFailedError(Exception):
    """A method of the host's store raised.

    The worker reads this as an outage: it gives the run up and lets its lease run out, and never fails the run.

    Attributes:
        method: The method that raised, such as ``"update"``.
        error: What it raised.
    """

    def __init__(self, method: str, error: Exception) -> None:
        super().__init__(f"the store's {method} raised {error!r}")
        self.method = method
        self.error = error


async def _guarded[T](method: str, answer: Awaitable[T]) -> T:
    """What one call of a store method gives, or ``StoreFailedError`` when it raises."""
    try:
        return await answer
    except Exception as error:
        raise StoreFailedError(method, error) from error


class GuardedStore:
    """A host's store, wrapped so that whatever one of its methods raises comes out as ``StoreFailedError``.

    It has the six methods of ``RunStore`` and passes each call on. Code that holds a ``GuardedStore`` can tell an
    outage apart from a fault in its own code by the exception's class alone.
    """

    def __init__(self, store: RunStore) -> None:
        self._store = store

    async def create(self, run: StoredRun) -> None:
        """``RunStore.create``."""
        await _guarded("create", self._store.create(run))

    async def get(self, run_id: str) -> StoredRun | None:
        """``RunStore.get``."""
        return await _guarded("get", self._store.get(run_id))

    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        """``RunStore.update``."""
        return await _guarded("update", self._store.update(run_id, rev, changes))

    async def expired(self, now: float) -> list[StoredRun]:
        """``RunStore.expired``."""
        return await _guarded("expired", self._store.expired(now))

    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        """``RunStore.add_event``."""
        return await _guarded("add_event", self._store.add_event(run_id, owner, body))

    async def events(self, run_id: str, after: int) -> list[tuple[int, str]]:
        """``RunStore.events``."""
        return await _guarded("events", self._store.events(run_id, after))
