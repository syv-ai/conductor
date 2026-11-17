"""``RunStore``: where the worker keeps runs and their events, written by the host, and the row it keeps.

A stored run is one row of text and numbers, ``StoredRun``, so any database holds it. Every
change is a compare-and-set on ``rev``: ``update`` writes only while the row is
still at the ``rev`` the writer read, and moves it one on. That one rule is
what lets several workers act on one run. An ``update`` can append one event
in the same write, which is how a run's ``run_end`` lands with the row that
says how the run stopped.

``GuardedStore`` is the worker's one door to a host's store. It reads each row
into a ``RunRow``, writes a row's changes back as text, and turns anything the
store raises into one exception, ``StoreFailedError``, which names the method.
``check_run_store`` calls a store with ``guarded`` alone, since it checks the
text a store keeps.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable
from dataclasses import dataclass, replace
from typing import Any, Generic, Protocol, TypedDict

from conductor import RunStatus
from pydantic import TypeAdapter, ValidationError

from conductor_providers.fastapi.state import M, RunOutcome, RunRow, RunState

log = logging.getLogger("conductor_providers.fastapi")


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
        owner: The worker resuming it.
        lease_until: Until when the owner's lease holds, as seconds since the epoch; a time already past for a run
            handed back for any worker to go on with; ``None`` when the run is not running.
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

    async def update(
        self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
    ) -> bool:
        """Write some columns of a run and move its ``rev`` one on, only if the run is still at ``rev``.

        With ``event``, the write also appends that event to the run, numbered as ``add_event`` numbers one, in the
        same statement or transaction: both land, or neither does. The worker writes a run's ``run_end`` this way,
        with the row that says how the run stopped, so a reader never finds one without the other. The event is
        appended whoever the row's ``owner`` is, since the ``rev`` already decides which writer wins.

        Args:
            run_id: The run's id.
            rev: The ``rev`` the writer read.
            changes: The columns to write.
            event: An event to append with the write, as JSON; ``None`` for none.

        Returns:
            Whether it was written, and the event with it: the run was still at ``rev``.
        """
        ...

    async def expired(self, now: float) -> list[StoredRun]:
        """The runs whose worker died or handed them back.

        Args:
            now: The time, as seconds since the epoch.

        Returns:
            The runs whose ``lease_until`` is before ``now``. A run that is not running has no ``lease_until``, so
            it is never one of them.
        """
        ...

    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        """Append an event to a run, numbered one past the run's last (the first is ``0``), only while ``owner`` holds the run.

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


async def guarded[T](method: str, answer: Awaitable[T]) -> T:
    """What one call of a store method gives, or ``StoreFailedError`` when it raises.

    It is internal to this package: ``GuardedStore`` calls the host's store through it, and so does
    ``check_run_store``, which checks the text a store keeps rather than the rows the worker reads.

    Args:
        method: The method called, such as ``"update"``, which the error names.
        answer: The call, not yet awaited.
    """
    try:
        return await answer
    except Exception as error:
        raise StoreFailedError(method, error) from error


def refuse_an_update_without_an_event(store: RunStore) -> None:
    """Refuse a store whose ``update`` can't be given ``event`` as its fourth argument.

    Every call to it would raise, and the worker would read each one as an outage, so the store is refused once,
    before it is used.

    Raises:
        TypeError: The store's ``update`` can't be given ``event`` as its fourth argument.
    """
    try:
        inspect.signature(store.update).bind("run_id", 0, {}, None)
    except TypeError:
        raise TypeError(
            f"{type(store).__name__}.update can't be given an event as its fourth argument; add the parameter "
            "`event: str | None = None` after changes, not keyword-only, and "
            "when it is given, append it as add_event does, in the same statement or transaction as the write"
        ) from None


class GuardedStore(Generic[M]):
    """A host's store as the worker uses it: rows read into ``RunRow``, and outages raised as ``StoreFailedError``.

    It has the six methods of ``RunStore``. ``get`` and ``expired`` read each row into a ``RunRow``, and ``create``
    and ``update`` write a ``RunRow``'s columns back as text. Code that holds a ``GuardedStore`` works on a row's
    metadata, state and outcome as their types, and can tell an outage apart from a fault in its own code by the
    exception's class alone.

    Args:
        store: The host's store.
        metadata: Reads a run's metadata: the worker's ``metadata_type``.

    Raises:
        TypeError: The store's ``update`` can't be given ``event`` as its fourth argument.
    """

    def __init__(self, store: RunStore, metadata: TypeAdapter[M]) -> None:
        refuse_an_update_without_an_event(store)
        self._store = store
        self._metadata = metadata

    async def create(self, row: RunRow[M]) -> None:
        """Insert a new run: ``RunStore.create`` with the row written as text."""
        columns = self._columns(
            {"owner": row.owner, "lease_until": row.lease_until},
            state=row.state,
            run=row.run,
            outcome=None,
        )
        columns["outcome"] = row.outcome_json
        stored = StoredRun(
            id=row.id,
            graph=row.graph,
            metadata=row.metadata_json,
            rev=row.rev,
            **columns,
        )
        await guarded("create", self._store.create(stored))

    async def get(self, run_id: str) -> RunRow[M] | None:
        """``RunStore.get``, its row read.

        Raises:
            ValidationError: The row's state can't be read.
        """
        stored = await guarded("get", self._store.get(run_id))
        return None if stored is None else self._read(stored)

    async def update(
        self,
        row: RunRow[M],
        holder: RunChanges | None = None,
        *,
        state: RunState | None = None,
        run: str | None = None,
        outcome: RunOutcome | None = None,
        event: str | None = None,
    ) -> RunRow[M] | None:
        """``RunStore.update`` at the ``rev`` of ``row``: write some of its columns, and ``event`` with them.

        Of two writers at one ``rev``, one wins. The columns and the event are one ``update``, so both land or
        neither does: a run that stops with no resume call is written with the ``run_end`` that says so.

        Args:
            row: The run's row as read.
            holder: The row's new ``owner`` or ``lease_until``, or both; ``None`` to leave them.
            state: The row's new state, written as its ``state`` and ``status``; ``None`` to leave both.
            run: The row's new ``run``, as JSON; ``None`` to leave it.
            outcome: The row's new outcome; ``None`` to leave it.
            event: An event to append in the same write, as JSON; ``None`` for none.

        Returns:
            The row as written; ``None`` when another writer moved its ``rev`` on first, and nothing was written.
        """
        if outcome is None:
            columns = self._columns(holder or {}, state=state, run=run, outcome=None)
        else:  # an outcome can be megabytes, so it is written on a thread
            columns = await asyncio.to_thread(
                self._columns, holder or {}, state=state, run=run, outcome=outcome
            )
        if not await guarded("update", self._store.update(row.id, row.rev, columns, event)):
            return None
        written: dict[str, Any] = {**(holder or {}), "rev": row.rev + 1}
        if state is not None:
            written["state"] = state
        if run is not None:
            written["run"] = run
        if outcome is not None:
            written["outcome_json"] = columns["outcome"]
        return replace(row, **written)

    async def expired(self, now: float) -> list[RunRow[M]]:
        """``RunStore.expired``, each row read.

        A row whose state can't be read is logged and left out, so that it doesn't keep the other runs from being
        taken over. A row whose metadata the worker's type refuses is given, since its metadata is read only when
        it is first asked for: its resume call then fails the run.
        """
        rows = []
        for stored in await guarded("expired", self._store.expired(now)):
            try:
                rows.append(self._read(stored))
            except ValidationError:
                log.exception(
                    "run %s, whose lease ran out, was not taken over; the next sweep tries again",
                    stored.id,
                )
        return rows

    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        """``RunStore.add_event``."""
        return await guarded("add_event", self._store.add_event(run_id, owner, body))

    async def events(self, run_id: str, after: int) -> list[tuple[int, str]]:
        """``RunStore.events``."""
        return await guarded("events", self._store.events(run_id, after))

    def _read(self, stored: StoredRun) -> RunRow[M]:
        """A row with its state read; its metadata and outcome are read when they are first asked for."""
        return RunRow(
            id=stored.id,
            graph=stored.graph,
            metadata_json=stored.metadata,
            metadata_type=self._metadata,
            run=stored.run,
            state=RunState.model_validate_json(stored.state),
            outcome_json=stored.outcome,
            rev=stored.rev,
            owner=stored.owner,
            lease_until=stored.lease_until,
        )

    @staticmethod
    def _columns(
        holder: RunChanges,
        *,
        state: RunState | None,
        run: str | None,
        outcome: RunOutcome | None,
    ) -> RunChanges:
        """The columns a write changes, as the store keeps them: the one place the worker writes a row's state.

        ``status`` is written with ``state``, from it.
        """
        columns: RunChanges = {}
        if state is not None:
            columns["status"], columns["state"] = state.status, state.model_dump_json()
        if run is not None:
            columns["run"] = run
        if outcome is not None:
            columns["outcome"] = outcome.model_dump_json()
        columns.update(holder)
        return columns
