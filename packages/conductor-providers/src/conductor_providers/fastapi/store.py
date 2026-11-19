"""``RunStore``: where the worker keeps runs and their events, written by the host, and the row it keeps.

A stored run is one row of text and numbers, ``StoredRun``, so any database holds it. Every change is a
compare-and-set on ``rev``: ``update`` writes only while the row is still at the ``rev`` the writer read, and moves
it one on. That one rule is what lets several workers act on one run. A writer that loses reads the row again, and
works its change out again from what it reads. ``GuardedStore.change`` is that loop, and every write the worker
tries again goes through it. An ``update`` can append one event in the same write, which is how a run's ``run_end``
lands with the row that says how the run stopped.

``GuardedStore`` is the worker's one door to a host's store. It wraps each row it reads in a ``RunRow``, which
reads the row's JSON into its types. It writes a row's changes back as text, and turns anything the store raises
into one exception, ``StoreFailedError``, which names the method. ``check_run_store`` calls a store with
``guarded`` alone, since it checks the text a store keeps.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Generic, Protocol, TypedDict

from conductor import InvalidCallError, RunStatus
from conductor.events import RunEnd
from pydantic import TypeAdapter, ValidationError
from typing_extensions import TypeVar

from conductor_providers.fastapi.problems import no_such_run
from conductor_providers.fastapi.state import RunOutcome, RunState

#: The type of a run's metadata: what the host keeps on it. ``dict[str, str]`` unless the host names a model.
M = TypeVar("M", default=dict[str, str])

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


@dataclass(frozen=True)
class RunRow(Generic[M]):
    """A run's row as the worker reads it: the ``StoredRun`` the host's store gave, with its JSON read into types.

    The worker's store wraps each row in one as it gets it, so nothing else in the worker parses a row's JSON. The
    ``state`` is read at once: it is small, and every action on a run looks at it first. The ``outcome`` can be
    megabytes, and only a reader's view and a resume call need it, so it is read the first time it is asked for.
    So is the ``metadata``, so that a row whose metadata the worker's type refuses can still be taken over, and its
    resume call then fails it.

    Attributes:
        stored: The row as the host's store keeps it.
        state: The row's ``state``, read: where the run is, and what was asked of it while it was running.
        metadata_type: Reads the row's ``metadata``: the worker's ``metadata_type``.
    """

    stored: StoredRun
    state: RunState
    metadata_type: TypeAdapter[M] = field(repr=False, compare=False)

    @cached_property
    def metadata(self) -> M:
        """The run's metadata, read by its type the first time it is asked for.

        Raises:
            ValidationError: The worker's ``metadata_type`` refuses the row's metadata.
        """
        return self.metadata_type.validate_json(self.stored.metadata)

    @cached_property
    def outcome(self) -> RunOutcome:
        """What a reader gets without loading the run, read the first time it is asked for.

        Raises:
            ValidationError: The row's outcome can't be read.
        """
        return RunOutcome.model_validate_json(self.stored.outcome)


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

    It has the six methods of ``RunStore``. ``get`` and ``expired`` wrap each row in a ``RunRow``. ``create`` and
    ``update`` take the columns they write as their types, write them as text, and give the row as written.
    ``change`` tries a write again until it lands. Code that holds a ``GuardedStore`` works on a row's metadata,
    state and outcome as their types, and can tell an outage apart from a fault in its own code by the exception's
    class alone.

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

    async def create(
        self,
        run_id: str,
        graph: str,
        metadata: M,
        holder: RunChanges,
        *,
        state: RunState,
        run: str,
        outcome: RunOutcome,
    ) -> RunRow[M]:
        """Insert a new run: ``RunStore.create`` with the row written as text.

        Args:
            run_id: The run's id.
            graph: The ``GraphRecord`` the run is of, as JSON.
            metadata: The run's metadata, written by alias.
            holder: The row's ``owner`` and ``lease_until``.
            state: The row's state, written as its ``state`` and ``status``.
            run: The row's ``run``, as JSON.
            outcome: The row's outcome.

        Returns:
            The row as written.
        """
        stored = StoredRun(
            id=run_id,
            graph=graph,
            metadata=self._metadata.dump_json(metadata, by_alias=True).decode(),
            rev=0,
            **await self._columns(holder, state=state, run=run, outcome=outcome),
        )
        await guarded("create", self._store.create(stored))
        return RunRow(stored, state, self._metadata)

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
        call_end: RunEnd | None = None,
    ) -> RunRow[M] | None:
        """``RunStore.update`` at the ``rev`` of ``row``: write some of its columns, and a ``run_end`` with them.

        Of two writers at one ``rev``, one wins. A write whose state stops the run appends a ``run_end`` that says
        what the state says, so a row that stops always carries a matching ``run_end``. A write that leaves the run
        running appends ``call_end``, the ``run_end`` of a resume call that ended with its run going on. The columns
        and the event are one ``update``, so both land or neither does.

        Args:
            row: The run's row as read.
            holder: The row's new ``owner`` or ``lease_until``, or both; ``None`` to leave them.
            state: The row's new state, written as its ``state`` and ``status``; ``None`` to leave both.
            run: The row's new ``run``, as JSON; ``None`` to leave it.
            outcome: The row's new outcome; ``None`` to leave it.
            call_end: The ``run_end`` of the resume call this write ends, appended when the run goes on; ``None``
                for none.

        Returns:
            The row as written; ``None`` when another writer moved its ``rev`` on first, and nothing was written.
        """
        columns = await self._columns(holder or {}, state=state, run=run, outcome=outcome)
        if state is not None and not state.is_running:
            call_end = RunEnd(status=state.status, cause=state.cause)
        event = None if call_end is None else call_end.model_dump_json()
        stored = row.stored
        if not await guarded("update", self._store.update(stored.id, stored.rev, columns, event)):
            return None
        written = replace(stored, **columns, rev=stored.rev + 1)
        return RunRow(written, row.state if state is None else state, self._metadata)

    async def change(
        self,
        row: RunRow[M],
        step: Callable[[RunRow[M]], Awaitable[RunRow[M] | None]],
        *,
        read_first: bool = False,
    ) -> RunRow[M]:
        """Change a run's row by ``step``, trying again each time another writer moved its ``rev`` on first.

        ``step`` is one try. It is given the row as read, works out what to write from it, and writes it with
        ``update``. When ``update`` loses, the row is read again and ``step`` is called on what was read, so a
        change made meanwhile, such as a cancel, is never written over.

        Args:
            row: The run's row as read.
            step: One try at the change. It returns the row as the try leaves it: as written, or as given when the
                try writes nothing. It returns ``None`` when another writer moved the ``rev`` on first.
            read_first: Whether the first try is given the row as the store has it now, read again, rather than
                ``row``.

        Returns:
            The row as the last try left it.

        Raises:
            InvalidCallError: The row is gone when it is read again (``no_such_run``).
        """
        read_again = read_first
        while True:
            if read_again:
                read = await self.get(row.stored.id)
                if read is None:
                    raise InvalidCallError([no_such_run()])
                row = read
            if (left := await step(row)) is not None:
                return left
            read_again = True

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
        return RunRow(stored, RunState.model_validate_json(stored.state), self._metadata)

    @staticmethod
    async def _columns(
        holder: RunChanges,
        *,
        state: RunState | None,
        run: str | None,
        outcome: RunOutcome | None,
    ) -> RunChanges:
        """The columns a write changes, as the store keeps them: the one place the worker writes a row's JSON.

        ``status`` is written with ``state``, from it. An outcome can be megabytes, so it is written on a thread.
        """
        columns: RunChanges = {}
        if state is not None:
            columns["status"], columns["state"] = state.status, state.model_dump_json()
        if run is not None:
            columns["run"] = run
        if outcome is not None:
            columns["outcome"] = await asyncio.to_thread(outcome.model_dump_json)
        columns.update(holder)
        return columns
