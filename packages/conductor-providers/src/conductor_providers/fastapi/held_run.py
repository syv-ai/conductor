"""``HeldRun``: one worker's lease on one run, while the worker resumes it.

When a worker takes a run, it starts ``HeldRun.run()`` as a task. That makes one resume call after another: it loads
the run from its row, calls ``run.stream()``, and stores the events and a checkpoint every few seconds. Meanwhile it
renews the lease, so no other worker takes the run. When a resume call ends, it writes how it ended, and lets the run
go unless an answer queued meanwhile means it should go on at once.

A held run only ever gets a ``GuardedStore``, so anything the host's store raises comes out as
``StoreFailedError``. That is an outage, never the run's fault: the worker gives the run up and lets the lease run
out, and another worker goes on from the last checkpoint.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from asyncio import Event as AsyncioEvent
from asyncio import Lock
from collections.abc import Callable
from dataclasses import dataclass, replace
from threading import Event
from typing import Any

from conductor import Graph, InvalidCallError, NodeRegistry, Problem, Run
from conductor.events import RunCheckpoint, RunEnd
from pydantic import TypeAdapter

from conductor_providers.fastapi.problems import raised_outside_a_node
from conductor_providers.fastapi.state import RunOutcome, RunState
from conductor_providers.fastapi.store import GuardedStore, RunChanges, StoredRun, StoreFailedError

log = logging.getLogger("conductor_providers.fastapi")

#: The owner and lease of a row whose run isn't running.
_UNHELD: RunChanges = {"owner": None, "lease_until": None}

#: The owner and lease of a row handed back: its lease has already run out, so the next sweep of any worker takes it.
_HANDED_BACK: RunChanges = {"owner": None, "lease_until": 0.0}


@dataclass(frozen=True)
class ResumeSettings:
    """What every resume call of one worker is made with.

    Attributes:
        worker_id: The worker's id, written as the ``owner`` of the runs it holds.
        nodes_for: The registry for a run's metadata.
        context: The run's context, from its metadata and the resume call's stop event; ``None`` for no context.
        metadata: Reads a run's metadata from its JSON.
        lease: The seconds the worker holds a run between renewals.
        checkpoint_every: The seconds between saves of a running run.
        resume_timeout: The seconds one resume call may take; ``None`` for no limit.
        handing_back: Set when the worker shuts down, so every resume call hands its run back.
    """

    worker_id: str
    nodes_for: Callable[[Any], NodeRegistry[Any]]
    context: Callable[[Any, Event], Any] | None
    metadata: TypeAdapter[Any]
    lease: float
    checkpoint_every: float
    resume_timeout: float | None
    handing_back: AsyncioEvent

    def holder(self) -> RunChanges:
        """The ``owner`` and ``lease_until`` of a row this worker holds, from now."""
        return {"owner": self.worker_id, "lease_until": time.time() + self.lease}

    def context_for(self, metadata: Any, stop: Event) -> Any:
        """The context of a resume call: ``context(metadata, stop)``, or ``None`` when the worker has none."""
        return None if self.context is None else self.context(metadata, stop)


def loaded(row: StoredRun, nodes: NodeRegistry[Any]) -> Run[Any]:
    """The run a row keeps, loaded on ``nodes``.

    Raises:
        InvalidCallError: The graph doesn't compile on ``nodes``, or the saved run can't be read.
    """
    return Run.from_json(row.run, Graph.compile(row.graph, nodes))


class HeldRun:
    """One worker's lease on one run, while the worker resumes it.

    The worker starts ``run()`` as a task, and calls ``heard_of_a_change()`` when it has written a cancel into the
    run's row. Nothing else of it is used from outside.

    Args:
        store: The worker's store, guarded.
        row: The run's row, as the worker just wrote it to take the run.
        settings: The worker's settings for resume calls.

    Attributes:
        run_id: The run's id.
    """

    def __init__(self, store: GuardedStore, row: StoredRun, settings: ResumeSettings) -> None:
        self.run_id = row.id
        self._store = store
        self._settings = settings
        # the row as the next resume call starts from it, and its rev and state as last read or written
        self._row = row
        self._rev = row.rev
        self._state = RunState.model_validate_json(row.state)
        # set to end the resume call before its next node call: a cancel was asked, another worker took the run, or
        # the store failed. The same event goes to the host's ``context``, so a node can stop a slow call too.
        self._stop = Event()
        if self._state.cancel_asked:
            self._stop.set()
        # held for each write to the row, so this worker's writes land one at a time
        self._lock = Lock()
        # whether the run was let go or taken by another worker; nothing more is written then
        self._released = False
        # how many times the row was read again, each time taking in what another writer changed
        self._reads = 0

    async def run(self) -> None:
        """Make one resume call after another, renewing the lease, until the run is let go or taken.

        A ``StoreFailedError`` is the host's outage and never fails the run. It is logged, the stop event is set so
        that a node call still going on can stop, and the run is left as the store has it, ``running``. Once its
        lease runs out, a worker resumes it from the last checkpoint; a node call still on a thread may then be made
        again while it runs.
        """
        renewing = asyncio.create_task(self._renew())
        try:
            while not self._released:
                await self._finish(*await self._resume_call())
        except StoreFailedError:
            self._stop.set()  # another worker makes the calls going on again
            log.exception(
                "the store raised while run %s was resumed; a worker resumes the run once its lease runs out",
                self.run_id,
            )
        finally:
            renewing.cancel()

    async def heard_of_a_change(self) -> None:
        """Read the row again, and stop the resume call if a cancel was asked or another worker took the run."""
        async with self._lock:
            await self._read_again()

    async def _resume_call(
        self,
    ) -> tuple[RunEnd | None, Callable[[RunState], tuple[RunState, RunChanges]]]:
        """Make one resume call and store its events and checkpoints.

        Returns:
            The call's own ``run_end``, and how the row's state changes with it, as a function of the state as it is
            when it is written (one of ``RunState.ended_with`` and ``RunState.failed``). A run handed back, as the
            worker shuts down, has no ``run_end`` and keeps its state.

        Raises:
            StoreFailedError: The store failed. Anything else that raises fails the run: conductor's refusal with
                its first problem, and anything else with ``raised_outside_a_node``.
        """
        settings = self._settings
        # the run as it last stopped: a checkpoint changes only its progress
        stopped = RunOutcome.model_validate_json(self._row.outcome)
        # the run as the row has it, which a failed run lists; nothing, until the run is loaded
        held = RunOutcome(waiting=[], outputs=None, progress=None)
        try:
            metadata = settings.metadata.validate_json(self._row.metadata)
            run = await asyncio.to_thread(loaded, self._row, settings.nodes_for(metadata))
            held = await asyncio.to_thread(RunOutcome.of, run)
            stream = run.stream(
                context=settings.context_for(metadata, self._stop),
                timeout=settings.resume_timeout,
                cancel=self._stop,
                hand_back=settings.handing_back,
                checkpoint_every=settings.checkpoint_every,
            )
            async with contextlib.aclosing(
                stream
            ):  # leaving it ends the call before its next node call
                async for event in stream:
                    if isinstance(event, RunCheckpoint):
                        saved = await asyncio.to_thread(event.run.to_json)
                        held = await asyncio.to_thread(RunOutcome.of, event.run)
                        outcome = stopped.model_copy(update={"progress": held.progress})
                        await self._write({"run": saved, "outcome": outcome.model_dump_json()})
                    elif isinstance(event, RunEnd):
                        ended = event.run
                        saved, written = await asyncio.to_thread(
                            lambda: (ended.to_json(), RunOutcome.of(ended).model_dump_json())
                        )
                        return event, lambda state: state.ended_with(ended, saved, written)[:2]
                    elif not await self._store.add_event(
                        self.run_id, settings.worker_id, event.model_dump_json()
                    ):
                        self._stop.set()  # another worker holds the run
            return None, lambda state: (state, {})
        except StoreFailedError:
            raise
        except InvalidCallError as refusal:
            every = "; ".join(
                f"{problem.type} at {'.'.join(map(str, problem.loc))}: {problem.msg}"
                for problem in refusal.problems
            )
            log.error("resuming run %s was refused: %s", self.run_id, every, exc_info=True)
            cause: Problem = refusal.problems[0]  # a cause is one problem; the log has them all
        except Exception as error:
            log.exception("resuming run %s failed", self.run_id)
            cause = raised_outside_a_node(error=type(error).__name__)
        outcome_json = held.model_dump_json()
        return RunEnd(status="failed", cause=cause), lambda state: state.failed(cause, outcome_json)

    async def _finish(
        self, run_end: RunEnd | None, after: Callable[[RunState], tuple[RunState, RunChanges]]
    ) -> None:
        """Write how a resume call ended.

        What is written is worked out again from the row's state as it is on each try of the write, so a cancel or
        an answer written meanwhile is never written over. The ``run_end`` is written with the row, in one
        ``update``, so it lands once, with the write that lands, and a follower that reads that the run stopped has
        the event to read. When the run stops, it says what the row says: a decline queued meanwhile fails the run,
        though the call's own ``run_end`` said ``waiting``. When the run goes on, it is the call's own. A run another
        worker took is left as it is. A run that is to go on as the worker shuts down is handed back for another
        worker.

        Args:
            run_end: The call's own ``run_end``; ``None`` for a run handed back.
            after: The row's new state and the columns that change with it, from the state as it is.
        """
        settings = self._settings
        async with (
            self._lock
        ):  # so no renewal moves the rev between this read and what is worked out from it
            await self._read_again()
        while not self._released:
            reads = self._reads
            state, changes = await asyncio.to_thread(after, self._state)
            ended = None
            if run_end is not None:
                ended = (
                    run_end if state.is_running else RunEnd(status=state.status, cause=state.cause)
                ).model_dump_json()
            goes_on = state.is_running and not settings.handing_back.is_set()
            if goes_on:
                holder = settings.holder()
            elif state.is_running:
                holder = _HANDED_BACK
            else:
                holder = _UNHELD
            written: RunChanges = {**state.columns(), **changes, **holder}
            async with self._lock:  # no renewal lands after the write that lets the run go
                if self._reads == reads and await self._write_once(written, ended):
                    self._state, self._released = state, not goes_on
                    self._row = replace(self._row, **changes)
                    if not state.cancel_asked:
                        # the cancel that stopped this resume call doesn't stop the next; a call of this one still
                        # on its thread past its attempt_timeout goes on seeing it set
                        self._stop = Event()
                    return

    async def _renew(self) -> None:
        """Renew the lease three times a lease, until the run is let go or taken.

        A renewal the store fails is logged, and the next one is tried, so one failed statement doesn't cost the
        run. A renewal the store hasn't answered within a third of a lease is given up, and the next is made at
        once. It doesn't hold up the other writes, so a worker whose run was taken meanwhile learns it at its next
        checkpoint or event.
        """
        lease = self._settings.lease
        wait = lease / 3
        while not self._released:
            await asyncio.sleep(wait)
            wait = lease / 3
            try:
                async with asyncio.timeout(lease / 3):
                    await self._write({"lease_until": time.time() + lease})
            except TimeoutError:
                log.error(
                    "the store did not answer a renewal of the lease of run %s; the next is made at once",
                    self.run_id,
                )
                wait = 0.0
            except StoreFailedError:
                log.exception(
                    "the lease of run %s was not renewed; the next renewal tries again", self.run_id
                )

    async def _write(self, changes: RunChanges) -> bool:
        """Write a checkpoint or a renewal, again for as long as another writer (an answer queued, a cancel) moved
        the ``rev`` on first.

        Returns:
            Whether it was written. A run let go, or taken by another worker, takes no write.
        """
        async with self._lock:
            while not self._released:
                if await self._write_once(changes):
                    return True
            return False

    async def _write_once(self, changes: RunChanges, event: str | None = None) -> bool:
        """Write ``changes``, and ``event`` with them, at the ``rev`` this worker knows, under the lock; a write that
        loses reads the row again.

        Returns:
            Whether it was written.
        """
        if await self._store.update(self.run_id, self._rev, changes, event):
            self._rev += 1
            return True
        await self._read_again()
        return False

    async def _read_again(self) -> None:
        """Read the row, under the lock, and take in what another writer changed: its state, a cancel asked of it,
        and whether another worker took it."""
        row = await self._store.get(self.run_id)
        if row is None:
            raise RuntimeError(f"run {self.run_id} is gone from the store while it is running")
        self._reads += 1
        self._rev, self._state = row.rev, RunState.model_validate_json(row.state)
        self._released = row.owner != self._settings.worker_id
        if self._released or self._state.cancel_asked:
            self._stop.set()
