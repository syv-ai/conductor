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
from asyncio import Event as AsyncioEvent
from asyncio import Lock
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from threading import Event
from typing import Any

from conductor import Graph, InvalidCallError, NodeRegistry, Problem, Run
from conductor.events import RunCheckpoint, RunEnd

from conductor_providers.fastapi.problems import raised_outside_a_node
from conductor_providers.fastapi.store import GuardedStore, RunRow, StoreFailedError

log = logging.getLogger("conductor_providers.fastapi")


@dataclass(frozen=True)
class ResumeSettings:
    """What every resume call of one worker is made with.

    The worker loads every stored run through ``load``, for a resume call and for any other call that needs the run
    itself, so a run is compiled and read in one place.

    Attributes:
        nodes_for: The registry for a run's metadata.
        context: The run's context, from its metadata and the resume call's stop event; ``None`` for no context.
        checkpoint_every: The seconds between saves of a running run.
        resume_timeout: The seconds one resume call may take; ``None`` for no limit.
        handing_back: Set when the worker shuts down, so every resume call hands its run back.
    """

    nodes_for: Callable[[Any], NodeRegistry[Any]]
    context: Callable[[Any, Event], Any] | None
    checkpoint_every: float
    resume_timeout: float | None
    handing_back: AsyncioEvent

    async def load(self, row: RunRow[Any], metadata: Any) -> Run[Any]:
        """The run ``row`` keeps, loaded on a thread on the registry for ``metadata``.

        A resume call, an answer and a decline load a row on its own metadata. ``again`` and ``create`` with
        ``previous`` load it on the metadata of the run they start.

        Raises:
            InvalidCallError: The row's graph doesn't compile on that registry, or its saved run can't be read.
        """
        nodes = self.nodes_for(metadata)
        return await asyncio.to_thread(
            lambda: Run.from_json(row.stored.run, Graph.compile(row.stored.graph, nodes))
        )

    def context_for(self, metadata: Any, stop: Event) -> Any:
        """The context of a resume call: ``context(metadata, stop)``, or ``None`` when the worker has none."""
        return None if self.context is None else self.context(metadata, stop)


class HeldRun:
    """One worker's lease on one run, while the worker resumes it.

    The worker starts ``run()`` as a task, and calls ``cancel()`` when it has written a cancel into the run's row.
    Nothing else of it is used from outside.

    Args:
        store: The worker's store, guarded.
        row: The run's row, as the worker just wrote it to take the run.
        settings: The worker's settings for resume calls.

    Attributes:
        run_id: The run's id.
    """

    def __init__(
        self, store: GuardedStore[Any], row: RunRow[Any], settings: ResumeSettings
    ) -> None:
        self.run_id = row.stored.id
        self._store = store
        self._settings = settings
        # the row as last read or written
        self._row = row
        # set to end the resume call before its next node call: a cancel was asked, another worker took the run, or
        # the store failed. The same event goes to the host's ``context``, so a node can stop a slow call too.
        self._stop = Event()
        if row.state.cancel_asked:
            self._stop.set()
        # held for each write to the row, so this worker's writes land one at a time
        self._lock = Lock()
        # whether the run was let go or taken by another worker; nothing more is written then
        self._released = False

    async def run(self) -> None:
        """Make one resume call after another, renewing the lease, until the run is let go or taken.

        A ``StoreFailedError`` is the host's outage and never fails the run. It is logged, the stop event is set so
        that a node call still going on can stop, and the run is left as the store has it, ``running``. Once its
        lease runs out, a worker resumes it from the last checkpoint; a node call still on a thread may then be made
        again while it runs.

        Anything else that raises here is a fault the run can't be failed with, such as a row whose JSON the worker
        can't read. It is logged with the run's id, and the run is left as the store has it.
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
        except Exception:
            self._stop.set()
            log.exception(
                "the worker stopped resuming run %s, and left it as the store has it; read its row to see why",
                self.run_id,
            )
        finally:
            renewing.cancel()

    def cancel(self) -> None:
        """End the resume call before its next node call: the worker has just written a cancel into the run's row.

        It takes no lock, so a cancel is answered at once, even while the end of a resume call is being written.
        The row itself is taken in by the next write, whose compare-and-set finds the cancel there.
        """
        self._stop.set()

    async def _resume_call(
        self,
    ) -> tuple[tuple[RunEnd, str] | None, tuple[Problem, Run[Any] | None] | None]:
        """Make one resume call and store its events and checkpoints.

        Returns:
            What ``_finish`` takes, ``(ended, failed)``. A call that ended gives ``ended``: its own ``run_end``,
            which carries the run it returned, and that run as JSON. A call that failed gives ``failed``: why, and
            the run as the row has it (``None`` when it couldn't be loaded). A run handed back, as the worker shuts
            down, gives neither.

        Raises:
            StoreFailedError: The store failed. Anything else that raises fails the run: conductor's refusal with
                its first problem, and anything else with ``raised_outside_a_node``.
        """
        settings, row = self._settings, self._row
        # the run as it last stopped: a checkpoint changes only its progress
        stopped = row.outcome
        # the run as the row has it, which a failed run lists and is given its queue; nothing, until it is loaded
        last: Run[Any] | None = None
        try:
            run = await settings.load(row, row.metadata)
            last = run
            stream = run.stream(
                context=settings.context_for(row.metadata, self._stop),
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
                        checkpoint = event.run
                        saved, progress = await asyncio.to_thread(
                            lambda: (checkpoint.to_json(), checkpoint.progress)
                        )
                        last = checkpoint
                        progressed = stopped.model_copy(update={"progress": progress})
                        await self._write(
                            lambda row: self._store.update(row, run=saved, outcome=progressed)
                        )
                    elif isinstance(event, RunEnd):
                        # written here, so a run that can't be written fails this call (run_unwritable)
                        saved = await asyncio.to_thread(event.run.to_json)
                        return (event, saved), None
                    elif not await self._store.add_event(self.run_id, event.model_dump_json()):
                        self._stop.set()  # another worker holds the run
            return None, None
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
        return None, (cause, last)

    async def _finish(
        self,
        ended: tuple[RunEnd, str] | None,
        failed: tuple[Problem, Run[Any] | None] | None,
    ) -> None:
        """Write how a resume call ended.

        What is written is worked out from the row's state on each try of the write, so a cancel or an answer
        written meanwhile is never written over. It is worked out under the lock, so no renewal lands between
        working it out and writing it. It runs on a thread, so other runs on this worker go on meanwhile. The call's
        ``run_end`` goes to ``GuardedStore.update``, which writes it with the row in one ``update``, so it lands
        once, with the write that lands. Once the run stops, the store writes the state's own instead: a decline
        queued meanwhile ends the run ``declined``, though the call's own ``run_end`` said ``waiting``. A run another
        worker took is left as it is. A run that is to go on as the worker shuts down is handed back for another
        worker.

        Args:
            ended: For a call that ended, its own ``run_end`` and the run it returned as JSON, which
                ``RunState.ended_with`` takes; ``None`` for a call that failed or a run handed back.
            failed: For a call that failed, why and the run as the row has it, which ``RunState.failed`` takes;
                ``None`` for a call that didn't.
        """
        handing_back = self._settings.handing_back

        async def end(row: RunRow[Any]) -> RunRow[Any] | None:
            state, run, outcome, call_end = row.state, None, None, None
            if failed is not None:
                state, run, outcome = await asyncio.to_thread(state.failed, *failed)
            elif ended is not None:
                call_end, saved = ended
                state, run, outcome = await asyncio.to_thread(state.ended_with, call_end.run, saved)
            goes_on = state.is_running and not handing_back.is_set()
            written = await self._store.update(
                row, holds=goes_on, state=state, run=run, outcome=outcome, call_end=call_end
            )
            if written is not None:
                self._released = not goes_on
                if not state.cancel_asked:
                    # the cancel that stopped this resume call doesn't stop the next; a call of this one still on
                    # its thread past its attempt_timeout goes on seeing it set
                    self._stop = Event()
            return written

        # read first: this worker's own calls may have queued an answer, and the end is the largest write
        await self._write(end, read_first=True)

    async def _renew(self) -> None:
        """Renew the lease three times a lease, until the run is let go or taken.

        A renewal the store fails is logged, and the next one is tried, so one failed statement doesn't cost the
        run. A write of the lease the store hasn't answered within a third of a lease is given up, and the next
        renewal is made at once. Only the store's answer is timed, not the wait for another write of this worker,
        such as the end of a resume call. A renewal doesn't hold up the other writes, so a worker whose run was taken
        meanwhile learns it at its next checkpoint or event.
        """
        lease = self._store.lease

        async def renewed(row: RunRow[Any]) -> RunRow[Any] | None:
            async with asyncio.timeout(lease / 3):
                return await self._store.update(row, holds=True)

        wait = lease / 3
        while not self._released:
            await asyncio.sleep(wait)
            wait = lease / 3
            try:
                await self._write(renewed)
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

    async def _write(
        self,
        write: Callable[[RunRow[Any]], Awaitable[RunRow[Any] | None]],
        *,
        read_first: bool = False,
    ) -> None:
        """Write to the run's row by ``write``, under the lock, as ``GuardedStore.change`` writes: again on the row
        read again, each time another writer moved its ``rev`` on first.

        Each try first takes in the row it is given, so a cancel asked meanwhile stops the resume call, and a run
        that was let go or taken by another worker is left as it is.

        Args:
            write: One try, as ``GuardedStore.change`` takes it: it writes with ``GuardedStore.update`` and returns
                what that returns.
            read_first: Whether the first try is given the row as the store has it now, read under the same hold of
                the lock, rather than the row as this worker last read or wrote it (``GuardedStore.change``).

        Raises:
            InvalidCallError: The row is gone from the store (``no_such_run``), as ``GuardedStore.change`` says.
        """

        async def one_try(row: RunRow[Any]) -> RunRow[Any] | None:
            self._take_in(row)
            return row if self._released else await write(row)

        async with self._lock:
            self._row = await self._store.change(self._row, one_try, read_first=read_first)

    def _take_in(self, row: RunRow[Any]) -> None:
        """Take in the row as read: what another writer changed in its state, a cancel asked of it, and whether
        another worker took it."""
        self._row = row
        self._released = not self._store.owns(row)
        if self._released or row.state.cancel_asked:
            self._stop.set()
