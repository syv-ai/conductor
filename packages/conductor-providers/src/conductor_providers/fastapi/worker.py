"""``Worker``: resumes stored runs for one process, on any number of processes sharing one store.

To resume a run, a worker loads it from its row and makes one resume call:
``run.stream()``, which is ``run.resume()`` with its events. It calls every
node that is ready, until the run is complete, waiting, failed or stopped.
The row's status is ``running`` from when a resume call is wanted until it
has ended and its end is written.

A worker makes each resume call in the background, holds a lease on the run
meanwhile, and writes a checkpoint every few seconds. Another worker takes
over a run whose lease ran out, from its last checkpoint, and a worker that
shuts down hands its runs back. Every write is a compare-and-set on the run's
``rev``, so of two workers acting on one run, one goes on and the other is
told so.

Nothing here knows HTTP: ``run_router`` makes one worker and its routes call
it.
"""

import asyncio
import contextlib
import logging
import time
import uuid
from asyncio import Event, Lock, Task
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Literal, TypeVar

from conductor import Graph, InvalidCallError, NodeRegistry, Run, RunProgress
from conductor.events import RunCheckpoint, RunEnd
from pydantic import TypeAdapter
from pydantic_core import from_json

from conductor_providers.fastapi import router_problems
from conductor_providers.fastapi.resume_end import HandedBack, Raised, ResumeEnd, end_of
from conductor_providers.fastapi.router_problems import NoSuchRunError, RefusedError
from conductor_providers.fastapi.store import RunStore
from conductor_providers.fastapi.stored_run import (
    Outcome,
    RunAsRead,
    RunChanges,
    StoredRun,
    StoredState,
)

C = TypeVar("C")

log = logging.getLogger("conductor_providers.fastapi")

#: The owner and lease of a row whose run is not running.
_UNHELD: RunChanges = {"owner": None, "lease_until": None}

#: The owner and lease of a row handed back: its lease already run out, so the next sweep of any worker takes it.
_HANDED_BACK: RunChanges = {"owner": None, "lease_until": 0.0}


@dataclass(eq=False)
class _Hold:
    """A run this worker holds while it resumes it.

    Attributes:
        run_id: The run's id.
        rev: The row's ``rev`` as this worker last wrote or read it.
        state: The row's state at that ``rev``.
        stop: Set to end the resume call before its next node call, from the row as this worker reads it: a cancel
            is asked in its state, or another worker holds it. Cleared when the end of a resume call is written and
            no cancel is asked any more.
        lock: Held for each of this worker's writes to the row, so they land one at a time.
        released: Whether the row was given up or taken by another worker; this worker writes nothing more to it.
        reads: How many times this worker read the row again, each time taking in what another writer changed.
    """

    run_id: str
    rev: int
    state: StoredState
    stop: Event = field(default_factory=Event)
    lock: Lock = field(default_factory=Lock)
    released: bool = False
    reads: int = 0


class Worker(Generic[C]):
    """Resumes stored runs for one process, holding a lease on each run while it does.

    Each method that takes a run takes it as the caller read it (``read``). A method that makes a run running
    (``answer`` on a run that is not running, ``resume``) writes only if the run is still at that ``rev``. What is
    asked of a running run (an answer to queue, a cancel) is written however the ``rev`` moved, for as long as the
    run is running.
    """

    def __init__(
        self,
        *,
        store: RunStore,
        nodes_for: Callable[[C], NodeRegistry[Any]],
        callers: TypeAdapter[C],
        context: Callable[[C], Any],
        max_items: int | None,
        lease: float,
        checkpoint_every: float,
        resume_timeout: float | None,
        poll: float,
    ) -> None:
        """A worker on ``store``.

        Args:
            store: Where runs and their events are kept.
            nodes_for: The nodes a caller's graphs may use; called for every method call and every resume call.
            callers: Reads and writes a caller as the JSON a run keeps.
            context: The run's context a resume call is given, from the caller who started the run.
            max_items: The ceiling on the items of any one Series that a run starts with.
            lease: The seconds this worker holds a run between renewals.
            checkpoint_every: The seconds between saves of a running run.
            resume_timeout: The seconds one call of ``resume()`` may take before the run ends ``timed_out``; ``None`` for no limit.
            poll: The seconds between a follower's reads of new events.
        """
        self.id = uuid.uuid4().hex
        self._store = store
        self._nodes_for = nodes_for
        self._callers = callers
        self._context = context
        self._max_items = max_items
        self._lease = lease
        self._checkpoint_every = checkpoint_every
        self._resume_timeout = resume_timeout
        self._poll = poll
        self._holds: dict[_Hold, Task[None]] = {}
        self._handing_back = Event()

    # -- what a route calls -----------------------------------------------------------------------

    def started_by(self, row: StoredRun) -> C:
        """The caller who started a run.

        Args:
            row: The run's row.

        Returns:
            The caller, read from the JSON the row keeps.
        """
        return self._callers.validate_json(row.caller)

    async def read(self, run_id: str) -> RunAsRead:
        """The run as the store has it.

        Args:
            run_id: The run's id.

        Returns:
            The row, with its state read.

        Raises:
            NoSuchRunError: The store has no run by that id.
        """
        row = await self._store.get(run_id)
        if row is None:
            raise NoSuchRunError
        return RunAsRead.of(row)

    async def begin(self, graph: Any, inputs: Mapping[str, Any], who: C) -> str:
        """Start a run: store it and make its first resume call in the background.

        Args:
            graph: The graph record, as a ``GraphRecord`` or as the plain data a caller sent.
            inputs: Values for the graph's inputs, by name.
            who: The caller who starts it, on whose nodes the graph is loaded.

        Returns:
            The run's id.

        Raises:
            InvalidCallError: The graph can't run on ``who``'s nodes, or refuses the inputs.
        """
        nodes = self._nodes_for(who)
        run = await asyncio.to_thread(lambda: Graph.compile(graph, nodes).with_inputs(inputs, max_items=self._max_items))
        return await self._create(run.graph.record.model_dump_json(), who, run)

    async def estimate(self, graph: Any, inputs: Mapping[str, Any], who: C) -> RunProgress:
        """How many calls a run of ``graph`` on ``inputs`` will make, from ``run.estimate()``, which calls only the
        nodes that cost nothing; nothing is stored.

        The call is made as a stored run's is: with the caller's context and the router's ``resume_timeout`` and
        ``max_items``.

        Args:
            graph: The graph record, as a ``GraphRecord`` or as the plain data a caller sent.
            inputs: Values for the graph's inputs, by name.
            who: The caller, on whose nodes the graph is loaded.

        Returns:
            The run's progress after that call.

        Raises:
            InvalidCallError: The graph can't run on ``who``'s nodes, or refuses the inputs.
        """
        nodes = self._nodes_for(who)

        def estimated() -> RunProgress:
            ready = Graph.compile(graph, nodes).with_inputs(inputs, max_items=self._max_items)
            return ready.estimate(context=self._context(who), timeout=self._resume_timeout).progress

        return await asyncio.to_thread(estimated)

    async def again(self, run: RunAsRead, inputs: Mapping[str, Any], who: C) -> str:
        """Start a run that is another with some inputs changed; a call that receives what it received before
        isn't made again.

        Args:
            run: The run to start from, as read.
            inputs: The inputs to change, by name; the rest stay as ``run`` had them.
            who: The caller who starts the new run, on whose own nodes it is loaded.

        Returns:
            The new run's id.

        Raises:
            InvalidCallError: The graph can't run on ``who``'s nodes, or refuses the inputs.
        """
        nodes = self._nodes_for(who)
        again = await asyncio.to_thread(lambda: _loaded(run.row, nodes).with_inputs(inputs))
        return await self._create(run.row.graph, who, again)

    async def answer(self, run: RunAsRead, answers: Mapping[str, Any]) -> Literal["started", "queued"]:
        """Answer what waits in a run, by waiting id.

        While the run is running, the answers are queued until the resume call ends. Otherwise the run takes them
        as ``run.answer`` does, whatever its status, and it is resumed; an earlier answer to the same question that
        was refused, or that stayed queued through a resume call that raised, is removed from the row's state.

        Args:
            run: The run as read.
            answers: An answer for each waiting call, by its waiting id.

        Returns:
            ``queued`` when the run was running, else ``started``.

        Raises:
            RefusedError: An answer is queued for one of the ids already (``already_answered``, one problem for
                each), the run is complete (``run_complete``), or another call claimed it first (``run_changed``).
            InvalidCallError: The run refuses an answer; every answer it refuses is listed.
        """
        def queued_with_these(state: StoredState) -> StoredState:
            if twice := state.queued_answers.keys() & answers.keys():
                raise RefusedError(*(router_problems.already_answered(waiting_id=waiting_id) for waiting_id in sorted(twice)))
            return state.model_copy(update={"queued_answers": {**state.queued_answers, **answers}})

        run = await self._ask_of_a_running_run(run, queued_with_these)
        if run.state.is_running:
            return "queued"
        if run.state.status == "complete":
            raise RefusedError(router_problems.run_complete())
        row, nodes = run.row, self._nodes_for(self.started_by(run.row))
        answered = await asyncio.to_thread(lambda: _loaded(row, nodes).answer(answers).to_json())
        state = run.state.answered(answers.keys()).running()
        if not await self._claim(run, state, changed=answered):
            raise RefusedError(router_problems.run_changed())
        return "started"

    async def resume(self, run: RunAsRead, max_items: int | None = None) -> None:
        """Resume a run that ended early; no node call that finished is made again.

        Args:
            run: The run as read.
            max_items: A higher ceiling on the items of any one Series, for a run that stopped at its ceiling. The
                run is stored with it as the run is claimed, so every later resume call has it.

        Raises:
            RefusedError: The run did not end early (``run_not_resumable``), or another call claimed
                the run first (``run_changed``).
            InvalidCallError: With ``max_items``, the stored run can't be loaded to raise its ceiling.
        """
        if run.state.cause is None:
            raise RefusedError(router_problems.run_not_resumable(status=run.state.status))
        state = run.state.running()
        raised = None
        if max_items is not None:
            row, nodes = run.row, self._nodes_for(self.started_by(run.row))
            raised = await asyncio.to_thread(lambda: _loaded(row, nodes).with_max_items(max_items).to_json())
        if not await self._claim(run, state, changed=raised):
            raise RefusedError(router_problems.run_changed())

    async def cancel(self, run: RunAsRead) -> bool:
        """Ask a running run to stop before its next node call: at once when this worker resumes it, at the next
        renewal of its lease when another does.

        The cancel is written to the row's state, and a resume call is stopped only by what its row says: this
        worker reads the row again here, and another worker reads it at its next renewal.

        Args:
            run: The run as read.

        Returns:
            Whether the run was running as read, or since, so that the cancel was written.
        """
        run = await self._ask_of_a_running_run(run, lambda state: state.model_copy(update={"cancel_asked": True}))
        if run.state.is_running:
            for hold in [hold for hold in self._holds if hold.run_id == run.row.id]:
                async with hold.lock:
                    await self._read_again(hold)
        return run.state.is_running

    async def events(self, run: RunAsRead, after: int = -1) -> AsyncGenerator[tuple[int, str]]:
        """The events of a run as they are stored, from whichever worker resumes it, until the run is not running.

        A row stops saying ``running`` only after the ``run_end`` of its resume call is stored. So the row is read
        only once a ``run_end`` was the last event, or when there was no event at all, and otherwise events alone
        are read.

        Args:
            run: The run as read.
            after: The number of the last event the follower has; ``-1`` for none.

        Yields:
            Each event numbered above ``after``, in order, as ``(number, body)``.
        """
        run_id = run.row.id
        may_have_ended = True
        while True:
            new = await self._store.events(run_id, after)
            for number, body in new:
                after = number
                yield number, body
            if new:
                may_have_ended = from_json(new[-1][1])["type"] == "run_end"
            elif may_have_ended and not (await self.read(run_id)).state.is_running:
                for event in await self._store.events(run_id, after):  # stored before the row said so
                    yield event
                return
            else:  # the row is written after its run_end, so it may still say running for a poll or two
                await asyncio.sleep(self._poll)

    async def sweep(self) -> None:
        """Go on with every run whose lease ran out: its worker died or handed it back. A run that can't be taken
        (its row can't be read, the store fails) is logged and left for the next call, and the others are taken."""
        held = {hold.run_id for hold in self._holds}
        for row in await self._store.expired(time.time()):
            if row.id not in held:
                try:
                    run = RunAsRead.of(row)
                    await self._claim(run, run.state)
                except Exception:
                    log.exception("run %s, whose lease ran out, was not taken over; the next sweep tries again", row.id)

    @asynccontextmanager
    async def serving(self) -> AsyncGenerator[None]:
        """The context a worker serves in: a lifespan enters it.

        While entered, runs whose lease ran out are taken over, three times a lease; a sweep that raises is logged
        and the next one runs. On leaving, every resume call this worker has going is stopped and its run handed back,
        a run begun or taken meanwhile included. A run handed back keeps the node calls already made, so the next
        worker repeats none of them once the last checkpoint is written.

        Yields:
            Nothing; the worker serves until the context is left.
        """
        async def keep_sweeping() -> None:
            while True:
                try:
                    await self.sweep()
                except Exception:
                    log.exception("the sweep for runs whose lease ran out failed; the next one tries again")
                await asyncio.sleep(self._lease / 3)

        self._handing_back.clear()
        sweeping = asyncio.create_task(keep_sweeping())
        try:
            yield
        finally:
            self._handing_back.set()
            sweeping.cancel()
            while self._holds:  # a run begun or taken while the others were handed back is handed back too
                await asyncio.gather(*self._holds.values())

    # -- taking a run -----------------------------------------------------------------------------

    def _mine(self) -> RunChanges:
        """The owner and lease of a row this worker holds."""
        return {"owner": self.id, "lease_until": time.time() + self._lease}

    async def _create(self, graph: str, who: C, run: Run[Any]) -> str:
        """Store a run, held by this worker, and make its first resume call in the background.

        Args:
            graph: The run's graph, as JSON.
            who: The caller who starts it.
            run: The run with its inputs, ready.

        Returns:
            The run's id.
        """
        saved, outcome = await asyncio.to_thread(lambda: (run.to_json(), Outcome.so_far(run).model_dump_json()))
        state = StoredState.started()
        written: RunChanges = {"run": saved, **state.columns(), "outcome": outcome, **self._mine()}
        row = StoredRun(id=uuid.uuid4().hex, graph=graph, caller=self._callers.dump_json(who).decode(), rev=0, **written)
        await self._store.create(row)
        self._resume_in_background(row, state)
        return row.id

    async def _claim(self, run: RunAsRead, state: StoredState, *, changed: str | None = None) -> bool:
        """Take a run for this worker if it is still at the ``rev`` read, and resume it in the background. Of two
        workers, one wins.

        Args:
            run: The run as read.
            state: The state the row takes with the claim; its ``status`` and ``state`` are written from it.
            changed: The run as the call that claims it changed it, as JSON (a caller's answers given, or its
                ceiling raised); the row's ``run`` then. Left out, the row keeps the run it has.

        Returns:
            Whether this worker took the run.
        """
        written: RunChanges = {**state.columns(), **self._mine()}
        if changed is not None:
            written["run"] = changed
        if not await self._store.update(run.row.id, run.row.rev, written):
            return False
        self._resume_in_background(replace(run.row, **written, rev=run.row.rev + 1), state)
        return True

    def _resume_in_background(self, row: StoredRun, state: StoredState) -> None:
        """Start a task that resumes a run this worker has just written. A cancel already asked in ``state`` stops
        the first resume call before its first node call.

        Args:
            row: The row as written.
            state: The row's state.
        """
        hold = _Hold(row.id, row.rev, state)
        if state.cancel_asked:
            hold.stop.set()
        self._holds[hold] = asyncio.get_running_loop().create_task(self._resume_while_held(hold, row))

    # -- a held run -------------------------------------------------------------------------------

    async def _resume_while_held(self, hold: _Hold, row: StoredRun) -> None:
        """Make one resume call after another on a held run, renewing its lease, until the hold is released.

        The hold is released when a resume call ends and the run is not to be resumed at once (no queued answer
        made it running again), or when another worker took the run.

        What a host's code raises and what conductor refuses are handled where they happen (``_resume_steps``,
        ``Ended.row_after``), so an exception that reaches here is the store's, raised when an event, a checkpoint
        or the end of a resume call was written. It is the host's outage and never fails the run: it is logged,
        no more nodes are started and the run is left as it is in the store, ``running``. Its lease runs out, and
        a worker resumes it from the last checkpoint; a node call that is still on a thread then may be made again
        while it runs.

        Args:
            hold: The hold on the run.
            row: The row as this worker wrote it.
        """
        renewing = asyncio.create_task(self._renew(hold))
        try:
            while not hold.released:
                row = await self._finish(hold, row, await self._resume_once(hold, row))
        except Exception:
            log.exception("the store raised while run %s was resumed; a worker resumes the run once its lease runs out", row.id)
        finally:
            renewing.cancel()
            del self._holds[hold]

    async def _renew(self, hold: _Hold) -> None:
        """Renew the lease three times a lease, until the run is given up or taken.

        A renewal that raises is logged and the next one is tried, so one failed statement doesn't cost this
        worker the run. A renewal the store hasn't answered within a third of a lease is given up and the next is
        made at once. It doesn't keep the worker's other writes to the row waiting, so a worker whose run another
        worker took meanwhile learns it at its next checkpoint or event, and a single statement that never returns
        leaves a third of the lease to renew in.

        Args:
            hold: The hold on the run.
        """
        wait = self._lease / 3
        while not hold.released:
            await asyncio.sleep(wait)
            wait = self._lease / 3
            try:
                async with asyncio.timeout(self._lease / 3):
                    await self._write(hold, {"lease_until": time.time() + self._lease})
            except TimeoutError:
                log.error("the store did not answer a renewal of the lease of run %s; the next is made at once", hold.run_id)
                wait = 0.0
            except Exception:
                log.exception("the lease of run %s was not renewed; the next renewal tries again", hold.run_id)

    async def _resume_once(self, hold: _Hold, row: StoredRun) -> ResumeEnd:
        """Make one resume call on a held run and store its events and checkpoints.

        The run is loaded and resumed in ``_resume_steps``, and every call to the store is made here. So an
        exception that leaves this method is the store's: the host's outage, which never fails a run.

        Args:
            hold: The hold on the run.
            row: The row, whose ``run`` is loaded and resumed.

        Returns:
            How the resume call ended: with the run of its ``run_end``; handed back, when the worker shuts down and
            no cancel was asked, the run stored as its last checkpoint; or as a call that raised.
        """
        end: ResumeEnd = HandedBack()
        async with contextlib.aclosing(self._resume_steps(hold, row)) as steps:  # a raise in this loop ends the resume call before its next node call
            async for step in steps:
                match step:
                    case ResumeEnd():
                        end = step
                    case str():
                        if not await self._store.add_event(row.id, self.id, step):
                            hold.stop.set()  # another worker holds the run
                    case _:
                        await self._write(hold, step)
        return end

    async def _resume_steps(self, hold: _Hold, row: StoredRun) -> AsyncGenerator[str | RunChanges | ResumeEnd]:
        """Load a held run, make one resume call on it, and yield what is to be stored. Nothing here calls the store.

        Args:
            hold: The hold on the run.
            row: The row, whose ``run`` is loaded and resumed.

        Yields:
            Each event to store, as JSON; each checkpoint, as the row's ``run`` and ``outcome`` to write (the
            outcome as the run last stopped, with the checkpoint's progress); and last, unless the run
            is handed back, how the resume call ended (a ``ResumeEnd``). An exception raised outside a node is
            logged and ends the call as ``Raised``, with the outcome of the run as the row has it: the run as
            loaded, or the last checkpoint yielded; ``Outcome.empty()``, for a run that could not be loaded. Its cause is
            conductor's own problem where conductor refused the call (a context missing or of another class, a
            run that can't be read or written), the first of them where there are several; else ``resume_failed``.
        """
        stopped = Outcome.model_validate_json(row.outcome)  # as the run last stopped: a checkpoint changes its progress only
        held = Outcome.empty()  # the run as the row holds it; nothing, until the run is loaded
        try:
            who = self.started_by(row)
            run = await asyncio.to_thread(_loaded, row, self._nodes_for(who))
            held = await asyncio.to_thread(Outcome.so_far, run)
            stream = run.stream(
                context=self._context(who), timeout=self._resume_timeout,
                cancel=hold.stop, hand_back=self._handing_back, checkpoint_every=self._checkpoint_every,
            )
            async with contextlib.aclosing(stream):
                async for event in stream:
                    if isinstance(event, RunCheckpoint):
                        saved = await asyncio.to_thread(event.run.to_json)
                        held = await asyncio.to_thread(Outcome.so_far, event.run)
                        yield {"run": saved, "outcome": stopped.model_copy(update={"progress": held.progress}).model_dump_json()}
                    elif isinstance(event, RunEnd):
                        yield await asyncio.to_thread(end_of, event, held)
                    else:
                        yield event.model_dump_json()
        except InvalidCallError as refusal:
            every = "; ".join(f"{problem.type} at {'.'.join(map(str, problem.loc))}: {problem.msg}" for problem in refusal.problems)
            log.error("resuming run %s was refused: %s", row.id, every, exc_info=True)  # the cause is the first problem only
            yield Raised.refused(refusal, held)
        except Exception:
            log.exception("resuming run %s failed", row.id)
            yield Raised.because(router_problems.resume_failed(), held)

    async def _finish(self, hold: _Hold, row: StoredRun, end: ResumeEnd) -> StoredRun:
        """Write how a resume call ended. What is written is worked out from the row's state as read for each try
        of the write, so a cancel or an answer that is written meanwhile is never written over.

        The call's ``run_end`` is stored once, and before the row says the run stopped running, so a follower that
        reads that it stopped has the event to read. A run another worker took is left as it is, and the hold is
        released. A run that is to be resumed again as the worker shuts down is left for another worker.

        The lease is renewed until the end is written, and the run is made JSON outside the lock, so a run however
        long it takes to write stays this worker's.

        Args:
            hold: The hold on the run.
            row: The row as the resume call started from it.
            end: How the resume call ended.

        Returns:
            ``row`` with the ``run`` and ``outcome`` that were written. The hold is released unless this worker
            resumes the run again at once.
        """
        if end.run_end is not None and not await self._store.add_event(hold.run_id, self.id, end.run_end):
            hold.released = True  # another worker holds the run, so the write of the row would lose too
            return row
        async with hold.lock:  # so no renewal moves the rev between this read and what is taken in from it
            await self._read_again(hold)
        while not hold.released:
            reads = hold.reads
            after = await asyncio.to_thread(end.row_after, hold.state)
            goes_on = after.state.is_running and not self._handing_back.is_set()
            holder = self._mine() if goes_on else _HANDED_BACK if after.state.is_running else _UNHELD
            changes: RunChanges = {**after.state.columns(), **after.changes, **holder}
            async with hold.lock:  # no renewal lands after the write that gives the run up
                if hold.reads == reads and await self._write_once(hold, changes):  # decided from the state as it still is
                    hold.state, hold.released = after.state, not goes_on
                    if not after.state.cancel_asked:
                        hold.stop.clear()  # the cancel that stopped this resume call does not stop the next
                    return replace(row, **after.changes)
        return row

    async def _write(self, hold: _Hold, changes: RunChanges) -> bool:
        """Write a checkpoint or a renewal to a held run, again for as long as another writer (an answer queued, a
        cancel) moved its ``rev`` on first.

        Args:
            hold: The hold on the run.
            changes: The columns to write.

        Returns:
            Whether it was written. A run given up, or that another worker took, takes no write.
        """
        async with hold.lock:
            while not hold.released:
                if await self._write_once(hold, changes):
                    return True
            return False

    async def _write_once(self, hold: _Hold, changes: RunChanges) -> bool:
        """Write ``changes`` to a held run at the ``rev`` this worker knows, under the hold's lock; a write that
        loses reads the row again.

        Args:
            hold: The hold on the run.
            changes: The columns to write.

        Returns:
            Whether it was written.
        """
        if await self._store.update(hold.run_id, hold.rev, changes):
            hold.rev += 1
            return True
        await self._read_again(hold)
        return False

    async def _read_again(self, hold: _Hold) -> None:
        """Read the row of a held run, under the hold's lock, and take in what another writer changed: its state,
        a cancel asked of it, and whether another worker took it.

        Args:
            hold: The hold on the run.
        """
        run = await self.read(hold.run_id)
        hold.reads += 1
        hold.rev, hold.state = run.row.rev, run.state
        hold.released = run.row.owner != self.id
        if hold.released or hold.state.cancel_asked:
            hold.stop.set()

    # -- a run another worker may hold ------------------------------------------------------------

    async def _ask_of_a_running_run(self, run: RunAsRead, asked: Callable[[StoredState], StoredState]) -> RunAsRead:
        """Write what a caller asks of a running run (an answer to queue, a cancel) into the row's state. It is
        tried for as long as the run is running, reading the row again each time another writer moved its ``rev``
        on first.

        Args:
            run: The run as read.
            asked: The state to write, made from the state as read for each try.

        Returns:
            The run after the call: still running, with what was asked written; or as last read, not running, with
            nothing written.
        """
        while run.state.is_running:
            state = asked(run.state)
            changes = state.columns()
            if await self._store.update(run.row.id, run.row.rev, changes):
                return RunAsRead(replace(run.row, **changes, rev=run.row.rev + 1), state)
            run = await self.read(run.row.id)
        return run


def _loaded(row: StoredRun, nodes: NodeRegistry[Any]) -> Run[Any]:
    """A stored run, loaded.

    Args:
        row: The run's row.
        nodes: The nodes its graph is loaded on.

    Returns:
        The run as the row's ``run`` saved it.
    """
    return Run.from_json(row.run, Graph.compile(row.graph, nodes))
