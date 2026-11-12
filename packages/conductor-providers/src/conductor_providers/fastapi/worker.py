"""``Worker``: runs the legs of stored runs for one process, on any number of processes sharing one store.

A worker runs each leg in the background, holds a lease on its run while it
does, and writes a checkpoint every few seconds. Another worker takes over a
run whose lease ran out, from its last checkpoint, and a worker that shuts
down hands its runs back. Every write is a compare-and-set on the run's
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
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Literal, TypeVar

from conductor import Graph, GraphRecord, InvalidCallError, NodeRegistry, Run, Waiting
from conductor.events import RunCheckpoint, RunEnd
from pydantic import TypeAdapter
from pydantic_core import from_json

from conductor_providers.fastapi import router_problem
from conductor_providers.fastapi.leg_end import HandedBack, LegEnd, Raised, end_of
from conductor_providers.fastapi.router_problem import NoSuchRunError, RefusedError
from conductor_providers.fastapi.store import RunStore
from conductor_providers.fastapi.stored_run import (
    RUNNING,
    Outcome,
    RunAsRead,
    RunChanges,
    StoredRun,
    StoredState,
)

C = TypeVar("C")

log = logging.getLogger("conductor_providers.fastapi")

#: The owner and lease of a row no leg is wanted for.
_UNHELD: RunChanges = {"owner": None, "lease_until": None}

#: The owner and lease of a row handed back: its lease already run out, so the next sweep of any worker takes it.
_HANDED_BACK: RunChanges = {"owner": None, "lease_until": 0.0}


@dataclass(eq=False)
class _Hold:
    """A run this worker holds while it runs legs of it.

    Attributes:
        run_id: The run's id.
        rev: The row's ``rev`` as this worker last wrote or read it.
        state: The row's state at that ``rev``.
        stop: Set to end the running leg before its next unit, from the row as this worker reads it: a cancel is
            asked in its state, or another worker holds it. Cleared when a leg's end leaves no cancel asked.
        lock: Held for each of this worker's writes to the row, so they land one at a time.
        released: Whether the row was given up or taken by another worker; this worker writes nothing more to it.
        reads: How many times this worker read the row again, each time taking in what another writer changed.
    """

    run_id: str
    rev: int
    state: StoredState
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    released: bool = False
    reads: int = 0


class Worker(Generic[C]):
    """Runs the legs of stored runs for one process, holding a lease on each run while it does.

    Each method that takes a run takes it as the caller read it (``read``). A call that starts a leg (an answer
    to a run no leg runs for, a resume) writes only if the run is still at that ``rev``; what is asked of a running
    leg (an answer to queue, a cancel) is written however the ``rev`` moved, for as long as the leg is wanted.
    """

    def __init__(
        self,
        *,
        store: RunStore,
        nodes_for: Callable[[C], NodeRegistry],
        callers: TypeAdapter[C],
        from_run: Callable[[C], Mapping[type, Any]],
        max_items: int | None,
        lease: float,
        checkpoint_every: float,
        leg_timeout: float | None,
        poll: float,
    ) -> None:
        """A worker on ``store``.

        Args:
            store: Where runs and their events are kept.
            nodes_for: The nodes a caller's graphs may use; called for every call and every leg.
            callers: Reads and writes a caller as the JSON a run keeps.
            from_run: The ``FromRun`` values of a leg, from the caller who started its run.
            max_items: The ceiling on the items of any one Series that a run starts with.
            lease: The seconds this worker holds a run between renewals.
            checkpoint_every: The seconds between saves of a running leg.
            leg_timeout: The seconds one leg may run.
            poll: The seconds between a follower's reads of new events.
        """
        self.id = uuid.uuid4().hex
        self._store = store
        self._nodes_for = nodes_for
        self._callers = callers
        self._from_run = from_run
        self._max_items = max_items
        self._lease = lease
        self._checkpoint_every = checkpoint_every
        self._leg_timeout = leg_timeout
        self._poll = poll
        self._holds: dict[_Hold, asyncio.Task[None]] = {}
        self._handing_back = asyncio.Event()

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

    async def begin(self, graph: GraphRecord, inputs: Mapping[str, Any], who: C) -> str:
        """Start a run and run its first leg.

        Args:
            graph: The graph to run.
            inputs: Values for the graph's open inputs, by name or by address.
            who: The caller who starts it, on whose nodes the graph is loaded.

        Returns:
            The run's id.

        Raises:
            InvalidCallError: The graph can't run on ``who``'s nodes, or refuses the inputs.
        """
        nodes = self._nodes_for(who)
        saved = await asyncio.to_thread(lambda: Graph.compile(graph, nodes).with_inputs(inputs, max_items=self._max_items).to_json())
        return await self._create(graph.model_dump_json(), who, saved)

    async def again(self, run: RunAsRead, inputs: Mapping[str, Any], who: C) -> str:
        """Start a run that is another with some inputs changed; a call that receives what it received before
        isn't made again.

        Args:
            run: The run to start from, as read.
            inputs: The inputs to change, by name or by address; the rest stay as ``run`` had them.
            who: The caller who starts the new run, on whose own nodes it is loaded.

        Returns:
            The new run's id.

        Raises:
            InvalidCallError: The graph can't run on ``who``'s nodes, or refuses the inputs.
        """
        nodes = self._nodes_for(who)
        saved = await asyncio.to_thread(lambda: _loaded(run.row, nodes).with_inputs(inputs).to_json())
        return await self._create(run.row.graph, who, saved)

    async def answer(self, run: RunAsRead, answers: Mapping[str, Any]) -> Literal["started", "queued"]:
        """Answer what waits in a run, by waiting id.

        While a leg runs, the answers are queued for its end. Otherwise the run takes them as ``run.answer`` does,
        whatever its status, and its next leg starts; an earlier answer to the same question that was refused, or
        that stayed queued through a leg that raised, is then done with.

        Args:
            run: The run as read.
            answers: An answer for each waiting unit, by its waiting id.

        Returns:
            ``queued`` when a leg was running, else ``started``.

        Raises:
            RefusedError: An answer is queued for one of the ids already (``already_answered``, one problem for
                each), the run is complete (``run_complete``), or another call claimed it first (``run_changed``).
            InvalidCallError: The run refuses an answer; every answer it refuses is listed.
        """
        def queued_with_these(state: StoredState) -> StoredState:
            if twice := state.queued_answers.keys() & answers.keys():
                raise RefusedError(*(router_problem.already_answered(waiting_id=waiting_id) for waiting_id in sorted(twice)))
            return state.model_copy(update={"queued_answers": {**state.queued_answers, **answers}})

        run = await self._ask_of_a_running_leg(run, queued_with_these)
        if run.state.leg_wanted:
            return "queued"
        if run.state.run_state.status == "complete":
            raise RefusedError(router_problem.run_complete())
        row, nodes = run.row, self._nodes_for(self.started_by(run.row))
        answered = await asyncio.to_thread(lambda: _loaded(row, nodes).answer(answers).to_json())
        state = run.state.answered(answers.keys()).model_copy(update={"run_state": RUNNING})
        if not await self._claim(run, state, changed=answered):
            raise RefusedError(router_problem.run_changed())
        return "started"

    async def resume(self, run: RunAsRead, max_items: int | None = None) -> None:
        """Go on with a run whose last leg ended early; nothing finished is called again.

        Args:
            run: The run as read.
            max_items: A higher ceiling on the items of any one Series, for a run that stopped at its ceiling. The
                run is stored with it as the run is claimed, so every later leg goes on under it.

        Raises:
            RefusedError: The run's last leg did not end early (``run_not_resumable``), or another call claimed
                the run first (``run_changed``).
            InvalidCallError: With ``max_items``, the stored run can't be loaded to raise its ceiling.
        """
        if run.state.cause is None:
            raise RefusedError(router_problem.run_not_resumable(status=run.state.run_state.status))
        state = run.state.model_copy(update={"run_state": RUNNING})
        raised = None
        if max_items is not None:
            row, nodes = run.row, self._nodes_for(self.started_by(run.row))
            raised = await asyncio.to_thread(lambda: _loaded(row, nodes).with_max_items(max_items).to_json())
        if not await self._claim(run, state, changed=raised):
            raise RefusedError(router_problem.run_changed())

    async def cancel(self, run: RunAsRead) -> bool:
        """Ask the running leg of a run to stop before its next unit: at once when this worker runs it, at its next
        renewal when another does.

        The cancel is written to the run's state, and a leg is stopped only by what its row says: this worker's
        leg reads the row again here, where another worker's reads it at its next renewal.

        Args:
            run: The run as read.

        Returns:
            Whether a leg was wanted for the run as read, or since, so that the cancel was written.
        """
        run = await self._ask_of_a_running_leg(run, lambda state: state.model_copy(update={"cancel_asked": True}))
        if run.state.leg_wanted:
            for hold in [hold for hold in self._holds if hold.run_id == run.row.id]:
                async with hold.lock:
                    await self._read_again(hold)
        return run.state.leg_wanted

    async def events(self, run: RunAsRead, after: int = -1) -> AsyncGenerator[tuple[int, str]]:
        """The events of a run as they are stored, from whichever worker runs it, until no leg is wanted for it.

        A row stops saying a leg is wanted only after the leg's ``run_end`` is stored. So the row is read only once
        a ``run_end`` was the last event, or when there was no event at all, and otherwise events alone are read.

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
            elif may_have_ended and not (await self.read(run_id)).state.leg_wanted:
                for event in await self._store.events(run_id, after):  # stored before the row said so
                    yield event
                return
            else:  # the row follows its run_end, so it may still say a leg is wanted for a poll or two
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
        and the next one runs. On leaving, every leg this worker runs is stopped and its run handed back, a run
        begun or taken meanwhile included. A leg handed back keeps the calls already made, so the next worker
        repeats none of them once the leg's last checkpoint is written.

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

    async def _create(self, graph: str, who: C, saved: str) -> str:
        """Store a run, held by this worker, and run its first leg.

        Args:
            graph: The run's graph, as JSON.
            who: The caller who starts it.
            saved: The run with its inputs, as JSON.

        Returns:
            The run's id.
        """
        state = StoredState.started()
        written: RunChanges = {"run": saved, **state.columns(), "outcome": Outcome.empty().model_dump_json(), **self._mine()}
        row = StoredRun(id=uuid.uuid4().hex, graph=graph, caller=self._callers.dump_json(who).decode(), rev=0, **written)
        await self._store.create(row)
        self._run_legs(row, state)
        return row.id

    async def _claim(self, run: RunAsRead, state: StoredState, *, changed: str | None = None) -> bool:
        """Take a run for this worker if it is still at the ``rev`` read, and run its next leg. Of two workers, one
        wins.

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
        self._run_legs(replace(run.row, **written, rev=run.row.rev + 1), state)
        return True

    def _run_legs(self, row: StoredRun, state: StoredState) -> None:
        """Run the legs of a run this worker has just written, in the background; a cancel asked of it the first leg
        hears before its first unit.

        Args:
            row: The row as written.
            state: The row's state.
        """
        hold = _Hold(row.id, row.rev, state)
        if state.cancel_asked:
            hold.stop.set()
        self._holds[hold] = asyncio.get_running_loop().create_task(self._legs(hold, row))

    # -- a held run -------------------------------------------------------------------------------

    async def _legs(self, hold: _Hold, row: StoredRun) -> None:
        """Run legs of a held run, renewing its lease, until the end of one gives the run up.

        What a host's code raises and what conductor refuses are dealt with where they are met (``_leg_steps``,
        ``Ended.row_after``), so a raise that reaches here is the store's, at an event, a checkpoint or a leg's
        end. It is the host's outage and never fails the run: it is logged, the leg starts nothing more and the
        run is given up as it is in the store, ``running``. Its lease runs out, and a worker goes on from the last
        checkpoint; a call the given-up leg still has on a thread then may be made again while it runs.

        Args:
            hold: The hold on the run.
            row: The row as this worker wrote it.
        """
        renewing = asyncio.create_task(self._renew(hold))
        try:
            while not hold.released:
                row = await self._finish(hold, row, await self._leg(hold, row))
        except Exception:
            log.exception("the store raised while a leg of run %s ran; a worker goes on with the run once its lease runs out", row.id)
        finally:
            renewing.cancel()
            del self._holds[hold]

    async def _renew(self, hold: _Hold) -> None:
        """Renew the lease three times a lease, until the run is given up or taken.

        A renewal that raises is logged and the next one is tried, so one failed statement doesn't cost a leg its
        run. One the store hasn't answered a third of a lease on is given up and the next is made at once: it
        doesn't keep the leg's other writes waiting, so a leg whose run another worker took meanwhile learns it at
        its next checkpoint or event, and a single statement that never returns leaves a third of the lease to
        renew in.

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

    async def _leg(self, hold: _Hold, row: StoredRun) -> LegEnd:
        """Run one leg of a held run and store its events and checkpoints.

        The leg's own work is done in ``_leg_steps``, and every call to the store is made here. So a raise that
        leaves this method is the store's: the host's outage, which never fails a run.

        Args:
            hold: The hold on the run.
            row: The row, whose ``run`` the leg starts from.

        Returns:
            How the leg ended: with the run of its ``run_end``; handed back, when the worker shuts down and no
            cancel was asked, its run stored as the leg's last checkpoint; or as a leg that raised.
        """
        end: LegEnd = HandedBack()
        async with contextlib.aclosing(self._leg_steps(hold, row)) as steps:  # a raise in this loop ends the leg before its next unit
            async for step in steps:
                match step:
                    case LegEnd():
                        end = step
                    case str():
                        if not await self._store.add_event(row.id, self.id, step):
                            hold.stop.set()  # another worker holds the run
                    case _:
                        await self._write(hold, step)
        return end

    async def _leg_steps(self, hold: _Hold, row: StoredRun) -> AsyncGenerator[str | RunChanges | LegEnd]:
        """Run one leg of a held run, and say what the store is to take of it. Nothing here calls the store.

        Args:
            hold: The hold on the run.
            row: The row, whose ``run`` the leg starts from.

        Yields:
            Each event to store, as JSON; each checkpoint, as the row's ``run`` to write; and last, unless the leg
            is handed back, how the leg ended. A raise anywhere in the leg outside a node is logged and ends it
            as a leg that raised, which lists what waits in the run as the row holds it: the run as loaded, or
            the last checkpoint the store was given; nothing, for a run that could not be loaded. Its cause is
            conductor's own problem where conductor refused the leg (a ``FromRun`` value nobody gives, a run
            that can't be read or written), the first of them where there are several; else ``leg_failed``.
        """
        waiting: list[Waiting] = []  # in the run as the row holds it; nothing, until the run is loaded
        try:
            who = self.started_by(row)
            run = await asyncio.to_thread(_loaded, row, self._nodes_for(who))
            waiting = run.waiting
            stream = run.stream(
                from_run=self._from_run(who), timeout=self._leg_timeout,
                cancel=hold.stop, hand_back=self._handing_back, checkpoint_every=self._checkpoint_every,
            )
            async with contextlib.aclosing(stream):
                async for event in stream:
                    if isinstance(event, RunCheckpoint):
                        yield {"run": await asyncio.to_thread(event.run.to_json)}
                        waiting = event.run.waiting
                    elif isinstance(event, RunEnd):
                        yield await asyncio.to_thread(end_of, event, waiting)
                    else:
                        yield event.model_dump_json()
        except InvalidCallError as refusal:
            every = "; ".join(f"{problem.type} at {'.'.join(map(str, problem.loc))}: {problem.msg}" for problem in refusal.errors())
            log.error("the leg of run %s was refused: %s", row.id, every, exc_info=True)  # a cause holds the first only
            yield Raised.refused(refusal, waiting)
        except Exception:
            log.exception("the leg of run %s failed", row.id)
            yield Raised.because(router_problem.leg_failed(), waiting)

    async def _finish(self, hold: _Hold, row: StoredRun, end: LegEnd) -> StoredRun:
        """Write how a leg ended, decided from the row's state as read for each try of the write, so a cancel or an
        answer that lands meanwhile is never written over.

        A leg's ``run_end`` is stored once, and before the row says the run stopped running, so a follower that
        reads that it stopped has the event to read. A run another worker took is left as it is, and the hold is
        released. A next leg that is wanted as the worker shuts down is left for another worker.

        The lease is renewed until the end is written, and the run is made JSON outside the lock, so a run however
        long it takes to write stays this worker's.

        Args:
            hold: The hold on the run.
            row: The row as the leg started from it.
            end: How the leg ended.

        Returns:
            ``row`` with the ``run`` and ``outcome`` that were written. The hold is released unless this worker
            goes on with a next leg of it.
        """
        if end.run_end is not None and not await self._store.add_event(hold.run_id, self.id, end.run_end):
            hold.released = True  # another worker holds the run, so the write of the row would lose too
            return row
        async with hold.lock:  # so no renewal moves the rev between this read and what is taken in from it
            await self._read_again(hold)
        while not hold.released:
            reads = hold.reads
            after = await asyncio.to_thread(end.row_after, hold.state)
            goes_on = after.state.leg_wanted and not self._handing_back.is_set()
            holder = self._mine() if goes_on else _HANDED_BACK if after.state.leg_wanted else _UNHELD
            changes: RunChanges = {**after.state.columns(), **after.changes, **holder}
            async with hold.lock:  # no renewal lands after the write that gives the run up
                if hold.reads == reads and await self._write_once(hold, changes):  # decided from the state as it still is
                    hold.state, hold.released = after.state, not goes_on
                    if not after.state.cancel_asked:
                        hold.stop.clear()  # the cancel this leg heard is not the next leg's
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

    async def _ask_of_a_running_leg(self, run: RunAsRead, asked: Callable[[StoredState], StoredState]) -> RunAsRead:
        """Write what a caller asks of a run's running leg into its state, for as long as a leg is wanted for the
        run, reading the run again each time another writer moved its ``rev`` on first.

        Args:
            run: The run as read.
            asked: The state to write, made from the state as read for each try.

        Returns:
            The run as it stands after the call: with what was asked written, a leg still wanted for it; or as last
            read, with no leg wanted and nothing written.
        """
        while run.state.leg_wanted:
            state = asked(run.state)
            changes = state.columns()
            if await self._store.update(run.row.id, run.row.rev, changes):
                return RunAsRead(replace(run.row, **changes, rev=run.row.rev + 1), state)
            run = await self.read(run.row.id)
        return run


def _loaded(row: StoredRun, nodes: NodeRegistry) -> Run:
    """A stored run, loaded.

    Args:
        row: The run's row.
        nodes: The nodes its graph is loaded on.

    Returns:
        The run as the row's ``run`` saved it.
    """
    return Run.from_json(row.run, Graph.compile(row.graph, nodes))
