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

from conductor import EndedEarly, GraphRecord, NodeRegistry, Run
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
        stop: Set to end the running leg before its next unit.
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

    Each method that takes a run takes it as the caller read it (``read``), and writes only if the run is still
    at that ``rev``.
    """

    def __init__(
        self,
        *,
        store: RunStore,
        nodes_for: Callable[[C], NodeRegistry],
        caller_type: Any,
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
            caller_type: The type of a caller, which a run keeps as JSON.
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
        self._callers: TypeAdapter[C] = TypeAdapter(caller_type)
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
        """The caller who started the run of ``row``."""
        return self._callers.validate_json(row.caller)

    async def read(self, run_id: str) -> RunAsRead:
        """The run as the store has it.

        Raises:
            NoSuchRunError: The store has no run by that id.
        """
        row = await self._store.get(run_id)
        if row is None:
            raise NoSuchRunError
        return RunAsRead.of(row)

    async def begin(self, graph: GraphRecord, inputs: Mapping[str, Any], who: C) -> str:
        """Start a run of ``graph`` for ``who`` and run its first leg.

        Returns:
            The run's id.

        Raises:
            InvalidCallError: The graph can't run on ``who``'s nodes, or refuses the inputs.
        """
        nodes = self._nodes_for(who)
        saved = await asyncio.to_thread(lambda: nodes.load(graph).with_inputs(inputs, max_items=self._max_items).to_json())
        return await self._create(graph.model_dump_json(), who, saved)

    async def again(self, run: RunAsRead, inputs: Mapping[str, Any], who: C) -> str:
        """Start a run of ``who``'s, on ``who``'s own nodes, that is ``run`` with ``inputs`` changed; a call that
        receives what it received before isn't made again.

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
        whatever its status, and its next leg starts.

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
        state = run.state.model_copy(update={"run_state": RUNNING})
        if not await self._claim(run, state, answered=answered):
            raise RefusedError(router_problem.run_changed())
        return "started"

    async def resume(self, run: RunAsRead, max_items: int | None = None) -> None:
        """Go on with a run whose last leg ended early; nothing finished is called again.

        Args:
            run: The run as read.
            max_items: A higher ceiling on the items of any one Series, for a run that stopped at its ceiling.

        Raises:
            RefusedError: The run's last leg did not end early (``run_not_resumable``), or another call claimed
                the run first (``run_changed``).
        """
        if not isinstance(run.state.run_state, EndedEarly):
            raise RefusedError(router_problem.run_not_resumable(status=run.state.run_state.status))
        state = run.state.model_copy(update={"run_state": RUNNING})
        if not await self._claim(run, state, max_items=max_items):
            raise RefusedError(router_problem.run_changed())

    async def cancel(self, run: RunAsRead) -> bool:
        """Ask the running leg of a run to stop before its next unit: at once when this worker runs it, at its next
        renewal when another does.

        Returns:
            Whether a leg was running to ask.
        """
        run = await self._ask_of_a_running_leg(run, lambda state: state.model_copy(update={"cancel_asked": True}))
        for hold in self._holds:
            if hold.run_id == run.row.id:
                hold.stop.set()
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
        """While entered, take over runs whose lease ran out, three times a lease. On leaving, stop every leg this
        worker runs and hand its run back: a leg handed back keeps the calls already made, so the next worker
        repeats none. A sweep that raises is logged and the next one runs."""
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
            await asyncio.gather(*self._holds.values(), return_exceptions=True)

    # -- taking a run -----------------------------------------------------------------------------

    def _mine(self) -> RunChanges:
        """The owner and lease of a row this worker holds."""
        return {"owner": self.id, "lease_until": time.time() + self._lease}

    async def _create(self, graph: str, who: C, saved: str) -> str:
        """Store a run of ``who``'s, held by this worker, and run its first leg."""
        state = StoredState()
        written: RunChanges = {"run": saved, **state.columns(), "outcome": Outcome().model_dump_json(), **self._mine()}
        row = StoredRun(id=uuid.uuid4().hex, graph=graph, caller=self._callers.dump_json(who).decode(), **written)
        await self._store.create(row)
        self._run_legs(row, state)
        return row.id

    async def _claim(self, run: RunAsRead, state: StoredState, *, answered: str | None = None, max_items: int | None = None) -> bool:
        """Take a run for this worker if it is still at the ``rev`` read, and run its next leg. Of two workers, one
        wins.

        Args:
            run: The run as read.
            state: The state the row takes with the claim; its ``status`` and ``state`` are written from it.
            answered: The run with a caller's answers given, as JSON, for a claim that an answer makes; the row's
                ``run`` then. Left out, the row keeps the run it has.
            max_items: A ceiling on the items of any one Series for the leg, where the run's own is to be raised.

        Returns:
            Whether this worker took the run.
        """
        written: RunChanges = {**state.columns(), **self._mine()}
        if answered is not None:
            written["run"] = answered
        if not await self._store.update(run.row.id, run.row.rev, written):
            return False
        self._run_legs(replace(run.row, **written, rev=run.row.rev + 1), state, max_items)
        return True

    def _run_legs(self, row: StoredRun, state: StoredState, max_items: int | None = None) -> None:
        """Run the legs of ``row``, as this worker has just written it, in the background; a cancel asked of it the
        first leg hears before its first unit."""
        hold = _Hold(row.id, row.rev, state)
        if state.cancel_asked:
            hold.stop.set()
        self._holds[hold] = asyncio.get_running_loop().create_task(self._legs(hold, row, max_items))

    # -- a held run -------------------------------------------------------------------------------

    async def _legs(self, hold: _Hold, row: StoredRun, max_items: int | None) -> None:
        """Run legs of a held run, renewing its lease, until the end of one gives the run up.

        A raise while a leg's end is written, which may come after its ``run_end`` is stored, is logged and writes
        no second end: the row stays ``running``, its lease runs out, and a worker goes on from the last checkpoint.
        """
        renewing = asyncio.create_task(self._renew(hold))
        try:
            while not hold.released:
                end = await self._leg(hold, row, max_items)
                row, max_items = await self._finish(hold, row, end), None
        except Exception:
            log.exception("the end of the leg of run %s was not written; a worker goes on with the run once its lease runs out", row.id)
        finally:
            renewing.cancel()
            del self._holds[hold]

    async def _renew(self, hold: _Hold) -> None:
        """Renew the lease three times a lease, until the run is given up or taken. A renewal that raises is logged
        and the next one is tried, so one failed statement doesn't cost a leg its run."""
        while not hold.released:
            await asyncio.sleep(self._lease / 3)
            try:
                await self._write(hold, {"lease_until": time.time() + self._lease})
            except Exception:
                log.exception("the lease of run %s was not renewed; the next renewal tries again", hold.run_id)

    async def _leg(self, hold: _Hold, row: StoredRun, max_items: int | None) -> LegEnd:
        """Run one leg of ``row`` and store its events and checkpoints.

        Returns:
            How the leg ended: with the run of its ``run_end``; handed back, when the worker shuts down and no
            cancel was asked, its run stored as the leg's last checkpoint; or, for a raise anywhere in it outside
            a node, which is logged, as a leg that raised.
        """
        try:
            who = self.started_by(row)
            run = await asyncio.to_thread(_loaded, row, self._nodes_for(who))
            stream = run.stream(
                from_run=self._from_run(who), max_items=max_items, timeout=self._leg_timeout,
                cancel=hold.stop, hand_back=self._handing_back, checkpoint_every=self._checkpoint_every,
            )
            end: LegEnd = HandedBack()
            async with contextlib.aclosing(stream):  # a raise in this loop ends the leg before its next unit
                async for event in stream:
                    if isinstance(event, RunCheckpoint):
                        await self._write(hold, {"run": await asyncio.to_thread(event.run.to_json)})
                    elif isinstance(event, RunEnd):
                        end = await asyncio.to_thread(end_of, event)
                    elif not await self._store.add_event(row.id, self.id, event.model_dump_json()):
                        hold.stop.set()  # another worker holds the run
            return end
        except Exception:
            log.exception("the leg of run %s failed", row.id)
            return Raised(router_problem.leg_failed())

    async def _finish(self, hold: _Hold, row: StoredRun, end: LegEnd) -> StoredRun:
        """Write how a leg ended, decided from the row's state as read for each try of the write, so a cancel or an
        answer that lands meanwhile is never written over.

        A leg's ``run_end`` is stored once, and before the row says the run stopped running, so a follower that
        reads that it stopped has the event to read. A run another worker took is left as it is, and the hold is
        released. A next leg that is wanted as the worker shuts down is left for another worker.

        The lease is renewed until the end is written, and the run is made JSON outside the lock, so a run however
        long it takes to write stays this worker's.

        Returns:
            ``row`` with the ``run`` and ``outcome`` that were written. The hold is released unless this worker
            goes on with a next leg of it.
        """
        end_stored = False
        async with hold.lock:  # so no renewal moves the rev between this read and what is taken in from it
            await self._read_again(hold)
        while not hold.released:
            reads = hold.reads
            after = await asyncio.to_thread(end.row_after, hold.state)
            if after.run_end is not None and not end_stored:
                if not await self._store.add_event(hold.run_id, self.id, after.run_end):
                    hold.released = True  # another worker holds the run, so the write of the row would lose too
                    break
                end_stored = True
            goes_on = after.leg_wanted and not self._handing_back.is_set()
            holder = self._mine() if goes_on else _HANDED_BACK if after.leg_wanted else _UNHELD
            changes: RunChanges = {**after.state.columns(), **after.changes, **holder}
            async with hold.lock:  # no renewal lands after the write that gives the run up
                if hold.reads == reads and await self._write_once(hold, changes):  # decided from the state as it still is
                    hold.state, hold.released = after.state, not goes_on
                    return replace(row, **after.changes)
        return row

    async def _write(self, hold: _Hold, changes: RunChanges) -> bool:
        """Write a checkpoint or a renewal to a held run, again for as long as another writer (an answer queued, a
        cancel) moved its ``rev`` on first.

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
        a cancel asked of it, and whether another worker took it."""
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
    """The stored run, on ``nodes``."""
    return nodes.load(row.graph).load_run(row.run)
