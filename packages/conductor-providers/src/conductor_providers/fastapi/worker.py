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
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Generic, Literal, TypeVar

from conductor import (
    EndedEarly,
    Going,
    GraphRecord,
    InvalidCallError,
    NodeRegistry,
    Problem,
    Run,
    RunState,
    RunStatus,
    Waiting,
)
from conductor.events import RunCheckpoint, RunEnd
from pydantic import BaseModel, TypeAdapter
from pydantic_core import from_json

from conductor_providers.fastapi.store import RunChanges, RunStore, StoredRun

C = TypeVar("C")

log = logging.getLogger("conductor_providers.fastapi")

#: A code the router itself writes, beside the codes of the runs it serves.
_RouterCode = Literal["no_such_run", "already_answered", "run_complete", "run_not_resumable", "run_changed", "leg_failed"]

#: The English of each of the router's codes.
_ROUTER_PROBLEMS: dict[_RouterCode, str] = {
    "no_such_run": "There is no run by that id that the caller may see; check the id and who is calling.",
    "already_answered": "An answer to {waiting_id!r} already waits for the running leg to end; read the run once it has, and answer what still waits.",
    "run_complete": "The run is complete, so nothing in it waits for an answer; start a new run to ask again.",
    "run_not_resumable": "The run is {status}, and only a run whose last leg ended early is resumed; read the run to see what it needs.",
    "run_changed": "The run changed since it was read, so nothing was done; read it again and repeat the call if it still applies.",
    "leg_failed": "The run's worker failed before the leg could end; its operator finds why in the server log. Resume it to carry on.",
}

#: The codes the router itself writes, for a host translating by code to cover beside conductor's ``PROBLEM_CODES``.
ROUTER_PROBLEM_CODES: frozenset[str] = frozenset(_ROUTER_PROBLEMS)


def _problem(code: _RouterCode, **ctx: Any) -> Problem:
    """The problem ``code`` of the router's own, about the run as a whole, its message made from ``ctx``."""
    return Problem(type=code, msg=_ROUTER_PROBLEMS[code].format(**ctx), ctx=ctx)


class RefusedError(Exception):
    """A call the worker refused because of how the run stands.

    Attributes:
        problems: Why, each with one of the router's own codes.
    """

    def __init__(self, *problems: Problem) -> None:
        super().__init__(" ".join(problem.msg for problem in problems))
        self.problems = list(problems)


class NoSuchRunError(RefusedError):
    """No run has the id, or the caller may not see the one that has."""

    def __init__(self) -> None:
        super().__init__(_problem("no_such_run"))


#: The state of a run a leg is wanted for.
_RUNNING = Going("running")


class Outcome(BaseModel):
    """What a leg's end leaves for a reader, so that reading a run doesn't load it: a row's ``outcome``, as JSON.

    Attributes:
        waiting: What waits for a person.
        outputs: The graph's outputs by address once the run is complete; ``None`` until then.
    """

    waiting: list[Waiting] = []
    outputs: dict[str, Any] | None = None


class StoredState(BaseModel):
    """How a run stands and what was asked of its running leg: a row's ``state``, as JSON.

    It is small whatever the run holds, so a route that queues an answer or asks a cancel reads and writes it
    without touching the run's ``outcome``.

    Attributes:
        run_state: ``running`` while a leg is wanted; else how the last leg left the run, as the run itself says
            it, with its cause when it ended early.
        queued_answers: Answers that came while a leg ran, by waiting id; the run gets them when the leg ends.
        refused: Queued answers the run refused when a leg ended, by waiting id, each with its problems.
        cancel_asked: Whether someone asked the running leg to stop; a leg's end clears it.
    """

    run_state: RunState = _RUNNING
    queued_answers: dict[str, Any] = {}
    refused: dict[str, list[Problem]] = {}
    cancel_asked: bool = False

    @property
    def leg_wanted(self) -> bool:
        """Whether a leg is wanted for the run: a worker runs one, or the run waits for a worker to take it."""
        return self.run_state == _RUNNING

    def columns(self) -> RunChanges:
        """The row's ``status`` and ``state`` for this state. Nothing else writes either column, so each write of
        ``state`` writes the ``status`` it holds in the same statement. ``status`` is for the host's own queries:
        the worker and the routes read ``state`` and never ``status``."""
        return {"status": self.run_state.status, "state": self.model_dump_json()}


@dataclass(frozen=True)
class RunAsRead:
    """A run as read from the store: its row, with the row's state read from its JSON once.

    Attributes:
        row: The row.
        state: The row's ``state``.
    """

    row: StoredRun
    state: StoredState

    @classmethod
    def of(cls, row: StoredRun) -> "RunAsRead":
        """``row`` with its state read."""
        return cls(row, StoredState.model_validate_json(row.state))


class RunView(Outcome):
    """The body of ``GET /runs/{id}``: a run as its last leg left it, read without loading the run.

    Attributes:
        waiting: What waits for a person, as of the last leg's end; each ``id`` is what
            ``POST /runs/{id}/answers`` takes. Empty after a leg that raised, which ended with no run to ask.
        outputs: The graph's outputs by address once the run is complete, each as its type writes it (a table as its
            columns); ``None`` until then.
        id: The run's id, as ``POST /runs`` gave it.
        status: Where the run is: ``running`` while a leg runs, ``waiting`` for answers, ``complete``, or how its
            last leg ended early, which ``POST /runs/{id}/resume`` goes on from.
        refused: Answers sent while a leg ran that were refused when it ended, by waiting id, each with its problems.
        cause: Why the run's last leg ended early; ``None`` while it runs, waits or once complete.
    """

    id: str
    status: RunStatus
    refused: dict[str, list[Problem]] = {}
    cause: Problem | None = None

    @classmethod
    def of(cls, run: RunAsRead) -> "RunView":
        """The view of ``run``, from its state and its row's ``outcome``."""
        outcome = Outcome.model_validate_json(run.row.outcome)
        cause = run.state.run_state.cause if isinstance(run.state.run_state, EndedEarly) else None
        return cls(id=run.row.id, status=run.state.run_state.status, refused=run.state.refused, cause=cause, waiting=outcome.waiting, outputs=outcome.outputs)


@dataclass(frozen=True)
class _RowAfterALeg:
    """What a row holds once the end of a leg is written.

    Attributes:
        state: The row's state.
        changes: The row's ``run`` and ``outcome``, each where it changes.
        run_ended: Whether the leg ended, so that its ``run_end`` is stored; a leg handed back did not.
        leg_wanted: Whether a next leg is wanted: the leg was handed back, or the run took an answer queued during it.
    """

    state: StoredState
    changes: RunChanges
    run_ended: bool
    leg_wanted: bool


@dataclass(frozen=True)
class _LegEnd(ABC):
    """How a leg ended.

    Attributes:
        run_state: How the leg left the run.
        outcome: The row's ``outcome`` after the leg, as JSON.
        event: The leg's ``run_end`` event, as JSON.
    """

    run_state: RunState
    outcome: str
    event: str

    @abstractmethod
    def row_after(self, state: StoredState, *, shutting_down: bool) -> _RowAfterALeg:
        """What a row holds after this end, once the answers queued during the leg are dealt with.

        Args:
            state: The row's state as last read.
            shutting_down: Whether the worker is shutting down, so that a leg it stopped is handed back.

        Returns:
            The row's state and the columns that change with it, whether the leg ended, and whether a next leg
            is wanted.
        """


@dataclass(frozen=True)
class _Ended(_LegEnd):
    """A leg that ended with a run.

    Attributes:
        run: The run the leg ended with.
    """

    run: Run

    @classmethod
    def of(cls, end: RunEnd) -> "_Ended":
        """The end of a leg whose stream ended with ``end``: what waits in its run and, once complete, its
        outputs, each written by its type as the run writes it."""
        run = end.run
        outcome = Outcome(waiting=run.waiting, outputs=from_json(run.outputs_json()) if run.status == "complete" else None)
        return cls(run.state, outcome.model_dump_json(), end.model_dump_json(), run)

    def row_after(self, state: StoredState, *, shutting_down: bool) -> _RowAfterALeg:
        """What a row holds after this end, once the run has been given the answers queued during the leg.

        A leg its worker stopped to shut down, with no cancel asked of the run, is handed back: the run did not
        end, so the row keeps its state and its outcome and takes the run as a checkpoint. Otherwise each queued
        answer is given to the run on its own, as ``run.answer`` takes one whatever the run's status: one the run
        refuses stays in ``refused`` with its problems, and the ones it takes make the run the next leg starts
        from.
        """
        if shutting_down and self.run_state.status == "cancelled" and not state.cancel_asked:
            return _RowAfterALeg(state, {"run": self.run.to_json()}, run_ended=False, leg_wanted=True)
        run, refused, taken = self.run, dict(state.refused), False
        for waiting_id, value in state.queued_answers.items():
            try:
                run, taken = run.answer(waiting_id, value), True
            except InvalidCallError as error:
                refused[waiting_id] = error.errors()
        after = state.model_copy(update={"run_state": _RUNNING if taken else self.run_state, "queued_answers": {}, "refused": refused, "cancel_asked": False})
        return _RowAfterALeg(after, {"run": run.to_json(), "outcome": self.outcome}, run_ended=True, leg_wanted=taken)


@dataclass(frozen=True)
class _Raised(_LegEnd):
    """A leg that raised outside a node: it ended with no run."""

    def row_after(self, state: StoredState, *, shutting_down: bool) -> _RowAfterALeg:
        """What a row holds after this end: the run failed, also when the worker is shutting down, and the row
        keeps its last checkpoint and the answers queued during the leg, for the end of the next leg to deal with."""
        after = state.model_copy(update={"run_state": self.run_state, "cancel_asked": False})
        return _RowAfterALeg(after, {"outcome": self.outcome}, run_ended=True, leg_wanted=False)


_LEG_FAILED = _problem("leg_failed")

#: The end of a leg that raised: ``failed`` with the cause ``leg_failed``, nothing waiting and no outputs. Its event
#: is made without validation: a ``RunEnd`` requires the run it ended with, which it never writes, and a leg that
#: raised has none.
_RAISED = _Raised(EndedEarly("failed", _LEG_FAILED), Outcome().model_dump_json(), RunEnd.model_construct(status="failed", cause=_LEG_FAILED).model_dump_json())

#: The owner and lease of a row no leg is wanted for.
_NOBODYS: RunChanges = {"owner": None, "lease_until": None}

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
    """

    run_id: str
    rev: int
    state: StoredState
    stop: asyncio.Event = field(default_factory=asyncio.Event)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    released: bool = False


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
        self._holds: dict[_Hold, asyncio.Task[None]] = {}
        self._stopping = False

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
                raise RefusedError(*(_problem("already_answered", waiting_id=waiting_id) for waiting_id in sorted(twice)))
            return state.model_copy(update={"queued_answers": {**state.queued_answers, **answers}})

        run = await self._ask_of_a_running_leg(run, queued_with_these)
        if run.state.leg_wanted:
            return "queued"
        if run.state.run_state.status == "complete":
            raise RefusedError(_problem("run_complete"))
        row, nodes = run.row, self._nodes_for(self.started_by(run.row))
        answered = await asyncio.to_thread(lambda: _loaded(row, nodes).answer(answers).to_json())
        state = run.state.model_copy(update={"run_state": _RUNNING})
        if not await self._claim(run, state, answered=answered):
            raise RefusedError(_problem("run_changed"))
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
            raise RefusedError(_problem("run_not_resumable", status=run.state.run_state.status))
        state = run.state.model_copy(update={"run_state": _RUNNING})
        if not await self._claim(run, state, ceiling=max_items):
            raise RefusedError(_problem("run_changed"))

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

    async def take_over(self) -> None:
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
    async def running(self) -> AsyncGenerator[None]:
        """While entered, take over runs whose lease ran out, three times a lease. On leaving, stop every leg this
        worker runs and hand its run back: a cancel keeps the calls already made, so the next worker repeats none.
        A sweep that raises is logged and the next one runs."""
        async def sweep() -> None:
            while True:
                try:
                    await self.take_over()
                except Exception:
                    log.exception("the sweep for runs whose lease ran out failed; the next one tries again")
                await asyncio.sleep(self._lease / 3)

        sweeping = asyncio.create_task(sweep())
        try:
            yield
        finally:
            self._stopping = True
            sweeping.cancel()
            for hold in self._holds:
                hold.stop.set()
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

    async def _claim(self, run: RunAsRead, state: StoredState, *, answered: str | None = None, ceiling: int | None = None) -> bool:
        """Take a run for this worker if it is still at the ``rev`` read, and run its next leg. Of two workers, one
        wins.

        Args:
            run: The run as read.
            state: The state the row takes with the claim; its ``status`` and ``state`` are written from it.
            answered: The run with a caller's answers given, as JSON, for a claim that an answer makes; the row's
                ``run`` then. Left out, the row keeps the run it has.
            ceiling: A ceiling on the items of any one Series for the leg, where the run's own is to be raised.

        Returns:
            Whether this worker took the run.
        """
        written: RunChanges = {**state.columns(), **self._mine()}
        if answered is not None:
            written["run"] = answered
        if not await self._store.update(run.row.id, run.row.rev, written):
            return False
        self._run_legs(replace(run.row, **written, rev=run.row.rev + 1), state, ceiling)
        return True

    def _run_legs(self, row: StoredRun, state: StoredState, ceiling: int | None = None) -> None:
        """Run the legs of ``row``, as this worker has just written it, in the background; a cancel asked of it the
        first leg hears before its first unit."""
        hold = _Hold(row.id, row.rev, state)
        if state.cancel_asked:
            hold.stop.set()
        self._holds[hold] = asyncio.get_running_loop().create_task(self._legs(hold, row, ceiling))

    # -- a held run -------------------------------------------------------------------------------

    async def _legs(self, hold: _Hold, row: StoredRun, ceiling: int | None) -> None:
        """Run legs of a held run, renewing its lease, until the end of one gives the run up.

        A raise while a leg's end is written, which may come after its ``run_end`` is stored, is logged and writes
        no second end: the row stays ``running``, its lease runs out, and a worker goes on from the last checkpoint.
        """
        renewing = asyncio.create_task(self._renew(hold))
        try:
            while not hold.released:
                end = await self._leg(hold, row, ceiling)
                row, ceiling = await self._finish(hold, row, end), None
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

    async def _leg(self, hold: _Hold, row: StoredRun, ceiling: int | None) -> _LegEnd:
        """Run one leg of ``row`` and store its events and checkpoints.

        Returns:
            How the leg ended. A raise anywhere in it, outside a node, is logged and ends it as ``_RAISED``.
        """
        try:
            who = self.started_by(row)
            run = await asyncio.to_thread(_loaded, row, self._nodes_for(who))
            stream = run.stream(
                from_run=self._from_run(who), max_items=ceiling, timeout=self._leg_timeout,
                cancel=hold.stop, checkpoint_every=self._checkpoint_every,
            )
            async with contextlib.aclosing(stream):  # a raise in this loop ends the leg before its next unit
                async for event in stream:
                    if isinstance(event, RunCheckpoint):
                        await self._write(hold, {"run": await asyncio.to_thread(event.run.to_json)})
                    elif isinstance(event, RunEnd):
                        end: _LegEnd = await asyncio.to_thread(_Ended.of, event)
                    elif not await self._store.add_event(row.id, self.id, event.model_dump_json()):
                        hold.stop.set()  # another worker holds the run
            return end
        except Exception:
            log.exception("the leg of run %s failed", row.id)
            return _RAISED

    async def _finish(self, hold: _Hold, row: StoredRun, end: _LegEnd) -> StoredRun:
        """Write how a leg ended, decided from the row's state as read for each try of the write, so a cancel or an
        answer that lands meanwhile is never written over.

        A leg that is handed back stores no ``run_end``: the run did not end. Otherwise the leg's ``run_end`` is
        stored, once and before the row says the run stopped running, so a follower that reads that it stopped has
        the event to read. A run another worker took is left as it is, and the hold is released.

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
            state = hold.state
            after = await asyncio.to_thread(end.row_after, state, shutting_down=self._stopping and not end_stored)
            if after.run_ended and not end_stored:
                if not await self._store.add_event(hold.run_id, self.id, end.event):
                    hold.released = True  # another worker holds the run, so the write of the row would lose too
                    break
                end_stored = True
            goes_on = after.leg_wanted and not self._stopping
            holder = self._mine() if goes_on else _HANDED_BACK if after.leg_wanted else _NOBODYS
            changes: RunChanges = {**after.state.columns(), **after.changes, **holder}
            async with hold.lock:  # no renewal lands after the write that gives the run up
                if hold.state is state and await self._write_once(hold, changes):
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
