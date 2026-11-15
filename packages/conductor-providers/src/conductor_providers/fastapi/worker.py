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
it, and a host with routes of its own makes one and calls it the same way.
"""

import asyncio
import contextlib
import logging
import time
import uuid
from asyncio import Event as AsyncioEvent
from asyncio import Lock, Task
from collections.abc import AsyncGenerator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass, field, replace
from threading import Event
from typing import Any, Generic

from conductor import Graph, InvalidCallError, NodeRegistry, Run, RunProgress
from conductor.events import RunCheckpoint, RunEnd
from pydantic import TypeAdapter
from pydantic_core import from_json
from typing_extensions import TypeVar

from conductor_providers.fastapi.problems import (
    no_such_run,
    previous_run_unloadable,
    raised_outside_a_node,
    run_not_resumable,
)
from conductor_providers.fastapi.resume_end import HandedBack, Raised, ResumeEnd, end_of
from conductor_providers.fastapi.store import RunStore
from conductor_providers.fastapi.stored_run import (
    M,
    Outcome,
    RunChanges,
    RunView,
    StoredRun,
    StoredState,
)

#: The class of the run's context, the registry's ``context_type``.
X = TypeVar("X", default=None)

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
            is asked in its state, or another worker holds it; and set when the store raised and the run is given
            up. The same event is handed to the host's ``context``
            function, so a node can stop a slow call too. A new one, not set, is made for the next resume call when
            the end of one is written and no cancel is asked any more.
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


class Worker(Generic[M, X]):
    """Starts, answers, resumes and cancels stored runs for one process, on any number of processes sharing one store.

    ``run_router`` is a thin layer of routes over one worker. A host with its own routes, and its own say on who may
    start which run, makes a worker and calls it as the routes do: ``begin`` a run (or ``estimate`` one first),
    ``read`` it, check whose it is with ``metadata_of``, follow its ``events``,
    ``answer``, ``resume`` or ``cancel`` it, start it ``again`` with some inputs changed, and serve inside
    ``serving()``, which a lifespan enters. Each run is held under a lease while this worker resumes it, written to a
    checkpoint every few seconds, and taken over by another worker when its lease runs out.

    ``M`` is the type of a run's metadata: what the host keeps on a run, such as who started it and in which team.
    It is ``dict[str, str]`` unless you name a pydantic model with ``metadata_type``. The metadata is written to the
    run once, as JSON, when the run starts, and never changed. Every resume call, on any worker, reads it back and
    builds the run's registry and context from it, so keep in it nothing that changes. ``X`` is the class of the
    run's context, the registry's ``context_type``.

    Each method that acts on a run takes its id, reads the run itself, acts on it as it is, and returns the run as
    written. Every write is at the ``rev`` the method read: when another call wrote the run first, the method reads
    it again and acts on it as it is then.

    A worker that has never entered ``serving()`` raises ``RuntimeError`` from ``begin``, ``again``, and an
    ``answer``, ``resume`` or ``cancel`` that would make a run running. Once it has left ``serving()``, such a call is
    taken and its run handed back at once, for a worker that serves to take over.

    Attributes:
        metadata_type: The type of a run's metadata, as the worker was given it; the router's answers are declared
            with it.
        nodes_for: The function that gives the registry for a run's metadata, as the worker was given it; the routes
            describe it for ``/nodes`` and compile on it for ``/compile``.
    """

    def __init__(
        self,
        store: RunStore,
        *,
        nodes_for: Callable[[M], NodeRegistry[X]],
        context: Callable[[M, Event], X] | None = None,
        # mypy can't tie a parameter's default to the TypeVar's default, so it asks for this one ignore
        metadata_type: type[M] = dict[str, str],  # type: ignore[assignment]
        max_items: int | None = None,
        lease: float = 30.0,
        checkpoint_every: float = 5.0,
        resume_timeout: float | None = 300.0,
        poll: float = 0.1,
    ) -> None:
        """A worker on ``store``.

        Args:
            store: Where runs and their events are kept.
            nodes_for: The nodes a run's graph may use, from the run's metadata: ``lambda metadata: registry`` for
                one registry, or a registry per team. Called for every method call and every resume call, so it
                caches what it builds.
            context: The run's context, an instance of the registry's ``context_type``, from the run's metadata and
                the resume call's stop event, a ``threading.Event``; called for every resume call. The event is set
                when that call stops early because the run was cancelled, because another worker took the run over,
                or because the store raised and this worker gave the run up, so a node given it in the context can
                stop a slow call and raise (what it returns would be kept); a node reads it and never sets it. It is
                not set when this worker shuts down, since the calls going on then are waited for and kept, and the
                next resume call gets a new one. Left out for a registry that names no ``context_type``.
            metadata_type: The type of a run's metadata, ``M``: a pydantic model of yours, or ``dict[str, str]``.
                The metadata is kept on the run as JSON and read back by this type.
            max_items: The ceiling on the items of any one Series that a run starts with; ``None`` for none. A
                resume may raise it for that run, and the raised ceiling holds for that run from then on.
            lease: The seconds this worker holds a run between renewals.
            checkpoint_every: The seconds between saves of a running run.
            resume_timeout: The seconds one call of ``resume()`` may take before the run ends ``timed_out``; ``None``
                for no limit.
            poll: The seconds between a follower's reads of new events.
        """
        self._id = uuid.uuid4().hex
        self._store = store
        self._metadata: TypeAdapter[M] = TypeAdapter(metadata_type)
        # pydantic reads the metadata by the type given; mypy reads a subscript as a type, known only at run time
        self._views: type[RunView[M]] = RunView[metadata_type]  # type: ignore[valid-type]
        self.metadata_type = metadata_type
        self.nodes_for = nodes_for
        self._served = False
        self._context = context
        self._max_items = max_items
        self._lease = lease
        self._checkpoint_every = checkpoint_every
        self._resume_timeout = resume_timeout
        self._poll = poll
        self._holds: dict[_Hold, Task[None]] = {}
        self._handing_back = AsyncioEvent()

    # -- what a host calls ------------------------------------------------------------------------

    async def read(self, run_id: str) -> RunView[M]:
        """The run as a reader sees it: its status, what waits in it, its outputs once complete.

        Args:
            run_id: The run's id.

        Returns:
            The run's view, as it was when the run last stopped running.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``).
        """
        return self._view(await self._row(run_id))

    async def metadata_of(self, run_id: str) -> M:
        """A run's metadata, as it was written when the run started.

        Only the row's ``metadata`` column is read, so this stays cheap however much waits in the run. The router's
        access check calls it on every route under ``/runs/{run_id}``.

        Args:
            run_id: The run's id.

        Returns:
            The metadata, read back by the worker's ``metadata_type``.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``).
        """
        return self._metadata_in(await self._row(run_id))

    async def begin(
        self,
        graph: Any,
        inputs: Mapping[str, Any],
        metadata: M,
        *,
        previous: str | None = None,
    ) -> RunView[M]:
        """Start a run: store it and make its first resume call in the background.

        Args:
            graph: The graph record, as a ``GraphRecord``, as its JSON text or as the plain data a caller sent.
            inputs: Values for the graph's inputs, by name, every one the graph needs.
            metadata: The run's metadata, stored with it; the graph is compiled on its nodes.
            previous: The id of a run of this graph, or of an earlier version of it, that lends its work as
                ``graph.with_inputs(previous=...)`` takes it: a call that receives what a call of the same node
                received there is not made again. It is loaded on ``metadata``'s nodes. The new run starts at this
                worker's ``max_items``, not at a ceiling ``previous`` was resumed with; ``again`` keeps that one.

        Returns:
            The new run, as written.

        Raises:
            InvalidCallError: The graph can't run on ``metadata``'s nodes, or refuses the inputs, each with
                conductor's problems; else no run has the id ``previous`` (``no_such_run``), or it can't be loaded
                on those nodes (``previous_run_unloadable``, which says why). Start without ``previous`` then.
            RuntimeError: The worker has never served.
        """
        run = await self._ready(graph, inputs, metadata, previous)
        return await self._create(run.graph.record.model_dump_json(), metadata, run)

    async def estimate(
        self,
        graph: Any,
        inputs: Mapping[str, Any],
        metadata: M,
        *,
        previous: str | None = None,
    ) -> RunProgress:
        """How many calls the run ``begin`` would start will make, from ``run.resume(only_costs_nothing=True)``,
        which calls only the nodes that cost nothing; nothing is stored.

        The call is made as a stored run's is: with the metadata's context and this worker's ``resume_timeout`` and
        ``max_items``. The context is given a stop event that is never set, since nothing cancels an estimate.

        Args:
            graph: The graph record, as a ``GraphRecord``, as its JSON text or as the plain data a caller sent.
            inputs: Values for the graph's inputs, by name.
            metadata: The metadata the run would have, on whose nodes the graph is compiled.
            previous: The id of a run whose work the run would reuse, as ``begin`` takes it.

        Returns:
            The run's progress after that call.

        Raises:
            InvalidCallError: As ``begin`` refuses the same call.
        """
        run = await self._ready(graph, inputs, metadata, previous)

        def estimated() -> RunProgress:
            context = self._context_for(metadata, Event())
            return run.resume(
                context=context, timeout=self._resume_timeout, only_costs_nothing=True
            ).progress

        return await asyncio.to_thread(estimated)

    async def again(self, run_id: str, inputs: Mapping[str, Any], metadata: M) -> RunView[M]:
        """Start a run of a run's graph with some inputs changed, as ``run.with_inputs`` makes one; a call that
        receives what it received before isn't made again.

        The new run has the run's graph, the inputs it had except those given, and its ceiling on items, a raised
        one included. To start another graph, such as an edited one, from a run's work, ``begin`` with
        ``previous``.

        Args:
            run_id: The id of the run to start from.
            inputs: The inputs to change, by name; the rest stay as the run had them.
            metadata: The new run's metadata, stored with it; the run is loaded on its nodes.

        Returns:
            The new run, as written.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``), or the graph can't run on ``metadata``'s nodes,
                or refuses the inputs.
            RuntimeError: The worker has never served.
        """
        row, nodes = await self._row(run_id), self.nodes_for(metadata)
        again = await asyncio.to_thread(lambda: _loaded(row, nodes).with_inputs(inputs))
        return await self._create(row.graph, metadata, again)

    async def answer(self, run_id: str, answers: Mapping[str, Any]) -> RunView[M]:
        """Answer what waits in a run, by waiting id.

        While the run is running, the answers are queued, as one call sent them, and given to the run when its
        resume call ends. Otherwise the run takes them as ``run.answer`` does, whatever its status, and it is
        resumed. An answer to a question that nothing waits on any more, such as one in a complete run, is refused
        by conductor as ``not_waiting``. No answers at all change nothing, as ``run.answer({})`` doesn't, so nothing
        is written.

        When another call takes the run between this call's read and its write, the run is read again, and the
        answers are given to it as it is then: queued, if the other call started a resume.

        Args:
            run_id: The run's id.
            answers: An answer for each waiting call, by its waiting id.

        Returns:
            The run as written: ``running`` whether the answers were queued or resumed the run.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``), or the run refuses an answer; every answer it
                refuses is listed.
            RuntimeError: The run is not running, so the answers would resume it, and the worker has never served.
        """
        row = await self._row(run_id)
        if not answers:
            return self._view(row)
        while True:
            row = await self._ask_of_a_running_run(row, lambda state: state.queued(answers))
            state = _state_of(row)
            if state.is_running:
                return self._view(row)
            loaded, nodes = row, self.nodes_for(self._metadata_in(row))
            answered = await asyncio.to_thread(
                lambda: _loaded(loaded, nodes).answer(answers).to_json()
            )
            claimed = await self._claim(
                row, state.answered(answers.keys()).running(), changed=answered
            )
            if claimed is not None:
                return self._view(claimed)
            row = await self._row(run_id)

    async def resume(self, run_id: str, max_items: int | None = None) -> RunView[M]:
        """Resume a run that ended early; no node call that finished is made again.

        When another call takes the run between this call's read and its write, the run is read again: it is
        resumed if it still ended early, and refused if it is running now.

        Args:
            run_id: The run's id.
            max_items: A higher ceiling on the items of any one Series, for a run that stopped at its ceiling. The
                run is stored with it as the run is claimed, so every later resume call has it.

        Returns:
            The run as written, ``running``.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``), the run did not end early
                (``run_not_resumable``), or, with ``max_items``, the stored run can't be loaded to raise its
                ceiling.
            RuntimeError: The worker has never served.
        """
        while True:
            row = await self._row(run_id)
            state = _state_of(row)
            if state.cause is None:
                raise InvalidCallError([run_not_resumable(status=state.status)])
            raised = None
            if max_items is not None:
                nodes = self.nodes_for(self._metadata_in(row))
                raised = await asyncio.to_thread(
                    lambda: _loaded(row, nodes).with_max_items(max_items).to_json()
                )
            claimed = await self._claim(row, state.running(), changed=raised)
            if claimed is not None:
                return self._view(claimed)

    async def cancel(self, run_id: str) -> RunView[M]:
        """Ask a running or waiting run to stop before its next node call, so that it ends ``cancelled``.

        A running run is asked to stop: at once when this worker resumes it, at the next renewal of its lease when
        another does. The cancel is written to the row's state, and a resume call is stopped only by what its row
        says: this worker reads the row again here, and another worker reads it at its next renewal.

        A waiting run is claimed with the cancel already asked, as ``resume`` claims a run, and resumed: its resume
        call ends ``cancelled`` before it calls any node, as ``run.stream(cancel=...)`` ends a waiting run, and its
        questions still wait. Like any resume call it loads the run with its metadata's registry and context, so a
        run that can't be resumed ends ``failed`` instead, as it would on its next answer.

        A run that is complete or already ended early has nothing to cancel, and is left as it is. When another
        call changes the run between this call's read and its write, the run is read again and the cancel goes by
        what it is then.

        Args:
            run_id: The run's id.

        Returns:
            The run as written: ``running`` when the cancel was asked, else the run as it is.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``).
            RuntimeError: The run is waiting, so the cancel would resume it, and the worker has never served.
        """

        def asked_to_stop(state: StoredState) -> StoredState:
            return state.model_copy(update={"cancel_asked": True})

        row = await self._row(run_id)
        while True:
            row = await self._ask_of_a_running_run(row, asked_to_stop)
            state = _state_of(row)
            if state.is_running:
                for hold in [hold for hold in self._holds if hold.run_id == run_id]:
                    async with hold.lock:
                        await self._read_again(hold)
                return self._view(row)
            if state.status != "waiting":
                return self._view(row)
            claimed = await self._claim(row, asked_to_stop(state.running()))
            if claimed is not None:
                return self._view(claimed)
            row = await self._row(run_id)

    async def events(self, run_id: str, after: int = -1) -> AsyncGenerator[tuple[int, str]]:
        """The events of a run as they are stored, from whichever worker resumes it, until the run is not running.

        A row stops saying ``running`` only after the ``run_end`` of its resume call is stored. So the row is read
        only once a ``run_end`` was the last event, or when there was no event at all, and otherwise events alone
        are read.

        Args:
            run_id: The run's id.
            after: The number of the last event the follower has; ``-1`` for none.

        Yields:
            Each event numbered above ``after``, in order, as ``(number, body)``.
        """
        may_have_ended = True
        while True:
            new = await self._store.events(run_id, after)
            for number, body in new:
                after = number
                yield number, body
            if new:
                may_have_ended = from_json(new[-1][1])["type"] == "run_end"
            elif may_have_ended and not _state_of(await self._row(run_id)).is_running:
                for event in await self._store.events(
                    run_id, after
                ):  # stored before the row said so
                    yield event
                return
            else:  # the row is written after its run_end, so it may still say running for a poll or two
                await asyncio.sleep(self._poll)

    async def _sweep(self) -> None:
        """Go on with every run whose lease ran out: its worker died or handed it back. A run that can't be taken
        (its row can't be read, the store fails) is logged and left for the next call, and the others are taken."""
        held = {hold.run_id for hold in self._holds}
        for row in await self._store.expired(time.time()):
            if row.id not in held:
                try:
                    await self._claim(row, _state_of(row))
                except Exception:
                    log.exception(
                        "run %s, whose lease ran out, was not taken over; the next sweep tries again",
                        row.id,
                    )

    @asynccontextmanager
    async def serving(self) -> AsyncGenerator[None]:
        """The context a worker serves in: a lifespan enters it.

        A worker that has never entered it raises ``RuntimeError`` from a method that would make a run running
        (``begin``, ``again``, ``answer`` and ``resume`` of a run that is not running, ``cancel`` of a waiting one),
        since no sweep of its own would ever take over its runs. While entered, runs whose lease ran out are taken over, three times a lease; a sweep that raises
        is logged and the next one runs. On leaving, every resume call this worker has going is stopped and its run
        handed back, and so is a run begun or taken after that, at once: another worker that serves takes it over. A run handed back keeps the node calls already made, so the next
        worker repeats none of them once the last checkpoint is written.

        Yields:
            Nothing; the worker serves until the context is left.
        """

        async def keep_sweeping() -> None:
            while True:
                try:
                    await self._sweep()
                except Exception:
                    log.exception(
                        "the sweep for runs whose lease ran out failed; the next one tries again"
                    )
                await asyncio.sleep(self._lease / 3)

        self._handing_back.clear()
        self._served = True
        sweeping = asyncio.create_task(keep_sweeping())
        try:
            yield
        finally:
            self._handing_back.set()
            sweeping.cancel()
            # a run begun or taken while the others were handed back is handed back too
            while self._holds:
                await asyncio.gather(*self._holds.values())

    # -- taking a run -----------------------------------------------------------------------------

    def _check_served(self) -> None:
        """Refuse to make a run running on a worker that never entered ``serving()``. A worker that served and left
        it takes the run and hands it back at once, for a worker that serves to take over.

        Raises:
            RuntimeError: The worker has never served.
        """
        if not self._served:
            raise RuntimeError(
                "the worker has never served, so no sweep of its own would take over its runs; enter "
                "`async with worker.serving():` (a server's lifespan does) before it begins, answers, resumes or "
                "cancels a run"
            )

    async def _ready(
        self, graph: Any, inputs: Mapping[str, Any], metadata: M, previous: str | None
    ) -> Run[Any]:
        """The run ``begin`` starts and ``estimate`` counts: the graph compiled on the metadata's nodes, with its
        inputs, and the work of ``previous`` lent to it.

        The graph's own problems and the inputs' are refused before the previous run is loaded.

        Raises:
            InvalidCallError: The graph or the inputs are refused, no run has the id ``previous``
                (``no_such_run``), or it can't be loaded on these nodes (``previous_run_unloadable``).
        """
        nodes = self.nodes_for(metadata)
        lends = None if previous is None else await self._row(previous)

        def ready() -> Run[Any]:
            compiled = Graph.compile(graph, nodes)
            run = compiled.with_inputs(inputs, max_items=self._max_items)
            if lends is None:
                return run
            try:
                lent = _loaded(lends, nodes)
            except InvalidCallError as refusal:
                reason = "; ".join(problem.msg for problem in refusal.problems)
                raise InvalidCallError(
                    [previous_run_unloadable(run_id=lends.id, reason=reason)]
                ) from refusal
            return compiled.with_inputs(inputs, max_items=self._max_items, previous=lent)

        return await asyncio.to_thread(ready)

    def _mine(self) -> RunChanges:
        """The owner and lease of a row this worker holds."""
        return {"owner": self._id, "lease_until": time.time() + self._lease}

    async def _row(self, run_id: str) -> StoredRun:
        """A run's row, as the store has it.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``).
        """
        row = await self._store.get(run_id)
        if row is None:
            raise InvalidCallError([no_such_run()])
        return row

    def _view(self, row: StoredRun) -> RunView[M]:
        """The run in ``row`` as a reader sees it, from its state, its outcome and its metadata."""
        state, outcome = _state_of(row), Outcome.model_validate_json(row.outcome)
        return self._views(
            id=row.id,
            metadata=self._metadata_in(row),
            status=state.status,
            cause=state.cause,
            waiting=outcome.waiting,
            outputs=outcome.outputs,
            progress=outcome.progress,
            refused=state.refused,
        )

    def _metadata_in(self, row: StoredRun) -> M:
        """The metadata a row keeps, read from its JSON."""
        return self._metadata.validate_json(row.metadata)

    def _context_for(self, metadata: M, stop: Event) -> X | None:
        """The context of a resume call: ``context(metadata, stop)``, or ``None`` when the worker was given none."""
        return None if self._context is None else self._context(metadata, stop)

    async def _create(self, graph: str, metadata: M, run: Run[Any]) -> RunView[M]:
        """Store a run, held by this worker, and make its first resume call in the background.

        Args:
            graph: The run's graph, as JSON.
            metadata: The run's metadata.
            run: The run with its inputs, ready.

        Returns:
            The run, as written.

        Raises:
            RuntimeError: The worker has never served.
        """
        self._check_served()
        saved, outcome = await asyncio.to_thread(
            lambda: (run.to_json(), Outcome.so_far(run).model_dump_json())
        )
        state = StoredState.started()
        written: RunChanges = {"run": saved, **state.columns(), "outcome": outcome, **self._mine()}
        row = StoredRun(
            id=uuid.uuid4().hex,
            graph=graph,
            metadata=self._metadata.dump_json(metadata).decode(),
            rev=0,
            **written,
        )
        await self._store.create(row)
        self._resume_in_background(row, state)
        return self._view(row)

    async def _claim(
        self, row: StoredRun, state: StoredState, *, changed: str | None = None
    ) -> StoredRun | None:
        """Take a run for this worker if it is still at the ``rev`` read, and resume it in the background. Of two
        workers, one wins.

        Args:
            row: The run's row as read.
            state: The state the row takes with the claim; its ``status`` and ``state`` are written from it.
            changed: The run as the call that claims it changed it, as JSON (answers given, or its
                ceiling raised); the row's ``run`` then. Left out, the row keeps the run it has.

        Returns:
            The row as written when this worker took the run; ``None`` when another call wrote it first.

        Raises:
            RuntimeError: The worker has never served.
        """
        self._check_served()
        written: RunChanges = {**state.columns(), **self._mine()}
        if changed is not None:
            written["run"] = changed
        if not await self._store.update(row.id, row.rev, written):
            return None
        claimed = replace(row, **written, rev=row.rev + 1)
        self._resume_in_background(claimed, state)
        return claimed

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
        self._holds[hold] = asyncio.get_running_loop().create_task(
            self._resume_while_held(hold, row)
        )

    # -- a held run -------------------------------------------------------------------------------

    async def _resume_while_held(self, hold: _Hold, row: StoredRun) -> None:
        """Make one resume call after another on a held run, renewing its lease, until the hold is released.

        The hold is released when a resume call ends and the run is not to be resumed at once (no queued answer
        made it running again), or when another worker took the run.

        What a host's code raises and what conductor refuses are handled where they happen (``_resume_steps``,
        ``Ended.row_after``), so an exception that reaches here is the store's, raised when an event, a checkpoint
        or the end of a resume call was written. It is the host's outage and never fails the run: it is logged,
        no more nodes are started, the stop event is set so that a node call still going on can stop, and the run
        is left as it is in the store, ``running``. Its lease runs out, and a worker resumes it from the last
        checkpoint; a node call that is still on a thread then may be made again while it runs.

        Args:
            hold: The hold on the run.
            row: The row as this worker wrote it.
        """
        renewing = asyncio.create_task(self._renew(hold))
        try:
            while not hold.released:
                row = await self._finish(hold, row, await self._resume_once(hold, row))
        except Exception:
            hold.stop.set()  # another worker makes the calls going on again
            log.exception(
                "the store raised while run %s was resumed; a worker resumes the run once its lease runs out",
                row.id,
            )
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
                log.error(
                    "the store did not answer a renewal of the lease of run %s; the next is made at once",
                    hold.run_id,
                )
                wait = 0.0
            except Exception:
                log.exception(
                    "the lease of run %s was not renewed; the next renewal tries again", hold.run_id
                )

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
        async with contextlib.aclosing(
            self._resume_steps(hold, row)
        ) as steps:  # a raise in this loop ends the resume call before its next node call
            async for step in steps:
                match step:
                    case ResumeEnd():
                        end = step
                    case str():
                        if not await self._store.add_event(row.id, self._id, step):
                            hold.stop.set()  # another worker holds the run
                    case _:
                        await self._write(hold, step)
        return end

    async def _resume_steps(
        self, hold: _Hold, row: StoredRun
    ) -> AsyncGenerator[str | RunChanges | ResumeEnd]:
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
            run that can't be read or written), the first of them where there are several; else
            ``raised_outside_a_node``.
        """
        # as the run last stopped: a checkpoint changes its progress only
        stopped = Outcome.model_validate_json(row.outcome)
        held = Outcome.empty()  # the run as the row holds it; nothing, until the run is loaded
        try:
            metadata = self._metadata_in(row)
            run = await asyncio.to_thread(_loaded, row, self.nodes_for(metadata))
            held = await asyncio.to_thread(Outcome.so_far, run)
            stream = run.stream(
                context=self._context_for(metadata, hold.stop),
                timeout=self._resume_timeout,
                cancel=hold.stop,
                hand_back=self._handing_back,
                checkpoint_every=self._checkpoint_every,
            )
            async with contextlib.aclosing(stream):
                async for event in stream:
                    if isinstance(event, RunCheckpoint):
                        saved = await asyncio.to_thread(event.run.to_json)
                        held = await asyncio.to_thread(Outcome.so_far, event.run)
                        yield {
                            "run": saved,
                            "outcome": stopped.model_copy(
                                update={"progress": held.progress}
                            ).model_dump_json(),
                        }
                    elif isinstance(event, RunEnd):
                        yield await asyncio.to_thread(end_of, event, held)
                    else:
                        yield event.model_dump_json()
        except InvalidCallError as refusal:
            every = "; ".join(
                f"{problem.type} at {'.'.join(map(str, problem.loc))}: {problem.msg}"
                for problem in refusal.problems
            )
            log.error(
                "resuming run %s was refused: %s", row.id, every, exc_info=True
            )  # the cause is the first problem only
            yield Raised.refused(refusal, held)
        except Exception as error:
            log.exception("resuming run %s failed", row.id)
            yield Raised.because(raised_outside_a_node(error=type(error).__name__), held)

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
        if end.run_end is not None and not await self._store.add_event(
            hold.run_id, self._id, end.run_end
        ):
            hold.released = (
                True  # another worker holds the run, so the write of the row would lose too
            )
            return row
        # so no renewal moves the rev between this read and what is taken in from it
        async with hold.lock:
            await self._read_again(hold)
        while not hold.released:
            reads = hold.reads
            after = await asyncio.to_thread(end.row_after, hold.state)
            goes_on = after.state.is_running and not self._handing_back.is_set()
            if goes_on:
                holder = self._mine()
            elif after.state.is_running:
                holder = _HANDED_BACK
            else:
                holder = _UNHELD
            changes: RunChanges = {**after.state.columns(), **after.changes, **holder}
            async with hold.lock:  # no renewal lands after the write that gives the run up
                if hold.reads == reads and await self._write_once(
                    hold, changes
                ):  # decided from the state as it still is
                    hold.state, hold.released = after.state, not goes_on
                    if not after.state.cancel_asked:
                        # the cancel that stopped this resume call does not stop the next; an attempt of this one
                        # still on its thread past its attempt_timeout goes on seeing it set
                        hold.stop = Event()
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
        row = await self._row(hold.run_id)
        hold.reads += 1
        hold.rev, hold.state = row.rev, _state_of(row)
        hold.released = row.owner != self._id
        if hold.released or hold.state.cancel_asked:
            hold.stop.set()

    # -- a run another worker may hold ------------------------------------------------------------

    async def _ask_of_a_running_run(
        self, row: StoredRun, asked: Callable[[StoredState], StoredState]
    ) -> StoredRun:
        """Write what a caller asks of a running run (an answer to queue, a cancel) into the row's state. It is
        tried for as long as the run is running, reading the row again each time another writer moved its ``rev``
        on first.

        Args:
            row: The run's row as read.
            asked: The state to write, made from the state as read for each try.

        Returns:
            The row after the call: still running, with what was asked written; or as last read, not running, with
            nothing written.
        """
        while (state := _state_of(row)).is_running:
            changes = asked(state).columns()
            if await self._store.update(row.id, row.rev, changes):
                return replace(row, **changes, rev=row.rev + 1)
            row = await self._row(row.id)
        return row


def _state_of(row: StoredRun) -> StoredState:
    """The state a row keeps, read from its JSON."""
    return StoredState.model_validate_json(row.state)


def _loaded(row: StoredRun, nodes: NodeRegistry[Any]) -> Run[Any]:
    """A stored run, loaded.

    Args:
        row: The run's row.
        nodes: The nodes its graph is loaded on.

    Returns:
        The run as the row's ``run`` saved it.
    """
    return Run.from_json(row.run, Graph.compile(row.graph, nodes))
