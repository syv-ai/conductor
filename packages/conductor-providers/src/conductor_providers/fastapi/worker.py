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
from typing import Any, Generic, Literal, TypeVar, overload

from conductor import Graph, InvalidCallError, NodeRegistry, Run, RunProgress
from conductor.events import RunCheckpoint, RunEnd
from pydantic import TypeAdapter
from pydantic_core import from_json

from conductor_providers.fastapi.problems import (
    no_such_run,
    previous_run_unloadable,
    raised_outside_a_node,
    run_not_resumable,
)
from conductor_providers.fastapi.resume_end import HandedBack, Raised, ResumeEnd, end_of
from conductor_providers.fastapi.store import RunStore
from conductor_providers.fastapi.stored_run import (
    Outcome,
    RunAsRead,
    RunChanges,
    StoredRun,
    StoredState,
)

C = TypeVar("C")
#: The class of the run's context that the registry names.
X = TypeVar("X")

log = logging.getLogger("conductor_providers.fastapi")

#: The owner and lease of a row whose run is not running.
_UNHELD: RunChanges = {"owner": None, "lease_until": None}

#: The owner and lease of a row handed back: its lease already run out, so the next sweep of any worker takes it.
_HANDED_BACK: RunChanges = {"owner": None, "lease_until": 0.0}


def _no_context(caller: object, stop: Event) -> None:
    """The run's context when the host gives none: none, for a registry that names no ``context_type``."""
    return None


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


class Worker(Generic[C]):
    """Starts, answers, resumes and cancels stored runs for one process, on any number of processes sharing one store.

    ``run_router`` is a thin layer of routes over one worker. A host with its own routes, and its own say on who may
    start which run, makes a worker and calls it as the routes do: ``begin`` a run (or ``estimate`` one first),
    ``read`` it, show it with ``run.view()``, check whom it is for with ``caller_of``, follow its ``events``,
    ``answer``, ``resume`` or ``cancel`` it, start it ``again`` with some inputs changed, and serve inside
    ``serving()``, which a lifespan enters. Each run is held under a lease while this worker resumes it, written to a
    checkpoint every few seconds, and taken over by another worker when its lease runs out.

    ``C`` is the type of a run's caller: what the run was admitted with, which is who started it and whatever else
    holds for the run's whole life, such as a tenant or the graph they were let run. The caller is written to the
    run once, as JSON, when the run is created, and never changed; every resume call, on any worker, reads it back
    and makes the run's registry and context from it, so a host keeps its own data about a run there and needs no
    hook for it. Hold nothing in it that changes: a run keeps it as it was when the run began. Every worker on one
    store takes over any run in it, so they all read the caller, and make the registry and context from it, the
    same way.

    Each method that takes a run takes it as the caller read it (``read``). A method that makes a run running
    (``answer`` on a run that is not running, ``resume``, ``cancel`` on a waiting run) writes only if the run is
    still at that ``rev``. What is asked of a running run (an answer to queue, a cancel) is written however the
    ``rev`` moved, for as long as the run is running.

    A worker that has never entered ``serving()`` raises ``RuntimeError`` from ``begin``, ``again``, and an
    ``answer``, ``resume`` or ``cancel`` that would make a run running. Once it has left ``serving()``, such a call is
    taken and its run handed back at once, for a worker that serves to take over.

    Attributes:
        nodes_for: The function that gives a caller's registry, as the worker was given it; the routes describe it
            for ``/nodes`` and compile on it for ``/compile``.
    """

    @overload
    def __init__(
        self,
        *,
        store: RunStore,
        caller_type: type[C],
        nodes_for: Callable[[C], NodeRegistry[X]],
        context: Callable[[C, Event], X],
        max_items: int | None = ...,
        lease: float = ...,
        checkpoint_every: float = ...,
        resume_timeout: float | None = ...,
        poll: float = ...,
    ) -> None: ...

    @overload
    def __init__(
        self,
        *,
        store: RunStore,
        caller_type: type[C],
        nodes_for: Callable[[C], NodeRegistry[None]],
        max_items: int | None = ...,
        lease: float = ...,
        checkpoint_every: float = ...,
        resume_timeout: float | None = ...,
        poll: float = ...,
    ) -> None: ...

    @overload
    def __init__(
        self: "Worker[None]",
        *,
        store: RunStore,
        caller_type: None,
        nodes_for: Callable[[None], NodeRegistry[X]],
        context: Callable[[None, Event], X],
        max_items: int | None = ...,
        lease: float = ...,
        checkpoint_every: float = ...,
        resume_timeout: float | None = ...,
        poll: float = ...,
    ) -> None: ...

    @overload
    def __init__(
        self: "Worker[None]",
        *,
        store: RunStore,
        caller_type: None,
        nodes_for: Callable[[None], NodeRegistry[None]],
        max_items: int | None = ...,
        lease: float = ...,
        checkpoint_every: float = ...,
        resume_timeout: float | None = ...,
        poll: float = ...,
    ) -> None: ...

    def __init__(
        self,
        *,
        store: RunStore,
        caller_type: Any,
        nodes_for: Callable[[Any], NodeRegistry[Any]],
        context: Callable[[Any, Event], Any] = _no_context,
        max_items: int | None = None,
        lease: float = 30.0,
        checkpoint_every: float = 5.0,
        resume_timeout: float | None = 300.0,
        poll: float = 0.1,
    ) -> None:
        """A worker on ``store``.

        Args:
            store: Where runs and their events are kept.
            caller_type: The type of a run's caller, ``C``; the caller is kept on the run as its JSON, read back by
                this type. ``None`` for a host with no logins, whose every run's caller is ``None``.
            nodes_for: The nodes a run's graph may use, from its caller: ``lambda caller: registry`` for one
                registry, or a registry per tenant. Called for every method call and every resume call, so it caches
                what it builds.
            context: The run's context, an instance of the registry's ``context_type``, from its caller and the
                resume call's stop event, a ``threading.Event``; called for every resume call. The event is set when
                that call stops early because the run was cancelled, because another worker took the run over, or
                because the store raised and this worker gave the run up, so a node given it in the context can stop
                a slow call and raise (what it returns would be kept); a node reads it and never sets it. It is not
                set when this worker shuts down, since the calls going on then are waited for and kept, and the next
                resume call gets a new one. Left out for a registry that names no ``context_type``.
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
        self._caller_type: TypeAdapter[C] = TypeAdapter(caller_type)
        self.nodes_for: Callable[[C], NodeRegistry[Any]] = nodes_for
        self._served = False
        self._context: Callable[[C, Event], Any] = context
        self._max_items = max_items
        self._lease = lease
        self._checkpoint_every = checkpoint_every
        self._resume_timeout = resume_timeout
        self._poll = poll
        self._holds: dict[_Hold, Task[None]] = {}
        self._handing_back = AsyncioEvent()

    # -- what a host calls ------------------------------------------------------------------------

    def caller_of(self, run: RunAsRead) -> C:
        """A run's caller: what it was admitted with, as it was when the run began.

        Args:
            run: The run as read.

        Returns:
            The caller, read from the JSON the run keeps.
        """
        return self._caller_type.validate_json(run._row.caller)

    async def read(self, run_id: str) -> RunAsRead:
        """The run as the store has it.

        Args:
            run_id: The run's id.

        Returns:
            The run as read.

        Raises:
            InvalidCallError: The store has no run by that id (``no_such_run``).
        """
        row = await self._store.get(run_id)
        if row is None:
            raise InvalidCallError([no_such_run()])
        return RunAsRead._of(row)

    async def begin(
        self,
        graph: Any,
        inputs: Mapping[str, Any],
        caller: C,
        *,
        previous: RunAsRead | None = None,
    ) -> str:
        """Start a run: store it and make its first resume call in the background.

        Args:
            graph: The graph record, as a ``GraphRecord``, as its JSON text or as the plain data a caller sent.
            inputs: Values for the graph's inputs, by name, every one the graph needs.
            caller: The run's caller, stored with it; the graph is compiled on its nodes.
            previous: A run of this graph, or of an earlier version of it, as read, that lends its work as
                ``graph.with_inputs(previous=...)`` takes it: a call that receives what a call of the same node
                received there is not made again. It is loaded on ``caller``'s nodes. The new run starts at this
                worker's ``max_items``, not at a ceiling ``previous`` was resumed with; ``again`` keeps that one.

        Returns:
            The run's id.

        Raises:
            InvalidCallError: The graph can't run on ``caller``'s nodes, or refuses the inputs, each with
                conductor's problems; else ``previous`` can't be loaded on ``caller``'s nodes
                (``previous_run_unloadable``, which says why). Start without ``previous`` then.
            RuntimeError: The worker has never served.
        """
        nodes = self.nodes_for(caller)

        def ready() -> Run[Any]:
            compiled = Graph.compile(graph, nodes)
            # the graph's own problems and the inputs' are refused before the previous run is loaded
            run = compiled.with_inputs(inputs, max_items=self._max_items)
            if previous is None:
                return run
            try:
                lent = _loaded(previous._row, nodes)
            except InvalidCallError as refusal:
                raise InvalidCallError(
                    [
                        previous_run_unloadable(
                            run_id=previous.id,
                            reason="; ".join(problem.msg for problem in refusal.problems),
                        )
                    ]
                ) from refusal
            return compiled.with_inputs(inputs, max_items=self._max_items, previous=lent)

        run = await asyncio.to_thread(ready)
        return await self._create(run.graph.record.model_dump_json(), caller, run)

    async def estimate(self, graph: Any, inputs: Mapping[str, Any], caller: C) -> RunProgress:
        """How many calls a run of ``graph`` on ``inputs`` will make, from ``run.resume(only_costs_nothing=True)``,
        which calls only the nodes that cost nothing; nothing is stored.

        The call is made as a stored run's is: with the caller's context and this worker's ``resume_timeout`` and
        ``max_items``. The context is given a stop event that is never set, since nothing cancels an estimate.

        Args:
            graph: The graph record, as a ``GraphRecord``, as its JSON text or as the plain data a caller sent.
            inputs: Values for the graph's inputs, by name.
            caller: The caller the run would have, on whose nodes the graph is compiled.

        Returns:
            The run's progress after that call.

        Raises:
            InvalidCallError: The graph can't run on ``caller``'s nodes, or refuses the inputs.
        """
        nodes = self.nodes_for(caller)

        def estimated() -> RunProgress:
            ready = Graph.compile(graph, nodes).with_inputs(inputs, max_items=self._max_items)
            return ready.resume(
                context=self._context(caller, Event()),
                timeout=self._resume_timeout,
                only_costs_nothing=True,
            ).progress

        return await asyncio.to_thread(estimated)

    async def again(self, run: RunAsRead, inputs: Mapping[str, Any], caller: C) -> str:
        """Start a run of ``run``'s graph with some inputs changed, as ``run.with_inputs`` makes one; a call that
        receives what it received before isn't made again.

        The new run has ``run``'s graph, the inputs it had except those given, and its ceiling on items, a raised
        one included. To start another graph, such as an edited one, from a run's work, ``begin`` with
        ``previous``.

        Args:
            run: The run to start from, as read.
            inputs: The inputs to change, by name; the rest stay as ``run`` had them.
            caller: The new run's caller, stored with it; ``run`` is loaded on its nodes.

        Returns:
            The new run's id.

        Raises:
            InvalidCallError: The graph can't run on ``caller``'s nodes, or refuses the inputs.
            RuntimeError: The worker has never served.
        """
        nodes = self.nodes_for(caller)
        again = await asyncio.to_thread(lambda: _loaded(run._row, nodes).with_inputs(inputs))
        return await self._create(run._row.graph, caller, again)

    async def answer(
        self, run: RunAsRead, answers: Mapping[str, Any]
    ) -> Literal["started", "queued"]:
        """Answer what waits in a run, by waiting id.

        While the run is running, the answers are queued, as one call sent them, and given to the run when its
        resume call ends. Otherwise the run takes them as ``run.answer`` does, whatever its status, and it is
        resumed. An answer to a question that nothing waits on any more, such as one in a complete run, is refused
        by conductor as ``not_waiting``.

        When another call takes the run between this call's read and its write, the run is read again, and the
        answers are given to it as it is then: queued, if the other call started a resume.

        Args:
            run: The run as read.
            answers: An answer for each waiting call, by its waiting id.

        Returns:
            ``queued`` when the run was running, else ``started``.

        Raises:
            InvalidCallError: The run refuses an answer; every answer it refuses is listed.
            RuntimeError: The run is not running, so the answers would resume it, and the worker has never served.
        """
        while True:
            run = await self._ask_of_a_running_run(run, lambda state: state.queued(answers))
            if run._state.is_running:
                return "queued"
            row, nodes = run._row, self.nodes_for(self.caller_of(run))
            answered = await asyncio.to_thread(
                lambda: _loaded(row, nodes).answer(answers).to_json()
            )
            state = run._state.answered(answers.keys()).running()
            if await self._claim(run, state, changed=answered):
                return "started"
            run = await self.read(run.id)

    async def resume(self, run: RunAsRead, max_items: int | None = None) -> None:
        """Resume a run that ended early; no node call that finished is made again.

        When another call takes the run between this call's read and its write, the run is read again: it is
        resumed if it still ended early, and refused if it is running now.

        Args:
            run: The run as read.
            max_items: A higher ceiling on the items of any one Series, for a run that stopped at its ceiling. The
                run is stored with it as the run is claimed, so every later resume call has it.

        Raises:
            InvalidCallError: The run did not end early (``run_not_resumable``), or, with ``max_items``, the stored
                run can't be loaded to raise its ceiling.
            RuntimeError: The worker has never served.
        """
        while True:
            if run._state.cause is None:
                raise InvalidCallError([run_not_resumable(status=run._state.status)])
            raised = None
            if max_items is not None:
                row, nodes = run._row, self.nodes_for(self.caller_of(run))
                raised = await asyncio.to_thread(
                    lambda: _loaded(row, nodes).with_max_items(max_items).to_json()
                )
            if await self._claim(run, run._state.running(), changed=raised):
                return
            run = await self.read(run.id)

    async def cancel(self, run: RunAsRead) -> bool:
        """Ask a running or waiting run to stop before its next node call, so that it ends ``cancelled``.

        A running run is asked to stop: at once when this worker resumes it, at the next renewal of its lease when
        another does. The cancel is written to the row's state, and a resume call is stopped only by what its row
        says: this worker reads the row again here, and another worker reads it at its next renewal.

        A waiting run is claimed with the cancel already asked, as ``resume`` claims a run, and resumed: its resume
        call ends ``cancelled`` before it calls any node, as ``run.stream(cancel=...)`` ends a waiting run, and its
        questions still wait. Like any resume call it loads the run with the caller's registry and context, so a run
        that can't be resumed ends ``failed`` instead, as it would on its next answer.

        What is done is decided by the run's state as last read. When another call changed the run first (an answer
        claimed it, or the run stopped), the run is read again and the cancel goes by what it is now. Each read again
        follows a write of another call that landed, so this goes round only while other calls act on the run.

        Args:
            run: The run as read.

        Returns:
            Whether the cancel was asked: ``False`` only for a run that is complete or already ended early.

        Raises:
            RuntimeError: The run is waiting, so the cancel would resume it, and the worker has never served.
        """

        def asked_to_stop(state: StoredState) -> StoredState:
            return state.model_copy(update={"cancel_asked": True})

        while True:
            run = await self._ask_of_a_running_run(run, asked_to_stop)
            if run._state.is_running:
                for hold in [hold for hold in self._holds if hold.run_id == run._row.id]:
                    async with hold.lock:
                        await self._read_again(hold)
                return True
            if run._state.status != "waiting":
                return False
            if await self._claim(run, asked_to_stop(run._state.running())):
                return True
            run = await self.read(run._row.id)

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
        run_id = run._row.id
        may_have_ended = True
        while True:
            new = await self._store.events(run_id, after)
            for number, body in new:
                after = number
                yield number, body
            if new:
                may_have_ended = from_json(new[-1][1])["type"] == "run_end"
            elif may_have_ended and not (await self.read(run_id))._state.is_running:
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
                    run = RunAsRead._of(row)
                    await self._claim(run, run._state)
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

    def _mine(self) -> RunChanges:
        """The owner and lease of a row this worker holds."""
        return {"owner": self._id, "lease_until": time.time() + self._lease}

    async def _create(self, graph: str, caller: C, run: Run[Any]) -> str:
        """Store a run, held by this worker, and make its first resume call in the background.

        Args:
            graph: The run's graph, as JSON.
            caller: The run's caller.
            run: The run with its inputs, ready.

        Returns:
            The run's id.

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
            caller=self._caller_type.dump_json(caller).decode(),
            rev=0,
            **written,
        )
        await self._store.create(row)
        self._resume_in_background(row, state)
        return row.id

    async def _claim(
        self, run: RunAsRead, state: StoredState, *, changed: str | None = None
    ) -> bool:
        """Take a run for this worker if it is still at the ``rev`` read, and resume it in the background. Of two
        workers, one wins.

        Args:
            run: The run as read.
            state: The state the row takes with the claim; its ``status`` and ``state`` are written from it.
            changed: The run as the call that claims it changed it, as JSON (a caller's answers given, or its
                ceiling raised); the row's ``run`` then. Left out, the row keeps the run it has.

        Returns:
            Whether this worker took the run.

        Raises:
            RuntimeError: The worker has never served.
        """
        self._check_served()
        written: RunChanges = {**state.columns(), **self._mine()}
        if changed is not None:
            written["run"] = changed
        if not await self._store.update(run._row.id, run._row.rev, written):
            return False
        self._resume_in_background(replace(run._row, **written, rev=run._row.rev + 1), state)
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
            caller = self._caller_type.validate_json(row.caller)
            run = await asyncio.to_thread(_loaded, row, self.nodes_for(caller))
            held = await asyncio.to_thread(Outcome.so_far, run)
            stream = run.stream(
                context=self._context(caller, hold.stop),
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
        run = await self.read(hold.run_id)
        hold.reads += 1
        hold.rev, hold.state = run._row.rev, run._state
        hold.released = run._row.owner != self._id
        if hold.released or hold.state.cancel_asked:
            hold.stop.set()

    # -- a run another worker may hold ------------------------------------------------------------

    async def _ask_of_a_running_run(
        self, run: RunAsRead, asked: Callable[[StoredState], StoredState]
    ) -> RunAsRead:
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
        while run._state.is_running:
            state = asked(run._state)
            changes = state.columns()
            if await self._store.update(run._row.id, run._row.rev, changes):
                return RunAsRead(replace(run._row, **changes, rev=run._row.rev + 1), state)
            run = await self.read(run._row.id)
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
