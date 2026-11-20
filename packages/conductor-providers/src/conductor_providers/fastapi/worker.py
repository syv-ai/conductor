"""``Worker``: starts, answers, declines, resumes and cancels stored runs, on any number of processes sharing one store.

A worker keeps runs in your store. To go on with a run, it takes the run's lease and starts a ``HeldRun``, which
makes the resume calls in the background and writes a checkpoint every few seconds. Another worker takes over a run
whose lease ran out, from its last checkpoint, and a worker that shuts down hands its runs back. Every write is at the
``rev`` the writer read, so of two workers acting on one run, one goes on and the other reads the run again.

Nothing here knows HTTP: ``run_router`` makes its routes over one worker, and a host with routes of its own calls a
worker the same way.
"""

import asyncio
import logging
import time
import uuid
from asyncio import Event as AsyncioEvent
from asyncio import Task
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from threading import Event
from types import MappingProxyType
from typing import Any, Generic

from conductor import Graph, InvalidCallError, NodeRegistry, Run, RunProgress
from pydantic import TypeAdapter
from pydantic_core import from_json
from typing_extensions import TypeVar

from conductor_providers.fastapi.held_run import HANDED_BACK, HeldRun, ResumeSettings
from conductor_providers.fastapi.problems import (
    RESUMABLE,
    no_such_run,
    run_not_resumable,
)
from conductor_providers.fastapi.state import RunOutcome, RunState, RunView
from conductor_providers.fastapi.store import GuardedStore, M, RunRow, RunStore

#: The class of the run's context, the registry's ``context_type``.
X = TypeVar("X", default=None)

log = logging.getLogger("conductor_providers.fastapi")


class Worker(Generic[M, X]):
    """Starts, answers, declines, resumes and cancels the runs kept in one store, on as many processes as share it.

    ``run_router`` is a thin layer of routes over one worker. With routes of your own, you call the worker as the
    routes do: ``begin`` a run (or ``estimate`` one first) and ``resume`` it to start it, ``read`` it, ``answer``,
    ``decline``, ``resume`` or ``cancel`` it, follow its ``events``, or store it ``again`` with some inputs changed and
    ``resume`` that. Each method that acts on a run takes its id, reads the run itself, and returns the run as it is
    after the call. When another call writes the run between this call's read and its write, the run is read again,
    and the call goes by what it is then.

    Each method that acts on a run also takes ``allows``, the request's run filter: a test of a run's metadata. The
    row is read once, its metadata goes to ``allows``, and the call acts on the same row. A run the filter turns away
    is refused as ``no_such_run``, as one that isn't there. Pass ``lambda metadata: True`` to let every run through.

    How a run takes answers and declines, while it runs and after, is in the reference:
    ``python -m conductor.about answers-and-declines``.

    Every run keeps some metadata, of type ``M``: what you want to know about the run, such as who started it. It is
    a ``dict[str, str]`` unless you name a pydantic model with ``metadata_type``. It is written once, when the run
    starts. Each time any worker resumes the run, it builds the run's registry and context from it, so keep in it
    nothing that changes. ``X`` is the class of the run's context, the registry's ``context_type``.

    A worker works only inside ``serving()``, which your server's lifespan enters. While this worker resumes a run, it
    holds a lease on it and saves it every few seconds. When a worker dies, another one takes the run over once the
    lease runs out.
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
            nodes_for: The registry a run's graph compiles on, from the run's metadata: ``lambda metadata:
                registry`` for one registry, or one per team. It is called on every call and every resume, so let it
                cache what it builds.
            context: The run's context, an instance of the registry's ``context_type``, from the run's metadata and
                a stop event. It is called on every resume. The worker sets the stop event when that resume call
                stops early: the run was cancelled, another worker took it over, or the store failed. A slow node
                reads it and raises; it is not set when the worker shuts down, since the calls going on then are
                kept. Leave it out for a registry with no ``context_type``.
            metadata_type: The type of a run's metadata: a pydantic model of yours, or ``dict[str, str]``.
            max_items: The item limit a run starts with, for any one Series; ``None`` for none. A resume may raise
                it for one run.
            lease: The seconds this worker holds a run between renewals.
            checkpoint_every: The seconds between saves of a running run.
            resume_timeout: The seconds one resume call may take before the run ends ``timed_out``; ``None`` for no
                limit.
            poll: The seconds between a follower's reads of new events.
        """
        self._metadata = TypeAdapter(metadata_type)
        self._store = GuardedStore(store, self._metadata)
        self._settings = ResumeSettings(
            worker_id=uuid.uuid4().hex,
            nodes_for=nodes_for,
            context=context,
            lease=lease,
            checkpoint_every=checkpoint_every,
            resume_timeout=resume_timeout,
            handing_back=AsyncioEvent(),
        )
        self._max_items = max_items
        self._poll = poll
        self._served = False
        # every run this worker holds, by id, with the task that resumes it
        self._held: dict[str, tuple[HeldRun, Task[None]]] = {}

    @property
    def nodes_for(self) -> Callable[[M], NodeRegistry[X]]:
        """The function that gives the registry for a run's metadata.

        It is the ``nodes_for`` the worker was made with. ``/nodes`` describes what it gives, and ``/compile``
        compiles on it. It is read-only: make a new worker to give it another.
        """
        return self._settings.nodes_for

    # -- what a host calls ------------------------------------------------------------------------

    async def read(self, run_id: str, *, allows: Callable[[M], bool]) -> RunView:
        """The run as a reader sees it: its status, what waits in it, its outputs once complete.

        Args:
            run_id: The run's id.
            allows: The request's run filter.

        Returns:
            The run's view, as it was when the run last stopped running.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``).
        """
        return self._view(await self._row(run_id, allows))

    async def begin(
        self,
        graph: Any,
        inputs: Mapping[str, Any],
        metadata: M,
        *,
        previous: str | None = None,
        allows: Callable[[M], bool],
    ) -> RunView:
        """Store a new run, ready for its first resume call; ``resume`` starts it.

        Nothing else happens, so a host can call it inside a database transaction of its own: a retried
        transaction leaves no run behind and starts none. Call ``resume`` once the run is stored for good, such as
        after the commit. A run nobody resumes is started by the next sweep of any serving worker, within a third of
        a lease, so a forgotten ``resume`` only delays it.

        Args:
            graph: The graph record, as a ``GraphRecord``, as its JSON text or as the plain data a caller sent.
            inputs: Values for the graph's inputs, by name, every one the graph needs.
            metadata: The run's metadata, stored with it; the graph is compiled on its nodes.
            previous: The id of a run of this graph, or of an earlier version of it, that lends its work as
                ``graph.with_inputs(previous=...)`` takes it: a call that receives what a call of the same node
                received there is not made again. It is loaded on ``metadata``'s nodes. The new run starts at this
                worker's ``max_items``, not at a ceiling ``previous`` was resumed with; ``again`` keeps that one.
            allows: The run filter of the request, which ``previous`` must pass: a run it turns away is refused as
                ``no_such_run``. It is required, as on every other method, so a host's own route can't leave the
                access check out by accident; ``lambda metadata: True`` lets every run through.

        Returns:
            The new run, as written: ``running``, since a worker is about to resume it.

        Raises:
            InvalidCallError: The graph can't run on ``metadata``'s nodes, or refuses the inputs, each with
                conductor's problems. Else no run has the id ``previous`` (``no_such_run``, at ``loc`` ``("previous",)``). Or the previous run
                can't be loaded on those nodes, with the problems ``Run.from_json`` gives for its saved run, such as
                ``saved_run_unreadable``. Start without ``previous`` then.
        """
        metadata = self._checked(metadata)
        run = await self._ready(graph, inputs, metadata, previous, allows)
        return await self._create(run.graph.record.model_dump_json(), metadata, run)

    async def estimate(
        self,
        graph: Any,
        inputs: Mapping[str, Any],
        metadata: M,
        *,
        previous: str | None = None,
        allows: Callable[[M], bool],
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
            allows: The run filter ``previous`` must pass, as ``begin`` takes it.

        Returns:
            The run's progress after that call.

        Raises:
            InvalidCallError: As ``begin`` refuses the same call.
        """
        metadata = self._checked(metadata)
        run = await self._ready(graph, inputs, metadata, previous, allows)

        def estimated() -> RunProgress:
            context = self._settings.context_for(metadata, Event())
            return run.resume(
                context=context, timeout=self._settings.resume_timeout, only_costs_nothing=True
            ).progress

        return await asyncio.to_thread(estimated)

    async def again(
        self,
        run_id: str,
        inputs: Mapping[str, Any],
        metadata: M,
        *,
        allows: Callable[[M], bool],
    ) -> RunView:
        """Store a run of a run's graph with some inputs changed, as ``run.with_inputs`` makes one; a call that
        receives what it received before isn't made again. ``resume`` starts it, as it starts a run ``begin`` stores.

        The new run has the run's graph, the inputs it had except those given, and its ceiling on items, a raised
        one included. To start another graph, such as an edited one, from a run's work, ``begin`` with
        ``previous``.

        Args:
            run_id: The id of the run to start from.
            inputs: The inputs to change, by name; the rest stay as the run had them.
            metadata: The new run's metadata, stored with it; the run is loaded on its nodes.
            allows: The request's run filter.

        Returns:
            The new run, as written: ``running``, since a worker is about to resume it.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``), or the graph can't run on ``metadata``'s nodes,
                or refuses the inputs.
        """
        metadata = self._checked(metadata)
        row = await self._row(run_id, allows)
        loaded = await self._settings.load(row, metadata)
        again = await asyncio.to_thread(loaded.with_inputs, inputs)
        return await self._create(row.stored.graph, metadata, again)

    async def answer(
        self,
        run_id: str,
        answers: Mapping[str, Any],
        *,
        allows: Callable[[M], bool],
    ) -> RunView:
        """Answer what waits in a run, by waiting id.

        A running run queues the answers until its resume call ends. A run that isn't running takes them as
        ``run.with_answers`` does, whatever its status, and is resumed. A ``declined`` run waits on nothing, so it
        refuses every answer as ``not_waiting``. No answers at all change nothing. The queue's rules are in
        ``python -m conductor.about answers-and-declines``.

        Args:
            run_id: The run's id.
            answers: An answer for each waiting call, by its waiting id.
            allows: The request's run filter.

        Returns:
            The run as written: ``running``.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``), or an answer is refused, and nothing is written.
            RuntimeError: The answers would resume the run, and the worker has never served.
        """
        if not answers:
            return await self.read(run_id, allows=allows)
        row = await self._row(run_id, allows)
        return self._view(
            await self._store.change(row, lambda row: self._reply(row, answers=answers))
        )

    async def decline(
        self,
        run_id: str,
        waiting_id: str,
        *,
        reason: str | None = None,
        allows: Callable[[M], bool],
    ) -> RunView:
        """Decline to answer a question that waits in a run, so the run ends for good, as ``run.with_decline`` ends it.

        A run that takes the decline is ``declined``, with ``answer_declined``: nothing more is called, and a later
        answer, decline or resume is refused. A running run queues the decline as it queues an answer, and takes it
        when its resume call ends. A run that isn't running takes it here, with no resume call, so this needs no
        serving worker. To stop a running run's calls at once too, ``cancel`` it as well. The queue's rules are in
        ``python -m conductor.about answers-and-declines``.

        Args:
            run_id: The run's id.
            waiting_id: The id of the waiting call, as the run's ``waiting`` lists it.
            reason: Why the person declined, which the run's cause keeps in ``ctx.reason``.
            allows: The request's run filter.

        Returns:
            The run as written: ``declined`` when it took the decline, else ``running``.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``), ``reason`` can't be written as JSON
                (``invalid_decline_reason``), a run that isn't running refuses the decline (``not_waiting``,
                ``answer_stale``), or it can't be loaded on its metadata's nodes. Nothing is written then.
        """

        async def declined(row: RunRow[M]) -> RunRow[M] | None:
            state = row.state
            if state.is_running:
                return await self._store.update(
                    row, state=state.with_queued(declines={waiting_id: reason})
                )
            run = await self._settings.load(row, row.metadata)
            ended, saved, outcome = await asyncio.to_thread(
                lambda: state.declined(run.with_decline(waiting_id, reason=reason))
            )
            return await self._store.update(row, state=ended, run=saved, outcome=outcome)

        row = await self._row(run_id, allows)
        return self._view(await self._store.change(row, declined))

    async def resume(
        self,
        run_id: str,
        max_items: int | None = None,
        *,
        allows: Callable[[M], bool],
    ) -> RunView:
        """Start a run ``begin`` or ``again`` stored, or resume one that ended early; no node call that finished is
        made again.

        A ``running`` run that no worker holds, one just stored or handed back, is taken by this worker and resumed.
        One that a worker holds is left as it is, so a resume that comes after a sweep took the run changes nothing.
        A ``waiting`` or ``complete`` run didn't end early, and a ``declined`` one has ended for good, so each is
        refused.

        Args:
            run_id: The run's id.
            max_items: A higher ceiling on the items of any one Series, for a run that stopped at its ceiling. The
                run is stored with it as the run is claimed, so every later resume call has it. A ``running`` run
                hasn't stopped, so it is refused with it.
            allows: The request's run filter.

        Returns:
            The run as written: ``running``.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``), the run did not end early or was declined
                (``run_not_resumable``), or, with ``max_items``, the run is running (``run_not_resumable``) or the
                stored run can't be loaded to raise its ceiling.
            RuntimeError: The run would be resumed, and the worker has never served.
        """

        async def resumed(row: RunRow[M]) -> RunRow[M] | None:
            state = row.state
            if state.is_running and max_items is None:
                return row if row.stored.owner is not None else await self._claim(row, state)
            if state.status not in RESUMABLE:
                raise InvalidCallError([run_not_resumable(status=state.status)])
            return await self._reply(row, max_items=max_items)

        row = await self._row(run_id, allows)
        return self._view(await self._store.change(row, resumed))

    async def cancel(self, run_id: str, *, allows: Callable[[M], bool]) -> RunView:
        """Ask a running or waiting run to stop before its next node call, so that it ends ``cancelled``.

        A running run is asked to stop: at once when this worker resumes it, at its next checkpoint or lease renewal
        when another does. The cancel is written to the row's state. This worker then stops its own resume call here,
        and another worker reads the cancel when its next checkpoint or renewal finds the row changed. A run that
        completes before its worker hears the cancel still ends ``cancelled``, since the worker reads the row again
        as it writes the run's end. Every call's outcome is kept, so a ``resume`` then makes no call and completes
        it.

        A waiting run is claimed with the cancel already asked, as ``resume`` claims a run, and resumed: its resume
        call ends ``cancelled`` before it calls any node, as ``run.stream(cancel=...)`` ends a waiting run, and its
        questions still wait. Like any resume call it loads the run with its metadata's registry and context, so a
        run that can't be resumed ends ``failed`` instead, as it would on its next answer.

        A run that is complete or already ended early has nothing to cancel, and is left as it is.

        Args:
            run_id: The run's id.
            allows: The request's run filter.

        Returns:
            The run as written: ``running`` when the cancel was asked, else the run as it is.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``).
            RuntimeError: The run is waiting, so the cancel would resume it, and the worker has never served.
        """

        async def cancelled(row: RunRow[M]) -> RunRow[M] | None:
            if row.state.status == "waiting":
                return await self._claim(row, row.state.running().asked_to_stop())
            if not row.state.is_running:
                return row
            written = await self._store.update(row, state=row.state.asked_to_stop())
            if written is not None and run_id in self._held:
                held, _ = self._held[run_id]
                held.cancel()
            return written

        row = await self._row(run_id, allows)
        return self._view(await self._store.change(row, cancelled))

    async def events(
        self, run_id: str, after: int = -1, *, allows: Callable[[M], bool]
    ) -> AsyncIterator[tuple[int, str]]:
        """A run's events as they are stored, from whichever worker resumes it, until the run isn't running.

        The run is read, and checked with ``allows``, when this is awaited, so a run that isn't there is refused
        before any event is given. ``async for number, body in await worker.events(run_id)`` follows it.

        Args:
            run_id: The run's id.
            after: The number of the last event the follower has; ``-1`` for none.
            allows: The request's run filter.

        Returns:
            Each event numbered above ``after``, in order, as ``(number, body)``.

        Raises:
            InvalidCallError: No run has the id, or ``allows`` turns it away (``no_such_run``).
        """
        row = await self._row(run_id, allows)
        return self._events_from(row, after)

    async def _events_from(self, row: RunRow[M], after: int) -> AsyncGenerator[tuple[int, str]]:
        """The events of the run in ``row`` above ``after``, until the run isn't running.

        A row stops saying ``running`` in the same write that stores its ``run_end``. So once a read of the row says
        the run stopped, the events stored then are all there will be. The row is read again only when a ``run_end``
        was the last event, or there was no event yet; otherwise events alone are read.
        """
        run_id, ended = row.stored.id, not row.state.is_running
        may_have_ended = True
        while True:
            new = await self._store.events(run_id, after)
            for number, body in new:
                after = number
                yield number, body
            if ended:
                return
            if new:
                may_have_ended = from_json(new[-1][1])["type"] == "run_end"
                continue
            if may_have_ended:
                ended = not (await self._row(run_id)).state.is_running
            if not ended:
                await asyncio.sleep(self._poll)

    async def _sweep(self) -> None:
        """Go on with every run whose lease ran out: its worker died or handed it back, or it was just stored and
        nobody resumed it. A run that can't be taken (its row can't be read, the store fails) is logged and left for
        the next call, and the others are taken."""
        for row in await self._store.expired(time.time()):
            if row.stored.id not in self._held:
                try:
                    await self._claim(row, row.state)
                except Exception:
                    log.exception(
                        "run %s, whose lease ran out, was not taken over; the next sweep tries again",
                        row.stored.id,
                    )

    @asynccontextmanager
    async def serving(self) -> AsyncGenerator[None]:
        """The time the worker serves: your server's lifespan enters it.

        While it is entered, the worker takes over the runs whose lease ran out, three times a lease. On leaving, it
        hands back every run it is resuming: no more nodes are started, the calls going on are waited for and kept,
        and another worker that serves goes on with the run. A run taken while it leaves is handed back too.

        A worker that has never served raises ``RuntimeError`` from any call that would make a run running, since it
        would hand no run back and take over none. One that served and left takes such a call and hands the run back
        at once. A server with ``run_router`` starts serving when FastAPI starts it, and FastAPI's ``TestClient``
        starts it only inside ``with TestClient(app) as client:``.

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
                await asyncio.sleep(self._settings.lease / 3)

        self._settings.handing_back.clear()
        self._served = True
        sweeping = asyncio.create_task(keep_sweeping())
        try:
            yield
        finally:
            self._settings.handing_back.set()
            sweeping.cancel()
            # a run taken while the others were handed back is handed back too
            while self._held:
                await asyncio.gather(*(task for _, task in self._held.values()))

    # -- taking a run -----------------------------------------------------------------------------

    def _check_served(self) -> None:
        """Refuse to make a run running on a worker that never entered ``serving()``. A worker that served and left
        it takes the run and hands it back at once, for a worker that serves to take over.

        Raises:
            RuntimeError: The worker has never served.
        """
        if not self._served:
            raise RuntimeError(
                "the worker isn't serving, so it would hand no run back when it stops, and take over none whose "
                "worker stopped. Enter `async with worker.serving():` first. run_router does that when FastAPI starts the "
                "server, so with FastAPI's TestClient write `with TestClient(app) as client:`, since a TestClient "
                "without `with` never starts it"
            )

    async def _ready(
        self,
        graph: Any,
        inputs: Mapping[str, Any],
        metadata: M,
        previous: str | None,
        allows: Callable[[M], bool],
    ) -> Run[Any]:
        """The run ``begin`` starts and ``estimate`` counts: the graph compiled on the metadata's nodes, with its
        inputs, and the work of ``previous`` lent to it.

        The graph's own problems and the inputs' are refused before the previous run is loaded.

        Raises:
            InvalidCallError: The graph or the inputs are refused, no run has the id ``previous``
                (``no_such_run``), or it can't be loaded on these nodes (conductor's problems, as they are).
        """
        nodes = self.nodes_for(metadata)
        lends = None if previous is None else await self._row(previous, allows, "previous")

        def checked() -> tuple[Graph[Any], Run[Any]]:
            compiled = Graph.compile(graph, nodes)
            return compiled, compiled.with_inputs(inputs, max_items=self._max_items)

        compiled, run = await asyncio.to_thread(checked)
        if lends is None:
            return run
        previous_run = await self._settings.load(lends, metadata)
        return await asyncio.to_thread(
            compiled.with_inputs, inputs, max_items=self._max_items, previous=previous_run
        )

    async def _row(
        self, run_id: str, allows: Callable[[M], bool] | None = None, *loc: str
    ) -> RunRow[M]:
        """A run's row, as the store has it, if ``allows`` lets it through.

        Args:
            run_id: The run's id.
            allows: The request's run filter, or ``None`` for the worker's own reads.
            loc: Where the id was given, for the problem: ``"previous"``, or nothing for the route's own run.

        Raises:
            InvalidCallError: No run has the id, or ``allows`` turns its metadata away (``no_such_run``). The two
                read the same, so a request learns nothing about a run it may not see.
        """
        row = await self._store.get(run_id)
        if row is None or (allows is not None and not allows(row.metadata)):
            raise InvalidCallError([no_such_run(*loc)])
        return row

    def _view(self, row: RunRow[M]) -> RunView:
        """The run in ``row`` as a reader sees it, from its state, its outcome and its metadata."""
        state, outcome = row.state, row.outcome
        return RunView(
            id=row.stored.id,
            metadata=from_json(row.stored.metadata),
            status=state.status,
            cause=state.cause,
            waiting=outcome.waiting,
            outputs=outcome.outputs,
            progress=outcome.progress,
            refused=state.refused,
        )

    def _checked(self, metadata: M) -> M:
        """The metadata for a new run, checked once against the worker's ``metadata_type`` before anything is stored.

        It comes from the host's own code, a dependency or a call, never from the client, so a value of another type
        is a bug in that code: pydantic's ``ValidationError`` is raised as it is, a 500, as a ``nodes_for`` that
        raises is.
        """
        return self._metadata.validate_python(metadata)

    async def _create(self, graph: str, metadata: M, run: Run[Any]) -> RunView:
        """Store a new run as a run handed back is stored: ``running``, held by no worker, its lease already out.

        The store's ``create`` is all that happens, so a host's transaction around it leaves nothing behind when it
        rolls back. ``resume`` takes the row at once, and the next sweep of any serving worker takes it otherwise.

        Args:
            graph: The run's graph, as JSON.
            metadata: The run's metadata.
            run: The run with its inputs, ready.

        Returns:
            The run, as written.
        """
        state = RunState.started()
        saved, outcome = await asyncio.to_thread(
            lambda: (run.to_json(), RunOutcome.of(run, state.status))
        )
        row = await self._store.create(
            uuid.uuid4().hex,
            graph,
            metadata,
            HANDED_BACK,
            state=state,
            run=saved,
            outcome=outcome,
        )
        return self._view(row)

    async def _claim(
        self, row: RunRow[M], state: RunState, *, changed: str | None = None
    ) -> RunRow[M] | None:
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
        claimed = await self._store.update(row, self._settings.holder(), state=state, run=changed)
        if claimed is not None:
            self._resume_in_background(claimed)
        return claimed

    async def _reply(
        self,
        row: RunRow[M],
        *,
        answers: Mapping[str, Any] = MappingProxyType({}),
        max_items: int | None = None,
    ) -> RunRow[M] | None:
        """One try at giving a run this call's answers, or its higher ceiling: ``answer`` and ``resume`` both come
        here.

        A running run queues the answers. Otherwise they go to ``run.with_answers``, and the run is claimed and resumed.
        A row that isn't running holds no queue. The rules are in ``python -m conductor.about answers-and-declines``.

        Args:
            row: The run's row as read.
            answers: This call's answers, by waiting id.
            max_items: A higher ceiling on the items of any one Series, written with the claim.

        Returns:
            The row as written; ``None`` when another call wrote it first.

        Raises:
            InvalidCallError: The run refuses an answer of this call, with every problem it gave them. Or the run
                can't be loaded on its metadata's nodes.
            RuntimeError: The run would be resumed, and the worker has never served.
        """
        state = row.state
        queued = state.with_queued(answers)
        if state.is_running:
            return await self._store.update(row, state=queued)
        changed = None
        if answers or max_items is not None:
            run = await self._settings.load(row, row.metadata)

            def given() -> str:
                answered = run.with_answers(answers)
                return (
                    answered if max_items is None else answered.with_max_items(max_items)
                ).to_json()

            changed = await asyncio.to_thread(given)
        return await self._claim(row, state.answered(answers.keys()).running(), changed=changed)

    def _resume_in_background(self, row: RunRow[M]) -> None:
        """Start the task that resumes a run this worker has just taken.

        Args:
            row: The row as this worker wrote it to take the run.
        """
        held = HeldRun(self._store, row, self._settings)

        async def hold() -> None:
            try:
                await held.run()
            finally:
                # The run may have been taken again while this task wrote that it lets the run go. Then the entry
                # is the newer task's, and that task removes it.
                if self._held.get(row.stored.id, (None,))[0] is held:
                    del self._held[row.stored.id]

        self._held[row.stored.id] = (held, asyncio.get_running_loop().create_task(hold()))
