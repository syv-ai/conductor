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
from dataclasses import replace
from threading import Event
from typing import Any, Generic

from conductor import Graph, InvalidCallError, NodeRegistry, Run, RunProgress
from pydantic import TypeAdapter
from pydantic_core import from_json
from typing_extensions import TypeVar

from conductor_providers.fastapi.bodies import Decline
from conductor_providers.fastapi.held_run import HeldRun, ResumeSettings, loaded
from conductor_providers.fastapi.problems import (
    no_such_run,
    previous_run_unloadable,
    run_not_resumable,
)
from conductor_providers.fastapi.state import M, RunOutcome, RunState, RunView
from conductor_providers.fastapi.store import GuardedStore, RunChanges, RunStore, StoredRun

#: The class of the run's context, the registry's ``context_type``.
X = TypeVar("X", default=None)

log = logging.getLogger("conductor_providers.fastapi")


class Worker(Generic[M, X]):
    """Starts, answers, declines, resumes and cancels the runs kept in one store, on as many processes as share it.

    ``run_router`` is a thin layer of routes over one worker. With routes of your own, you call the worker as the
    routes do: ``begin`` a run (or ``estimate`` one first), ``read`` it, ``answer``, ``decline``, ``resume`` or
    ``cancel`` it,
    follow its ``events``, or start it ``again`` with some inputs changed. Each method that acts on a run takes its id,
    reads the run itself, and returns the run as it is after the call.

    Every run keeps some metadata, of type ``M``: what you want to know about the run, such as who started it. It is
    a ``dict[str, str]`` unless you name a pydantic model with ``metadata_type``. It is written once, when the run
    starts. Each time any worker resumes the run, it builds the run's registry and context from it, so keep in it
    nothing that changes. ``X`` is the class of the run's context, the registry's ``context_type``.

    A worker works only inside ``serving()``, which your server's lifespan enters. While this worker resumes a run, it
    holds a lease on it and saves it every few seconds. When a worker dies, another one takes the run over once the
    lease runs out.

    Attributes:
        nodes_for: The function that gives the registry for a run's metadata, as given; ``/nodes`` describes it and
            ``/compile`` compiles on it.
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
        self.nodes_for = nodes_for
        self._store = GuardedStore(store)
        self._settings = ResumeSettings(
            worker_id=uuid.uuid4().hex,
            nodes_for=nodes_for,
            context=context,
            metadata=TypeAdapter(metadata_type),
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

    # -- what a host calls ------------------------------------------------------------------------

    async def read(self, run_id: str, *, allows: Callable[[M], bool]) -> RunView:
        """The run as a reader sees it: its status, what waits in it, its outputs once complete.

        The row is read once: its metadata goes to ``allows``, and the view is built from the same row.

        Args:
            run_id: The run's id.
            allows: The run filter of the request, a test of a run's metadata: a run it turns away is refused as
                ``no_such_run``, as one that isn't there. Pass ``lambda metadata: True`` to let every run through.

        Returns:
            The run's view, as it was when the run last stopped running.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``).
        """
        return self._view(await self._row(run_id, allows))

    async def metadata_of(self, run_id: str) -> M:
        """A run's metadata, as it was written when the run started.

        The store gives the whole row, and only its ``metadata`` is parsed. For a check of who may see a run, pass
        the filter to the method that acts on it instead (``allows=``), which reads the row once for both.

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
        allows: Callable[[M], bool] | None = None,
    ) -> RunView:
        """Start a run: store it and make its first resume call in the background.

        Args:
            graph: The graph record, as a ``GraphRecord``, as its JSON text or as the plain data a caller sent.
            inputs: Values for the graph's inputs, by name, every one the graph needs.
            metadata: The run's metadata, stored with it; the graph is compiled on its nodes.
            previous: The id of a run of this graph, or of an earlier version of it, that lends its work as
                ``graph.with_inputs(previous=...)`` takes it: a call that receives what a call of the same node
                received there is not made again. It is loaded on ``metadata``'s nodes. The new run starts at this
                worker's ``max_items``, not at a ceiling ``previous`` was resumed with; ``again`` keeps that one.
            allows: The run filter of the request, which ``previous`` must pass: a run it turns away is refused as
                ``no_such_run``. It must be given with ``previous``; ``lambda metadata: True`` lets every run through.

        Returns:
            The new run, as written.

        Raises:
            InvalidCallError: The graph can't run on ``metadata``'s nodes, or refuses the inputs, each with
                conductor's problems; else no run has the id ``previous`` (``no_such_run``), or it can't be loaded
                on those nodes (``previous_run_unloadable``, which says why). Start without ``previous`` then.
            RuntimeError: The worker has never served.
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
        allows: Callable[[M], bool] | None = None,
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
        """Start a run of a run's graph with some inputs changed, as ``run.with_inputs`` makes one; a call that
        receives what it received before isn't made again.

        The new run has the run's graph, the inputs it had except those given, and its ceiling on items, a raised
        one included. To start another graph, such as an edited one, from a run's work, ``begin`` with
        ``previous``.

        Args:
            run_id: The id of the run to start from.
            inputs: The inputs to change, by name; the rest stay as the run had them.
            metadata: The new run's metadata, stored with it; the run is loaded on its nodes.
            allows: The run filter of the request, a test of a run's metadata: a run it turns away is refused as
                ``no_such_run``, as one that isn't there. Pass ``lambda metadata: True`` to let every run through.

        Returns:
            The new run, as written.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``), or the graph can't run on ``metadata``'s nodes,
                or refuses the inputs.
            RuntimeError: The worker has never served.
        """
        metadata = self._checked(metadata)
        row, nodes = await self._row(run_id, allows), self.nodes_for(metadata)
        again = await asyncio.to_thread(lambda: loaded(row, nodes).with_inputs(inputs))
        return await self._create(row.graph, metadata, again)

    async def answer(
        self,
        run_id: str,
        answers: Mapping[str, Any],
        *,
        allows: Callable[[M], bool],
    ) -> RunView:
        """Answer what waits in a run, by waiting id.

        While the run is running, the answers are queued, as one call sent them, and given to the run when its
        resume call ends. Otherwise the run takes them as ``run.answer`` does, whatever its status, and it is
        resumed. An answer to a question that nothing waits on any more, such as one in a complete run, is refused
        by conductor as ``not_waiting``. No answers at all change nothing, as ``run.answer({})`` doesn't, so nothing
        is written.

        When another call takes the run between this call's read and its write, the run is read again, and the
        answers are given to it as it is then: queued if it is running, refused as ``not_waiting`` if the other
        call already answered the question and the run finished.

        Args:
            run_id: The run's id.
            answers: An answer for each waiting call, by its waiting id.
            allows: The run filter of the request, a test of a run's metadata: a run it turns away is refused as
                ``no_such_run``, as one that isn't there. Pass ``lambda metadata: True`` to let every run through.

        Returns:
            The run as written: ``running`` whether the answers were queued or resumed the run.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``), or the run refuses an answer; every answer it
                refuses is listed.
            RuntimeError: The run is not running, so the answers would resume it, and the worker has never served.
        """
        row = await self._row(run_id, allows)
        if not answers:
            return self._view(row)
        while True:
            row = await self._ask_of_a_running_run(row, lambda state: state.with_queued(answers))
            state = RunState.model_validate_json(row.state)
            if state.is_running:
                return self._view(row)
            taken, nodes = row, self.nodes_for(self._metadata_in(row))
            answered = await asyncio.to_thread(
                lambda: loaded(taken, nodes).answer(answers).to_json()
            )
            claimed = await self._claim(
                row, state.answered(answers.keys()).running(), changed=answered
            )
            if claimed is not None:
                return self._view(claimed)
            row = await self._row(run_id)

    async def decline(
        self,
        run_id: str,
        waiting_id: str,
        *,
        reason: str | None = None,
        allows: Callable[[M], bool],
    ) -> RunView:
        """Decline to answer a question that waits in a run, so the run fails, as ``run.decline`` fails it.

        A decline is queued as an answer is. While the run is running, it waits in the queue until the run's resume
        call ends, and the run is given it then, in order with the answers queued with it. Otherwise the run is
        checked first, so a decline the run refuses is refused here, and then claimed with the decline queued. The
        worker that holds it gives it the queue without a resume call, so the declined call doesn't ask again. A run
        that takes the decline ends ``failed``, with ``answer_declined`` as its cause, and isn't resumed. A running
        run can still refuse it when it is given it, as ``not_waiting`` after an answer to the same question or as
        ``answer_stale``. It is then in the run's ``refused``, and the run goes on as it would have.

        When another call takes the run between this call's read and its write, the run is read again, and the
        decline goes by what it is then: queued if it is running, refused as ``not_waiting`` if the question was
        answered.

        Args:
            run_id: The run's id.
            waiting_id: The id of the waiting call, as the run's ``waiting`` lists it.
            reason: Why the person declined, which the run's cause keeps in ``ctx.reason``.
            allows: The run filter of the request, a test of a run's metadata: a run it turns away is refused as
                ``no_such_run``, as one that isn't there. Pass ``lambda metadata: True`` to let every run through.

        Returns:
            The run as written, ``running``.

        Raises:
            ValidationError: ``reason`` isn't a ``str`` or ``None``, such as ``b"no"``. It is raised before the run
                is read.
            InvalidCallError: No run has the id (``no_such_run``). Or the run isn't running, and refuses the
                decline: ``not_waiting``, ``answer_stale``, or ``invalid_answer``, variant ``reason``, for a ``str``
                JSON can't write, such as one with a lone surrogate.
            PydanticSerializationError: The run is running, and ``reason`` is a ``str`` JSON can't write, so the
                decline can't be queued in the row's state.
            RuntimeError: The run is not running, so the decline would be claimed, and the worker has never served.

            Nothing is written when any of these is raised.
        """
        declined = Decline(waiting_id=waiting_id, reason=reason)
        row = await self._row(run_id, allows)
        while True:
            row = await self._ask_of_a_running_run(row, lambda state: state.with_queued(declined))
            state = RunState.model_validate_json(row.state)
            if state.is_running:
                return self._view(row)
            taken, nodes = row, self.nodes_for(self._metadata_in(row))
            await asyncio.to_thread(lambda: loaded(taken, nodes).decline(waiting_id, reason=reason))
            claimed = await self._claim(row, state.with_queued(declined).running())
            if claimed is not None:
                return self._view(claimed)
            row = await self._row(run_id)

    async def resume(
        self,
        run_id: str,
        max_items: int | None = None,
        *,
        allows: Callable[[M], bool],
    ) -> RunView:
        """Resume a run that ended early; no node call that finished is made again.

        When another call takes the run between this call's read and its write, the run is read again: it is
        resumed if it still ended early, and refused if it is running now.

        Args:
            run_id: The run's id.
            max_items: A higher ceiling on the items of any one Series, for a run that stopped at its ceiling. The
                run is stored with it as the run is claimed, so every later resume call has it.
            allows: The run filter of the request, a test of a run's metadata: a run it turns away is refused as
                ``no_such_run``, as one that isn't there. Pass ``lambda metadata: True`` to let every run through.

        Returns:
            The run as written, ``running``.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``), the run did not end early
                (``run_not_resumable``), or, with ``max_items``, the stored run can't be loaded to raise its
                ceiling.
            RuntimeError: The worker has never served.
        """
        row = await self._row(run_id, allows)
        while True:
            state = RunState.model_validate_json(row.state)
            if state.cause is None:
                raise InvalidCallError([run_not_resumable(status=state.status)])
            raised = None
            if max_items is not None:
                nodes = self.nodes_for(self._metadata_in(row))
                raised = await asyncio.to_thread(
                    lambda: loaded(row, nodes).with_max_items(max_items).to_json()
                )
            claimed = await self._claim(row, state.running(), changed=raised)
            if claimed is not None:
                return self._view(claimed)
            row = await self._row(run_id)

    async def cancel(self, run_id: str, *, allows: Callable[[M], bool]) -> RunView:
        """Ask a running or waiting run to stop before its next node call, so that it ends ``cancelled``.

        A running run is asked to stop: at once when this worker resumes it, at its next checkpoint or lease renewal
        when another does. The cancel is written to the row's state, and a resume call is stopped only by what its
        row says: this worker reads the row again here, and another worker reads it when its next checkpoint or
        renewal finds the row changed.

        A waiting run is claimed with the cancel already asked, as ``resume`` claims a run, and resumed: its resume
        call ends ``cancelled`` before it calls any node, as ``run.stream(cancel=...)`` ends a waiting run, and its
        questions still wait. Like any resume call it loads the run with its metadata's registry and context, so a
        run that can't be resumed ends ``failed`` instead, as it would on its next answer.

        A run that is complete or already ended early has nothing to cancel, and is left as it is. When another
        call changes the run between this call's read and its write, the run is read again and the cancel goes by
        what it is then.

        Args:
            run_id: The run's id.
            allows: The run filter of the request, a test of a run's metadata: a run it turns away is refused as
                ``no_such_run``, as one that isn't there. Pass ``lambda metadata: True`` to let every run through.

        Returns:
            The run as written: ``running`` when the cancel was asked, else the run as it is.

        Raises:
            InvalidCallError: No run has the id (``no_such_run``).
            RuntimeError: The run is waiting, so the cancel would resume it, and the worker has never served.
        """

        row = await self._row(run_id, allows)
        while True:
            row = await self._ask_of_a_running_run(row, RunState.asked_to_stop)
            state = RunState.model_validate_json(row.state)
            if state.is_running:
                if run_id in self._held:
                    held, _ = self._held[run_id]
                    await held.heard_of_a_change()
                return self._view(row)
            if state.status != "waiting":
                return self._view(row)
            claimed = await self._claim(row, state.running().asked_to_stop())
            if claimed is not None:
                return self._view(claimed)
            row = await self._row(run_id)

    async def events(
        self, run_id: str, after: int = -1, *, allows: Callable[[M], bool]
    ) -> AsyncIterator[tuple[int, str]]:
        """A run's events as they are stored, from whichever worker resumes it, until the run isn't running.

        The run is read, and checked with ``allows``, when this is awaited, so a run that isn't there is refused
        before any event is given. ``async for number, body in await worker.events(run_id)`` follows it.

        Args:
            run_id: The run's id.
            after: The number of the last event the follower has; ``-1`` for none.
            allows: The run filter of the request, a test of a run's metadata: a run it turns away is refused as
                ``no_such_run``, as one that isn't there. Pass ``lambda metadata: True`` to let every run through.

        Returns:
            Each event numbered above ``after``, in order, as ``(number, body)``.

        Raises:
            InvalidCallError: No run has the id, or ``allows`` turns it away (``no_such_run``).
        """
        row = await self._row(run_id, allows)
        return self._events_from(row, after)

    async def _events_from(self, row: StoredRun, after: int) -> AsyncGenerator[tuple[int, str]]:
        """The events of the run in ``row`` above ``after``, until the run isn't running.

        A row stops saying ``running`` only after the ``run_end`` of its resume call is stored. So once a read of
        the row says the run stopped, the events stored then are all there will be. The row is read again only when
        a ``run_end`` was the last event, or there was no event yet; otherwise events alone are read.
        """
        run_id, ended = row.id, not RunState.model_validate_json(row.state).is_running
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
                ended = not RunState.model_validate_json((await self._row(run_id)).state).is_running
            if not ended:  # the row is written after its run_end, so it may still say running for a poll or two
                await asyncio.sleep(self._poll)

    async def _sweep(self) -> None:
        """Go on with every run whose lease ran out: its worker died or handed it back. A run that can't be taken
        (its row can't be read, the store fails) is logged and left for the next call, and the others are taken."""
        for row in await self._store.expired(time.time()):
            if row.id not in self._held:
                try:
                    await self._claim(row, RunState.model_validate_json(row.state))
                except Exception:
                    log.exception(
                        "run %s, whose lease ran out, was not taken over; the next sweep tries again",
                        row.id,
                    )

    @asynccontextmanager
    async def serving(self) -> AsyncGenerator[None]:
        """The time the worker serves: your server's lifespan enters it.

        While it is entered, the worker takes over the runs whose lease ran out, three times a lease. On leaving, it
        hands back every run it is resuming: no more nodes are started, the calls going on are waited for and kept,
        and another worker that serves goes on with the run. A run begun or taken while it leaves is handed back too.

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
            # a run begun or taken while the others were handed back is handed back too
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
        allows: Callable[[M], bool] | None,
    ) -> Run[Any]:
        """The run ``begin`` starts and ``estimate`` counts: the graph compiled on the metadata's nodes, with its
        inputs, and the work of ``previous`` lent to it.

        The graph's own problems and the inputs' are refused before the previous run is loaded.

        Raises:
            InvalidCallError: The graph or the inputs are refused, no run has the id ``previous``
                (``no_such_run``), or it can't be loaded on these nodes (``previous_run_unloadable``).
            TypeError: ``previous`` is given without ``allows``.
        """
        if previous is not None and allows is None:
            raise TypeError(
                "a run started from a previous one needs allows=, the request's run filter; pass "
                "`lambda metadata: True` to let every run through"
            )
        nodes = self.nodes_for(metadata)
        lends = None if previous is None else await self._row(previous, allows)

        def ready() -> Run[Any]:
            compiled = Graph.compile(graph, nodes)
            run = compiled.with_inputs(inputs, max_items=self._max_items)
            if lends is None:
                return run
            try:
                lent = loaded(lends, nodes)
            except InvalidCallError as refusal:
                reason = "; ".join(problem.msg for problem in refusal.problems)
                raise InvalidCallError(
                    [previous_run_unloadable(run_id=lends.id, reason=reason)]
                ) from refusal
            return compiled.with_inputs(inputs, max_items=self._max_items, previous=lent)

        return await asyncio.to_thread(ready)

    async def _row(self, run_id: str, allows: Callable[[M], bool] | None = None) -> StoredRun:
        """A run's row, as the store has it, if ``allows`` lets it through.

        Raises:
            InvalidCallError: No run has the id, or ``allows`` turns its metadata away (``no_such_run``). The two
                read the same, so a request learns nothing about a run it may not see.
        """
        row = await self._store.get(run_id)
        if row is None or (allows is not None and not allows(self._metadata_in(row))):
            raise InvalidCallError([no_such_run()])
        return row

    def _view(self, row: StoredRun) -> RunView:
        """The run in ``row`` as a reader sees it, from its state, its outcome and its metadata."""
        state = RunState.model_validate_json(row.state)
        outcome = RunOutcome.model_validate_json(row.outcome)
        return RunView(
            id=row.id,
            metadata=from_json(row.metadata),
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
        return self._settings.metadata.validate_python(metadata)

    def _metadata_in(self, row: StoredRun) -> M:
        """The metadata a row keeps, read from its JSON."""
        return self._settings.metadata.validate_json(row.metadata)

    async def _create(self, graph: str, metadata: M, run: Run[Any]) -> RunView:
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
            lambda: (run.to_json(), RunOutcome.of(run).model_dump_json())
        )
        state = RunState.started()
        written: RunChanges = {
            "run": saved,
            **state.columns(),
            "outcome": outcome,
            **self._settings.holder(),
        }
        row = StoredRun(
            id=uuid.uuid4().hex,
            graph=graph,
            metadata=self._settings.metadata.dump_json(metadata, by_alias=True).decode(),
            rev=0,
            **written,
        )
        await self._store.create(row)
        self._resume_in_background(row)
        return self._view(row)

    async def _claim(
        self, row: StoredRun, state: RunState, *, changed: str | None = None
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
        written: RunChanges = {**state.columns(), **self._settings.holder()}
        if changed is not None:
            written["run"] = changed
        if not await self._store.update(row.id, row.rev, written):
            return None
        claimed = replace(row, **written, rev=row.rev + 1)
        self._resume_in_background(claimed)
        return claimed

    def _resume_in_background(self, row: StoredRun) -> None:
        """Start the task that resumes a run this worker has just taken.

        Args:
            row: The row as this worker wrote it to take the run.
        """
        held = HeldRun(self._store, row, self._settings)

        async def hold() -> None:
            try:
                await held.run()
            finally:
                del self._held[row.id]

        self._held[row.id] = (held, asyncio.get_running_loop().create_task(hold()))

    # -- a run another worker may hold ------------------------------------------------------------

    async def _ask_of_a_running_run(
        self, row: StoredRun, asked: Callable[[RunState], RunState]
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
        while (state := RunState.model_validate_json(row.state)).is_running:
            changes = asked(state).columns()
            if await self._store.update(row.id, row.rev, changes):
                return replace(row, **changes, rev=row.rev + 1)
            row = await self._row(row.id)
        return row
