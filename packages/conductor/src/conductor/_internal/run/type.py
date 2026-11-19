"""``Run``, one run of a graph: its inputs and the outcome of every call so far.

Take the README's graph. ``graph.run({"text": "a\\nb"})`` calls ``lines`` once, ``shout`` once per line and ``join``
once. The run stores one outcome per call, such as ``Done`` when the call returned (``outcome.py`` has the four
kinds). A call with no outcome hasn't been made yet.

``resume()`` calls every node that is ready, until the run is complete or has to stop. It may stop to wait for a
person, after a failure, at the item limit or at a timeout. Wherever it stops, the run can be saved as JSON and
go on later, in another process too.

A node inside a subgraph is called like any other node: once per item its subgraph node is called for, and
inside that once per item of its own loops. The subgraph node itself is never called. What a run says about it
comes from the nodes inside. A call at any depth is named by one path of node ids and item indexes:
``("e", 0, "ask", 2)`` is the call of ``ask`` for its third item, inside the first call of ``e``.
"""

from __future__ import annotations

import dataclasses
import functools
from asyncio import Event as AsyncioEvent
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass
from functools import cache
from threading import Event as ThreadEvent
from types import MappingProxyType
from typing import Any, assert_never, overload

from conductor._internal.ask import AnswerAsJsonValidator, AnswerValidator, DeclineValidator
from conductor._internal.base import (
    ConductorModel,
    ItemIndexes,
    Loc,
    NodePath,
    parse_waiting_id,
    write_waiting_id,
)
from conductor._internal.graph import Graph, LimitsValidator
from conductor._internal.problem import InvalidCallError, Problem, call_problems, run_problems

from .blueprint import BlueprintFunctionNode, outcome_loc
from .events import (
    Event,
    EventBase,
    RunCheckpoint,
    RunEnd,
    RunStart,
    with_run,
)
from .json import outputs_to_json, run_from_json, run_to_json
from .node_events import NodeEvents
from .outcome import Done, NoneInput, Outcome, Skipped, Unanswered
from .outcomes import RunOutcomes
from .progress import RunProgress, call_count, is_complete, progress_of
from .results import outputs_of, results_of
from .reuse import ReusableOutcomes
from .scheduler import Scheduler
from .state import EndedEarly, LastResume, RunStatus, deadline, kept_by_an_answer, status_of
from .stream import stream
from .validator import ForgetValidator, WaitingIdValidator
from .waiting import Waiting, question_fingerprint, waiting_record


class NoneInputCall(ConductorModel):
    """A call that wasn't made because one of its arguments was ``None`` and its input doesn't take ``None``.

    Say ``strip(line: str) -> str | None`` returns ``None`` for the line ``" "``, and ``shout(line: str)`` is wired
    to it. Then ``run.none_input_calls`` lists ``NoneInputCall(loc=("shout", 1), input="line")``.

    Attributes:
        loc: Which call it was, written as a ``Problem``'s ``loc``: the node's id, then its item index in each loop
            it is inside. A call inside a subgraph starts with the subgraph node and its item.
        input: The input that got ``None``.
    """

    loc: Loc
    input: str


@dataclass(frozen=True, eq=False, repr=False)
class Run[C]:
    """One run of a graph: its inputs and the outcome of every call so far.

    A run you hold never changes. Each ``with_...`` method (``with_answers``, ``with_decline``, ``with_inputs``,
    ``with_forgotten``, ``with_max_items``) returns a new ``Run`` with that change and calls no node; ``resume`` and
    ``again`` return a new ``Run`` after their calls. Each leaves this one as it was, so keep what they return:
    ``run = run.with_answers(waiting, True).resume()``. Its attributes can't be set, and ``inputs`` can't be changed.
    ``to_json`` saves all of it.

    A run is made by ``graph.with_inputs(...)``, ``graph.run(...)`` or ``Run.from_json(saved, graph)``. Each checks
    what the run is made of.

    ``C`` is the class of the run's context, the same as the graph's: a ``Graph[HostContext]`` makes
    ``Run[HostContext]``. The context is the object ``resume``, ``stream`` and ``again`` take as ``context=``. A run
    never saves it.

    Attributes:
        graph: The graph the run runs.
        inputs: The graph inputs the caller gave, after they were checked, by name (``"text"``). It is read-only.
        max_items: The item limit: the most items a series a node is called per item of may have. Past it, the
            run stops as ``stopped``. ``None`` means no limit.
    """

    graph: Graph[C]
    max_items: int | None
    #: The outcome of every call so far.
    _outcomes: RunOutcomes
    #: The outcomes of the previous run that this run's calls may still reuse.
    _reusable: ReusableOutcomes
    #: How the last ``resume()`` left the run (see ``LastResume``).
    _last_resume: LastResume = None
    #: The nodes that were complete when the last ``resume()`` ended. A later stream doesn't report them complete
    #: again, and their reusable outcomes from the previous run are dropped.
    _complete_nodes: frozenset[NodePath] = frozenset()

    @property
    def inputs(self) -> Mapping[str, Any]:
        """The graph inputs the caller gave, by name. The run's outcomes hold them, read-only."""
        return self._outcomes.graph_inputs

    def __repr__(self) -> str:
        outcomes = sum(
            len(self._outcomes.of(path)) for path in self.graph._blueprint.function_nodes
        )
        return f"<Run {self.status}: {outcomes} outcomes>"

    @property
    def status(self) -> RunStatus:
        """How the run stands, such as ``ready``, ``waiting`` or ``complete``.

        ``ready`` means a call is left to make, ``waiting`` that a call waits for a person, and ``complete`` that
        every call has an outcome. A run that a ``resume()`` ended early says how: ``failed``, ``stopped``,
        ``cancelled`` or ``timed_out``. A run a person ended with ``with_decline`` is ``declined``. A copy made while a
        ``resume()`` is still going is ``running``. ``state.py`` has the whole rule.
        """
        return status_of(self._outcomes, self._last_resume)[0]

    @property
    def cause(self) -> Problem | None:
        """The problem that ended the run early or declined it, with its ``loc`` and code. ``None`` otherwise."""
        return status_of(self._outcomes, self._last_resume)[1]

    # -- running ----------------------------------------------------------------

    def with_max_items(self, max_items: int | None) -> Run[C]:
        """Returns a copy of this run with a new item limit. No node is called.

        A run stops as ``stopped`` when a series that a node is called per item of has more items than the limit. Say
        a run started with ``max_items=30`` met 40 items. ``run.with_max_items(100).resume()`` goes on from there.
        ``None`` removes the limit. The limit is saved with the run.

        Raises:
            InvalidCallError: ``invalid_limit`` if ``max_items`` isn't a whole number of at least 1, or ``None``.
        """
        return dataclasses.replace(
            self, max_items=LimitsValidator()({"max_items": max_items})["max_items"]
        )

    def resume(
        self,
        *,
        context: C | None = None,
        timeout: float | None = None,
        only_costs_nothing: bool = False,
    ) -> Run[C]:
        """Calls every node that is ready, and returns a new run once it has to stop.

        It stops when the run is complete, waits for a person, fails, or reaches its item limit or its ``timeout``.
        A ``declined`` run has ended for good: it calls no node, and stays ``declined``.

        The context isn't saved with a run, so pass it again on every call. Once ``timeout`` has passed, no new call
        starts. The calls already running on threads are waited for, up to their ``attempt_timeout``, and their
        outcomes are kept. So ``resume`` may return later than ``timeout``, but a call that returned is never made
        again.

        ``only_costs_nothing=True`` calls only the nodes whose ``Policy`` says ``costs_nothing``. A host uses it to
        learn how many calls each costly node will have, before any of them is made. Say only ``files`` costs nothing
        in ``files -> summarize``. Then ``graph.with_inputs({"names": ["a.pdf", "b.pdf"]}).resume(only_costs_nothing=True)``
        calls only ``files``, and ``progress.nodes["summarize"].total`` is 2. The run is ``ready`` afterwards, unless
        one of these calls fails, asks, or meets a series over the item limit. The next ``resume()`` doesn't make those
        calls again.

        Args:
            context: The run's context, given to every node that takes ``Context[...]``.
            timeout: The seconds after which no new call starts. ``None`` means no limit.
            only_costs_nothing: Whether to call only the nodes whose ``Policy`` says ``costs_nothing``.

        Returns:
            A new run, after those calls. The run you called it on is left as it was, so keep this one.

        Raises:
            InvalidCallError: ``invalid_limit_seconds`` if ``timeout`` isn't a number of seconds above 0.
                ``invalid_context`` if ``context`` isn't an instance of the registry's ``context_type``, and
                ``context_missing`` if it is ``None`` while a node takes it.
        """
        limits = LimitsValidator()({"timeout": timeout})
        return self._run_scheduler(
            self.graph._context_validator(context),
            None,
            deadline(limits["timeout"]),
            only_costs_nothing=only_costs_nothing,
        )

    def stream(
        self,
        *,
        context: C | None = None,
        timeout: float | None = None,
        only_costs_nothing: bool = False,
        cancel: AsyncioEvent | ThreadEvent | None = None,
        hand_back: AsyncioEvent | None = None,
        checkpoint_every: float | None = None,
    ) -> AsyncGenerator[Event]:
        """Does what ``resume`` does, and yields an event for each thing that happens.

        Use it as ``async for event in run.stream(context=...): ...``. It takes ``context``, ``timeout`` and
        ``only_costs_nothing`` as ``resume`` does. The nodes are called on a thread of their own, so a slow node never
        blocks the event loop. The last event is a ``RunEnd``, whose ``run`` is the run as it ends.

        It can also stop in three ways a host controls:

        - Set ``cancel`` to stop before the next call. Leaving the loop early stops it too, once the generator is
          closed. Python only closes an async generator when it is garbage collected, so close it yourself:
          ``async with contextlib.aclosing(run.stream()) as events:``, then loop over ``events``. ``cancel`` can be an
          ``asyncio.Event`` or a ``threading.Event``. A ``threading.Event`` can go in the context too, so a slow node
          can check it from any thread.
        - Set ``hand_back`` to stop the same way but leave the run running. This is for a host that shuts down and lets
          another process go on. The calls on threads are waited for and kept. The last event is then a
          ``RunCheckpoint`` whose run is ``running``, and there is no ``RunEnd``. If ``cancel`` is set as well, it wins.
        - Pass ``checkpoint_every`` (seconds) to get a ``RunCheckpoint`` at most that often while calls return, and one
          more when the stream stops with calls still on threads. If the process dies, a host that saves each
          checkpoint loses the calls in flight, and the outcomes recorded since the last checkpoint.

        Raises:
            InvalidCallError: Whatever ``resume`` raises, and ``invalid_limit_seconds`` if ``checkpoint_every`` isn't a number
                of seconds above 0. Both are raised at this call, before any event.
        """
        limits = LimitsValidator()({"timeout": timeout, "checkpoint_every": checkpoint_every})
        run_scheduler = functools.partial(
            self._run_scheduler, only_costs_nothing=only_costs_nothing
        )
        return stream(
            run_scheduler,
            self.graph._context_validator(context),
            limits["timeout"],
            cancel,
            hand_back,
            limits["checkpoint_every"],
        )

    def _run_scheduler(
        self,
        context: C | None,
        emit: Callable[[EventBase], None] | None,
        stop: Callable[[], LastResume] | None,
        checkpoint_every: float | None = None,
        only_costs_nothing: bool = False,
    ) -> Run[C]:
        """Runs the scheduler on a copy of this run's outcomes, and returns a new run with them.

        The new run has the outcomes the scheduler recorded and says how the scheduler stopped. The scheduler sends the
        events about each node. The events about the run itself (that it started, a checkpoint, how it ended) are sent
        here, since they carry the run.

        Args:
            context: The run's context, given to each node that takes it.
            emit: Called with each event, for ``stream``. ``None`` for ``resume``.
            stop: Called before each call. It says whether the run must stop now, and why.
            checkpoint_every: The fewest seconds between two checkpoints. ``None`` means no checkpoints.
            only_costs_nothing: Whether to call only the nodes whose ``Policy`` says ``costs_nothing``. Every call of
                any other node is treated as not ready.

        Returns:
            The new run, with the outcomes as the scheduler left them.

        Raises:
            RuntimeError: The scheduler wasn't stopped, yet it left a call untried that was ready. That is a bug in the
                scheduler.
        """
        # The scheduler records outcomes into copies of this run's outcomes and of what it may reuse. The run
        # returned at the end is made from them, with how the scheduler stopped and which nodes are complete.
        outcomes, reusable = self._outcomes.copy(), self._reusable.copy()
        nodes = self.graph._blueprint.function_nodes
        complete_before = self._complete_nodes
        node_events: NodeEvents | None = None
        checkpoint: Callable[[LastResume], None] | None = None
        if emit is not None:
            emit(RunStart(waiting=tuple(self.waiting)))
            node_events = NodeEvents(emit, outcomes, complete_before)
            if checkpoint_every is not None:
                listener, announced = emit, node_events.complete

                # A checkpoint lists the nodes announced complete so far, so a run read back from it doesn't
                # announce them again.
                def checkpoint(stopped: LastResume) -> None:
                    saved = dataclasses.replace(
                        self,
                        _outcomes=outcomes.copy(),
                        _reusable=reusable.copy(),
                        _last_resume="running" if stopped is None else stopped,
                        _complete_nodes=frozenset(
                            path for path, node in nodes.items() if node in announced
                        ),
                    )
                    listener(with_run(RunCheckpoint(), saved))

        stopped: LastResume
        if self.status == "declined":
            # A person ended the run for good, so no node is called and it stays declined.
            stopped = self._last_resume
        else:
            stopped = Scheduler(
                outcomes=outcomes,
                reusable=reusable,
                max_items=self.max_items,
                only_costs_nothing=only_costs_nothing,
                node_events=node_events,
                stop=stop,
                context=context,
                checkpoint=checkpoint,
                checkpoint_every=checkpoint_every,
            ).run()
        complete = frozenset(
            path
            for path, node in nodes.items()
            if path in complete_before or is_complete(call_count(outcomes, node))
        )
        # A complete node has no call left to reuse an outcome of the previous run.
        reusable.drop(complete)
        run = dataclasses.replace(
            self,
            _outcomes=outcomes.copy(),
            _reusable=reusable,
            _last_resume=stopped,
            _complete_nodes=complete,
        )
        if (
            stopped is None
            and not only_costs_nothing
            and len(complete) < len(nodes)
            and run.status == "ready"
        ):
            lacking = sorted(path for path in nodes if path not in complete)
            raise RuntimeError(
                f"the scheduler has nothing to try, and no outcome for every call of {lacking}"
            )
        # A run handed back is still running. It didn't end, and the host saves it to go on elsewhere.
        if emit is not None:
            if stopped == "running":
                emit(with_run(RunCheckpoint(), run))
            else:
                status, cause = status_of(run._outcomes, run._last_resume)
                emit(with_run(RunEnd(status=status, cause=cause), run))
        return run

    @overload
    def with_answers(self, waiting: Waiting | str, value: Any, /) -> Run[C]: ...

    @overload
    def with_answers(self, answers: Mapping[str, Any], /) -> Run[C]: ...

    def with_answers(self, waiting: Waiting | str | Mapping[str, Any], /, *value: Any) -> Run[C]:
        """Returns a new run with these answers given to the calls waiting for them. No node is called.

        ``run.with_answers(waiting, value)`` answers one call, and ``run.with_answers({waiting_id: value, ...})``
        answers several. Each answer is checked against the questions its call asked (``AnswerValidator`` has the
        rules). An answer becomes the call's outputs, and the call's outcome becomes ``Done``. The next ``resume``
        then calls the nodes that read those outputs. A run that ended early, even a cancelled one, takes answers the
        same way. A ``declined`` run waits on nothing, so every answer to it is ``not_waiting``. An empty mapping
        answers no call, so it changes nothing: the run is returned as it is.

        Args:
            waiting: The call to answer, as ``run.waiting`` lists it or by its id. Or a mapping of answers by waiting
                id.
            value: The answer, when ``waiting`` names one call. Give nothing here with a mapping.

        Returns:
            A new run with the answers in it. The run you called it on is left as it was, so keep this one:
            ``run = run.with_answers(waiting, True)``.

        Raises:
            TypeError: A call is named without its answer, or a mapping is given with one.
            InvalidCallError: Every fault at once, and the run is left as it was. ``answer_given_twice`` for a call
                answered twice in one ``with_answers``, and ``invalid_answer`` for an answer its question refuses. ``not_waiting`` for an id
                that names no waiting call. ``answer_stale`` for an id whose call now asks another question, with
                ``ctx.moved_to`` when that question now waits on another call.
        """
        match waiting, value:
            case ((Waiting(id=key) | str(key)), (one,)):
                given: Mapping[str, Any] = {key: one}
            case Mapping(), ():
                given = waiting
            case _:
                raise TypeError(
                    "run.with_answers takes a waiting call and its answer, or a mapping of waiting ids to answers"
                )
        problems: list[Problem] = []
        answered: dict[tuple[BlueprintFunctionNode, ItemIndexes], Outcome] = {}
        # Working out the status looks at every outcome, so it is only done when a refusal needs it.
        status = cache(lambda: self.status)
        for waiting_id, one in given.items():
            try:
                node, item_indexes, loc, found = self._waiting_call(waiting_id, status)
            except InvalidCallError as refused:
                problems.extend(refused.problems)
                continue
            if (node, item_indexes) in answered:
                problems.append(call_problems.answer_given_twice(*loc, waiting_id=waiting_id))
                continue
            try:
                read = AnswerValidator(
                    node.compiled_node.outputs,
                    found.choices,
                    found.defaults,
                    loc=loc,
                    waiting_id=waiting_id,
                )(one)
            except InvalidCallError as refused:
                problems.extend(refused.problems)
                answered[(node, item_indexes)] = found
                continue
            answered[(node, item_indexes)] = Done(read)
        if problems:
            raise InvalidCallError(problems)
        # A cancel or a timeout is dropped only once a call is answered after it.
        if not answered:
            return self
        return dataclasses.replace(
            self,
            _outcomes=self._outcomes.replacing(answered),
            _last_resume=kept_by_an_answer(self._last_resume),
        )

    def with_decline(self, waiting: Waiting | str, /, *, reason: str | None = None) -> Run[C]:
        """Returns a new run in which a person declined to answer a waiting call, and so ended the run for good.

        The run is ``declined``, with ``answer_declined`` as its ``cause``. Every call made before keeps its
        outcome, so ``run.results`` still shows what ran. But no call is made any more: nothing waits, ``resume``
        and ``stream`` call no node, and a later ``with_answers`` or ``with_decline`` is refused as ``not_waiting``,
        as on a complete run. To try again, start a new run with ``again`` or ``with_inputs``, which reuses what this
        one ran and asks again. Say ``approve`` waits under ``"approve@1~3fa2c0d1"``. Then
        ``run.with_decline("approve@1~3fa2c0d1", reason="Over budget")`` is ``declined``, and its
        ``cause.ctx["reason"]`` is ``"Over budget"``.

        Args:
            waiting: The call to decline, as ``run.waiting`` lists it or by its id.
            reason: Why the person declined, for the host to show. It goes in the problem's ``ctx``, and the
                message doesn't quote it.

        Returns:
            A new run, ``declined``. The run you called it on is left as it was, so keep this one.

        Raises:
            InvalidCallError: ``not_waiting`` for an id that names no waiting call, and ``answer_stale`` for an id
                whose call now asks another question, as ``with_answers`` refuses them. ``invalid_decline_reason``
                for a reason that isn't text JSON can write, or ``None``. It is checked as its JSON text
                would be, so ``bytes`` are taken as their base64 text and a ``str`` enum member as its text, and ``1.5`` and a ``str``
                with a lone surrogate are refused. The run is left as it was.
        """
        waiting_id = waiting.id if isinstance(waiting, Waiting) else waiting
        _, _, loc, _ = self._waiting_call(waiting_id, cache(lambda: self.status))
        reason = DeclineValidator(loc=loc, waiting_id=waiting_id)(reason)
        cause = run_problems.answer_declined(*loc, reason=reason)
        # Nothing waits on a run that has ended: the declined question and every other one lose their outcome.
        asked = [(node, item_indexes) for node, item_indexes, _ in self._outcomes.calls(Unanswered)]
        return dataclasses.replace(
            self,
            _outcomes=self._outcomes.replacing(removing=asked),
            _last_resume=EndedEarly(status="declined", cause=cause),
        )

    @staticmethod
    def queued_as_json(
        answers: Mapping[str, Any] = MappingProxyType({}),
        /,
        *,
        declines: Mapping[str, Any] = MappingProxyType({}),
    ) -> tuple[dict[str, Any], dict[str, str | None]]:
        """Turns the answers and declines a host queues for a run into the JSON data the run will be given.

        A server can't give a running run an answer or a decline, so it queues them as JSON until the run stops.
        Each answer becomes the JSON data ``run.with_answers`` reads it from, and each reason the text
        ``run.with_decline`` keeps. What JSON can't write is refused now: a reason with the very problem
        ``run.with_decline`` gives it, and an answer as a whole, since its questions aren't known here
        (``run.with_answers`` refuses the same value at its question). What depends on the question, such as an
        answer of the wrong type, is checked when the run is given the data.

        It is provisional: the name and shape may change before 2.0.

        Args:
            answers: Answers by waiting id, as ``run.with_answers`` takes them.
            declines: Reasons by waiting id, as ``run.with_decline`` takes them.

        Returns:
            The answers and the reasons by waiting id, as the run is to be given them. For example
            ``{"approve~3fa2c0d1": "2026-11-15"}`` for an answer of ``date(2026, 11, 15)``.

        Raises:
            InvalidCallError: Every refusal at once, each at the call its waiting id names. ``invalid_answer`` for
                an answer JSON can't write, such as a ``str`` with a lone surrogate. ``invalid_decline_reason``
                for a reason that isn't a ``str`` JSON can write, or ``None``.
        """
        problems: list[Problem] = []
        queued: dict[str, Any] = {}
        reasons: dict[str, str | None] = {}
        for waiting_id, answer in answers.items():
            try:
                queued[waiting_id] = AnswerAsJsonValidator(
                    loc=parse_waiting_id(waiting_id)[0], waiting_id=waiting_id
                )(answer)
            except InvalidCallError as refused:
                problems.extend(refused.problems)
        for waiting_id, reason in declines.items():
            try:
                reasons[waiting_id] = DeclineValidator(
                    loc=parse_waiting_id(waiting_id)[0], waiting_id=waiting_id
                )(reason)
            except InvalidCallError as refused:
                problems.extend(refused.problems)
        if problems:
            raise InvalidCallError(problems)
        return queued, reasons

    def _waiting_call(
        self, waiting_id: str, status: Callable[[], RunStatus]
    ) -> tuple[BlueprintFunctionNode, ItemIndexes, Loc, Unanswered]:
        """Finds the call that ``waiting_id`` names, still waiting on the question the id was written for.

        ``with_answers`` and ``with_decline`` name a call this way.

        Args:
            waiting_id: The id, as a host gave it.
            status: Returns the run's status, for a refusal's message. It is only called when the id is refused.

        Returns:
            The call's node, its item indexes, its loc, and its outcome.

        Raises:
            InvalidCallError: ``not_waiting`` for an id that names no waiting call. ``answer_stale`` for an id whose
                call now asks another question, with ``ctx.moved_to`` when that question now waits on another call.
        """
        loc, nodes, item_indexes, fingerprint = WaitingIdValidator(status=status)(waiting_id)
        # None when the id names a subgraph, a node the graph doesn't have, or no node at all ("" reads as ())
        node = self.graph._blueprint.function_nodes.get(nodes)
        outcome = self._outcomes.of(nodes).get(item_indexes) if node is not None else None
        match node, outcome:
            case BlueprintFunctionNode(), Unanswered() as found if (
                outcome_loc(node, item_indexes) == loc
            ):  # each item index in the id comes right after the node whose loop it counts
                pass
            case _, Unanswered() | Done() | Skipped() | NoneInput() | None:
                moved_to = self._moved_to(nodes, fingerprint) if fingerprint is not None else None
                raise InvalidCallError(
                    [
                        call_problems.not_waiting(
                            *loc, waiting_id=waiting_id, status=status(), moved_to=moved_to
                        )
                    ]
                )
            case _:
                assert_never(outcome)
        asked = question_fingerprint(node, item_indexes, self._outcomes)
        if fingerprint is not None and asked != fingerprint:
            raise InvalidCallError(
                [
                    call_problems.answer_stale(
                        *loc,
                        waiting_id=waiting_id,
                        current=write_waiting_id(outcome_loc(node, item_indexes), asked),
                        moved_to=self._moved_to(nodes, fingerprint),
                    )
                ]
            )
        return node, item_indexes, loc, found

    def _moved_to(self, nodes: NodePath, fingerprint: str) -> str | None:
        """Returns the id of the call where a stale question now waits, or ``None``.

        A waiting id names a call and the question it asked. After an edit, the same question may be waiting on another
        call of the node at ``nodes``. This finds it by the question's fingerprint.
        """
        # None when the id names a subgraph, a node the graph doesn't have, or no node at all ("" reads as ())
        node = self.graph._blueprint.function_nodes.get(nodes)
        if node is None:
            return None
        for item_indexes, outcome in self._outcomes.of(nodes).items():
            if (
                isinstance(outcome, Unanswered)
                and question_fingerprint(node, item_indexes, self._outcomes) == fingerprint
            ):
                return write_waiting_id(outcome_loc(node, item_indexes), fingerprint)
        return None

    def with_inputs(
        self,
        inputs: Mapping[str, Any] = MappingProxyType({}),
        *,
        max_items: int | None = None,
    ) -> Run[C]:
        """Returns a new run with some inputs changed and the rest as before, ready to ``resume`` or ``stream``.

        The new run may reuse this run's outcomes. A call that gets the same arguments as a call of its node did before
        takes that call's outcome, and the node isn't called for it. That includes a person's answer, and a question
        still waiting. Inputs are given and checked as ``graph.with_inputs`` takes them. ``python -m conductor.about
        running-again`` has the rules for what is reused.

        ``max_items`` left out, or ``None``, keeps this run's item limit. ``run.with_max_items(None)`` removes it.
        """
        return self.graph.with_inputs(
            {**self.inputs, **inputs},
            max_items=max_items if max_items is not None else self.max_items,
            previous=self,
        )

    def with_forgotten(self, *node_ids: str) -> Run[C]:
        """Returns a run of the same inputs that has forgotten these nodes' outcomes, ready to ``resume`` or ``stream``.

        Only the forgotten nodes are called again. Everything else is reused as ``with_inputs`` reuses it, so a node
        downstream that then gets the same arguments as before isn't called. A host forgets a node when it has to
        check the node's access again.
        """
        node_ids = ForgetValidator(self.graph._nodes)(node_ids)
        fresh = self.with_inputs()
        # Forgetting a subgraph node forgets every node inside it.
        fresh._reusable.drop(
            {path for path in self.graph._blueprint.function_nodes if path[0] in node_ids}
        )
        return fresh

    def again(
        self,
        inputs: Mapping[str, Any] = MappingProxyType({}),
        *,
        max_items: int | None = None,
        context: C | None = None,
        timeout: float | None = None,
    ) -> Run[C]:
        """Runs again with some inputs changed and the rest as before, and returns the new run.

        It is ``run.with_inputs(...).resume(context=context, timeout=timeout)``. A call that gets the same arguments as
        before isn't made again: its outcome is reused, whatever the context is.
        """
        return self.with_inputs(inputs, max_items=max_items).resume(
            context=context, timeout=timeout
        )

    # -- what a host reads ------------------------------------------------------

    @property
    def waiting(self) -> list[Waiting]:
        """Every call waiting for a person, in node order and then item order.

        A call inside a subgraph is listed where its subgraph node is.
        """
        return [
            waiting_record(node, item_indexes, outcome, self._outcomes)
            for node, item_indexes, outcome in self._outcomes.calls(Unanswered)
        ]

    @property
    def progress(self) -> RunProgress:
        """How far the run is: for the run and for each node, how many calls are finished, waiting and expected.

        ``progress.py`` explains the numbers. They are worked out from the outcomes each time, so a run read back with
        ``Run.from_json`` has the same progress.

        The total of a node is ``None`` while the series that decides it doesn't exist yet. In the README graph, before
        any call, ``run.progress.nodes["approve"].total`` is ``None``: how many invoices there are is only known once
        ``read`` has returned. If ``read``'s ``Policy`` says ``costs_nothing``, ``run.resume(only_costs_nothing=True)``
        calls it first.
        """
        return progress_of(self._outcomes)

    @property
    def none_input_calls(self) -> list[NoneInputCall]:
        """The calls that weren't made because an input got ``None``: which node, which item and which input.

        A call inside a subgraph is listed where its subgraph node is.
        """
        return [
            NoneInputCall(loc=outcome_loc(node, item_indexes), input=outcome.input)
            for node, item_indexes, outcome in self._outcomes.calls(NoneInput)
        ]

    @property
    def results(self) -> dict[str, dict[str, Any]]:
        """Every node's outputs so far, by node id and output name. It works whatever the run's status.

        A node called once shows what it returned. A node called per item shows a list with one entry per item, and
        ``None`` for an item with no value yet. A node called per item of a list that is itself made per item shows a
        list of lists. Say three documents are split into lines, and the second document is ``None``, so it is never
        split. Then the lines show as ``[["A", "B"], None, ["C"]]``. An empty second document shows ``[]`` there.
        ``None`` in place of a list means the list being looped over was never made, or no item of it could have a
        value.
        """
        return results_of(self.graph, self._outcomes)

    @property
    def outputs(self) -> dict[str, Any]:
        """The graph's outputs by name, once the run is complete.

        ``results`` works on a run that isn't complete.
        """
        status, cause = status_of(self._outcomes, self._last_resume)
        if status != "complete":
            reason = cause.msg if cause is not None else ""
            raise InvalidCallError(
                [call_problems.run_not_complete("outputs", status=status, reason=reason)]
            )
        return outputs_of(self.graph, self._outcomes)

    def outputs_to_json(self) -> str:
        """Returns ``run.outputs`` as JSON text, such as ``{"result":"A!+B!+C!"}`` for the README graph.

        Each value is written by its output's type, the same way a saved run writes it. A table is written as its
        columns, ``inf`` and ``nan`` as text, and a host's own type by its own schema.

        Raises:
            InvalidCallError: ``run_not_complete`` if the run isn't complete. ``run_unwritable``, naming the output,
                if a value can't be written as JSON.
        """
        return outputs_to_json(self)

    # -- JSON -------------------------------------------------------------------

    @classmethod
    def from_json[R](cls, saved: str | bytes, graph: Graph[R]) -> Run[R]:
        """Returns the run that ``run.to_json()`` saved as ``saved``, on the same ``graph``.

        ``Run.from_json(run.to_json(), graph)`` has the inputs, the outcomes and the status of ``run``, in another
        process too. Each value is converted back by its type, so a date comes back from its text, for example.
        What the saved run could still reuse is kept only for a node whose ``Policy`` says ``reuse_when_inputs_match``
        in ``graph``.

        Raises:
            InvalidCallError: The graph's own problems, if it has any, before anything of ``saved`` is read.
                ``saved_run_of_another_graph`` if ``saved`` was written for another graph. ``saved_run_unreadable`` if
                ``saved`` isn't a saved run of this graph. Its ``where`` names the key, node, output and call number,
                and it quotes no value.
        """
        if graph.problems:
            raise InvalidCallError(graph.problems)
        return run_from_json(Run, saved, graph)

    def to_json(self) -> str:
        """Returns the whole run as JSON text.

        It has the inputs, how the last ``resume()`` left the run, every node's outcomes, and the outcomes it may still
        reuse from a previous run. ``json/type.py`` shows the JSON.

        Raises:
            InvalidCallError: ``run_unwritable``, naming the input or output, if a value can't be written as JSON.
        """
        return run_to_json(self)


def make_run[C](
    graph: Graph[C], inputs: dict[str, Any], max_items: int | None, previous: Run[Any] | None
) -> Run[C]:
    """Returns a run of ``graph`` that hasn't started: what ``graph.with_inputs({"text": "a\\nb"})`` returns.

    No node has an outcome yet. The inputs have already been checked. With ``previous``, the new run may reuse
    that run's outcomes (see ``Run.with_inputs``).
    """
    blueprint = graph._blueprint
    return Run(
        graph,
        max_items,
        RunOutcomes(blueprint, inputs),
        ReusableOutcomes.from_previous(previous._outcomes, previous._reusable, into=blueprint)
        if previous is not None
        else ReusableOutcomes(),
    )
