"""``Run``: a graph's inputs and the outcome of every call so far, as one value that writes itself as JSON.

For the README graph, ``graph.run(text="a\\nb")`` calls ``lines`` once, ``shout`` once per line and ``join`` once, and
the run keeps one outcome per call: ``Done`` (it returned, or a person answered for it), ``Skipped`` (it
is on a branch that was not chosen), ``NoneInput`` (an argument was ``None`` and its input does not accept ``None``)
or ``Unanswered`` (it asked a person and waits for an answer). A call with no outcome has not been made yet.
``resume()`` calls every node that is ready, until the run is complete, waiting, failed or stopped, so a run can
stop anywhere (a person to ask, a failure, the item limit, a timeout) and go on from its JSON in another process.

A node inside a subgraph has calls as any node has: one per item its subgraph node is called for, and in each of
those one per item of its own loops. The subgraph node is never called itself: what a run says of it is read from
the calls of the nodes inside. A place at any depth of subgraphs is one path of node ids and item indexes:
``("e", 0, "ask", 2)`` is the call of ``ask`` for its third item inside the first call of ``e``.
"""

from __future__ import annotations

import dataclasses
import functools
from asyncio import Event as AsyncioEvent
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import cache
from types import MappingProxyType
from typing import Any, Literal, overload

from conductor._internal.base import (
    ConductorModel,
    ItemIndexes,
    Loc,
    NodePath,
)
from conductor._internal.gate import read_limits
from conductor._internal.graph import Graph
from conductor._internal.problem import InvalidCallError, Problem, call_problems

from .arguments import ResultReader
from .blueprint import BlueprintFunctionNode, RunBlueprint, outcome_loc
from .events import (
    Event,
    EventBase,
    RunCheckpoint,
    RunEnd,
    RunStart,
)
from .json import outputs_to_json, run_from_json, run_to_json
from .outcome import (
    Done,
    NoneInput,
    NotReady,
    Outcome,
    Skipped,
    Unanswered,
)
from .progress import RunProgress, is_complete, progress_of
from .results import outputs_of, results_of
from .reuse import ReusableOutcomes
from .scheduler import Scheduler
from .state import LastResume, RunStatus, deadline, kept_by_an_answer, status_of
from .stream import stream
from .waiting import (
    Waiting,
    question_fingerprint,
    read_waiting_id,
    received,
    waiting_record,
    write_waiting_id,
)


class NotCalled(ConductorModel):
    """A call a node was not given because one of its arguments was ``None`` and the input does not accept ``None``.

    ``strip(line: str) -> str | None`` returns ``None`` for the line ``" "``, and ``shout(line: str)`` is wired to it:
    ``run.not_called`` lists ``NotCalled(loc=("shout", 1), input="line")``.

    Attributes:
        loc: Which call it was, as a ``Problem``'s ``loc`` says it: the node, then the index of its item in each
            loop the node is inside; inside a subgraph, the subgraph node and its item first.
        input: The input whose argument was ``None``.
    """

    loc: Loc
    input: str


@dataclass(eq=False, repr=False)
class Run[C]:
    """One run of a graph. Every call returns a new ``Run``; ``to_json`` is all of it.

    ``C`` is the class of the run's context, as the graph's (``Run[HostContext]`` for a ``Graph[HostContext]``): the
    object ``resume``, ``stream`` and ``again`` take as ``context=``. A run never saves it.

    A run is made by ``graph.with_inputs(...)``, ``graph.run(...)`` or ``Run.from_json(saved, graph)``, which check
    what it is made of.

    Attributes:
        graph: The graph the run runs on.
        inputs: The caller's inputs as checked, by the graph input's name (``"text"``).
        max_items: The most items any one Series a node works through may hold before the run stops as
            ``stopped``; ``None`` for no ceiling.
    """

    graph: Graph[C]
    inputs: dict[str, Any]
    max_items: int | None
    #: Each function node's outcomes by item indexes, in item order, by the node's path: ``_outcomes[("w",)]`` for a
    #: node of the graph itself, ``_outcomes[("e", "ask")]`` for one inside ``e``. Only the scheduler writes out of that
    #: order, and it sorts them again before it hands the run on.
    _outcomes: dict[NodePath, dict[ItemIndexes, Outcome]]
    #: The outcomes of the run this one went on from that its calls may reuse.
    _reusable: ReusableOutcomes
    #: How the last ``resume()`` left the run: ``"running"`` in a copy made while it went on, how it ended early, or
    #: ``None`` when it made every call that was ready.
    _last_resume: LastResume = None
    #: The nodes that were complete (``is_complete``) when the last ``resume()`` ended, so a later stream does not say
    #: them complete again, and their outcomes from the previous run are dropped.
    _complete_nodes: frozenset[NodePath] = frozenset()

    @property
    def _blueprint(self) -> RunBlueprint:
        """The blueprint of the run's graph: the function nodes the run calls."""
        return self.graph._blueprint

    def __repr__(self) -> str:
        return f"<Run {self.status}: {sum(len(outcomes) for outcomes in self._outcomes.values())} outcomes>"

    @property
    def status(self) -> RunStatus:
        """How the run stands: ``ready`` (a call is left to make), ``waiting`` (a call waits for a person) or
        ``complete``, as its outcomes say; ``running`` in a copy made while a ``resume()`` went on; or, while a call is
        left, how the last ``resume()`` ended it early: ``failed``, ``stopped`` (at the item limit), ``cancelled`` or
        ``timed_out``. A failed call and a series over the item limit stand until a ``resume()``; an answer reads a
        cancelled or timed-out run from its outcomes again."""
        return status_of(self._last_resume, self._blueprint.function_nodes.values(), self._outcomes)[0]

    @property
    def cause(self) -> Problem | None:
        """The problem that ended the run early, with its ``loc`` and its code; ``None`` when the run did not end early."""
        return status_of(self._last_resume, self._blueprint.function_nodes.values(), self._outcomes)[1]

    # -- going on ---------------------------------------------------------------

    def with_max_items(self, max_items: int) -> Run[C]:
        """A copy of this run with ``max_items`` as its item limit. No node is called; ``resume`` goes on.

        A run stops with status ``stopped`` when a list that a node is called
        per item of has more items than the limit. Example: a run started with
        ``max_items=30`` that met 40 items goes on with
        ``run.with_max_items(100).resume()``. The limit is saved by ``to_json()``.

        Raises:
            InvalidCallError: ``max_items`` is not a whole number of at least 1 (``invalid_limit``).
        """
        return dataclasses.replace(self, max_items=read_limits(max_items=max_items)["max_items"])

    def resume(self, *, context: C | None = None, timeout: float | None = None, only_costs_nothing: bool = False) -> Run[C]:
        """Go on: call every node that is ready, until the run is complete, waits for a person, fails, reaches its
        item limit or its ``timeout``, and return the run as it then is.

        ``context`` is the run's context, the object every node that takes ``Context[...]`` is given; a run never
        saves it, so it is passed again on every call. Past ``timeout`` no more calls start, but the calls already
        on threads are waited for, as long as their ``attempt_timeout``, and kept: so ``resume`` can return later
        than ``timeout``, and a call that returned is never made again.

        With ``only_costs_nothing=True`` only the nodes whose ``Policy`` says ``costs_nothing`` are called, so that
        ``run.progress`` says how many calls each costly node will have before any of them is made. For
        ``files -> summarize`` where only ``files`` costs nothing,
        ``graph.with_inputs(names=["a.pdf", "b.pdf"]).resume(only_costs_nothing=True)`` calls ``files`` alone,
        and its ``progress.nodes["summarize"].total`` is 2. The run is ``ready`` afterwards, unless every node costs
        nothing (then it is ``complete``), a series has more items than ``max_items`` (``stopped``), or one of
        those calls fails or asks a person (``failed`` or ``waiting``). The next ``resume()`` makes none of those
        calls again.

        Args:
            context: The run's context, given to every node that takes ``Context[...]``.
            timeout: The seconds after which no more calls start; ``None`` for no limit.
            only_costs_nothing: Whether to call only the nodes whose ``Policy`` says ``costs_nothing``.

        Returns:
            The run after those calls.

        Raises:
            InvalidCallError: ``timeout`` is not a number of seconds above 0 (``invalid_limit``); ``context`` is not an instance of
                the registry's ``context_type`` (``invalid_context``), or is ``None`` while a node takes it
                (``context_missing``).
        """
        limits = read_limits(timeout=timeout)
        return self._run_scheduler(self.graph._context_door(context), None, deadline(limits["timeout"]), only_costs_nothing=only_costs_nothing)

    def stream(
        self, *, context: C | None = None, timeout: float | None = None, only_costs_nothing: bool = False,
        cancel: AsyncioEvent | None = None, hand_back: AsyncioEvent | None = None, checkpoint_every: float | None = None,
    ) -> AsyncGenerator[Event]:
        """``resume``, with an event for each thing that happens: ``async for event in run.stream(context=...): ...``.

        It takes ``context``, ``timeout`` and ``only_costs_nothing`` as ``resume`` does. The nodes are called on a thread of their own, so a
        slow node never blocks the event loop. The last event is ``RunEnd``, whose ``run`` is the run as it then is.

        - Setting ``cancel`` stops before the next call, as does leaving the loop early once the stream is closed.
          Python closes an async generator only when it is collected, so hold it with
          ``async with contextlib.aclosing(run.stream()) as events:`` and loop over ``events``.
        - Setting ``hand_back`` stops the same way and leaves the run running, for a host that shuts down and lets
          another process go on: the calls on threads are waited for and kept, and the last event is a
          ``RunCheckpoint`` whose ``run`` is ``running``, with no ``RunEnd``. A ``cancel`` set as well wins.
        - With ``checkpoint_every`` (seconds), a ``RunCheckpoint`` comes at most that often while calls return, and
          once when the stream stops with calls still on threads: a host that stores its ``run`` loses only the
          calls in flight if the process dies.

        Raises:
            InvalidCallError: As ``resume`` does, and for a ``checkpoint_every`` that is not a number of seconds
                above 0 (``invalid_limit``), at this call, before any event.
        """
        limits = read_limits(timeout=timeout, checkpoint_every=checkpoint_every)
        run_scheduler = functools.partial(self._run_scheduler, only_costs_nothing=only_costs_nothing)
        return stream(run_scheduler, self.graph._context_door(context), limits["timeout"], cancel, hand_back, limits["checkpoint_every"])

    def _run_scheduler(
        self, context: C | None, emit: Callable[[EventBase], None] | None, stop: Callable[[], LastResume] | None,
        checkpoint_every: float | None = None, only_costs_nothing: bool = False,
    ) -> Run[C]:
        """Runs the scheduler on a copy of this run and returns the copy with the outcomes it wrote and how it left the run.

        The scheduler emits the events about each node. The events about the run itself (that it started, a
        checkpoint of it, how it ended) are emitted here, since they carry the run.

        Args:
            context: The run's context, passed to each node that takes it.
            emit: Called with each event, for ``stream``; ``None`` for ``resume``.
            stop: Asked before each call whether the run must stop now, and why.
            checkpoint_every: The least number of seconds between two checkpoints; ``None`` for none.
            only_costs_nothing: Whether only the nodes whose ``Policy`` says ``costs_nothing`` are called: every
                call of another node is not ready.

        Returns:
            The copy, as the scheduler left it.

        Raises:
            RuntimeError: The scheduler did not stop early, and left a call that was ready untried: the run is
                ``ready`` after a ``resume()`` that calls every node.
        """
        run = self._copy()
        checkpoint: Callable[[LastResume], None] | None = None
        if emit is not None:
            emit(RunStart(waiting=tuple(run.waiting)))
            if checkpoint_every is not None:
                listener = emit

                def checkpoint(stopped: LastResume) -> None:
                    listener(RunCheckpoint(run=dataclasses.replace(run._copy(), _last_resume="running" if stopped is None else stopped)))

        blueprint = self._blueprint
        nodes = blueprint.function_nodes
        complete_before = run._complete_nodes
        run._last_resume = stopped = Scheduler(
            nodes=list(nodes.values()), threads=blueprint.threads, outcomes=run._outcomes, reusable=run._reusable,
            complete=complete_before,
            max_items=run.max_items, arguments_for=_only_costs_nothing(run._reader.arguments_for) if only_costs_nothing else run._reader.arguments_for,
            emit=emit, stop=stop, context=context, checkpoint=checkpoint, checkpoint_every=checkpoint_every,
        ).run()
        outcomes = run._outcomes
        run._complete_nodes = frozenset(path for path, node in nodes.items() if path in complete_before or is_complete(outcomes, node))
        run._reusable.drop(run._complete_nodes)  # a complete node has no call left to take them
        if stopped is None and not only_costs_nothing and len(run._complete_nodes) < len(nodes) and run.status == "ready":
            lacking = sorted(path for path in nodes if path not in run._complete_nodes)
            raise RuntimeError(f"the scheduler has nothing to try, and no outcome for every call of {lacking}")
        if emit is not None:  # a run handed back is still running: it did not end, and the host stores it to go on elsewhere
            emit(RunCheckpoint(run=run) if stopped == "running" else RunEnd.of(run))
        return run

    @overload
    def answer(self, waiting: Waiting | str, value: Any, /) -> Run[C]: ...

    @overload
    def answer(self, answers: Mapping[str, Any], /) -> Run[C]: ...

    def answer(self, waiting: Waiting | str | Mapping[str, Any], /, *value: Any) -> Run[C]:
        """Give waiting calls their answers, checked against their questions; nothing runs until ``resume``.

        ``run.answer(waiting, value)`` answers one, ``run.answer({waiting_id: value, ...})`` several. One question
        is answered with its value, several (a node with several outputs) with a mapping by question name, where a
        question with a default or a blank-taking type without choices may be left out, and of an
        ``exclusive_group`` exactly one is answered; a question limited by choices is answered unless it has a
        default. The answer is the call's outputs, and its outcome ``Done``; ``resume`` then calls the nodes
        that read them. A run that ended early takes answers as a waiting one does, a cancelled one included.

        Args:
            waiting: The call to answer, as ``run.waiting`` lists it or by its id; or the answers by waiting id.
            value: The answer to ``waiting`` when it names one call, given exactly once; nothing with a mapping.

        Returns:
            A copy of the run with the answers in it.

        Raises:
            TypeError: A call is named without its answer, or a mapping is given one.
            InvalidCallError: Every fault at once, and the run is left as it was: a call answered twice in one
                call of ``answer`` (``invalid_answer``), an id that names nothing waiting (``not_waiting``), one
                whose call now asks another question (``answer_stale``, with ``ctx.moved_to`` when that question
                waits elsewhere), or an answer its question refuses.
        """
        match waiting, value:
            case (Waiting(id=key) | str(key)), (one,):
                given: Mapping[str, Any] = {key: one}
            case Mapping(), ():
                given = waiting
            case _:
                raise TypeError("run.answer takes a waiting call and its answer, or a mapping of waiting ids to answers")
        problems: list[Problem] = []
        answered: dict[tuple[BlueprintFunctionNode, ItemIndexes], Outcome] = {}
        arguments_for = self._reader.arguments_for
        status = cache(lambda: self.status)  # read from every outcome, so only when a refusal names it
        for waiting_id, one in given.items():
            try:
                loc, nodes, item_indexes, fingerprint = read_waiting_id(waiting_id, status=status)
            except InvalidCallError as refused:
                problems.extend(refused.problems)
                continue
            node = self._blueprint.function_nodes.get(nodes)  # None for an id of a subgraph, or of no node
            found = self._outcomes[nodes].get(item_indexes) if node is not None else None
            if node is None or type(found) is not Unanswered or outcome_loc(node, item_indexes) != loc:  # the indexes stand after the node of their loop
                moved_to = self._moved_to(nodes, fingerprint) if fingerprint is not None else None
                problems.append(call_problems.not_waiting(*loc, waiting_id=waiting_id, status=status(), moved_to=moved_to))
                continue
            compiled_node = node.compiled_node
            asked = question_fingerprint(compiled_node, received(node, item_indexes, arguments_for))
            if fingerprint is not None and asked != fingerprint:
                current = write_waiting_id(outcome_loc(node, item_indexes), asked)
                problems.append(call_problems.answer_stale(*loc, waiting_id=waiting_id, current=current, moved_to=self._moved_to(nodes, fingerprint)))
                continue
            if (node, item_indexes) in answered:
                problems.append(call_problems.invalid_answer(*loc, waiting_id=waiting_id, reason="the call is answered twice"))
                continue
            try:
                read = compiled_node.answer_door(one, found.choices, found.defaults, loc=loc, waiting_id=waiting_id)
            except InvalidCallError as refused:
                problems.extend(refused.problems)
                answered[(node, item_indexes)] = found
                continue
            answered[(node, item_indexes)] = Done(read)
        if problems:
            raise InvalidCallError(problems)
        run = dataclasses.replace(self._copy(share_outcomes=True), _last_resume=kept_by_an_answer(self._last_resume))
        for changed in {node for node, _ in answered}:  # its own outcomes for each node it changes; every other node's are shared
            run._outcomes[changed.node_path] = dict(self._outcomes[changed.node_path])
        for (node, item_indexes), now in answered.items():
            run._outcomes[node.node_path][item_indexes] = now
        return run

    def _moved_to(self, nodes: NodePath, fingerprint: str) -> str | None:
        """The id of the call where the question a stale id names now waits, if it waits elsewhere on the node at
        the path ``nodes``."""
        node = self._blueprint.function_nodes.get(nodes)  # None for an id of a subgraph, or of no node
        if node is None:
            return None
        arguments_for = self._reader.arguments_for
        for item_indexes, outcome in self._outcomes[nodes].items():
            if type(outcome) is Unanswered and question_fingerprint(node.compiled_node, received(node, item_indexes, arguments_for)) == fingerprint:
                return write_waiting_id(outcome_loc(node, item_indexes), fingerprint)
        return None

    def with_inputs(self, inputs: Mapping[str, Any] = MappingProxyType({}), /, *, max_items: int | None = None, **by_name: Any) -> Run[C]:
        """A run of these changed inputs, the rest as before, ready to ``resume`` or ``stream``.

        It carries this run's outcomes, and those this run carried from the previous run and did not use for a
        node that hasn't finished (a finished node's are dropped, so a save stays its size over edits): a call that
        receives what a call of its node received before takes that call's outcome, an answer or a question still
        waiting included, and the node is not called for it. Inputs are given as ``graph.with_inputs`` takes them,
        and checked the same way.
        """
        return self.graph.with_inputs({**self.inputs, **inputs, **by_name}, max_items=max_items if max_items is not None else self.max_items, previous=self)

    def forget(self, *node_ids: str) -> Run[C]:
        """A run of the same inputs that has forgotten these nodes' work, ready to ``resume`` or ``stream``.

        Every other outcome is carried as ``with_inputs`` carries it, so only the
        forgotten nodes are called again, and a node below them that then
        receives what it received before is not. A host forgets a node whose
        access it must check again.
        """
        unknown = [n for n in node_ids if n not in self.graph._nodes]
        if unknown:
            raise InvalidCallError([call_problems.node_not_in_graph(n, nodes=list(self.graph._nodes)) for n in unknown])
        fresh = self.with_inputs()
        fresh._reusable.drop({path for path in self._blueprint.function_nodes if path[0] in node_ids})  # a subgraph's id forgets every node inside it
        return fresh

    def again(
        self, inputs: Mapping[str, Any] = MappingProxyType({}), /, *, max_items: int | None = None, context: C | None = None,
        timeout: float | None = None, **by_name: Any,
    ) -> Run[C]:
        """Run again with changed inputs, the rest as before: ``run.with_inputs(...).resume(context=context, timeout=timeout)``.

        A call whose arguments are the ones it had before is not made again: its outcome is reused, whatever the
        context is.
        """
        return self.with_inputs(inputs, max_items=max_items, **by_name).resume(context=context, timeout=timeout)

    # -- reading ----------------------------------------------------------------

    @property
    def waiting(self) -> list[Waiting]:
        """Every call waiting for a person, in node then item order; those inside a subgraph where its
        node stands."""
        arguments_for = self._reader.arguments_for
        return [waiting_record(node, item_indexes, outcome, arguments_for) for node, item_indexes, outcome in self._waiting_outcomes()]

    def _waiting_outcomes(self) -> Iterator[tuple[BlueprintFunctionNode, ItemIndexes, Unanswered]]:
        """Every call waiting for a person, with its node and item indexes, in node then item order."""
        for node in self._blueprint.function_nodes.values():
            for item_indexes, outcome in self._outcomes[node.node_path].items():
                if type(outcome) is Unanswered:
                    yield node, item_indexes, outcome

    @property
    def progress(self) -> RunProgress:
        """How far the run is: for the run and for each node, how many calls have an outcome, how many wait for a
        person, how many are known so far, and how many there will be (``None`` while a series that decides it
        does not exist yet).

        It is read from the run's outcomes each time, so a run read back with ``Run.from_json`` has the same.
        For the README graph before any call, ``run.progress.nodes["approve"].total`` is ``None``: how many
        invoices there are is known once ``read`` has returned. ``run.resume(only_costs_nothing=True)`` calls ``read``
        first when its ``Policy`` says ``costs_nothing``.
        """
        return progress_of(self._blueprint, self._outcomes)

    @property
    def not_called(self) -> list[NotCalled]:
        """The calls a node didn't get because an input was blank: which node, which item and which input; those
        inside a subgraph where its node stands."""
        return [
            NotCalled(loc=outcome_loc(node, item_indexes), input=outcome.input)
            for node in self._blueprint.function_nodes.values() for item_indexes, outcome in self._outcomes[node.node_path].items() if type(outcome) is NoneInput
        ]

    @property
    def results(self) -> dict[str, dict[str, Any]]:
        """Every node's outputs so far, by node id and output name; it can be read whatever the run's status.

        A node called once shows its return value. A node called per item shows a list with one entry per
        item, ``None`` for an item with no value yet. A node called per item of a list that is itself made
        per item shows a list of lists: three documents split into lines, the second empty, give
        ``[["A", "B"], None, ["C"]]``. ``None`` instead of a list means the list being looped over was never
        made, or no item of it could have a value.
        """
        return results_of(self.graph, self._blueprint, self._outcomes, self.inputs)

    @property
    def outputs(self) -> dict[str, Any]:
        """The graph's outputs (``graph.outputs``), by name, once the run is complete; ``results`` reads a run in any state."""
        if (status := self.status) != "complete":
            reason = cause.msg if (cause := self.cause) is not None else ""
            raise InvalidCallError([call_problems.run_not_complete("outputs", status=status, reason=reason)])
        return outputs_of(self.graph, self._blueprint, self._outcomes, self.inputs)

    def outputs_to_json(self) -> str:
        """Returns ``run.outputs`` as JSON text, by name: ``{"result":"A!+B!+C!"}`` for the README graph.

        Each value is written by its output's type, as in a saved run: a table as its columns, ``inf`` and
        ``nan`` as text, a host type by its own schema.

        Raises:
            InvalidCallError: The run is not complete (``run_not_complete``), or a value cannot be written as
                JSON (``run_unwritable``, naming the output).
        """
        return outputs_to_json(self)

    # -- what a call reads ------------------------------------------------------

    @property
    def _reader(self) -> ResultReader:
        """Reads this run's outcomes for one item: what a call of a node receives there."""
        return ResultReader(self._outcomes, self.inputs)

    # -- JSON -------------------------------------------------------------------

    @classmethod
    def from_json[R](cls, saved: str | bytes, graph: Graph[R]) -> Run[R]:
        """Returns the run that ``run.to_json()`` wrote as ``saved``, on the same ``graph``.

        ``Run.from_json(run.to_json(), graph)`` has the inputs, the outcomes and the status of ``run``, also in
        another process. Each value is read the way its type reads JSON: a date from its text, for example.

        Raises:
            InvalidCallError: ``graph`` has problems (they are raised, and nothing of ``saved`` is read);
                ``saved`` was written for another graph (``saved_run_of_another_graph``); or ``saved`` is not a
                saved run of this graph (``saved_run_unreadable``, whose ``where`` names the key, the node, the
                output and the call number, and which quotes no value).
        """
        if graph.problems:
            raise InvalidCallError(graph.problems)
        return run_from_json(Run, saved, graph, graph._blueprint, graph._codec)

    def to_json(self) -> str:
        """Returns the whole run as JSON text: its inputs, how the last ``resume()`` left it, every node's outcomes, and the outcomes
        it may still reuse from the previous run.

        Raises:
            InvalidCallError: A value cannot be written as JSON (``run_unwritable``, naming the input or the
                output).
        """
        return run_to_json(self)

    def _copy(self, *, share_outcomes: bool = False) -> Run[C]:
        """A copy to change: its own outcomes to write into and its own reusable outcomes to take from; or with
        ``share_outcomes`` this run's dict of outcomes for each node and its reusable outcomes, for a caller that
        replaces whole nodes and takes nothing.

        Args:
            share_outcomes: Whether the copy shares this run's dict of outcomes for each node and its reusable outcomes.
        """
        if share_outcomes:
            return dataclasses.replace(self, _outcomes=dict(self._outcomes))
        return dataclasses.replace(self, _outcomes={name: dict(held) for name, held in self._outcomes.items()}, _reusable=self._reusable.copy())


RunEnd.model_rebuild(_types_namespace={"Run": Run})
RunCheckpoint.model_rebuild(_types_namespace={"Run": Run})


def _only_costs_nothing(
    arguments_for: Callable[[BlueprintFunctionNode, ItemIndexes], dict[str, Any] | Skipped | NoneInput | Literal[NotReady.NOT_READY]],
) -> Callable[[BlueprintFunctionNode, ItemIndexes], dict[str, Any] | Skipped | NoneInput | Literal[NotReady.NOT_READY]]:
    """``arguments_for``, with every call of a node whose ``Policy`` does not say ``costs_nothing`` not ready: what
    ``resume(only_costs_nothing=True)`` hands the scheduler, so that it calls ``lines`` and never ``shout`` when only
    ``lines`` costs nothing."""

    def restricted(node: BlueprintFunctionNode, item_indexes: ItemIndexes) -> dict[str, Any] | Skipped | NoneInput | Literal[NotReady.NOT_READY]:
        return arguments_for(node, item_indexes) if node.policy.costs_nothing else NotReady.NOT_READY

    return restricted


def make_run[C](graph: Graph[C], inputs: dict[str, Any], max_items: int | None, previous: Run[Any] | None) -> Run[C]:
    """Returns a run of ``graph`` that has not started, with the inputs the inputs door let in: what
    ``graph.with_inputs(text="a\\nb")`` returns. No node has an outcome yet; with ``previous``, the run may reuse
    that run's outcomes (``Run.with_inputs``)."""
    blueprint = graph._blueprint
    outcomes: dict[NodePath, dict[ItemIndexes, Outcome]] = {path: {} for path in blueprint.function_nodes}
    return Run(graph, inputs, max_items, outcomes, ReusableOutcomes.from_run(previous, into=blueprint) if previous is not None else ReusableOutcomes())
