"""``Run``: a graph's inputs and every unit's result, as one value that writes itself as JSON.

A unit is one node in one instance of its scope: the node's one call, or its
call for one item. A unit is ``Done`` (its outputs kept), ``Answered`` (done
by a person), ``Skipped`` (a branch not taken), ``NotCalled`` (an input got
nothing, so the node wasn't called) or ``outcomes.Waiting`` (it waits for a person). A
unit with no entry hasn't run yet. Going on is one leg: start every unit
whose inputs exist, until nothing more can start. So a run can stop anywhere
(a person to ask, a failure, the item ceiling, a timeout) and carry on from
its JSON in another process.

A node inside a subgraph has units as any node has: one for each item
its subgraph node runs for, and under it for each item of its own scope. Its
open inputs read what the subgraph node is given, and what reads the
subgraph node reads the unit inside that holds the output. The subgraph
node itself holds no unit: what a run says of it is read from the units
inside. Where something is, at any depth of subgraphs, is one path of
node ids and item positions: ``("e", 0, "ask", 2)`` is ``ask``'s third item
inside ``e``'s first.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import time
from asyncio import AbstractEventLoop, Queue
from asyncio import Event as AsyncioEvent
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from threading import Event as ThreadEvent
from threading import Thread
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast, overload

from conductor._internal import outcomes
from conductor._internal.address import (
    ItemIndexes,
    Loc,
    NodePath,
    node_path_text,
    parse_loc_text,
)
from conductor._internal.ask import (
    Waiting,
    answer_outputs,
    question_fingerprint,
    waiting_record,
)
from conductor._internal.compiled import (
    FlatFunctionNode,
    FlatSubgraph,
    display_plans_of,
    outcome_loc,
)
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.node_definition import context_values
from conductor._internal.outcomes import (
    Answered,
    Outcome,
    Skipped,
)
from conductor._internal.problem import InvalidCallError, Problem
from conductor._internal.problem import catalogue as problem
from conductor._internal.run.arguments import (
    DisplayReader,
    ResultReader,
)
from conductor._internal.run.json import outputs_to_json, run_from_json, run_to_json
from conductor._internal.run.reuse import (
    ReusableOutcomes,
    arguments_fingerprint,
    outer_items_fingerprint,
)
from conductor._internal.run.state import READY, RUNNING, EndedEarly, Going, RunState, RunStatus
from conductor._internal.scheduler import (
    Event,
    EventBase,
    RunCheckpoint,
    RunEnd,
    RunStart,
    Scheduler,
)

if TYPE_CHECKING:
    from conductor._internal.graph import Graph



#: ``run.answer``'s second argument when it is given a mapping of answers instead.
_NOT_GIVEN: Any = object()


class NotCalled(ConductorModel):
    """A call a node didn't get, because an input got a blank there.

    Attributes:
        loc: Which call it was, as a ``Problem``'s ``loc`` says it: the node, then the place of its item in each
            Series the node runs per item of; inside a subgraph, the subgraph node and its item first.
        input: The input that got nothing.
    """

    loc: Loc
    input: str


@dataclass(eq=False, repr=False)
class Run:
    """One run of a graph. Every call returns a new ``Run``; ``to_json`` is all of it.

    A run is made by ``graph.with_inputs(...)``, ``graph.run(...)`` or ``Run.from_json(saved, graph)``, which check
    what it is made of.

    Attributes:
        graph: The graph the run runs on.
        inputs: The caller's inputs as checked, by address (``"upper.text"``) whether they were given by name or by
            address.
        max_items: The most items any one Series a node works through may hold before the run stops as
            ``stopped``; ``None`` for no ceiling.
        state: How the run stands: ``Going`` (``ready`` before the first leg, then ``running``, ``waiting`` or
            ``complete``), or ``EndedEarly`` (``failed``, ``stopped`` at the item ceiling, ``cancelled`` or
            ``timed_out``) with the ``Problem`` that caused it.
    """

    graph: Graph
    inputs: dict[str, Any]
    max_items: int | None
    #: Each function's units by instance, in instance order, by the function's name: ``_outcomes["w"]`` for a node of
    #: the graph itself, ``_outcomes["e/ask"]`` for one inside ``e``. Only a leg records out of that order, and it puts
    #: the units back in it before it hands the run on.
    _outcomes: dict[str, dict[ItemIndexes, Outcome]]
    #: What each node may take from the run this one went on from.
    _reusable: dict[NodePath, ReusableOutcomes]
    state: RunState = READY
    #: For a node whose units a saved run made with other code or another signature, what that code was; they are never earlier work for a later run.
    _code_changed: dict[NodePath, str] = field(default_factory=dict[NodePath, str])
    #: The nodes every unit of which was finished when the last leg ended, so a later leg doesn't say them complete again.
    _finished_nodes: frozenset[NodePath] = frozenset()

    def __repr__(self) -> str:
        return f"<Run {self.status}: {sum(len(outcomes) for outcomes in self._outcomes.values())} outcomes>"

    @property
    def status(self) -> RunStatus:
        """How the run stands, as ``state`` says it."""
        return self.state.status

    # -- going on ---------------------------------------------------------------

    def with_max_items(self, max_items: int) -> Run:
        """A copy of this run with ``max_items`` as its item limit. No node is called; ``resume`` goes on.

        A run stops with status ``stopped`` when a list that a node is called
        per item of has more items than the limit. Example: a run started with
        ``max_items=30`` that met 40 items goes on with
        ``run.with_max_items(100).resume()``. The limit is saved by ``to_json()``.

        Raises:
            InvalidCallError: ``max_items`` is below 1 (``invalid_max_items``).
        """
        checked_max_items(max_items=max_items)
        return dataclasses.replace(self, max_items=max_items)

    def resume(self, *, context: Mapping[type, Any] = MappingProxyType({}), timeout: float | None = None) -> Run:
        """Go on: start every unit whose inputs exist, until the run is done, waits, fails, reaches its ceiling or its ``timeout``.

        ``context`` gives each ``Context`` type its value. Past its
        ``timeout`` a leg starts nothing more but waits for the calls already
        on threads, as far as their ``attempt_timeout``, and keeps them; so it
        can end later than ``timeout``, and a call that came back is never made
        again.
        """
        checked_max_items(timeout=timeout)
        return self._run_scheduler(context_values(self.graph._needs_context, context), None, _deadline(timeout))

    def stream(
        self, *, context: Mapping[type, Any] = MappingProxyType({}), timeout: float | None = None, cancel: AsyncioEvent | None = None,
        hand_back: AsyncioEvent | None = None, checkpoint_every: float | None = None,
    ) -> AsyncGenerator[Event]:
        """``resume``, saying what happens as it happens: ``async for event in run.stream(): ...``.

        The leg runs on its own thread, so a slow node never holds the event
        loop. The last event is ``RunEnd``, whose ``run`` is the run as the leg
        left it. Setting ``cancel`` ends the leg before its next unit, and so
        does leaving the loop early once the stream is closed, which Python
        does for an async generator only when it is collected: hold it with
        ``async with contextlib.aclosing(run.stream()) as events:`` and loop
        over ``events``. Setting ``hand_back`` stops the leg the same way and
        leaves the run going, for a host that shuts down and lets another
        process go on: the calls on threads are waited for and kept, and the
        last event is a ``RunCheckpoint`` whose ``run`` is ``running``, with
        no ``RunEnd``, since the run didn't end. A ``cancel`` set as well
        wins. With ``checkpoint_every`` (seconds), a
        ``RunCheckpoint`` comes at most that often while units finish, and once
        when the leg is stopped with calls still on threads: a host that stores
        its ``run`` loses only the calls in flight if the process dies. A call
        that can't start raises here, before anything is iterated.
        """
        checked_max_items(timeout=timeout, checkpoint_every=checkpoint_every)
        return self._stream(context_values(self.graph._needs_context, context), timeout, cancel, hand_back, checkpoint_every)

    async def _stream(
        self, context: dict[type, Any], timeout: float | None, cancel: AsyncioEvent | None, hand_back: AsyncioEvent | None,
        checkpoint_every: float | None,
    ) -> AsyncGenerator[Event]:
        loop, events = asyncio.get_running_loop(), Queue["Event | None"]()
        cancelled = ThreadEvent()
        deadline = _deadline(timeout)

        def stop() -> RunState | None:
            if cancelled.is_set() or (cancel is not None and cancel.is_set()):
                return EndedEarly("cancelled", problem.run_cancelled())
            if hand_back is not None and hand_back.is_set():
                return RUNNING  # the leg stops and the run goes on elsewhere
            return deadline() if deadline is not None else None

        failed: list[BaseException] = []

        def leg() -> None:
            try:
                self._run_scheduler(context, lambda event: _hand_over(loop, events, event), stop, checkpoint_every)
            except BaseException as error:  # the engine's own defect reaches the caller
                failed.append(error)
            finally:
                _hand_over(loop, events, None)

        thread = Thread(target=leg, name="conductor-scheduler", daemon=True)
        thread.start()
        try:
            while (event := await events.get()) is not None:
                yield event
            if failed:
                raise failed[0]
        finally:
            cancelled.set()

    def _run_scheduler(
        self, context: dict[type, Any], emit: Callable[[EventBase], None] | None, stop: Callable[[], RunState | None] | None,
        checkpoint_every: float | None = None,
    ) -> Run:
        """One leg over a copy of this run, which is handed back as the leg left it.

        The leg is given what it changes and reads (the copy's units and earlier work, the graph's functions
        where they stand, and what a node comes to for an item), and says how it ended. What a listener hears of
        the run itself (that it started, a checkpoint of it, how it ended) is said here, where the run is.
        """
        run = self._copy()
        checkpoint: Callable[[RunState], None] | None = None
        if emit is not None:
            emit(RunStart(waiting=tuple(run.waiting)))
            if checkpoint_every is not None:
                listener = emit

                def checkpoint(state: RunState) -> None:
                    listener(RunCheckpoint(run=dataclasses.replace(run._copy(), state=state)))

        graph = self.graph
        run.state, run._finished_nodes = Scheduler(
            nodes=graph._flat_function_nodes, threads=graph._threads, outcomes=run._outcomes, reusable=run._reusable, finished=run._finished_nodes,
            max_items=run.max_items, arguments_for=run._reader.arguments_for, emit=emit, stop=stop, context=context,
            checkpoint=checkpoint, checkpoint_every=checkpoint_every,
        ).run()
        if emit is not None:  # a leg handed back left its run running: the run didn't end, and the host stores it as it stands
            emit(RunCheckpoint(run=run) if run.state is RUNNING else RunEnd.of(run))
        return run

    @overload
    def answer(self, waiting: Waiting | str, value: Any, /) -> Run: ...

    @overload
    def answer(self, answers: Mapping[str, Any], /) -> Run: ...

    def answer(self, waiting: Waiting | str | Mapping[str, Any], value: Any = _NOT_GIVEN, /) -> Run:
        """Give waiting units their answers, checked against their questions; nothing runs until ``resume``.

        ``run.answer(unit, value)`` answers one, ``run.answer({unit_id: value, ...})``
        several. A node with one output is answered with its value, one with
        several with a mapping by output name, where a question with a default
        or a blank-taking type without choices may be left out, and of an
        ``exclusive_group`` exactly one is answered; a question limited by
        choices is answered unless it has a default. Every fault is refused at
        once and the run is left as it was: a unit answered twice in one call,
        an id that names nothing waiting (``not_waiting``), or one whose place
        now holds another question (``answer_stale``, with ``ctx.moved_to``
        when that question waits elsewhere). A run that ended early takes
        answers as a waiting one does, a cancelled one included.
        """
        if value is not _NOT_GIVEN:
            given: Mapping[str, Any] = {waiting.id if isinstance(waiting, Waiting) else str(waiting): value}
        elif isinstance(waiting, Mapping):
            given = {str(key): one for key, one in waiting.items()}
        else:
            raise TypeError("run.answer takes a unit and its answer, or a mapping of unit ids to answers")
        problems: list[Problem] = []
        answered: dict[tuple[FlatFunctionNode, ItemIndexes], Outcome] = {}
        for unit_id, one in given.items():
            place, _, fingerprint = unit_id.partition("~")
            try:
                loc, nodes, item_indexes = parse_loc_text(place)
            except ValueError:  # a place in it isn't a number
                problems.append(problem.not_waiting(unit_id.partition("@")[0], waiting_id=unit_id, status=self.status))
                continue
            node = self.graph._nodes_by_name.get(node_path_text(nodes))
            found = self._outcomes[node.name].get(item_indexes) if type(node) is FlatFunctionNode else None
            if type(node) is not FlatFunctionNode or type(found) is not outcomes.Waiting:
                moved_to = self._moved_to(nodes, fingerprint) if fingerprint else None
                problems.append(problem.not_waiting(*loc, waiting_id=unit_id, status=self.status, moved_to=moved_to))
                continue
            compiled_node = node.compiled_node
            if fingerprint and question_fingerprint(compiled_node, found) != fingerprint:
                current = waiting_record(compiled_node, outcome_loc(node, item_indexes), found).id
                problems.append(problem.answer_stale(*loc, waiting_id=unit_id, current=current, moved_to=self._moved_to(nodes, fingerprint)))
                continue
            if (node, item_indexes) in answered:
                problems.append(problem.invalid_answer(*loc, waiting_id=unit_id, reason="the unit is answered twice in one call"))
                continue
            read, faults = answer_outputs(compiled_node, one, found.question.overrides)
            problems.extend(
                problem.invalid_answer(*loc, waiting_id=unit_id, reason=reason, question=q or None) for q, reason in faults
            )
            answered[(node, item_indexes)] = Answered(read, found.question) if not faults else found
        if problems:
            raise InvalidCallError(problems)
        run = self._copy(share_outcomes=True)
        for changed in {node for node, _ in answered}:  # its own units for each node it changes; every other node's are shared
            run._outcomes[changed.name] = dict(self._outcomes[changed.name])
        for (node, item_indexes), now in answered.items():
            run._outcomes[node.name][item_indexes] = now
        return run

    def _moved_to(self, nodes: NodePath, fingerprint: str) -> str | None:
        """The id of the unit where the question a stale id names now waits, if it waits elsewhere on the node at
        the path ``nodes``."""
        for node, item_indexes, outcome in self._waiting_outcomes():
            if node.node_path == nodes and question_fingerprint(node.compiled_node, outcome) == fingerprint:
                return waiting_record(node.compiled_node, outcome_loc(node, item_indexes), outcome).id
        return None

    def with_inputs(self, inputs: Mapping[str, Any] = MappingProxyType({}), /, *, max_items: int | None = None, **by_name: Any) -> Run:
        """A run of these changed inputs, the rest as before, ready to ``resume`` or ``stream``.

        It carries this run's work, and what this run carried and didn't use
        for a node that hasn't finished (a finished node's is dropped, so a
        save stays its size over edits): a unit that receives what an earlier
        unit of its node received takes that unit's outcome, an answer or a
        question still waiting included, and the node is not called for it.
        Inputs are given as ``graph.with_inputs`` takes them, and checked the
        same way.
        """
        changed = {**inputs, **by_name}
        named = self.graph._given_by_address(changed)[0]
        given = {graph_input.name: self.inputs[address] for address, graph_input in self.graph._input_fields.items() if address not in named}
        return self.graph.with_inputs({**given, **changed}, max_items=max_items if max_items is not None else self.max_items, previous=self)

    def forget(self, *node_ids: str) -> Run:
        """A run of the same inputs that has forgotten these nodes' work, ready to ``resume`` or ``stream``.

        Every other unit is carried as ``with_inputs`` carries it, so only the
        forgotten nodes are called again, and a node below them that then
        receives what it received before is not. A host forgets a node whose
        access it must check again.
        """
        unknown = [n for n in node_ids if n not in self.graph._nodes]
        if unknown:
            raise InvalidCallError([problem.node_not_in_graph(n, nodes=list(self.graph._nodes)) for n in unknown])
        fresh = self.with_inputs()
        for path in [path for path in fresh._reusable if path[0] in node_ids]:  # a subgraph's id forgets every node inside it
            del fresh._reusable[path]
        return fresh

    def _reusable_outcomes(self, into: Graph | None = None) -> dict[NodePath, ReusableOutcomes]:
        """What a run going on from this one may take, by node: each node's finished and waiting units, in
        instance order, then the earlier work this run held and didn't take.

        ``into`` another graph keeps only the nodes it holds as they are here (by
        ``code_fingerprint``): the others' units could never be taken there, and kept they would only make its
        save larger.
        """
        work: dict[NodePath, ReusableOutcomes] = {}
        reader = self._reader
        for node in self.graph._flat_function_nodes:
            if not node.policy.reuse_when_inputs_match or node.node_path in self._code_changed:
                continue
            compiled_node = node.compiled_node
            of_node = work[node.node_path] = ReusableOutcomes()
            for item_indexes, outcome in self._outcomes[node.name].items():
                if type(outcome) is Skipped or type(outcome) is outcomes.NotCalled:
                    continue
                arguments = cast("dict[str, Any]", reader.arguments_for(node, item_indexes))  # the unit was called, so what it reads is settled and it received a call's arguments
                fingerprint = arguments_fingerprint(compiled_node.code_fingerprint, arguments, plain=compiled_node.plain_inputs)
                if fingerprint is None:
                    continue
                if type(outcome) is Answered or type(outcome) is outcomes.Waiting:
                    of_node.answers.setdefault((fingerprint, outer_items_fingerprint(self._outcomes, node, item_indexes)), []).append(outcome)
                else:
                    of_node.by_item_indexes[item_indexes] = (fingerprint, outcome)
                    of_node.by_fingerprint.setdefault(fingerprint, outcome)
        for path, unused in self._reusable.items():  # the earlier work this run held and no unit of it took
            of_node = work.setdefault(path, ReusableOutcomes())
            for fingerprint, done in unused.by_fingerprint.items():
                of_node.by_fingerprint.setdefault(fingerprint, done)
            for asked, answers in unused.answers.items():
                of_node.answers.setdefault(asked, []).extend(answers)
        ours, theirs = self.graph._functions, (into or self.graph)._functions
        return {
            path: of_node for path, of_node in work.items()
            if (of_node.by_fingerprint or of_node.answers) and (there := theirs.get(path)) is not None and there.code_fingerprint == ours[path].code_fingerprint
        }

    def again(
        self, inputs: Mapping[str, Any] = MappingProxyType({}), /, *, max_items: int | None = None, context: Mapping[type, Any] = MappingProxyType({}),
        timeout: float | None = None, **by_name: Any,
    ) -> Run:
        """Run again with changed inputs, the rest as before: ``run.with_inputs(...).resume(...)``.

        A unit that receives what it received before is not called again.
        """
        return self.with_inputs(inputs, max_items=max_items, **by_name).resume(context=context, timeout=timeout)

    # -- reading ----------------------------------------------------------------

    @property
    def waiting(self) -> list[Waiting]:
        """Every unit waiting for a person, in node then item order; those inside a subgraph where its
        node stands."""
        return [waiting_record(node.compiled_node, outcome_loc(node, item_indexes), outcome) for node, item_indexes, outcome in self._waiting_outcomes()]

    def _waiting_outcomes(self) -> Iterator[tuple[FlatFunctionNode, ItemIndexes, outcomes.Waiting]]:
        """Every unit waiting for a person, with its node and instance, in node then instance order."""
        for node in self.graph._flat_function_nodes:
            for item_indexes, outcome in self._outcomes[node.name].items():
                if type(outcome) is outcomes.Waiting:
                    yield node, item_indexes, outcome

    @property
    def not_called(self) -> list[NotCalled]:
        """The calls a node didn't get because an input was blank: which node, which item and which input; those
        inside a subgraph where its node stands."""
        return [
            NotCalled(loc=outcome_loc(node, item_indexes), input=outcome.input)
            for node in self.graph._flat_function_nodes for item_indexes, outcome in self._outcomes[node.name].items() if type(outcome) is outcomes.NotCalled
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
        reader = self._display_reader
        return {n: {o: reader.display_value(display_plans_of(*self._function_of(f"{n}.{o}"))) for o in compiled_node.outputs} for n, compiled_node in self.graph._nodes.items()}

    @property
    def outputs(self) -> dict[str, Any]:
        """The graph's outputs (``graph.outputs``), by address, once the run is complete; ``results`` reads a run in any state."""
        match self.state:
            case Going(status="complete"):
                pass
            case Going(status=status):
                raise InvalidCallError([problem.run_not_complete("outputs", status=status, reason="")])
            case EndedEarly(status=status, cause=cause):
                raise InvalidCallError([problem.run_not_complete("outputs", status=status, reason=cause.msg)])
        reader = self._display_reader
        return {address: reader.display_value(display_plans_of(*self._function_of(address))) for address in self.graph.outputs}

    def outputs_json(self) -> str:
        """The graph's outputs (``run.outputs``) as JSON, by address, each written by its output's type as a save
        writes it: a table as its columns, a float that isn't finite as its text, a host's type by its own schema.

        Raises:
            InvalidCallError: The run is not complete (``run_not_complete``), or an output holds a value with no
                JSON form (``run_unwritable``, naming the output).
        """
        return outputs_to_json(self)

    def _function_of(self, address: str) -> tuple[FlatFunctionNode, str]:
        """The function node that returns the output at ``address`` (``node.output``), and the output's name on
        that function. For a subgraph (``e.upper.result``) it is the function inside the
        subgraph, since the subgraph's node is never called itself."""
        node_id, _, output = address.partition(".")
        node = self.graph._nodes_by_name[node_id]
        return node.output_sources[output] if type(node) is FlatSubgraph else (node, output)

    # -- what a leg reads -------------------------------------------------------

    @property
    def _reader(self) -> ResultReader:
        """Reads what this run's units hold for one item: what a call of a node receives there."""
        return ResultReader(self._outcomes, self.inputs)

    @property
    def _display_reader(self) -> DisplayReader:
        """Reads what this run's units hold as they stand, waiting for nothing: what a host is shown."""
        return DisplayReader(self._outcomes, self.inputs)

    # -- JSON -------------------------------------------------------------------

    @classmethod
    def from_json(cls, saved: str | bytes, graph: Graph) -> Run:
        """The run of ``graph`` that ``run.to_json()`` wrote as ``saved``: ``Run.from_json(run.to_json(), graph)`` is ``run`` again.

        A graph with problems takes no run, so they are what is refused, before anything of the save is read. The
        save is read once, by the graph's ``RunCodec``, from its JSON text, so each value is read as its type reads
        JSON: a date, and a float that isn't finite, from their text.

        Raises:
            InvalidCallError: The graph's problems; ``saved_run_of_another_graph``; or ``saved_run_unreadable`` at the place
                pydantic names (the section, the node, the output, the slot), quoting nothing of a value.
        """
        return run_from_json(cls, saved, graph)

    def to_json(self) -> str:
        """The run as JSON: its inputs, each node's units as a column per output, and what it went on from."""
        return run_to_json(self)

    def _copy(self, *, share_outcomes: bool = False) -> Run:
        """A copy to change: its own units to record into and its own earlier work to take from; or with
        ``share_outcomes`` this run's units of every node, for a caller that replaces whole nodes and takes nothing.

        Args:
            share_outcomes: Whether the copy holds this run's own dict of units for each node.
        """
        if share_outcomes:
            outcomes, previous = dict(self._outcomes), dict(self._reusable)
        else:  # its own answers to take, as this run holds them too
            outcomes = {name: dict(held) for name, held in self._outcomes.items()}
            previous = {path: ReusableOutcomes(of_node.by_item_indexes, of_node.by_fingerprint, {asked: list(left) for asked, left in of_node.answers.items()}) for path, of_node in self._reusable.items()}
        return dataclasses.replace(self, _outcomes=outcomes, _reusable=previous)


RunEnd.model_rebuild(_types_namespace={"Run": Run})
RunCheckpoint.model_rebuild(_types_namespace={"Run": Run})


def no_outcomes(graph: Graph) -> dict[str, dict[ItemIndexes, Outcome]]:
    """The units of a run of ``graph`` that hasn't started: none, for every function of the graph and of each
    graph embedded in it."""
    return {node.name: {} for node in graph._flat_function_nodes}


def checked_max_items(*, max_items: int | None = None, timeout: float | None = None, checkpoint_every: float | None = None) -> None:
    """A run's ceiling (at least 1 item) and a leg's seconds (a finite number above 0) are refused at the call."""
    problems = []
    if max_items is not None and max_items < 1:
        problems.append(problem.invalid_max_items("max_items", max_items=max_items))
    if timeout is not None and not _seconds(timeout):
        problems.append(problem.invalid_timeout("timeout", timeout=timeout))
    if checkpoint_every is not None and not _seconds(checkpoint_every):
        problems.append(problem.invalid_checkpoint_every("checkpoint_every", checkpoint_every=checkpoint_every))
    if problems:
        raise InvalidCallError(problems)


def _seconds(value: float) -> bool:
    return math.isfinite(value) and value > 0


# -- values -------------------------------------------------------------------------------------


def _hand_over(loop: AbstractEventLoop, events: Queue[Any], event: Any) -> None:
    """Put an event on the listener's queue from the leg's thread; a listener whose loop has closed is gone, and hears nothing."""
    try:
        loop.call_soon_threadsafe(events.put_nowait, event)
    except RuntimeError:
        pass


def _deadline(timeout: float | None) -> Callable[[], EndedEarly | None] | None:
    if timeout is None:
        return None
    end = time.monotonic() + timeout
    return lambda: EndedEarly("timed_out", problem.run_timed_out(timeout=timeout)) if time.monotonic() > end else None
