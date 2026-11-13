"""``Run``: a graph's inputs and the outcome of every call so far, as one value that writes itself as JSON.

For the README graph, ``graph.run(text="a\\nb")`` calls ``lines`` once, ``shout`` once per line and ``join`` once, and
the run keeps one outcome per call: ``Done`` (it returned), ``Answered`` (a person answered for it), ``Skipped`` (it
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
from asyncio import Event as AsyncioEvent
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast, overload

from conductor._internal.ask import Answer
from conductor._internal.base import (
    ConductorModel,
    ItemIndexes,
    Loc,
    NodePath,
    node_path_text,
    parse_loc_text,
)
from conductor._internal.gate import RefusedError, check_limits
from conductor._internal.problem import InvalidCallError, Problem
from conductor._internal.problem import catalogue as problem

from .arguments import ResultReader
from .events import (
    Event,
    EventBase,
    RunCheckpoint,
    RunEnd,
    RunStart,
)
from .flat import FlatFunctionNode, FlatGraph, flat_graph_of, outcome_loc
from .json import codec_of, outputs_to_json, run_from_json, run_to_json
from .outcome import (
    Answered,
    NoneInput,
    Outcome,
    Skipped,
    Unanswered,
)
from .progress import Progress, progress_of
from .results import outputs_of, results_of
from .reuse import (
    ReusableOutcomes,
    arguments_fingerprint,
    outer_items_fingerprint,
)
from .scheduler import Scheduler
from .state import (
    READY,
    RUNNING,
    EndedEarly,
    Going,
    RunState,
    RunStatus,
    deadline,
)
from .stream import stream
from .waiting import Waiting, question_fingerprint, waiting_record

if TYPE_CHECKING:
    from conductor._internal.graph import Graph



#: ``run.answer``'s second argument when it is given a mapping of answers instead.
_NOT_GIVEN: Any = object()


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
        state: How the run stands: ``Going`` (``ready`` before the first leg, then ``running``, ``waiting`` or
            ``complete``), or ``EndedEarly`` (``failed``, ``stopped`` at the item ceiling, ``cancelled`` or
            ``timed_out``) with the ``Problem`` that caused it.
    """

    graph: Graph[C]
    inputs: dict[str, Any]
    max_items: int | None
    #: Each function's units by instance, in instance order, by the function's name: ``_outcomes["w"]`` for a node of
    #: the graph itself, ``_outcomes["e/ask"]`` for one inside ``e``. Only a leg records out of that order, and it puts
    #: the units back in it before it hands the run on.
    _outcomes: dict[str, dict[ItemIndexes, Outcome]]
    #: What each node may take from the run this one went on from.
    _reusable: dict[NodePath, ReusableOutcomes]
    #: The flat graph of ``graph``: the function nodes the run calls.
    _flat: FlatGraph = field(kw_only=True)
    state: RunState = READY
    #: For a node whose units a saved run made with other code or another signature, what that code was; they are never earlier work for a later run.
    _code_changed: dict[NodePath, str] = field(default_factory=dict[NodePath, str])
    #: The nodes every unit of which was finished when the last leg ended, so a later leg doesn't say them complete again.
    _finished_nodes: frozenset[NodePath] = frozenset()
    #: A person's answers to a node's ``Asked`` parameter, by node name and item indexes, each kept until its call is
    #: made: ``_answers["approve"][(1,)]`` once ``run.answer`` is given ``True`` for ``approve@1``. Never changed in place.
    _answers: dict[str, dict[ItemIndexes, Answer]] = field(default_factory=dict[str, dict[ItemIndexes, Answer]])

    def __repr__(self) -> str:
        return f"<Run {self.status}: {sum(len(outcomes) for outcomes in self._outcomes.values())} outcomes>"

    @property
    def status(self) -> RunStatus:
        """How the run stands, as ``state`` says it."""
        return self.state.status

    # -- going on ---------------------------------------------------------------

    def with_max_items(self, max_items: int) -> Run[C]:
        """A copy of this run with ``max_items`` as its item limit. No node is called; ``resume`` goes on.

        A run stops with status ``stopped`` when a list that a node is called
        per item of has more items than the limit. Example: a run started with
        ``max_items=30`` that met 40 items goes on with
        ``run.with_max_items(100).resume()``. The limit is saved by ``to_json()``.

        Raises:
            InvalidCallError: ``max_items`` is not a whole number of at least 1 (``invalid_max_items``).
        """
        _check_limits(max_items=max_items)
        return dataclasses.replace(self, max_items=max_items)

    def resume(self, *, context: C | None = None, timeout: float | None = None, only_costs_nothing: bool = False) -> Run[C]:
        """Go on: call every node that is ready, until the run is complete, waits for a person, fails, reaches its
        item limit or its ``timeout``, and return the run as it then is.

        ``context`` is the run's context, the object every node that takes ``Context[...]`` is given; a run never
        saves it, so it is passed again on every call. Past ``timeout`` no more calls start, but the calls already
        on threads are waited for, as long as their ``attempt_timeout``, and kept: so ``resume`` can return later
        than ``timeout``, and a call that returned is never made again.

        ``only_costs_nothing=True`` calls only the nodes whose ``Policy`` says ``costs_nothing``, as far as they
        are ready, and the run is ``ready`` again afterwards: ``graph.with_inputs(csv=csv).resume(only_costs_nothing=True)``,
        then ``run.progress``, says how many calls each costly node will have before any is made. A call that
        raises then has no outcome, and fails or asks at the next ``resume()``. That ``resume()`` makes none of
        the calls again, and a stream of it does not say again the nodes that were complete.

        Raises:
            InvalidCallError: ``timeout`` is not a number of seconds above 0 (``invalid_timeout``); ``context`` is not an instance of
                the registry's ``context_type`` (``invalid_context``), or is ``None`` while a node takes it
                (``context_missing``).
        """
        _check_limits(timeout=timeout)
        return self._run_scheduler(self.graph._checked_context(context), None, deadline(timeout), only_costs_nothing=only_costs_nothing)

    def stream(
        self, *, context: C | None = None, timeout: float | None = None, cancel: AsyncioEvent | None = None,
        hand_back: AsyncioEvent | None = None, checkpoint_every: float | None = None, only_costs_nothing: bool = False,
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
                above 0 (``invalid_checkpoint_every``), at this call, before any event.
        """
        _check_limits(timeout=timeout, checkpoint_every=checkpoint_every)
        return stream(self._run_scheduler, self.graph._checked_context(context), timeout, cancel, hand_back, checkpoint_every, only_costs_nothing)

    def _run_scheduler(
        self, context: C | None, emit: Callable[[EventBase], None] | None, stop: Callable[[], RunState | None] | None,
        checkpoint_every: float | None = None, only_costs_nothing: bool = False,
    ) -> Run[C]:
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

        flat = self._flat
        run.state, run._finished_nodes = Scheduler(
            nodes=list(flat.function_nodes.values()), threads=flat.threads, outcomes=run._outcomes, reusable=run._reusable, answers=run._answers,
            finished=run._finished_nodes,
            max_items=run.max_items, arguments_for=run._reader.arguments_for, emit=emit, stop=stop, context=context,
            checkpoint=checkpoint, checkpoint_every=checkpoint_every, only_costs_nothing=only_costs_nothing,
        ).run()
        if run._answers:  # an answer goes once its call has an outcome
            outcomes_of = run._outcomes
            run._answers = {
                name: left for name, of_node in run._answers.items() if (left := {i: a for i, a in of_node.items() if i not in outcomes_of[name]})
            }
        if emit is not None:  # a leg handed back left its run running: the run didn't end, and the host stores it as it stands
            emit(RunCheckpoint(run=run) if run.state is RUNNING else RunEnd.of(run))
        return run

    @overload
    def answer(self, waiting: Waiting | str, value: Any, /) -> Run[C]: ...

    @overload
    def answer(self, answers: Mapping[str, Any], /) -> Run[C]: ...

    def answer(self, waiting: Waiting | str | Mapping[str, Any], value: Any = _NOT_GIVEN, /) -> Run[C]:
        """Give waiting units their answers, checked against their questions; nothing runs until ``resume``.

        ``run.answer(waiting, value)`` answers one, ``run.answer({waiting_id: value, ...})``
        several. One question is answered with its value, several (a node
        with several outputs) with a mapping by question name, where a
        question with a default or a blank-taking type without choices may be
        left out, and of an ``exclusive_group`` exactly one is answered; a
        question limited by choices is answered unless it has a default. The
        answer to a node's outputs is the call's outcome. The answer to an
        ``Asked`` parameter is kept by the run, and ``resume`` calls the node
        with it. Every fault is refused at once and the run is left as it
        was: a call answered twice in one call of ``answer``, an id that names
        nothing waiting (``not_waiting``), or one whose call now asks another
        question (``answer_stale``, with ``ctx.moved_to`` when that question
        waits elsewhere). A run that ended early takes answers as a waiting
        one does, a cancelled one included.
        """
        if value is not _NOT_GIVEN:
            given: Mapping[str, Any] = {waiting.id if isinstance(waiting, Waiting) else str(waiting): value}
        elif isinstance(waiting, Mapping):
            given = {str(key): one for key, one in waiting.items()}
        else:
            raise TypeError("run.answer takes a unit and its answer, or a mapping of unit ids to answers")
        problems: list[Problem] = []
        answered: dict[tuple[FlatFunctionNode, ItemIndexes], Outcome | Answer] = {}
        for unit_id, one in given.items():
            place, _, fingerprint = unit_id.partition("~")
            try:
                loc, nodes, item_indexes = parse_loc_text(place)
            except ValueError:  # a place in it isn't a number
                problems.append(problem.not_waiting(unit_id.partition("@")[0], waiting_id=unit_id, status=self.status))
                continue
            node = self._flat.function_nodes.get(node_path_text(nodes))  # None for an id of a subgraph, or of no node
            found = self._outcomes[node.name].get(item_indexes) if node is not None else None
            if node is None or type(found) is not Unanswered:
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
            try:
                read = compiled_node.answer_door(one, found.question.overrides, loc=loc, waiting_id=unit_id)
            except RefusedError as refused:
                problems.extend(refused.problems)
                answered[(node, item_indexes)] = found
                continue
            asked = compiled_node.definition.asked
            answered[(node, item_indexes)] = Answered(read, found.question) if asked is None else Answer(found.question, asked.field.name, read[asked.field.name])
        if problems:
            raise InvalidCallError(problems)
        run = self._copy(share_outcomes=True)
        for changed in {node for node, _ in answered}:  # its own outcomes and answers for each node it changes; every other node's are shared
            run._outcomes[changed.name] = dict(self._outcomes[changed.name])
            if changed.compiled_node.definition.asked is not None:
                run._answers = {**run._answers, changed.name: dict(self._answers.get(changed.name, {}))}
        for (node, item_indexes), now in answered.items():
            if isinstance(now, Answer):  # the call has no outcome until the node is called with the answer
                del run._outcomes[node.name][item_indexes]
                run._answers[node.name][item_indexes] = now
            else:
                run._outcomes[node.name][item_indexes] = now
        return run

    def _moved_to(self, nodes: NodePath, fingerprint: str) -> str | None:
        """The id of the unit where the question a stale id names now waits, if it waits elsewhere on the node at
        the path ``nodes``."""
        for node, item_indexes, outcome in self._waiting_outcomes():
            if node.node_path == nodes and question_fingerprint(node.compiled_node, outcome) == fingerprint:
                return waiting_record(node.compiled_node, outcome_loc(node, item_indexes), outcome).id
        return None

    def with_inputs(self, inputs: Mapping[str, Any] = MappingProxyType({}), /, *, max_items: int | None = None, **by_name: Any) -> Run[C]:
        """A run of these changed inputs, the rest as before, ready to ``resume`` or ``stream``.

        It carries this run's work, and what this run carried and didn't use
        for a node that hasn't finished (a finished node's is dropped, so a
        save stays its size over edits): a unit that receives what an earlier
        unit of its node received takes that unit's outcome, an answer or a
        question still waiting included, and the node is not called for it.
        Inputs are given as ``graph.with_inputs`` takes them, and checked the
        same way.
        """
        return self.graph.with_inputs({**self.inputs, **inputs, **by_name}, max_items=max_items if max_items is not None else self.max_items, previous=self)

    def forget(self, *node_ids: str) -> Run[C]:
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

    def _reusable_outcomes(self, into: FlatGraph) -> dict[NodePath, ReusableOutcomes]:
        """What a run going on from this one may take, by node: each node's finished and waiting units, in
        instance order, then the earlier work this run held and didn't take.

        ``into`` is the flat graph of the graph the new run is of, this run's or another's. Only the nodes it has
        as they are here (by ``code_fingerprint``) are kept: the others' units could never be taken there, and kept
        they would only make its save larger.
        """
        work: dict[NodePath, ReusableOutcomes] = {}
        reader = self._reader
        for node in self._flat.function_nodes.values():
            if not node.policy.reuse_when_inputs_match or node.node_path in self._code_changed:
                continue
            compiled_node = node.compiled_node
            of_node = work[node.node_path] = ReusableOutcomes()
            for item_indexes, outcome in self._outcomes[node.name].items():
                if type(outcome) is Skipped or type(outcome) is NoneInput:
                    continue
                arguments = cast("dict[str, Any]", reader.arguments_for(node, item_indexes))  # the unit was called, so what it reads is settled and it received a call's arguments
                fingerprint = arguments_fingerprint(compiled_node.code_fingerprint, arguments, plain=compiled_node.plain_inputs)
                if fingerprint is None:
                    continue
                if type(outcome) is Answered or type(outcome) is Unanswered:
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
        ours, theirs = self._flat.function_nodes, into.function_nodes
        return {
            path: of_node for path, of_node in work.items()
            if (of_node.by_fingerprint or of_node.answers) and (there := theirs.get(name := node_path_text(path))) is not None
            and there.compiled_node.code_fingerprint == ours[name].compiled_node.code_fingerprint
        }

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
        """Every unit waiting for a person, in node then item order; those inside a subgraph where its
        node stands."""
        return [waiting_record(node.compiled_node, outcome_loc(node, item_indexes), outcome) for node, item_indexes, outcome in self._waiting_outcomes()]

    def _waiting_outcomes(self) -> Iterator[tuple[FlatFunctionNode, ItemIndexes, Unanswered]]:
        """Every unit waiting for a person, with its node and instance, in node then instance order."""
        for node in self._flat.function_nodes.values():
            for item_indexes, outcome in self._outcomes[node.name].items():
                if type(outcome) is Unanswered:
                    yield node, item_indexes, outcome

    @property
    def progress(self) -> Progress:
        """How far the run is: for the run and for each node, how many calls have an outcome, how many wait for a
        person, how many are known so far, and how many there will be (``None`` while a series that decides it
        does not exist yet).

        It is read from the run's outcomes each time, so a run read back with ``Run.from_json`` has the same.
        For the README graph before any call, ``run.progress.nodes["approve"].total`` is ``None``: how many
        invoices there are is known once ``read`` has returned. ``run.resume(only_costs_nothing=True)`` calls
        ``read`` first when its ``Policy`` says ``costs_nothing``.
        """
        return progress_of(self._flat, self._outcomes)

    @property
    def not_called(self) -> list[NotCalled]:
        """The calls a node didn't get because an input was blank: which node, which item and which input; those
        inside a subgraph where its node stands."""
        return [
            NotCalled(loc=outcome_loc(node, item_indexes), input=outcome.input)
            for node in self._flat.function_nodes.values() for item_indexes, outcome in self._outcomes[node.name].items() if type(outcome) is NoneInput
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
        return results_of(self.graph, self._flat, self._outcomes, self.inputs)

    @property
    def outputs(self) -> dict[str, Any]:
        """The graph's outputs (``graph.outputs``), by name, once the run is complete; ``results`` reads a run in any state."""
        match self.state:
            case Going(status="complete"):
                pass
            case Going(status=status):
                raise InvalidCallError([problem.run_not_complete("outputs", status=status, reason="")])
            case EndedEarly(status=status, cause=cause):
                raise InvalidCallError([problem.run_not_complete("outputs", status=status, reason=cause.msg)])
        return outputs_of(self.graph, self._flat, self._outcomes, self.inputs)

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
        """Reads what this run's units hold for one item: what a call of a node receives there."""
        return ResultReader(self._outcomes, self.inputs)

    # -- JSON -------------------------------------------------------------------

    @classmethod
    def from_json[R](cls, saved: str | bytes, graph: Graph[R]) -> Run[R]:
        """Returns the run that ``run.to_json()`` wrote as ``saved``, on the same ``graph``.

        ``Run.from_json(run.to_json(), graph)`` has the inputs, the outcomes and the state of ``run``, also in
        another process. Each value is read the way its type reads JSON: a date from its text, for example.

        Raises:
            InvalidCallError: ``graph`` has problems (they are raised, and nothing of ``saved`` is read);
                ``saved`` was written for another graph (``saved_run_of_another_graph``); or ``saved`` is not a
                saved run of this graph (``saved_run_unreadable``, whose ``where`` names the key, the node, the
                output and the call number, and which quotes no value).
        """
        if graph.problems:
            raise InvalidCallError(graph.problems)
        return run_from_json(Run, saved, graph, codec_of(graph))

    def to_json(self) -> str:
        """Returns the whole run as JSON text: its inputs, its state, every node's outcomes, and the outcomes
        it may still reuse from the previous run.

        Raises:
            InvalidCallError: A value cannot be written as JSON (``run_unwritable``, naming the input or the
                output).
        """
        return run_to_json(self)

    def _copy(self, *, share_outcomes: bool = False) -> Run[C]:
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


def make_run[C](graph: Graph[C], inputs: dict[str, Any], max_items: int | None, previous: Run[Any] | None) -> Run[C]:
    """Returns a run of ``graph`` that has not started, with the inputs the inputs door let in: what
    ``graph.with_inputs(text="a\\nb")`` returns. No node has an outcome yet; with ``previous``, the run may reuse
    that run's outcomes (``Run.with_inputs``)."""
    flat = flat_graph_of(graph)
    outcomes: dict[str, dict[ItemIndexes, Outcome]] = {name: {} for name in flat.function_nodes}
    return Run(graph, inputs, max_items, outcomes, previous._reusable_outcomes(into=flat) if previous is not None else {}, _flat=flat)


def _check_limits(*, max_items: int | None = None, timeout: float | None = None, checkpoint_every: float | None = None) -> None:
    """Raises ``InvalidCallError`` with the problems of the limits a public method was given that no run can have.

    Raises:
        InvalidCallError: The problems ``check_limits`` found (``invalid_max_items``, ``invalid_timeout``,
            ``invalid_checkpoint_every``).
    """
    try:
        check_limits(max_items=max_items, timeout=timeout, checkpoint_every=checkpoint_every)
    except RefusedError as refused:
        raise InvalidCallError(refused.problems) from None
