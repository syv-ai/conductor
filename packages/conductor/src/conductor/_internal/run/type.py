"""``Run``: a graph's inputs and every unit's result, as one value that writes itself as JSON.

A unit is one node in one instance of its scope: the node's one call, or its
call for one item. A unit is ``Done`` (its outputs kept), ``Answered`` (done
by a person), ``Skipped`` (a branch not taken), ``Blank`` (an input got
nothing, so the node wasn't called) or ``Asking`` (it waits for a person). A
unit with no entry hasn't run yet. Going on is one leg: start every unit
whose inputs exist, until nothing more can start. So a run can stop anywhere
(a person to ask, a failure, the item ceiling, a timeout) and carry on from
its JSON in another process.

A node inside an embedded graph has units as any node has: one for each item
its embedding node runs for, and under it for each item of its own scope. Its
open inputs read what the embedding node is given, and what reads the
embedding node reads the unit inside that holds the output. The embedding
node itself holds no unit: what a run says of it is read from the units
inside. Where something is, at any depth of embedded graphs, is one path of
node ids and item positions: ``("e", 0, "ask", 2)`` is ``ask``'s third item
inside ``e``'s first.
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
import threading
import time
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, cast, overload

import pydantic_core
from pydantic import ValidationError

from conductor._internal import problem
from conductor._internal.address import (
    ItemPath,
    Loc,
    NodePath,
    node_name,
    node_path,
    parse_loc_text,
)
from conductor._internal.ask import (
    Waiting,
    answer_outputs,
    question_fingerprint,
    waiting_unit,
)
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.earlier_work import EarlierWork, fingerprint_of, lineage
from conductor._internal.leg import Event, Leg, RunCheckpoint, RunEnd, RunStart, _EventBase
from conductor._internal.loaded_graph import (
    EmbeddedGraphAtPath,
    FunctionAtPath,
    InputGathersItems,
    NodeAtPath,
    host_reads,
    unit_loc,
)
from conductor._internal.node_definition import from_run_values
from conductor._internal.problem import InvalidCallError, Problem, value_free_errors
from conductor._internal.run.call_inputs import (
    AsItStandsReader,
    CallInputs,
    ItemReader,
)
from conductor._internal.run_status import _READY, _RUNNING, EndedEarly, Going, RunState, RunStatus
from conductor._internal.saved_run import (
    SavedEarlier,
    SavedRun,
    SavedUnits,
    _earlier_columns,
    _unit_columns,
    _unwritable_at,
)
from conductor._internal.units import (
    Answered,
    Asking,
    Blank,
    Skipped,
    Unit,
    Units,
    has_every_unit,
    instances,
)

if TYPE_CHECKING:
    from conductor._internal.graph import Graph



#: ``run.answer``'s second argument when it is given a mapping of answers instead.
_ONE_ANSWER: Any = object()


class NotCalled(ConductorModel):
    """A call a node didn't get, because an input got a blank there.

    Attributes:
        loc: Which call it was, as a ``Problem``'s ``loc`` says it: the node, then the place of its item in each
            Series the node runs per item of; inside an embedded graph, the embedding node and its item first.
        input: The input that got nothing.
    """

    loc: Loc
    input: str


@dataclass(eq=False, repr=False)
class Run:
    """One run of a graph. Every call returns a new ``Run``; ``to_json`` is all of it.

    A run is made by ``graph.with_inputs(...)``, ``graph.run(...)`` or ``graph.load_run(saved)``, which check
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
    #: Each function's units by instance, in instance order, by the function's name: ``_units["w"]`` for a node of
    #: the graph itself, ``_units["e/ask"]`` for one inside ``e``. Only a leg records out of that order, and it puts
    #: the units back in it before it hands the run on.
    _units: Units
    #: What each node may take from the run this one went on from.
    _earlier_work: dict[NodePath, EarlierWork]
    state: RunState = _READY
    #: For a node whose units a saved run made with other code or another signature, what that code was; they are never earlier work for a later run.
    _ran_as: dict[NodePath, str] = field(default_factory=dict[NodePath, str])
    #: The nodes every unit of which was finished when the last leg ended, so a later leg doesn't say them complete again.
    _complete: frozenset[NodePath] = frozenset()

    def __repr__(self) -> str:
        return f"<Run {self.status}: {sum(len(units) for units in self._units.values())} units>"

    @property
    def status(self) -> RunStatus:
        """How the run stands, as ``state`` says it."""
        return self.state.status

    # -- going on ---------------------------------------------------------------

    def resume(self, *, max_items: int | None = None, from_run: Mapping[type, Any] = MappingProxyType({}), timeout: float | None = None) -> Run:
        """Go on: start every unit whose inputs exist, until the run is done, waits, fails, reaches its ceiling or its ``timeout``.

        ``max_items`` raises (or sets) the ceiling a stopped run carries on
        under. ``from_run`` gives each ``FromRun`` type its value. Past its
        ``timeout`` a leg starts nothing more but waits for the calls already
        on threads, as far as their ``attempt_timeout``, and keeps them; so it
        can end later than ``timeout``, and a call that came back is never made
        again.
        """
        checked_leg(max_items=max_items, timeout=timeout)
        return self._leg(max_items, from_run_values(self.graph._needs_from_run, from_run), None, _deadline(timeout))

    def stream(
        self, *, max_items: int | None = None, from_run: Mapping[type, Any] = MappingProxyType({}), timeout: float | None = None, cancel: asyncio.Event | None = None,
        hand_back: asyncio.Event | None = None, checkpoint_every: float | None = None,
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
        checked_leg(max_items=max_items, timeout=timeout, checkpoint_every=checkpoint_every)
        return self._stream(max_items, from_run_values(self.graph._needs_from_run, from_run), timeout, cancel, hand_back, checkpoint_every)

    async def _stream(
        self, max_items: int | None, from_run: dict[type, Any], timeout: float | None, cancel: asyncio.Event | None, hand_back: asyncio.Event | None,
        checkpoint_every: float | None,
    ) -> AsyncGenerator[Event]:
        loop, events = asyncio.get_running_loop(), asyncio.Queue["Event | None"]()
        cancelled = threading.Event()
        deadline = _deadline(timeout)

        def stop() -> RunState | None:
            if cancelled.is_set() or (cancel is not None and cancel.is_set()):
                return EndedEarly("cancelled", problem.run_cancelled())
            if hand_back is not None and hand_back.is_set():
                return _RUNNING  # the leg stops and the run goes on elsewhere
            return deadline() if deadline is not None else None

        failed: list[BaseException] = []

        def leg() -> None:
            try:
                self._leg(max_items, from_run, lambda event: _hand_over(loop, events, event), stop, checkpoint_every)
            except BaseException as error:  # the engine's own defect reaches the caller
                failed.append(error)
            finally:
                _hand_over(loop, events, None)

        thread = threading.Thread(target=leg, name="conductor-leg", daemon=True)
        thread.start()
        try:
            while (event := await events.get()) is not None:
                yield event
            if failed:
                raise failed[0]
        finally:
            cancelled.set()

    def _leg(
        self, max_items: int | None, from_run: dict[type, Any], emit: Callable[[_EventBase], None] | None, stop: Callable[[], RunState | None] | None,
        checkpoint_every: float | None = None,
    ) -> Run:
        """One leg over a copy of this run, which is handed back as the leg left it.

        The leg is given what it changes and reads (the copy's units and earlier work, the graph's functions
        where they stand, and what a node comes to for an item), and says how it ended. What a listener hears of
        the run itself (that it started, a checkpoint of it, how it ended) is said here, where the run is.
        """
        run = self._copy(max_items=max_items if max_items is not None else self.max_items)
        checkpoint: Callable[[RunState], None] | None = None
        if emit is not None:
            emit(RunStart(waiting=tuple(run.waiting)))
            if checkpoint_every is not None:
                listener = emit

                def checkpoint(state: RunState) -> None:
                    listener(RunCheckpoint(run=dataclasses.replace(run._copy(), state=state)))

        graph = self.graph
        leg = Leg(graph._tree, graph._threads, run._units, run._earlier_work, run._complete, run.max_items, run._reader.comes_to, emit, stop, from_run, checkpoint, checkpoint_every)
        leg.go()
        run.state, run._complete = leg.state, leg.finished
        if emit is not None:  # a leg handed back left its run running: the run didn't end, and the host stores it as it stands
            emit(RunCheckpoint(run=run) if run.state is _RUNNING else RunEnd.of(run))
        return run

    @overload
    def answer(self, unit: Waiting | str, value: Any, /) -> Run: ...

    @overload
    def answer(self, answers: Mapping[str, Any], /) -> Run: ...

    def answer(self, unit: Waiting | str | Mapping[str, Any], value: Any = _ONE_ANSWER, /) -> Run:
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
        if value is not _ONE_ANSWER:
            given: Mapping[str, Any] = {unit.id if isinstance(unit, Waiting) else str(unit): value}
        elif isinstance(unit, Mapping):
            given = {str(key): one for key, one in unit.items()}
        else:
            raise TypeError("run.answer takes a unit and its answer, or a mapping of unit ids to answers")
        problems: list[Problem] = []
        answered: dict[tuple[FunctionAtPath, ItemPath], Unit] = {}
        for unit_id, one in given.items():
            place, _, fingerprint = unit_id.partition("~")
            try:
                loc, nodes, key = parse_loc_text(place)
            except ValueError:  # a place in it isn't a number
                problems.append(problem.not_waiting(unit_id.partition("@")[0], waiting_id=unit_id, status=self.status))
                continue
            node = self.graph._node_at.get(node_name(nodes))
            found = self._units[node.name].get(key) if type(node) is FunctionAtPath else None
            if type(node) is not FunctionAtPath or type(found) is not Asking:
                moved_to = self._moved_to(nodes, fingerprint) if fingerprint else None
                problems.append(problem.not_waiting(*loc, waiting_id=unit_id, status=self.status, moved_to=moved_to))
                continue
            placed = node.placed
            if fingerprint and question_fingerprint(placed, found) != fingerprint:
                current = waiting_unit(placed, unit_loc(node, key), found).id
                problems.append(problem.answer_stale(*loc, waiting_id=unit_id, current=current, moved_to=self._moved_to(nodes, fingerprint)))
                continue
            if (node, key) in answered:
                problems.append(problem.invalid_answer(*loc, waiting_id=unit_id, reason="the unit is answered twice in one call"))
                continue
            read, faults = answer_outputs(placed, one, found.question.overrides)
            problems.extend(
                problem.invalid_answer(*loc, waiting_id=unit_id, reason=reason, question=q or None) for q, reason in faults
            )
            answered[(node, key)] = Answered(read, found.question) if not faults else found
        if problems:
            raise InvalidCallError(problems)
        run = self._copy(share_units=True)
        for changed in {node for node, _ in answered}:  # its own units for each node it changes; every other node's are shared
            run._units[changed.name] = dict(self._units[changed.name])
        for (node, key), now in answered.items():
            run._units[node.name][key] = now
        return run

    def _moved_to(self, nodes: NodePath, fingerprint: str) -> str | None:
        """The id of the unit where the question a stale id names now waits, if it waits elsewhere on the node at
        the path ``nodes``."""
        for node, key, unit in self._asking():
            if node.path == nodes and question_fingerprint(node.placed, unit) == fingerprint:
                return waiting_unit(node.placed, unit_loc(node, key), unit).id
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
        given = {offered.name: self.inputs[address] for address, offered in self.graph._input_fields.items() if address not in named}
        return self.graph.with_inputs({**given, **changed}, max_items=max_items if max_items is not None else self.max_items, earlier=self)

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
        for path in [path for path in fresh._earlier_work if path[0] in node_ids]:  # an embedded graph's id forgets every node inside it
            del fresh._earlier_work[path]
        return fresh

    def _as_earlier_work(self, into: Graph | None = None) -> dict[NodePath, EarlierWork]:
        """What a run going on from this one may take, by node: each node's finished and waiting units, in
        instance order, then the earlier work this run held and didn't take.

        ``into`` another graph keeps only the nodes it holds as they are here (by
        ``reuse_digest``): the others' units could never be taken there, and kept they would only make its
        save larger.
        """
        work: dict[NodePath, EarlierWork] = {}
        reader = self._reader
        for node in self.graph._tree:
            if not node.policy.reuse_when_inputs_match or node.path in self._ran_as:
                continue
            placed = node.placed
            of_node = work[node.path] = EarlierWork()
            for key, unit in self._units[node.name].items():
                if type(unit) is Skipped or type(unit) is Blank:
                    continue
                received = cast("CallInputs", reader.comes_to(node, key))  # the unit was called, so what it reads is settled and it received a call's arguments
                fingerprint = fingerprint_of(placed.reuse_digest, received, plain=placed.plain_inputs)
                if fingerprint is None:
                    continue
                if type(unit) is Answered or type(unit) is Asking:
                    of_node.asked.setdefault((fingerprint, lineage(self._units, node, key)), []).append(unit)
                else:
                    of_node.at[key] = (fingerprint, unit)
                    of_node.computed.setdefault(fingerprint, unit)
        for path, untaken in self._earlier_work.items():  # the earlier work this run held and no unit of it took
            of_node = work.setdefault(path, EarlierWork())
            for fingerprint, done in untaken.computed.items():
                of_node.computed.setdefault(fingerprint, done)
            for asked, units in untaken.asked.items():
                of_node.asked.setdefault(asked, []).extend(units)
        ours, theirs = self.graph._functions, (into or self.graph)._functions
        return {
            path: of_node for path, of_node in work.items()
            if (of_node.computed or of_node.asked) and (there := theirs.get(path)) is not None and there.reuse_digest == ours[path].reuse_digest
        }

    def again(
        self, inputs: Mapping[str, Any] = MappingProxyType({}), /, *, max_items: int | None = None, from_run: Mapping[type, Any] = MappingProxyType({}),
        timeout: float | None = None, **by_name: Any,
    ) -> Run:
        """Run again with changed inputs, the rest as before: ``run.with_inputs(...).resume(...)``.

        A unit that receives what it received before is not called again.
        """
        return self.with_inputs(inputs, max_items=max_items, **by_name).resume(from_run=from_run, timeout=timeout)

    # -- reading ----------------------------------------------------------------

    @property
    def waiting(self) -> list[Waiting]:
        """Every unit waiting for a person, in node then item order; those inside an embedded graph where its
        node stands."""
        return [waiting_unit(node.placed, unit_loc(node, key), unit) for node, key, unit in self._asking()]

    def _asking(self) -> Iterator[tuple[FunctionAtPath, ItemPath, Asking]]:
        """Every unit waiting for a person, with its node and instance, in node then instance order."""
        for node in self.graph._tree:
            for key, unit in self._units[node.name].items():
                if type(unit) is Asking:
                    yield node, key, unit

    @property
    def not_called(self) -> list[NotCalled]:
        """The calls a node didn't get because an input was blank: which node, which item and which input; those
        inside an embedded graph where its node stands."""
        return [
            NotCalled(loc=unit_loc(node, key), input=unit.input)
            for node in self.graph._tree for key, unit in self._units[node.name].items() if type(unit) is Blank
        ]

    @property
    def results(self) -> dict[str, dict[str, Any]]:
        """Each node's outputs as the run stands: one value on a node that runs once; for one that runs per
        item, a list in item order with ``None`` where an item has none, or ``None`` in place of the list where
        no item could bring a value or the Series isn't made yet."""
        nodes, reader = self.graph._node_at, self._reader_as_it_stands
        return {n: {o: self._shown(reader, nodes[n], o) for o in placed.outputs} for n, placed in self.graph._nodes.items()}

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
        nodes, reader = self.graph._node_at, self._reader_as_it_stands
        return {address: self._shown(reader, nodes[address.partition(".")[0]], address.partition(".")[2]) for address in self.graph.outputs}

    def outputs_json(self) -> str:
        """The graph's outputs (``run.outputs``) as JSON, by address, each written by its output's type as a save
        writes it: a table as its columns, a float that isn't finite as its text, a host's type by its own schema.

        Raises:
            InvalidCallError: The run is not complete (``run_not_complete``), or an output holds a value with no
                JSON form (``state_unwritable``, naming the output).
        """
        nodes = self.graph._node_at
        written = {}
        for address, value in self.outputs.items():
            node = nodes[address.partition(".")[0]]
            output = node.placed.outputs[address.partition(".")[2]]
            try:
                written[address] = (output.column if node.maps else output.slot).to_json(value)
            except pydantic_core.PydanticSerializationError:
                raise InvalidCallError([problem.state_unwritable("outputs", address, at=address, reason="a value has no JSON form")]) from None
        return pydantic_core.to_json(written).decode()

    def _shown(self, reader: AsItStandsReader, node: NodeAtPath, output: str) -> Any:
        """One output of a node as a host is shown it: what a node reading it would be given (``host_reads``),
        as the run stands.

        An output made per item is what a ``Series`` input that keeps blanks would gather of it: ``None`` where
        no item could bring a value, for a blank or a branch not taken outside the items; else every item in
        its place, ``None`` for one with no value or not settled yet. An embedded graph's node holds no unit:
        its output is the one of the function inside that makes it, and for one that runs per item, an output
        made per item inside is a list for each of its items.

        Args:
            reader: Reads the run's units as they stand.
            node: The node, one of the run's own graph.
            output: The output's name.
        """
        if type(node) is EmbeddedGraphAtPath:
            function, output = node.offers[output]
            around = len(node.maps)
        else:
            function, around = node, 0
        if 0 < around < len(function.maps):  # an embedded graph's node that runs per item, of an output made per item inside
            return reader.shown_for_each_item(cast("InputGathersItems", host_reads(function, output, 0)), around, host_reads(function, output, around))
        return reader.shown(host_reads(function, output, 0))

    # -- what a leg reads -------------------------------------------------------

    @property
    def _reader(self) -> ItemReader:
        """Reads what this run's units hold for one item: what a call of a node receives there."""
        return ItemReader(self._units, self.inputs)

    @property
    def _reader_as_it_stands(self) -> AsItStandsReader:
        """Reads what this run's units hold as they stand, waiting for nothing: what a host is shown."""
        return AsItStandsReader(self._units, self.inputs)

    # -- JSON -------------------------------------------------------------------

    def to_json(self) -> str:
        """The run as JSON: its inputs, each node's units as a column per output, and what it went on from."""
        saved, form = self._saved(), self.graph._save_form
        try:
            return form.serializer.to_json(saved, exclude_defaults=True).decode()
        except pydantic_core.PydanticSerializationError:
            at = _unwritable_at(form.inputs, form.outputs, saved)
            raise InvalidCallError([problem.state_unwritable("state", at, at=at, reason="a value has no JSON form")]) from None

    def _saved(self) -> SavedRun[SavedUnits, SavedEarlier[SavedUnits]]:
        """The run as its graph's ``SaveForm`` writes it. A node inside an embedded graph is saved as any node,
        under its path. An embedded graph's node holds no unit, so nothing is written of it."""
        functions = self.graph._functions
        units = {
            node.name: _unit_columns(node.placed.outputs, self._slots(node))
            for node in self.graph._tree if self._units[node.name]
        }
        earlier = {node_name(path): _earlier_columns(functions[path].outputs, of_node) for path, of_node in self._earlier_work.items() if of_node.computed or of_node.asked}
        code = {name: self._ran_as.get(path) or placed.reuse_digest for path, placed in functions.items() if (name := node_name(path)) in units or name in earlier}
        complete = [node.name for node in self.graph._tree if node.path in self._complete]
        return SavedRun(graph=self.graph.fingerprint, state=self.state, max_items=self.max_items, inputs=self.inputs, units=units, earlier=earlier, code=code, complete=complete)

    def _slots(self, node: FunctionAtPath) -> list[Unit | None]:
        """A node's units with a slot for each instance of the node, in instance order; ``None`` where the
        instance hasn't run."""
        units = self._units[node.name]
        if has_every_unit(self._units, node):  # the units as they stand
            return list(units.values())
        return [units.get(instance) for instance in instances(self._units, node.maps, ())[0]]

    def _copy(self, *, share_units: bool = False, max_items: int | None = None) -> Run:
        """A copy to change: its own units to record into and its own earlier work to take from; or with
        ``share_units`` this run's units of every node, for a caller that replaces whole nodes and takes nothing.

        Args:
            share_units: Whether the copy holds this run's own dict of units for each node.
            max_items: The copy's item ceiling; this run's when ``None``.
        """
        if share_units:
            units, earlier = dict(self._units), dict(self._earlier_work)
        else:  # its own answers to take, as this run holds them too
            units = {name: dict(held) for name, held in self._units.items()}
            earlier = {path: EarlierWork(of_node.at, of_node.computed, {asked: list(left) for asked, left in of_node.asked.items()}) for path, of_node in self._earlier_work.items()}
        return dataclasses.replace(self, max_items=max_items if max_items is not None else self.max_items, _units=units, _earlier_work=earlier)


RunEnd.model_rebuild(_types_namespace={"Run": Run})
RunCheckpoint.model_rebuild(_types_namespace={"Run": Run})


def no_units(graph: Graph) -> Units:
    """The units of a run of ``graph`` that hasn't started: none, for every function of the graph and of each
    graph embedded in it."""
    return {node.name: {} for node in graph._tree}


def run_from_json(graph: Graph, saved: str | bytes | Mapping[str, Any]) -> Run:
    """A run of ``graph`` from what ``run.to_json()`` wrote; a caller reaches it through ``graph.load_run``.

    A graph with problems takes no run, so they are what is refused, before anything of the save is read. The
    save is read once, by the graph's ``SaveForm``, from its JSON text, so each value is read as its type reads
    JSON: a date, and a float that isn't finite, from their text.

    Raises:
        InvalidCallError: The graph's problems; ``state_of_another_graph``; or ``state_unreadable`` at the place
            pydantic names (the section, the node, the output, the slot), quoting nothing of a value.
    """
    if graph.problems:
        raise InvalidCallError(graph.problems)
    try:
        read = graph._save_form.validator.validate_json(saved if isinstance(saved, (str, bytes)) else pydantic_core.to_json(saved, inf_nan_mode="strings"))
    except pydantic_core.PydanticSerializationError:
        raise InvalidCallError([problem.state_unreadable("state", at="the run", reason="it is not JSON")]) from None
    except ValidationError as error:
        first = error.errors(include_url=False, include_input=False)[0]
        if first["type"] == "union_tag_invalid" and not first["loc"]:  # a save is read as one of this graph, which its fingerprint says it is
            raise InvalidCallError([problem.state_of_another_graph("state", saved=first["ctx"]["tag"], graph=graph.fingerprint)]) from None
        at = ".".join(str(step) for step in first["loc"][1:]) or "the run"  # a place in the save comes after the fingerprint it was read under
        raise InvalidCallError([problem.state_unreadable("state", at=at, reason=value_free_errors(error)[0]["msg"])]) from None
    return _run_of(graph, read)


def _run_of(graph: Graph, saved: SavedRun[list[Unit | None], EarlierWork]) -> Run:
    """A run of ``graph`` from its save as the graph's ``SaveForm`` read it: each node's units in the order of
    their slots, its earlier work an ``EarlierWork``.

    Raises:
        InvalidCallError: A node doesn't hold one slot for each instance the saved run made of it
            (``state_unreadable``).
    """
    run = Run(graph, saved.inputs, saved.max_items, no_units(graph), {}, state=saved.state)
    for node in graph._tree:  # each after the nodes it reads, so the Series its instances are of are in place
        if (slots := saved.units.get(node.name)) is not None:
            made, _ = instances(run._units, node.maps, ())
            if len(slots) != len(made):
                at, reason = f"units.{node.name}", f"it holds {len(slots)} slots for the {len(made)} instances the saved run made of the node"
                raise InvalidCallError([problem.state_unreadable("state", at=at, reason=reason)])
            run._units[node.name] = {instance: unit for instance, unit in zip(made, slots, strict=True) if unit is not None}
    functions = graph._functions
    run._ran_as = {path: digest for name, digest in saved.code.items() if digest != functions[path := node_path(name)].reuse_digest}
    run._earlier_work = {path: of_node for name, of_node in saved.earlier.items() if (path := node_path(name)) not in run._ran_as}  # work of other code is never taken
    run._complete = frozenset(node_path(name) for name in saved.complete)
    return run


def checked_leg(*, max_items: int | None = None, timeout: float | None = None, checkpoint_every: float | None = None) -> None:
    """A leg's ceiling (at least 1 item) and its seconds (a finite number above 0) are refused at the call."""
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


def _hand_over(loop: asyncio.AbstractEventLoop, events: asyncio.Queue[Any], event: Any) -> None:
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
