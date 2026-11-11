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
open inputs read what the embedding node is given, and the embedding node's
own unit is what the inner graph's outputs came to for that item, there once
every one of them is settled. Where something is, at any depth of embedded
graphs, is one path of node ids and item positions: ``("e", 0, "ask", 2)`` is
``ask``'s third item inside ``e``'s first.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import threading
import time
from collections.abc import AsyncGenerator, Callable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Literal, overload

import pydantic_core
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PlainSerializer,
    PositiveInt,
    TypeAdapter,
    ValidationError,
)
from pydantic import Field as PydanticField

from conductor._internal.ask import Question, Waiting, answer_outputs, waiting_questions
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import InvalidCallError, Problem, make_problem, value_free_errors
from conductor._internal.fields import SKIPPED, Field, FieldValueError, written_table
from conductor._internal.graph import (
    Cell,
    Fixed,
    FunctionAtPath,
    Gather,
    GraphAtPath,
    Item,
    Maps,
    NodeAtPath,
    NodePath,
    Open,
    PlacedFunction,
    SeriesOfOne,
)
from conductor._internal.node_definition import Returns, from_run_values

if TYPE_CHECKING:
    from conductor._internal.graph import Graph, PlacedNode
    from conductor._internal.leg import Event, _EventBase

#: An instance of a node: the place of its item in each map the node runs in, outermost first; ``()`` for a node
#: that runs once.
Key = tuple[int, ...]

#: Where something is in a run: a node's id, then the place of its item in each of its scope's maps; for what is
#: inside an embedded graph, the embedding node and its item first, ``("e", 0, "ask", 2)``.
Path = tuple[str | int, ...]

#: Each node's units by instance, for the nodes of one graph of the tree by id.
_UnitsOfAGraph = dict[str, dict[Key, "Unit"]]


def unit_loc(node: NodeAtPath, key: Key) -> Path:
    """Where the instance ``key`` of ``node`` is: the node's id, then the places of its own scope's items, after
    the same for each embedded graph's node it is inside, ``("e", 0, "ask", 2)``.

    Args:
        node: The node.
        key: The instance; one cut off after the places of the embedded graph's node gives the node's id alone
            at the end, ``("e", 0, "ask")``.
    """
    if node.embedding is None:
        return (node.placed.id, *key)
    base = node.base
    return (*unit_loc(node.embedding, key[:base]), node.placed.id, *key[base:])


def path_text(path: Path) -> str:
    """A path as a waiting id writes it: each node id, ``@`` before each place of its item, and ``/`` between
    nodes, ``e@0/ask@2``."""
    return "".join(f"@{step}" if isinstance(step, int) else f"/{step}" for step in path)[1:]

#: How a run stands while nothing has ended it early: not started, in a leg, waiting for a person, or done.
GoingStatus = Literal["ready", "running", "complete", "waiting"]
#: How a leg ended before the run was done: a node failed, a Series outgrew ``max_items``, a cancel, or the deadline.
EndedStatus = Literal["failed", "stopped", "cancelled", "timed_out"]
#: How a run stands after a leg.
RunStatus = GoingStatus | EndedStatus


@dataclass(frozen=True, slots=True)
class Going:
    """A run nothing has ended early.

    Attributes:
        status: Not started (``ready``), in a leg (``running``), waiting for a person, or ``complete``.
    """

    status: GoingStatus


@dataclass(frozen=True, slots=True)
class EndedEarly:
    """A run whose last leg ended before it was done, and why; ``resume`` goes on from where it stopped.

    Attributes:
        status: ``failed`` (a node failed), ``stopped`` (a Series outgrew ``max_items``), ``cancelled`` or
            ``timed_out``.
        cause: Why, as a problem with its place and code.
    """

    status: EndedStatus
    cause: Problem


#: How a run stands: going on, or ended early with a cause.
RunState = Going | EndedEarly

_READY, _RUNNING, _WAITING, _COMPLETE = Going("ready"), Going("running"), Going("waiting"), Going("complete")
#: The one ``Going`` of each status, so a run is asked how it stands by identity.
_GOING: Mapping[GoingStatus, Going] = {going.status: going for going in (_READY, _RUNNING, _WAITING, _COMPLETE)}


class _Pending:
    def __repr__(self) -> str:
        return "PENDING"


#: What a unit's inputs are while something above it hasn't run yet.
PENDING: Any = _Pending()

#: ``run.answer``'s second argument when it is given a mapping of answers instead.
_ONE_ANSWER: Any = object()


class NotCalled(ConductorModel):
    """A call a node didn't get, because an input got a blank there.

    Attributes:
        loc: Which call it was, as a ``Problem``'s ``loc`` says it: the node, then the place of its item in each
            Series the node runs per item of; inside an embedded graph, the embedding node and its item first.
        input: The input that got nothing.
    """

    loc: Path
    input: str


# -- what a unit came to ------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Done:
    """A unit whose node was called and returned.

    Attributes:
        out: Its outputs by name: a value, ``None`` for a blank, ``SKIPPED`` for a branch not taken, a list for a
            ``Series``.
    """

    out: dict[str, Any]


@dataclass(frozen=True, slots=True)
class Answered(Done):
    """A unit a person answered: its outputs are the answer.

    Attributes:
        question: What the person was asked.
    """

    question: Question


@dataclass(frozen=True, slots=True)
class Skipped:
    """A unit on a branch not taken: an input came only from branches not taken, or the node returned ``SKIPPED``.
    What reads it is skipped too."""


@dataclass(frozen=True, slots=True)
class Blank:
    """A unit whose node wasn't called because an input that takes no blank got one; its outputs are blank.

    Attributes:
        input: The input that got nothing.
    """

    input: str


@dataclass(frozen=True, slots=True)
class Asking:
    """A unit waiting for a person: its node raised ``Ask``.

    Attributes:
        question: What the person is asked.
    """

    question: Question


#: What one node came to in one instance of its scope. A run and its copies share their units, so none is changed.
Unit = Done | Answered | Skipped | Blank | Asking

#: The one skipped unit.
SKIPPED_UNIT = Skipped()


@dataclass(slots=True)
class Earlier:
    """The work of one node that a run may take from the run it went on from, in place of calling the node.

    A person's answer, or a question still waiting, is taken once, by the next unit in order that received the
    same under the same items above it. A node's own result is taken by the unit at the same place when it
    received the same there, and else by what was received, whichever unit made it.

    Attributes:
        at: Each result by the instance its unit stood in, with the fingerprint of what it received.
        computed: A node's results by the fingerprint of what the unit received.
        asked: A person's answers and the questions still waiting, by the fingerprint of what the unit received and
            of the items above it, in the order they stood.
    """

    at: dict[Key, tuple[str, Done]] = field(default_factory=dict[Key, tuple[str, Done]])
    computed: dict[str, Done] = field(default_factory=dict[str, Done])
    asked: dict[tuple[str, str], list[Answered | Asking]] = field(default_factory=dict[tuple[str, str], list["Answered | Asking"]])


class Run:
    """One run of a graph. Every call returns a new ``Run``; ``to_json`` is all of it.

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
    state: RunState
    #: Each node's units by instance, by the node's id, for each graph of the tree by the path of the node that
    #: embeds it: ``_units[()]["w"]`` for a node of the graph itself, ``_units[("e",)]["ask"]`` for one inside ``e``.
    _units: dict[NodePath, _UnitsOfAGraph]
    #: What each node may take from the run this one went on from.
    _earlier: dict[NodePath, Earlier]
    #: For a node whose units a saved run made with other code or another signature, what that code was; they are never carried.
    _ran_as: dict[NodePath, str]
    #: The nodes whose units were recorded out of instance order (threads finish when they finish), sorted when read.
    _unordered: set[NodePath]
    #: The nodes every unit of which was finished when the last leg ended, so a later leg doesn't say them complete again.
    _complete: frozenset[NodePath]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("a run is made with graph.with_inputs(...), graph.run(...) or graph.load_run(saved)")

    def __repr__(self) -> str:
        return f"<Run {self.status}: {sum(len(units) for graph in self._units.values() for units in graph.values())} units>"

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
        checkpoint_every: float | None = None,
    ) -> AsyncGenerator[Event]:
        """``resume``, saying what happens as it happens: ``async for event in run.stream(): ...``.

        The leg runs on its own thread, so a slow node never holds the event
        loop. The last event is ``RunEnd``, whose ``run`` is the run as the leg
        left it. Setting ``cancel`` ends the leg before its next unit, and so
        does leaving the loop early once the stream is closed, which Python
        does for an async generator only when it is collected: hold it with
        ``async with contextlib.aclosing(run.stream()) as events:`` and loop
        over ``events``. With ``checkpoint_every`` (seconds), a
        ``RunCheckpoint`` comes at most that often while units finish, and once
        when the leg is stopped with calls still on threads: a host that stores
        its ``run`` loses only the calls in flight if the process dies. A call
        that can't start raises here, before anything is iterated.
        """
        checked_leg(max_items=max_items, timeout=timeout, checkpoint_every=checkpoint_every)
        return self._stream(max_items, from_run_values(self.graph._needs_from_run, from_run), timeout, cancel, checkpoint_every)

    async def _stream(
        self, max_items: int | None, from_run: dict[type, Any], timeout: float | None, cancel: asyncio.Event | None, checkpoint_every: float | None,
    ) -> AsyncGenerator[Event]:
        loop, events = asyncio.get_running_loop(), asyncio.Queue["Event | None"]()
        cancelled = threading.Event()
        deadline = _deadline(timeout)

        def stop() -> EndedEarly | None:
            if cancelled.is_set() or (cancel is not None and cancel.is_set()):
                return EndedEarly("cancelled", make_problem("run_cancelled"))
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
        self, max_items: int | None, from_run: dict[type, Any], emit: Callable[[_EventBase], None] | None, stop: Callable[[], EndedEarly | None] | None,
        checkpoint_every: float | None = None,
    ) -> Run:
        from conductor._internal.leg import Leg

        run = self._copy(max_items=max_items if max_items is not None else self.max_items)
        Leg(run, emit, stop, from_run, checkpoint_every).go()
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
        when that question waits elsewhere). A cancelled run takes no answers
        (``run_cancelled``): ``resume`` it, and it carries on and waits again,
        or start a new run.
        """
        if self.status == "cancelled":
            raise InvalidCallError([make_problem("run_cancelled", "status")])
        if value is not _ONE_ANSWER:
            given: Mapping[str, Any] = {unit.id if isinstance(unit, Waiting) else str(unit): value}
        elif isinstance(unit, Mapping):
            given = {str(key): one for key, one in unit.items()}
        else:
            raise TypeError("run.answer takes a unit and its answer, or a mapping of unit ids to answers")
        problems: list[Problem] = []
        answered: dict[tuple[FunctionAtPath, Key], Unit] = {}
        for unit_id, one in given.items():
            place, _, fingerprint = unit_id.partition("~")
            try:
                loc, nodes, key = _parse_place(place)
            except ValueError:  # a place in it isn't a number
                problems.append(make_problem("not_waiting", unit_id.partition("@")[0], waiting_id=unit_id, status=self.status))
                continue
            node = self.graph._node_at.get(nodes)
            found = self._units[node.inside][node.placed.id].get(key) if node is not None else None
            if not isinstance(node, FunctionAtPath) or not isinstance(found, Asking):
                moved = {"moved_to": went} if fingerprint and (went := self._moved_to(nodes, fingerprint)) else {}
                problems.append(make_problem("not_waiting", *loc, waiting_id=unit_id, status=self.status, **moved))
                continue
            placed = node.placed
            if fingerprint and _question_fingerprint(placed, found) != fingerprint:
                moved = {"moved_to": went} if (went := self._moved_to(nodes, fingerprint)) else {}
                problems.append(make_problem("answer_stale", *loc, waiting_id=unit_id, current=waiting_unit(node, key, found).id, **moved))
                continue
            if (node, key) in answered:
                problems.append(make_problem("invalid_answer", *loc, waiting_id=unit_id, reason="the unit is answered twice in one call"))
                continue
            read, faults = answer_outputs(placed, one, found.question.overrides)
            problems.extend(
                make_problem("invalid_answer", *loc, waiting_id=unit_id, reason=reason, **({"question": q} if q else {})) for q, reason in faults
            )
            answered[(node, key)] = Answered(kept_outputs(placed.outputs, read), found.question) if not faults else found
        if problems:
            raise InvalidCallError(problems)
        run = self._copy(share_units=True)
        for node in {changed for changed, _ in answered}:  # its own units for each node it changes; every other node's are shared
            run._units[node.inside][node.placed.id] = dict(self._units[node.inside][node.placed.id])
        for (node, key), now in answered.items():
            run._units[node.inside][node.placed.id][key] = now
        return run

    def _moved_to(self, nodes: NodePath, fingerprint: str) -> str | None:
        """The id of the unit where the question a stale id names now waits, if it waits elsewhere on the node at
        the path ``nodes``."""
        for node, key, unit in self._asking():
            if node.path == nodes and _question_fingerprint(node.placed, unit) == fingerprint:
                return waiting_unit(node, key, unit).id
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
        fresh = self.graph.with_inputs({**given, **changed}, max_items=max_items if max_items is not None else self.max_items)
        fresh._earlier = self._carried()
        return fresh

    def forget(self, *node_ids: str) -> Run:
        """A run of the same inputs that has forgotten these nodes' work, ready to ``resume`` or ``stream``.

        Every other unit is carried as ``with_inputs`` carries it, so only the
        forgotten nodes are called again, and a node below them that then
        receives what it received before is not. A host forgets a node whose
        access it must check again.
        """
        unknown = [n for n in node_ids if n not in self.graph._nodes]
        if unknown:
            raise InvalidCallError([make_problem("node_not_in_graph", n, node_id=n, nodes=list(self.graph._nodes)) for n in unknown])
        fresh = self.with_inputs()
        for path in [path for path in fresh._earlier if path[0] in node_ids]:  # an embedded graph's id forgets every node inside it
            del fresh._earlier[path]
        return fresh

    def _carried(self, into: Graph | None = None) -> dict[NodePath, Earlier]:
        """What a run going on from this one may take, by node: each node's finished and waiting units, in
        instance order, then what this run carried and didn't take.

        An embedded graph's own unit is never carried: the units of the nodes inside it are, so a later run
        takes them node by node. ``into`` another graph keeps only the nodes it holds as they are here (by
        ``reuse_digest``): the others' units could never be taken there, and carried they would only make its
        save larger.
        """
        carried: dict[NodePath, Earlier] = {}
        for node in self.graph._tree:
            if isinstance(node, GraphAtPath) or not node.policy.reuse_when_inputs_match or node.path in self._ran_as:
                continue
            placed = node.placed
            earlier = carried[node.path] = Earlier()
            for key, unit in self._in_order(node):
                if not isinstance(unit, (Done, Asking)):
                    continue
                fingerprint = _fingerprint(placed.reuse_digest, self._inputs(node, key), plain=placed.plain_inputs)
                if fingerprint is None:
                    continue
                if isinstance(unit, (Answered, Asking)):
                    earlier.asked.setdefault((fingerprint, self._lineage(node, key)), []).append(unit)
                else:
                    earlier.at[key] = (fingerprint, unit)
                    earlier.computed.setdefault(fingerprint, unit)
        for path, left in self._earlier.items():  # what this run was lent and no unit of it received
            earlier = carried.setdefault(path, Earlier())
            for fingerprint, done in left.computed.items():
                earlier.computed.setdefault(fingerprint, done)
            for asked, units in left.asked.items():
                earlier.asked.setdefault(asked, []).extend(units)
        ours, theirs = self.graph._functions, (into or self.graph)._functions
        return {
            path: earlier for path, earlier in carried.items()
            if (earlier.computed or earlier.asked) and (there := theirs.get(path)) is not None and there.reuse_digest == ours[path].reuse_digest
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
        return [waiting_unit(node, key, unit) for node, key, unit in self._asking()]

    def _asking(self) -> Iterator[tuple[FunctionAtPath, Key, Asking]]:
        """Every unit waiting for a person, with its node and instance, in node then instance order."""
        for node in self.graph._tree:
            if isinstance(node, FunctionAtPath):
                for key, unit in self._in_order(node):
                    if isinstance(unit, Asking):
                        yield node, key, unit

    def _in_order(self, node: NodeAtPath) -> Any:
        """A node's units in instance order."""
        units = self._units[node.inside][node.placed.id]
        return sorted(units.items()) if node.path in self._unordered else units.items()

    @property
    def not_called(self) -> list[NotCalled]:
        """The calls a node didn't get because an input was blank: which node, which item and which input; those
        inside an embedded graph where its node stands."""
        return [
            NotCalled(loc=unit_loc(node, key), input=unit.input)
            for node in self.graph._tree for key, unit in self._units[node.inside][node.placed.id].items() if isinstance(unit, Blank)
        ]

    @property
    def results(self) -> dict[str, dict[str, Any]]:
        """Each node's outputs: one value on a node that ran once, else a list in item order, ``None`` where a
        unit has none."""
        nodes = self.graph._node_at
        return {n: {o: self._collect(nodes[(n,)], o) for o in placed.outputs} for n, placed in self.graph._nodes.items()}

    @property
    def outputs(self) -> dict[str, Any]:
        """The graph's outputs (``graph.outputs``), by address, once the run is complete; ``results`` reads a run in any state."""
        match self.state:
            case Going(status="complete"):
                pass
            case Going(status=status):
                raise InvalidCallError([make_problem("run_not_complete", "outputs", status=status, reason="")])
            case EndedEarly(status=status, cause=cause):
                raise InvalidCallError([make_problem("run_not_complete", "outputs", status=status, reason=cause.msg)])
        nodes = self.graph._node_at
        return {address: self._collect(nodes[(address.partition(".")[0],)], address.partition(".")[2]) for address in self.graph.outputs}

    def _collect(self, node: NodeAtPath, output: str) -> Any:
        """One output of a node over its units, in instance order."""
        units = self._units[node.inside][node.placed.id]
        values = []
        for key in self._instances(node.maps, ())[0]:
            unit = units.get(key)
            value = unit.out[output] if isinstance(unit, Done) else None
            values.append(None if value is SKIPPED else [None if v is SKIPPED else v for v in value] if isinstance(value, list) else value)
        return values if node.maps else values[0]

    def _offered(self, node: GraphAtPath, key: Key) -> Any:
        """What the embedded graph's node ``node`` offers in the instance ``key``: each output of its inner graph
        (``graph.outputs``), by address, as a unit keeps it.

        An output made once inside is the inner unit's own: its value, ``SKIPPED`` for a branch not taken, ``None``
        for a blank, so what reads the embedded graph's node is skipped or left uncalled as it would be reading
        the node inside. One made per item inside, at any depth, is every item's in order, and a ``Series`` made
        per item is a list of lists.

        Returns:
            The outputs by address, or ``PENDING`` until every one of them is settled for the instance.
        """
        units, base = self._units[node.path], len(key)
        out: dict[str, Any] = {}
        for address, cell, maps, lists in node.offers:
            if len(maps) == base:  # made once inside: the inner unit's own
                unit = units[cell.node_id].get(key)
                kind = type(unit)
                if kind is Done or kind is Answered:
                    value = unit.out[cell.output]  # type: ignore[union-attr]
                elif kind is Skipped or kind is Blank:
                    value = SKIPPED if kind is Skipped else None
                else:
                    return PENDING
            else:
                instances, known = self._instances(maps, key)
                if not known:
                    return PENDING
                value = [_cell(units, base, cell, instance) for instance in instances]
                for one in value:
                    if one is PENDING:
                        return PENDING
                if lists:
                    value = [[None if item is SKIPPED else item for item in one] if type(one) is list else one for one in value]
            out[address] = value
        return out

    # -- what a leg reads -------------------------------------------------------

    def _reused(self, earlier: Earlier, node: FunctionAtPath, key: Key, kwargs: dict[str, Any], in_turn: Callable[[], bool]) -> Any:
        """The unit of ``earlier`` that ``node`` takes in ``key`` in place of a call on ``kwargs``; ``None`` when
        there is none.

        A person's answer, or a question still waiting, is taken once, in order: when one is there to take and
        ``in_turn`` says a unit before this one hasn't been tried yet, this is ``PENDING``, tried again once it
        has. It is taken only under the same items above, so two documents' identical lines keep each document's
        answers when the documents move. A node's own result is the one of the unit at the same place, when that
        received the same, and else one any earlier unit computed from the same.

        Args:
            earlier: What the node may take: this run's own to take from.
            node: The node.
            key: The instance.
            kwargs: What the unit received.
            in_turn: Whether every unit of the node before this one has been tried.
        """
        placed = node.placed
        fingerprint = _fingerprint(placed.reuse_digest, kwargs, plain=placed.plain_inputs)
        if fingerprint is None:
            return None
        if earlier.asked:
            answers = earlier.asked.get((fingerprint, self._lineage(node, key)))
            if answers:
                if not in_turn():
                    return PENDING
                return answers.pop(0)
        there = earlier.at.get(key)
        if there is not None and there[0] == fingerprint:
            return there[1]
        return earlier.computed.get(fingerprint)

    def _lineage(self, node: NodeAtPath, key: Key) -> str:
        """A fingerprint of the item of each map above the innermost that the instance ``key`` of ``node`` sits
        in, those of the embedded graphs' nodes around it included. A unit's innermost item is in what it
        received."""
        above = [
            self._units[series.inside][series.placed.id][key[:depth]].out[output][key[depth]]  # type: ignore[union-attr]
            for depth, (series, output) in enumerate(node.maps[:-1])
        ]
        return (_fingerprint("", {"items": above}) or "") if above else ""

    def _call(self, node: FunctionAtPath, kwargs: dict[str, Any], from_run: Mapping[type, Any]) -> tuple[Any, BaseException | None]:
        """Call ``node`` on what a unit received: its outputs as a run keeps them, or what went wrong.

        A value an edge brings is checked against the input first where its source's type doesn't settle it, and what
        the node returns against its declared outputs; either refusal is a ``FieldValueError``, which quotes no value.
        """
        placed = node.placed
        definition = placed.definition
        try:
            for name in node.checked:
                kwargs = {**kwargs, name: _checked(placed.inputs[name].spec, name, kwargs[name], code="invalid_edge_value")}
            if definition.from_run:
                kwargs = {**kwargs, **{name: from_run[needed] for name, needed in definition.from_run.items()}}
            returned = definition.function(**kwargs)
        except BaseException as error:  # noqa: BLE001  a node's own failure, or its Ask, is the run's to record
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return None, error
        try:
            return checked_outputs(definition.returns, placed.outputs, returned), None
        except FieldValueError as wrong:
            return None, wrong

    def _instances(self, maps: Maps, under: Key, *, counted: bool = False) -> tuple[list[Key], bool]:
        """The instances of a node that runs in ``maps`` under the instance ``under`` of a node that runs in the
        first of them, and whether all of them exist yet: a Series not made yet, or one a person is asked for,
        has none so far.

        With ``counted``, the innermost map's instances are only counted: ``[(n,)]`` stands for ``n`` of them, unbuilt.
        """
        start, deepest = len(under), len(maps)
        if start >= deepest:
            return [under[:deepest]], True
        keys, complete = [under], True
        for depth in range(start, deepest):
            series, output = maps[depth]
            units = self._units[series.inside][series.placed.id]
            below: list[Key] = []
            for parent in keys:
                unit = units.get(parent)
                if unit is None or type(unit) is Asking:
                    complete = False
                    continue
                count = _length(unit.out[output]) if isinstance(unit, Done) else 0
                if counted and depth == deepest - 1:
                    below = [(below[0][0] + count,)] if below else [(count,)]
                else:
                    below.extend((*parent, i) for i in range(count))
            keys = below
        return keys, complete

    def _inputs(self, node: FunctionAtPath, key: Key) -> Any:
        """What ``node`` receives in the instance ``key``.

        An open input of a node inside an embedded graph is read as the embedded graph's node is given it: by
        that node's plan, in the graph that node is in.

        Returns:
            The keyword arguments of the call; ``PENDING`` while something it reads hasn't run; or the unit it
            comes to uncalled: ``Skipped`` when an input comes only from branches not taken, else ``Blank`` when
            an input that takes no blank got one. Every input is read before a blank is the answer.
        """
        kwargs: dict[str, Any] = {}
        blank: Blank | None = None
        gathered: dict[str, tuple[Gather, _UnitsOfAGraph, list[Any]]] = {}
        own, own_base = self._units[node.inside], node.base
        for name, (plan, giver) in node.inputs.items():
            kind = type(plan)
            if kind is Fixed:
                kwargs[name] = plan.value  # type: ignore[union-attr]
                continue
            if kind is Open:  # a first node of the graph itself, or of a graph embedded in one: the caller gives it
                kwargs[name] = self.inputs[plan.address]  # type: ignore[union-attr]
                continue
            if giver is None:
                units, base = own, own_base
            else:
                units, base = self._units[giver.inside], giver.base
            if kind is Gather:
                value = _gathered(units, base, plan, key)  # type: ignore[arg-type]
            else:  # one value: the first edge whose branch was taken
                value = SKIPPED
                for source in plan.sources:  # type: ignore[union-attr]
                    value = _cell(units, base, source, key)
                    if value is not SKIPPED:
                        break
                if kind is SeriesOfOne and value is not None and value is not SKIPPED and value is not PENDING:
                    value = [value]
            if value is PENDING:
                return PENDING
            if value is SKIPPED:
                return SKIPPED_UNIT
            if value is None:
                if not plan.spec.whole.optional and blank is None:
                    blank = Blank(name)
                kwargs[name] = None
            elif kind is Gather:
                gathered[name] = (plan, units, value)  # type: ignore[assignment]
            else:
                kwargs[name] = value
        if blank is not None:
            return blank
        for name, (plan, units, items) in gathered.items():
            kwargs[name] = _aligned(plan, units, items, gathered)
        return kwargs

    # -- JSON -------------------------------------------------------------------

    def to_json(self) -> str:
        """The run as JSON: its inputs, every unit, and what it went on from, a column per output of each node."""
        try:
            return self._saved().model_dump_json(exclude_defaults=True)
        except _SaveError as unwritable:
            raise InvalidCallError([make_problem("state_unwritable", "state", unwritable.at, at=unwritable.at, reason=unwritable.reason)]) from None

    def _saved(self) -> _SavedGoingRun | _SavedEndedRun:
        """The run as its save. A node inside an embedded graph is saved as any node, under its path. The unit
        of an embedded graph's node is not written: it is what the units inside came to, and is read from them.

        Raises:
            _SaveError: A value has no JSON form; it names where.
        """
        functions = self.graph._functions
        units: dict[str, _SavedUnits] = {}
        for path, placed in functions.items():
            kept = self._units[path[:-1]][path[-1]]
            if kept:
                try:
                    units["/".join(path)] = _dump_units(placed, dict(sorted(kept.items())) if path in self._unordered else kept)
                except _SaveError as unwritable:
                    raise _SaveError("/".join((*path[:-1], unwritable.at)), unwritable.reason) from None
        earlier = {"/".join(path): _dump_earlier(functions[path], lent) for path, lent in self._earlier.items() if lent.computed or lent.asked}
        inputs = {address: _dump_input(self.graph, address, v) for address, v in self.inputs.items()}
        non_finite: dict[str, list[_Path]] = {}
        for address, written_input in inputs.items():
            if not self.graph._input_fields[address].spec.type.hashes_plainly:
                _find_non_finite(written_input, [], non_finite.setdefault(address, []))
        code = {
            "/".join(path): self._ran_as.get(path) or placed.reuse_digest
            for path, placed in functions.items() if "/".join(path) in earlier or "/".join(path) in units
        }
        rest: dict[str, Any] = {
            "graph": self.graph.fingerprint, "max_items": self.max_items, "inputs": inputs, "non_finite_inputs": {a: paths for a, paths in non_finite.items() if paths},
            "units": units, "earlier": earlier, "code": code, "complete": ["/".join(node.path) for node in self.graph._tree if node.path in self._complete],
        }
        match self.state:
            case Going(status=status):
                return _SavedGoingRun.model_construct(status=status, **rest)
            case EndedEarly(status=ended, cause=cause):
                return _SavedEndedRun.model_construct(status=ended, cause=cause, **rest)

    def _copy(self, *, share_units: bool = False, **changes: Any) -> Run:
        """A copy to change: its own units to record into and its own carried answers to take from; or with
        ``share_units`` this run's units of every node, for a caller that replaces whole nodes and takes nothing."""
        if share_units:
            units = {inside: dict(nodes) for inside, nodes in self._units.items()}
        else:
            units = {inside: {n: dict(held) for n, held in nodes.items()} for inside, nodes in self._units.items()}
        copy = start_run(self.graph, self.inputs, changes.get("max_items", self.max_items), dict(self._earlier), units)
        if not share_units:
            for path, earlier in self._earlier.items():  # its own answers to take, as this run holds them too
                copy._earlier[path] = Earlier(earlier.at, earlier.computed, {asked: list(units) for asked, units in earlier.asked.items()})
        copy.state, copy._unordered = self.state, set(self._unordered)
        copy._ran_as, copy._complete = self._ran_as, self._complete
        return copy


def _cell(units: _UnitsOfAGraph, base: int, cell: Cell, key: Key) -> Any:
    """What ``cell`` holds for the instance ``key`` or one under it, as an edge brings it.

    Args:
        units: The units of the graph the cell's node is in.
        base: How many places of ``key`` are of the items of the embedded graph's node that graph is inside.
        cell: The cell.
        key: The instance of the node that reads.

    Returns:
        The value (an ``Item``'s is the item of ``key``, and a field named on the edge is read from it);
        ``PENDING`` while its unit hasn't run or waits for a person; ``SKIPPED`` for a branch not taken;
        ``None`` for a blank.
    """
    depth = base + len(cell.scope)
    unit = units[cell.node_id].get(key[:depth])
    kind = type(unit)
    if kind is Done or kind is Answered:
        value = unit.out[cell.output]  # type: ignore[union-attr]
    elif kind is Skipped:
        return SKIPPED
    elif kind is Blank:
        return None
    else:
        return PENDING
    if value is None or value is SKIPPED or type(cell) is Cell:
        return value
    if type(cell) is Item:
        value = value[key[depth]]
    field = cell.field  # type: ignore[attr-defined]
    return value if field is None or value is None or value is SKIPPED else _field_of(value, field)


def _gathered(units: _UnitsOfAGraph, base: int, plan: Gather, key: Key) -> Any:
    """The items a gathering input gets in the instance ``key``, with ``None`` for an item that has no value.

    Args:
        units: The units of the graph the plan was made in.
        base: How many places of ``key`` are of the items of the embedded graph's node that graph is inside.
        plan: The input's plan.
        key: The instance of the node that gathers.

    Returns:
        The list, one place per item of the Series; ``PENDING`` until every item is settled; ``SKIPPED`` when
        every edge is skipped where the node runs or above; ``None`` when the Series is blank.
    """
    key = key[: base + len(plan.series.scope)]
    live = []
    for source, gate in zip(plan.sources, plan.gates, strict=True):
        taken = True
        for cells in gate:
            held = [_cell(units, base, cell, key) for cell in cells]
            if PENDING in held:
                return PENDING
            if all(value is SKIPPED for value in held):
                taken = False
        if taken:
            live.append(source)
    if not live:
        return SKIPPED
    series = _cell(units, base, plan.series, key)
    if series is None:
        return None
    if len(live) == 1 and type(live[0]) is Item:  # the Series itself: its items, or one field of each
        wanted = live[0].field
        if wanted is None:
            return [None if item is SKIPPED else item for item in series] if any(item is SKIPPED for item in series) else series
        return [None if item is SKIPPED or item is None else _field_of(item, wanted) for item in series]
    items = []
    for place in range(len(series)):
        value = SKIPPED
        for source in live:
            value = _cell(units, base, source, (*key, place))
            if value is PENDING:
                return PENDING
            if value is not SKIPPED:
                break
        items.append(None if value is SKIPPED else value)
    return items


def no_units(graph: Graph) -> dict[NodePath, _UnitsOfAGraph]:
    """The units of a run of ``graph`` that hasn't started: none, for every node of the graph and of each graph
    embedded in it."""
    return {inside: {node_id: {} for node_id in ids} for inside, ids in graph._ids_inside.items()}


def start_run(graph: Graph, inputs: dict[str, Any], max_items: int | None, earlier: dict[NodePath, Earlier], units: dict[NodePath, _UnitsOfAGraph]) -> Run:
    """A run of ``graph`` not started; the one way a ``Run`` is made.

    Args:
        graph: The graph.
        inputs: The caller's inputs, already checked, by address.
        max_items: The item ceiling, or ``None``.
        earlier: What each node may take from an earlier run, by node.
        units: Each node's units, as ``Run._units`` holds them.
    """
    run = Run.__new__(Run)
    run.graph, run.inputs, run.max_items, run._units = graph, inputs, max_items, units
    run.state = _READY
    run._earlier = earlier
    run._ran_as = {}
    run._unordered = set()
    run._complete = frozenset()
    return run


def waiting_unit(node: FunctionAtPath, key: Key, unit: Asking) -> Waiting:
    """A waiting unit as a host shows it. Its id is where it is as text, then ``~`` and the fingerprint of the
    question asked there, as ``run.answer`` reads it.

    Args:
        node: The node that asks.
        key: The instance it asks in.
        unit: The unit.
    """
    question, loc, placed = unit.question, unit_loc(node, key), node.placed
    return Waiting(
        id=f"{path_text(loc)}~{_question_fingerprint(placed, unit)}", loc=loc, prompt=question.prompt,
        questions=waiting_questions(placed, question.overrides), inputs=question.inputs,
    )


def _aligned(plan: Gather, units: _UnitsOfAGraph, items: list[Any], gathered: Mapping[str, tuple[Gather, _UnitsOfAGraph, list[Any]]]) -> list[Any]:
    """The items one gathering input receives: those that have a value in every input of the node that gathers
    the same Series, so those inputs stay in step; an input whose items may be blank keeps a blank in its place.

    Args:
        plan: The input's plan.
        units: The units of the graph the plan was made in, which with the plan's Series says which Series it is.
        items: What the input gathered, one place per item of the Series.
        gathered: Every gathering input of the node, by name: its plan, the units of its plan's graph, its items.
    """
    together = [(others, other.spec.keeps_blanks) for other, theirs, others in gathered.values() if other.series == plan.series and theirs is units]
    if len(together) == 1:
        return items if together[0][1] else [item for item in items if item is not None]  # by identity: an item may be a frame, which has no plain ==
    return [item for place, item in enumerate(items) if all(keeps or others[place] is not None for others, keeps in together)]


def _field_of(value: Any, name: str) -> Any:
    """One field of a record: a mapping's key, else an attribute."""
    return value[name] if type(value) is dict or isinstance(value, Mapping) else getattr(value, name)


def run_from_json(graph: Graph, saved: str | bytes | Mapping[str, Any]) -> Run:
    """A run of ``graph`` from what ``run.to_json()`` wrote; a caller reaches it through ``graph.load_run``.

    A graph with problems takes no run, so they are what is refused, before anything of the save is read.
    """
    if graph.problems:
        raise InvalidCallError(graph.problems)
    try:
        # read from its JSON text, so the values it holds are the save's own and its marked floats are put in place
        data = _SAVED.validate_json(saved if isinstance(saved, (str, bytes)) else pydantic_core.to_json(saved, inf_nan_mode="strings"))
    except pydantic_core.PydanticSerializationError:
        raise InvalidCallError([make_problem("state_unreadable", "state", at="the run", reason="it is not JSON")]) from None
    except ValidationError as error:
        # The save's own shape is conductor's, so its whole place is named: node ids, output names, indexes, never a value.
        first = error.errors(include_input=False)[0]
        # a save is read as the kind its status names, so a place inside it starts with that status, and a fault of the
        # status itself has no place of its own
        loc = ("status",) if first["type"].startswith("union_tag") else first["loc"][1:]
        at = ".".join(str(step) for step in loc) or "the run"
        raise InvalidCallError([make_problem("state_unreadable", "state", at=at, reason=value_free_errors(error)[0]["msg"])]) from None
    try:
        return _run_from_saved(graph, data)
    except _SaveError as unreadable:
        raise InvalidCallError([make_problem("state_unreadable", "state", unreadable.at, at=unreadable.at, reason=unreadable.reason)]) from None


def _run_from_saved(graph: Graph, data: _SavedRun) -> Run:
    """A run of ``graph`` from its read save.

    Raises:
        InvalidCallError: The save is of another graph.
        _SaveError: Something in the save doesn't fit the graph, naming where.
    """
    if data.graph != graph.fingerprint:
        raise InvalidCallError([make_problem("state_of_another_graph", "state", saved=data.graph, graph=graph.fingerprint)])
    unknown = sorted({*data.inputs, *data.non_finite_inputs} - graph._input_fields.keys())
    if unknown:
        raise _SaveError(unknown[0], "the graph has no such input")
    functions = graph._functions
    unknown = sorted(text for text in data.units if tuple(text.split("/")) not in functions)
    if unknown:
        raise _SaveError(unknown[0], "the graph has no such node")
    written = {address: _with_non_finite(v, data.non_finite_inputs.get(address, ()), address) for address, v in data.inputs.items()}
    run = start_run(graph, {address: _load_input(graph, address, v) for address, v in written.items()}, data.max_items, {}, no_units(graph))
    for text, dumped in data.units.items():
        path = tuple(text.split("/"))
        if len(dumped.keys) != len(dumped.states):
            raise _SaveError(text, f"it holds {len(dumped.keys)} places for {len(dumped.states)} units")
        try:
            units = run._units[path[:-1]][path[-1]] = dict(zip(dumped.keys, _load_units(functions[path], dumped), strict=True))
        except _SaveError as unreadable:
            raise _SaveError("/".join((*path[:-1], unreadable.at)), unreadable.reason) from None
        if len(units) != len(dumped.keys):
            raise _SaveError(text, "a unit is listed twice")
    other_code = {path for text, digest in data.code.items() if (path := tuple(text.split("/"))) in functions and digest != functions[path].reuse_digest}
    run._ran_as = {path: data.code["/".join(path)] for path in other_code}
    for text, lent in data.earlier.items():
        path = tuple(text.split("/"))
        if path not in functions:
            raise _SaveError(text, "the graph has no such node")
        if path not in other_code:  # units made with other code are never taken
            run._earlier[path] = _load_earlier(functions[path], lent)
    match data:
        case _SavedGoingRun(status=status):
            run.state = _GOING[status]
        case _SavedEndedRun(status=ended, cause=cause):
            run.state = EndedEarly(ended, cause)
    run._complete = frozenset(path for text in data.complete if (path := tuple(text.split("/"))) in graph._node_at)
    for node in graph._tree:  # each after the nodes it reads, so the Series its instances are of are in place
        units = run._units[node.inside][node.placed.id]
        made, _ = run._instances(node.maps, ())
        if isinstance(node, GraphAtPath):  # its units are what the units inside it came to
            for key in made:
                offered = run._offered(node, key)
                if offered is not PENDING:
                    units[key] = Done(offered)
        elif stray := sorted(set(units) - set(made)):
            raise _SaveError("/".join(node.path), f"it holds the unit {list(stray[0])}, of an item the saved run never made")
    return run


def checked_leg(*, max_items: int | None = None, timeout: float | None = None, checkpoint_every: float | None = None) -> None:
    """A leg's ceiling (at least 1 item) and its seconds (a finite number above 0) are refused at the call."""
    problems = []
    if max_items is not None and max_items < 1:
        problems.append(make_problem("invalid_max_items", "max_items", max_items=max_items))
    if timeout is not None and not _seconds(timeout):
        problems.append(make_problem("invalid_timeout", "timeout", timeout=timeout))
    if checkpoint_every is not None and not _seconds(checkpoint_every):
        problems.append(make_problem("invalid_checkpoint_every", "checkpoint_every", checkpoint_every=checkpoint_every))
    if problems:
        raise InvalidCallError(problems)


def _seconds(value: float) -> bool:
    return math.isfinite(value) and value > 0


# -- values -------------------------------------------------------------------------------------


def kept_outputs(outputs: Mapping[str, Field], spread: dict[str, Any]) -> dict[str, Any]:
    """A node's checked outputs by name as a run keeps them: a ``Series`` as a list."""
    for name, spec in outputs.items():
        if spec.many and spread[name] is not SKIPPED and spread[name] is not None:
            spread[name] = list(spread[name])  # an item may be SKIPPED: a branch not taken for it
    return spread


def checked_outputs(returns: Returns, outputs: Mapping[str, Field], returned: Any) -> Any:
    """What a node returned, as its outputs by name as a run keeps them; ``SKIPPED`` for a whole branch not taken.

    Each is read strictly by its declared type (a value already of its class as it is, a model from a dict, never
    ``"12"`` for an ``int``), and of an ``exclusive_group`` at most one output holds a value.

    Args:
        returns: How the node's function returns its outputs.
        outputs: The node's outputs, by name.
        returned: What the function returned.

    Raises:
        FieldValueError: An output is not of its type, or two of one group hold a value; its code is
            ``invalid_output``.
    """
    if returned is SKIPPED:
        return SKIPPED
    spread, settled = returns.spread(returned, outputs)
    if settled:
        return spread
    groups: dict[str, str] = {}
    for name, spec in outputs.items():
        spread[name] = _checked(spec, name, spread[name], code="invalid_output")
        group = spec.param.exclusive_group
        if group is not None and spread[name] is not SKIPPED:
            if group in groups:
                raise FieldValueError(f"'{groups[group]}' and '{name}' are alternatives of '{group}', and both hold a value", code="invalid_output", name=name)
            groups[group] = name
    return kept_outputs(outputs, spread)


def _checked(spec: Field, name: str, value: Any, *, code: str) -> Any:
    if value is SKIPPED:
        return value
    try:
        read = spec.whole.read(value, strict=code == "invalid_output")
    except FieldValueError as refused:
        refused.code, refused.ctx = code, {"name": name, **refused.ctx}
        raise
    exact = spec.whole.exact_classes
    if code == "invalid_output" and exact and not isinstance(read, exact):
        raise FieldValueError(f"its type's own check gave a {type(read).__name__}, not a {spec.whole.id}", code="invalid_output", name=name)
    return read


def _length(value: Any) -> int:
    """How many items a ``Series`` output holds; none when it is blank or a branch not taken."""
    return 0 if value is SKIPPED or value is None else len(value)


#: The state a save writes for each kind of unit.
_STATE: Mapping[type, Literal["done", "skipped", "blank", "waiting"]] = {Done: "done", Answered: "done", Skipped: "skipped", Blank: "blank", Asking: "waiting"}


def _dump_units(placed: PlacedNode, units: Mapping[Key, Unit]) -> _SavedUnits:
    """A node's units as columns: their places, their states, and one list per output."""
    return _SavedUnits.model_construct(keys=list(units), **_unit_columns(placed, list(units.values())))


def _dump_earlier(placed: PlacedNode, earlier: Earlier) -> _SavedEarlier:
    """What a node may take from an earlier run, as columns: each unit once, with what it received."""
    listed: list[Unit] = []
    places: dict[int, Key] = {}
    received: list[str] = []
    lineages: dict[int, str] = {}
    held = {id(done): key for key, (_, done) in earlier.at.items()}
    for fingerprint, done in earlier.computed.items():
        if id(done) in held:
            places[len(listed)] = held.pop(id(done))
        listed.append(done)
        received.append(fingerprint)
    for key, (fingerprint, done) in earlier.at.items():
        if id(done) in held:  # received the same as an earlier unit of the node, which holds the place in ``computed``
            places[len(listed)] = key
            listed.append(done)
            received.append(fingerprint)
    for (fingerprint, lineage), asked in earlier.asked.items():
        for unit in asked:
            lineages[len(listed)] = lineage
            listed.append(unit)
            received.append(fingerprint)
    return _SavedEarlier.model_construct(fingerprints=received, places=places, lineages=lineages, **_unit_columns(placed, listed))


def _unit_columns(placed: PlacedNode, listed: list[Unit]) -> dict[str, Any]:
    """Units as the columns a save writes: states, questions, blank inputs, and each output across the done ones."""
    done = [u for u in listed if isinstance(u, Done)]
    outputs: dict[str, _SavedOutput] = {}
    for name, spec in placed.outputs.items():
        values = [u.out[name] for u in done]
        at = f"{placed.id}.{name}"
        plain = [None if v is SKIPPED or v is None else [None if i is SKIPPED else i for i in v] if spec.many else v for v in values]
        dumped = _written_as_json(spec.column.adapter, plain, at)
        non_finite: list[_Path] = []
        if not spec.type.hashes_plainly:
            _find_non_finite(dumped, [], non_finite)
        items = {n: skips for n, v in enumerate(values) if spec.many and v is not SKIPPED and v is not None and (skips := [i for i, item in enumerate(v) if item is SKIPPED])}
        outputs[name] = _SavedOutput.model_construct(values=dumped, skipped=[i for i, v in enumerate(values) if v is SKIPPED], skipped_items=items, non_finite=non_finite)
    return dict(
        states=[_STATE[type(u)] for u in listed], outputs=outputs,
        questions={i: u.question for i, u in enumerate(listed) if isinstance(u, (Answered, Asking))},
        blank_inputs={i: u.input for i, u in enumerate(listed) if isinstance(u, Blank)},
    )


#: A place in a saved value: a key of a mapping, or an index of a list.
_Path = list[str | int]

#: A long column of a save that the writer made of the right items already: read typed, written as it is, since checking
#: each item against its type on the way out costs as much as writing it.
_AS_IS = PlainSerializer(lambda column: column, return_type=Any)

#: The places of a value's floats that aren't finite.
_Paths = Annotated[list[_Path], _AS_IS]


#: Every part of a save refuses what it doesn't declare, and writes a float that isn't finite as its text.
_SAVE_CONFIG = ConfigDict(extra="forbid", ser_json_inf_nan="strings")


class _SavedOutput(BaseModel):
    """One output of a node across its finished units, as a column.

    Attributes:
        values: The output's value on each finished unit, in the units' order, as JSON; ``None`` where it was blank
            or a branch not taken.
        skipped: The places in ``values`` whose output was a branch not taken (``SKIPPED``), not a blank.
        skipped_items: For a ``Series`` output, by place in ``values``, the items that were a branch not taken.
        non_finite: The places, from the place in ``values`` down, of each float that isn't finite where the output's
            type wouldn't read its text back as a float (an ``Any``, a ``dict``).
    """

    model_config = _SAVE_CONFIG
    values: list[Any]
    skipped: list[NonNegativeInt] = []
    skipped_items: dict[NonNegativeInt, list[NonNegativeInt]] = {}
    non_finite: _Paths = []


class _SavedUnitColumns(BaseModel):
    """Units of one node as columns, one place per unit.

    Attributes:
        states: What each unit came to: ``done``, ``skipped``, ``blank`` or ``waiting``.
        questions: By place, the question a waiting unit asks, or a done one was answered on.
        blank_inputs: By place, for a blank unit, the input that got nothing.
        outputs: By output name, that output across the done units, in their order.
    """

    model_config = _SAVE_CONFIG
    states: Annotated[list[Literal["done", "skipped", "blank", "waiting"]], _AS_IS]
    questions: dict[NonNegativeInt, Question] = {}
    blank_inputs: dict[NonNegativeInt, str] = {}
    outputs: dict[str, _SavedOutput] = {}


class _SavedUnits(_SavedUnitColumns):
    """A node's units.

    Attributes:
        keys: Each unit's instance: the place of its item in each map the node runs in, those of the embedded
            graphs' nodes around it first.
    """

    keys: list[tuple[NonNegativeInt, ...]]


class _SavedEarlier(_SavedUnitColumns):
    """Work a run carried from the one before it: the units, and what each received.

    Attributes:
        fingerprints: By place, the fingerprint of what the unit received.
        places: By place, the instance a computed unit stood in in the earlier run, where one of this run that
            received the same takes it first.
        lineages: By place, for an answer or a question still waiting, the fingerprint of the items above it.
    """

    fingerprints: list[str]
    places: dict[NonNegativeInt, tuple[NonNegativeInt, ...]] = {}
    lineages: dict[NonNegativeInt, str] = {}


class _SavedRunBase(BaseModel):
    """What ``run.to_json()`` writes beside how the run stands, read once at ``load_run``: a malformed save is refused
    here, by where.

    Attributes:
        graph: The fingerprint of the graph that ran, so another graph's save is refused.
        max_items: The item ceiling the run goes on under.
        inputs: The caller's inputs by address, as JSON, each as its type writes it.
        non_finite_inputs: By address, the places of each float that isn't finite, as ``_SavedOutput.non_finite``.
        units: The units of each function node with any, by node: its id, after the ids of the embedded
            graphs' nodes it is inside, joined by ``/`` (``e/ask``). An embedded graph's node has none of its own.
        earlier: The carried work not yet taken, by node, named as in ``units``.
        code: By node, named as in ``units``, the ``reuse_digest`` of the code that made its saved units and its
            carried work, so units made by other code are never carried.
        complete: The nodes, named as in ``units``, every unit of which was finished when the last leg ended.
    """

    model_config = _SAVE_CONFIG
    graph: str
    max_items: PositiveInt | None
    inputs: dict[str, Any]
    non_finite_inputs: dict[str, _Paths] = {}
    units: dict[str, _SavedUnits]
    earlier: dict[str, _SavedEarlier] = {}
    code: dict[str, str] = {}
    complete: list[str] = []


class _SavedGoingRun(_SavedRunBase):
    """A saved run nothing ended early. ``cause`` is taken as ``null`` only, and never written.

    Attributes:
        status: How the run stood when it was saved: ``ready``, ``running``, ``waiting`` or ``complete``.
        cause: Always ``None``.
    """

    status: GoingStatus
    cause: None = None


class _SavedEndedRun(_SavedRunBase):
    """A saved run whose last leg ended early: its ``cause`` is required.

    Attributes:
        status: How the run ended: ``failed``, ``stopped``, ``cancelled`` or ``timed_out``.
        cause: The problem that ended it.
    """

    status: EndedStatus
    cause: Problem


#: A save, told apart by its ``status``: a status and a cause that disagree are refused at the parse.
_SavedRun = Annotated[_SavedGoingRun | _SavedEndedRun, PydanticField(discriminator="status")]
_SAVED: TypeAdapter[_SavedRun] = TypeAdapter(_SavedRun)


class _SaveError(Exception):
    """A run that can't be written to its save, or a save this graph can't read.

    Attributes:
        at: Where: an input's address, ``node.output`` or ``node.output.column``.
        reason: Why, quoting nothing of the value.
    """

    def __init__(self, at: str, reason: str) -> None:
        super().__init__(reason)
        self.at, self.reason = at, reason


def _load_earlier(placed: PlacedNode, dumped: _SavedEarlier) -> Earlier:
    """What a node may take from an earlier run, from its save."""
    if len(dumped.fingerprints) != len(dumped.states):
        raise _SaveError(placed.id, f"it holds {len(dumped.fingerprints)} fingerprints for {len(dumped.states)} units")
    earlier = Earlier()
    for place, (fingerprint, unit) in enumerate(zip(dumped.fingerprints, _load_units(placed, dumped), strict=True)):
        match unit:
            case Answered() | Asking():
                earlier.asked.setdefault((fingerprint, dumped.lineages.get(place, "")), []).append(unit)
            case Done():
                earlier.computed.setdefault(fingerprint, unit)
                if place in dumped.places:
                    earlier.at[dumped.places[place]] = (fingerprint, unit)
            case Skipped() | Blank():
                raise _SaveError(placed.id, "it carries a unit that was never called")
    return earlier


def _load_units(placed: PlacedNode, dumped: _SavedUnitColumns) -> list[Unit]:
    """A node's saved units, in their saved order, each output read by its own type.

    Raises:
        _SaveError: Something doesn't fit the node, naming where.
    """
    states = dumped.states
    done = states.count("done")
    columns: dict[str, list[Any]] = {}
    for name, spec in placed.outputs.items():
        at = f"{placed.id}.{name}"
        column = dumped.outputs.get(name)
        if column is None:
            if done:
                raise _SaveError(at, "the output is missing")
            continue
        values: list[Any]
        try:
            values = spec.column.adapter.validate_python(_with_non_finite(column.values, column.non_finite, at))
        except ValidationError as error:
            raise _SaveError(at, value_free_errors(error)[0]["msg"]) from None
        if len(values) != done:
            raise _SaveError(at, f"it holds {len(values)} values for {done} finished units")
        for i in column.skipped:
            if i >= done:
                raise _SaveError(at, f"it skips the value {i} of {done}")
            values[i] = SKIPPED
        for n, positions in column.skipped_items.items():
            if n >= done or not isinstance(values[n], list) or any(i >= len(values[n]) for i in positions):
                raise _SaveError(at, f"it skips an item the value {n} doesn't hold")
            for i in positions:
                values[n][i] = SKIPPED
        if spec.many and not spec.whole.optional and any(v is None for v in values):
            raise _SaveError(at, "a finished unit has no value on an output of many")
        columns[name] = values
    unknown = sorted(dumped.outputs.keys() - placed.outputs.keys())
    if unknown:
        raise _SaveError(f"{placed.id}.{unknown[0]}", "the node has no such output")
    loaded: list[Unit] = []
    n = 0
    for i, state in enumerate(states):
        question = dumped.questions.get(i)
        match state:
            case "done":
                out = {name: values[n] for name, values in columns.items()}
                n += 1
                loaded.append(Done(out) if question is None else Answered(out, question))
            case "waiting":
                if question is None:
                    raise _SaveError(placed.id, f"the waiting unit {i} holds no question")
                loaded.append(Asking(question))
            case "blank":
                if i not in dumped.blank_inputs:
                    raise _SaveError(placed.id, f"the blank unit {i} names no input")
                loaded.append(Blank(dumped.blank_inputs[i]))
            case "skipped":
                loaded.append(SKIPPED_UNIT)
    return loaded


def _find_non_finite(value: Any, path: list[Any], found: list[list[Any]]) -> None:
    """The place of every float that isn't finite in a dumped value its type doesn't read back as a float, such as a ``dict``'s."""
    if type(value) is float:
        if not math.isfinite(value):
            found.append(path)
    elif type(value) is dict:
        for k, v in value.items():
            _find_non_finite(v, [*path, k], found)
    elif type(value) is list:
        for i, v in enumerate(value):
            _find_non_finite(v, [*path, i], found)


def _with_non_finite(written: Any, paths: Any, at: str) -> Any:
    """A saved value with each float a save marked read back from the text it was written as, before its type reads it.

    The marks are places in the JSON, so only its mappings and lists are walked. The value is the save's own, read
    from its text, so it is changed in place.
    """
    root: list[Any] = [written]
    try:
        for path in paths:
            holder: Any = root
            steps: list[Any] = [0, *path]
            for step in steps[:-1]:
                holder = holder[step]
            last = steps[-1]
            if holder[last] not in _NON_FINITE_TEXT:
                raise LookupError(last)
            holder[last] = float(holder[last])  # a text holder can't take it: TypeError
    except (LookupError, TypeError):
        raise _SaveError(at, "it marks a float where the value has none") from None
    return root[0]


#: How a save writes a float that isn't finite.
_NON_FINITE_TEXT = frozenset({"NaN", "Infinity", "-Infinity"})


def _written_as_json(adapter: TypeAdapter[Any], values: list[Any], at: str) -> Any:
    """``values`` as JSON; a value with no JSON form raises ``_SaveError`` naming where, never quoting it."""
    try:
        written = adapter.dump_python(values, mode="json")
    except Exception as error:  # noqa: BLE001  pydantic's and the codec's errors both quote the value
        raise _SaveError(at, f"a value has no JSON form ({type(error).__name__})") from None
    return written


def _dump_input(graph: Graph, address: str, value: Any) -> Any:
    """A caller's input as JSON, as its field writes it.

    Raises:
        _SaveError: The value has no JSON form; it names the input.
    """
    return _written_as_json(graph._input_fields[address].spec.column.adapter, [value], address)[0]  # as a column writes it: a float that isn't finite is kept


def _load_input(graph: Graph, address: str, saved: Any) -> Any:
    """A caller's input from a save, read by its field."""
    spec = graph._input_fields[address].spec
    try:
        return spec.whole.adapter.validate_python(saved)
    except ValidationError as error:
        raise _SaveError(address, value_free_errors(error)[0]["msg"]) from None


def jsonable(kwargs: dict[str, Any]) -> Any:
    """What a unit received, as JSON; a float that isn't finite as text. ``FieldValueError`` as ``question_unwritable`` names
    the first input with no JSON form."""
    try:
        return pydantic_core.from_json(pydantic_core.to_json(kwargs, fallback=written_table, inf_nan_mode="strings"))
    except (TypeError, ValueError, pydantic_core.PydanticSerializationError) as error:
        for name, value in kwargs.items():
            try:
                pydantic_core.to_json(value, fallback=written_table, inf_nan_mode="strings")
            except (TypeError, ValueError, pydantic_core.PydanticSerializationError) as unwritable:
                raise FieldValueError(str(unwritable), code="question_unwritable", input=name) from None
        raise error


def _hand_over(loop: asyncio.AbstractEventLoop, events: asyncio.Queue[Any], event: Any) -> None:
    """Put an event on the listener's queue from the leg's thread; a listener whose loop has closed is gone, and hears nothing."""
    try:
        loop.call_soon_threadsafe(events.put_nowait, event)
    except RuntimeError:
        pass


def _fingerprint(digest: str, kwargs: dict[str, Any], *, plain: bool = False) -> str | None:
    """What a unit of a node (its ``reuse_digest``) received, as a short hash; ``None`` for a value with no JSON form, which is never reused.

    ``plain`` says no input holds a mapping, so no key came from a store: the inputs' own order is the node's.
    """
    try:
        # No key of a plain node's inputs came from a store, so they need no canonical form: its inputs' order is the node's.
        written = pydantic_core.to_json([digest, kwargs], fallback=written_table) if plain else canonical_json([digest, kwargs], fallback=written_table)
    except (TypeError, ValueError, pydantic_core.PydanticSerializationError):
        return None
    return hashlib.blake2b(written, digest_size=12).hexdigest()


def canonical_json(value: Any, fallback: Callable[[Any], Any] | None = None) -> bytes:
    """``value`` as JSON to hash, with every object's keys sorted, so a value a JSON store gave back in another key order is written alike.

    Every hash whose keys may come back from a store is written here. A hash of values that hold no mapping (a reuse
    fingerprint of a node whose inputs can't, a question whose inputs are plain) is their plain JSON, hashed as it
    is: there are no keys to sort, and those hashes are made on every call.
    """
    plain = pydantic_core.to_jsonable_python(value, fallback=fallback)
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _question_fingerprint(placed: PlacedFunction, unit: Asking) -> str:
    """Eight hex digits of which node asks, as which code, and what it received, so a waiting id stops matching when its place holds another question."""
    inputs = unit.question.inputs
    values = [inputs[name] for name in sorted(inputs)]  # saved, so the names may come back from a store in any order
    hashed = [placed.id, placed.reuse_digest, values]
    written = canonical_json(hashed) if {type(v) for v in values} & {dict, list} else pydantic_core.to_json(hashed)  # plain values have no keys to sort
    return hashlib.blake2b(written, digest_size=4).hexdigest()


def _parse_place(place: str) -> tuple[Path, NodePath, Key]:
    """The place a waiting id names, ``e@0/ask@2``: as a path, then the ids of its nodes and the places of its
    items apart, which name the node and its instance.

    Raises:
        ValueError: A place in it isn't a number.
    """
    loc: list[str | int] = []
    nodes: list[str] = []
    key: list[int] = []
    for step in place.split("/"):
        node_id, *places = step.split("@")
        nodes.append(node_id)
        key += [int(i) for i in places]
        loc += [node_id, *key[len(key) - len(places):]]
    return tuple(loc), tuple(nodes), tuple(key)


def _deadline(timeout: float | None) -> Callable[[], EndedEarly | None] | None:
    if timeout is None:
        return None
    end = time.monotonic() + timeout
    return lambda: EndedEarly("timed_out", make_problem("run_timed_out", timeout=timeout)) if time.monotonic() > end else None

