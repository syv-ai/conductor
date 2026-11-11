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
from collections.abc import AsyncGenerator, Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, NamedTuple, get_args, overload

import pydantic_core
from pydantic import ConfigDict, NonNegativeInt, PositiveInt, TypeAdapter, ValidationError
from pydantic import Field as PydanticField
from pydantic_core import (
    CoreConfig,
    CoreSchema,
    PydanticCustomError,
    SchemaSerializer,
    SchemaValidator,
    core_schema,
)

from conductor._internal.ask import Question, Waiting, answer_outputs, waiting_questions
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.fields import SKIPPED, Field, FieldValueError, Records, Type, written_table
from conductor._internal.graph import (
    Cell,
    CellRead,
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
from conductor._internal.outside_the_items import COULD, could_bring
from conductor._internal.problem import InvalidCallError, Problem, make_problem, value_free_errors

if TYPE_CHECKING:
    from conductor._internal.graph import Graph
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

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")  # a save that gives it a cause is refused
    status: GoingStatus


@dataclass(frozen=True, slots=True)
class EndedEarly:
    """A run whose last leg ended before it was done, and why; ``resume`` goes on from where it stopped.

    Attributes:
        status: ``failed`` (a node failed), ``stopped`` (a Series outgrew ``max_items``), ``cancelled`` or
            ``timed_out``.
        cause: Why, as a problem with its place and code.
    """

    __pydantic_config__: ClassVar[ConfigDict] = ConfigDict(extra="forbid")
    status: EndedStatus
    cause: Problem


#: How a run stands: going on, or ended early with a cause.
RunState = Going | EndedEarly

_READY, _RUNNING, _WAITING, _COMPLETE = Going("ready"), Going("running"), Going("waiting"), Going("complete")


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
        out: Its outputs by name: a value, ``None`` for a blank, ``SKIPPED`` for a branch not taken (every one
            of them, when the node returned ``SKIPPED``), and for a ``Series`` a list, or ``Records`` when the
            node returned a table for a ``Series`` of records.
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
    """A unit whose node wasn't called because a branch above it wasn't taken: an input came only from branches
    not taken. What reads it is skipped too."""


@dataclass(frozen=True, slots=True)
class Blank:
    """A unit whose node wasn't called because an input that takes no blank got one. Its outputs are blank,
    except those of an exclusive group, which are a branch not taken: no branch was chosen.

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

#: The one skipped unit: it holds nothing, so every node not called for a branch not taken shares it.
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
    #: Each node's units by instance, in instance order, by the node's id, for each graph of the tree by the path of
    #: the node that embeds it: ``_units[()]["w"]`` for a node of the graph itself, ``_units[("e",)]["ask"]`` for one
    #: inside ``e``. Only a leg records out of that order, and it puts the units back in it before it hands the run on.
    _units: dict[NodePath, _UnitsOfAGraph]
    #: What each node may take from the run this one went on from.
    _earlier: dict[NodePath, Earlier]
    #: For a node whose units a saved run made with other code or another signature, what that code was; they are never carried.
    _ran_as: dict[NodePath, str]
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
            answered[(node, key)] = Answered(read, found.question) if not faults else found
        if problems:
            raise InvalidCallError(problems)
        run = self._copy(share_units=True)
        around: dict[GraphAtPath, dict[Key, None]] = {}  # each embedded graph's node around an answered unit, and its instances there
        for node, key in answered:
            embedding = node.embedding
            while embedding is not None:
                around.setdefault(embedding, {})[key[: len(embedding.maps)]] = None
                embedding = embedding.embedding
        changes: list[NodeAtPath] = [*{node for node, _ in answered}, *around]
        for changed in changes:  # its own units for each node it changes; every other node's are shared
            run._units[changed.inside][changed.placed.id] = dict(self._units[changed.inside][changed.placed.id])
        for (node, key), now in answered.items():
            run._units[node.inside][node.placed.id][key] = now
        for embedding in sorted(around, key=lambda node: node.position):  # each once, after every embedded graph's node inside it
            run._settle(embedding, around[embedding])
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
            for key, unit in self._units[node.inside][placed.id].items():
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
                for key, unit in self._units[node.inside][node.placed.id].items():
                    if isinstance(unit, Asking):
                        yield node, key, unit

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
            node = nodes[(address.partition(".")[0],)]
            output = node.placed.outputs[address.partition(".")[2]]
            try:
                written[address] = (output.column if node.maps else output.slot).to_json(value)
            except pydantic_core.PydanticSerializationError:
                raise InvalidCallError([make_problem("state_unwritable", "outputs", address, at=address, reason="a value has no JSON form")]) from None
        return pydantic_core.to_json(written).decode()

    def _collect(self, node: NodeAtPath, output: str) -> Any:
        """One output of a node over its units, in instance order."""
        units = self._units[node.inside][node.placed.id]
        values = []
        for key in self._instances(node.maps, ())[0]:
            unit = units.get(key)
            value = unit.out[output] if isinstance(unit, Done) else None
            if type(value) is list:
                value = [None if item is SKIPPED else item for item in value]
            elif type(value) is Records:
                value = value.records()
            values.append(None if value is SKIPPED else value)
        return values if node.maps else values[0]

    def _offered(self, node: GraphAtPath, key: Key) -> Any:
        """What the embedded graph's node ``node`` offers in the instance ``key``: each output of its inner graph
        (``graph.outputs``), by address, as a unit keeps it.

        An output made once inside is the inner unit's own: its value, ``SKIPPED`` for a branch not taken, ``None``
        for a blank, so what reads the embedded graph's node is skipped or left uncalled as it would be reading
        the node inside. One made per item inside, at any depth, is every item's in order, and a ``Series`` made
        per item is a list of lists; where the Series those items are of was never made, for a blank or a branch
        not taken, it is that blank or that branch not taken, as the nodes drawn in the graph itself hold.

        Returns:
            The outputs by address, or ``PENDING`` until every one of them is settled for the instance, and
            every node inside that ends the graph with no output.
        """
        units, base = self._units[node.path], len(key)
        out: dict[str, Any] = {}
        for address, (cell, maps, lists, series) in node.offers.items():
            if len(maps) == base:  # made once inside: the inner unit's own
                unit = units[cell.node_id].get(key)
                kind = type(unit)
                if kind is Done or kind is Answered:
                    value = unit.out[cell.output]  # type: ignore[union-attr]
                elif kind is Skipped or kind is Blank:
                    value = SKIPPED if kind is Skipped or cell.exclusive else None
                else:
                    return PENDING
            else:
                made = _cell(units, base, series, key)  # type: ignore[arg-type]  # an output made per item has its Series
                if made is PENDING:
                    return PENDING
                if made is None or made is SKIPPED:  # the Series was never made, so nor was the output: no list of no items
                    out[address] = made
                    continue
                instances, known = self._instances(maps, key)
                if not known:
                    return PENDING
                value = [_cell(units, len(maps), cell, instance) for instance in instances]
                for one in value:
                    if one is PENDING:
                        return PENDING
                if lists:
                    value = [[None if item is SKIPPED else item for item in one] if type(one) is list else one.records() if type(one) is Records else one for one in value]
            out[address] = value
        for last in node.ends_silently:  # nothing to offer, and still what the graph inside came to
            instances, known = self._instances(last.maps, key)
            if not known:
                return PENDING
            held = units[last.placed.id]
            for instance in instances:
                unit = held.get(instance)
                if unit is None or type(unit) is Asking:
                    return PENDING
        return out

    def _settle(self, node: GraphAtPath, instances: Iterable[Key]) -> None:
        """Gives the embedded graph's node ``node`` its unit in each of ``instances`` where every output of the
        graph inside is settled.

        The unit is what the units inside came to and nothing else (``_offered``), so it is made of them and
        never saved. One an instance already holds is made again, to the same; the node's units stay in
        instance order.

        Args:
            node: The embedded graph's node.
            instances: The instances of it to look at.
        """
        units = self._units[node.inside][node.placed.id]
        last = next(reversed(units), None)
        in_order = True
        for key in instances:
            offered = self._offered(node, key)
            if offered is not PENDING:
                if last is None or key > last:
                    last = key
                elif key not in units:
                    in_order = False
                units[key] = Done(offered)
        if not in_order:  # a run holds every node's units in instance order
            held = sorted(units.items())
            units.clear()
            units.update(held)

    def _settle_embedded(self) -> None:
        """Settles every embedded graph's node in every instance that holds no unit (``_settle``), each after the
        nodes inside it."""
        for node in self.graph._tree:
            if type(node) is GraphAtPath:
                units = self._units[node.inside][node.placed.id]
                self._settle(node, [key for key in self._instances(node.maps, ())[0] if key not in units])

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

    def _has_every_unit(self, node: NodeAtPath) -> bool:
        """Whether every instance of ``node`` holds a unit, one waiting for a person included, and every Series
        it runs per item of is made, so no instance is still to come."""
        units = self._units[node.inside][node.placed.id]
        if not node.maps:
            return bool(units)
        counted, known = self._instances(node.maps, (), counted=True)
        return known and len(units) == (counted[0][0] if counted else 0)

    def _inputs(self, node: FunctionAtPath, key: Key) -> Any:
        """What ``node`` receives in the instance ``key``.

        An open input of a node inside an embedded graph is read as the embedded graph's node is given it: by
        that node's plan, in the graph that node is in.

        Returns:
            The keyword arguments of the call; or the unit it comes to uncalled: ``Skipped`` when an input comes
            only from branches not taken, whatever the others hold; else ``PENDING`` while something it reads
            hasn't run; else ``Blank`` when an input that takes no blank got one. Every input is read before
            pending or a blank is the answer, so the order of a node's parameters decides nothing. When an
            edge names a field the record it brings doesn't have, the ``Problem`` the unit fails with
            (``invalid_edge_value``).
        """
        kwargs: dict[str, Any] = {}
        blank: Blank | None = None
        pending = False
        gathered: dict[str, tuple[Gather, _UnitsOfAGraph, CellRead, list[Any]]] = {}
        own = self._units[node.inside]
        for name, (plan, giver, cells) in node.inputs.items():
            kind = type(plan)
            if kind is Fixed:
                kwargs[name] = plan.value  # type: ignore[union-attr]
                continue
            if kind is Open:  # a first node of the graph itself, or of a graph embedded in one: the caller gives it
                kwargs[name] = self.inputs[plan.address]  # type: ignore[union-attr]
                continue
            units = own if giver is None else self._units[giver.inside]
            try:
                if kind is Gather:
                    value = self._gathered(node.inside if giver is None else giver.inside, plan, cells, node.gathered_through[name], key)  # type: ignore[arg-type]
                else:  # one value: the first edge whose branch was taken
                    value = SKIPPED
                    for source, depth in cells:
                        value = _cell(units, depth, source, key)
                        if value is not SKIPPED:
                            break
                    if kind is SeriesOfOne and value is not SKIPPED and value is not PENDING and (value is not None or plan.spec.keeps_blanks):
                        value = [value]  # a Series of one, whose one item is blank where the items take a blank
            except FieldValueError as missing:  # an edge into this input names a field its record doesn't have
                return make_problem(missing.code, *unit_loc(node, key), name=name, reason=missing.reason, **missing.ctx)
            if value is SKIPPED:
                return SKIPPED_UNIT
            if value is PENDING:  # a later input may be skipped, which settles the unit whatever this one comes to
                pending = True
            elif value is None:
                if not plan.spec.whole.optional and blank is None:
                    blank = Blank(name)
                kwargs[name] = None
            elif kind is Gather:
                gathered[name] = (plan, units, cells[0], value)  # type: ignore[assignment]
            else:
                kwargs[name] = value
        if pending:
            return PENDING
        if blank is not None:
            return blank
        for name, (plan, units, series, items) in gathered.items():
            kwargs[name] = _aligned(plan, units, series, items, gathered)
        return kwargs

    def _gathered(self, inside: NodePath, plan: Gather, cells: tuple[CellRead, ...], through: tuple[NodeAtPath, ...], key: Key) -> Any:
        """The items a gathering input gets in the instance ``key``, with ``None`` for an item that has no value.

        What is outside the items is asked first whether any item could bring a value (``could_bring``). When
        none could, that is the answer for any number of items, none included. When one could, each item
        answers for itself: one with no value, for a blank or a branch not taken, holds ``None``.

        Args:
            inside: The path of the embedded graph's node the plan was made in the graph of; ``()`` for the
                run's own graph.
            plan: The input's plan.
            cells: The cells the input reads where its node stands: the Series, then what each edge brings.
            through: The nodes the input gathers through, as the tree keeps them; none when nothing outside
                the items but the Series itself has a say.
            key: The instance of the node that gathers.

        Returns:
            The list, one place per item of the Series; ``PENDING`` until what is outside and every item is
            settled; ``SKIPPED`` when no item could bring a value for a branch not taken outside the items;
            ``None`` when none could for a blank there.
        """
        (series_made, gathered_in), *edges = cells
        key = key[:gathered_in]
        units = self._units[inside]
        series = _cell(units, gathered_in, series_made, key)
        if through:
            certain = could_bring(plan, cells, through, inside, self.graph._node_at, _AsItStands(self, key))
            if certain is not COULD:
                return certain
        elif series is PENDING or series is SKIPPED or series is None:  # nothing outside but the Series has a say
            return series
        if len(edges) == 1 and type(edges[0][0]) is Item:  # the Series itself: its items, or one field of each
            wanted = edges[0][0].field
            if type(series) is Records:
                return series.records() if wanted is None else _field_of(series.columns, wanted)
            if wanted is None:
                return [None if item is SKIPPED else item for item in series] if any(item is SKIPPED for item in series) else series
            return [None if item is SKIPPED or item is None else _field_of(item, wanted) for item in series]
        items = []
        per_item = [(source, gathered_in if type(source) is Item else gathered_in + 1) for source, _ in edges]  # an item of the Series' unit, or a unit made for the item
        for place in range(len(series)):
            value = SKIPPED
            for source, depth in per_item:
                value = _cell(units, depth, source, (*key, place))
                if value is PENDING:
                    return PENDING
                if value is not SKIPPED:
                    break
            items.append(None if value is SKIPPED else value)
        return items

    # -- JSON -------------------------------------------------------------------

    def to_json(self) -> str:
        """The run as JSON: its inputs, each node's units as a column per output, and what it went on from."""
        saved = self._saved()
        try:
            return self.graph._save_form.serializer.to_json(saved, exclude_defaults=True).decode()
        except pydantic_core.PydanticSerializationError:
            at = _unwritable_at(self.graph, saved)
            raise InvalidCallError([make_problem("state_unwritable", "state", at, at=at, reason="a value has no JSON form")]) from None

    def _saved(self) -> SavedRun[SavedUnits, SavedEarlier[SavedUnits]]:
        """The run as its graph's ``SaveForm`` writes it. A node inside an embedded graph is saved as any node,
        under its path. The unit of an embedded graph's node is not written: it is what the units inside came
        to, and is read from them."""
        functions = self.graph._functions
        units = {
            "/".join(node.path): _unit_columns(node.placed.outputs, self._slots(node))
            for node in self.graph._tree if isinstance(node, FunctionAtPath) and self._units[node.inside][node.placed.id]
        }
        earlier = {"/".join(path): _earlier_columns(functions[path].outputs, lent) for path, lent in self._earlier.items() if lent.computed or lent.asked}
        code = {name: self._ran_as.get(path) or placed.reuse_digest for path, placed in functions.items() if (name := "/".join(path)) in units or name in earlier}
        complete = ["/".join(node.path) for node in self.graph._tree if node.path in self._complete]
        return SavedRun(graph=self.graph.fingerprint, state=self.state, max_items=self.max_items, inputs=self.inputs, units=units, earlier=earlier, code=code, complete=complete)

    def _slots(self, node: FunctionAtPath) -> list[Unit | None]:
        """A node's units with a slot for each instance of the node, in instance order; ``None`` where the
        instance hasn't run."""
        units = self._units[node.inside][node.placed.id]
        if self._has_every_unit(node):  # the units as they stand
            return list(units.values())
        return [units.get(instance) for instance in self._instances(node.maps, ())[0]]

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
        copy.state = self.state
        copy._ran_as, copy._complete = self._ran_as, self._complete
        return copy


class _AsItStands:
    """What was made outside the items of a gathering input, as it stands for the instance the input gathers in:
    the reader a run hands the question of what is outside the items.

    Attributes:
        run: The run.
        key: The instance the input gathers in.
    """

    __slots__ = ("key", "run")

    def __init__(self, run: Run, key: Key) -> None:
        self.run, self.key = run, key

    def cell(self, inside: NodePath, depth: int, cell: Cell) -> Any:
        """What ``cell`` holds there: ``COULD`` for a value, else ``SKIPPED``, ``None`` or ``PENDING``."""
        value = _cell(self.run._units[inside], depth, cell, self.key)
        return value if value is PENDING or value is SKIPPED or value is None else COULD

    def gathered(self, inside: NodePath, plan: Gather, cells: tuple[CellRead, ...], through: tuple[NodeAtPath, ...]) -> Any:
        """What the gathering input of ``plan`` gathers there: ``COULD`` for a list, else ``SKIPPED``, ``None``
        or ``PENDING``."""
        value = self.run._gathered(inside, plan, cells, through, self.key)
        return value if value is PENDING or value is SKIPPED or value is None else COULD


def _cell(units: _UnitsOfAGraph, depth: int, cell: Cell, key: Key) -> Any:
    """What ``cell`` holds for the instance ``key`` or one under it, as an edge brings it.

    Args:
        units: The units of the graph the cell's node is in.
        depth: How many places of ``key`` say which unit of the cell's node holds it, as the reading node's
            ``inputs`` give it.
        cell: The cell.
        key: The instance of the node that reads.

    Returns:
        The value (an ``Item``'s is the item of ``key``, and a field named on the edge is read from it);
        ``PENDING`` while its unit hasn't run or waits for a person; ``SKIPPED`` for a branch not taken, which
        an output of an exclusive group is where its node wasn't called; ``None`` for a blank.
    """
    unit = units[cell.node_id].get(key[:depth])
    kind = type(unit)
    if kind is Done or kind is Answered:
        value = unit.out[cell.output]  # type: ignore[union-attr]
    elif kind is Skipped:
        return SKIPPED
    elif kind is Blank:  # not called: a blank, and for an output of an exclusive group no branch was chosen
        return SKIPPED if cell.exclusive else None
    else:
        return PENDING
    if value is None or value is SKIPPED or type(cell) is Cell:
        return value
    if type(cell) is Item:
        if type(value) is Records:  # held by column: the cell an edge names, with no record built
            wanted = cell.field  # type: ignore[attr-defined]
            return value[key[depth]] if wanted is None else _field_of(value.columns, wanted)[key[depth]]
        value = value[key[depth]]
    field = cell.field  # type: ignore[attr-defined]
    return value if field is None or value is None or value is SKIPPED else _field_of(value, field)


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


def _aligned(plan: Gather, units: _UnitsOfAGraph, series: CellRead, items: list[Any], gathered: Mapping[str, tuple[Gather, _UnitsOfAGraph, CellRead, list[Any]]]) -> list[Any]:
    """The items one gathering input receives: those that have a value in every input of the node that gathers
    the same Series, so those inputs stay in step; an input whose items may be blank keeps a blank in its place.

    Args:
        plan: The input's plan.
        units: The units of the graph the plan was made in, which with ``series`` says which Series it is.
        series: The Series the input gathers, as the node's ``inputs`` give it.
        items: What the input gathered, one place per item of the Series.
        gathered: Every gathering input of the node, by name: its plan, the units of its plan's graph, its
            Series, its items.
    """
    together = [(others, other.spec.keeps_blanks) for other, theirs, of, others in gathered.values() if of == series and theirs is units]
    if len(together) == 1:
        return items if together[0][1] else [item for item in items if item is not None]  # by identity: an item may be a frame, which has no plain ==
    return [item for place, item in enumerate(items) if all(keeps or others[place] is not None for others, keeps in together)]


def _field_of(value: Any, name: str) -> Any:
    """One field of a record: a mapping's key, else an attribute; of records held by column, that column.

    Raises:
        FieldValueError: The record has no such field (``invalid_edge_value``). Only a record whose type
            declares no fields can lack one: a field a type declares is checked when the graph loads.
    """
    try:
        return value[name] if type(value) is dict or isinstance(value, Mapping) else getattr(value, name)
    except (KeyError, AttributeError):
        raise _no_such_field(name) from None


def _no_such_field(name: str) -> FieldValueError:
    """The refusal of an edge that names the field ``name`` of a record that has none."""
    return FieldValueError(f"it has no field '{name}'", code="invalid_edge_value", field=name)


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
        raise InvalidCallError([make_problem("state_unreadable", "state", at="the run", reason="it is not JSON")]) from None
    except ValidationError as error:
        first = error.errors(include_url=False, include_input=False)[0]
        if first["type"] == "union_tag_invalid" and not first["loc"]:  # a save is read as one of this graph, which its fingerprint says it is
            raise InvalidCallError([make_problem("state_of_another_graph", "state", saved=first["ctx"]["tag"], graph=graph.fingerprint)]) from None
        at = ".".join(str(step) for step in first["loc"][1:]) or "the run"  # a place in the save comes after the fingerprint it was read under
        raise InvalidCallError([make_problem("state_unreadable", "state", at=at, reason=value_free_errors(error)[0]["msg"])]) from None
    return _run_of(graph, read)


def _run_of(graph: Graph, saved: SavedRun[list[Unit | None], Earlier]) -> Run:
    """A run of ``graph`` from its save as the graph's ``SaveForm`` read it: each node's units in the order of
    their slots, its carried work an ``Earlier``.

    Raises:
        InvalidCallError: A node doesn't hold one slot for each instance the saved run made of it
            (``state_unreadable``).
    """
    run = start_run(graph, saved.inputs, saved.max_items, {}, no_units(graph))
    run.state = saved.state
    for node in graph._tree:  # each after the nodes it reads, so the Series its instances are of are in place
        instances, _ = run._instances(node.maps, ())
        if isinstance(node, GraphAtPath):  # its units are what the units inside it came to
            run._settle(node, instances)
        elif (slots := saved.units.get("/".join(node.path))) is not None:
            if len(slots) != len(instances):
                at, reason = f"units.{'/'.join(node.path)}", f"it holds {len(slots)} slots for the {len(instances)} instances the saved run made of the node"
                raise InvalidCallError([make_problem("state_unreadable", "state", at=at, reason=reason)])
            run._units[node.inside][node.placed.id] = {instance: unit for instance, unit in zip(instances, slots, strict=True) if unit is not None}
    functions = graph._functions
    run._ran_as = {path: digest for name, digest in saved.code.items() if digest != functions[path := tuple(name.split("/"))].reuse_digest}
    run._earlier = {path: lent for name, lent in saved.earlier.items() if (path := tuple(name.split("/"))) not in run._ran_as}  # work of other code is never taken
    run._complete = frozenset(tuple(name.split("/")) for name in saved.complete)
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


def checked_outputs(returns: Returns, outputs: Mapping[str, Field], returned: Any) -> dict[str, Any]:
    """What a node returned, as its outputs by name as a run keeps them; ``SKIPPED`` returned for the whole node
    is ``SKIPPED`` on every output.

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
        return dict.fromkeys(outputs, SKIPPED)
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
    return spread


def _checked(spec: Field, name: str, value: Any, *, code: str) -> Any:
    if value is SKIPPED:
        return value
    returned = code == "invalid_output"
    try:
        read = spec.whole.read_returned(value) if returned else spec.whole.read(value)
    except FieldValueError as refused:
        refused.code, refused.ctx = code, {"name": name, **refused.ctx}
        raise
    exact = spec.whole.exact_classes
    if returned and exact and not isinstance(read, exact):
        raise FieldValueError(f"its type's own check gave a {type(read).__name__}, not a {spec.whole.id}", code="invalid_output", name=name)
    return read


def _length(value: Any) -> int:
    """How many items a ``Series`` output holds; none when it is blank or a branch not taken."""
    return 0 if value is SKIPPED or value is None else len(value)


# -- the save -----------------------------------------------------------------------------------


#: Every record of a save refuses a key it doesn't declare, and writes a float that isn't finite as its text.
_SAVED = ConfigDict(extra="forbid", ser_json_inf_nan="strings")

#: What a save says of a unit that holds outputs, and of one that is a branch not taken; a unit with neither
#: word holds no outputs.
UnitState = Literal["done", "skipped"]
_DONE, _SKIPPED = get_args(UnitState)


#: The slots of one output whose value is a branch not taken.
SkippedSlots = list[NonNegativeInt]

#: By slot, the items of a ``Series`` output there that are a branch not taken.
SkippedItems = dict[NonNegativeInt, list[NonNegativeInt]]


@dataclass(slots=True, kw_only=True)
class SavedUnits:
    """Units of one node as a save holds them: columns with a slot for each unit. A node's own units have a slot
    for each instance of the node, in instance order, so an instance is a place in them and is never written.

    The record is what is written and what is read: a key of a save is a field here and nowhere else.

    Attributes:
        states: What each unit came to; ``None`` for a unit that holds no outputs.
        outputs: By output, its value in each slot, read and written by the output's own type (the one part
            composed for each graph); ``None`` where the unit holds none.
        questions: By slot, the question a unit asked; ``None``, as for each field after it, when there is none,
            which leaves the key out of the save. On a slot that isn't done the unit waits for a person, so a
            waiting unit can't be written without its question.
        blank_inputs: By slot, the input that got a blank, for a unit that wasn't called for it.
        skipped: By output, the slots whose value is a branch not taken (``SKIPPED``); ``None`` in the column.
            A graph's form takes it for the node's own outputs only.
        skipped_items: By output and slot, the items of a ``Series`` there that are a branch not taken; ``None``
            in the column. A graph's form takes it for the node's ``Series`` outputs only, so an output of one
            value can't be saved with any.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _SAVED
    states: list[UnitState | None]
    outputs: dict[str, list[Any]]
    questions: dict[NonNegativeInt, Question] | None = None
    blank_inputs: dict[NonNegativeInt, str] | None = None
    skipped: dict[str, SkippedSlots] | None = None
    skipped_items: dict[str, SkippedItems] | None = None


@dataclass(slots=True, kw_only=True)
class SavedEarlier[UnitsT]:
    """The work of one node carried from an earlier run, as a save holds it: each carried unit once.

    Attributes:
        units: The carried units, a slot for each: a ``SavedUnits`` when written, the units it holds when read.
        fingerprints: For each slot, the fingerprint of what the unit received.
        places: By slot, the instance a computed unit stood in in the earlier run; ``None``, as for ``lineages``,
            when there is none, which leaves the key out of the save.
        lineages: By slot, for an answer or a question still waiting, the fingerprint of the items above it.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _SAVED
    units: UnitsT
    fingerprints: list[str]
    places: dict[NonNegativeInt, tuple[NonNegativeInt, ...]] | None = None
    lineages: dict[NonNegativeInt, str] | None = None


@dataclass(slots=True, kw_only=True)
class SavedRun[UnitsT, EarlierT]:
    """A run as a save holds it. A node is named by its id after the ids of the embedded graphs' nodes it is
    inside, joined by ``/`` (``e/ask``).

    Attributes:
        graph: The fingerprint of the run's graph, so a save of another graph is told apart before anything in
            it is read.
        state: How the run stands, with its cause when a leg ended early.
        max_items: The item ceiling the run goes on under.
        inputs: The caller's inputs by address, each read and written by its field's type.
        units: By node, the units of each function node that has any: a ``SavedUnits`` when written, the units
            it holds when read. An embedded graph's node has none of its own.
        earlier: By node, the work carried from an earlier run and not yet taken: a ``SavedEarlier`` when
            written, an ``Earlier`` when read.
        code: By node, the ``reuse_digest`` of the code that made its units and its carried work, so work made
            by other code is never taken.
        complete: The nodes every unit of which was finished when the last leg ended.
    """

    __pydantic_config__: ClassVar[ConfigDict] = _SAVED
    graph: str
    state: Annotated[Going | EndedEarly, PydanticField(discriminator="status")]
    max_items: PositiveInt | None
    inputs: dict[str, Any]
    units: dict[str, UnitsT]
    earlier: dict[str, EarlierT]
    code: dict[str, str]
    complete: list[str]


#: Each record's own schema, as pydantic makes it from the record: what a save's schema is composed from.
_UNITS, _EARLIER, _RUN = (TypeAdapter(record).core_schema for record in (SavedUnits, SavedEarlier, SavedRun))

#: The schemas of what ``SavedUnits`` holds for one output beside its column.
_SKIPPED_SLOTS, _SKIPPED_ITEMS = TypeAdapter(SkippedSlots).core_schema, TypeAdapter(SkippedItems).core_schema

#: The key of a save that names its graph, as ``SavedRun`` declares it.
_GRAPH = SavedRun.__dataclass_fields__["graph"].name  # a KeyError where this is read, should the field be renamed


class SaveForm(NamedTuple):
    """How the runs of one graph are read from JSON and written as it.

    Attributes:
        validator: Reads a save into the ``SavedRun`` that ``_run_of`` makes a run of.
        serializer: Writes the ``SavedRun`` that ``Run._saved`` gives.
    """

    validator: SchemaValidator
    serializer: SchemaSerializer


def save_form(graph: Graph) -> SaveForm:
    """Composes how the runs of ``graph`` are read from JSON and written as it.

    What a save holds is declared once, by ``SavedRun``, ``SavedUnits`` and ``SavedEarlier``, and their own
    schemas are what is written and read by. Only what depends on the graph is put in here: each input's and
    each output's own type, the nodes a save may name, and the graph's fingerprint. The parts are pydantic's own
    schemas of those types, put together as core schemas, since generating a model of a whole graph costs more
    than reading a small run. A graph's form is made the first time a run of it is saved or loaded.
    """
    shared: dict[str, CoreSchema] = {}
    units_record, earlier_record = _inlined(_UNITS, shared), _inlined(_EARLIER, shared)
    blocks: dict[tuple[tuple[str, Type, bool], ...], tuple[CoreSchema, CoreSchema]] = {}
    units, earlier = {}, {}
    for path, placed in graph._functions.items():
        columns = tuple((name, output.column, output.many) for name, output in placed.outputs.items())
        if columns not in blocks:  # nodes of the same outputs share one schema
            block = len(blocks)
            of_block = _with_fields(
                units_record,
                outputs=_closed({name: _inlined(column.saved_schema, shared) for name, column, _ in columns}),
                skipped=_closed({}, {name: _SKIPPED_SLOTS for name, _, _ in columns}),
                skipped_items=_closed({}, {name: _SKIPPED_ITEMS for name, _, many in columns if many}),
            )
            shared[f"units:{block}"] = core_schema.no_info_after_validator_function(_units_of, of_block, ref=f"units:{block}")
            carried = _with_fields(earlier_record, units=core_schema.definition_reference_schema(f"units:{block}"))
            shared[f"earlier:{block}"] = core_schema.no_info_after_validator_function(_earlier_of, carried, ref=f"earlier:{block}")
            blocks[columns] = core_schema.definition_reference_schema(f"units:{block}"), core_schema.definition_reference_schema(f"earlier:{block}")
        units["/".join(path)], earlier["/".join(path)] = blocks[columns]
    run = _with_fields(
        _inlined(_RUN, shared),
        graph=core_schema.literal_schema([graph.fingerprint]),
        inputs=_closed({address: _inlined(offered.spec.whole.saved_schema, shared) for address, offered in graph._input_fields.items()}),
        units=_closed({}, units),
        earlier=_closed({}, earlier),
        code=_closed({}, dict.fromkeys(units, core_schema.str_schema())),
        complete=core_schema.list_schema(core_schema.literal_schema(["/".join(node.path) for node in graph._tree])),
    )
    schema = core_schema.definitions_schema(core_schema.tagged_union_schema({graph.fingerprint: run}, discriminator=_GRAPH), list(shared.values()))
    config = CoreConfig(ser_json_inf_nan="strings")
    return SaveForm(SchemaValidator(schema, config), SchemaSerializer(schema, config))


def _with_fields(record: Any, **schemas: CoreSchema) -> CoreSchema:
    """Gives a record's schema with the schema of each named field replaced by the one composed for a graph.

    Args:
        record: The schema pydantic made of a record of a save.
        **schemas: The schema of a field, by the field's name.

    Raises:
        KeyError: The record declares no field of a name given.
    """
    fields = {declared["name"]: declared for declared in record["schema"]["fields"]}
    for name, schema in schemas.items():
        declared = fields[name]
        fields[name] = {**declared, "schema": {**declared["schema"], "schema": schema} if declared["schema"]["type"] == "default" else schema}
    composed = {**record, "schema": {**record["schema"], "fields": list(fields.values())}}
    del composed["ref"]  # pydantic's name for the record as declared, which this no longer is
    return composed  # type: ignore[return-value]  # a dataclass schema, as the record's was


def _closed(required: Mapping[str, CoreSchema], optional: Mapping[str, CoreSchema] = MappingProxyType({})) -> CoreSchema:
    """Gives the schema of an object of exactly these keys, each of ``required`` there: the names a graph gives,
    of its inputs, its nodes and a node's outputs."""
    # each field is written out: a graph makes three for every node, and pydantic's own helper costs more than the field
    fields: dict[str, Any] = {name: {"type": "typed-dict-field", "schema": schema} for name, schema in required.items()}
    fields.update((name, {"type": "typed-dict-field", "schema": schema, "required": False}) for name, schema in optional.items())
    return core_schema.typed_dict_schema(fields, extra_behavior="forbid")


def _inlined(schema: CoreSchema, shared: dict[str, CoreSchema]) -> CoreSchema:
    """Gives ``schema`` as it stands inside another: what pydantic defined beside it (a model met twice, one that
    holds itself) moves to ``shared``, since one validator takes each definition once."""
    if schema["type"] != "definitions":
        return schema
    shared.update((definition["ref"], definition) for definition in schema["definitions"])  # type: ignore[typeddict-item]  # every definition has its ref
    return schema["schema"]


#: The state a save writes for a unit that holds outputs or is a branch not taken; any other has none.
_STATE_WORD: Mapping[type, UnitState] = {Done: _DONE, Answered: _DONE, Skipped: _SKIPPED}

#: The kinds of unit that hold outputs.
_HOLDS_OUTPUTS = frozenset({Done, Answered})


def _unit_columns(outputs: Mapping[str, Field], slots: list[Unit | None]) -> SavedUnits:
    """Gives units as the columns a save holds of them.

    Args:
        outputs: The outputs of the units' node, by name.
        slots: The units, in the order of their slots; ``None`` for an instance that hasn't run.
    """
    kinds = set(map(type, slots))
    columns: dict[str, list[Any]] = {}
    skipped: dict[str, list[int]] = {}
    skipped_items: dict[str, dict[int, list[int]]] = {}
    questions: dict[int, Question] | None = None
    blank_inputs: dict[int, str] | None = None
    if kinds & _HOLDS_OUTPUTS:
        every = kinds <= _HOLDS_OUTPUTS  # every unit holds outputs, as those of a node that finished do
        for name, output in outputs.items():
            column = [unit.out[name] for unit in slots] if every else [unit.out[name] if type(unit) in _HOLDS_OUTPUTS else None for unit in slots]  # type: ignore[union-attr]
            columns[name] = column
            gone = [slot for slot, value in enumerate(column) if value is SKIPPED]
            if gone:
                skipped[name] = gone
                for slot in gone:
                    column[slot] = None
            if output.many:
                for slot, items in enumerate(column):
                    if type(items) is list and (gaps := [place for place, item in enumerate(items) if item is SKIPPED]):
                        skipped_items.setdefault(name, {})[slot] = gaps
                        column[slot] = [None if item is SKIPPED else item for item in items]
    else:  # no unit holds outputs yet
        columns = {name: [None] * len(slots) for name in outputs}
    if Answered in kinds or Asking in kinds:
        questions = {slot: unit.question for slot, unit in enumerate(slots) if type(unit) is Answered or type(unit) is Asking}  # type: ignore[union-attr]
    if Blank in kinds:
        blank_inputs = {slot: unit.input for slot, unit in enumerate(slots) if type(unit) is Blank}  # type: ignore[union-attr]
    return SavedUnits(
        states=list(map(_STATE_WORD.get, map(type, slots))), outputs=columns, questions=questions, blank_inputs=blank_inputs,
        skipped=skipped or None, skipped_items=skipped_items or None,
    )


def _units_of(saved: SavedUnits) -> list[Unit | None]:
    """Makes the units a node's saved columns hold, in the order of their slots; ``None`` for an instance that
    hasn't run.

    Raises:
        PydanticCustomError: An output doesn't hold one slot for each state (``slots_differ``); or a skip, a
            question or a blank input is placed on a slot that can't hold one (``misplaced``).
    """
    states, outputs = saved.states, saved.outputs
    if any(len(column) != len(states) for column in outputs.values()):
        raise PydanticCustomError("slots_differ", "an output doesn't hold one slot for each state")
    try:
        for name, slots in (saved.skipped or {}).items():
            for slot in slots:
                outputs[name][slot] = SKIPPED
        for name, by_slot in (saved.skipped_items or {}).items():
            for slot, places in by_slot.items():
                for place in places:
                    outputs[name][slot][place] = SKIPPED
        if len(outputs) == 1:  # most nodes have one output
            ((name, column),) = outputs.items()
            outs = [{name: value} for value in column]
        else:
            names = tuple(outputs)
            outs = [dict(zip(names, values, strict=True)) for values in zip(*outputs.values(), strict=True)] if outputs else [{} for _ in states]
        done, skipped = _DONE, _SKIPPED
        units: list[Unit | None] = [Done(out) if state == done else SKIPPED_UNIT if state == skipped else None for state, out in zip(states, outs, strict=True)]
        for slot, question in (saved.questions or {}).items():
            units[slot] = Asking(question) if states[slot] is None else Answered(units[slot].out, question)  # type: ignore[union-attr]
        for slot, blank in (saved.blank_inputs or {}).items():
            if units[slot] is not None:
                raise LookupError(slot)
            units[slot] = Blank(blank)
    except (LookupError, TypeError, AttributeError):  # a slot the columns don't have, or one that holds another kind of unit
        raise PydanticCustomError("misplaced", "a skip, a question or a blank input is placed on a slot that can't hold one") from None
    return units


def _earlier_columns(outputs: Mapping[str, Field], earlier: Earlier) -> SavedEarlier[SavedUnits]:
    """Gives what a node may take from an earlier run as a save holds it, each unit once: the computed units by
    the instance they stood in, then the ones only known by what they received, then the answers and the
    questions still waiting.

    Args:
        outputs: The node's outputs, by name.
        earlier: The carried work.
    """
    slots: list[Unit | None] = [done for _, done in earlier.at.values()]
    fingerprints = [fingerprint for fingerprint, _ in earlier.at.values()]
    places = dict(enumerate(earlier.at))
    stood = set(fingerprints)  # each of these is one of the units above: the first that received it
    for fingerprint, done in earlier.computed.items():
        if fingerprint not in stood:
            slots.append(done)
            fingerprints.append(fingerprint)
    lineages: dict[int, str] = {}
    for (fingerprint, lineage), asked in earlier.asked.items():
        for unit in asked:
            if lineage:
                lineages[len(slots)] = lineage
            slots.append(unit)
            fingerprints.append(fingerprint)
    return SavedEarlier(units=_unit_columns(outputs, slots), fingerprints=fingerprints, places=places or None, lineages=lineages or None)


def _earlier_of(saved: SavedEarlier[list[Unit | None]]) -> Earlier:
    """Makes the carried work a node's save holds.

    Raises:
        PydanticCustomError: It doesn't hold one fingerprint for each unit (``slots_differ``), or it carries a
            unit that was never called (``not_carried``).
    """
    if len(saved.fingerprints) != len(saved.units):
        raise PydanticCustomError("slots_differ", "it doesn't hold one fingerprint for each unit")
    places, lineages = saved.places or {}, saved.lineages or {}
    earlier = Earlier()
    for slot, (fingerprint, unit) in enumerate(zip(saved.fingerprints, saved.units, strict=True)):
        kind = type(unit)
        if kind is Done:
            earlier.computed.setdefault(fingerprint, unit)  # type: ignore[arg-type]
            if slot in places:
                earlier.at[places[slot]] = (fingerprint, unit)  # type: ignore[assignment]
        elif kind is Answered or kind is Asking:
            earlier.asked.setdefault((fingerprint, lineages.get(slot, "")), []).append(unit)  # type: ignore[arg-type]
        else:
            raise PydanticCustomError("not_carried", "it carries a unit that was never called")
    return earlier


def _unwritable_at(graph: Graph, saved: SavedRun[SavedUnits, SavedEarlier[SavedUnits]]) -> str:
    """Says where a run that pydantic refused to write holds a value with no JSON form: the first input, or
    output of a node, that isn't written alone either. It is looked for only once the whole save was refused.

    Args:
        graph: The run's graph.
        saved: The run as ``Run._saved`` gives it.

    Returns:
        The input's address, or the node as a save names it and the output (``e/ask.result``).
    """
    columns = {address: (graph._input_fields[address].spec.whole, value) for address, value in saved.inputs.items()}
    by_node = [*saved.units.items(), *((name, carried.units) for name, carried in saved.earlier.items())]
    for name, units in by_node:
        outputs = graph._functions[tuple(name.split("/"))].outputs
        columns.update({f"{name}.{output}": (outputs[output].column, column) for output, column in units.outputs.items()})
    for at, (written_by, value) in columns.items():
        try:
            written_by.adapter.serializer.to_json(value)
        except pydantic_core.PydanticSerializationError:
            return at
    return "the run"


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

