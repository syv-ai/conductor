"""``Run``: a graph's inputs and every unit's result, as one value that writes itself as JSON.

A unit is one node on one row. A unit is ``done`` (its outputs kept),
``skipped`` (a branch not taken), ``blank`` (a plain input got nothing, so the
node wasn't called) or ``waiting`` for a person. A unit with no entry hasn't
run yet. Going on is one leg: start every unit whose inputs exist, until
nothing more can start. So a run can stop anywhere (a person to ask, a
failure, the row ceiling, a timeout) and carry on from its JSON in another
process.
"""

from __future__ import annotations

import asyncio
import datetime
import decimal
import hashlib
import json
import math
import threading
import time
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Literal, overload

import polars as pl
import pydantic_core
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PlainSerializer,
    PositiveInt,
    TypeAdapter,
    ValidationError,
)

from conductor._internal.ask import RowQuestion, Waiting, answer_outputs, waiting_questions
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import InvalidCallError, Problem, make_problem, value_free_errors
from conductor._internal.fields import (
    SKIPPED,
    FieldSpec,
    FieldValueError,
    carried_list,
    read_value,
    takes_a_blank,
)
from conductor._internal.node_definition import (
    from_run_values,
)

if TYPE_CHECKING:
    from conductor._internal.graph import Graph, InputPlan, Level, PlacedNode, Source
    from conductor._internal.leg import Event, _EventBase
    from conductor._internal.node_definition import FunctionDefinition

#: A row: the parent's key plus a number, ``()`` for the graph's own row.
Key = tuple[int, ...]

#: How a run stands while nothing has ended it early: not started, in a leg, waiting for a person, or done.
GoingStatus = Literal["ready", "running", "complete", "waiting"]
#: How a leg ended before the run was done: a node failed, a table outgrew ``max_rows``, a cancel, or the deadline.
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
        status: ``failed`` (a node failed), ``stopped`` (a table outgrew ``max_rows``), ``cancelled`` or ``timed_out``.
        cause: Why, as a problem with its place and code.
    """

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
    """A row a node wasn't called on, because a plain input got a blank there.

    Attributes:
        node_id: The node, as the graph's caller names it.
        row: The row's place under each table above the node.
        input: The input that got nothing, as the node's caller names it.
    """

    node_id: str
    row: tuple[int, ...]
    input: str


@dataclass(frozen=True)
class Frame:
    """A table as a run keeps it: columns of plain values, so a cell is a list index."""

    rows: int
    columns: dict[str, list[Any]]

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.columns, strict=False) if self.columns else pl.DataFrame()


@dataclass(slots=True)
class Unit:
    """What one node came to on one row.

    A run and its copies share their units, so a unit is never changed in place: a new state is a new ``Unit``.
    """

    state: Literal["done", "skipped", "blank", "waiting"]
    out: dict[str, Any] | None = None
    #: For a waiting unit, what the person is asked, and for a done one, what they were asked when a person answered it.
    question: RowQuestion | None = None
    #: For a blank unit, the plain input that got nothing.
    blank_input: str | None = None
    #: For a skipped unit, the depth of the row where the branch not taken was decided.
    skipped_from: int | None = None


class WaitingUnit(Unit):
    """A unit waiting for a person: its node raised ``Ask`` on the row, and it always holds the question."""

    __slots__ = ()
    state: Literal["waiting"]
    question: RowQuestion

    def __init__(self, question: RowQuestion) -> None:  # one is made for every row that asks: positional, and no more
        self.state, self.out, self.question, self.blank_input, self.skipped_from = "waiting", None, question, None, None


class Run:
    """One run of a graph. Every call returns a new ``Run``; ``to_json`` is all of it.

    Attributes:
        graph: The graph the run runs on: the one it was started from, or, when a caller gave a list to an input that
            takes one value per row, that graph compiled for those lists, whose ``plan`` says where each node ran.
        inputs: The caller's inputs as checked, by address (``"upper.text"``) whether they were given by name or by
            address.
        max_rows: The most rows any table may hold before the run stops as ``stopped``; ``None`` for no ceiling.
        state: How the run stands: ``Going`` (``ready`` before the first leg, then ``running``, ``waiting`` or
            ``complete``), or ``EndedEarly`` (``failed``, ``stopped`` at the row ceiling, ``cancelled`` or
            ``timed_out``) with the ``Problem`` that caused it.
    """

    graph: Graph
    inputs: dict[str, Any]
    #: The caller's inputs and the lists an author typed in, by address: what a graph input's edge reads.
    _given_values: dict[str, Any]
    max_rows: int | None
    _units: dict[str, dict[Key, Unit]]
    state: RunState
    #: node -> fingerprint of what a unit received -> the earlier units that received it, in row order, not yet taken.
    _earlier: dict[str, dict[str, list[Unit]]]
    #: The nodes whose ``_earlier`` entry is this run's own to take from, not shared with the run it was copied from.
    _earlier_own: set[str]
    #: Whether a node's earlier units include answers or questions, which are kept by the rows above them too.
    _asking: dict[str, bool]
    #: For a node whose units a saved run made with other code or another signature, what that code was; they are never carried.
    _ran_as: dict[str, str]
    #: The nodes whose units were recorded out of row order (threads finish when they finish), sorted when read.
    _unordered: set[str]
    #: The nodes every row of which was finished when the last leg ended, so a later leg doesn't say them complete again.
    _complete: frozenset[str]

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("a run is made with graph.with_inputs(...), graph.run(...) or graph.load_run(saved)")

    def __repr__(self) -> str:
        return f"<Run {self.status}: {sum(len(u) for u in self._units.values())} units>"

    @property
    def status(self) -> RunStatus:
        """How the run stands, as ``state`` says it."""
        return self.state.status

    # -- going on ---------------------------------------------------------------

    def resume(self, *, max_rows: int | None = None, from_run: Mapping[type, Any] = MappingProxyType({}), timeout: float | None = None) -> Run:
        """Go on: start every unit whose inputs exist, until the run is done, waits, fails, reaches its ceiling or its ``timeout``.

        ``max_rows`` raises (or sets) the ceiling a stopped run carries on
        under. ``from_run`` gives each ``FromRun`` type its value. Past its
        ``timeout`` a leg starts nothing more but waits for the calls already
        on threads, as far as their ``attempt_timeout``, and keeps them; so it
        can end later than ``timeout``, and a call that came back is never made
        again.
        """
        checked_leg(max_rows=max_rows, timeout=timeout)
        return self._leg(max_rows, from_run_values(self.graph._needs_from_run, from_run), None, _deadline(timeout))

    def stream(
        self, *, max_rows: int | None = None, from_run: Mapping[type, Any] = MappingProxyType({}), timeout: float | None = None, cancel: asyncio.Event | None = None,
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
        checked_leg(max_rows=max_rows, timeout=timeout, checkpoint_every=checkpoint_every)
        return self._stream(max_rows, from_run_values(self.graph._needs_from_run, from_run), timeout, cancel, checkpoint_every)

    async def _stream(
        self, max_rows: int | None, from_run: dict[type, Any], timeout: float | None, cancel: asyncio.Event | None, checkpoint_every: float | None,
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
                self._leg(max_rows, from_run, lambda event: _hand_over(loop, events, event), stop, checkpoint_every)
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
        self, max_rows: int | None, from_run: dict[type, Any], emit: Callable[[_EventBase], None] | None, stop: Callable[[], EndedEarly | None] | None,
        checkpoint_every: float | None = None,
    ) -> Run:
        from conductor._internal.leg import Leg

        run = self._copy(max_rows=max_rows if max_rows is not None else self.max_rows)
        Leg(run, emit, stop, from_run, checkpoint_every).go()
        return run

    @overload
    def answer(self, unit: Waiting | str, value: Any, /) -> Run: ...

    @overload
    def answer(self, answers: Mapping[str, Any], /) -> Run: ...

    def answer(self, unit: Waiting | str | Mapping[str, Any], value: Any = _ONE_ANSWER, /) -> Run:
        """Give waiting rows their answers, checked against their questions; nothing runs until ``resume``.

        ``run.answer(unit, value)`` answers one row, ``run.answer({unit_id: value, ...})``
        several. A node with one output is answered with its value, one with
        several with a mapping by output name, where a question with a default
        or a blank-taking type without choices may be left out, and of an
        ``exclusive_group`` exactly one is answered; a question limited by
        choices is answered unless it has a default. Every fault is refused at once and the run is left as it was:
        a row answered twice in one call, an id that names nothing waiting
        (``not_waiting``), or one whose row now holds another question
        (``answer_stale``, with ``ctx.moved_to`` when that question waits on
        another row). A cancelled run takes no answers (``run_cancelled``):
        ``resume`` it, and it carries on and waits again, or start a new run.
        """
        if self.status == "cancelled":
            raise InvalidCallError([make_problem("run_cancelled", "status")])
        if value is not _ONE_ANSWER:
            given: Mapping[str, Any] = {unit.id if isinstance(unit, Waiting) else str(unit): value}
        elif isinstance(unit, Mapping):
            given = unit if all(isinstance(key, str) for key in unit) else {str(key): one for key, one in unit.items()}
        else:
            raise TypeError("run.answer takes a unit and its answer, or a mapping of unit ids to answers")
        problems: list[Problem] = []
        answered: dict[str, dict[Key, Unit]] = {}
        for unit_id, one in given.items():
            outer, key, fingerprint = _parse_unit(unit_id)
            if key is None:
                problems.append(make_problem("not_waiting", outer, waiting_id=unit_id, status=self.status))
                continue
            here = [(n, u) for n in self.graph._placed_as(outer) if isinstance(u := self._units[n].get(key), WaitingUnit)]
            matching = [(n, u) for n, u in here if fingerprint is None or _question_fingerprint(self.graph._nodes[n], u) == fingerprint]
            if len(matching) != 1:
                moved = {"moved_to": went} if fingerprint is not None and (went := self._moved_to(outer, fingerprint)) else {}
                if len(here) == 1 and fingerprint is not None:
                    current = self._waiting(here[0][0], key, here[0][1]).id
                    problems.append(make_problem("answer_stale", outer, *key, waiting_id=unit_id, current=current, **moved))
                else:
                    problems.append(make_problem("not_waiting", outer, *key, waiting_id=unit_id, status=self.status, **moved))
                continue
            ((node_id, found),) = matching
            if key in answered.get(node_id, {}):
                problems.append(make_problem("invalid_answer", outer, *key, waiting_id=unit_id, reason="the row is answered twice in one call"))
                continue
            placed = self.graph._nodes[node_id]
            shown = self.graph._shown_outputs(node_id) if node_id in self.graph._spliced else {}  # an embedding's questions are asked, and answered, by the names it offers
            if shown and isinstance(one, Mapping):
                own = {name: inner for inner, name in shown.items()}
                one = {own.get(name, name): value for name, value in one.items()}
            read, faults = answer_outputs(placed, one, found.question.overrides)
            if faults:
                problems.extend(
                    make_problem("invalid_answer", outer, *key, waiting_id=unit_id, reason=reason, **({"question": shown.get(q, q)} if q else {})) for q, reason in faults
                )
            answered.setdefault(node_id, {})[key] = Unit("done", kept_outputs(placed.definition.outputs, read), found.question) if not faults else found
        if problems:
            raise InvalidCallError(problems)
        run = self._copy(share_units=True)
        for node_id, units in answered.items():
            run._units[node_id] = {**self._units[node_id], **units}
        return run

    def _moved_to(self, outer: str, fingerprint: str) -> str | None:
        """The id of the row where the question a stale id names now waits, if it waits on another row of the same node."""
        for n in self.graph._placed_as(outer):
            for k, u in self._units[n].items():
                if isinstance(u, WaitingUnit) and _question_fingerprint(self.graph._nodes[n], u) == fingerprint:
                    return self._waiting(n, k, u).id
        return None

    def with_inputs(self, inputs: Mapping[str, Any] = MappingProxyType({}), /, *, max_rows: int | None = None, **by_name: Any) -> Run:
        """A run of these changed inputs, the rest as before, ready to ``resume`` or ``stream``.

        It carries this run's work, and what this run carried and didn't use
        for a node that hasn't finished (a finished node's is dropped, so a
        save stays its size over edits): a row that receives what an earlier unit of its node received takes
        that unit's outcome, an answer or a question still waiting included, and
        the node is not called for it. Identical rows match the earlier ones in
        row order: the first identical row takes the first earlier one, the
        second the second. Inputs are given as ``graph.with_inputs`` takes
        them, and checked the same way.
        """
        changed = {**inputs, **by_name}
        named = self.graph._given_by_key(changed)[0]
        given = {field.name: self.inputs[key] for key, field in self.graph._input_fields.items() if key in self.inputs and key not in named}
        fresh = self.graph.with_inputs({**given, **changed}, max_rows=max_rows if max_rows is not None else self.max_rows)
        fresh._earlier = self._carried()
        return fresh

    def forget(self, *node_ids: str) -> Run:
        """A run of the same inputs that has forgotten these nodes' work, ready to ``resume`` or ``stream``.

        Every other unit is carried as ``with_inputs`` carries it, so only the
        forgotten nodes are called again, and a node below them that then
        receives what it received before is not. A host forgets a node whose
        access it must check again. An embedded graph's id forgets its nodes.
        """
        unknown = [n for n in node_ids if not self.graph._placed_as(n)]
        if unknown:
            raise InvalidCallError([make_problem("node_not_in_graph", n, node_id=n, nodes=list(self.graph._readable)) for n in unknown])
        fresh = self.with_inputs()
        for n in node_ids:
            for placed_id in self.graph._placed_as(n):
                fresh._earlier.pop(placed_id, None)
        return fresh

    def _carried(self, into: Graph | None = None) -> dict[str, dict[str, list[Unit]]]:
        """What a run going on from this one may take: each node's finished and waiting units by what they received, then what this run carried and didn't take.

        ``into`` another graph keeps only the nodes it holds as they are here (by ``reuse_digest``): the others' units could
        never be taken there, and carried they would only make its save larger.
        """
        earlier: dict[str, dict[str, list[Unit]]] = {}
        for n, units in self._units.items():
            placed = self.graph._nodes[n]
            if not placed.definition.policy.reuse_when_inputs_match or n in self._ran_as:
                continue
            if into is not None and (n not in into._nodes or into._nodes[n].reuse_digest != placed.reuse_digest):
                continue
            by: dict[str, list[Unit]] = {}
            for k, u in self._in_order(n):
                if u.state == "done" or u.state == "waiting":
                    fingerprint = _fingerprint(placed.reuse_digest, self._inputs(placed, k), plain=placed.plain_inputs)
                    if fingerprint is not None:
                        by.setdefault(self._asked_key(placed, k, fingerprint) if u.question is not None else fingerprint, []).append(u)
            for fingerprint, left in self._earlier.get(n, {}).items():
                if left:
                    by.setdefault(fingerprint, []).extend(left)
            if by:
                earlier[n] = by
        return earlier

    def again(
        self, inputs: Mapping[str, Any] = MappingProxyType({}), /, *, max_rows: int | None = None, from_run: Mapping[type, Any] = MappingProxyType({}),
        timeout: float | None = None, **by_name: Any,
    ) -> Run:
        """Run again with changed inputs, the rest as before: ``run.with_inputs(...).resume(...)``.

        A row that receives what it received before is not called again.
        """
        return self.with_inputs(inputs, max_rows=max_rows, **by_name).resume(from_run=from_run, timeout=timeout)

    # -- reading ----------------------------------------------------------------

    @property
    def waiting(self) -> list[Waiting]:
        """Every row waiting for a person, in node then row order."""
        return [self._waiting(n, k, u) for n in self._units for k, u in self._in_order(n) if isinstance(u, WaitingUnit)]

    def _in_order(self, node_id: str) -> Any:
        """A node's units in row order."""
        units = self._units[node_id]
        return sorted(units.items()) if node_id in self._unordered else units.items()

    def _waiting(self, node_id: str, key: Key, unit: WaitingUnit) -> Waiting:
        question = unit.question
        outer = self.graph._outward(node_id)
        shown = self.graph._shown_outputs(node_id) if node_id in self.graph._spliced else {}
        asked = [q.model_copy(update={"name": shown[q.name]}) if q.name in shown else q for q in waiting_questions(self.graph._nodes[node_id], question.overrides)]
        row = "".join(f"@{i}" for i in key)  # a waiting id is node@row~fingerprint, as _parse_unit reads it
        return Waiting(
            id=f"{outer}{row}~{_question_fingerprint(self.graph._nodes[node_id], unit)}", node_id=outer, row=key, prompt=question.prompt,
            questions=asked, inputs=question.inputs,
        )

    @property
    def not_called(self) -> list[NotCalled]:
        """Rows a node wasn't called on because a plain input was blank: which node, row and input."""
        found: list[NotCalled] = []
        seen = set()
        for n, units in self._units.items():
            for k, u in units.items():
                if u.blank_input is not None and (place := (self.graph._outward(n), k)) not in seen:  # an embedding's row once, by its first node
                    seen.add(place)
                    found.append(NotCalled(node_id=place[0], row=k, input=self.graph._outward(f"{n}.{u.blank_input}").partition(".")[2] or u.blank_input))
        return found

    @property
    def results(self) -> dict[str, dict[str, Any]]:
        """Each node's outputs: one value on a node that ran once, else a list in row order, ``None`` where a row has none.

        An embedded graph is one node here, with the outputs it offers.
        """
        nodes = self.graph._nodes
        found = {drawn: {name: self._collect(nodes[n], o) for name, (n, o) in fields.items()} for drawn, fields in self.graph._readable.items()}
        return found

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
        results = self.results
        return {address: results[address.split(".")[0]][address.split(".", 1)[1]] for address in self.graph.outputs}

    def _collect(self, placed: PlacedNode, output: str) -> Any:
        keys, _ = self._rows(placed.level, ())
        values = []
        for key in keys:
            unit = self._units[placed.id].get(key)
            value = unit.out.get(output) if unit is not None and unit.out is not None else None
            values.append(None if value is SKIPPED else value.frame() if isinstance(value, Frame) else [None if v is SKIPPED else v for v in value] if isinstance(value, list) else value)
        return values[0] if not placed.level else values

    # -- what a leg reads -------------------------------------------------------

    def _reused(self, placed: PlacedNode, key: Key, kwargs: dict[str, Any], in_turn: Callable[[], bool]) -> Any:
        """The next earlier unit of ``placed`` that received ``kwargs``, taken so an identical row after it takes the one after.

        Identical rows take them in row order: when an earlier unit is there to take and ``in_turn`` says a row
        before this one hasn't been tried yet, this is ``PENDING``, tried again once it has.

        A person's answer, or a question still waiting, is taken only by a row under the same rows above it, so two
        documents' identical lines keep each document's answers when the documents move.
        """
        earlier = self._earlier.get(placed.id)
        if not earlier:
            return None
        fingerprint = _fingerprint(placed.reuse_digest, kwargs, plain=placed.plain_inputs)
        if fingerprint is None:
            return None
        asks = self._asking.get(placed.id)
        if asks is None:
            asks = self._asking[placed.id] = any(k.startswith("asked:") for k in earlier)
        if asks and earlier.get(asked := self._asked_key(placed, key, fingerprint)):
            fingerprint = asked
        if not earlier.get(fingerprint):
            return None
        if not in_turn():
            return PENDING
        if placed.id not in self._earlier_own:  # the lists are the earlier run's too until this run takes from them
            earlier = self._earlier[placed.id] = {fp: list(units) for fp, units in earlier.items()}
            self._earlier_own.add(placed.id)
        return earlier[fingerprint].pop(0)

    def _asked_key(self, placed: PlacedNode, key: Key, fingerprint: str) -> str:
        """Where an answered or waiting unit is kept for reuse: what it received, and the value of each row above its own."""
        above = [self._row_value(placed.level[i], key[: i + 1]) for i in range(len(placed.level) - 1)]
        lineage = _fingerprint("", {"rows": above}) if above else ""
        return f"asked:{fingerprint}:{lineage}"

    def _given(self, address: str) -> Any:
        """The list a caller gave, or an author typed in, at ``address``, whose items are the rows of the table it makes."""
        return self._given_values[address]

    def _row_value(self, table: str, key: Key) -> Any:
        """What one row of a table holds: an item of a list, or a table's row as a mapping."""
        node_id, output = self.graph._makers[table]
        if node_id is None:
            return self._given(output)[key[-1]]
        unit = self._units[node_id].get(key[:-1])
        made = unit.out.get(output) if unit is not None and unit.out else None
        if isinstance(made, Frame):
            return {c: v[key[-1]] for c, v in made.columns.items()}
        return made[key[-1]] if isinstance(made, list) else None

    def _call(self, placed: PlacedNode, kwargs: dict[str, Any], from_run: Mapping[type, Any]) -> tuple[Any, BaseException | None]:
        """Call ``placed`` on what a unit received: its outputs as a run keeps them, or what went wrong.

        A value an edge brings is checked against the input first where its source's type doesn't settle it, and what
        the node returns against its declared outputs; either refusal is a ``FieldValueError``, which quotes no value.
        """
        try:
            for name in placed.checked:
                kwargs = {**kwargs, name: _checked(placed.definition.inputs[name], name, kwargs[name], code="invalid_edge_value")}
            if placed.definition.from_run:
                kwargs = {**kwargs, **{name: from_run[needed] for name, needed in placed.definition.from_run.items()}}
            returned = placed.definition.function(**kwargs)
        except BaseException as error:  # noqa: BLE001  a node's own failure, or its Ask, is the run's to record
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return None, error
        try:
            return checked_outputs(placed.definition, returned), None
        except FieldValueError as wrong:
            return None, wrong

    def _stop(self, table: str, count: int) -> None:
        """Stop at the row ceiling, unless the leg already ended for a failure, a cancel or its timeout, which stays the cause."""
        match self.state:
            case EndedEarly():
                return
            case Going():
                maker = self.graph._makers[table][0]
                self.state = EndedEarly("stopped", make_problem(
                    "row_ceiling", self.graph._outward(maker) if maker else "inputs", table=self.graph._outward(table, table=True), rows=count, max_rows=self.max_rows,
                ))

    def _rows_made(self) -> dict[str, int]:
        """How many rows each table has so far; a typed list's are its items times the rows they go under."""
        made: dict[str, int] = {}
        for table, (node_id, output) in self.graph._makers.items():
            if node_id is not None:
                made[table] = sum(_length(u.out[output]) for u in self._units[node_id].values() if u.state == "done" and u.out)

        stack = [("", 1), *((t, made[t]) for t, (node_id, _) in self.graph._makers.items() if node_id is not None)]
        while stack:
            parent, rows = stack.pop()
            for address in self.graph._lists_under.get(parent, ()):
                made[address] = rows * len(self._given(address))
                stack.append((address, made[address]))
        return made

    def _rows(self, level: Level, row: Key, *, counted: bool = False) -> tuple[list[Key], bool]:
        """The rows of ``level`` under ``row`` (or the one above it), and whether all of them exist yet.

        With ``counted``, the last table's rows are only counted: ``[(n,)]`` stands for ``n`` rows, unbuilt.
        """
        start, deepest = len(row), len(level)
        if start >= deepest:
            return [row[:deepest]], True
        keys, complete = [row], True
        for depth, table in enumerate(level[start:], start):
            node_id, output = self.graph._makers[table]
            below: list[Key] = []
            for parent in keys:
                if node_id is None:
                    count: int | None = len(self._given(output))
                else:
                    unit = self._units[node_id].get(parent)
                    if unit is None or unit.state == "waiting":
                        count = None
                    else:
                        count = _length(unit.out.get(output)) if unit.state == "done" and unit.out else 0
                if count is None:
                    complete = False
                    continue
                if counted and depth == deepest - 1:
                    below = [(below[0][0] + count,)] if below else [(count,)]
                else:
                    below.extend((*parent, i) for i in range(count))
            keys = below
        return keys, complete

    def _not_made(self, source: Source, row: Key) -> Literal["skipped", "blank"] | None:
        """Why the table ``source`` reads under ``row`` was never made: its maker took no branch there, or wasn't called for
        a blank input, or gave a blank in place of it."""
        if len(row) >= len(source.level) or source.node_id is None:
            return None
        node_id, output = self.graph._makers[source.level[len(row)]]
        unit = self._units[node_id].get(row) if node_id is not None else None
        if unit is None:
            return None
        if unit.state == "skipped" or (unit.state == "done" and unit.out is not None and unit.out.get(output) is SKIPPED):
            return "skipped"
        return "blank" if unit.state == "blank" or (unit.state == "done" and unit.out is not None and output in unit.out and unit.out[output] is None) else None

    def _skip_origin(self, source: Source, key: Key) -> int:
        """The depth of the row where the branch behind a skipped value was decided."""
        if source.node_id is None:
            return len(key)
        unit_key = key[:-1] if source.made else key
        unit = self._units[source.node_id].get(unit_key)
        if unit is not None and unit.state == "skipped":
            return unit.skipped_from if unit.skipped_from is not None else len(unit_key)
        if unit is not None and unit.out is not None and unit.out.get(source.output) is SKIPPED:
            return len(unit_key)  # the whole output was the branch not taken on its node's row
        return len(key)

    def _cell(self, source: Source, key: Key) -> Any:
        if source.node_id is None:
            value = self._given(source.output)
            return value[key[-1]] if source.made else value
        unit = self._units[source.node_id].get(key[:-1] if source.made else key)
        if unit is None or unit.state == "waiting":
            return PENDING
        if unit.state != "done":
            return SKIPPED if unit.state == "skipped" else None
        value = unit.out[source.output]  # type: ignore[index]
        if value is SKIPPED or not source.made:
            return value
        if isinstance(value, Frame):
            if source.whole_table:
                return {c: v[key[-1]] for c, v in value.columns.items()}
            return value.columns[source.column][key[-1]]  # type: ignore[index]
        return value[key[-1]]

    def _inputs(self, placed: PlacedNode, row: Key) -> Any:
        """The keyword arguments for ``placed`` on ``row``; PENDING; or the ``Unit`` it comes to uncalled, skipped or blank.

        Every input is read before a blank is the answer: the call is skipped when any input is, and blank only when none is.
        """
        kwargs: dict[str, Any] = {}
        uncalled: Unit | None = None
        gathered: dict[str, list[tuple[Level, Key, Any]]] = {}
        for name, plan in placed.inputs.items():
            spec = plan.spec
            if not plan.sources:
                kwargs[name] = plan.value
            elif spec.shape == "one":
                value = SKIPPED
                origins = []
                for source in plan.sources:
                    value = self._cell(source, row[: len(source.level)])
                    if value is PENDING:
                        return PENDING
                    if value is not SKIPPED:
                        break
                    origins.append(self._skip_origin(source, row[: len(source.level)]))
                if value is SKIPPED:
                    return Unit("skipped", skipped_from=min(origins))
                if value is None and not spec.optional:
                    uncalled = uncalled or Unit("blank", blank_input=name)
                kwargs[name] = value
            elif spec.shape == "many":
                if len(plan.sources) == 1 and plan.sources[0].node_id is None and self.inputs[plan.sources[0].output] is None:
                    kwargs[name] = None  # a Series that may be None, and nothing was given
                    continue
                items: list[tuple[Level, Key, Any]] = []
                skipped_above = blank_above = 0
                skips = []  # for each item that is a skip: whether its branch was decided at or above the row it is gathered under, and where
                lone = plan.sources[0]
                on: dict[Level, list[Source]] = {}  # sources on one table are read row by row, in the table's order
                if len(plan.sources) == 1 and (lone.node_id is not None or lone.made):
                    on[lone.level] = plan.sources
                else:
                    for source in plan.sources:
                        if source.node_id is None and not source.made:  # the caller's list, given whole
                            items.extend(((source.output,), (i,), v) for i, v in enumerate(self.inputs[source.output]))
                            continue
                        on.setdefault(source.level, []).append(source)
                for level, sources in on.items():
                    under = row[: sources[0].anchor]
                    keys, complete = self._rows(level, under)
                    if not complete:
                        return PENDING
                    if not keys:
                        why = [self._not_made(source, under) for source in sources]
                        skipped_above, blank_above = skipped_above + why.count("skipped"), blank_above + why.count("blank")
                    if len(sources) == 1:  # one source: each row is its value, or a skip
                        (source,) = sources
                        for key in keys:
                            value = self._cell(source, key)
                            if value is PENDING:
                                return PENDING
                            if value is SKIPPED:
                                origin = self._skip_origin(source, key)
                                skips.append((len(key) > len(under) and origin <= len(under), origin))
                                value = None
                            items.append((level, key, value))
                        continue
                    for key in keys:
                        values, origins = [], []
                        for source in sources:
                            value = self._cell(source, key)
                            if value is PENDING:
                                return PENDING
                            if value is SKIPPED:
                                origins.append(self._skip_origin(source, key))
                            else:
                                values.append(value)
                        if values:
                            items.extend((level, key, value) for value in values)
                        else:
                            items.append((level, key, None))
                            skips.append((len(key) > len(under) and max(origins) <= len(under), min(origins)))
                if skipped_above and skipped_above == len(plan.sources):
                    return Unit("skipped", skipped_from=len(row))  # every source was a branch not taken above this row: so is the call
                if blank_above and skipped_above + blank_above == len(plan.sources):
                    if spec.optional:  # a Series that may be None receives the blank in place of the list, as a Table does
                        kwargs[name] = None
                    else:  # the rows were never made, as their maker had a blank input: not the same as no rows
                        uncalled = uncalled or Unit("blank", blank_input=name)
                    continue
                if items and len(skips) == len(items) and all(above for above, _ in skips):
                    return Unit("skipped", skipped_from=min(origin for _, origin in skips))  # every row under this one was skipped by a decision at or above it
                gathered[name] = items
            elif plan.sources[0].node_id is None:
                kwargs[name] = self.inputs[plan.sources[0].output]  # a caller's table, read to its columns at the call
            else:
                table = self._table(plan, row)
                if table is PENDING or isinstance(table, Unit):
                    return table
                if table is None and not spec.optional:
                    uncalled = uncalled or Unit("blank", blank_input=name)
                kwargs[name] = table
        if uncalled is not None:
            return uncalled
        if gathered:
            # A row a Series input can't take leaves every Series input on that table.
            blank = {name: {(lvl, k) for lvl, k, v in items if v is None} for name, items in gathered.items() if not takes_a_blank(placed.inputs[name].spec.type)}
            for name, items in gathered.items():
                others = [rows for other, rows in blank.items() if other != name]
                keep_blank = takes_a_blank(placed.inputs[name].spec.type)
                kwargs[name] = [v for lvl, k, v in items if (keep_blank or v is not None) and not any((lvl, k) in rows for rows in others)]
        return kwargs

    def _table(self, plan: InputPlan, row: Key) -> Any:
        """The table a ``Table`` input receives on ``row``; for alternatives, the one whose branch ran, or a skipped ``Unit`` when none did.

        ``None`` when the table was never made there because its maker had a blank input, whether it is the one maker or the
        branch that ran.
        """
        if not plan.alternatives:
            under = row[: plan.table_anchor]
            if all(self._not_made(source, under) == "blank" for source in plan.sources):
                return None  # its maker had a blank input there: the table was never made
            return self._columns(plan.sources, plan.table_level, under, plan.spec.columns)
        for source in plan.sources:
            under = row[: source.anchor]
            why = self._not_made(source, under)
            if why == "blank":
                return None
            if why is None:
                return self._columns([source], source.level, under, plan.spec.columns)
        return Unit("skipped", skipped_from=len(row))

    def _columns(self, sources: list[Source], level: Level, row: Key, declared: Mapping[str, Any] | None) -> Any:
        """``sources``' columns on the rows of ``level`` under ``row``, only the declared ones when the input names them."""
        keys, complete = self._rows(level, row)
        if not complete:
            return PENDING
        columns: dict[str, list[Any]] = {}
        for source in sources:
            for column in (source.columns or {}) if source.whole_table else ():
                columns[column] = [None] * len(keys)  # a table with no rows under this row still has its columns
        for source in sources:
            depth = len(source.level)
            cells = [self._cell(source, key[:depth]) for key in keys]
            if any(c is PENDING for c in cells):
                return PENDING
            if source.whole_table:
                for i, cell in enumerate(cells):
                    for column, value in (cell if isinstance(cell, dict) else {}).items():
                        columns.setdefault(column, [None] * len(keys))[i] = value
            else:
                columns[source.name] = [None if c is SKIPPED else c for c in cells]
        if declared is not None:  # a declared column a table of unknown columns didn't bring is left missing, for the input's check to name
            columns = {c: columns.get(c, [None] * len(keys)) for c in declared if c in columns or not keys}
        return pl.DataFrame(columns, strict=False)

    # -- JSON -------------------------------------------------------------------

    def to_json(self) -> str:
        """The run as JSON: its inputs, every unit, and what it went on from, a column per output of each node."""
        try:
            return self._written()
        except _SaveError as unwritable:
            raise InvalidCallError([make_problem("state_unwritable", "state", unwritable.at, at=unwritable.at, reason=unwritable.reason)]) from None

    def _written(self) -> str:
        units = {n: _dump_units(self.graph._nodes[n], dict(self._in_order(n)) if n in self._unordered else us) for n, us in self._units.items() if us}
        earlier = {}
        for n, by in self._earlier.items():
            listed = [(fp, unit) for fp, units in by.items() for unit in units]
            if listed:
                numbered: dict[Key, Unit] = {(i,): unit for i, (_, unit) in enumerate(listed)}
                earlier[n] = _dump_units(self.graph._nodes[n], numbered, fingerprints=[fp for fp, _ in listed])
        inputs = {address: _dump_input(self.graph, address, v) for address, v in self.inputs.items()}
        non_finite: dict[str, list[_Path]] = {}
        for address, written_input in inputs.items():
            if not self.graph._input_fields[address].spec.plain:
                _find_non_finite(written_input, [], non_finite.setdefault(address, []))
        code = {n: self._ran_as.get(n) or self.graph._nodes[n].reuse_digest for n, us in self._units.items() if us or n in earlier}
        rest: dict[str, Any] = {
            "graph": self.graph.fingerprint, "max_rows": self.max_rows, "inputs": inputs, "non_finite_inputs": {a: paths for a, paths in non_finite.items() if paths},
            "units": units, "earlier": earlier, "code": code, "complete": [n for n in self._units if n in self._complete],
        }
        written: _SavedRun
        match self.state:
            case Going(status=status):
                written = _SavedGoingRun.model_construct(status=status, **rest)
            case EndedEarly(status=status, cause=cause):
                written = _SavedEndedRun.model_construct(status=status, cause=cause, **rest)
        return written.model_dump_json(exclude_defaults=True)

    def _copy(self, *, share_units: bool = False, **changes: Any) -> Run:
        """A copy to change: its own units to record into, or with ``share_units`` this run's, for a caller that replaces whole nodes."""
        copy = start_run(self.graph, self.inputs, changes.get("max_rows", self.max_rows))
        copy._units = dict(self._units) if share_units else {n: dict(us) for n, us in self._units.items()}
        copy.state, copy._earlier, copy._unordered = self.state, dict(self._earlier), set(self._unordered)
        copy._ran_as, copy._complete = self._ran_as, self._complete
        return copy


def start_run(graph: Graph, inputs: dict[str, Any], max_rows: int | None) -> Run:
    """A run of ``graph`` on ``inputs`` already checked, by address, not started; the one way a ``Run`` is made."""
    run = Run.__new__(Run)
    run.graph, run.inputs, run.max_rows = graph, inputs, max_rows
    run._given_values = {**inputs, **graph._typed_lists}
    run._units = {n: {} for n in graph._nodes}
    run.state = _READY
    run._earlier = {}
    run._earlier_own = set()
    run._asking = {}
    run._ran_as = {}
    run._unordered = set()
    run._complete = frozenset()
    return run


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
        if data.graph != graph.fingerprint:
            raise InvalidCallError([make_problem("state_of_another_graph", "state", saved=data.graph, graph=graph.fingerprint)])
        fields = graph._input_fields
        unknown = sorted(data.inputs.keys() - fields.keys())
        if unknown:
            raise _SaveError(unknown[0], "the graph has no such input")
        graph = graph._compiled_for(frozenset(a for a, v in data.inputs.items() if fields[a].spec.per_item and isinstance(v, list)))
        if graph.problems:
            raise InvalidCallError(graph.problems)
        unknown = sorted(data.non_finite_inputs.keys() - data.inputs.keys())
        if unknown:
            raise _SaveError(unknown[0], "the graph has no such input")
        written = {address: _with_non_finite(v, data.non_finite_inputs.get(address, ()), address) for address, v in data.inputs.items()}
        inputs = {address: _load_input(graph, address, v) for address, v in written.items()}
        run = start_run(graph, inputs, data.max_rows)
        for saved_nodes in (data.units, data.earlier):
            unknown = sorted(saved_nodes.keys() - graph._nodes.keys())
            if unknown:
                raise _SaveError(unknown[0], "the graph has no such node")
        for n, dumped in data.units.items():
            loaded = _load_units(graph._nodes[n], dumped)
            run._units[n] = dict(loaded)
            if len(run._units[n]) != len(loaded):
                raise _SaveError(n, "a row is listed twice")
        run._ran_as = {n: digest for n, digest in data.code.items() if n in graph._nodes and digest != graph._nodes[n].reuse_digest}
        for n, dumped in data.earlier.items():
            if n in run._ran_as:
                continue  # made with other code: never reused
            if len(dumped.fingerprints) != len(dumped.states):
                raise _SaveError(n, f"it holds {len(dumped.fingerprints)} fingerprints for {len(dumped.states)} units")
            by: dict[str, list[Unit]] = {}
            for fp, (_, unit) in zip(dumped.fingerprints, _load_units(graph._nodes[n], dumped), strict=True):
                by.setdefault(fp, []).append(unit)
            run._earlier[n] = by
        match data:
            case _SavedGoingRun(status=status):
                run.state = Going(status)
            case _SavedEndedRun(status=status, cause=cause):
                run.state = EndedEarly(status, cause)
        run._complete = frozenset({n for n in data.complete if n in graph._nodes})
        for n, units in run._units.items():  # every row is one the saved makers made
            placed = graph._nodes[n]
            made, _ = run._rows(placed.level, ())
            stray = sorted(set(units) - set(made))
            if stray:
                raise _SaveError(n, f"it holds the row {list(stray[0])}, which the saved run never made")
        return run
    except _SaveError as unreadable:
        raise InvalidCallError([make_problem("state_unreadable", "state", unreadable.at, at=unreadable.at, reason=unreadable.reason)]) from None


def checked_leg(*, max_rows: int | None = None, timeout: float | None = None, checkpoint_every: float | None = None) -> None:
    """A leg's ceiling (at least 1 row) and its seconds (a finite number above 0) are refused at the call."""
    problems = []
    if max_rows is not None and max_rows < 1:
        problems.append(make_problem("invalid_max_rows", "max_rows", max_rows=max_rows))
    if timeout is not None and not _seconds(timeout):
        problems.append(make_problem("invalid_timeout", "timeout", timeout=timeout))
    if checkpoint_every is not None and not _seconds(checkpoint_every):
        problems.append(make_problem("invalid_checkpoint_every", "checkpoint_every", checkpoint_every=checkpoint_every))
    if problems:
        raise InvalidCallError(problems)


def _seconds(value: float) -> bool:
    return math.isfinite(value) and value > 0


# -- values -------------------------------------------------------------------------------------


def kept_outputs(outputs: Mapping[str, FieldSpec], spread: dict[str, Any]) -> dict[str, Any]:
    """A node's checked outputs by name as a run keeps them: a table as columns, a ``Series`` as a list."""
    for name, spec in outputs.items():
        value = spread[name]
        if value is SKIPPED or value is None:  # a branch not taken, or the blank of a field that may be nothing
            continue
        if spec.shape == "table":
            spread[name] = Frame(value.height, value.to_dict(as_series=False))
        elif spec.shape == "many":
            spread[name] = list(value)  # a SKIPPED item is a row whose branch wasn't taken
    return spread


def checked_outputs(definition: FunctionDefinition[..., Any], returned: Any) -> Any:
    """What a node returned, as its outputs by name as a run keeps them; ``SKIPPED`` for a whole branch not taken.

    Each is read strictly by its declared type (a value already of its class as it is, a model from a dict, never
    ``"12"`` for an ``int``), and of an ``exclusive_group`` at most one output holds a value. A fault raises
    ``FieldValueError`` with the code ``invalid_output``.
    """
    if returned is SKIPPED:
        return SKIPPED
    outputs = definition.outputs
    spread, settled = definition.returns.spread(returned, outputs)
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


def _checked(spec: FieldSpec, name: str, value: Any, *, code: str) -> Any:
    try:
        read = read_value(spec, value, strict=code == "invalid_output")
    except FieldValueError as refused:
        refused.code, refused.ctx = code, {"name": name, **refused.ctx}
        raise
    if code == "invalid_output" and spec.shape == "one" and spec.plain_classes and read is not SKIPPED and not isinstance(read, spec.plain_classes):
        raise FieldValueError(f"its type's own check gave a {type(read).__name__}, not a {spec.type_id}", code="invalid_output", name=name)
    return read


def _length(value: Any) -> int:
    return 0 if value is SKIPPED or value is None else value.rows if isinstance(value, Frame) else len(value)


def _dump_units(placed: PlacedNode, units: Mapping[Key, Unit], fingerprints: list[str] | None = None) -> _SavedUnits:
    """Units as columns: keys, states, questions, and one list per output; carried work by what each unit received, in place of its row."""
    done = [u for u in units.values() if u.state == "done" and u.out is not None]
    outputs: dict[str, _SavedOutput] = {}
    for name, spec in placed.definition.outputs.items():
        values = [u.out[name] for u in done]  # type: ignore[index]
        at = f"{placed.id}.{name}"
        if spec.shape == "table":
            dumped: Any = [None if v is SKIPPED or v is None else _dump_frame(spec, v, at) for v in values]
        else:
            plain = [None if v is SKIPPED or v is None else [None if i is SKIPPED else i for i in v] if spec.shape == "many" else v for v in values]
            dumped = _written_as_json(spec.column_adapter, plain, at)
        non_finite: list[_Path] = []
        if spec.shape != "table" and not spec.plain:
            _find_non_finite(dumped, [], non_finite)
        items = {n: skips for n, v in enumerate(values) if spec.shape == "many" and v is not SKIPPED and v is not None and (skips := [i for i, item in enumerate(v) if item is SKIPPED])}
        outputs[name] = _SavedOutput.model_construct(values=dumped, skipped=[i for i, v in enumerate(values) if v is SKIPPED], skipped_items=items, non_finite=non_finite)
    listed = list(units.values())
    states = [u.state for u in listed]
    columns: dict[str, Any] = dict(
        states=states, questions={i: u.question for i, u in enumerate(listed) if u.question is not None}, outputs=outputs,
        blank_inputs={i: u.blank_input for i, u in enumerate(listed) if u.blank_input is not None} if "blank" in states else {},
        skipped_from={i: u.skipped_from for i, u in enumerate(listed) if u.skipped_from is not None} if "skipped" in states else {},
    )
    return _SavedUnits.model_construct(keys=list(units), **columns) if fingerprints is None else _SavedEarlier.model_construct(fingerprints=fingerprints, **columns)


#: A place in a saved value: a key of a mapping, or an index of a list.
_Path = list[str | int]

#: A long column of a save that the writer made of the right items already: read typed, written as it is, since checking
#: each item against its type on the way out costs as much as writing it.
_AS_IS = PlainSerializer(lambda column: column, return_type=Any)

#: The places of a value's floats that aren't finite.
_Paths = Annotated[list[_Path], _AS_IS]


#: Every part of a save refuses what it doesn't declare, and writes a float that isn't finite as its text.
_SAVE_CONFIG = ConfigDict(extra="forbid", ser_json_inf_nan="strings")


class _SavedTable(BaseModel):
    """A table in a save: a table output of one finished unit, or a caller's table input.

    Attributes:
        rows: How many rows the table has, which a run counts its rows by.
        columns: Each column as ``(name, values)``, in the table's own order, each value as JSON.
        types: For a column the row type doesn't name, the plain type its values were written from (``"date"``,
            ``"decimal"``, ...), so they read back as that type and not as their JSON text.
    """

    model_config = _SAVE_CONFIG
    rows: NonNegativeInt
    columns: list[tuple[str, list[Any]]]
    types: dict[str, str] = {}


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


class _SavedUnits(BaseModel):
    """A node's units as columns, one place per unit.

    Attributes:
        keys: Each unit's row; ``None`` for carried work, which is kept by what it received, not by row.
        states: What each unit came to: ``done``, ``skipped``, ``blank`` or ``waiting``.
        questions: By place, the question a waiting unit asks, or a done one was answered on.
        blank_inputs: By place, for a blank unit, the plain input that got nothing.
        skipped_from: By place, for a skipped unit, the depth of the row where the branch not taken was decided.
        outputs: By output name, that output across the done units, in their order.
    """

    model_config = _SAVE_CONFIG
    keys: list[tuple[NonNegativeInt, ...]] | None = None
    states: Annotated[list[Literal["done", "skipped", "blank", "waiting"]], _AS_IS]
    questions: dict[NonNegativeInt, RowQuestion] = {}
    blank_inputs: dict[NonNegativeInt, str] = {}
    skipped_from: dict[NonNegativeInt, NonNegativeInt] = {}
    outputs: dict[str, _SavedOutput] = {}


class _SavedEarlier(_SavedUnits):
    """Work a run carried from the one before it and hasn't taken yet: the units, and what each received.

    Attributes:
        fingerprints: By place, the fingerprint of what the unit received, where a row receiving the same takes it.
    """

    fingerprints: list[str]


class _SavedRunBase(BaseModel):
    """What ``run.to_json()`` writes beside how the run stands, read once at ``load_run``: a malformed save is refused
    here, by where.

    Attributes:
        graph: The fingerprint of the graph that ran, so another graph's save is refused.
        max_rows: The row ceiling the run goes on under.
        inputs: The caller's inputs by address, as JSON; a table as a ``_SavedTable``.
        non_finite_inputs: By address, the places of each float that isn't finite, as ``_SavedOutput.non_finite``.
        units: By node id, the units of each node with any.
        earlier: By node id, the carried work not yet taken.
        code: By node id, the ``reuse_digest`` of the code that made its saved units, so units made by other code are
            never carried.
        complete: The nodes every row of which was finished when the last leg ended.
    """

    model_config = _SAVE_CONFIG
    graph: str
    max_rows: PositiveInt | None
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
_SavedRun = Annotated[_SavedGoingRun | _SavedEndedRun, Field(discriminator="status")]
_SAVED: TypeAdapter[_SavedRun] = TypeAdapter(_SavedRun)
_SAVED_TABLE = TypeAdapter(_SavedTable)


class _SaveError(Exception):
    """A run that can't be written to its save, or a save this graph can't read.

    Attributes:
        at: Where: an input's address, ``node.output`` or ``node.output.column``.
        reason: Why, quoting nothing of the value.
    """

    def __init__(self, at: str, reason: str) -> None:
        super().__init__(reason)
        self.at, self.reason = at, reason


def _load_units(placed: PlacedNode, dumped: _SavedUnits) -> list[tuple[Key, Unit]]:
    """A node's saved units, each output read by its own type; anything that doesn't fit raises ``_SaveError`` naming where."""
    states = dumped.states
    done = states.count("done")
    columns: dict[str, list[Any]] = {}
    for name, spec in placed.definition.outputs.items():
        at = f"{placed.id}.{name}"
        column = dumped.outputs.get(name)
        if column is None:
            if done:
                raise _SaveError(at, "the output is missing")
            continue
        values: list[Any]
        try:
            if spec.shape == "table":
                values = [None if v is None else _load_frame(spec, _SAVED_TABLE.validate_python(v), at) for v in column.values]
            else:
                values = spec.column_adapter.validate_python(_with_non_finite(column.values, column.non_finite, at))
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
        if spec.shape != "one" and not spec.optional and any(v is None for v in values):
            raise _SaveError(at, "a finished unit has no value on an output that makes rows")
        columns[name] = values
    unknown = sorted(dumped.outputs.keys() - placed.definition.outputs.keys())
    if unknown:
        raise _SaveError(f"{placed.id}.{unknown[0]}", "the node has no such output")
    keys = dumped.keys if dumped.keys is not None else [() for _ in states]
    if len(keys) != len(states):
        raise _SaveError(placed.id, f"it holds {len(keys)} rows for {len(states)} units")
    loaded: list[tuple[Key, Unit]] = []
    n = 0
    for i, (key, state) in enumerate(zip(keys, states, strict=True)):
        out = None
        if state == "done":
            out = {name: values[n] for name, values in columns.items()}
            n += 1
        question = dumped.questions.get(i)
        if state == "waiting":
            if question is None:
                raise _SaveError(placed.id, f"the waiting unit {i} holds no question")
            loaded.append((key, WaitingUnit(question)))
        else:
            loaded.append((key, Unit(state, out, question, dumped.blank_inputs.get(i), dumped.skipped_from.get(i))))
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


def _load_frame(spec: FieldSpec, saved: _SavedTable, at: str) -> Frame:
    """A saved table, each column read by its type; a column whose length isn't the row count is refused."""
    rows, columns = saved.rows, {}
    for c, values in saved.columns:
        if len(values) != rows:
            raise _SaveError(f"{at}.{c}", f"it holds {len(values)} cells for {rows} rows")
        try:
            columns[c] = _column_adapter(spec, c, saved.types.get(c)).validate_python(values)
        except ValidationError as error:
            raise _SaveError(f"{at}.{c}", value_free_errors(error)[0]["msg"]) from None
    return Frame(rows, columns)


def _column_adapter(spec: FieldSpec, column: str, written_as: str | None = None) -> TypeAdapter[Any]:
    """Checks and writes one column of a table output: by the column's type where the row type names it, else by the type its values were written as."""
    return spec.column_adapters.get(column) or _COLUMN_OF.get(written_as or "", _ANY_COLUMN)


def _dump_frame(spec: FieldSpec, frame: Frame, at: str) -> dict[str, Any]:
    """A table as JSON: its row count, its columns in order as ``[name, values]`` pairs, and the value type of each column its row type doesn't name, so it reads back as written."""
    written, types = {}, {}
    for c, values in frame.columns.items():
        written_as = None if c in spec.column_adapters else _value_type(values)
        if written_as is not None:
            types[c] = written_as
        written[c] = _written_as_json(_column_adapter(spec, c, written_as), values, f"{at}.{c}")
    return {"rows": frame.rows, "columns": [[c, v] for c, v in written.items()], **({"types": types} if types else {})}


def _written_as_json(adapter: TypeAdapter[Any], values: list[Any], at: str) -> Any:
    """``values`` as JSON; a value with no JSON form raises ``_SaveError`` naming where, never quoting it."""
    try:
        written = adapter.dump_python(values, mode="json")
    except Exception as error:  # noqa: BLE001  pydantic's and the codec's errors both quote the value
        raise _SaveError(at, f"a value has no JSON form ({type(error).__name__})") from None
    return written


def _value_type(values: list[Any]) -> str | None:
    """The name of the plain type a column's values have, for one whose JSON doesn't say it (a date, a decimal, a float that isn't finite)."""
    first = next((v for v in values if v is not None), None)
    for name, kind in _PLAIN:
        if isinstance(first, kind):
            return name
    return None


#: The plain types a column's values can be written as and read back from, the narrower first.
_PLAIN: tuple[tuple[str, type], ...] = (
    ("bool", bool), ("int", int), ("float", float), ("datetime", datetime.datetime), ("date", datetime.date),
    ("time", datetime.time), ("timedelta", datetime.timedelta), ("decimal", decimal.Decimal), ("str", str),
)
_COLUMN_OF = {name: carried_list(kind | None) for name, kind in _PLAIN}
_ANY_COLUMN = carried_list(Any)


def _dump_input(graph: Graph, address: str, value: Any) -> Any:
    """A caller's input as JSON: a table as a saved table output is, anything else by its field; a value with no JSON
    form raises ``_SaveError`` naming the input."""
    field = graph._input_fields[address]
    if field.spec.shape == "table":
        return None if value is None else _dump_frame(field.spec, Frame(value.height, value.to_dict(as_series=False)), address)
    if field.listed:
        return _written_as_json(field.spec.column_adapter, value, address)
    return _written_as_json(field.spec.column_adapter, [value], address)[0]  # written as a list is, a float that isn't finite kept


def _load_input(graph: Graph, address: str, saved: Any) -> Any:
    field = graph._input_fields[address]
    try:
        if field.spec.shape != "table":
            return (field.spec.list_adapter if field.listed else field.spec.adapter).validate_python(saved)
        return None if saved is None else _load_frame(field.spec, _SAVED_TABLE.validate_python(saved), address).frame()
    except ValidationError as error:
        raise _SaveError(address, value_free_errors(error)[0]["msg"]) from None


def _frame_json(value: Any) -> Any:
    if isinstance(value, pl.DataFrame):
        return value.to_dict(as_series=False)
    raise TypeError(type(value).__name__)


def jsonable(kwargs: dict[str, Any]) -> Any:
    """What a unit received, as JSON; a float that isn't finite as text. ``FieldValueError`` as ``question_unwritable`` names
    the first input with no JSON form."""
    try:
        return pydantic_core.from_json(pydantic_core.to_json(kwargs, fallback=_frame_json, inf_nan_mode="strings"))
    except (TypeError, ValueError, pydantic_core.PydanticSerializationError) as error:
        for name, value in kwargs.items():
            try:
                pydantic_core.to_json(value, fallback=_frame_json, inf_nan_mode="strings")
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
        written = pydantic_core.to_json([digest, kwargs], fallback=_frame_columns) if plain else canonical_json([digest, kwargs], fallback=_frame_columns)
    except (TypeError, ValueError, pydantic_core.PydanticSerializationError):
        return None
    return hashlib.blake2b(written, digest_size=12).hexdigest()


def canonical_json(value: Any, fallback: Callable[[Any], Any] | None = None) -> bytes:
    """``value`` as JSON to hash, with every object's keys sorted, so a value a JSON store gave back in another key order is written alike.

    Every hash whose keys may come back from a store is written here. A hash of values that hold no mapping (a reuse
    fingerprint of a node whose inputs can't, a question whose inputs are plain) is their plain JSON, hashed as it
    is: there are no keys to sort, and those hashes are made on every row.
    """
    plain = pydantic_core.to_jsonable_python(value, fallback=fallback)
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _frame_columns(value: Any) -> Any:
    """A table as its columns in order, for a fingerprint, where an object's keys are sorted."""
    if isinstance(value, pl.DataFrame):
        return [[c, s] for c, s in value.to_dict(as_series=False).items()]
    raise TypeError(type(value).__name__)


def _question_fingerprint(placed: PlacedNode, unit: WaitingUnit) -> str:
    """Eight hex digits of which node asks, as which code, and what it received, so a waiting id stops matching when its row holds another question."""
    inputs = unit.question.inputs
    values = [inputs[name] for name in sorted(inputs)]  # saved, so the names may come back from a store in any order
    hashed = [placed.id, placed.reuse_digest, values]
    written = canonical_json(hashed) if {type(v) for v in values} & {dict, list} else pydantic_core.to_json(hashed)  # plain values have no keys to sort
    return hashlib.blake2b(written, digest_size=4).hexdigest()


def _parse_unit(unit_id: str) -> tuple[str, Key | None, str | None]:
    """A waiting id's node, row (``None`` when it isn't one) and question fingerprint (``None`` when the id names only the place, ``node@row``)."""
    place, _, fingerprint = unit_id.partition("~")
    node_id, *rest = place.split("@")
    try:
        return node_id, tuple(int(i) for i in rest), fingerprint or None
    except ValueError:
        return node_id, None, fingerprint or None


def _deadline(timeout: float | None) -> Callable[[], EndedEarly | None] | None:
    if timeout is None:
        return None
    end = time.monotonic() + timeout
    return lambda: EndedEarly("timed_out", make_problem("run_timed_out", timeout=timeout)) if time.monotonic() > end else None

