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
import hashlib
import json
import threading
import time
from collections.abc import AsyncIterator, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import polars as pl
import pydantic_core
from pydantic import TypeAdapter, ValidationError

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import InvalidCallError, Problem, make_problem, value_free_errors
from conductor._internal.fields import SKIPPED, FieldSpec, carried_list

if TYPE_CHECKING:
    from conductor._internal.graph import Graph, Level, PlacedNode, Source
    from conductor._internal.leg import Event

#: A row: the parent's key plus a number, ``()`` for the graph's own row.
Key = tuple[int, ...]

#: How a run stands after a leg.
RunStatus = Literal["ready", "running", "complete", "waiting", "failed", "stopped", "cancelled", "timed_out"]


class _Pending:
    def __repr__(self) -> str:
        return "PENDING"


#: What a unit's inputs are while something above it hasn't run yet.
PENDING: Any = _Pending()


@dataclass(frozen=True)
class Frame:
    """A table as a run keeps it: columns of plain values, so a cell is a list index."""

    rows: int
    columns: dict[str, list[Any]]

    def frame(self) -> pl.DataFrame:
        return pl.DataFrame(self.columns, strict=False) if self.columns else pl.DataFrame()


@dataclass(frozen=True, slots=True)
class Unit:
    """What one node came to on one row."""

    state: Literal["done", "skipped", "blank", "waiting"]
    out: dict[str, Any] | None = None
    #: For a waiting unit, what the person is asked; for a blank one, ``{"input": name}``.
    question: dict[str, Any] | None = None


class Waiting(ConductorModel):
    """A row of a node waiting for a person.

    ``id`` is what ``run.answer`` takes: the node's id, then ``@`` and each
    place of the row, ``"approve@1"``. ``prompt``, ``choices``, ``default`` and
    ``widget`` are what the node's ``Ask`` said; ``answer_schema`` is the JSON
    schema of the answer, from the node's return type; ``inputs`` is what the
    node received, as JSON.
    """

    id: str
    node_id: str
    row: tuple[int, ...]
    prompt: str
    choices: Any = None
    default: Any = None
    widget: Any = None
    answer_schema: dict[str, Any]
    inputs: dict[str, Any]


class Run:
    """One run of a graph. Every call returns a new ``Run``; ``to_json`` is all of it.

    ``status`` is ``ready`` before the first leg, then ``complete``,
    ``waiting``, ``failed``, ``stopped`` (the row ceiling), ``cancelled`` or
    ``timed_out``; ``cause`` is the ``Problem`` behind any of the last four.
    """

    def __init__(self, graph: Graph, inputs: dict[str, Any], *, max_rows: int | None = None) -> None:
        """A run of ``graph`` on checked ``inputs``, by address, not started: what ``graph.with_inputs`` gives."""
        self.graph, self.inputs, self.max_rows = graph, inputs, max_rows
        self._units: dict[str, dict[Key, Unit]] = {n: {} for n in graph._nodes}
        self.status: RunStatus = "ready"
        self.cause: Problem | None = None
        #: node -> fingerprint of what a unit received -> its outputs, from the run this one went on from.
        self._earlier: dict[str, dict[str, dict[str, Any]]] = {}

    def __repr__(self) -> str:
        return f"<Run {self.status}: {sum(len(u) for u in self._units.values())} units>"

    # -- going on ---------------------------------------------------------------

    def resume(self, *, max_rows: int | None = None, from_run: Mapping[type, Any] | None = None, timeout: float | None = None) -> Run:
        """Go on: start every unit whose inputs exist, until the run is done, waits, fails, reaches its ceiling or its ``timeout``.

        ``max_rows`` raises (or sets) the ceiling a stopped run carries on
        under. ``from_run`` gives each ``FromRun`` type its value.
        """
        checked_max_rows(max_rows)
        return self._leg(max_rows, from_run, None, _deadline(timeout))

    def stream(
        self, *, max_rows: int | None = None, from_run: Mapping[type, Any] | None = None, timeout: float | None = None, cancel: asyncio.Event | None = None,
    ) -> AsyncIterator[Event]:
        """``resume``, saying what happens as it happens: ``async for event in run.stream(): ...``.

        The leg runs on its own thread, so a slow node never holds the event
        loop. The last event is ``RunEnd``, whose ``run`` is the run as the leg
        left it. Setting ``cancel``, or leaving the loop early, ends the leg
        before its next unit. A call that can't start raises here, before
        anything is iterated.
        """
        checked_max_rows(max_rows)
        self._check_from_run(from_run)
        return self._stream(max_rows, from_run, timeout, cancel)

    async def _stream(self, max_rows: int | None, from_run: Mapping[type, Any] | None, timeout: float | None, cancel: asyncio.Event | None) -> AsyncIterator[Event]:
        loop, events = asyncio.get_running_loop(), asyncio.Queue["Event | None"]()
        cancelled = threading.Event()
        deadline = _deadline(timeout)

        def stop() -> Problem | None:
            if cancelled.is_set() or (cancel is not None and cancel.is_set()):
                return make_problem("run_cancelled")
            return deadline() if deadline is not None else None

        failed: list[BaseException] = []

        def leg() -> None:
            try:
                self._leg(max_rows, from_run, lambda event: _hand_over(loop, events, event), stop)
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

    def _check_from_run(self, from_run: Mapping[type, Any] | None) -> None:
        missing = [
            make_problem("from_run_missing", p.id, name, needs=needed.__name__, parameter=name)
            for p in self.graph._nodes.values() for name, needed in p.definition.from_run.items() if needed not in (from_run or {})
        ]
        if missing:
            raise InvalidCallError(missing)

    def _leg(self, max_rows: int | None, from_run: Mapping[type, Any] | None, emit: Any, stop: Any) -> Run:
        from conductor._internal.leg import Leg

        self._check_from_run(from_run)
        run = self._copy(max_rows=max_rows if max_rows is not None else self.max_rows)
        Leg(run, emit, stop, dict(from_run or {})).go()
        return run

    def answer(self, unit: Waiting | str, value: Any) -> Run:
        """Give a waiting row its answer, checked against what the node returns. Nothing runs until ``resume``."""
        unit_id = unit.id if isinstance(unit, Waiting) else unit
        node_id, key = _parse_unit(unit_id)
        found = self._units.get(node_id, {}).get(key) if node_id in self.graph._nodes else None
        if found is None or found.state != "waiting":
            raise InvalidCallError([make_problem("not_waiting", node_id, *key, waiting_id=unit_id)])
        try:
            out = _answer_outputs(self.graph._nodes[node_id], value, found.question or {})
        except ValidationError as error:
            raise InvalidCallError([make_problem("invalid_answer", node_id, *key, reason=value_free_errors(error)[0]["msg"])]) from None
        except (ValueError, TypeError) as error:
            raise InvalidCallError([make_problem("invalid_answer", node_id, *key, reason=str(error))]) from None
        run = self._copy()
        run._units[node_id] = {**self._units[node_id], key: Unit("done", out)}
        return run

    def with_inputs(self, inputs: Mapping[str, Any] | None = None, /, *, max_rows: int | None = None, **by_name: Any) -> Run:
        """A run of these changed inputs, the rest as before, ready to ``resume`` or ``stream``.

        It carries this run's work: a row that receives what an earlier unit of
        its node received takes that unit's outputs, answers included, and the
        node is not called for it. Inputs are given as ``graph.with_inputs``
        takes them, and checked the same way.
        """
        given = {name: self.inputs[graph_input.key] for name, graph_input in self.graph.inputs.items() if graph_input.key in self.inputs}
        fresh = self.graph.with_inputs({**given, **(inputs or {}), **by_name}, max_rows=max_rows if max_rows is not None else self.max_rows)
        earlier: dict[str, dict[str, dict[str, Any]]] = {}
        for n, units in self._units.items():
            placed = self.graph._nodes[n]
            if placed.definition.policy.reuse_when_inputs_match:
                by = earlier[n] = {}
                for k, u in units.items():
                    fingerprint = _fingerprint(placed.definition.version, self._inputs(placed, k)) if u.state == "done" and u.out is not None else None
                    if fingerprint is not None:
                        by[fingerprint] = u.out  # type: ignore[assignment]
        fresh._earlier = earlier
        return fresh

    def again(
        self, inputs: Mapping[str, Any] | None = None, /, *, max_rows: int | None = None, from_run: Mapping[type, Any] | None = None,
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
        return [self._waiting(n, k, u) for n, units in self._units.items() for k, u in units.items() if u.state == "waiting"]

    def _waiting(self, node_id: str, key: Key, unit: Unit) -> Waiting:
        return Waiting(id=_unit_id(node_id, key), node_id=node_id, row=key, answer_schema=_answer_schema(self.graph._nodes[node_id]), **(unit.question or {}))

    @property
    def not_called(self) -> list[dict[str, Any]]:
        """Rows a node wasn't called on because a plain input was blank: which node, row and input."""
        return [
            {"node_id": self.graph._outer_id(n), "row": k, "input": self.graph._outer.get(f"{n}.{u.question['input']}", f".{u.question['input']}").partition(".")[2]}
            for n, units in self._units.items() for k, u in units.items() if u.state == "blank" and u.question
        ]

    @property
    def results(self) -> dict[str, dict[str, Any]]:
        """Each node's outputs: one value on a node that ran once, else a list in row order, ``None`` where a row has none.

        An embedded graph is one node here, with the outputs it offers.
        """
        found = {n: {o: self._collect(p, o) for o in p.definition.outputs} for n, p in self.graph._nodes.items() if "/" not in n}
        for e, (_, outs) in self.graph._offers.items():
            found[e] = {
                name: self._collect(self.graph._nodes[a.partition(".")[0]], a.partition(".")[2]) for name, a in outs.items() if a.partition(".")[0] in self.graph._nodes
            }
        return found

    @property
    def outputs(self) -> dict[str, Any]:
        """The graph's outputs (``graph.outputs``), by address, once the run is complete; ``results`` reads a run in any state."""
        if self.status != "complete":
            reason = self.cause.msg if self.cause is not None else ""
            raise InvalidCallError([make_problem("run_not_complete", "outputs", status=self.status, reason=reason)])
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

    def _reused(self, placed: PlacedNode, kwargs: dict[str, Any]) -> dict[str, Any] | None:
        earlier = self._earlier.get(placed.id)
        if not earlier:
            return None
        fingerprint = _fingerprint(placed.definition.version, kwargs)
        return earlier.get(fingerprint) if fingerprint is not None else None

    def _call(self, placed: PlacedNode, kwargs: dict[str, Any], from_run: Mapping[type, Any]) -> tuple[Any, BaseException | None]:
        try:
            if placed.definition.from_run:
                kwargs = {**kwargs, **{name: from_run[needed] for name, needed in placed.definition.from_run.items()}}
            return _outputs(placed, placed.definition.function(**kwargs)), None  # type: ignore[misc]
        except BaseException as error:  # noqa: BLE001  a node's own failure, or its Ask, is the run's to record
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return None, error

    def _outputs_json(self, placed: PlacedNode) -> dict[str, Any]:
        """A node's outputs as JSON, by its own types: one value, or a list in row order; a float that isn't finite as text."""
        dumped = {name: _dump(spec, self._collect(placed, name), bool(placed.level)) for name, spec in placed.definition.outputs.items()}
        written: dict[str, Any] = json.loads(pydantic_core.to_json(dumped, inf_nan_mode="strings"))
        return written

    def _stop(self, table: str, count: int) -> None:
        self.status = "stopped"
        self.cause = make_problem("row_ceiling", self.graph._makers[table][0] or "inputs", table=table, rows=count, max_rows=self.max_rows)

    def _rows_made(self) -> dict[str, int]:
        made: dict[str, int] = {}
        for table, (node_id, output) in self.graph._makers.items():
            if node_id is None:
                made[table] = len(self.inputs.get(output, []))
            else:
                made[table] = sum(_length(u.out[output]) for u in self._units[node_id].values() if u.state == "done" and u.out)
        return made

    def _rows(self, level: Level, row: Key) -> tuple[list[Key], bool]:
        """The rows of ``level`` under ``row`` (or the one above it), and whether all of them exist yet."""
        if len(row) >= len(level):
            return [row[: len(level)]], True
        keys, complete = [row], True
        for table in level[len(row):]:
            node_id, output = self.graph._makers[table]
            below: list[Key] = []
            for parent in keys:
                if node_id is None:
                    count: int | None = len(self.inputs[output])
                else:
                    unit = self._units[node_id].get(parent)
                    if unit is None or unit.state == "waiting":
                        count = None
                    else:
                        count = _length(unit.out.get(output)) if unit.state == "done" and unit.out else 0
                if count is None:
                    complete = False
                    continue
                below.extend((*parent, i) for i in range(count))
            keys = below
        return keys, complete

    def _skipped_above(self, source: Source, row: Key) -> bool:
        """Whether the table ``source`` reads under ``row`` was never made because its maker took no branch there."""
        if len(row) >= len(source.level) or source.node_id is None:
            return False
        node_id, _ = self.graph._makers[source.level[len(row)]]
        unit = self._units[node_id].get(row) if node_id is not None else None
        return unit is not None and unit.state == "skipped"

    def _cell(self, source: Source, key: Key) -> Any:
        if source.node_id is None:
            value = self.inputs[source.output]
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
        """The keyword arguments for ``placed`` on ``row``; or PENDING, SKIPPED, or the name of a blank input."""
        kwargs: dict[str, Any] = {}
        gathered: dict[str, list[tuple[Level, Key, Any]]] = {}
        for name, plan in placed.inputs.items():
            spec = plan.spec
            if not plan.sources:
                kwargs[name] = plan.value
            elif spec.shape == "one":
                value = SKIPPED
                for source in plan.sources:
                    value = self._cell(source, row[: len(source.level)])
                    if value is PENDING:
                        return PENDING
                    if value is not SKIPPED:
                        break
                if value is SKIPPED:
                    return SKIPPED
                if value is None and not spec.optional:
                    return name
                kwargs[name] = value
            elif spec.shape == "many":
                if len(plan.sources) == 1 and plan.sources[0].node_id is None and self.inputs[plan.sources[0].output] is None:
                    kwargs[name] = None  # a Series that may be None, and nothing was given
                    continue
                items: list[tuple[Level, Key, Any]] = []
                skipped_above = 0
                for source in plan.sources:
                    if source.node_id is None and not source.made:  # the caller's list, given whole
                        items.extend(((source.output,), (i,), v) for i, v in enumerate(self.inputs[source.output]))
                        continue
                    keys, complete = self._rows(source.level, row)
                    if not complete:
                        return PENDING
                    if not keys and self._skipped_above(source, row):
                        skipped_above += 1
                    for key in keys:
                        value = self._cell(source, key)
                        if value is PENDING:
                            return PENDING
                        items.append((source.level, key, None if value is SKIPPED else value))
                if skipped_above and skipped_above == len(plan.sources):
                    return SKIPPED  # every source was a branch not taken above this row: so is the call
                gathered[name] = items
            else:
                table = self._table(plan.sources, plan.table_level, row, spec.columns)
                if table is PENDING:
                    return PENDING
                kwargs[name] = table
        if gathered:
            # A row a Series input can't take leaves every Series input on that table.
            blank = {name: {(lvl, k) for lvl, k, v in items if v is None} for name, items in gathered.items() if not placed.inputs[name].spec.optional}
            for name, items in gathered.items():
                others = [rows for other, rows in blank.items() if other != name]
                keep_blank = placed.inputs[name].spec.optional
                kwargs[name] = [v for lvl, k, v in items if (keep_blank or v is not None) and not any((lvl, k) in rows for rows in others)]
        return kwargs

    def _table(self, sources: list[Source], level: Level, row: Key, declared: Mapping[str, Any] | None = None) -> Any:
        """The table a ``Table`` input receives on ``row``: its sources' columns on the rows of ``level``, only the declared ones when it names them."""
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
        if declared is not None:
            columns = {c: columns.get(c, [None] * len(keys)) for c in declared}
        return pl.DataFrame(columns, strict=False)

    # -- JSON -------------------------------------------------------------------

    def to_json(self) -> str:
        """The run as JSON: its inputs, every unit, and what it went on from, a column per output of each node."""
        units = {n: _dump_units(self.graph._nodes[n], us) for n, us in self._units.items()}
        earlier = {
            n: {"fingerprints": list(by), **_dump_units(self.graph._nodes[n], {(i,): Unit("done", out) for i, out in enumerate(by.values())}, keyed=False)}
            for n, by in self._earlier.items()
        }
        inputs = {address: _input_adapter(self.graph, address).dump_python(v, mode="json") for address, v in self.inputs.items()}
        cause = self.cause.model_dump(mode="json") if self.cause else None
        return pydantic_core.to_json({"status": self.status, "cause": cause, "max_rows": self.max_rows, "inputs": inputs, "units": units, "earlier": earlier}, inf_nan_mode="strings").decode()

    def _copy(self, **changes: Any) -> Run:
        copy = Run(self.graph, self.inputs, max_rows=changes.get("max_rows", self.max_rows))
        copy._units = {n: dict(us) for n, us in self._units.items()}
        copy.status, copy.cause, copy._earlier = self.status, self.cause, self._earlier
        return copy


def run_from_json(graph: Graph, saved: str | bytes | Mapping[str, Any]) -> Run:
    """A run of ``graph`` from what ``run.to_json()`` wrote; ``graph.load_run`` is the door to it."""
    try:
        data = pydantic_core.from_json(saved) if isinstance(saved, (str, bytes)) else saved
        inputs = {address: _input_adapter(graph, address).validate_python(v) for address, v in data["inputs"].items()}
        run = Run(graph, inputs, max_rows=data["max_rows"])
        for n, dumped in data["units"].items():
            run._units[n] = {tuple(k): u for k, u in _load_units(graph._nodes[n], dumped)}
        run._earlier = {
            n: {fp: u.out for fp, (_, u) in zip(dumped["fingerprints"], _load_units(graph._nodes[n], dumped), strict=True) if u.out is not None}
            for n, dumped in data["earlier"].items()
        }
        run.status = data["status"]
        run.cause = Problem.model_validate(data["cause"]) if data["cause"] else None
        return run
    except (KeyError, TypeError, ValueError, StopIteration) as error:
        reason = value_free_errors(error)[0]["msg"] if isinstance(error, ValidationError) else f"{type(error).__name__} {error}"
        raise InvalidCallError([make_problem("state_unreadable", "state", reason=reason)]) from None


def checked_max_rows(max_rows: int | None) -> None:
    """A ceiling below one row is refused at the call."""
    if max_rows is not None and (not isinstance(max_rows, int) or max_rows < 1):
        raise InvalidCallError([make_problem("invalid_max_rows", "max_rows", max_rows=max_rows)])


# -- values -------------------------------------------------------------------------------------


def _outputs(placed: PlacedNode, returned: Any) -> Any:
    spread = placed.definition.spread(returned)
    if spread is SKIPPED:
        return SKIPPED
    for name, spec in placed.definition.outputs.items():
        value = spread[name]
        if value is SKIPPED:
            continue
        if spec.shape == "table":
            if not isinstance(value, pl.DataFrame):
                raise TypeError(f"{name} should be a polars DataFrame, got {type(value).__name__}")
            missing = sorted((spec.columns or {}).keys() - set(value.columns))
            if missing:
                raise TypeError(f"{name} lacks the columns {missing} its row type declares")
            spread[name] = Frame(value.height, value.to_dict(as_series=False))
        elif spec.shape == "many":
            spread[name] = list(value)  # a SKIPPED item is a row whose branch wasn't taken
    return spread


def _length(value: Any) -> int:
    return 0 if value is SKIPPED or value is None else value.rows if isinstance(value, Frame) else len(value)


def _answer_outputs(placed: PlacedNode, value: Any, question: dict[str, Any]) -> dict[str, Any]:
    definition = placed.definition
    if question.get("choices") is not None:
        several = len(definition.outputs) == 1 and next(iter(definition.outputs.values())).shape == "many"
        if several and not isinstance(value, list):
            raise ValueError("pick a list of the choices")
        for picked in value if several else [value]:
            if picked not in _choice_ids(question["choices"]):
                raise ValueError("the answer is not one of the choices")
    if "result" in definition.outputs and len(definition.outputs) == 1:
        spec = definition.outputs["result"]
        if spec.shape == "table":
            return _outputs(placed, pl.DataFrame(TypeAdapter(list[dict[str, Any]]).validate_python(value)))
        return _outputs(placed, spec.adapter.validate_python(value))
    return _outputs(placed, TypeAdapter(definition.returns).validate_python(value))


def _choice_ids(choices: Any) -> list[Any]:
    if isinstance(choices, Mapping):
        return list(choices.keys())
    return [getattr(choice, "id", choice) for choice in choices]


def _answer_schema(placed: PlacedNode) -> dict[str, Any]:
    definition = placed.definition
    if len(definition.outputs) == 1 and "result" in definition.outputs:
        spec = definition.outputs["result"]
        if spec.shape == "table":
            return {"type": "array", "items": {"type": "object", "properties": spec.describe()["columns"]}}
        schema: dict[str, Any] = spec.adapter.json_schema()
        return schema
    whole: dict[str, Any] = TypeAdapter(definition.returns).json_schema()
    return whole


def _dump_units(placed: PlacedNode, units: Mapping[Key, Unit], keyed: bool = True) -> dict[str, Any]:
    """Units as columns: keys, states, questions, and one list per output."""
    done = [u for u in units.values() if u.state == "done" and u.out is not None]
    outputs: dict[str, Any] = {}
    for name, spec in placed.definition.outputs.items():
        values = [u.out[name] for u in done]  # type: ignore[index]
        skipped = [i for i, v in enumerate(values) if v is SKIPPED]
        if spec.shape == "table":
            dumped: Any = [
                None if v is SKIPPED else {"rows": v.rows, "columns": {c: _column_adapter(spec, c).dump_python(col, mode="json") for c, col in v.columns.items()}}
                for v in values
            ]
        else:
            plain = [None if v is SKIPPED else [None if i is SKIPPED else i for i in v] if spec.shape == "many" else v for v in values]
            dumped = spec.column_adapter.dump_python(plain, mode="json")
        items = {str(n): [i for i, item in enumerate(v) if item is SKIPPED] for n, v in enumerate(values) if spec.shape == "many" and v is not SKIPPED}
        outputs[name] = {"values": dumped, "skipped": skipped, **({"skipped_items": {k: v for k, v in items.items() if v}} if any(items.values()) else {})}
    out: dict[str, Any] = {"states": [u.state for u in units.values()], "questions": {str(i): u.question for i, u in enumerate(units.values()) if u.question}, "outputs": outputs}
    if keyed:
        out["keys"] = [list(k) for k in units]
    return out


def _load_units(placed: PlacedNode, dumped: Mapping[str, Any]) -> list[tuple[Key, Unit]]:
    states, questions = dumped["states"], dumped["questions"]
    columns: dict[str, list[Any]] = {}
    for name, column in dumped["outputs"].items():
        spec = placed.definition.outputs[name]
        if spec.shape == "table":
            values = [
                None if v is None else Frame(v["rows"], {c: _column_adapter(spec, c).validate_python(col) for c, col in v["columns"].items()})
                for v in column["values"]
            ]
        else:
            values = spec.column_adapter.validate_python(column["values"])
        for i in column["skipped"]:
            values[i] = SKIPPED
        for n, positions in column.get("skipped_items", {}).items():
            for i in positions:
                values[int(n)][i] = SKIPPED  # type: ignore[index]
        columns[name] = values
    keys = dumped.get("keys") or [[] for _ in states]
    loaded, n = [], 0
    for i, (key, state) in enumerate(zip(keys, states, strict=True)):
        out = None
        if state == "done":
            out = {name: values[n] for name, values in columns.items()}
            n += 1
        loaded.append((tuple(key), Unit(state, out, questions.get(str(i)))))
    return loaded


def _column_adapter(spec: FieldSpec, column: str) -> TypeAdapter[Any]:
    """Checks and writes one column of a table output: by the column's type where the row type names it."""
    return spec.column_adapters.get(column) or _ANY_COLUMN


_ANY_COLUMN = carried_list(Any)


def _input_adapter(graph: Graph, address: str) -> TypeAdapter[Any]:
    graph_input = next(p for p in graph.inputs.values() if p.key == address)
    return carried_list(graph_input.spec.type) if graph_input.many else graph_input.spec.adapter


def _frame_json(value: Any) -> Any:
    if isinstance(value, pl.DataFrame):
        return value.to_dict(as_series=False)
    raise TypeError(type(value).__name__)


def jsonable(kwargs: dict[str, Any]) -> Any:
    """What a unit received, as JSON; a float that isn't finite as text."""
    return json.loads(pydantic_core.to_json(kwargs, fallback=_frame_json, inf_nan_mode="strings"))


def _hand_over(loop: asyncio.AbstractEventLoop, events: asyncio.Queue[Any], event: Any) -> None:
    """Put an event on the listener's queue from the leg's thread; a listener whose loop has closed is gone, and hears nothing."""
    try:
        loop.call_soon_threadsafe(events.put_nowait, event)
    except RuntimeError:
        pass


def _fingerprint(version: int, kwargs: dict[str, Any]) -> str | None:
    """What a unit received, as a short hash; ``None`` for a value with no JSON form, which is never reused."""
    try:
        written = pydantic_core.to_json([version, kwargs], fallback=_frame_json)
    except (TypeError, ValueError, pydantic_core.PydanticSerializationError):
        return None
    return hashlib.blake2b(written, digest_size=12).hexdigest()


def _unit_id(node_id: str, key: Key) -> str:
    return node_id + "".join(f"@{i}" for i in key)


def _parse_unit(unit_id: str) -> tuple[str, Key]:
    node_id, *rest = unit_id.split("@")
    try:
        return node_id, tuple(int(i) for i in rest)
    except ValueError:
        return node_id, ()


def _dump(spec: FieldSpec, value: Any, many_rows: bool) -> Any:
    values = value if many_rows else [value]
    if spec.shape == "table":
        dumped = [None if v is None else v.to_dict(as_series=False) for v in values]
    else:
        dumped = spec.column_adapter.dump_python(values, mode="json")
    return dumped if many_rows else dumped[0]


def _deadline(timeout: float | None) -> Callable[[], Problem | None] | None:
    if timeout is None:
        return None
    end = time.monotonic() + timeout
    return lambda: make_problem("run_timed_out", timeout=timeout) if time.monotonic() > end else None

