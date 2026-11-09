"""``RunState``: everything a run is, as one record a host can store between legs.

A *leg* is one call of ``CompiledGraph.run``; a run takes several when a node
waits on a person. The state is what a leg ends on, what a checkpoint during
one hands over, and what the next one starts from. A unit in flight is in no
state: going on from one calls it again, so a unit runs at least once. It says
how the last leg ended (``status``, ``failure``) and, in ``nodes``, what each
node did: one ``WorkOfNode``. Everything else a run knows — which rows each
index has, which indexes will gain no more — follows from these and is worked
out again when a leg starts.

A state is the only source of reuse: a later leg re-planning a node reuses the
finished units in the state it is given, nothing else, so a state belongs to the
runner it came from. A leg that stops short hands on, in ``earlier_units``, what
it had not yet re-planned, so stopping doesn't lose it.

The module knows nothing of graphs, so a state loads with no registry in
sight and everything a form needs is on it. Its values are read through a
compiled graph: ``compiled.results(state)`` and ``compiled.outputs(state)``.

A state a leg just built holds its values typed; one read from JSON holds
their wire form, and the graph's validators read that when it is next used.
Dumped as JSON, each value is written by its own type, in one document that
holds each table once: under ``tables``, by the digest of what it holds, with
``{"table": digest}`` in every place that held it. Tables are read back
when a state is loaded, so a loaded state holds one frame per digest.
"""

from __future__ import annotations

from collections.abc import Mapping
from itertools import chain, islice
from typing import Any, Literal, Self

from pydantic import (
    Field,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    field_serializer,
    model_serializer,
    model_validator,
)

from conductor._internal.codec import TypedValue, each_to_jsonable
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import Problem
from conductor._internal.index import Row
from conductor._internal.ref import Ref
from conductor._internal.table import (
    tables_read_once,
    with_table_references_read,
    writing_each_table_once,
)

#: How the last leg ended. ``complete``: every node finished. ``waiting``: the
#: leg went quiet with units waiting on a person. ``failed``: a node failed.
#: ``cancelled`` and ``timed_out``: the host stopped the leg, or its deadline
#: passed. ``running``: not an ending, a state saved while its leg ran
#: (``RunCheckpoint``). A stored ``running`` state cannot tell a leg still alive
#: from one that crashed, so a host resuming from one keeps its own lease.
RunStatus = Literal["running", "complete", "waiting", "failed", "cancelled", "timed_out"]

#: A person's answers, by ``WaitingUnit.id``: each a value per output name, or
#: ``None`` for a unit that has no value on its row. Unit ids are opaque.
Answers = Mapping[str, Mapping[str, Any] | None]


class OutputColumn(ConductorModel):
    """What one output holds: a value per row, and the rows it was skipped at.

    ``rows`` and ``values`` line up, one value at the row of the unit that
    wrote it. On an output that makes rows, the value is the list of cells of the rows
    under that row. A ``Table`` is one value at its parent's row, its rows
    inside it. A row in ``skipped`` holds nothing, and neither does any row
    under it. A row one step deeper than a row in ``rows`` is a cell of that row's
    list skipped alone; a ``None`` stands in its place in the list.
    """

    rows: list[Row] = Field(default_factory=list)
    values: list[Any] = Field(default_factory=list)
    skipped: list[Row] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rows_line_up_with_values(self) -> Self:
        if len(self.rows) != len(self.values):
            raise ValueError(f"{len(self.rows)} rows for {len(self.values)} values")
        return self

    @field_serializer("values", when_used="json")
    def _values_as_wire(self, values: list[Any]) -> list[Any]:
        return _column_values_to_wire(values)


def _output_columns_to_wire(node_id: str, columns: Mapping[str, OutputColumn]) -> dict[str, Any]:
    """Each column of this node written by its own types, naming the output whose value has no JSON form."""
    written: dict[str, Any] = {}
    for name, column in columns.items():
        try:
            values = _column_values_to_wire(column.values)
        except Exception as unwritable:
            raise TypeError(f"{Ref(node_id, name)}: a value has no JSON form ({unwritable})") from unwritable
        written[name] = {"rows": column.rows, "values": values, "skipped": column.skipped}
    return written


def _column_values_to_wire(values: list[Any]) -> list[Any]:
    """A column's values, each written by its own type: a list holds a row's cells and is
    written cell by cell, all of them in one pass."""
    if not values or not all(type(value) is list for value in values):
        return each_to_jsonable(values)
    written = iter(each_to_jsonable(list(chain.from_iterable(values))))
    return [list(islice(written, len(value))) for value in values]


class Question(ConductorModel):
    """One thing a person is asked: an output of the waiting node.

    Plain data, so a form renders from a stored state with no graph loaded.
    ``dtype`` and ``widget`` are the dumps of the output's type and control;
    ``json_schema`` is the shape of one answer, for a client to check before
    sending. ``default`` fills the question when ``optional`` and left out; a
    state holds it as its cells are held, and writes it as JSON the same way.
    Questions sharing an ``exclusive_group`` are alternatives: the person answers exactly one.
    """

    name: str
    title: str
    description: str | None = None
    dtype: dict[str, Any] | None
    #: The answer is a list of values of ``dtype``, one row each.
    many: bool = False
    json_schema: dict[str, Any]
    widget: dict[str, Any] | None = None
    default: TypedValue = None
    optional: bool = False
    exclusive_group: str | None = None


class WaitingUnit(ConductorModel):
    """One unit waiting on a person: a node, on one row when it runs per row.

    ``id`` is what an answer is keyed by: ``node@row#hash``, the row and the
    first twelve characters of ``input_hash``, so an answer meant for a row that
    has since moved or changed is refused, never written on another row. Nothing
    ever reads it apart. ``questions`` are frozen until the unit is answered, or
    the inputs of its row change. ``inputs`` says what the row is about. A unit in
    a state refers to its tables as the state's values do; one on the event
    stream carries them, since an event stands alone.
    """

    id: str
    node_id: str
    row: Row
    prompt: str | None = None
    questions: tuple[Question, ...]
    #: The hash of what this unit received, for a later leg re-planning its node.
    input_hash: str = ""
    #: What this unit received, by input name, so a form can show the row it is about: exactly the
    #: values its call would have got, no more; many values as a list.
    inputs: dict[str, TypedValue] = Field(default_factory=dict)


#: A finished unit: its row and the hash of what it received, ``None`` for one that wasn't called.
FinishedUnit = tuple[Row, str | None]


class EarlierUnit(ConductorModel):
    """A unit of a re-planned node a later leg can still reuse, at the row it really had.

    What a finished or waiting unit left behind, one record each, so units of
    several legs can sit in one list, two of them at the same row. ``outputs``
    holds one value per output at this row, read and written as a state's
    columns are; empty on a waiting unit."""

    row: Row
    input_hash: str
    answered: bool = False
    #: Inputs alone can't say which of several rows' answers this was, so only
    #: its own row may take it: a person's answer has no second chance.
    only_its_own_row_may_claim: bool = False
    waiting: WaitingUnit | None = None
    outputs: dict[str, OutputColumn] = Field(default_factory=dict)


def id_of_waiting_unit(node_id: str, row: Row, input_hash: str) -> str:
    """The ``WaitingUnit.id`` of this unit: the one place it is spelled."""
    at_row = f"@{','.join(str(n) for n in row)}" if row else ""
    return f"{node_id}{at_row}#{input_hash[:12]}"


class WorkOfNode(ConductorModel):
    """What one node did in a run: everything a state knows about it, so a stale
    node is one key.

    ``finished_units`` is ``None`` while nothing of the node has finished, and
    ``[]`` for a node complete on no rows; a node that ran once has the one row ``()``."""

    #: ``CompiledNode.fingerprint`` when it last ran: what tells a later leg
    #: over an edited graph that the node changed.
    fingerprint: str | None = None
    #: Each finished unit's row and hash, in row order.
    finished_units: list[FinishedUnit] | None = None
    #: The finished rows a person answered.
    answered_rows: list[Row] = Field(default_factory=list)
    #: What each output holds, by output name.
    outputs: dict[str, OutputColumn] = Field(default_factory=dict)
    #: Units of this node waiting on a person, in row order.
    waiting: tuple[WaitingUnit, ...] = ()
    #: How many rows it wasn't called on, per input that was blank there.
    rows_not_called: dict[str, int] = Field(default_factory=dict)
    #: How many of its units this leg took from earlier work.
    reused_units: int = 0
    #: What a later leg may still reuse of the units this leg has not re-planned yet.
    #: A leg that stopped short (failed, cancelled, a checkpoint, waiting on a
    #: person) hands them on, so going on from it reuses what going on from
    #: the state before would.
    earlier_units: list[EarlierUnit] = Field(default_factory=list)

    def _with_values_as_wire(self, node_id: str) -> dict[str, Any]:
        """This record with the fields that hold values written, leaving the bookkeeping for the JSON writer."""
        return {
            **{name: getattr(self, name) for name in type(self).model_fields},
            "outputs": _output_columns_to_wire(node_id, self.outputs),
            "waiting": [unit.model_dump(mode="json") for unit in self.waiting],
            "earlier_units": [{**unit.model_dump(mode="json", exclude={"outputs"}), "outputs": _output_columns_to_wire(node_id, unit.outputs)} for unit in self.earlier_units],
        }

    def _with_tables_read(self, frames: Mapping[str, Any]) -> WorkOfNode:
        return self.model_copy(update={name: with_table_references_read(getattr(self, name), frames) for name in ("outputs", "waiting", "earlier_units")})


class RunState(ConductorModel):
    """A run, as of the end of its last leg or of a checkpoint during one.

    ``compiled.run(state)`` goes on from one, ``compiled.answer(state, ...)``
    returns a new one with answers written in, and ``without_nodes`` returns one
    that forgets some nodes. Nothing changes one in place.

    A state goes back only to the runner it came from: a leg takes whatever
    earlier results it holds, so a state from another run's values would hand
    those to this one (``FromRun`` values, a person's identity, are not hashed).

    Its JSON has ``tables``, each table the state holds, by digest (see the
    module docstring), so a dump is one self-contained document.
    """

    #: The tables of a JSON document, by digest. Read back into the values that
    #: refer to them when a state is loaded, so a state in memory never holds any here.
    tables: dict[str, Any] = Field(default_factory=dict)
    status: RunStatus
    #: Why the leg stopped short: the failed node's cause, or a host's stop.
    #: ``None`` when the leg went quiet by itself (``complete``, ``waiting``).
    failure: Problem | None = None
    #: What each node did, by node id, in execution order.
    nodes: dict[str, WorkOfNode] = Field(default_factory=dict)

    @property
    def waiting(self) -> tuple[WaitingUnit, ...]:
        """Every unit waiting on a person, in execution order and row order within a node."""
        return tuple(chain.from_iterable(work.waiting for work in self.nodes.values()))

    @model_serializer(mode="wrap")
    def _each_table_once(self, write_fields: SerializerFunctionWrapHandler, info: SerializationInfo):  # type: ignore[no-untyped-def]  # a return type here would become the record's published schema
        """As JSON, one document that has in ``tables`` what its values refer to."""
        if not info.mode_is_json():
            return write_fields(self)
        with writing_each_table_once() as tables:
            document = write_fields(self) if info.include or info.exclude or info.exclude_none or info.exclude_unset or info.exclude_defaults else self._document_written_in_one_pass()
        return {**document, "tables": tables}

    def _document_written_in_one_pass(self) -> dict[str, Any]:
        """Every field, with those that hold values written here, so their tables go in the one document.

        The bookkeeping fields stay as they are for the JSON writer, which is what keeps a dump of a state with no table as fast as one field at a time."""
        return {
            **{name: getattr(self, name) for name in type(self).model_fields},
            "nodes": {node_id: work._with_values_as_wire(node_id) for node_id, work in self.nodes.items()},
        }

    @model_validator(mode="after")
    def _tables_read_back(self) -> Self:
        """A document's ``tables`` read once each, and put where its values refer to them."""
        if self.tables:
            frames = tables_read_once(self.tables)
            object.__setattr__(self, "tables", {})
            object.__setattr__(self, "nodes", {node_id: work._with_tables_read(frames) for node_id, work in self.nodes.items()})
        return self

    def without_nodes(self, *node_ids: str) -> RunState:
        """This state as if these nodes had never run: a host's "run from here".

        What they hold goes, which is what makes the next leg run them again. A
        node reading one of them is re-planned, and reused where what it receives
        came out the same: to run a node and everything after it again, name
        them all. A node id also covers the nodes of a graph embedded under it
        (``approve`` covers ``approve/check``).
        """

        def is_dropped(node_id: str) -> bool:
            return any(node_id == dropped or node_id.startswith(f"{dropped}/") for dropped in node_ids)

        return self.model_copy(update={"nodes": {node_id: work for node_id, work in self.nodes.items() if not is_dropped(node_id)}})
