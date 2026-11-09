"""``RunState``: everything a run is, as one record a host can store between legs.

A *leg* is one call of ``CompiledGraph.run``; a run takes several when a node
waits on a person. The state is what a leg ends on, what a checkpoint during
one hands over, and what the next one starts from. A unit in flight is in no
state: going on from one calls it again, so a unit runs at least once. It says how the last leg ended (``status``, ``failure``), who is
waiting and what they were asked (``waiting``), what every output holds
(``output_columns``), which units finished (``finished_rows``), what each of them
received (``input_hashes``), a hash of how the graph used each node
(``node_fingerprints``), and how much this leg took from earlier work
(``reused_units``). Everything else a run knows — which rows each index has,
which indexes will gain no more — follows from these and is worked out again
when a leg starts.

A state is the only source of reuse: a later leg re-planning a node reuses the
finished units in the state it is given, nothing else, so a state belongs to the
runner it came from. ``earlier_work`` keeps what a leg that stopped short had not
yet re-planned, so stopping doesn't lose it.

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

from collections.abc import Callable, Mapping
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

from conductor._internal.codec import to_wire, values_to_wire
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


def _output_columns_to_wire(columns: dict[Ref, OutputColumn]) -> dict[str, Any]:
    """Each column's values written by their own types, naming the output whose value has no JSON form."""
    written: dict[str, Any] = {}
    for ref, column in columns.items():
        try:
            values = _column_values_to_wire(column.values)
        except Exception as unwritable:
            raise TypeError(f"{ref}: a value has no JSON form ({unwritable})") from unwritable
        written[ref] = {"rows": column.rows, "values": values, "skipped": column.skipped}
    return written


def _column_values_to_wire(values: list[Any]) -> list[Any]:
    """A column's values, each written by its own type: a list holds a row's cells and is
    written cell by cell, all of them in one pass."""
    if not values or not all(type(value) is list for value in values):
        return values_to_wire(values)
    written = iter(values_to_wire(list(chain.from_iterable(values))))
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
    default: Any = None
    optional: bool = False
    exclusive_group: str | None = None

    @field_serializer("default", when_used="json")
    def _default_as_wire(self, default: Any) -> Any:
        """The default written by its own type, each value of a list by its own."""
        return [to_wire(item, type(item)) for item in default] if isinstance(default, list) else to_wire(default, type(default))


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
    inputs: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("inputs", when_used="json")
    def _inputs_as_wire(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Each input written by its own type; held typed until a dump, as a state's values are."""
        return dict(zip(inputs, values_to_wire(list(inputs.values())), strict=True))


class EarlierWork(ConductorModel):
    """What a re-planned node's earlier units still hold, for a later leg to reuse: nodes
    a leg has not finished, the finished and waiting units none of its units claimed yet.

    The same shape as the same fields of a state, at the rows those units had."""

    finished_rows: dict[str, list[Row]] = Field(default_factory=dict)
    input_hashes: dict[str, list[str | None]] = Field(default_factory=dict)
    answered_rows: dict[str, list[Row]] = Field(default_factory=dict)
    output_columns: dict[Ref, OutputColumn] = Field(default_factory=dict)
    waiting: tuple[WaitingUnit, ...] = ()

    @field_serializer("output_columns", when_used="json")
    def _output_columns_as_wire(self, columns: dict[Ref, OutputColumn]) -> dict[str, Any]:
        return _output_columns_to_wire(columns)

    def without_nodes(self, is_dropped: Callable[[str], bool]) -> EarlierWork:
        """This work without the nodes ``is_dropped`` names."""
        return self.model_copy(update={
            "finished_rows": {node_id: rows for node_id, rows in self.finished_rows.items() if not is_dropped(node_id)},
            "input_hashes": {node_id: hashes for node_id, hashes in self.input_hashes.items() if not is_dropped(node_id)},
            "answered_rows": {node_id: rows for node_id, rows in self.answered_rows.items() if not is_dropped(node_id)},
            "output_columns": {ref: column for ref, column in self.output_columns.items() if not is_dropped(ref.node_id)},
            "waiting": tuple(unit for unit in self.waiting if not is_dropped(unit.node_id)),
        })


def id_of_waiting_unit(node_id: str, row: Row, input_hash: str) -> str:
    """The ``WaitingUnit.id`` of this unit: the one place it is spelled."""
    at_row = f"@{','.join(str(n) for n in row)}" if row else ""
    return f"{node_id}{at_row}#{input_hash[:12]}"


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
    waiting: tuple[WaitingUnit, ...] = ()
    output_columns: dict[Ref, OutputColumn] = Field(default_factory=dict)
    #: Per node, the rows that finished; ``()`` for a node that ran once.
    finished_rows: dict[str, list[Row]] = Field(default_factory=dict)
    #: Per node, ``CompiledNode.fingerprint`` when it last ran: what tells a
    #: later leg over an edited graph which nodes changed.
    node_fingerprints: dict[str, str] = Field(default_factory=dict)
    #: Per node, how many rows it wasn't called on, per input that was blank there.
    rows_not_called: dict[str, dict[str, int]] = Field(default_factory=dict)
    #: Per node, the hash of what each finished unit received, in the order of
    #: ``finished_rows``; ``None`` for a unit that wasn't called (skip or blank).
    input_hashes: dict[str, list[str | None]] = Field(default_factory=dict)
    #: Per node, the finished rows a person answered.
    answered_rows: dict[str, list[Row]] = Field(default_factory=dict)
    #: Per node, how many of its units this leg took from earlier work, zeros left out.
    reused_units: dict[str, int] = Field(default_factory=dict)
    #: What a later leg may still reuse of the work this leg has not re-planned
    #: yet, kept while a re-planned node isn't finished: a leg that stopped
    #: short (failed, cancelled, a checkpoint, waiting on a person) hands it on, so
    #: going on from it reuses as much as going on from the state before would.
    earlier_work: EarlierWork | None = None

    @field_serializer("output_columns", when_used="json")
    def _output_columns_as_wire(self, columns: dict[Ref, OutputColumn]) -> dict[str, Any]:
        return _output_columns_to_wire(columns)

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
            "waiting": [unit.model_dump(mode="json") for unit in self.waiting],
            "output_columns": _output_columns_to_wire(self.output_columns),
            "earlier_work": None if self.earlier_work is None else self.earlier_work.model_dump(mode="json"),
        }

    @model_validator(mode="after")
    def _tables_read_back(self) -> Self:
        """A document's ``tables`` read once each, and put where its values refer to them."""
        if self.tables:
            frames = tables_read_once(self.tables)
            object.__setattr__(self, "tables", {})
            for name in type(self).model_fields:
                object.__setattr__(self, name, with_table_references_read(getattr(self, name), frames))
        return self

    def without_nodes(self, *node_ids: str) -> RunState:
        """This state as if these nodes had never run: a host's "run from here".

        Their values, finished rows, waiting units, hashes and fingerprints go,
        which is what makes the next leg run them again. A node reading one of
        them is re-planned, and reused where what it receives came out the same:
        to run a node and everything after it again, name them all. A node id
        also covers the nodes of a graph embedded under it (``approve`` covers
        ``approve/check``).
        """

        def is_dropped(node_id: str) -> bool:
            return any(node_id == dropped or node_id.startswith(f"{dropped}/") for dropped in node_ids)

        return self.model_copy(update={
            "waiting": tuple(unit for unit in self.waiting if not is_dropped(unit.node_id)),
            "output_columns": {ref: column for ref, column in self.output_columns.items() if not is_dropped(ref.node_id)},
            "finished_rows": {node_id: rows for node_id, rows in self.finished_rows.items() if not is_dropped(node_id)},
            "node_fingerprints": {node_id: fingerprint for node_id, fingerprint in self.node_fingerprints.items() if not is_dropped(node_id)},
            "rows_not_called": {node_id: counts for node_id, counts in self.rows_not_called.items() if not is_dropped(node_id)},
            "input_hashes": {node_id: hashes for node_id, hashes in self.input_hashes.items() if not is_dropped(node_id)},
            "answered_rows": {node_id: rows for node_id, rows in self.answered_rows.items() if not is_dropped(node_id)},
            "earlier_work": None if self.earlier_work is None else self.earlier_work.without_nodes(is_dropped),
        })
