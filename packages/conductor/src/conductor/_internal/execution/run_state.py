"""``RunState``: everything a run is, as one record a host can store between legs.

A *leg* is one call of ``CompiledGraph.run``; a run takes several when a node
waits on a person. The state is what a leg ends on, what a checkpoint during
one hands over, and what the next one starts from. A unit in flight is in no
state: going on from one calls it again, so a unit runs at least once. It says how the last leg ended (``status``, ``failure``), who is
waiting and what they were asked (``waiting``), what every output holds
(``output_columns``), which units finished (``finished_rows``), and a hash of
how the graph used each node (``node_fingerprints``). Everything else a run
knows — which rows each index has, which indexes will gain no more — follows
from these and is worked out again when a leg starts.

The module knows nothing of graphs, so a state loads with no registry in
sight and everything a form needs is on it. Its values are read through a
compiled graph: ``compiled.results(state)`` and ``compiled.outputs(state)``.

A state a leg just built holds its values typed; one read from JSON holds
their wire form, and the graph's validators read that when it is next used.
Dumped as JSON, each value is written by its own type.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Self

from pydantic import Field, field_serializer, model_validator

from conductor._internal.codec import values_to_wire
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import Problem
from conductor._internal.ref import Ref
from conductor._internal.series import Row, Series

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

    ``rows`` and ``values`` line up. A row in ``skipped`` holds nothing, and
    neither does any row under it. A ``Table`` is one value at its parent's
    row, its rows inside it.
    """

    rows: list[Row] = Field(default_factory=list)
    values: list[Any] = Field(default_factory=list)
    skipped: list[Row] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rows_line_up_with_values(self) -> Self:
        if len(self.rows) != len(self.values):
            raise ValueError(f"{len(self.rows)} rows for {len(self.values)} values")
        return self


class Question(ConductorModel):
    """One thing a person is asked: an output of the waiting node.

    Plain data, so a form renders from a stored state with no graph loaded.
    ``dtype`` and ``widget`` are the dumps of the output's type and control;
    ``json_schema`` is the shape of one answer, for a client to check before
    sending. ``default`` is in wire form and fills the question when
    ``optional`` and left out. Questions sharing a ``choice`` are alternatives:
    the person answers exactly one.
    """

    name: str
    title: str
    description: str | None = None
    dtype: dict[str, Any] | None
    json_schema: dict[str, Any]
    widget: dict[str, Any] | None = None
    default: Any = None
    optional: bool = False
    choice: str | None = None


class WaitingUnit(ConductorModel):
    """One unit waiting on a person: a node, on one row when it runs per row.

    ``id`` is what an answer is keyed by. It is written once, when the node
    asks, and nothing ever reads it apart. ``questions`` are frozen until the
    unit is answered or its node, or something above it, changes. ``inputs``
    says what the row is about, in a state's size no more than the unit received.
    """

    id: str
    node_id: str
    row: Row
    prompt: str | None = None
    questions: tuple[Question, ...]
    #: What this unit received, by input name, so a form can show the row it is about: exactly the
    #: values its call would have got, no more; a ``Series`` as a list, a table in its JSON form.
    inputs: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("inputs", when_used="json")
    def _inputs_as_wire(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Each input written by its own type; held typed until a dump, as a state's values are."""
        return dict(zip(inputs, values_to_wire([value.values if isinstance(value, Series) else value for value in inputs.values()]), strict=True))


class RunState(ConductorModel):
    """A run, as of the end of its last leg or of a checkpoint during one.

    ``compiled.run(state)`` goes on from one, ``compiled.answer(state, ...)``
    returns a new one with answers written in, and ``without_nodes`` returns one
    that forgets some nodes. Nothing changes one in place.
    """

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

    @field_serializer("output_columns", when_used="json")
    def _output_columns_as_wire(self, columns: dict[Ref, OutputColumn]) -> dict[str, Any]:
        """Each column's values written by their own types, naming the output whose value has no JSON form."""
        written: dict[str, Any] = {}
        for ref, column in columns.items():
            try:
                values = values_to_wire(column.values)
            except Exception as unwritable:
                raise TypeError(f"{ref}: a value has no JSON form ({unwritable})") from unwritable
            written[ref] = {"rows": column.rows, "values": values, "skipped": column.skipped}
        return written

    def without_nodes(self, *node_ids: str) -> RunState:
        """This state as if these nodes had never run: a host's "run from here".

        Their values, finished rows, waiting units and fingerprints go, which
        is what makes the next leg run them again with everything that reads
        them. A node id also covers the nodes of a graph embedded under it
        (``approve`` covers ``approve/check``).
        """

        def is_dropped(node_id: str) -> bool:
            return any(node_id == dropped or node_id.startswith(f"{dropped}/") for dropped in node_ids)

        return self.model_copy(update={
            "waiting": tuple(unit for unit in self.waiting if not is_dropped(unit.node_id)),
            "output_columns": {ref: column for ref, column in self.output_columns.items() if not is_dropped(ref.node_id)},
            "finished_rows": {node_id: rows for node_id, rows in self.finished_rows.items() if not is_dropped(node_id)},
            "node_fingerprints": {node_id: fingerprint for node_id, fingerprint in self.node_fingerprints.items() if not is_dropped(node_id)},
            "rows_not_called": {node_id: counts for node_id, counts in self.rows_not_called.items() if not is_dropped(node_id)},
        })
