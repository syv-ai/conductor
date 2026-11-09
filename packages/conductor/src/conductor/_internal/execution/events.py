"""What a leg emits while it runs.

A *leg* is one call of ``CompiledGraph.stream`` or ``run``; a run takes several
when a node waits on a person. Each event is a frozen record, one of a union
discriminated on ``type``, so a caller can ``match`` on the class and a host
can send one over the network as it is. ``RunStart`` is always first and
``RunEnd`` always last; ``RunEnd.state`` says how the leg ended. A leg asked for
checkpoints also emits ``RunCheckpoint`` along the way.

In Python, a value in ``NodeComplete.outputs`` is a typed value (``{"rows": [...], "values": [...]}``
for an output holding many, ``ValuesOnRows``). ``model_dump(mode="json")``, ``model_dump_json()`` and
``TypeAdapter(Event).dump_json`` write any event as JSON, each value by its own
type (see ``conductor.to_jsonable``), so a host sends one as it is.
An event stands alone on the stream, so a table in one (in ``NodeComplete.outputs``, a
waiting unit's ``inputs``) is written whole. ``RunCheckpoint.state`` is a state, with its
own ``tables``. ``RunEnd.state`` is the host's to keep and stays out of what the event
writes, so a stream sends each value once and no state reaches a browser by default.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, computed_field

from conductor._internal.codec import TypedValue
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import Problem
from conductor._internal.execution.run_state import RunState, RunStatus, WaitingUnit
from conductor._internal.index import Row


class _EventBase(ConductorModel):
    """Every event: its schema marks ``type`` and every field a dump always writes as required, so a generated client can rely on them."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class RunStart(_EventBase):
    """A leg began. On a leg going on from a state: the nodes kept from it whole
    (``reused_nodes``), the nodes whose units run again because they or something
    above them changed (``changed_nodes``), and the units still waiting. These
    speak about nodes: which units of a changed node are reused is only known
    once its inputs are, and ``NodeComplete.reused_units`` says so."""

    type: Literal["run_start"] = "run_start"
    reused_nodes: list[str] = Field(default_factory=list)
    changed_nodes: list[str] = Field(default_factory=list)
    waiting: tuple[WaitingUnit, ...] = ()


class NodeStart(_EventBase):
    """A node's first unit started; never again for that node, on this or a later leg."""

    type: Literal["node_start"] = "node_start"
    node_id: str


class NodeProgress(_EventBase):
    """How far a node running once per row has got. At most four a second per
    node, and always its last count: ``done`` of ``total`` rows, ``total``
    ``None`` until the number of rows it will run over is known."""

    type: Literal["node_progress"] = "node_progress"
    node_id: str
    done: int
    total: int | None


class NodeComplete(_EventBase):
    """A node finished; ``outputs`` is its outputs by name: a value, or
    ``{"rows": [...], "values": [...]}`` for an output holding many. ``rows_not_called`` counts, per
    input, the rows the node wasn't called on because that input was blank.
    ``reused_units`` counts the units this leg took from an earlier finished
    unit instead of calling the node."""

    type: Literal["node_complete"] = "node_complete"
    node_id: str
    outputs: dict[str, TypedValue]
    rows_not_called: dict[str, int] = Field(default_factory=dict)
    reused_units: int = 0


class NodeSkipped(_EventBase):
    """Every unit of this node was skipped and it produced nothing, not even a blank."""

    type: Literal["node_skipped"] = "node_skipped"
    node_id: str


class NodeFailed(_EventBase):
    """A node failed, or one row of a node that runs per row. ``cause.type``
    says what kind of failure, for a consumer to key on."""

    type: Literal["node_failed"] = "node_failed"
    node_id: str
    cause: Problem


class NodeRetry(_EventBase):
    """An attempt failed on the outside world and is being tried again: try
    ``attempt`` of ``retries``, on ``row``, after ``delay`` seconds. ``error`` is what failed."""

    type: Literal["node_retry"] = "node_retry"
    node_id: str
    row: Row
    attempt: int
    retries: int
    error: str
    delay: float


class UnitWaiting(_EventBase):
    """A unit is waiting on a person, as of this leg: its node asked, or an
    earlier unit that received the same inputs was still waiting and goes on
    waiting at this row. The unit waits, whatever reads it waits, and the rest
    of the graph runs on."""

    type: Literal["unit_waiting"] = "unit_waiting"
    unit: WaitingUnit


class RunCheckpoint(_EventBase):
    """A state to save while the leg runs on, status ``running``: what has
    finished and who is waiting, as of the moment it was taken. Sent only when
    ``stream`` was given ``checkpoint_every``. It is a copy, so it can be dumped
    later or on another thread. ``run`` and ``answer`` go on from it as from any
    state: a finished unit is not called again and one in flight is."""

    type: Literal["run_checkpoint"] = "run_checkpoint"
    state: RunState


class RunEnd(_EventBase):
    """The leg ended. ``state`` is the run as this leg left it, for any way of
    ending. It is left out of what the event writes, which says only ``status``:
    every value already went out on its own event. A host that wants the state
    on the wire writes ``event.state.model_dump_json()``."""

    type: Literal["run_end"] = "run_end"
    state: RunState = Field(exclude=True)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def status(self) -> RunStatus:
        return self.state.status


Event = Annotated[
    RunStart | NodeStart | NodeProgress | NodeComplete | NodeSkipped | NodeFailed | NodeRetry | UnitWaiting | RunCheckpoint | RunEnd,
    Field(discriminator="type"),
]
