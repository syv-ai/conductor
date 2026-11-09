"""What a leg emits while it runs.

A *leg* is one call of ``CompiledGraph.stream`` or ``run``; a run takes several
when a node waits on a person. Each event is a frozen record, one of a union
discriminated on ``type``, so a caller can ``match`` on the class and a host
can send one over the network as it is. ``RunStart`` is always first and
``RunEnd`` always last; ``RunEnd.state`` says how the leg ended. A leg asked for
checkpoints also emits ``RunCheckpoint`` along the way.

A value in ``NodeComplete.result`` is a typed value (a ``Series`` for anything
on an index), and a host that sends the event on serialises it at that edge.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field

from conductor.errors import Problem
from conductor.execution.run_state import RunState, WaitingUnit
from conductor.model import ConductorModel
from conductor.series import Row


class _EventRecord(ConductorModel):
    """Every event: its schema marks ``type`` and every field a dump always writes as required, so a generated client can rely on them."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class RunStart(_EventRecord):
    """A leg began. On a leg going on from a state: the nodes kept from it
    (``reused_nodes``), the nodes it left out because they or something above
    them changed (``changed_nodes``), and the units still waiting."""

    type: Literal["run_start"] = "run_start"
    reused_nodes: list[str] = Field(default_factory=list)
    changed_nodes: list[str] = Field(default_factory=list)
    waiting: tuple[WaitingUnit, ...] = ()


class NodeStart(_EventRecord):
    type: Literal["node_start"] = "node_start"
    node_id: str


class NodeProgress(_EventRecord):
    """A node running once per row finished one more row. ``total`` is
    ``None`` until the number of rows it will run over is known."""

    type: Literal["node_progress"] = "node_progress"
    node_id: str
    done: int
    total: int | None


class NodeComplete(_EventRecord):
    """A node finished; ``result`` is its outputs by name: a value, or a
    ``Series`` for anything on an index. ``rows_not_called`` counts, per
    input, the rows the node wasn't called on because that input was blank."""

    type: Literal["node_complete"] = "node_complete"
    node_id: str
    result: dict[str, Any]
    rows_not_called: dict[str, int] = Field(default_factory=dict)


class NodeSkipped(_EventRecord):
    type: Literal["node_skipped"] = "node_skipped"
    node_id: str


class NodeFailed(_EventRecord):
    """A node failed, or one row of a node that runs per row. ``cause.type``
    says what kind of failure, for a consumer to key on."""

    type: Literal["node_failed"] = "node_failed"
    node_id: str
    error: str
    cause: Problem


class NodeRetry(_EventRecord):
    type: Literal["node_retry"] = "node_retry"
    node_id: str
    row: Row
    attempt: int
    retries: int
    error: str
    delay: float


class UnitWaiting(_EventRecord):
    """A node asked a person something, on this leg. The unit waits, whatever
    reads it waits, and the rest of the graph runs on."""

    type: Literal["unit_waiting"] = "unit_waiting"
    unit: WaitingUnit


class RunCheckpoint(_EventRecord):
    """A state to save while the leg runs on, status ``running``: what has
    finished and who is waiting, as of the moment it was taken. Sent only when
    ``stream`` was given ``checkpoint_every``. It is a copy, so it can be dumped
    later or on another thread. ``run`` and ``answer`` go on from it as from any
    state: a finished unit is not called again and one in flight is."""

    type: Literal["run_checkpoint"] = "run_checkpoint"
    state: RunState


class RunEnd(_EventRecord):
    """The leg ended. ``state`` is the run as of now, for any way of ending."""

    type: Literal["run_end"] = "run_end"
    state: RunState


Event = Annotated[
    RunStart | NodeStart | NodeProgress | NodeComplete | NodeSkipped | NodeFailed | NodeRetry | UnitWaiting | RunCheckpoint | RunEnd,
    Field(discriminator="type"),
]
