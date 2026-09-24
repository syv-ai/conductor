"""The events ``execute`` yields while a leg runs — a leg being one call of
``execute``; a run takes several when a node waits on a person in between.

Each event is a frozen ``ConductorModel``, one of a union discriminated on
``type``: a caller reads ``ending.results`` and can ``match`` on the class.
The engine builds each one validated, like every other record; an event is
built about once a unit and costs well under a microsecond, where a unit
costs a hundred or more. An event carries records, not their serialisations — an ``ErrorCause`` on ``node_error``
and ``graph_error``, a ``Series`` (a value with many rows) inside
``results`` — and the host that sends an event over the network
serialises it at that edge.

Every event that ends a leg carries ``results`` (what the leg produced,
by node and output) and ``state`` (the whole ``RunState`` of the run so
far, in wire form, which ``execute(state=...)`` starts the next leg from).
A row on an event is a ``Row``, the path of positions the ledger uses.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from conductor.errors import ErrorCause
from conductor.execution.state import RunState
from conductor.model import ConductorModel
from conductor.series import Row


class NodeStartEvent(ConductorModel):
    type: Literal["node_start"]
    node_id: str


class NodeProgressEvent(ConductorModel):
    """A node running once per row finished one more row. ``total`` is
    ``None`` until the number of rows it will run over is known, then the
    count ("4 of 10")."""

    type: Literal["node_progress"]
    node_id: str
    done: int
    total: int | None


class NodeCompleteEvent(ConductorModel):
    """A node finished; ``result`` is its outputs by name. ``cached`` is
    ``True`` when the outputs came from ``execute(cache=...)`` and the node
    did not run."""

    type: Literal["node_complete"]
    node_id: str
    #: The node's outputs by name: a value, or a ``Series`` for anything on an index.
    result: dict[str, Any]
    cached: bool = False


class NodeSkippedEvent(ConductorModel):
    type: Literal["node_skipped"]
    node_id: str


class NodeErrorEvent(ConductorModel):
    """A node failed — or one row of a node that runs per row. ``cause.code``
    says what kind of failure — ``invalid_input``, ``timeout``,
    ``execution_failed`` — for a consumer to key on."""

    type: Literal["node_error"]
    node_id: str
    error: str
    cause: ErrorCause


class NodeRetryEvent(ConductorModel):
    type: Literal["node_retry"]
    node_id: str
    row: Row | None
    attempt: int
    retries: int
    error: str
    delay: float


class GraphCompleteEvent(ConductorModel):
    """The leg completed. ``results`` is what it produced, by node and
    output; ``state`` is the run's whole state, which a host can
    store and hand back to ``execute(state=...)`` to start a new run from
    this one."""

    type: Literal["graph_complete"]
    results: dict[str, dict[str, Any]]
    state: RunState


class PendingUnit(ConductorModel):
    """One node waiting on a person — or one row of a node that runs per
    row: the node, its row (when it runs per row), the prompt, and the
    questions as ``Input`` records named by address (``node.field``)."""

    node_id: str
    row: Row | None
    prompt: str | None
    questions: tuple[Any, ...]


class GraphPendingEvent(ConductorModel):
    """The leg ended with nodes (or rows of them) waiting on a person — all
    of them at once. ``results`` is what the leg completed and ``state``
    the run's whole state; the next leg starts from it with the answers
    in ``cache``."""

    type: Literal["graph_pending"]
    pending: list[PendingUnit]
    results: dict[str, dict[str, Any]]
    state: RunState


class GraphErrorEvent(ConductorModel):
    """A node (or one row of one) failed and the leg stopped. Like every
    ending it carries ``results`` so far and ``state`` beside the cause, so
    a host can start a new run from a failed one without losing what ran."""

    type: Literal["graph_error"]
    node_id: str
    error: str
    cause: ErrorCause
    results: dict[str, dict[str, Any]]
    state: RunState


class GraphCancelledEvent(ConductorModel):
    """The host set ``cancel``. ``results`` so far and ``state`` travel
    with the reason, as on every ending."""

    type: Literal["graph_cancelled"]
    results: dict[str, dict[str, Any]]
    state: RunState


class GraphTimeoutEvent(ConductorModel):
    """The leg ran longer than the ``timeout`` its caller set (carried as
    ``timeout_seconds``). ``results`` so far and ``state`` travel with the
    reason, as on every ending."""

    type: Literal["graph_timeout"]
    results: dict[str, dict[str, Any]]
    state: RunState
    elapsed_seconds: float
    timeout_seconds: float


ExecutionEvent = Annotated[
    NodeStartEvent
    | NodeProgressEvent
    | NodeCompleteEvent
    | NodeSkippedEvent
    | NodeErrorEvent
    | NodeRetryEvent
    | GraphCompleteEvent
    | GraphPendingEvent
    | GraphErrorEvent
    | GraphCancelledEvent
    | GraphTimeoutEvent,
    Field(discriminator="type"),
]
