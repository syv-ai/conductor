"""The events ``execute`` yields while a leg runs — a leg being one call of
``execute``; a run takes several when a node waits on a person in between.

Each event is a ``TypedDict`` discriminated on ``type``. An event carries
records, not their serialisations — an ``ErrorCause`` on ``node_error``
and ``graph_error``, a ``Series`` (a value with many rows) inside
``results`` — and the host that sends an event over the network
serialises it at that edge.

Every event that ends a leg carries ``results`` (what the leg produced,
by node and output) and ``record`` (the whole ``RunRecord`` of the run so
far, in wire form, which ``execute(record=...)`` starts the next leg from).
A row on an event is a ``Row``, the path of positions the ledger uses.
"""

from __future__ import annotations

from typing import Any, Literal, Required, TypedDict

from conductor.errors import ErrorCause
from conductor.execution.record import RunRecord
from conductor.series import Row


class NodeStartEvent(TypedDict):
    type: Literal["node_start"]
    node_id: str


class NodeProgressEvent(TypedDict):
    """A node running once per row finished one more row. ``total`` is
    ``None`` until the number of rows it will run over is known, then the
    count ("4 of 10")."""

    type: Literal["node_progress"]
    node_id: str
    done: int
    total: int | None


class NodeCompleteEvent(TypedDict, total=False):
    """A node finished; ``result`` is its outputs by name. ``cached`` is set
    when the outputs came from ``execute(cache=...)`` and the node did not run.

    ``type`` and ``node_id`` are ``Required`` even though the class is
    ``total=False``: a generated client must not see the discriminant as
    optional.
    """

    type: Required[Literal["node_complete"]]
    node_id: Required[str]
    #: The node's outputs by name: a value, or a ``Series`` for anything on an index.
    result: Required[dict[str, Any]]
    cached: bool


class NodeSkippedEvent(TypedDict):
    type: Literal["node_skipped"]
    node_id: str


class NodeErrorEvent(TypedDict):
    """A node failed — or one row of a node that runs per row. ``cause.code``
    says what kind of failure — ``invalid_input``, ``timeout``,
    ``execution_failed`` — for a consumer to key on."""

    type: Literal["node_error"]
    node_id: str
    error: str
    cause: ErrorCause


class NodeRetryEvent(TypedDict):
    type: Literal["node_retry"]
    node_id: str
    row: Row | None
    attempt: int
    retries: int
    error: str
    delay: float


class GraphCompleteEvent(TypedDict):
    """The leg completed. ``results`` is what it produced, by node and
    output; ``record`` is the whole record of the run, which a host can
    store and hand back to ``execute(record=...)`` to start a new run from
    this one."""

    type: Literal["graph_complete"]
    results: dict[str, dict[str, Any]]
    record: RunRecord


class PendingUnit(TypedDict):
    """One node waiting on a person — or one row of a node that runs per
    row: the node, its row (when it runs per row), the prompt, and the
    questions as ``Input`` records named by address (``node.field``)."""

    node_id: str
    row: Row | None
    prompt: str | None
    questions: tuple[Any, ...]


class GraphPendingEvent(TypedDict):
    """The leg ended with nodes (or rows of them) waiting on a person — all
    of them at once. ``results`` is what the leg completed and ``record``
    the whole record; the next leg starts from the record with the answers
    in ``cache``."""

    type: Literal["graph_pending"]
    pending: list[PendingUnit]
    results: dict[str, dict[str, Any]]
    record: RunRecord


class GraphErrorEvent(TypedDict, total=False):
    """A node (or one row of one) failed and the leg stopped. Like every
    ending it carries ``results`` so far and ``record`` beside the cause, so
    a host can start a new run from a failed one without losing what ran."""

    type: Required[Literal["graph_error"]]
    node_id: str
    error: Required[str]
    cause: Required[ErrorCause]
    results: Required[dict[str, dict[str, Any]]]
    record: Required[RunRecord]


class GraphCancelledEvent(TypedDict):
    """The host set ``cancel``. ``results`` so far and ``record`` travel
    with the reason, as on every ending."""

    type: Literal["graph_cancelled"]
    results: dict[str, dict[str, Any]]
    record: RunRecord


class GraphTimeoutEvent(TypedDict):
    """The leg ran longer than the ``timeout`` its caller set (carried as
    ``timeout_seconds``). ``results`` so far and ``record`` travel with the
    reason, as on every ending."""

    type: Literal["graph_timeout"]
    results: dict[str, dict[str, Any]]
    record: RunRecord
    elapsed_seconds: float
    timeout_seconds: float


ExecutionEvent = (
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
    | GraphTimeoutEvent
)
