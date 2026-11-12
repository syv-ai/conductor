"""The events ``run.stream()`` yields, one class per event, and ``Event``, the union that reads any of them back
from JSON.

For the graph ``lines -> shout -> join`` a stream is ``RunStart``, a ``NodeStart`` and a ``NodeComplete`` for each
of the three nodes with ``NodeProgress`` for ``shout`` between them, and ``RunEnd``. ``node_events.py`` says when
each event about a node is made.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import ConfigDict, Field, InstanceOf
from pydantic.json_schema import SkipJsonSchema

from conductor._internal.address import Loc
from conductor._internal.ask import Waiting
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.problem import Problem
from conductor._internal.run.state import EndedEarly, Going, RunStatus

if TYPE_CHECKING:
    from conductor._internal.run import Run


class EventBase(ConductorModel):
    """The base of every event.

    Its JSON schema marks as required every field that ``model_dump`` always writes, ``type`` included, so a
    client generated from the schema can rely on them.
    """

    model_config = ConfigDict(json_schema_serialization_defaults_required=True, arbitrary_types_allowed=True, ser_json_inf_nan="strings")


class _NodeEvent(EventBase):
    """The base of every event about one node. No event is of this class itself.

    Attributes:
        loc: The node's id. For a node inside a subgraph, the id of each subgraph node comes first:
            ``("e", "upper")``. A node inside a loop, or inside a subgraph that is inside a loop, has each of
            its events once, for all of its calls together.
    """

    loc: Loc


class RunStart(EventBase):
    """The first event of a stream. ``waiting`` is the calls that were already waiting for a person."""

    type: Literal["run_start"] = "run_start"
    waiting: tuple[Waiting, ...] = ()


class NodeStart(_NodeEvent):
    """A node started: its first call is about to be made, or its first outcome was written without a call
    (reused from an earlier run, or not called). A node has this event once, however often the run is resumed."""

    type: Literal["node_start"] = "node_start"


class NodeProgress(_NodeEvent):
    """How many calls of a node inside a loop are done, at most four times a second; ``total`` is ``None`` until
    the number of calls is known."""

    type: Literal["node_progress"] = "node_progress"
    done: int
    total: int | None


class NodeComplete(_NodeEvent):
    """Every call of a node is done.

    The values are not in the event. They are in the run (``run.results``, ``run.not_called``), so the event is
    small however large the values are.

    Attributes:
        outcomes: How many outcomes the node has: one, or one per item.
        not_called: How many of them are calls that were not made, because an argument was ``None``.
        reused: How many of them were taken from an earlier run.
    """

    type: Literal["node_complete"] = "node_complete"
    outcomes: int
    not_called: int = 0
    reused: int = 0


class NodeSkipped(_NodeEvent):
    """Every call of a node was skipped: the node is downstream of a branch that was not chosen, or its function
    returned ``SKIPPED``."""

    type: Literal["node_skipped"] = "node_skipped"


class NodeFailed(_NodeEvent):
    """A call of the node failed and is not retried, so the run fails.

    Attributes:
        cause: Why it failed. Its ``loc`` says which call: the node's id, then the call's item indexes. Inside a
            subgraph, the subgraph node and its item indexes come first: ``("e", 0, "upper")``.
    """

    type: Literal["node_failed"] = "node_failed"
    cause: Problem


class NodeRetry(_NodeEvent):
    """A call failed with an error its node's ``Policy`` retries, and will be made again.

    Its ``loc`` says which call, as the ``loc`` of ``cause`` does: the node's id, then the call's item indexes.
    Inside a subgraph, the subgraph node and its item indexes come first.

    Attributes:
        attempt: The number of the retry that will be made: 1 for the first.
        retries: How many retries the node's ``Policy`` allows.
        delay: How many seconds from now the retry is made.
        cause: Why the call failed.
    """

    type: Literal["node_retry"] = "node_retry"
    attempt: int
    retries: int
    delay: float
    cause: Problem


class NodeWaiting(EventBase):
    """A call asked a person a question and is waiting for the answer. The calls downstream of it are not made
    until it is answered. Every other call of the graph is made."""

    type: Literal["node_waiting"] = "node_waiting"
    waiting: Waiting


class RunEnd(EventBase):
    """The last event of a stream: the run is complete, waiting for a person, or ended early.

    Attributes:
        status: The run's status.
        cause: Why the run ended early (``failed``, ``stopped``, ``cancelled`` or ``timed_out``). ``None`` otherwise.
        run: The run with all its values, for the host to store. It is not written when the event is dumped.
    """

    type: Literal["run_end"] = "run_end"
    status: RunStatus
    cause: Problem | None = None
    run: SkipJsonSchema[InstanceOf[Run]] = Field(exclude=True)

    @classmethod
    def of(cls, run: Run) -> RunEnd:
        """The ``RunEnd`` for ``run``, with the status and the cause of its state."""
        match run.state:
            case Going(status=status):
                return cls(status=status, run=run)
            case EndedEarly(status=status, cause=cause):
                return cls(status=status, cause=cause, run=run)


class RunCheckpoint(EventBase):
    """A copy of the run while calls are still being made, for a host to store.

    It comes at most every ``checkpoint_every`` seconds, and once more when the stream is stopped while calls are
    on threads. A host that stores the copy loses little if the process dies: ``Run.from_json`` reads it back,
    and ``resume()`` makes only the calls that have no outcome in it. With ``stream(hand_back=)`` it is also the
    last event, in place of ``RunEnd``: that copy has the outcome of every call that was on a thread.

    Attributes:
        run: The copy. Its status is ``running``, or, once the stream is stopped, how it stopped (``cancelled``,
            ``failed``, ``stopped``, ``timed_out``). It is not written when the event is dumped.
    """

    type: Literal["run_checkpoint"] = "run_checkpoint"
    run: SkipJsonSchema[InstanceOf[Run]] = Field(exclude=True)


Event = Annotated[
    RunStart | NodeStart | NodeProgress | NodeComplete | NodeSkipped | NodeFailed | NodeRetry | NodeWaiting | RunCheckpoint | RunEnd,
    Field(discriminator="type"),
]
Event.__doc__ = "Any event ``run.stream()`` yields, told apart by ``type``: ``TypeAdapter(Event)`` reads one back from JSON."
