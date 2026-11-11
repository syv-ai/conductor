"""The events of a leg: what ``run.stream()`` yields, a record each, and ``Event``, which reads any of them back from JSON."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import ConfigDict, Field, InstanceOf
from pydantic.json_schema import SkipJsonSchema

from conductor._internal.address import Loc
from conductor._internal.ask import Waiting
from conductor._internal.conductor_model import ConductorModel
from conductor._internal.problem import Problem
from conductor._internal.run_status import EndedEarly, Going, RunStatus

if TYPE_CHECKING:
    from conductor._internal.run import Run


class EventBase(ConductorModel):
    """Every event: its schema marks ``type`` and every field a dump always writes as required, so a generated client can rely on them."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True, arbitrary_types_allowed=True, ser_json_inf_nan="strings")


class _NodeEvent(EventBase):
    """Every event about one node, or one unit of it. Never made on its own.

    Attributes:
        loc: Which node it is: its id, and for a node inside an embedded graph the id of each embedding node
            first, ``("e", "upper")``. A node inside an embedded graph says its events once, for every item of
            the embedding node together, as a node that runs per item does.
    """

    loc: Loc


class RunStart(EventBase):
    """A leg began; ``waiting`` is the units already waiting for a person."""

    type: Literal["run_start"] = "run_start"
    waiting: tuple[Waiting, ...] = ()


class NodeStart(_NodeEvent):
    """A node's first unit started, called or taken from earlier work; never again for that node, on this or a later leg."""

    type: Literal["node_start"] = "node_start"


class NodeProgress(_NodeEvent):
    """How many units of a node that runs per item are done, at most four times a second; ``total`` ``None`` until
    every Series above it is made."""

    type: Literal["node_progress"] = "node_progress"
    done: int
    total: int | None


class NodeComplete(_NodeEvent):
    """Every unit of a node is done.

    Its values stay in the run, as ``run.results`` and ``run.not_called`` read them, so an event is small whatever
    the run holds.

    Attributes:
        outcomes: How many outcomes it has: one, or one per item.
        not_called: How many of them a blank input kept the node from.
        reused: How many of them ``run.again`` took from the earlier run.
    """

    type: Literal["node_complete"] = "node_complete"
    outcomes: int
    not_called: int = 0
    reused: int = 0


class NodeSkipped(_NodeEvent):
    """Every unit of a node was a branch not taken."""

    type: Literal["node_skipped"] = "node_skipped"


class NodeFailed(_NodeEvent):
    """A unit of the node failed, finally; ``cause`` says why, and its ``loc`` which unit: the node's id, then the
    unit's item, and inside an embedded graph the embedding node and its item first, ``("e", 0, "upper")``."""

    type: Literal["node_failed"] = "node_failed"
    cause: Problem


class NodeRetry(_NodeEvent):
    """An attempt failed on the outside world and is tried again: attempt ``attempt`` of ``retries``, after
    ``delay`` seconds. Its ``loc`` is the unit's, as its ``cause`` has it: the node's id, then the place of the
    unit's item in each Series the node runs per item of; inside an embedded graph, the embedding node and its
    item first."""

    type: Literal["node_retry"] = "node_retry"
    attempt: int
    retries: int
    delay: float
    cause: Problem


class NodeWaiting(EventBase):
    """A unit started waiting for a person. It waits, whatever reads it waits, and the rest of the graph runs on."""

    type: Literal["node_waiting"] = "node_waiting"
    waiting: Waiting


class RunEnd(EventBase):
    """The leg ended, with ``status`` and, for a run that ``failed``, ``stopped``, was ``cancelled`` or ``timed_out``, its ``cause``.

    ``run`` is the run as the leg left it, every value in it; it stays out of what the event writes, for the host to store.
    """

    type: Literal["run_end"] = "run_end"
    status: RunStatus
    cause: Problem | None = None
    run: SkipJsonSchema[InstanceOf[Run]] = Field(exclude=True)

    @classmethod
    def of(cls, run: Run) -> RunEnd:
        """The end of the leg that left ``run`` as it is."""
        match run.state:
            case Going(status=status):
                return cls(status=status, run=run)
            case EndedEarly(status=status, cause=cause):
                return cls(status=status, cause=cause, run=run)


class RunCheckpoint(EventBase):
    """The run as it stands mid-leg, for a host to store, every ``checkpoint_every`` seconds and once when the leg is stopped.

    ``run`` is a copy, its ``status`` ``running``, or once stopped how it stopped (``cancelled``, ``failed``,
    ``stopped``, ``timed_out``) with its ``cause``: ``graph.load_run`` takes its JSON and ``resume()`` calls only
    the units it holds no record of, so a process that dies mid-leg repeats at most the calls that were in
    flight. Like ``RunEnd``'s, it stays out of what the event writes. A leg handed back (``stream(hand_back=)``)
    says one last, in place of ``RunEnd``: its ``run`` is ``running`` and holds every call the leg waited for.
    """

    type: Literal["run_checkpoint"] = "run_checkpoint"
    run: SkipJsonSchema[InstanceOf[Run]] = Field(exclude=True)


Event = Annotated[
    RunStart | NodeStart | NodeProgress | NodeComplete | NodeSkipped | NodeFailed | NodeRetry | NodeWaiting | RunCheckpoint | RunEnd,
    Field(discriminator="type"),
]
Event.__doc__ = "Any event ``run.stream()`` yields, told apart by ``type``: ``TypeAdapter(Event)`` reads one back from JSON."
