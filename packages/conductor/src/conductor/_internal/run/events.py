"""The events ``run.stream()`` yields while it runs a graph, one class per event, and ``Event``, their union.

The example graph, the first in conductor's README::

    lines(text: str) -> Series[str]          # called once; returns ["a", "b"]
    shout(line: str, mark: str = "!") -> str # wired to lines.result: called once per line
    join(texts: Series[str]) -> str          # wired to shout.result: called once, with ["A!", "B!"]

``async for event in graph.with_inputs(text="a\\nb").stream()`` yields these events, in this order::

    RunStart(waiting=())
    NodeStart(loc=("lines",))
    NodeStart(loc=("shout",))
    NodeProgress(loc=("shout",), finished=1, total=2)
    NodeStart(loc=("join",))
    NodeComplete(loc=("lines",), outcomes=1)
    NodeProgress(loc=("shout",), finished=2, total=2)
    NodeComplete(loc=("shout",), outcomes=2)
    NodeComplete(loc=("join",), outcomes=1)
    RunEnd(status="complete")

``RunStart`` is always first and ``RunEnd`` last (a stream stopped with ``hand_back`` ends with ``RunCheckpoint``
instead). In between, a node's ``NodeStart``, when it has one, comes before every other event about it, and its
``NodeComplete`` (or ``NodeSkipped``) after every other one. Events about different nodes interleave, and two
rules explain the order above:

- ``NodeProgress`` comes at most four times a second per node, so ``shout``'s second call was made before
  ``NodeStart(join)``, but ``finished=2`` is told only with the final count, just before ``NodeComplete(shout)``.
- The engine looks for complete nodes only when no call is ready to be made. Here that is first after ``join``'s
  call, so even ``lines`` is complete only after ``NodeStart(join)``.

Every event is a frozen pydantic model with a ``type`` (``"node_complete"``). ``event.model_dump(mode="json")``
writes it as JSON, for example to send to a web page, and ``TypeAdapter(Event)`` reads it back.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import ConfigDict, Field, InstanceOf
from pydantic.json_schema import SkipJsonSchema

from conductor._internal.base import ConductorModel, Loc
from conductor._internal.problem import Problem

from .state import EndedEarly, Going, RunStatus
from .waiting import Waiting

if TYPE_CHECKING:
    from . import Run


class EventBase(ConductorModel):
    """The base class of every event.

    In its JSON schema every field is required, ``type`` and the fields with a default included, because
    ``model_dump`` always writes them: a client generated from the schema can rely on every field being there.
    """

    model_config = ConfigDict(json_schema_serialization_defaults_required=True, arbitrary_types_allowed=True)


class _NodeEvent(EventBase):
    """The base class of every event about one node: ``NodeStart(loc=("shout",))``. It is never made itself.

    Attributes:
        loc: Which node: its id, ``("shout",)``. For a node inside a subgraph, the id of the subgraph node comes
            first: ``("loud", "shout")`` for ``shout`` inside the subgraph node ``loud``. The loc names the node,
            not one of its calls: a node inside a loop has one ``NodeStart``, one ``NodeComplete`` and one
            ``NodeFailed`` for all its calls, not one per call. Only ``NodeRetry`` is about one call, and its loc
            says which.
    """

    loc: Loc


class RunStart(EventBase):
    """The first event of every stream: ``RunStart(waiting=())``.

    Attributes:
        waiting: The calls that were waiting for a person's answer when the stream started, in the order of
            ``run.waiting``. A run that ended waiting and is streamed again without an answer starts with its
            waiting calls here, and then ends at once with ``RunEnd(status="waiting")``.
    """

    type: Literal["run_start"] = "run_start"
    waiting: tuple[Waiting, ...] = ()


class NodeStart(_NodeEvent):
    """A node started: its first call is about to be made, or its first outcome was written without a call.

    An outcome is written without a call when it is reused from the previous run (the run that
    ``run.with_inputs(...)`` or ``run.again(...)`` was called on), or when the function is not called because an
    argument is ``None``. A node whose every call is skipped because of a branch that was not
    chosen has no ``NodeStart``, only ``NodeSkipped``. A node inside a subgraph has its ``NodeStart`` right after the subgraph node's:
    ``NodeStart(loc=("loud",))``, then ``NodeStart(loc=("loud", "shout"))``.

    A node has at most one ``NodeStart`` in a run, however many streams the run takes. A node that already had an
    outcome when the stream started has none in this stream: after an answer, the stream that goes on has
    ``NodeComplete`` for the node that asked, with no ``NodeStart`` before it.
    """

    type: Literal["node_start"] = "node_start"


class NodeProgress(_NodeEvent):
    """How many calls of a node inside a loop have an outcome that is not waiting, at most four times a second;
    ``total`` is ``None`` until the number of calls is known.

    It comes after a call's outcome is written, and once more with the final count before the node's
    ``NodeComplete`` or at the end of the stream, unless the count has not changed. A node inside no loop has no
    ``NodeProgress``. In the example ``shout`` has ``NodeProgress(finished=1, total=2)``, then ``finished=2, total=2``.

    Attributes:
        finished: How many calls of the node are finished: they have an outcome that is not waiting (done,
            answered, skipped or not called).
        total: How many calls the node will have. ``None`` while that depends on a series that does not exist yet.
    """

    type: Literal["node_progress"] = "node_progress"
    finished: int
    total: int | None


class NodeComplete(_NodeEvent):
    """Every call of a node has an outcome.

    None of the outcomes is waiting, and no call of the node or of a node upstream of it is running. The outcomes
    may be done, answered or not called, and some may be skipped; a node whose every call is skipped has
    ``NodeSkipped`` instead. A node has at most one ``NodeComplete`` in a run.

    The values are not in the event: they are in the run (``run.results``, ``run.not_called``), so the event is
    small however large the values are. With ``text="a\\n \\nb"`` and a node ``strip`` before ``shout`` that
    returns ``None`` for the line ``" "``, ``shout`` has ``NodeComplete(loc=("shout",), outcomes=3, not_called=1)``.

    Attributes:
        outcomes: How many outcomes the node has: 1 for a node inside no loop, else one per call. For a subgraph
            node, one per call of the subgraph.
        not_called: How many of the outcomes are not called: the function was not called because an argument was
            ``None`` and its input does not accept ``None``. Always 0 for a subgraph node: the nodes inside it
            have their own counts.
        reused: How many of the outcomes this stream reused from the previous run in place of a call. Always 0 for
            a subgraph node, for the same reason.
    """

    type: Literal["node_complete"] = "node_complete"
    outcomes: int
    not_called: int = 0
    reused: int = 0


class NodeSkipped(_NodeEvent):
    """Every call of a node was skipped: the node is downstream of a branch that was not chosen, or its function
    returned ``SKIPPED``.

    It comes in place of ``NodeComplete``, when the node is complete. A node skipped because of a branch was never
    called, so it has no ``NodeStart``. A subgraph node has ``NodeSkipped`` when every node inside it was skipped.
    A node with some calls skipped and some not has ``NodeComplete``.
    """

    type: Literal["node_skipped"] = "node_skipped"


class NodeFailed(_NodeEvent):
    """A call of the node failed and is not retried, so the run fails.

    The function raised an exception that its ``Policy`` does not retry, or it has no retries left. The stream then
    goes on with the ``NodeComplete`` of nodes that are complete (a node that is not complete has no more events),
    and ends with ``RunEnd(status="failed")``, whose ``cause`` is this event's ``cause``. For a node ``broken``
    after ``lines`` that raises for the first line:
    ``NodeFailed(loc=("broken",), cause=Problem(type="execution_failed", loc=("broken", 0), ...))``.

    Attributes:
        cause: Why it failed. Its ``loc`` says which call: the node's id, then the call's item indexes (one index
            per loop the node is inside). Inside a subgraph, the subgraph node's id and item indexes come first:
            ``("loud", 0, "shout")``.
    """

    type: Literal["node_failed"] = "node_failed"
    cause: Problem


class NodeRetry(_NodeEvent):
    """A call failed with an error its node's ``Policy`` retries, and will be made again.

    For a node ``fetch`` with ``Policy(retries=1)`` whose first call raises ``ExternalError``:
    ``NodeRetry(loc=("fetch", 0), attempt=1, retries=1, delay=1.0, cause=Problem(type="external_failed", ...))``.
    If the retry succeeds, the node goes on to ``NodeProgress`` and ``NodeComplete`` as any other node.

    Unlike the other node events, its ``loc`` says which call, as the ``loc`` of ``cause`` does: the node's id,
    then the call's item indexes. Inside a subgraph, the subgraph node's id and item indexes come first.

    Attributes:
        attempt: Which retry will be made: 1 for the first.
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
    """A call asked a person a question, by raising ``Ask``, and is waiting for the answer.

    It comes as soon as the call's outcome is written. The calls downstream of the waiting call are not made until
    it is answered; every call that does not need its outputs is made. The node that asked has no ``NodeComplete``
    in this stream, and the stream ends with ``RunEnd(status="waiting")``. After ``run.answer(...)`` the next stream
    has the node's ``NodeComplete``.

    The event has no ``loc`` of its own: ``waiting.loc`` says which call, ``("approve", 1)`` for the call of a node
    ``approve`` for the second line.

    Attributes:
        waiting: The question and which call asked it. Its ``id`` is what ``run.answer`` takes.
    """

    type: Literal["node_waiting"] = "node_waiting"
    waiting: Waiting


class RunEnd(EventBase):
    """The last event of a stream: the run is complete, waiting for a person, or ended early.

    It comes after every other event, whatever the status. Its status is never ``running``: a stream stopped with
    ``stream(hand_back=)``, which leaves the run going, has no ``RunEnd`` and ends with ``RunCheckpoint`` instead.

    Attributes:
        status: ``complete``; ``waiting`` (for a person's answer); or how the run ended early: ``failed`` (a call
            failed), ``stopped`` (a series had more items than the run's item limit, set with ``run.with_max_items(...)``), ``cancelled`` (the
            stream's ``cancel`` was set) or ``timed_out`` (the stream's ``timeout`` passed).
        cause: The problem that ended the run early. ``None`` when the status is ``complete`` or ``waiting``.
        run: The run as the stream left it, with all its values, for the host (the program that uses conductor) to
            keep or save with ``run.to_json()``. It is not written by ``model_dump``.
    """

    type: Literal["run_end"] = "run_end"
    status: RunStatus
    cause: Problem | None = None
    run: SkipJsonSchema[InstanceOf[Run]] = Field(exclude=True)

    @classmethod
    def of(cls, run: Run) -> RunEnd:
        """Returns the ``RunEnd`` of ``run``: the status of ``run.state``, and its cause when the run ended early."""
        match run.state:
            case Going(status=status):
                return cls(status=status, run=run)
            case EndedEarly(status=status, cause=cause):
                return cls(status=status, cause=cause, run=run)


class RunCheckpoint(EventBase):
    """A copy of the run while calls are still being made, for the host to save.

    It comes in two cases:

    - With ``stream(checkpoint_every=seconds)``: after a call's outcome is written, when at least that many seconds
      have passed since the stream started or since the last checkpoint. With a node ``slow`` that takes 20
      milliseconds per line and ``checkpoint_every=0.01``, ``text="a\\nb"`` gives ``NodeStart(slow)``,
      ``RunCheckpoint``, ``NodeProgress(slow, finished=1, total=2)``, ``RunCheckpoint``, and so on. One more comes at
      the moment the run stops (a call failed, or ``cancel``, ``hand_back`` or ``timeout``) while calls are still
      running on threads, before those calls are back: if the process dies while it waits for them, the host has
      everything but them.
    - With ``stream(hand_back=event)``, once ``event`` is set: the stream makes no more calls, waits for the calls
      on threads, and ends with a ``RunCheckpoint`` in place of ``RunEnd``. The run did not end: its copy has the
      outcomes of those calls, and another process can go on with it.

    Calls run on threads only for a node whose ``Policy`` has a ``concurrency`` above 1 or an ``attempt_timeout``;
    every other call is made one at a time on the stream's own thread. A host that saves each copy loses little if
    its process dies: ``Run.from_json`` reads the copy back, and ``run.resume()`` makes only the calls that have no
    outcome in it.

    Attributes:
        run: The copy. Its status is ``running`` while the run goes on, and after ``hand_back``. A copy made as the
            run stops has the status it stops with: ``failed``, ``stopped``, ``cancelled`` or ``timed_out``. It is
            not written by ``model_dump``.
    """

    type: Literal["run_checkpoint"] = "run_checkpoint"
    run: SkipJsonSchema[InstanceOf[Run]] = Field(exclude=True)


Event = Annotated[
    RunStart | NodeStart | NodeProgress | NodeComplete | NodeSkipped | NodeFailed | NodeRetry | NodeWaiting | RunCheckpoint | RunEnd,
    Field(discriminator="type"),
]
Event.__doc__ = "Any event ``run.stream()`` yields, told apart by ``type``: ``TypeAdapter(Event)`` reads one back from JSON."
