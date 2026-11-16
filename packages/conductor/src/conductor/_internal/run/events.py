"""The events a host gets while ``run.stream()`` runs a graph.

A host that shows a run as it goes, on a web page say, listens to these events. There is one class per event, and
``Event`` is the union of them all.

Take the first graph in conductor's README. It has three nodes::

    lines(text: str) -> Series[str]          # called once; returns ["a", "b"]
    shout(line: str, mark: str = "!") -> str # wired to lines.result: called once per line
    join(texts: Series[str]) -> str          # wired to shout.result: called once, with ["A!", "B!"]

``async for event in graph.with_inputs(text="a\\nb").stream()`` gets these events, in this order::

    RunStart(waiting=())
    NodeStart(loc=("lines",))
    NodeStart(loc=("shout",))
    NodeProgress(loc=("shout",), count=CallCount(finished=1, known=2, total=2, ...))
    NodeStart(loc=("join",))
    NodeComplete(loc=("lines",), count=CallCount(finished=1, known=1, total=1, ...))
    NodeProgress(loc=("shout",), count=CallCount(finished=2, known=2, total=2, ...))
    NodeComplete(loc=("shout",), count=CallCount(finished=2, known=2, total=2, ...))
    NodeComplete(loc=("join",), count=CallCount(finished=1, known=1, total=1, ...))
    RunEnd(status="complete")

The ``...`` stands for the count's ``waiting``, ``skipped`` and ``not_called``, all 0 here.

``RunStart`` always comes first and ``RunEnd`` last. The one exception is a stream stopped with ``hand_back``,
which ends with ``RunCheckpoint`` instead. A node's ``NodeStart``, if it has one, comes before every other event
about that node, and its ``NodeComplete`` comes after all of them. Events about different nodes are mixed together.

Two rules explain the order above:

- ``NodeProgress`` comes at most four times a second for each node. ``shout``'s second call was made before
  ``NodeStart(join)``, but ``finished=2`` only shows up in the last count, just before ``NodeComplete(shout)``.
- The engine looks for complete nodes only when no call is ready to be made. Here that first happens after
  ``join`` is called, so even ``lines`` only completes after ``NodeStart(join)``.

Every event is a frozen pydantic model with a ``type``, such as ``"node_complete"``. ``event.model_dump(mode="json")``
turns it into JSON, to send to a web page for example. ``TypeAdapter(Event).validate_json(text)`` turns that JSON
back into an equal event. ``RunEnd`` and ``RunCheckpoint`` also hand the host the run itself, as ``event.run``, but
the run isn't in the JSON. The rules are in the reference: ``python -m conductor.about events``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import ConfigDict, Field, PrivateAttr

from conductor._internal.base import ConductorModel, Loc
from conductor._internal.problem import Problem

from .progress import CallCount
from .state import RunStatus
from .waiting import Waiting

if TYPE_CHECKING:
    from . import Run


class EventBase(ConductorModel):
    """The base class of every event.

    The JSON schema of an event marks every field as required, even ``type`` and the fields that have a default.
    That's true to what it writes, since ``model_dump`` always writes every field. So a client generated from the
    schema can count on each field being there.
    """

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class _NodeEvent(EventBase):
    """The base class of every event about one node, such as ``NodeStart(loc=("shout",))``.

    It is never made itself, only its subclasses are.

    Attributes:
        loc: Which node the event is about, by its id: ``("shout",)``. For a node inside a subgraph, the subgraph
            node's id comes first, so ``shout`` inside the subgraph node ``loud`` is ``("loud", "shout")``. The loc
            names the node, not one of its calls. A node inside a loop gets one ``NodeStart``, one ``NodeComplete``
            and one ``NodeFailed`` for all its calls together. When an event is about one call, the record it
            carries says which call: the ``loc`` of its ``Problem`` or of its ``Waiting``.
    """

    loc: Loc


class RunStart(EventBase):
    """The first event of every stream, such as ``RunStart(waiting=())``.

    Attributes:
        waiting: The calls that were waiting for a person's answer when the stream started, in the same order as
            ``run.waiting``. Say a run stopped to wait for an answer, and is streamed again before anyone answers.
            Its stream starts with those calls here, and then ends right away with ``RunEnd(status="waiting")``.
    """

    type: Literal["run_start"] = "run_start"
    waiting: tuple[Waiting, ...] = ()


class NodeStart(_NodeEvent):
    """A node has started: its first call is about to be made, or it got its first outcome without a call.

    A node can get an outcome without being called in two ways. The outcome may be reused from the previous run,
    the one ``run.with_inputs(...)`` or ``run.again(...)`` was called on. Or the function isn't called because an
    argument is ``None``. Either way the node has started.

    A node that is skipped on every call, because it sits on a skipped branch, gets no ``NodeStart``.
    It only gets ``NodeComplete``, with ``NodeProgress`` before it if it is inside a loop. A node whose function
    returned ``SKIPPED`` was still called, so it does get its ``NodeStart``.

    A node inside a subgraph starts right after the subgraph node does: ``NodeStart(loc=("loud",))``, then
    ``NodeStart(loc=("loud", "shout"))``.

    A node gets at most one ``NodeStart`` in a stream. A node that already had an outcome when a stream started gets
    none in that stream. For example, after a person answers, the next stream has ``NodeComplete`` for the node that
    asked, with no ``NodeStart`` before it. A node with no outcome yet, such as one whose only call failed, starts
    again in the next stream.
    """

    type: Literal["node_start"] = "node_start"


class NodeProgress(_NodeEvent):
    """Sent while a node inside a loop runs, to say how far it has got.

    Its ``count`` says how many of the node's calls have an outcome, and how many calls there will be. It is the
    same ``CallCount`` that ``run.progress`` shows. The event is sent at most four times a second.

    It comes after a call's outcome is written. One more comes with the final count, before the node's
    ``NodeComplete`` or at the end of the stream, unless the count hasn't changed. A node that isn't inside a loop
    has only one call, so it gets no ``NodeProgress``.

    In the README example, ``shout`` first has
    ``count=CallCount(finished=1, waiting=0, skipped=0, not_called=0, known=2, total=2)``, and then the same with
    ``finished=2``.

    Attributes:
        count: The node's ``CallCount``, the same as ``run.progress.nodes["shout"]`` at that moment.
    """

    type: Literal["node_progress"] = "node_progress"
    count: CallCount


class NodeComplete(_NodeEvent):
    """Sent when a node is finished: every call of it has an outcome.

    None of those outcomes is waiting for a person, and no call of the node, or of a node upstream of it, is still
    running. Each outcome may be done, answered, not called or skipped. A node gets at most one ``NodeComplete`` in
    a run.

    The event doesn't carry the node's values. They are in the run, in ``run.results`` and ``run.not_called``, so
    the event stays small however large the values are.

    Its ``count`` is the node's ``CallCount``, the same as ``run.progress`` has for the node. Some examples. Run the
    README graph with ``text="a\\n \\nb"``, and put a node ``strip`` before ``shout`` that returns ``None`` for
    the line ``" "``. Then ``shout``'s count has ``finished=3, not_called=1``. A node was skipped when every one of
    its calls was: ``finished > 0`` and ``skipped == finished``. A node inside a loop over an empty list has
    ``finished=0, skipped=0``, and wasn't skipped. A subgraph node completes once every node inside it has, and its
    count is the sum of theirs.

    Attributes:
        count: The node's ``CallCount``, the same as ``run.progress.nodes["shout"]`` at that moment. Every call is
            finished, so ``finished``, ``known`` and ``total`` are equal, and ``waiting`` is 0.
        reused: How many of the finished calls this stream reused from the previous run instead of making the
            call. For a subgraph node, it's the sum of the nodes inside it.
    """

    type: Literal["node_complete"] = "node_complete"
    count: CallCount
    reused: int = 0


class NodeFailed(_NodeEvent):
    """A call of the node failed and won't be retried, so the run ends.

    The function raised an exception that the node's ``Policy`` doesn't retry, or it ran out of retries. After this
    event, the stream still sends ``NodeComplete`` for the nodes that are complete. A node that isn't complete gets
    no more events. The stream ends with ``RunEnd(status="failed")``, with this event's ``cause``.

    If the stream's ``cancel`` was set by then, the stream ends with ``RunEnd(status="cancelled")`` instead. That's
    because a node can be written to stop a slow call when a cancel comes, and it stops by raising. The failed call
    has no outcome, so a resume makes it again.

    For example, take a node ``broken`` after ``lines`` that raises on the first line. It gets
    ``NodeFailed(loc=("broken",), cause=Problem(type="execution_failed", loc=("broken", 0), ...))``.

    Attributes:
        cause: Why the call failed. Its ``loc`` says which call it was: the node's id, then the call's item
            indexes, one index for each loop the node is inside. Inside a subgraph, the subgraph node's id and item
            indexes come first: ``("loud", 0, "shout")``.
    """

    type: Literal["node_failed"] = "node_failed"
    cause: Problem


class NodeRetry(_NodeEvent):
    """A call failed, and will be tried again, since the node's ``Policy`` retries that error.

    Say a node ``fetch`` has ``Policy(retries=1)``, and its first call raises ``ExternalError``. It gets
    ``NodeRetry(loc=("fetch",), attempt=1, retries=1, delay=1.0, cause=Problem(type="external_failed", loc=("fetch", 0), ...))``.
    If the retry works, the node goes on to ``NodeProgress`` and ``NodeComplete`` like any other node.

    Attributes:
        attempt: Which retry comes next: 1 for the first.
        retries: How many retries the node's ``Policy`` allows.
        delay: How many seconds from now the retry is made.
        cause: Why the call failed. Its ``loc`` says which call it was, as in ``NodeFailed``.
    """

    type: Literal["node_retry"] = "node_retry"
    attempt: int
    retries: int
    delay: float
    cause: Problem


class NodeWaiting(_NodeEvent):
    """A call asked a person a question, by raising ``Ask``, and is waiting for the answer.

    It comes as soon as the call's outcome is written. The calls downstream of the waiting call wait for the answer
    too. Every other call is still made. The node that asked gets no ``NodeComplete`` in this stream, and the stream
    ends with ``RunEnd(status="waiting")``. Once someone answers with ``run.answer(...)``, the next stream has the
    node's ``NodeComplete``.

    Say a node ``approve`` asks about each line. The call for the second line gets
    ``NodeWaiting(loc=("approve",), waiting=Waiting(loc=("approve", 1), ...))``. The event's ``loc`` names the
    node, and ``waiting.loc`` says which call it is.

    Attributes:
        waiting: The question, and which call asked it. Its ``id`` is what ``run.answer`` takes.
    """

    type: Literal["node_waiting"] = "node_waiting"
    waiting: Waiting


class _EventWithRun(EventBase):
    """The base class of the two events that also hand the host the run: ``RunEnd`` and ``RunCheckpoint``.

    The run rides along with the event but isn't part of it. It isn't in the event's JSON, and two events are
    compared without it. Only an event that a stream gives has a run. So an event read back from JSON equals the one
    the stream gave, but has no ``run``.
    """

    _run: Run | None = PrivateAttr(default=None)

    @property
    def run(self) -> Run:
        """The run, for the host to keep, or to save with ``run.to_json()``.

        The host is the program that uses conductor.

        Raises:
            AttributeError: The event was read back from JSON or made by hand, so it has no run.
        """
        if self._run is None:
            raise AttributeError(
                f"this {type(self).__name__} carries no run: it was read back from JSON or made by hand, "
                "and only an event a stream yields has the run"
            )
        return self._run

    def __eq__(self, other: object) -> bool:
        """Whether ``other`` is the same kind of event with the same fields. The run is left out.

        For any other kind of object, Python asks that object instead, as pydantic does. That's what makes
        ``event == unittest.mock.ANY`` true.
        """
        if type(other) is not type(self):
            return NotImplemented
        return self.__dict__ == other.__dict__


def with_run[E: _EventWithRun](event: E, run: Run) -> E:
    """Returns ``event`` with ``run`` attached, the way ``run.stream()`` gives its ``RunEnd`` or ``RunCheckpoint``."""
    event._run = run
    return event


class RunEnd(_EventWithRun):
    """The last event of a stream: the run is complete, is waiting for a person, or ended early.

    It comes after every other event, whatever the status. Its status is never ``running``. A stream stopped with
    ``stream(hand_back=)`` leaves the run going, so it gets no ``RunEnd``. It ends with ``RunCheckpoint`` instead.

    Attributes:
        status: ``complete``, ``waiting`` (for a person's answer), or how the run ended early. It ends early as
            ``failed`` when a call failed, and as ``stopped`` when a series had more items than the run's item
            limit, set with ``run.with_max_items(...)``. It ends as ``cancelled`` when the stream's ``cancel`` was
            set, and as ``timed_out`` when the stream's ``timeout`` passed.
        cause: The problem that ended the run early, or ``None`` when the status is ``complete`` or ``waiting``.
        run: The run as the stream left it, with all its values. It isn't in the event's JSON.
    """

    type: Literal["run_end"] = "run_end"
    status: RunStatus
    cause: Problem | None = None


class RunCheckpoint(_EventWithRun):
    """A copy of the run, handed over while calls are still being made, so the host can save it.

    A host that saves each copy loses little if its process dies. ``Run.from_json`` loads the copy back, and
    ``run.resume()`` makes only the calls that have no outcome in it.

    It comes in two cases. The first is ``stream(checkpoint_every=seconds)``. Then a copy comes after a call's
    outcome is written, once at least that many seconds have passed since the stream started or since the last
    copy. Say a node ``slow`` takes 20 milliseconds per line, and ``checkpoint_every=0.01``. Then ``text="a\\nb"``
    gives ``NodeStart(slow)``, ``RunCheckpoint``, ``NodeProgress(slow)``, ``RunCheckpoint``, and so on.

    With ``checkpoint_every``, one more copy comes when the run stops while calls are still running on threads. The
    run may stop because a call failed, or because of ``cancel``, ``hand_back`` or ``timeout``. That copy comes
    before those calls are back. If the process dies while it waits for them, the host has everything but them.

    The second case is ``stream(hand_back=event)``, once ``event`` is set. The stream makes no more calls, waits
    for the calls on threads, and ends with a ``RunCheckpoint`` instead of ``RunEnd``. The run hasn't ended. Its
    copy has the outcomes of those calls, and another process can go on with it.

    Calls only run on threads for a node whose ``Policy`` has a ``concurrency`` above 1 or an ``attempt_timeout``.
    Every other call is made one at a time, on the stream's own thread.

    Its JSON is just ``{"type": "run_checkpoint"}``. The copy is for the host, not for a web page.

    Attributes:
        run: The copy of the run. Its status is ``running`` while the run goes on, and after ``hand_back``. A copy
            made as the run stops early has the status it stops with: ``failed``, ``stopped``, ``cancelled`` or
            ``timed_out``. ``running`` is never saved, though. A copy loaded back from ``to_json()`` has the status
            its outcomes say.
    """

    type: Literal["run_checkpoint"] = "run_checkpoint"


# Any event ``run.stream()`` gives. The events are told apart by their ``type`` field, so ``TypeAdapter(Event)``
# can load any of them back from JSON. It is a ``type`` statement so that a JSON schema that uses it gives it a
# name: a model with a field ``event: Event`` gets ``{"$ref": "#/$defs/Event"}``, and FastAPI lists it as the
# component ``Event``.
type Event = Annotated[
    RunStart
    | NodeStart
    | NodeProgress
    | NodeComplete
    | NodeFailed
    | NodeRetry
    | NodeWaiting
    | RunCheckpoint
    | RunEnd,
    Field(discriminator="type"),
]
