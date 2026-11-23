"""Helpers for the tests of the worker and of the routes.

``MemoryStore`` is a ``RunStore`` kept in a dict, which a test can wait on. The functions below build small graphs
whose nodes wait on events. That lets a test pause a run in the middle of a resume call and check what the worker
does then.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from asyncio import Event as AsyncioEvent
from collections.abc import Callable
from threading import Event as ThreadEvent
from typing import Any

from conductor import (
    Ask,
    Context,
    Graph,
    GraphRecord,
    NodeError,
    NodeRegistry,
    Run,
    Series,
    make_graph,
)
from conductor_providers.fastapi import RunChanges, RunView, StoredRun, Worker

#: The ``state`` column of a run that has just created, written out by hand as a host's database stores it. Tests that
#: write a row the way an earlier worker left it use this text. If the stored form changes in a way that would break
#: the rows already in hosts' databases, those tests break too.
BEGUN_RUN_STATE = json.dumps(
    {"status": "running", "cause": None, "queued": [], "refused": {}, "cancel_asked": False}
)


#: A run filter that lets every run through, for tests that don't check access.
EVERY_RUN = lambda metadata: True  # noqa: E731


class MemoryStore:
    """A ``RunStore`` that keeps its rows in a dict.

    A test can wait on this store instead of sleeping and reading again. ``until`` waits until a run's row meets a
    condition, and checks again after each write. ``swept`` is set once a worker has asked for the runs whose lease
    ran out.
    """

    def __init__(self) -> None:
        self.rows: dict[str, StoredRun] = {}
        self.written: dict[str, list[str]] = {}
        self._changed = AsyncioEvent()
        self.swept = AsyncioEvent()

    async def create(self, run: StoredRun) -> None:
        self.rows[run.id], self.written[run.id] = run, []
        self._changed.set()

    async def get(self, run_id: str) -> StoredRun | None:
        return self.rows.get(run_id)

    async def update(
        self, run_id: str, rev: int, changes: RunChanges, event: str | None = None
    ) -> bool:
        row = self.rows.get(run_id)
        if row is None or row.rev != rev:
            return False
        self.rows[run_id] = dataclasses.replace(row, **changes, rev=rev + 1)
        if event is not None:
            self.written[run_id].append(event)
        self._changed.set()
        return True

    async def expired(self, now: float) -> list[StoredRun]:
        self.swept.set()
        return [
            row
            for row in self.rows.values()
            if row.lease_until is not None and row.lease_until < now
        ]

    async def add_event(self, run_id: str, owner: str, body: str) -> bool:
        if self.rows[run_id].owner != owner:
            return False
        self.written[run_id].append(body)
        return True

    async def events(self, run_id: str, after: int) -> list[tuple[int, str]]:
        return [(seq, body) for seq, body in enumerate(self.written[run_id]) if seq > after]

    async def until(self, run_id: str, condition: Callable[[StoredRun], bool]) -> StoredRun:
        """Waits until ``condition`` is true of the run's row, then returns the row.

        It checks again after each write to the store, and gives up after 10 seconds.
        """
        async with asyncio.timeout(10):
            while not condition(self.rows[run_id]):
                self._changed.clear()
                await self._changed.wait()
        return self.rows[run_id]

    async def ended(self, run_id: str) -> RunView:
        """Waits until the run's row is no longer running, then returns the run as a reader sees it."""
        await self.until(run_id, lambda row: row.status != "running")
        return await self.view(run_id)

    async def view(self, run_id: str) -> RunView:
        """Returns the run as a reader would see it now.

        It reads a copy of the row through a separate worker that serves nothing. That way, a store that counts its
        reads doesn't count these.
        """
        copy = MemoryStore()
        copy.rows[run_id] = self.rows[run_id]
        reader: Worker = Worker(copy, nodes_for=lambda metadata: NodeRegistry())
        return await reader.read(run_id, allows=EVERY_RUN)

    def state(self, run_id: str) -> dict[str, Any]:
        """Returns the run's ``state`` column, parsed from the JSON the worker wrote."""
        written: dict[str, Any] = json.loads(self.rows[run_id].state)
        return written

    async def stored_events(self, run_id: str) -> list[dict[str, Any]]:
        """Returns every event stored for the run, in order."""
        return [json.loads(body) for _, body in await self.events(run_id, -1)]


def left_running(run_id: str, graph: GraphRecord, run: Run, *, owner: str | None) -> StoredRun:
    """Returns a stored row the way a worker leaves it when it dies during a resume call.

    The run belongs to ``anna``. It is still ``running``, it is saved as ``run``, and its lease has run out.
    """
    return StoredRun(
        id=run_id,
        graph=graph.model_dump_json(),
        metadata='{"user": "anna"}',
        run=run.to_json(),
        status="running",
        state=BEGUN_RUN_STATE,
        outcome=json.dumps({"waiting": [], "outputs": None, "progress": None}),
        rev=0,
        owner=owner,
        lease_until=0.0,
    )


def asking(
    items: bool = False,
) -> tuple[NodeRegistry, GraphRecord, ThreadEvent, dict[str, Any], str]:
    """Builds a graph with a node that waits on a gate, beside a node that asks a question.

    ``busy`` waits until ``gate`` is set, and ``confirm`` asks a question. Nodes that no edge puts in order are called
    in the order of their ids, so ``busy`` is called before ``confirm`` asks.

    With ``items``, the graph also loops over three items, so an item limit of 1 stops the run.

    Returns:
        The registry, the graph, the gate, the graph's inputs, and the waiting id of the question.
    """
    registry, gate = NodeRegistry(), ThreadEvent()

    @registry.node("busy", title="Busy", description="For a test.")
    def busy(text: str) -> str:
        gate.wait(5)
        return text

    @registry.node("confirm", title="Confirm", description="For a test.", asks=True)
    def confirm(question: str) -> bool:
        raise Ask("Go on?")

    @registry.node("after", title="After", description="For a test.")
    def after(ok: bool) -> bool:
        return ok

    @registry.node("three", title="Three", description="For a test.")
    def three(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("each", title="Each", description="For a test.")
    def each(i: int) -> int:
        return i

    nodes: list[Any] = [busy, confirm, after, three, each] if items else [busy, confirm, after]
    edges = [("confirm.result", "after.ok"), *([("three.result", "each.i")] if items else [])]
    graph = make_graph(
        *nodes,
        edges=edges,
        outputs={
            "busy": "busy.result",
            "after": "after.result",
            **({"each": "each.result"} if items else {}),
        },
    )
    inputs: dict[str, Any] = {"text": "x", "question": "y", **({"n": 3} if items else {})}
    gate.set()
    (waiting,) = Graph.compile(graph, registry).run(inputs).waiting
    gate.clear()
    return registry, graph, gate, inputs, waiting.id


def gated() -> tuple[NodeRegistry, GraphRecord, dict[str, ThreadEvent], list[str]]:
    """Builds a graph whose two resume calls each wait on a gate of their own.

    In the first resume call, ``busy`` waits on the gate ``first``, beside the question ``approve``. Once the question
    is answered, the next resume call runs ``per_item`` for each item, and its first call waits on the gate
    ``second``. ``per_item_called`` is set as soon as that call starts.

    Returns:
        The registry, the graph, the three gates by name, and ``calls``, the items ``per_item`` was called with.
    """
    registry, gates, calls = (
        NodeRegistry(),
        {name: ThreadEvent() for name in ("first", "second", "per_item_called")},
        [],
    )

    @registry.node("items", title="Items", description="For a test.")
    def items(csv: str) -> Series[str]:
        return csv.split(",")

    @registry.node("busy", title="Busy", description="For a test.")
    def busy(text: str) -> str:
        gates["first"].wait(5)
        return text

    @registry.node("approve", title="Approve", description="For a test.", asks=True)
    def approve(question: str) -> bool:
        raise Ask("Go on?")

    @registry.node("per_item", title="Per item", description="For a test.")
    def per_item(item: str, ok: bool) -> str:
        calls.append(item)
        gates["per_item_called"].set()
        gates["second"].wait(5)
        return item

    graph = make_graph(
        items,
        busy,
        approve,
        per_item,
        edges=[("items.result", "per_item.item"), ("approve.result", "per_item.ok")],
        outputs={"busy": "busy.result", "per_item": "per_item.result"},
    )
    return registry, graph, gates, calls


@dataclasses.dataclass(frozen=True)
class Stoppable:
    """A run's context that carries the stop event of the current resume call.

    The worker gives its ``context`` function a new stop event for each resume call, and ``stoppable`` wraps it in
    this class.

    Attributes:
        stop: The resume call's stop event.
    """

    stop: ThreadEvent


@dataclasses.dataclass
class StopEventGraph:
    """The graph that ``stopping`` builds, with the events a test waits on.

    The graph has three nodes:

    - ``ask`` asks a question, and is called first.
    - ``stream`` waits up to ``seconds`` for its resume call's stop event, but only on its first call. If the event
      is set, it raises, as a node that stops a slow call does. A later call returns at once.
    - ``after`` runs once the question is answered, and records whether its own resume call's stop event is set.

    Attributes:
        registry: The nodes, on a registry whose context is ``Stoppable``.
        graph: The graph.
        waiting_id: The question's waiting id.
        started: Set once ``stream`` has started.
        returned: Set once a call of ``stream`` has returned or raised.
        seen: Whether each call of ``stream`` and ``after`` saw its stop event set, in the order they ended.
    """

    registry: NodeRegistry[Stoppable]
    graph: GraphRecord
    waiting_id: str
    started: ThreadEvent
    returned: ThreadEvent
    seen: list[bool]


def stoppable(metadata: object, stop: ThreadEvent) -> Stoppable:
    """A ``context`` function for a worker. It ignores the run's metadata and wraps the resume call's stop event."""
    return Stoppable(stop)


def stopping() -> StopEventGraph:
    """Builds a ``StopEventGraph`` graph.

    It runs the graph once on its own to learn the question's waiting id. Then it clears the events and ``seen``, so
    the test starts fresh.
    """
    registry, started, returned, seen = (
        NodeRegistry(context_type=Stoppable),
        ThreadEvent(),
        ThreadEvent(),
        [],
    )

    @registry.node("ask", title="Ask", description="For a test.", asks=True)
    def ask(question: str) -> bool:
        raise Ask("Go on?")

    @registry.node("stream", title="Stream", description="For a test.")
    def stream(ctx: Context[Stoppable], text: str, seconds: float) -> str:
        first = not seen
        started.set()
        try:
            seen.append(ctx.stop.wait(seconds if first else 0))
            if seen[-1]:
                raise NodeError("The call was stopped.", code="stopped")
            return text
        finally:
            returned.set()

    @registry.node("after", title="After", description="For a test.")
    def after(ctx: Context[Stoppable], text: str, ok: bool) -> str:
        seen.append(ctx.stop.is_set())
        return text

    graph = make_graph(
        ask,
        stream,
        after,
        edges=[("stream.result", "after.text"), ("ask.result", "after.ok")],
        outputs={"after": "after.result"},
    )
    run = Graph.compile(graph, registry).run(
        {"question": "y", "text": "x", "seconds": 0.0}, context=Stoppable(ThreadEvent())
    )
    (waiting,) = run.waiting
    started.clear()
    returned.clear()
    seen.clear()
    return StopEventGraph(registry, graph, waiting.id, started, returned, seen)
