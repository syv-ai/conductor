"""A ``RunStore`` in memory that a test can wait on, and the graphs the worker's and the routes' tests stop in the middle of a resume call."""

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

#: A just-begun run's ``state`` column in its stored form, as hosts' databases hold it: the rows a test writes as an
#: earlier worker left them read it, so a change to the form that would break those databases breaks them too.
BEGUN_RUN_STATE = json.dumps(
    {"status": "running", "cause": None, "queued_answers": [], "refused": {}, "cancel_asked": False}
)


class MemoryStore:
    """A ``RunStore`` in a dict. ``until`` waits for a run's row to change, so a test awaits what a worker writes
    where it would sleep and read again; ``swept`` is set once a worker has asked for the runs whose lease ran out."""

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

    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        row = self.rows.get(run_id)
        if row is None or row.rev != rev:
            return False
        self.rows[run_id] = dataclasses.replace(row, **changes, rev=rev + 1)
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
        """The run's row once ``condition`` is true of it, awaited on the store's own writes."""
        async with asyncio.timeout(10):
            while not condition(self.rows[run_id]):
                self._changed.clear()
                await self._changed.wait()
        return self.rows[run_id]

    async def ended(self, run_id: str) -> RunView:
        """The run as its view once its row is no longer running."""
        await self.until(run_id, lambda row: row.status != "running")
        return await self.view(run_id)

    async def view(self, run_id: str) -> RunView:
        """The run as a reader is shown it now, read by a worker that serves nothing, from a copy of its row, so a
        store that counts its reads counts none of these."""
        copy = MemoryStore()
        copy.rows[run_id] = self.rows[run_id]
        reader = Worker(store=copy, caller_type=object, nodes_for=lambda caller: NodeRegistry())
        return (await reader.read(run_id)).view()

    def state(self, run_id: str) -> dict[str, Any]:
        """The run's ``state`` column, as the JSON the worker wrote."""
        written: dict[str, Any] = json.loads(self.rows[run_id].state)
        return written

    async def stored_events(self, run_id: str) -> list[dict[str, Any]]:
        """Every event stored for the run, in order."""
        return [json.loads(body) for _, body in await self.events(run_id, -1)]


def left_running(run_id: str, graph: GraphRecord, run: Run, *, owner: str | None) -> StoredRun:
    """A row of ``anna``'s as a worker that died in a resume call left it: running, saved at ``run``, its lease run out."""
    return StoredRun(
        id=run_id,
        graph=graph.model_dump_json(),
        caller='"anna"',
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
    """A graph with a node that waits on ``gate`` beside a node that asks a question: the registry, the graph, the
    gate, its inputs and the waiting id of the question. ``busy`` is called before ``confirm`` asks, as nodes no edge
    orders are called in the order of their ids. With ``items``, the graph also loops over three items, so that an
    item limit of 1 stops the run."""
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
    """A graph whose first resume call waits on ``first`` beside a question, and whose resume call after the answer
    waits on ``second`` in the call for the first of two items; ``per_item_called`` is set once that call started.
    ``calls`` lists the items called."""
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
    """A run's context that is the stop event the worker gives its ``context`` function for each resume call.

    Attributes:
        stop: The resume call's stop event.
    """

    stop: ThreadEvent


@dataclasses.dataclass
class Stopping:
    """A graph whose node ``stream`` waits up to ``seconds`` for its resume call's stop event in its first call, and
    raises if it is set, as a node that stops a slow call does (a later call returns at once), beside a question
    that is asked first, and whose node ``after`` reads the stop event of its own resume call once the question is
    answered.

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


def stoppable(caller: object, stop: ThreadEvent) -> Stoppable:
    """A worker's ``context`` function: the resume call's stop event, whoever the caller."""
    return Stoppable(stop)


def stopping() -> Stopping:
    """A ``Stopping`` graph, its waiting id read from a run on its own."""
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
    return Stopping(registry, graph, waiting.id, started, returned, seen)
