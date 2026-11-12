"""A ``RunStore`` in memory that a test can wait on, and the graphs the worker's and the routes' tests stop in the middle of a resume call."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from asyncio import Event as AsyncioEvent
from collections.abc import Callable
from threading import Event as ThreadEvent
from typing import Any

from conductor import Ask, Graph, GraphRecord, NodeRegistry, Run, Series, make_graph
from conductor_providers.fastapi import RunChanges, StoredRun
from conductor_providers.fastapi.router import RunView
from conductor_providers.fastapi.stored_run import Outcome, RunAsRead, StoredState


class MemoryStore:
    """A ``RunStore`` in a dict. ``until`` waits for a run's row to change, so a test awaits what a worker writes
    where it would sleep and read again."""

    def __init__(self) -> None:
        self.rows: dict[str, StoredRun] = {}
        self.written: dict[str, list[str]] = {}
        self._changed = AsyncioEvent()

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
        return [row for row in self.rows.values() if row.lease_until is not None and row.lease_until < now]

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
        return RunView.of(RunAsRead.of(await self.until(run_id, lambda row: row.status != "running")))

    async def stored_events(self, run_id: str) -> list[dict[str, Any]]:
        """Every event stored for the run, in order."""
        return [json.loads(body) for _, body in await self.events(run_id, -1)]


def left_running(run_id: str, graph: GraphRecord, run: Run, *, owner: str | None) -> StoredRun:
    """A row of ``anna``'s as a worker that died in a resume call left it: running, saved at ``run``, its lease run out."""
    return StoredRun(
        id=run_id, graph=graph.model_dump_json(), caller='"anna"', run=run.to_json(), **StoredState.started().columns(),
        outcome=Outcome.empty().model_dump_json(), rev=0, owner=owner, lease_until=0.0,
    )


def asking(items: bool = False) -> tuple[NodeRegistry, GraphRecord, ThreadEvent, dict[str, Any], str]:
    """A graph with a node that waits on ``gate`` beside a node that asks a question: the registry, the graph, the
    gate, its inputs and the waiting id of the question. With ``items``, the graph also loops over three items, so
    that an item limit of 1 stops the run."""
    registry, gate = NodeRegistry(), ThreadEvent()

    @registry.node("busy", title="Busy", description="For a test.")
    def busy(text: str) -> str:
        gate.wait(5)
        return text

    @registry.node("approve", title="Approve", description="For a test.")
    def approve(text: str) -> bool:
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

    nodes: list[Any] = [busy, approve, after, three, each] if items else [busy, approve, after]
    graph = make_graph(*nodes, edges=[("approve.result", "after.ok"), *([("three.result", "each.i")] if items else [])])
    inputs: dict[str, Any] = {"busy.text": "x", "approve.text": "y", **({"three.n": 3} if items else {})}
    gate.set()
    (waiting,) = Graph.compile(graph, registry).run(inputs).waiting
    gate.clear()
    return registry, graph, gate, inputs, waiting.id


def gated() -> tuple[NodeRegistry, GraphRecord, dict[str, ThreadEvent], list[str]]:
    """A graph whose first resume call waits on ``first`` beside a question, and whose resume call after the answer
    waits on ``second`` in the call for the first of two items; ``per_item_called`` is set once that call started.
    ``calls`` lists the items called."""
    registry, gates, calls = NodeRegistry(), {name: ThreadEvent() for name in ("first", "second", "per_item_called")}, []

    @registry.node("items", title="Items", description="For a test.")
    def items(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("busy", title="Busy", description="For a test.")
    def busy(text: str) -> str:
        gates["first"].wait(5)
        return text

    @registry.node("approve", title="Approve", description="For a test.")
    def approve(text: str) -> bool:
        raise Ask("Go on?")

    @registry.node("per_item", title="Per item", description="For a test.")
    def per_item(item: str, ok: bool) -> str:
        calls.append(item)
        gates["per_item_called"].set()
        gates["second"].wait(5)
        return item

    graph = make_graph(items, busy, approve, per_item, edges=[("items.result", "per_item.item"), ("approve.result", "per_item.ok")])
    return registry, graph, gates, calls
