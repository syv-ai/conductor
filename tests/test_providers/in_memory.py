"""A ``RunStore`` in memory that a test can wait on, and the graphs the worker's and the routes' tests hold mid-leg."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
from collections.abc import Callable
from typing import Any

from conductor import Ask, GraphRecord, NodeRegistry, Series, make_graph
from conductor_providers.fastapi import RunChanges, StoredRun
from conductor_providers.fastapi.worker import RunAsRead, RunView


class MemoryStore:
    """A ``RunStore`` in a dict. ``until`` waits for a run's row to change, so a test awaits what a worker writes
    where it would sleep and read again."""

    def __init__(self) -> None:
        self.rows: dict[str, StoredRun] = {}
        self.written: dict[str, list[str]] = {}
        self._changed = asyncio.Event()

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

    async def until(self, run_id: str, holds: Callable[[StoredRun], bool]) -> StoredRun:
        """The run's row once ``holds`` is true of it, awaited on the store's own writes."""
        async with asyncio.timeout(10):
            while not holds(self.rows[run_id]):
                self._changed.clear()
                await self._changed.wait()
        return self.rows[run_id]

    async def ended(self, run_id: str) -> RunView:
        """The run as its view once no leg is wanted for it."""
        return RunView.of(RunAsRead.of(await self.until(run_id, lambda row: row.status != "running")))

    async def stored_events(self, run_id: str) -> list[dict[str, Any]]:
        """Every event stored for the run, in order."""
        return [json.loads(body) for _, body in await self.events(run_id, -1)]


def asking(items: bool = False) -> tuple[NodeRegistry, GraphRecord, threading.Event, dict[str, Any], str]:
    """A graph whose leg waits on ``gate`` beside a question, its inputs, and the waiting id the question gets; with
    ``items``, beside a Series of three items read one at a time too, for a ceiling of 1 to stop."""
    registry, gate = NodeRegistry(), threading.Event()

    @registry.node("busy", title="Busy", description="")
    def busy(text: str) -> str:
        gate.wait(5)
        return text

    @registry.node("approve", title="Approve", description="")
    def approve(text: str) -> bool:
        raise Ask("Go on?")

    @registry.node("after", title="After", description="")
    def after(ok: bool) -> bool:
        return ok

    @registry.node("three", title="Three", description="")
    def three(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("each", title="Each", description="")
    def each(i: int) -> int:
        return i

    nodes: list[Any] = [busy, approve, after, three, each] if items else [busy, approve, after]
    graph = make_graph(*nodes, edges=[("approve.result", "after.ok"), *([("three.result", "each.i")] if items else [])])
    inputs: dict[str, Any] = {"busy.text": "x", "approve.text": "y", **({"three.n": 3} if items else {})}
    gate.set()
    (waiting,) = registry.load(graph).run(inputs).waiting
    gate.clear()
    return registry, graph, gate, inputs, waiting.id


def gated() -> tuple[NodeRegistry, GraphRecord, dict[str, threading.Event], list[str]]:
    """A graph whose first leg waits on ``first`` beside a question, and whose leg after the answer waits on ``second``
    on its first item of two; ``held`` is set once that item is called."""
    registry, gates, calls = NodeRegistry(), {name: threading.Event() for name in ("first", "second", "held")}, []

    @registry.node("items", title="Items", description="")
    def items(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("busy", title="Busy", description="")
    def busy(text: str) -> str:
        gates["first"].wait(5)
        return text

    @registry.node("approve", title="Approve", description="")
    def approve(text: str) -> bool:
        raise Ask("Go on?")

    @registry.node("hold", title="Hold", description="")
    def hold(item: str, ok: bool) -> str:
        calls.append(item)
        gates["held"].set()
        gates["second"].wait(5)
        return item

    graph = make_graph(items, busy, approve, hold, edges=[("items.result", "hold.item"), ("approve.result", "hold.ok")])
    return registry, graph, gates, calls
