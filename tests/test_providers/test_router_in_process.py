"""``run_router`` served in this process over a store in memory, for what the page's own app doesn't set up: a host's
own ``may_see`` and a registry per caller."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import threading
import time
from typing import Annotated, Any, TypedDict

import pytest

httpx = pytest.importorskip("httpx")

import polars as pl  # noqa: E402
from conductor import Ask, GraphRecord, NodeRecord, NodeRegistry, Series, Table, make_graph  # noqa: E402
from conductor_providers.fastapi import RunChanges, StoredRun, run_router  # noqa: E402
from fastapi import FastAPI, Header  # noqa: E402


class MemoryStore:
    """A ``RunStore`` in a dict."""

    def __init__(self) -> None:
        self.rows: dict[str, StoredRun] = {}
        self.written: dict[str, list[str]] = {}

    async def create(self, run: StoredRun) -> None:
        self.rows[run.id], self.written[run.id] = run, []

    async def get(self, run_id: str) -> StoredRun | None:
        return self.rows.get(run_id)

    async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
        row = self.rows.get(run_id)
        if row is None or row.rev != rev:
            return False
        self.rows[run_id] = dataclasses.replace(row, **changes, rev=rev + 1)  # type: ignore[arg-type]
        return True

    async def expired(self, now: float) -> list[StoredRun]:
        return [row for row in self.rows.values() if row.status == "running" and row.lease_until < now]

    async def add_event(self, run_id: str, owner: str, body: str) -> int | None:
        if self.rows[run_id].owner != owner:
            return None
        self.written[run_id].append(body)
        return len(self.written[run_id]) - 1

    async def events(self, run_id: str, after: int) -> list[tuple[int, str]]:
        return [(seq, body) for seq, body in enumerate(self.written[run_id]) if seq > after]


def _caller(x_user: Annotated[str, Header()] = "anna") -> str:
    return x_user


async def _until_ended(client: Any, run_id: str, who: str) -> dict[str, Any]:
    for _ in range(500):
        row: dict[str, Any] = (await client.get(f"/runs/{run_id}", headers={"x-user": who})).json()
        if row["status"] != "running":
            return row
        await asyncio.sleep(0.01)
    raise AssertionError("the run never ended")


def test_again_by_another_caller_who_may_see_the_run_is_checked_on_that_callers_own_nodes() -> None:
    full, bare = NodeRegistry(), NodeRegistry()

    @full.node("double")
    def double(x: int) -> int:
        return 2 * x

    graph = GraphRecord(nodes=[NodeRecord(id="d", node="double")]).model_dump(mode="json")
    app = FastAPI()
    app.include_router(run_router(
        store=MemoryStore(), registry=lambda who: full if who == "anna" else bare, caller=_caller, may_see=lambda caller, started_by: True,
    ))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"x": 1}})).json()["id"]
            assert (await _until_ended(client, run_id, "anna"))["status"] == "complete"
            again = await client.post(f"/runs/{run_id}/again", json={"x": 2}, headers={"x-user": "bo"})
            assert again.status_code == 422
            assert [problem["type"] for problem in again.json()["detail"]] == ["unknown_node"]
            mine = await client.post(f"/runs/{run_id}/again", json={"x": 2})
            assert mine.status_code == 202
            assert (await _until_ended(client, mine.json()["id"], "anna"))["outputs"] == {"d.result": 4}

    asyncio.run(scenario())


def test_an_output_that_isnt_finite_reads_as_its_text() -> None:
    registry = NodeRegistry()

    @registry.node("edges")
    def edges(x: int) -> list[float]:
        return [float("inf"), float("-inf"), float("nan"), x]

    graph = GraphRecord(nodes=[NodeRecord(id="e", node="edges")]).model_dump(mode="json")
    app = FastAPI()
    app.include_router(run_router(store=MemoryStore(), registry=registry, caller=_caller))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"x": 1}})).json()["id"]
            assert (await _until_ended(client, run_id, "anna"))["outputs"] == {"e.result": ["Infinity", "-Infinity", "NaN", 1.0]}

    asyncio.run(scenario())


def test_a_run_stopped_at_its_ceiling_says_why_and_is_resumed_with_no_body_or_a_higher_ceiling() -> None:
    registry = NodeRegistry()

    @registry.node("up")
    def up(text: str) -> str:
        return text.upper()

    graph = GraphRecord(nodes=[NodeRecord(id="u", node="up")]).model_dump(mode="json")
    store = MemoryStore()
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=_caller, max_rows=1))

    def last_end(run_id: str) -> dict[str, Any]:
        ends: list[dict[str, Any]] = [json.loads(body) for body in store.written[run_id] if json.loads(body)["type"] == "run_end"]
        return ends[-1]

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"text": ["a", "b"]}})).json()["id"]
            stopped = await _until_ended(client, run_id, "anna")
            assert stopped["status"] == "stopped"
            assert stopped["cause"]["type"] == last_end(run_id)["cause"]["type"] == "row_ceiling"
            assert stopped["cause"]["ctx"]["max_rows"] == 1 and "rows" in stopped["cause"]["ctx"]
            below_one = await client.post(f"/runs/{run_id}/resume", json={"max_rows": 0})
            assert below_one.status_code == 422
            assert [(e["type"], e["loc"]) for e in below_one.json()["detail"]] == [("greater_than", ["body", "max_rows"])]
            assert (await client.get(f"/runs/{run_id}")).json()["status"] == "stopped"
            assert (await client.post(f"/runs/{run_id}/resume")).status_code == 202
            assert (await _until_ended(client, run_id, "anna"))["status"] == "stopped"
            assert (await client.post(f"/runs/{run_id}/resume", json={"max_rows": 2})).status_code == 202
            completed = await _until_ended(client, run_id, "anna")
            assert completed["outputs"] == {"u.result": ["A", "B"]}
            assert completed["cause"] is None and last_end(run_id)["cause"] is None

    asyncio.run(scenario())



def test_a_ceiling_raised_by_a_resume_holds_for_the_legs_after_it() -> None:
    registry = NodeRegistry()

    @registry.node("split")
    def split(n: int) -> Series[int]:
        return list(range(n))

    @registry.node("ask")
    def ask(x: int) -> int:
        raise Ask("ok?")

    graph = make_graph(split, ask, edges=[("split.result", "ask.x")], values={"split.n": 5}).model_dump(mode="json")
    app = FastAPI()
    app.include_router(run_router(store=MemoryStore(), registry=registry, caller=_caller, max_rows=3))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph})).json()["id"]
            assert (await _until_ended(client, run_id, "anna"))["status"] == "stopped"
            assert (await client.post(f"/runs/{run_id}/resume", json={"max_rows": 10})).status_code == 202
            waiting = (await _until_ended(client, run_id, "anna"))["waiting"]
            assert len(waiting) == 5
            assert (await client.post(f"/runs/{run_id}/answers", json={waiting[0]["id"]: 7})).status_code == 202
            after = await _until_ended(client, run_id, "anna")
            assert (after["status"], len(after["waiting"])) == ("waiting", 4)

    asyncio.run(scenario())


class Part(TypedDict):
    text: str


def test_a_table_made_on_each_row_reads_as_a_list_of_records_per_row() -> None:
    registry = NodeRegistry()

    @registry.node("split")
    def split(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("as-table")
    def as_table(text: str) -> Table[Part]:
        return pl.DataFrame({"text": [text]})

    graph = make_graph(split, as_table).model_dump(mode="json")
    app = FastAPI()
    app.include_router(run_router(store=MemoryStore(), registry=registry, caller=_caller))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"text": "a,b"}})).json()["id"]
            ended = await _until_ended(client, run_id, "anna")
            assert (ended["status"], ended["outputs"]) == ("complete", {"as-table.result": [[{"text": "a"}], [{"text": "b"}]]})

    asyncio.run(scenario())


def _gated() -> tuple[NodeRegistry, GraphRecord, dict[str, threading.Event], list[str]]:
    """A graph whose first leg waits on ``first`` beside a question, and whose leg after the answer waits on ``second``
    on its first row of two; ``held`` is set once that row is called."""
    registry, gates, calls = NodeRegistry(), {name: threading.Event() for name in ("first", "second", "held")}, []

    @registry.node("items")
    def items(text: str) -> Series[str]:
        return text.split(",")

    @registry.node("busy")
    def busy(text: str) -> str:
        gates["first"].wait(5)
        return text

    @registry.node("approve")
    def approve(text: str) -> bool:
        raise Ask("Go on?")

    @registry.node("hold")
    def hold(item: str, ok: bool) -> str:
        calls.append(item)
        gates["held"].set()
        gates["second"].wait(5)
        return item

    graph = make_graph(items, busy, approve, hold, edges=[("items.result", "hold.item"), ("approve.result", "hold.ok")])
    return registry, graph, gates, calls


def test_a_cancel_reaches_the_leg_an_answer_queued_during_the_last_one_started() -> None:
    registry, graph, gates, calls = _gated()
    for gate in ("first", "second"):
        gates[gate].set()
    (waiting,) = registry.load(graph).run({"items.text": "a,b", "busy.text": "x", "approve.text": "y"}).waiting
    gates["first"].clear()
    gates["second"].clear()
    calls.clear()
    app = FastAPI()
    app.include_router(run_router(store=MemoryStore(), registry=registry, caller=_caller))

    async def scenario() -> None:
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            inputs = {"items.text": "a,b", "busy.text": "x", "approve.text": "y"}
            run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
            assert (await client.post(f"/runs/{run_id}/answers", json={waiting.id: True})).json() == {"answers": "queued"}
            gates["first"].set()
            await asyncio.to_thread(gates["held"].wait, 5)
            assert (await client.post(f"/runs/{run_id}/cancel")).json() == {"cancelled": True}
            gates["second"].set()
            assert (await _until_ended(client, run_id, "anna"))["status"] == "cancelled" and calls == ["a"]

    asyncio.run(scenario())


def test_a_cancel_sent_while_a_dead_worker_held_the_run_is_kept_when_another_takes_it_over() -> None:
    registry, graph, gates, calls = _gated()
    for gate in gates.values():
        gate.set()
    run = registry.load(graph).run({"items.text": "a,b", "busy.text": "x", "approve.text": "y"})
    answered = run.answer({run.waiting[0].id: True})
    calls.clear()
    store = MemoryStore()
    app = FastAPI()
    app.include_router(run_router(store=store, registry=registry, caller=_caller, lease=0.6))

    async def scenario() -> None:
        await store.create(StoredRun(
            id="taken", graph=graph.model_dump_json(), caller='"anna"', run=answered.to_json(), status="running", owner="a-dead-worker", lease_until=time.time() + 0.3,
        ))
        async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
            assert (await client.post("/runs/taken/cancel")).json() == {"cancelled": True}
            assert (await _until_ended(client, "taken", "anna"))["status"] == "cancelled" and calls == []
            assert (await client.post("/runs/taken/resume", json={})).status_code == 202
            assert (await _until_ended(client, "taken", "anna"))["status"] == "complete" and calls == ["a", "b"]

    asyncio.run(scenario())
