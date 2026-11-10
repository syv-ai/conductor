"""``run_router`` served in this process over a store in memory, for what the page's own app doesn't set up: a host's
own ``may_see`` and a registry per caller."""

from __future__ import annotations

import asyncio
import dataclasses
from typing import Annotated, Any

import pytest

httpx = pytest.importorskip("httpx")

from conductor import GraphRecord, NodeRecord, NodeRegistry  # noqa: E402
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
