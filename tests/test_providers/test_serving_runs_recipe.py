"""The recipe in ``docs/serving-runs.md`` is what runs here: every Python block on that page, as written, as two workers.

Each worker is the page's module loaded afresh, with its own id, legs and sweeper, on one database file. A kill is a
worker's tasks cancelled with no final write, so its lease runs out as a dead process's would.
"""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import time
import types
from collections.abc import AsyncIterator, Callable, Coroutine, Iterator
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from conductor import Ask, FromRun, Param, Series, node  # noqa: E402
from conductor.widgets import Switch  # noqa: E402

PAGE = Path(__file__).parents[2] / "docs" / "serving-runs.md"
_gate = threading.Event()
_calls: list[str] = []


@dataclass(frozen=True)
class Caller:
    name: str


@dataclass(frozen=True)
class Missing:
    """A value no route puts in ``from_run``."""


@node("gate")
def gate() -> float:
    _calls.append("gate")
    assert _gate.wait(20), "the test never opened the gate"
    return float("nan")


@node("ask")
def ask(value: float) -> Annotated[bool, Param(title="Approved")]:
    raise Ask("Approve?", widget=Switch())


@node("who")
def who(caller: FromRun[Caller]) -> str:
    return caller.name


@node("greet")
def greet(name: str) -> str:
    return f"hi {name}"


@node("rows")
def rows(n: int) -> Series[int]:
    return list(range(n))


@node("work")
def work(i: int) -> int:
    _calls.append(f"work {i}")
    time.sleep(0.02)
    return i


@node("approve_each")
def approve_each(i: int) -> bool:
    raise Ask(f"Approve {i}?")


@node("after")
def after(approved: bool) -> bool:
    _calls.append("after")
    assert _gate.wait(20), "the test never opened the gate"
    return approved


@node("needs_missing")
def needs_missing(value: FromRun[Missing]) -> str:
    return ""


class _Worker:
    """One process of the recipe, spoken to over ASGI."""

    def __init__(self, database: Path) -> None:
        self.module = types.ModuleType("serving_runs_recipe")
        sys.modules["serving_runs_recipe"] = self.module
        for block in re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.S):
            exec(compile(block, str(PAGE), "exec"), self.module.__dict__)  # noqa: S102  the page, as a reader copies it
        module = self.module
        module.database = module.Database(str(database))
        module.LEASE, module.CHECKPOINT_EVERY = 0.6, 0.05
        module.registry.add(gate, ask, greet, who, rows, work, approve_each, after, needs_missing)
        module.Caller = Caller
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=module.app), base_url="http://recipe")

    async def start(self, graph: dict[str, Any]) -> str:
        created = await self.client.post("/runs", json=graph)
        assert created.status_code == 202, created.text
        return created.json()["id"]

    async def frames(self, run_id: str) -> list[dict[str, Any]]:
        response = await self.client.get(f"/runs/{run_id}/events")
        assert response.status_code == 200
        return [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]

    async def read(self, run_id: str) -> dict[str, Any]:
        return (await self.client.get(f"/runs/{run_id}")).json()

    async def until_status(self, run_id: str, *statuses: str) -> dict[str, Any]:
        for _ in range(2000):
            row = await self.read(run_id)
            if row["status"] in statuses:
                return row
            await asyncio.sleep(0.01)
        raise AssertionError(f"never {statuses}: {row}")

    def kill(self) -> None:
        """Stop every task of this worker with no final write, as a dead process would."""
        for task in [*self.module.tasks, *self._sweeping()]:
            task.cancel()

    def _sweeping(self) -> list[asyncio.Task[Any]]:
        return [t for t in asyncio.all_tasks() if t.get_coro().__qualname__.startswith("sweep") and t.get_coro().cr_frame.f_globals is self.module.__dict__]  # type: ignore[union-attr]


def _chain(*node_types: str, edges: tuple[tuple[str, str], ...] = (), values: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    nodes = [{"id": n, "node": n, **({"values": values[n]} if values and n in values else {})} for n in node_types]
    return {"nodes": nodes, "edges": [list(edge) for edge in edges]}


_GATE_ASK = _chain("gate", "ask", edges=(("gate.result", "ask.value"),))


@pytest.fixture(autouse=True)
def _a_closed_gate() -> Iterator[None]:
    _gate.clear()
    _calls.clear()
    yield
    _gate.set()


@asynccontextmanager
async def _workers(tmp_path: Path, count: int = 2) -> AsyncIterator[list[_Worker]]:
    async with AsyncExitStack() as stack:
        workers = []
        for _ in range(count):
            worker = _Worker(tmp_path / "runs.db")
            await stack.enter_async_context(worker.module.lifespan(worker.module.app))
            workers.append(worker)
        yield workers
        _gate.set()


def _run(scenario: Callable[[], Coroutine[Any, Any, None]]) -> None:
    asyncio.run(scenario())


def test_a_run_started_on_one_worker_is_followed_read_and_answered_on_the_other(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _workers(tmp_path) as (a, b):
            run_id = await a.start(_GATE_ASK)
            following = asyncio.create_task(b.frames(run_id))
            await asyncio.sleep(0.05)
            _gate.set()
            seen = await following

            assert seen[0]["type"] == "run_start" and seen[-1] == {"type": "run_end", "status": "waiting"}
            completed = next(frame for frame in seen if frame["type"] == "node_complete")
            assert completed == {"type": "node_complete", "node_id": "gate", "rows": 1, "not_called": 0, "reused_units": 0}
            assert await a.frames(run_id) == seen  # a follower that comes late gets every event

            (unit,) = (await b.read(run_id))["waiting"]
            assert (await b.client.post(f"/runs/{run_id}/answers", json={unit["id"]: True})).status_code == 202
            done = await a.until_status(run_id, "complete")
            assert done["outputs"] == {"ask.result": True}
            assert _calls == ["gate"]  # what ran did not run again

    _run(scenario)


def test_two_workers_answering_one_run_at_once_start_one_leg(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _workers(tmp_path) as (a, b):
            _gate.set()
            run_id = await a.start(_chain("rows", "approve_each", "after", edges=(("rows.result", "approve_each.i"), ("approve_each.result", "after.approved")), values={"rows": {"n": 1}}))
            (unit,) = (await a.until_status(run_id, "waiting"))["waiting"]

            first, second = await asyncio.gather(
                a.client.post(f"/runs/{run_id}/answers", json={unit["id"]: True}),
                b.client.post(f"/runs/{run_id}/answers", json={unit["id"]: False}),
            )
            assert sorted([first.status_code, second.status_code]) == [202, 409]
            await a.until_status(run_id, "complete")
            assert _calls == ["after"]

    _run(scenario)


def test_a_run_whose_worker_dies_is_finished_by_the_other_from_its_last_checkpoint(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _workers(tmp_path) as (a, b):
            run_id = await a.start(_chain("rows", "work", edges=(("rows.result", "work.i"),), values={"rows": {"n": 40}}))
            while len(_calls) < 15:
                await asyncio.sleep(0.01)
            a.kill()

            await b.until_status(run_id, "complete")
            made = [call for call in _calls if call.startswith("work")]
            assert sorted(set(made)) == sorted(f"work {i}" for i in range(40))
            assert len(made) - 40 <= 6, made  # only what the last checkpoint had no record of is called again

    _run(scenario)


def test_a_cancel_sent_to_either_worker_stops_the_leg_and_a_resume_finishes_it(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _workers(tmp_path) as (a, b):
            run_id = await a.start(_GATE_ASK)
            while not _calls:
                await asyncio.sleep(0.01)
            assert (await b.client.post(f"/runs/{run_id}/cancel")).json() == {"cancelled": True}
            await asyncio.sleep(a.module.LEASE)  # the worker running the leg sees the flag at its next renewal
            _gate.set()  # a cancel waits for the call already made, and keeps it
            await a.until_status(run_id, "cancelled")

            refused = await b.client.post(f"/runs/{run_id}/answers", json={})
            assert refused.status_code == 409 and "resume" in refused.text
            assert (await b.client.post(f"/runs/{run_id}/resume", json={})).status_code == 202
            await b.until_status(run_id, "waiting")
            assert _calls == ["gate"]

    _run(scenario)


def test_a_leg_that_timed_out_is_resumed_by_its_route(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _workers(tmp_path, count=1) as (a,):
            a.module.LEG_TIMEOUT = 0.1
            run_id = await a.start(_chain("rows", "work", edges=(("rows.result", "work.i"),), values={"rows": {"n": 30}}))
            await a.until_status(run_id, "timed_out")
            a.module.LEG_TIMEOUT = 300.0
            assert (await a.client.post(f"/runs/{run_id}/resume", json={})).status_code == 202
            await a.until_status(run_id, "complete")
            assert sorted(_calls) == sorted(f"work {i}" for i in range(30))  # a timed-out leg keeps what it called
            assert (await a.client.post(f"/runs/{run_id}/resume", json={})).status_code == 409

    _run(scenario)


def test_a_deploy_hands_its_legs_to_another_worker_and_no_call_is_made_twice(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _workers(tmp_path, count=1) as (b,):
            a = _Worker(tmp_path / "runs.db")
            async with a.module.lifespan(a.module.app):
                run_id = await a.start(_chain("rows", "work", edges=(("rows.result", "work.i"),), values={"rows": {"n": 30}}))
                while len(_calls) < 10:
                    await asyncio.sleep(0.01)
            await b.until_status(run_id, "complete")
            made = [call for call in _calls if call.startswith("work")]
            assert sorted(made) == sorted(f"work {i}" for i in range(30))

    _run(scenario)


def test_answers_sent_while_a_leg_runs_are_applied_when_it_ends(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _workers(tmp_path) as (a, b):
            graph = _chain("rows", "approve_each", "after", edges=(("rows.result", "approve_each.i"), ("approve_each.result", "after.approved")), values={"rows": {"n": 2}})
            run_id = await a.start(graph)
            first, second = (await a.until_status(run_id, "waiting"))["waiting"]
            assert (await a.client.post(f"/runs/{run_id}/answers", json={first["id"]: True})).json() == {"answers": "started"}
            while "after" not in _calls:
                await asyncio.sleep(0.01)
            assert (await b.client.post(f"/runs/{run_id}/answers", json={second["id"]: False})).json() == {"answers": "queued"}
            _gate.set()
            done = await b.until_status(run_id, "complete")
            assert done["outputs"] == {"after.result": [True, False]}

    _run(scenario)


def test_a_leg_that_cannot_start_leaves_the_run_failed_not_running(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _workers(tmp_path, count=1) as (a,):
            run_id = await a.start(_chain("needs_missing"))
            assert (await a.until_status(run_id, "failed"))["status"] == "failed"

    _run(scenario)


def test_a_node_connects_to_the_database_itself_on_the_legs_thread(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _workers(tmp_path, count=1) as (a,):
            run_id = await a.start(_chain("count_runs", values={"count_runs": {"status": "running"}}))
            assert (await a.until_status(run_id, "complete"))["outputs"] == {"count_runs.result": 1}

    _run(scenario)


def test_what_a_node_receives_comes_from_the_routes_own_dependency(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _workers(tmp_path, count=1) as (a,):
            a.module.app.dependency_overrides[a.module.current_caller] = lambda: Caller("bo")
            run_id = await a.start(_chain("who"))
            assert (await a.until_status(run_id, "complete"))["outputs"] == {"who.result": "bo"}

    _run(scenario)


def test_a_wrong_answer_is_a_422_listing_every_fault_and_a_run_not_waiting_is_a_409(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _workers(tmp_path, count=1) as (a,):
            _gate.set()
            run_id = await a.start(_GATE_ASK)
            (unit,) = (await a.until_status(run_id, "waiting"))["waiting"]

            wrong = await a.client.post(f"/runs/{run_id}/answers", json={unit["id"]: "not a bool", "nonesuch": True})
            assert wrong.status_code == 422 and len(wrong.json()["detail"]) == 2
            assert (await a.read(run_id))["status"] == "waiting"

            stale = await a.client.post(f"/runs/{run_id}/answers", json={unit["id"]: True}, headers={"If-Match": "99"})
            assert stale.status_code == 409
            assert (await a.client.post(f"/runs/{run_id}/answers", json={unit["id"]: False})).status_code == 202
            await a.until_status(run_id, "complete")
            assert (await a.client.post(f"/runs/{run_id}/answers", json={})).status_code == 409
            assert (await a.client.post("/runs/nonesuch/answers", json={})).status_code == 404

    _run(scenario)


def test_a_graph_that_cannot_run_is_a_422_before_anything_is_saved(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _workers(tmp_path, count=1) as (a,):
            refused = await a.client.post("/runs", json={"nodes": [{"id": "x", "node": "nonesuch"}]})
            assert refused.status_code == 422 and refused.json()["detail"]
            assert a.module.execute("SELECT COUNT(*) AS n FROM runs")[0]["n"] == 0

    _run(scenario)


def test_an_end_user_fills_the_inputs_of_a_published_graph_and_names_no_node(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _workers(tmp_path, count=1) as (a,):
            a.module.published["hello"] = a.module.GraphRecord.model_validate(_chain("greet"))
            created = await a.client.post("/graphs/hello/runs", json={"greet.name": "ida"})
            assert created.status_code == 202, created.text
            assert (await a.until_status(created.json()["id"], "complete"))["outputs"] == {"greet.result": "hi ida"}

    _run(scenario)


def test_the_published_schema_says_what_a_run_reads_as(tmp_path: Path) -> None:
    spec = _Worker(tmp_path / "runs.db").module.app.openapi()
    view = spec["components"]["schemas"]["RunView"]
    assert {"status", "rev", "waiting", "outputs"} <= set(view["properties"])
    waiting = spec["components"]["schemas"]["Waiting"]
    assert {"id", "node_id", "row", "prompt", "questions", "inputs"} <= set(waiting["required"])


def test_a_stream_says_not_to_cache_it_and_an_unknown_runs_events_are_a_404(tmp_path: Path) -> None:
    async def scenario() -> None:
        async with _workers(tmp_path, count=1) as (a,):
            _gate.set()
            run_id = await a.start(_chain("gate"))
            response = await a.client.get(f"/runs/{run_id}/events")
            assert response.headers["cache-control"] == "no-cache"
            assert response.headers["content-type"].startswith("text/event-stream")
            assert (await a.client.get("/runs/nope/events")).status_code == 404

    _run(scenario)
