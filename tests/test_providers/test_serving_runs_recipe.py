"""The recipe in ``docs/serving-runs.md`` is what runs here: every Python block on that page, as written."""

from __future__ import annotations

import asyncio
import json
import re
import sys
import threading
import types
from collections.abc import Callable, Coroutine, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import pytest

pytest.importorskip("fastapi")
httpx = pytest.importorskip("httpx")

from conductor import AskPerson, FromRun, NodeRegistry, Result, node  # noqa: E402
from conductor.widgets import Switch  # noqa: E402

PAGE = Path(__file__).parents[2] / "docs" / "serving-runs.md"
_gate = threading.Event()
_gate_calls: list[str] = []


@dataclass(frozen=True)
class Caller:
    name: str


@node("gate")
def gate() -> float:
    _gate_calls.append("gate")
    assert _gate.wait(20), "the test never opened the gate"
    return float("nan")


@node("ask")
def ask(value: float) -> Annotated[bool, Result(title="Approved")] | AskPerson:
    return AskPerson(widgets={"result": Switch()})


@node("who")
def who(caller: FromRun[Caller]) -> str:
    return caller.name


@node("greet")
def greet(name: str) -> str:
    return f"hi {name}"


class _Host:
    """One process of the recipe: a fresh module holding its own legs, spoken to over ASGI."""

    def __init__(self) -> None:
        self.module = types.ModuleType("serving_runs_recipe")
        sys.modules["serving_runs_recipe"] = self.module
        for block in re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.S):
            exec(compile(block, str(PAGE), "exec"), self.module.__dict__)
        self.module.registry = NodeRegistry([gate, ask, greet, who])
        self.module.Caller = Caller
        self.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.module.app), base_url="http://recipe")

    async def start(self, *node_types: str) -> str:
        created = await self.client.post("/runs", json=_graph(*node_types))
        assert created.status_code == 202, created.text
        return created.json()["id"]

    async def frames(self, run_id: str) -> list[dict[str, Any]]:
        response = await self.client.get(f"/runs/{run_id}/events")
        assert response.status_code == 200
        return [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]

    async def until(self, ready: Callable[[], bool]) -> None:
        for _ in range(2000):
            if ready():
                return
            await asyncio.sleep(0.005)
        raise AssertionError("never happened")

    async def emitted(self, run_id: str, kind: str) -> None:
        await self.until(lambda: any(event.type == kind for event in self.module.legs[run_id].events))


def _graph(*node_types: str) -> dict[str, Any]:
    """A chain of nodes, each with its type as its id, ``ask`` reading the one before it."""
    nodes: list[dict[str, Any]] = []
    for position, node_type in enumerate(node_types):
        bindings = {"value": {"refs": [f"{node_types[position - 1]}.result"]}} if node_type == "ask" else {}
        nodes.append({"id": node_type, "type": node_type, "version": 1, "bindings": bindings})
    return {"nodes": nodes}


@pytest.fixture(autouse=True)
def _a_closed_gate() -> Iterator[None]:
    _gate.clear()
    _gate_calls.clear()
    yield
    _gate.set()


def _run(scenario: Callable[[], Coroutine[Any, Any, None]]) -> None:
    asyncio.run(scenario())


def test_a_run_is_started_followed_late_saved_answered_on_a_new_process_and_cancelled():
    async def scenario() -> None:
        first = _Host()
        run_id = await first.start("gate", "ask")

        early = asyncio.create_task(first.frames(run_id))
        await first.emitted(run_id, "run_start")
        gone = asyncio.create_task(first.frames(run_id))
        await asyncio.sleep(0.01)
        gone.cancel()  # a closed connection
        await asyncio.gather(gone, return_exceptions=True)
        late = asyncio.create_task(first.frames(run_id))
        await asyncio.sleep(0.01)
        _gate.set()
        seen_early, seen_late = await early, await late

        assert seen_late == seen_early  # replay, then the rest
        assert [frame["type"] for frame in seen_late][0] == "run_start"
        assert seen_late[-1] == {"type": "run_end", "status": "waiting"}
        assert "finished_units" not in json.dumps(seen_late)
        completed = next(frame for frame in seen_late if frame["type"] == "node_complete")
        assert completed["result"]["result"] == "NaN"

        read = (await first.client.get(f"/runs/{run_id}")).json()
        assert read["status"] == "waiting" and "state" not in read and "graph" not in read
        (unit,) = read["waiting"]

        # The next process has only what the row holds.
        second = _Host()
        row = first.module.saved[run_id].model_dump_json()
        second.module.saved[run_id] = second.module.SavedRun.model_validate_json(row)
        assert (await second.client.post(f"/runs/{run_id}/answers", json={unit["id"]: {"result": True}})).status_code == 202
        assert (await second.frames(run_id))[-1] == {"type": "run_end", "status": "complete"}
        assert second.module.saved[run_id].status == "complete"
        assert _gate_calls == ["gate"]  # what ran did not run again

        _gate.clear()
        stopping = await second.start("gate")
        await second.emitted(stopping, "node_start")
        assert (await second.client.post(f"/runs/{stopping}/cancel")).json() == {"cancelled": True}
        assert (await second.frames(stopping))[-1] == {"type": "run_end", "status": "cancelled"}
        assert second.module.saved[stopping].status == "cancelled"
        assert (await second.client.post(f"/runs/{stopping}/cancel")).json() == {"cancelled": False}
        assert await second.frames(stopping) == []  # no leg running: read the row

    _run(scenario)


def test_a_wrong_answer_is_a_422_listing_every_fault_and_a_run_not_waiting_is_a_409():
    async def scenario() -> None:
        host = _Host()
        _gate.set()
        run_id = await host.start("gate", "ask")
        await host.frames(run_id)
        (unit,) = host.module.saved[run_id].waiting

        wrong = await host.client.post(f"/runs/{run_id}/answers", json={unit.id: {"result": "not a bool"}, "nonesuch": {}})
        assert wrong.status_code == 422 and len(wrong.json()["detail"]) == 2
        assert host.module.saved[run_id].status == "waiting"

        assert (await host.client.post(f"/runs/{run_id}/answers", json={unit.id: {"result": False}})).status_code == 202
        assert (await host.client.post(f"/runs/{run_id}/answers", json={})).status_code == 409
        await host.frames(run_id)
        assert (await host.client.post("/runs/nonesuch/answers", json={})).status_code == 404

    _run(scenario)


def test_a_graph_that_cannot_run_is_a_422_before_anything_is_saved():
    async def scenario() -> None:
        host = _Host()
        refused = await host.client.post("/runs", json={"nodes": [{"id": "x", "type": "nonesuch", "version": 1}]})
        assert refused.status_code == 422 and refused.json()["detail"]
        assert host.module.saved == {}

    _run(scenario)


def test_what_a_node_receives_comes_from_the_routes_own_dependency():
    async def scenario() -> None:
        host = _Host()
        module = host.module
        module.app.dependency_overrides[module.current_caller] = lambda: Caller("bo")
        run_id = await host.start("who")
        await host.frames(run_id)
        run = module.saved[run_id]
        assert module.CompiledGraph.from_graph(run.graph, module.registry).outputs(run.state) == {"who.result": "bo"}

    _run(scenario)


def test_an_end_user_fills_the_inputs_of_a_published_graph_and_names_no_node():
    async def scenario() -> None:
        host = _Host()
        host.module.published["hello"] = host.module.Graph.model_validate(_graph("greet"))
        created = await host.client.post("/graphs/hello/runs", json={"greet.name": "ida"})
        assert created.status_code == 202, created.text
        await host.frames(created.json()["id"])
        run = host.module.saved[created.json()["id"]]
        assert run.status == "complete"

    _run(scenario)


def test_the_published_schema_says_what_each_route_answers():
    host = _Host()
    spec = host.module.app.openapi()

    media = spec["paths"]["/runs/{run_id}/events"]["get"]["responses"]["200"]["content"]["text/event-stream"]
    union = media["itemSchema"]["properties"]["data"]["contentSchema"]
    assert union["discriminator"]["propertyName"] == "type" and {"$ref": "#/components/schemas/RunEnd"} in union["oneOf"]
    run_end = spec["components"]["schemas"]["RunEnd"]
    assert set(run_end["properties"]) == {"type", "status"}
    assert spec["paths"]["/nodes"]["get"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/RegistryDescription")
    assert spec["paths"]["/compile"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"].endswith("/GraphDescription")


def test_a_stream_says_not_to_cache_it():
    async def scenario() -> None:
        host = _Host()
        _gate.set()
        run_id = await host.start("gate")
        response = await host.client.get(f"/runs/{run_id}/events")
        assert response.headers["cache-control"] == "no-cache"
        assert response.headers["content-type"].startswith("text/event-stream")

    _run(scenario)
