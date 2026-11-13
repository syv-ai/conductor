"""``docs/serving-runs.md`` as written, served by real uvicorn workers on one SQLite file.

Each worker is a process running the page's Python blocks, in order, as one module, with the test nodes added and the
lease and checkpoint shortened. A kill is ``SIGKILL``, so its lease runs out as a dead process's would; a deploy is
``SIGTERM``, so uvicorn shuts the app down.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import socket
import sys
import time
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from pathlib import Path
from signal import Signals
from subprocess import Popen
from typing import Any

import pytest

httpx = pytest.importorskip("httpx")
pytest.importorskip("uvicorn")

PAGE = Path(__file__).parents[2] / "docs" / "serving-runs.md"
HERE = Path(__file__).parent

#: What a test changes on the page's app: short leases and checkpoints, the test nodes, and a caller named by header.
_TEST_SETTINGS = """
from conductor import Context
from fastapi import Header

from served_nodes import served

registry.add_subregistry(served)


@registry.node("who", title="Who", description="For a test.")
def who(ctx: Context[AppContext]) -> str:
    return ctx.caller.name


def header_caller(x_user: Annotated[str, Header()] = "anna") -> Caller:
    return Caller(x_user)


app.dependency_overrides[current_caller] = header_caller
"""


def _app_source() -> str:
    blocks = re.findall(r"```python\n(.*?)```", PAGE.read_text(), re.S)
    source = "from typing import Annotated\n" + "\n".join(blocks)
    settings = 'lease=0.6, checkpoint_every=0.05, resume_timeout=float(os.environ.get("RESUME_TIMEOUT", "300")), '
    assert source.count("run_router(\n") == 1
    return "import os\n" + source.replace("run_router(\n", "run_router(\n    " + settings, 1) + _TEST_SETTINGS


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class Worker:
    """One uvicorn process serving the page's app."""

    def __init__(self, directory: Path, **environment: str) -> None:
        self.port = _free_port()
        env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(directory), str(HERE)]), **environment}
        self.process = Popen(
            [sys.executable, "-m", "uvicorn", "served_app:app", "--port", str(self.port), "--log-level", "warning"],
            cwd=directory, env=env,
        )
        self.client = httpx.AsyncClient(base_url=f"http://127.0.0.1:{self.port}", timeout=30)

    def wait_until_serving(self) -> None:
        for _ in range(500):
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.1):
                    return
            except OSError:
                assert self.process.poll() is None, "the worker exited"
                time.sleep(0.02)
        raise AssertionError("the worker never served")

    def stop(self, how: Signals) -> None:
        if self.process.poll() is None:
            self.process.send_signal(how)
        self.process.wait(30)

    async def start(self, graph: dict[str, Any], inputs: dict[str, Any] | None = None) -> str:
        created = await self.client.post("/runs", json={"graph": graph, "inputs": inputs or {}})
        assert created.status_code == 202, created.text
        return str(created.json()["id"])

    async def read(self, run_id: str) -> dict[str, Any]:
        found: dict[str, Any] = (await self.client.get(f"/runs/{run_id}")).json()
        return found

    async def until_status(self, run_id: str, *statuses: str) -> dict[str, Any]:
        for _ in range(3000):
            row = await self.read(run_id)
            if row["status"] in statuses:
                return row
            await asyncio.sleep(0.01)
        raise AssertionError(f"never {statuses}: {row}")

    async def until_refused(self, run_id: str) -> dict[str, Any]:
        """The run once it has a refused answer, which is written just after the resume call's own last write."""
        for _ in range(3000):
            row = await self.read(run_id)
            if row["refused"]:
                return row
            await asyncio.sleep(0.01)
        raise AssertionError(f"never refused: {row}")

    async def frames(self, run_id: str) -> list[dict[str, Any]]:
        response = await self.client.get(f"/runs/{run_id}/events")
        assert response.status_code == 200
        return [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]


class Served:
    """The directory the workers share: the app, the database, the gate and the calls."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        (directory / "served_app.py").write_text(_app_source())

    def worker(self, **environment: str) -> Worker:
        started = Worker(self.directory, **environment)
        started.wait_until_serving()
        return started

    def open_the_gate(self) -> None:
        (self.directory / "open").touch()

    @property
    def calls(self) -> list[str]:
        path = self.directory / "calls"
        return path.read_text().splitlines() if path.exists() else []

    async def until_calls(self, count: int) -> None:
        while len(self.calls) < count:
            await asyncio.sleep(0.01)


@asynccontextmanager
async def _workers(served: Served, count: int = 2, **environment: str) -> AsyncIterator[list[Worker]]:
    workers = [served.worker(**environment) for _ in range(count)]
    try:
        yield workers
    finally:
        served.open_the_gate()
        for worker in workers:
            await worker.client.aclose()
            worker.stop(signal.SIGTERM)


def _run(scenario: Callable[[], Coroutine[Any, Any, None]]) -> None:
    asyncio.run(scenario())


def _source(node_id: str) -> dict[str, Any]:
    """The entry of an input fed by the output ``result`` of ``node_id``."""
    return {"sources": [{"node": node_id, "output": "result"}]}


def _graph(nodes: dict[str, dict[str, Any]], *, returns: str | None = None) -> dict[str, Any]:
    """A graph record of ``nodes``, each ``{node id: inputs}`` where the node's id is its definition's; with ``returns``,
    the graph returns that node's output ``result`` as ``result``."""
    record: dict[str, Any] = {"nodes": {node_id: {"node": node_id, "inputs": inputs} for node_id, inputs in nodes.items()}}
    if returns is not None:
        record["outputs"] = {"result": {"node": returns, "output": "result"}}
    return record


_GATE_ASK = _graph({"gate": {}, "ask": {"value": _source("gate")}}, returns="ask")
_WORK = _graph({"numbers": {"n": {"graph_input": "n"}}, "work": {"i": _source("numbers")}}, returns="work")
_APPROVE = _graph({"numbers": {"n": {"graph_input": "n"}}, "approve_each": {"i": _source("numbers")}, "after": {"approved": _source("approve_each")}}, returns="after")


@pytest.fixture
def served(tmp_path: Path) -> Served:
    return Served(tmp_path)


def test_a_run_started_on_one_worker_is_followed_read_and_answered_on_the_other(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served) as (a, b):
            run_id = await a.start(_GATE_ASK)
            following = asyncio.create_task(b.frames(run_id))
            await asyncio.sleep(0.1)
            served.open_the_gate()
            seen = await following

            assert seen[0]["type"] == "run_start" and seen[-1] == {"type": "run_end", "status": "waiting", "cause": None}
            completed = next(frame for frame in seen if frame["type"] == "node_complete")
            assert completed == {"type": "node_complete", "loc": ["gate"], "outcomes": 1, "not_called": 0, "reused": 0}
            assert await a.frames(run_id) == seen  # a follower that comes late gets every event

            (waiting_call,) = (await b.read(run_id))["waiting"]
            assert (await b.client.post(f"/runs/{run_id}/answers", json={waiting_call["id"]: True})).json() == {"answers": "started"}
            done = await a.until_status(run_id, "complete")
            assert done["outputs"] == {"result": True}
            assert served.calls == ["gate"]  # what ran did not run again

    _run(scenario)


def test_two_workers_answering_one_run_at_once_start_one_resume_call(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served) as (a, b):
            served.open_the_gate()
            run_id = await a.start(_APPROVE, {"n": 1})
            (waiting_call,) = (await a.until_status(run_id, "waiting"))["waiting"]

            first, second = await asyncio.gather(
                a.client.post(f"/runs/{run_id}/answers", json={waiting_call["id"]: True}),
                b.client.post(f"/runs/{run_id}/answers", json={waiting_call["id"]: False}),
            )
            said = sorted([first.json() if first.status_code == 202 else 409, second.json() if second.status_code == 202 else 409], key=str)
            # the later answer is refused at once, or, if it read the run once the resume call had started, queued and then refused
            assert said in ([409, {"answers": "started"}], [{"answers": "queued"}, {"answers": "started"}])
            await a.until_status(run_id, "complete")
            assert served.calls == ["after"]
            if said[0] != 409:
                refused = (await a.until_refused(run_id))["refused"]
                assert [problem["type"] for problem in refused[waiting_call["id"]]] == ["not_waiting"]

    _run(scenario)


def test_a_run_whose_worker_is_killed_is_finished_by_the_other_from_its_last_checkpoint(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served) as (a, b):
            run_id = await a.start(_WORK, {"n": 40})
            await served.until_calls(15)
            a.stop(signal.SIGKILL)

            await b.until_status(run_id, "complete")
            work_calls = [call for call in served.calls if call.startswith("work")]
            assert sorted(set(work_calls)) == sorted(f"work {i}" for i in range(40))
            assert len(work_calls) - 40 <= 6, work_calls  # only what the last checkpoint had no record of is called again

    _run(scenario)


def test_a_deploy_hands_its_runs_to_another_worker_and_no_call_is_made_twice(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served) as (a, b):
            run_id = await a.start(_WORK, {"n": 30})
            await served.until_calls(10)
            a.stop(signal.SIGTERM)

            await b.until_status(run_id, "complete")
            assert sorted(served.calls) == sorted(f"work {i}" for i in range(30))

    _run(scenario)


def test_a_cancel_sent_to_either_worker_stops_the_run_and_a_resume_finishes_it(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served) as (a, b):
            run_id = await a.start(_GATE_ASK)
            await served.until_calls(1)
            assert (await b.client.post(f"/runs/{run_id}/cancel")).json() == {"cancelled": True}
            await asyncio.sleep(0.6)  # the worker resuming the run sees the cancel at its next lease renewal
            served.open_the_gate()  # a cancel waits for the node call already started, and keeps its outcome
            await a.until_status(run_id, "cancelled")

            assert (await b.client.post(f"/runs/{run_id}/resume", json={})).status_code == 202
            await b.until_status(run_id, "waiting")
            assert served.calls == ["gate"]

    _run(scenario)


def test_a_run_that_timed_out_is_resumed_by_its_route_until_it_completes(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served, count=1, RESUME_TIMEOUT="0.1") as (a,):
            run_id = await a.start(_WORK, {"n": 30})
            await a.until_status(run_id, "timed_out")
            while (await a.client.post(f"/runs/{run_id}/resume", json={})).status_code == 202:
                await a.until_status(run_id, "timed_out", "complete")
                if (await a.read(run_id))["status"] == "complete":
                    break
            assert sorted(served.calls) == sorted(f"work {i}" for i in range(30))  # a resume call that timed out keeps what it called
            assert (await a.client.post(f"/runs/{run_id}/resume", json={})).status_code == 409

    _run(scenario)


def test_answers_sent_while_a_resume_call_runs_are_applied_when_it_ends(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served) as (a, b):
            run_id = await a.start(_APPROVE, {"n": 2})
            first, second = (await a.until_status(run_id, "waiting"))["waiting"]
            assert (await a.client.post(f"/runs/{run_id}/answers", json={first["id"]: True})).json() == {"answers": "started"}
            await served.until_calls(1)
            assert (await b.client.post(f"/runs/{run_id}/answers", json={second["id"]: False})).json() == {"answers": "queued"}
            served.open_the_gate()
            done = await b.until_status(run_id, "complete")
            assert done["outputs"] == {"result": [True, False]}

    _run(scenario)


def test_a_wrong_answer_sent_while_a_resume_call_runs_is_refused_alone_and_the_others_are_applied(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served) as (a, b):
            run_id = await a.start(_APPROVE, {"n": 3})
            first, second, third = (await a.until_status(run_id, "waiting"))["waiting"]
            assert (await a.client.post(f"/runs/{run_id}/answers", json={first["id"]: True})).json() == {"answers": "started"}
            await served.until_calls(1)
            queued = await b.client.post(f"/runs/{run_id}/answers", json={second["id"]: False, third["id"]: "not a yes or no"})
            assert queued.json() == {"answers": "queued"}
            served.open_the_gate()
            for _ in range(1000):  # the first resume call ends, the queued answers are applied, and the next one ends
                row = await b.read(run_id)
                if row["status"] == "waiting" and len(row["waiting"]) == 1:
                    break
                await asyncio.sleep(0.01)
            assert [waiting_call["id"] for waiting_call in row["waiting"]] == [third["id"]]
            assert [problem["type"] for problem in row["refused"][third["id"]]] == ["invalid_answer"]

    _run(scenario)


def test_an_answer_sent_while_a_resume_call_runs_to_an_id_nothing_waits_on_is_refused_when_it_ends(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served) as (a, b):
            run_id = await a.start(_APPROVE, {"n": 2})
            first, second = (await a.until_status(run_id, "waiting"))["waiting"]
            assert (await a.client.post(f"/runs/{run_id}/answers", json={first["id"]: True})).json() == {"answers": "started"}
            await served.until_calls(1)
            assert (await b.client.post(f"/runs/{run_id}/answers", json={"nobody": True})).json() == {"answers": "queued"}
            served.open_the_gate()
            row = await b.until_refused(run_id)
            assert row["status"] == "waiting" and [waiting_call["id"] for waiting_call in row["waiting"]] == [second["id"]]
            assert [problem["type"] for problem in row["refused"]["nobody"]] == ["not_waiting"]

    _run(scenario)


def test_a_second_answer_to_a_call_answered_while_a_resume_call_runs_is_a_409(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served) as (a, b):
            run_id = await a.start(_APPROVE, {"n": 2})
            first, second = (await a.until_status(run_id, "waiting"))["waiting"]
            assert (await a.client.post(f"/runs/{run_id}/answers", json={first["id"]: True})).json() == {"answers": "started"}
            await served.until_calls(1)
            assert (await a.client.post(f"/runs/{run_id}/answers", json={second["id"]: False})).json() == {"answers": "queued"}
            twice = await b.client.post(f"/runs/{run_id}/answers", json={second["id"]: True})
            assert twice.status_code == 409
            assert [(problem["type"], problem["ctx"]) for problem in twice.json()["detail"]] == [("already_answered", {"waiting_id": second["id"]})]
            served.open_the_gate()
            assert (await b.until_status(run_id, "complete"))["outputs"] == {"result": [True, False]}

    _run(scenario)


def test_an_answer_sent_while_the_last_resume_call_runs_is_refused_when_the_run_completes(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served) as (a, b):
            run_id = await a.start(_graph({"gate": {}}))
            await served.until_calls(1)
            assert (await b.client.post(f"/runs/{run_id}/answers", json={"nobody": True})).json() == {"answers": "queued"}
            served.open_the_gate()
            row = await a.until_refused(run_id)
            assert row["status"] == "complete"
            assert [problem["type"] for problem in row["refused"]["nobody"]] == ["not_waiting"]

    _run(scenario)


def test_a_node_the_run_context_does_not_fit_is_shown_at_compile_and_its_run_is_refused_before_it_starts(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served, count=1) as (a,):
            compiled = (await a.client.post("/compile", json=_graph({"needs_missing": {}}))).json()
            assert [(p["type"], p["loc"]) for p in compiled["problems"]] == [("context_does_not_fit", ["needs_missing", "value"])]
            created = await a.client.post("/runs", json={"graph": _graph({"needs_missing": {}}), "inputs": {}})
            assert created.status_code == 422
            assert [(p["type"], p["loc"]) for p in created.json()["detail"]] == [("context_does_not_fit", ["needs_missing", "value"])]

    _run(scenario)


def test_a_node_connects_to_the_database_itself_and_gets_the_caller_who_started_the_run(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served, count=1) as (a,):
            counted = await a.start(_graph({"count_runs": {"status": {"constant": "running"}}}, returns="count_runs"))
            assert (await a.until_status(counted, "complete"))["outputs"] == {"result": 1}

            a.client.headers["X-User"] = "bo"
            asked = await a.start(_graph({"who": {}}, returns="who"))
            assert (await a.until_status(asked, "complete"))["outputs"] == {"result": "bo"}

    _run(scenario)


def test_a_run_is_seen_only_by_the_caller_who_started_it(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served, count=1) as (a,):
            run_id = await a.start(_graph({"greet": {"name": {"graph_input": "name"}}}), {"name": "ida"})
            await a.until_status(run_id, "complete")
            for route in ("", "/events"):
                assert (await a.client.get(f"/runs/{run_id}{route}", headers={"X-User": "bo"})).status_code == 404
            assert (await a.client.post(f"/runs/{run_id}/answers", json={}, headers={"X-User": "bo"})).status_code == 404

    _run(scenario)


def test_a_wrong_answer_is_a_422_listing_every_problem_and_a_complete_run_is_a_409(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served, count=1) as (a,):
            served.open_the_gate()
            run_id = await a.start(_GATE_ASK)
            row = await a.until_status(run_id, "waiting")
            (waiting_call,) = row["waiting"]

            wrong = await a.client.post(f"/runs/{run_id}/answers", json={waiting_call["id"]: "not a bool", "nonesuch": True})
            assert wrong.status_code == 422 and len(wrong.json()["detail"]) == 2
            assert (await a.read(run_id))["status"] == "waiting"

            answered = await a.client.post(f"/runs/{run_id}/answers", json={waiting_call["id"]: False})
            assert answered.status_code == 202
            await a.until_status(run_id, "complete")
            assert (await a.client.post(f"/runs/{run_id}/answers", json={})).status_code == 409
            assert (await a.client.post("/runs/nonesuch/answers", json={})).status_code == 404

    _run(scenario)


def test_a_graph_that_cannot_run_is_a_422_before_anything_is_saved_and_compile_says_where(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served, count=1) as (a,):
            broken = {"nodes": {"x": {"node": "nonesuch"}}}
            refused = await a.client.post("/runs", json={"graph": broken})
            assert refused.status_code == 422 and [p["type"] for p in refused.json()["detail"]] == ["unknown_node"]
            described = (await a.client.post("/compile", json=broken)).json()
            assert [p["type"] for p in described["nodes"]["x"]["problems"]] == ["unknown_node"]
            assert "greet" in {node["id"] for node in (await a.client.get("/nodes")).json()["nodes"]}

    _run(scenario)


def test_an_edited_input_runs_again_calling_only_the_items_that_changed(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served, count=1) as (a,):
            run_id = await a.start(_WORK, {"n": 3})
            await a.until_status(run_id, "complete")
            again = await a.client.post(f"/runs/{run_id}/again", json={"n": 4})
            assert again.status_code == 202
            assert (await a.until_status(again.json()["id"], "complete"))["outputs"] == {"result": [0, 1, 2, 3]}
            assert served.calls == ["work 0", "work 1", "work 2", "work 3"]

    _run(scenario)


def test_an_end_user_fills_the_inputs_of_a_published_graph_and_names_no_node(served: Served) -> None:
    (served.directory / "served_app.py").write_text(
        _app_source() + '\noffered["hello"] = GraphRecord.model_validate({"nodes": {"greet": {"node": "greet", "inputs": {"name": {"graph_input": "name"}}}}, "outputs": {"greeting": {"node": "greet", "output": "result"}}})\n'
    )

    async def scenario() -> None:
        async with _workers(served, count=1) as (a,):
            created = await a.client.post("/graphs/hello/runs", json={"name": "ida"})
            assert created.status_code == 202, created.text
            assert (await a.until_status(created.json()["id"], "complete"))["outputs"] == {"greeting": "hi ida"}
            assert (await a.client.post("/graphs/nope/runs", json={})).status_code == 404

    _run(scenario)


def test_a_stream_says_not_to_cache_it_and_the_schema_says_what_a_run_reads_as(served: Served) -> None:
    async def scenario() -> None:
        async with _workers(served, count=1) as (a,):
            served.open_the_gate()
            run_id = await a.start(_graph({"gate": {}}))
            response = await a.client.get(f"/runs/{run_id}/events")
            assert response.headers["cache-control"] == "no-cache"
            assert response.headers["content-type"].startswith("text/event-stream")
            assert (await a.client.get("/runs/nope/events")).status_code == 404

            schemas = (await a.client.get("/openapi.json")).json()["components"]["schemas"]
            assert {"id", "status", "waiting", "outputs", "refused", "cause"} <= set(schemas["RunView"]["properties"])
            assert "rev" not in schemas["RunView"]["properties"]
            assert {"id", "loc", "prompt", "questions", "inputs"} <= set(schemas["Waiting"]["required"])

    _run(scenario)
