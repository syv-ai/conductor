"""What ``run_router``'s routes add to the worker they call, served in this process over a store in memory: who the
caller is and what they may see, the status and body of each answer and refusal, the stream of a follower, and the
schema. ``test_worker.py`` has what the worker does."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import pathlib
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, Any

import pytest

httpx = pytest.importorskip("httpx")

from conductor import FromRun, GraphRecord, NodeRecord, NodeRegistry, Problem, make_graph  # noqa: E402
from conductor_providers.fastapi import ROUTER_PROBLEM_CODES, RunChanges, StoredRun, run_router  # noqa: E402
from fastapi import FastAPI, Header  # noqa: E402

from tests.test_providers.in_memory import MemoryStore, asking  # noqa: E402


def _caller(x_user: Annotated[str, Header()] = "anna") -> str:
    return x_user


def _app(**settings: Any) -> FastAPI:
    app = FastAPI()
    app.include_router(run_router(**{"caller": _caller, **settings}))
    return app


@asynccontextmanager
async def _serving(app: FastAPI) -> AsyncIterator[Any]:
    async with app.router.lifespan_context(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client:
        yield client


def _frames(response: Any) -> list[dict[str, Any]]:
    return [json.loads(line.removeprefix("data: ")) for line in response.text.splitlines() if line.startswith("data: ")]


def _doubling() -> tuple[NodeRegistry, NodeRegistry, dict[str, Any]]:
    """A registry with a node, one without it, and a graph of that node."""
    full, bare = NodeRegistry(), NodeRegistry()

    @full.node("double", title="Double", description="")
    def double(x: int) -> int:
        return 2 * x

    return full, bare, GraphRecord(nodes=[NodeRecord(id="d", node="double")]).model_dump(mode="json")


# -- the caller -----------------------------------------------------------------------------------


async def test_again_by_another_caller_who_may_see_the_run_is_checked_on_that_callers_own_nodes() -> None:
    full, bare, graph = _doubling()
    store = MemoryStore()
    app = _app(store=store, registry=lambda who: full if who == "anna" else bare, may_see=lambda caller, started_by: True)

    async with _serving(app) as client:
        run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"x": 1}})).json()["id"]
        assert (await store.ended(run_id)).status == "complete"
        again = await client.post(f"/runs/{run_id}/again", json={"x": 2}, headers={"x-user": "bo"})
        assert again.status_code == 422
        assert [problem["type"] for problem in again.json()["detail"]] == ["unknown_node"]
        mine = await client.post(f"/runs/{run_id}/again", json={"x": 2})
        assert mine.status_code == 202
        assert (await store.ended(mine.json()["id"])).outputs == {"d.result": 4}


def _maybe_caller(x_user: Annotated[str | None, Header()] = None) -> str | None:
    return x_user


async def test_again_by_an_anonymous_caller_is_checked_on_the_nodes_of_no_caller_not_the_starters() -> None:
    full, bare, graph = _doubling()
    store = MemoryStore()
    app = _app(store=store, registry=lambda who: full if who == "anna" else bare, caller=_maybe_caller, may_see=lambda caller, started_by: True)

    async with _serving(app) as client:
        run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"x": 1}}, headers={"x-user": "anna"})).json()["id"]
        assert (await store.ended(run_id)).status == "complete"
        anonymous = await client.post(f"/runs/{run_id}/again", json={"x": 2})
        assert anonymous.status_code == 422
        assert [problem["type"] for problem in anonymous.json()["detail"]] == ["unknown_node"]


@dataclasses.dataclass(frozen=True)
class _Member:
    """A caller as FastAPI's class dependency: ``Depends(_Member)`` reads ``x_user`` from a header."""

    x_user: Annotated[str, Header()] = "anna"


async def test_a_class_as_the_caller_is_kept_as_that_class_so_its_starter_sees_the_run() -> None:
    registry, seen = NodeRegistry(), []

    @registry.node("whoami", title="Whoami", description="")
    def whoami(text: str, who: FromRun[_Member]) -> str:
        seen.append(who)
        return who.x_user

    store = MemoryStore()
    app = _app(store=store, registry=registry, caller=_Member, from_run=lambda caller: {_Member: caller})

    async with _serving(app) as client:
        started = await client.post("/runs", json={"graph": make_graph(whoami).model_dump(mode="json"), "inputs": {"text": "hi"}}, headers={"x-user": "bo"})
        await store.ended(started.json()["id"])
        row = (await client.get(f"/runs/{started.json()['id']}", headers={"x-user": "bo"})).json()
        assert (row["status"], row["outputs"]) == ("complete", {"whoami.result": "bo"})
        assert seen == [_Member("bo")]


def test_a_caller_with_no_return_annotation_is_refused_when_the_router_is_made() -> None:
    with pytest.raises(TypeError, match="return annotation"):
        run_router(store=MemoryStore(), registry=NodeRegistry(), caller=lambda: "anna")


# -- what each route answers ----------------------------------------------------------------------


async def test_a_route_answers_with_what_the_worker_did_and_the_run_reads_as_its_view() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()

    async with _serving(_app(store=store, registry=registry)) as client:
        started = await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})
        run_id = started.json()["id"]
        assert (started.status_code, (await client.get(f"/runs/{run_id}")).json()) == (
            202, {"id": run_id, "status": "running", "waiting": [], "outputs": None, "refused": {}, "cause": None},
        )
        cancelled = await client.post(f"/runs/{run_id}/cancel")
        assert (cancelled.status_code, cancelled.json()) == (200, {"cancelled": True})
        gate.set()
        assert (await store.ended(run_id)).status == "cancelled"
        assert (await client.post(f"/runs/{run_id}/cancel")).json() == {"cancelled": False}

        below_one = await client.post(f"/runs/{run_id}/resume", json={"max_items": 0})
        assert below_one.status_code == 422
        assert [(problem["type"], problem["loc"]) for problem in below_one.json()["detail"]] == [("greater_than", ["body", "max_items"])]
        assert (await client.get(f"/runs/{run_id}")).json()["status"] == "cancelled"
        resumed = await client.post(f"/runs/{run_id}/resume")
        assert (resumed.status_code, resumed.json()) == (202, {"id": run_id})

        waiting = await store.ended(run_id)
        assert (await client.get(f"/runs/{run_id}")).json() == waiting.model_dump(mode="json")
        answered = await client.post(f"/runs/{run_id}/answers", json={waiting_id: True})
        assert (answered.status_code, answered.json()) == (202, {"answers": "started"})
        assert (await store.ended(run_id)).outputs == {"busy.result": "x", "after.result": True}


async def test_every_refusal_of_the_router_is_a_list_of_problems_with_a_code_of_its_own() -> None:
    registry, graph, gate, inputs, waiting_id = asking()

    class ClaimLost(MemoryStore):
        """Loses every claim of a run that waits, as when another worker took it first."""

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            return not ("owner" in changes and self.rows[run_id].status == "waiting") and await super().update(run_id, rev, changes)

    store = ClaimLost()
    app = _app(store=store, registry=registry)

    def refusal(response: Any) -> tuple[int, list[tuple[str, dict[str, Any]]]]:
        problems = [Problem.model_validate(problem) for problem in response.json()["detail"]]
        assert all(problem.type in ROUTER_PROBLEM_CODES and problem.msg for problem in problems)
        return response.status_code, [(problem.type, dict(problem.ctx)) for problem in problems]

    async with _serving(app) as client:
        assert refusal(await client.get("/runs/nonesuch")) == (404, [("no_such_run", {})])
        run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
        assert refusal(await client.get(f"/runs/{run_id}", headers={"x-user": "bo"})) == (404, [("no_such_run", {})])
        assert (await client.post(f"/runs/{run_id}/answers", json={waiting_id: True})).json() == {"answers": "queued"}
        assert refusal(await client.post(f"/runs/{run_id}/answers", json={waiting_id: False})) == (409, [("already_answered", {"waiting_id": waiting_id})])
        assert refusal(await client.post(f"/runs/{run_id}/resume")) == (409, [("run_not_resumable", {"status": "running"})])
        gate.set()
        assert (await store.ended(run_id)).status == "complete"
        assert refusal(await client.post(f"/runs/{run_id}/answers", json={})) == (409, [("run_complete", {})])
        assert refusal(await client.post(f"/runs/{run_id}/resume")) == (409, [("run_not_resumable", {"status": "complete"})])
        waits = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
        assert (await store.ended(waits)).status == "waiting"
        assert refusal(await client.post(f"/runs/{waits}/answers", json={waiting_id: True})) == (409, [("run_changed", {})])
        wrong = await client.post(f"/runs/{waits}/answers", json={waiting_id: "not a bool", "nonesuch": True})
        assert wrong.status_code == 422 and len(wrong.json()["detail"]) == 2

    refused = {"application/json": {"schema": {"$ref": "#/components/schemas/Refused"}}}
    declared = {
        (verb, path): {code: response["content"] for code, response in operation["responses"].items() if code in ("404", "409")}
        for path, verbs in app.openapi()["paths"].items() for verb, operation in verbs.items() if path.startswith("/runs/")
    }
    assert declared == {
        ("get", "/runs/{run_id}"): {"404": refused},
        ("get", "/runs/{run_id}/events"): {"404": refused},
        ("post", "/runs/{run_id}/answers"): {"404": refused, "409": refused},
        ("post", "/runs/{run_id}/resume"): {"404": refused, "409": refused},
        ("post", "/runs/{run_id}/again"): {"404": refused},
        ("post", "/runs/{run_id}/cancel"): {"404": refused},
    }


@pytest.mark.parametrize(("route", "body"), [("answers", {"anything": True}), ("cancel", None)])
async def test_a_run_that_is_gone_when_a_write_to_it_is_tried_again_is_a_404(route: str, body: dict[str, Any] | None) -> None:
    registry, graph, gate, inputs, _ = asking()

    class GoneAtTheWrite(MemoryStore):
        """Loses what a caller asks of a running leg and has no run to read again, as when the host deleted it meanwhile."""

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            if changes.keys() == {"status", "state"}:
                del self.rows[run_id]
                return False
            return await super().update(run_id, rev, changes)

    async with _serving(_app(store=GoneAtTheWrite(), registry=registry)) as client:
        run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
        gone = await client.post(f"/runs/{run_id}/{route}", json=body)
        assert (gone.status_code, [problem["type"] for problem in gone.json()["detail"]]) == (404, ["no_such_run"])
        gate.set()


# -- a follower -----------------------------------------------------------------------------------


async def test_a_follower_reads_the_row_only_to_learn_whether_a_run_end_was_the_last() -> None:
    registry, graph, gate, inputs, _ = asking()

    class CountsReads(MemoryStore):
        reads = 0

        async def get(self, run_id: str) -> StoredRun | None:
            self.reads += 1
            return await super().get(run_id)

    store = CountsReads()

    async with _serving(_app(store=store, registry=registry, poll=0.01)) as client:
        run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
        following = asyncio.create_task(client.get(f"/runs/{run_id}/events"))
        await asyncio.sleep(0.1)  # the follower has the leg's first events by now
        before = store.reads
        await asyncio.sleep(0.3)  # thirty polls that find no event
        during = store.reads - before
        gate.set()
        frames = _frames(await following)
        assert (during, frames[-1]["type"], frames[-1]["status"]) == (0, "run_end", "waiting")
        again = (await client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": str(len(frames) - 1)})).text
        assert "data: " not in again  # a follower that has every event of a run that waits is told so at once


async def test_a_follower_that_sent_a_run_end_while_the_row_still_reads_running_keeps_asking_until_it_changes() -> None:
    registry = NodeRegistry()

    @registry.node("shout", title="Shout", description="")
    def shout(text: str) -> str:
        return text.upper()

    class RowLags(MemoryStore):
        """Reads a run that has ended as still running a few times, as a reader between the ``run_end`` and the row's write does."""

        lagged = 0

        async def get(self, run_id: str) -> StoredRun | None:
            row = await super().get(run_id)
            if row is not None and row.status == "complete" and self.lagged < 5:
                self.lagged += 1
                return dataclasses.replace(row, status="running")
            return row

    store = RowLags()

    async with _serving(_app(store=store, registry=registry, poll=0.01)) as client:
        run_id = (await client.post("/runs", json={"graph": make_graph(shout).model_dump(mode="json"), "inputs": {"text": "a"}})).json()["id"]
        await store.ended(run_id)
        frames = _frames(await asyncio.wait_for(client.get(f"/runs/{run_id}/events"), 5))
        assert [frame["type"] for frame in frames].count("run_end") == 1 and frames[-1]["type"] == "run_end"
        assert store.lagged == 5


# -- the schema, the surface, the page ------------------------------------------------------------


def test_the_schema_names_each_operation_and_body_as_a_generated_client_reads_them() -> None:
    schema = _app(store=MemoryStore(), registry=NodeRegistry(), published=lambda name: GraphRecord()).openapi()
    operations = {(verb, path): operation for path, verbs in schema["paths"].items() for verb, operation in verbs.items()}

    assert {key: operation["summary"] for key, operation in operations.items()} == {
        ("get", "/nodes"): "List Nodes",
        ("post", "/load"): "Load Graph",
        ("post", "/runs"): "Start Run",
        ("post", "/graphs/{name}/runs"): "Start Published",
        ("get", "/runs/{run_id}"): "Read Run",
        ("get", "/runs/{run_id}/events"): "Follow Run",
        ("post", "/runs/{run_id}/answers"): "Answer Run",
        ("post", "/runs/{run_id}/resume"): "Resume Run",
        ("post", "/runs/{run_id}/again"): "Run Again",
        ("post", "/runs/{run_id}/cancel"): "Cancel Run",
    }
    assert all(operation.get("description") for operation in operations.values())
    assert "Resume" in schema["components"]["schemas"] and not [name for name in schema["components"]["schemas"] if name.startswith("Body_")]
    assert operations["post", "/runs/{run_id}/resume"]["requestBody"]["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/Resume", "default": {}}
    for key in [("post", "/graphs/{name}/runs"), ("post", "/runs/{run_id}/answers"), ("post", "/runs/{run_id}/again")]:
        assert operations[key]["requestBody"]["content"]["application/json"]["schema"]["description"], key


def test_the_package_exports_what_a_host_writes_and_calls() -> None:
    import conductor_providers.fastapi as provider

    assert sorted(provider.__all__) == ["ROUTER_PROBLEM_CODES", "RunChanges", "RunStore", "StoredRun", "run_router"]


def test_the_problems_page_quotes_the_sentence_of_each_of_the_routers_codes() -> None:
    from conductor_providers.fastapi.worker import _ROUTER_PROBLEMS

    page = (pathlib.Path(__file__).parents[2] / "docs" / "problems.md").read_text()
    section = page[page.index("## The runs router"):]
    parts = re.split(r"^### `([a-z_]+)`$", section, flags=re.M)
    quoted = {code: body.split("\n\n")[1] for code, body in zip(parts[1::2], parts[2::2])}
    assert quoted == {code: f"> {sentence}" for code, sentence in _ROUTER_PROBLEMS.items()}
