"""What ``run_router``'s routes add to the worker they call, served in this process over a store in memory: who the
caller is and what they may see, the status and body of each answer and refusal, the stream of a follower, and the
schema. ``test_worker.py`` has what the worker does."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from threading import Event
from typing import Annotated, Any

import pytest

httpx = pytest.importorskip("httpx")

from conductor import (  # noqa: E402
    Ask,
    Context,
    GraphRecord,
    NodeRegistry,
    Policy,
    Problem,
    Series,
    make_graph,
)
from conductor_providers.fastapi import ROUTER_PROBLEMS, RunChanges, run_router  # noqa: E402
from fastapi import FastAPI, Header  # noqa: E402

from tests.test_providers.in_memory import MemoryStore, asking  # noqa: E402


def _caller(x_user: Annotated[str, Header()] = "anna") -> str:
    return x_user


def _starter_only(caller: object, started_by: object) -> bool:
    return caller == started_by


def _app(**settings: Any) -> FastAPI:
    app = FastAPI()
    app.include_router(run_router(**{"caller": _caller, "may_see": _starter_only, **settings}))
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

    @full.node("double", title="Double", description="For a test.")
    def double(x: int) -> int:
        return 2 * x

    return full, bare, GraphRecord.model_validate({"nodes": {"d": {"node": "double", "inputs": {"x": {"graph_input": "x"}}}}, "outputs": {"result": {"node": "d", "output": "result"}}}).model_dump(mode="json")


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
        assert (await store.ended(mine.json()["id"])).outputs == {"result": 4}


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
    registry, seen = NodeRegistry(context_type=_Member), []

    @registry.node("whoami", title="Whoami", description="For a test.")
    def whoami(who: Context[_Member], text: str) -> str:
        seen.append(who)
        return who.x_user

    store = MemoryStore()
    app = _app(store=store, registry=lambda _: registry, caller=_Member, context=lambda caller: caller)

    async with _serving(app) as client:
        started = await client.post("/runs", json={"graph": make_graph(whoami).model_dump(mode="json"), "inputs": {"text": "hi"}}, headers={"x-user": "bo"})
        await store.ended(started.json()["id"])
        row = (await client.get(f"/runs/{started.json()['id']}", headers={"x-user": "bo"})).json()
        assert (row["status"], row["outputs"]) == ("complete", {"result": "bo"})
        assert seen == [_Member("bo")]


def test_a_caller_with_no_return_annotation_is_refused_when_the_router_is_made() -> None:
    with pytest.raises(TypeError, match="return annotation"):
        run_router(store=MemoryStore(), registry=lambda _: NodeRegistry(), caller=lambda: "anna", may_see=_starter_only)


def _nobody() -> None:
    return None


async def test_a_router_with_no_logins_lets_every_request_see_every_run() -> None:
    registry, _, graph = _doubling()
    store = MemoryStore()
    app = FastAPI()
    app.include_router(run_router(store=store, registry=lambda _: registry, caller=_nobody, may_see=lambda caller, started_by: True))

    async with _serving(app) as client:
        run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"x": 2}})).json()["id"]
        assert (await store.ended(run_id)).outputs == {"result": 4}
        assert (await client.get(f"/runs/{run_id}", headers={"x-user": "bo"})).status_code == 200


async def test_the_registry_is_asked_for_the_callers_nodes_once_for_each_request_and_once_for_each_resume() -> None:
    full, _, graph = _doubling()
    asked: list[str] = []

    def nodes_of(who: str) -> NodeRegistry:
        asked.append(who)
        return full

    store = MemoryStore()
    async with _serving(_app(store=store, registry=nodes_of)) as client:
        assert (await client.post("/compile", json=graph, headers={"x-user": "bo"})).status_code == 200
        assert asked == ["bo"]
        run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"x": 1}})).json()["id"]
        assert (await store.ended(run_id)).status == "complete"
        assert asked == ["bo", "anna", "anna"]  # the start, then the worker's resume


async def test_one_registry_for_every_caller_serves_its_nodes() -> None:
    registry, _, _ = _doubling()

    async with _serving(_app(store=MemoryStore(), registry=lambda _: registry)) as client:
        assert [node["id"] for node in (await client.get("/nodes")).json()["nodes"]] == ["double"]


# -- what each route answers ----------------------------------------------------------------------


async def test_a_route_answers_with_what_the_worker_did_and_the_run_reads_as_its_view() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        started = await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})
        run_id = started.json()["id"]
        view = (await client.get(f"/runs/{run_id}")).json()
        progress = view.pop("progress")
        assert (started.status_code, view) == (
            202, {"id": run_id, "status": "running", "waiting": [], "outputs": None, "refused": {}, "cause": None},
        )
        assert (progress["finished"], progress["waiting"]) == (0, 0)  # the run as it was stored, before any call
        cancelled = await client.post(f"/runs/{run_id}/cancel")
        assert (cancelled.status_code, cancelled.json()) == (200, {"cancelled": True})
        gate.set()
        assert (await store.ended(run_id)).status == "cancelled"
        assert (await client.post(f"/runs/{run_id}/cancel")).json() == {"cancelled": False}

        below_one = await client.post(f"/runs/{run_id}/resume", json={"max_items": 0})
        assert below_one.status_code == 422
        assert [(problem["type"], problem["loc"]) for problem in below_one.json()["detail"]] == [("invalid_max_items", ["max_items"])]
        for not_a_whole_number in (True, 2.5, "5"):
            refused = await client.post(f"/runs/{run_id}/resume", json={"max_items": not_a_whole_number})
            assert (refused.status_code, [problem["type"] for problem in refused.json()["detail"]]) == (422, ["int_type"])
        assert (await client.get(f"/runs/{run_id}")).json()["status"] == "cancelled"
        resumed = await client.post(f"/runs/{run_id}/resume")
        assert (resumed.status_code, resumed.json()) == (202, {"id": run_id})

        waiting = await store.ended(run_id)
        assert (await client.get(f"/runs/{run_id}")).json() == waiting.model_dump(mode="json")
        answered = await client.post(f"/runs/{run_id}/answers", json={waiting_id: True})
        assert (answered.status_code, answered.json()) == (202, {"answers": "started"})
        assert (await store.ended(run_id)).outputs == {"busy": "x", "after": True}


async def test_an_answer_to_a_question_that_waits_in_a_cancelled_run_starts_the_next_resume_call() -> None:
    registry, asked, gate = NodeRegistry(), Event(), Event()

    @registry.node("busy", title="Busy", description="For a test.")
    def busy(text: str) -> str:
        gate.wait(5)
        return text

    @registry.node("approve", title="Approve", description="For a test.", asks=True)
    def approve(question: str) -> bool:
        asked.set()
        raise Ask("Go on?")

    @registry.node("then", title="Then", description="For a test.")
    def then(text: str) -> str:
        return text.upper()

    graph = make_graph(approve, busy, then, edges=[("busy.result", "then.text")], outputs={"then": "then.result", "approve": "approve.result"}).model_dump(mode="json")
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"text": "x", "question": "y"}})).json()["id"]
        await asyncio.to_thread(asked.wait, 5)
        assert (await client.post(f"/runs/{run_id}/cancel")).json() == {"cancelled": True}
        gate.set()
        cancelled = await store.ended(run_id)
        assert (cancelled.status, len(cancelled.waiting)) == ("cancelled", 1)

        answered = await client.post(f"/runs/{run_id}/answers", json={cancelled.waiting[0].id: True})
        assert (answered.status_code, answered.json()) == (202, {"answers": "started"})
        assert (await store.ended(run_id)).outputs == {"then": "X", "approve": True}


async def test_every_refusal_of_the_router_is_a_list_of_problems_with_a_code_of_its_own() -> None:
    registry, graph, gate, inputs, waiting_id = asking()

    class ClaimLost(MemoryStore):
        """Loses every claim of a run that waits, as when another worker took it first."""

        async def update(self, run_id: str, rev: int, changes: RunChanges) -> bool:
            return not ("owner" in changes and self.rows[run_id].status == "waiting") and await super().update(run_id, rev, changes)

    store = ClaimLost()
    app = _app(store=store, registry=lambda _: registry)

    def refusal(response: Any) -> tuple[int, list[tuple[str, dict[str, Any]]]]:
        problems = [Problem.model_validate(problem) for problem in response.json()["detail"]]
        assert all(problem.type in ROUTER_PROBLEMS.codes and problem.msg for problem in problems)
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


# -- a follower -----------------------------------------------------------------------------------


async def test_a_follower_gets_each_event_as_a_frame_with_its_number_and_last_event_id_goes_on_after_it() -> None:
    registry, graph, gate, inputs, _ = asking()
    gate.set()
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: registry, poll=0.01)) as client:
        run_id = (await client.post("/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs})).json()["id"]
        await store.ended(run_id)
        whole = await client.get(f"/runs/{run_id}/events")
        frames = _frames(whole)
        assert frames == await store.stored_events(run_id) and (frames[-1]["type"], frames[-1]["status"]) == ("run_end", "waiting")
        assert [line for line in whole.text.splitlines() if line.startswith("id: ")] == [f"id: {number}" for number in range(len(frames))]
        rest = await client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": str(len(frames) - 3)})
        assert _frames(rest) == frames[-2:]
        assert "data: " not in (await client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": str(len(frames) - 1)})).text


# -- progress -------------------------------------------------------------------------------------


def _uploads() -> tuple[NodeRegistry, dict[str, Any], list[str]]:
    """A graph whose first node costs nothing and hands on the files it is given, then a node per file."""
    registry, calls = NodeRegistry(), []

    @registry.node("upload", title="Upload", description="For a test.", policy=Policy(costs_nothing=True))
    def upload(files: list[str]) -> Series[str]:
        calls.append("upload")
        return files

    @registry.node("review", title="Review", description="For a test.")
    def review(file: str) -> str:
        calls.append("review")
        return file

    return registry, make_graph(upload, review).model_dump(mode="json"), calls


async def test_estimate_calls_only_the_nodes_that_cost_nothing_and_stores_nothing() -> None:
    registry, graph, calls = _uploads()
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: registry, published=lambda name: GraphRecord.model_validate(graph))) as client:
        estimated = await client.post("/estimate", json={"graph": graph, "inputs": {"files": ["a", "b", "c"]}})
        published = await client.post("/graphs/uploads/estimate", json={"files": ["a"]})

    assert estimated.status_code == 200
    assert estimated.json()["nodes"]["review"] == {"finished": 0, "waiting": 0, "known": 3, "total": 3}
    assert published.json()["nodes"]["review"]["total"] == 1
    assert calls == ["upload", "upload"] and store.rows == {}


async def test_estimate_refuses_inputs_the_graph_refuses() -> None:
    registry, graph, _ = _uploads()

    async with _serving(_app(store=MemoryStore(), registry=lambda _: registry)) as client:
        refused = await client.post("/estimate", json={"graph": graph, "inputs": {}})

    assert refused.status_code == 422 and [problem["type"] for problem in refused.json()["detail"]] == ["missing_input"]


async def test_a_graph_that_writes_one_node_id_twice_is_refused_by_every_route_that_reads_a_graph() -> None:
    registry, _, _ = _uploads()
    store = MemoryStore()
    node = '{"node": "review", "inputs": {"file": {"graph_input": "file"}}}'
    graph = f'{{"nodes": {{"review": {node}, "review": {node}}}}}'
    as_json = {"content-type": "application/json"}

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        answers = [
            await client.post("/compile", content=graph, headers=as_json),
            await client.post("/runs", content=f'{{"graph": {graph}, "inputs": {{"file": "a"}}}}', headers=as_json),
            await client.post("/estimate", content=f'{{"graph": {graph}, "inputs": {{"file": "a"}}}}', headers=as_json),
        ]

    for answer in answers:
        assert answer.status_code == 422, answer.text
        assert [(problem["type"], problem["ctx"]["error"]) for problem in answer.json()["detail"]] == [("invalid_record", "repeated_key")]
    assert store.rows == {}


async def test_a_run_body_that_writes_graph_twice_is_refused_and_nothing_is_stored() -> None:
    registry, good, _ = _uploads()
    store = MemoryStore()
    bad = '{"nodes": {"review": {"inputs": {}}}}'
    body = f'{{"graph": {bad}, "graph": {json.dumps(good)}, "inputs": {{"files": ["a"]}}}}'

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        answer = await client.post("/runs", content=body, headers={"content-type": "application/json"})

    assert answer.status_code == 422, answer.text
    assert [(problem["type"], problem["ctx"]["error"]) for problem in answer.json()["detail"]] == [("invalid_record", "repeated_key")]
    assert store.rows == {}


@pytest.mark.parametrize(
    ("graph", "error"),
    [
        ('{"nodes": {"review": {"inputs": {"file": {"graph_input": "file"}}}}}', "missing"),
        ('{"nodes": {"review": {"node": "review"}, "review": {"node": "review"}}}', "repeated_key"),
    ],
    ids=["a_node_with_no_node", "a_node_id_written_twice"],
)
async def test_a_run_of_a_graph_record_of_the_wrong_shape_is_refused_with_conductors_problems_never_fastapis(graph: str, error: str) -> None:
    registry, _, _ = _uploads()
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        answer = await client.post("/runs", content=f'{{"graph": {graph}, "inputs": {{}}}}', headers={"content-type": "application/json"})

    assert answer.status_code == 422, answer.text
    detail = answer.json()["detail"]
    assert [(problem["type"], problem["ctx"]["error"]) for problem in detail] == [("invalid_record", error)]
    assert all(problem["loc"][0] == "record" for problem in detail)
    assert store.rows == {}


async def test_a_run_read_once_it_stopped_has_its_progress() -> None:
    registry, graph, _ = _uploads()
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"files": ["a", "b"]}})).json()["id"]
        await store.ended(run_id)
        view = (await client.get(f"/runs/{run_id}")).json()

    assert (view["progress"]["finished"], view["progress"]["total"]) == (3, 3)
    assert view["progress"]["nodes"]["review"] == {"finished": 2, "waiting": 0, "known": 2, "total": 2}


# -- the schema, the surface, the page ------------------------------------------------------------


def test_an_operations_id_is_its_route_functions_name_without_fastapis_path_and_verb() -> None:
    schema = _app(store=MemoryStore(), registry=lambda _: NodeRegistry(), published=lambda name: GraphRecord()).openapi()

    assert schema["paths"]["/nodes"]["get"]["operationId"] == "list_nodes"
    assert schema["paths"]["/runs/{run_id}/answers"]["post"]["operationId"] == "answer_run"


def test_the_schema_names_each_operation_and_body_as_a_generated_client_reads_them() -> None:
    schema = _app(store=MemoryStore(), registry=lambda _: NodeRegistry(), published=lambda name: GraphRecord()).openapi()
    operations = {(verb, path): operation for path, verbs in schema["paths"].items() for verb, operation in verbs.items()}

    assert {key: operation["summary"] for key, operation in operations.items()} == {
        ("get", "/nodes"): "List Nodes",
        ("post", "/compile"): "Compile Graph",
        ("post", "/runs"): "Start Run",
        ("post", "/estimate"): "Estimate Run",
        ("post", "/graphs/{name}/runs"): "Start Published",
        ("post", "/graphs/{name}/estimate"): "Estimate Published",
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

    assert sorted(provider.__all__) == ["ROUTER_PROBLEMS", "RunChanges", "RunStore", "StoredRun", "run_router"]
