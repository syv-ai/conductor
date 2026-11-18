"""Tests of what ``run_router``'s routes add on top of the worker they call.

The app is served in this process, over a store in memory. The tests cover who the caller is and which runs they may
see, the HTTP status and body of each answer and refusal, the event stream a follower reads, and the OpenAPI schema.
What the worker itself does is tested in ``test_worker.py``.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from threading import Event
from typing import Annotated, Any

import pytest

httpx = pytest.importorskip("httpx")

from conductor import (  # noqa: E402
    Ask,
    Context,
    Graph,
    GraphRecord,
    InvalidCallError,
    NodeRegistry,
    Policy,
    Problem,
    Run,
    Series,
    make_graph,
)
from conductor_providers.fastapi import RunView, StoredRun, Worker, run_router  # noqa: E402
from fastapi import FastAPI, Header  # noqa: E402
from pydantic import BaseModel, Field, ValidationError  # noqa: E402

from tests.test_providers.in_memory import (  # noqa: E402
    MemoryStore,
    asking,
    left_running,
    stoppable,
    stopping,
)


def _metadata(x_user: Annotated[str, Header()] = "anna") -> dict[str, str]:
    return {"user": x_user}


def _own_runs(x_user: Annotated[str, Header()] = "anna") -> Callable[[dict[str, str]], bool]:
    return lambda metadata: metadata["user"] == x_user


def _every_run() -> Callable[[Any], bool]:
    return lambda metadata: True


def _app(
    *,
    registry: Any,
    metadata: Any = _metadata,
    run_filter: Any = _own_runs,
    **settings: Any,
) -> FastAPI:
    """Builds an app with the run routes, over a worker made from ``settings``.

    ``settings`` must include the worker's ``store``. By default, each run keeps the user named by the ``x-user``
    header as its metadata, and only that user may see it. ``metadata`` and ``run_filter`` change that.
    """
    worker: Worker[Any, Any] = Worker(settings.pop("store"), nodes_for=registry, **settings)
    app = FastAPI()
    app.include_router(run_router(worker, metadata=metadata, run_filter=run_filter))
    return app


@asynccontextmanager
async def _serving(app: FastAPI) -> AsyncIterator[Any]:
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t") as client,
    ):
        yield client


def _frames(response: Any) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


def _doubling() -> tuple[NodeRegistry, NodeRegistry, dict[str, Any]]:
    """Returns a registry with the node ``double``, a registry without it, and a graph that uses it."""
    full, bare = NodeRegistry(), NodeRegistry()

    @full.node("double", title="Double", description="For a test.")
    def double(x: int) -> int:
        return 2 * x

    return (
        full,
        bare,
        GraphRecord.model_validate(
            {
                "nodes": {"d": {"node": "double", "inputs": {"x": {"graph_input": "x"}}}},
                "outputs": {"result": {"node": "d", "output": "result"}},
            }
        ).model_dump(mode="json"),
    )


# -- metadata and the run filter -----------------------------------------------------------------


async def test_again_by_another_user_who_may_see_the_run_is_checked_on_that_users_own_nodes() -> (
    None
):
    full, bare, graph = _doubling()
    store = MemoryStore()
    app = _app(
        store=store,
        registry=lambda metadata: full if metadata["user"] == "anna" else bare,
        run_filter=_every_run,
    )

    async with _serving(app) as client:
        run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"x": 1}})).json()[
            "id"
        ]
        assert (await store.ended(run_id)).status == "complete"
        again = await client.post(f"/runs/{run_id}/again", json={"x": 2}, headers={"x-user": "bo"})
        assert again.status_code == 422
        assert [problem["type"] for problem in again.json()["detail"]] == ["unknown_node"]
        mine = await client.post(f"/runs/{run_id}/again", json={"x": 2})
        assert mine.status_code == 202
        assert (await store.ended(mine.json()["id"])).outputs == {"result": 4}


async def test_a_previous_run_whose_saved_form_conductor_refuses_is_refused_with_conductors_own_problems() -> (
    None
):
    """A run named as ``previous`` is refused as ``Run.from_json`` refuses the same saved run: code, loc and ctx."""
    full, _, graph = _doubling()
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: full)) as client:
        first = (await client.post("/runs", json={"graph": graph, "inputs": {"x": 2}})).json()["id"]
        assert (await store.ended(first)).status == "complete"
        edited = json.loads(store.rows[first].run)
        edited["outcomes"]["d"]["outputs"]["result"] = ["four"]
        store.rows[first] = dataclasses.replace(store.rows[first], run=json.dumps(edited))
        answer = await client.post(
            "/runs", json={"graph": graph, "inputs": {"x": 3}, "previous": first}
        )

    with pytest.raises(InvalidCallError) as in_python:
        Run.from_json(json.dumps(edited), Graph.compile(graph, full))
    assert answer.status_code == 422
    assert answer.json()["detail"] == [
        problem.model_dump(mode="json") for problem in in_python.value.problems
    ]
    (problem,) = answer.json()["detail"]
    assert (problem["type"], problem["loc"], problem["ctx"]["where"]) == (
        "saved_run_unreadable",
        ["saved_run"],
        "outcomes.d.outputs.result.0",
    )
    assert list(store.rows) == [first]


class _Member(BaseModel):
    """A run's metadata as a typed model: who started the run, and in which team."""

    user: str
    team: str


def _member(
    x_user: Annotated[str, Header()] = "anna", x_team: Annotated[str, Header()] = "a"
) -> _Member:
    return _Member(user=x_user, team=x_team)


def _team_runs(x_team: Annotated[str, Header()] = "a") -> Callable[[_Member], bool]:
    return lambda member: member.team == x_team


async def test_a_run_posted_to_an_app_that_never_started_is_refused_with_how_to_start_it() -> None:
    """A run posted before the app started is refused, and the error says to use the app's TestClient in ``with``.

    The router's worker serves while the app's lifespan runs. A ``TestClient`` used without ``with`` never runs it,
    and that is how a developer most often meets this error.
    """
    registry, _, graph = _doubling()
    app = _app(store=MemoryStore(), registry=lambda _: registry)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://t"
    ) as client:
        with pytest.raises(RuntimeError, match=r"`with TestClient\(app\) as client:`"):
            await client.post("/runs", json={"graph": graph, "inputs": {"x": 1}})


@pytest.mark.parametrize(
    ("metadata", "run_filter", "lets_in"),
    [
        (_metadata, _own_runs, {("anna", "a"), ("anna", "b")}),
        (_member, _team_runs, {("anna", "a"), ("bo", "a")}),
        (_metadata, _every_run, {("anna", "a"), ("anna", "b"), ("bo", "a"), ("bo", "b")}),
    ],
    ids=["own_runs", "a_team", "everyone"],
)
async def test_the_run_filter_lets_in_the_requests_it_says_and_turns_the_rest_away_as_an_id_that_isnt_there(
    metadata: Any, run_filter: Any, lets_in: set[tuple[str, str]]
) -> None:
    registry, _, graph = _doubling()
    store = MemoryStore()
    app = _app(
        store=store,
        registry=lambda _: registry,
        metadata=metadata,
        run_filter=run_filter,
        metadata_type=_Member if metadata is _member else dict[str, str],
    )

    async with _serving(app) as client:
        run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"x": 1}})).json()[
            "id"
        ]
        await store.ended(run_id)
        seen = set()
        for user in ("anna", "bo"):
            for team in ("a", "b"):
                headers = {"x-user": user, "x-team": team}
                read = await client.get(f"/runs/{run_id}", headers=headers)
                if read.status_code == 200:
                    seen.add((user, team))
                else:
                    unknown = await client.get("/runs/nonesuch", headers=headers)
                    assert (read.status_code, read.json()) == (404, unknown.json())

    assert seen == lets_in


async def test_a_typed_metadata_model_is_what_nodes_for_and_context_get_on_every_resume_and_on_another_worker() -> (
    None
):
    registry, seen = NodeRegistry(context_type=_Member), []

    @registry.node("whoami", title="Whoami", description="For a test.")
    def whoami(who: Context[_Member], text: str) -> str:
        seen.append(who)
        return who.user

    picked: list[object] = []

    def nodes_of(member: _Member) -> NodeRegistry[_Member]:
        picked.append(member)
        return registry

    store = MemoryStore()
    graph = make_graph(whoami)
    left = dataclasses.replace(
        left_running(
            "left", graph, Graph.compile(graph, registry).with_inputs({"text": "hi"}), owner=None
        ),
        metadata=_Member(user="bo", team="b").model_dump_json(),
    )
    await store.create(left)
    worker: Worker[_Member, _Member] = Worker(
        store, metadata_type=_Member, nodes_for=nodes_of, context=lambda member, stop: member
    )
    async with worker.serving():  # takes over the run a dead worker left
        assert (await store.ended("left")).outputs == {"result": "bo"}

    assert seen == picked == [_Member(user="bo", team="b")]


def _no_metadata() -> dict[str, str]:
    return {}


async def test_a_router_with_no_logins_lets_every_request_see_every_run() -> None:
    registry, _, graph = _doubling()
    store = MemoryStore()
    app = FastAPI()
    app.include_router(
        run_router(
            Worker(store, nodes_for=lambda _: registry),
            metadata=_no_metadata,
            run_filter=_every_run,
        )
    )

    async with _serving(app) as client:
        run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"x": 2}})).json()[
            "id"
        ]
        assert (await store.ended(run_id)).outputs == {"result": 4}
        assert (await client.get(f"/runs/{run_id}", headers={"x-user": "bo"})).status_code == 200


async def test_the_access_check_reads_only_the_runs_metadata() -> None:
    """A follower can read a run whose ``outcome`` column isn't valid JSON.

    The access check reads only the run's metadata, so no route parses the outcome just to check access.
    """
    registry, graph, gate, inputs, _ = asking()
    gate.set()
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: registry, poll=0.01)) as client:
        run_id = (
            await client.post(
                "/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs}
            )
        ).json()["id"]
        await store.ended(run_id)
        store.rows[run_id] = dataclasses.replace(store.rows[run_id], outcome="not JSON")

        followed = await client.get(f"/runs/{run_id}/events")
        turned_away = await client.get(f"/runs/{run_id}/events", headers={"x-user": "bo"})

    assert followed.status_code == 200 and _frames(followed)[-1]["type"] == "run_end"
    assert turned_away.status_code == 404


async def test_the_registry_is_asked_for_the_requests_nodes_once_for_each_request_and_once_for_each_resume() -> (
    None
):
    full, _, graph = _doubling()
    asked: list[str] = []

    def nodes_of(metadata: dict[str, str]) -> NodeRegistry:
        asked.append(metadata["user"])
        return full

    store = MemoryStore()
    async with _serving(_app(store=store, registry=nodes_of)) as client:
        assert (
            await client.post("/compile", json=graph, headers={"x-user": "bo"})
        ).status_code == 200
        assert asked == ["bo"]
        run_id = (await client.post("/runs", json={"graph": graph, "inputs": {"x": 1}})).json()[
            "id"
        ]
        assert (await store.ended(run_id)).status == "complete"
        assert asked == ["bo", "anna", "anna"]  # the start, then the worker's resume call


async def test_one_registry_for_every_request_serves_its_nodes() -> None:
    registry, _, _ = _doubling()

    async with _serving(_app(store=MemoryStore(), registry=lambda _: registry)) as client:
        assert [node["id"] for node in (await client.get("/nodes")).json()["nodes"]] == ["double"]


# -- what each route answers ----------------------------------------------------------------------


def _status(response: Any) -> tuple[int, str]:
    """Returns a route's answer as a pair: its HTTP status, and the status of the run in its body."""
    return response.status_code, response.json()["status"]


async def test_every_action_answers_with_the_run_as_it_wrote_it() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        started = await client.post(
            "/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs}
        )
        run_id = started.json()["id"]
        view = started.json()
        progress = view.pop("progress")
        assert (started.status_code, view) == (
            202,
            {
                "id": run_id,
                "metadata": {"user": "anna"},
                "status": "running",
                "cause": None,
                "waiting": [],
                "outputs": None,
                "refused": {},
            },
        )
        assert (progress["finished"], progress["waiting"]) == (
            0,
            0,
        )  # the run as it was stored, before any call
        assert _status(await client.post(f"/runs/{run_id}/cancel")) == (200, "running")
        gate.set()
        assert (await store.ended(run_id)).status == "cancelled"
        assert _status(await client.post(f"/runs/{run_id}/cancel")) == (200, "cancelled")

        below_one = await client.post(f"/runs/{run_id}/resume", json={"max_items": 0})
        assert below_one.status_code == 422
        assert [
            (problem["type"], problem["loc"], problem["ctx"])
            for problem in below_one.json()["detail"]
        ] == [
            ("invalid_limit", ["max_items"], {"limit": "max_items", "value": 0}),
        ]
        for not_a_whole_number in (True, 2.5, "5"):
            refused = await client.post(
                f"/runs/{run_id}/resume", json={"max_items": not_a_whole_number}
            )
            assert (
                refused.status_code,
                [problem["type"] for problem in refused.json()["detail"]],
            ) == (422, ["int_type"])
        assert (await client.get(f"/runs/{run_id}")).json()["status"] == "cancelled"
        assert _status(await client.post(f"/runs/{run_id}/resume")) == (202, "running")

        waiting = await store.ended(run_id)
        assert (await client.get(f"/runs/{run_id}")).json() == waiting.model_dump(mode="json")
        assert _status(await client.post(f"/runs/{run_id}/cancel")) == (200, "running")
        await store.ended(run_id)
        view = (await client.get(f"/runs/{run_id}")).json()
        assert (view["status"], view["cause"]["type"], view["waiting"]) == (
            "cancelled",
            "run_cancelled",
            waiting.model_dump(mode="json")["waiting"],
        )
        answered = await client.post(f"/runs/{run_id}/answers", json={waiting_id: True})
        assert _status(answered) == (202, "running")
        assert (await store.ended(run_id)).outputs == {"busy": "x", "after": True}
        nothing = await client.post(f"/runs/{run_id}/answers", json={})  # changes nothing
        assert _status(nothing) == (202, "complete")


async def test_a_view_shows_no_queued_answer_nor_a_cancel_asked() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    app = _app(store=store, registry=lambda _: registry)

    async with _serving(app) as client:
        run_id = (
            await client.post(
                "/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs}
            )
        ).json()["id"]
        queued = (await client.post(f"/runs/{run_id}/answers", json={waiting_id: True})).json()
        cancelled = (await client.post(f"/runs/{run_id}/cancel")).json()
        gate.set()
        await store.ended(run_id)

    fields = {"id", "metadata", "status", "cause", "waiting", "outputs", "progress", "refused"}
    assert set(queued) == set(cancelled) == fields
    assert set(app.openapi()["components"]["schemas"]["RunView"]["properties"]) == fields


async def test_an_answer_or_a_reason_json_cant_write_is_the_same_422_whether_or_not_the_run_runs() -> (
    None
):
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    app = _app(store=store, registry=lambda _: registry)
    # Valid JSON for a str with a lone surrogate, which no JSON writer can write back.
    answer, decline = (
        '{"%s": "\\ud800"}' % waiting_id,
        '{"waiting_id": "%s", "reason": "\\ud800"}' % waiting_id,
    )

    async def refusals(run_id: str) -> list[tuple[int, Any]]:
        sent = [
            await client.post(
                f"/runs/{run_id}/{route}",
                content=body,
                headers={"content-type": "application/json"},
            )
            for route, body in (("answers", answer), ("declines", decline))
        ]
        return [(response.status_code, response.json()) for response in sent]

    async with _serving(app) as client:
        run_id = (
            await client.post(
                "/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs}
            )
        ).json()["id"]
        while_running = await refusals(run_id)
        gate.set()
        await store.ended(run_id)
        stopped = await refusals(run_id)

    assert while_running == stopped
    assert [(status, body["detail"][0]["type"]) for status, body in stopped] == [
        (422, "invalid_answer"),
        (422, "invalid_decline_reason"),
    ]


async def test_a_decline_body_is_the_waiting_id_and_the_reason_and_nothing_else() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    gate.set()
    store = MemoryStore()
    app = _app(store=store, registry=lambda _: registry)

    assert set(app.openapi()["components"]["schemas"]["Decline"]["properties"]) == {
        "waiting_id",
        "reason",
    }
    async with _serving(app) as client:
        run_id = (
            await client.post(
                "/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs}
            )
        ).json()["id"]
        await store.ended(run_id)
        refused = await client.post(
            f"/runs/{run_id}/declines", json={"waiting_id": waiting_id, "kind": "decline"}
        )

    assert refused.status_code == 422
    assert [(error["type"], error["loc"]) for error in refused.json()["detail"]] == [
        ("extra_forbidden", ["body", "kind"])
    ]
    assert (await store.view(run_id)).status == "waiting"


async def test_an_answer_to_a_question_that_waits_in_a_cancelled_run_starts_the_next_resume_call() -> (
    None
):
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

    graph = make_graph(
        approve,
        busy,
        then,
        edges=[("busy.result", "then.text")],
        outputs={"then": "then.result", "approve": "approve.result"},
    ).model_dump(mode="json")
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        run_id = (
            await client.post(
                "/runs", json={"graph": graph, "inputs": {"text": "x", "question": "y"}}
            )
        ).json()["id"]
        await asyncio.to_thread(asked.wait, 5)
        assert _status(await client.post(f"/runs/{run_id}/cancel")) == (200, "running")
        gate.set()
        cancelled = await store.ended(run_id)
        assert (cancelled.status, len(cancelled.waiting)) == ("cancelled", 1)

        answered = await client.post(
            f"/runs/{run_id}/answers", json={cancelled.waiting[0].id: True}
        )
        assert _status(answered) == (202, "running")
        assert (await store.ended(run_id)).outputs == {"then": "X", "approve": True}


async def test_a_cancel_sent_to_the_route_sets_the_stop_event_a_node_of_the_run_reads() -> None:
    graph = stopping()
    store = MemoryStore()
    app = _app(store=store, registry=lambda _: graph.registry, context=stoppable)

    async with _serving(app) as client:
        started = await client.post(
            "/runs",
            json={
                "graph": graph.graph.model_dump(mode="json"),
                "inputs": {"question": "y", "text": "x", "seconds": 60.0},
            },
        )
        run_id = started.json()["id"]
        await asyncio.to_thread(graph.started.wait, 60)
        assert _status(await client.post(f"/runs/{run_id}/cancel")) == (200, "running")
        assert (await store.ended(run_id)).status == "cancelled"
    assert graph.seen == [True]


async def test_every_refusal_is_a_list_of_problems_with_the_status_of_its_code() -> None:
    registry, graph, gate, inputs, waiting_id = asking()
    store = MemoryStore()
    app = _app(store=store, registry=lambda _: registry)

    def refusal(response: Any) -> tuple[int, list[tuple[str, dict[str, Any]]]]:
        problems = [Problem.model_validate(problem) for problem in response.json()["detail"]]
        assert all(problem.msg for problem in problems)
        return response.status_code, [(problem.type, dict(problem.ctx)) for problem in problems]

    async with _serving(app) as client:
        assert refusal(await client.get("/runs/nonesuch")) == (404, [("no_such_run", {})])
        run_id = (
            await client.post(
                "/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs}
            )
        ).json()["id"]
        assert refusal(await client.get(f"/runs/{run_id}", headers={"x-user": "bo"})) == (
            404,
            [("no_such_run", {})],
        )
        for answer in (True, False):  # the second is queued too, and refused once the run stops
            queued = await client.post(f"/runs/{run_id}/answers", json={waiting_id: answer})
            assert _status(queued) == (202, "running")
        assert refusal(await client.post(f"/runs/{run_id}/resume")) == (
            409,
            [("run_not_resumable", {"status": "running"})],
        )
        gate.set()
        assert (await store.ended(run_id)).status == "complete"
        late = await client.post(f"/runs/{run_id}/answers", json={waiting_id: True})
        assert refusal(late) == (
            422,
            [("not_waiting", {"waiting_id": waiting_id, "status": "complete"})],
        )
        assert refusal(await client.post(f"/runs/{run_id}/resume")) == (
            409,
            [("run_not_resumable", {"status": "complete"})],
        )
        waits = (
            await client.post(
                "/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs}
            )
        ).json()["id"]
        assert (await store.ended(waits)).status == "waiting"
        wrong = await client.post(
            f"/runs/{waits}/answers", json={waiting_id: "not a bool", "nonesuch": True}
        )
        assert wrong.status_code == 422 and len(wrong.json()["detail"]) == 2
        declined = await client.post(
            f"/runs/{waits}/declines", json={"waiting_id": waiting_id, "reason": "No"}
        )
        assert _status(declined) == (202, "declined")
        assert declined.json()["cause"]["type"] == "answer_declined"
        assert refusal(
            await client.post(f"/runs/{waits}/declines", json={"waiting_id": waiting_id})
        ) == (422, [("not_waiting", {"waiting_id": waiting_id, "status": "declined"})])
        assert refusal(await client.post(f"/runs/{waits}/resume")) == (
            409,
            [("run_not_resumable", {"status": "declined"})],
        )


@pytest.mark.parametrize("dependencies", ["read a header", "read nothing"])
def test_each_route_declares_the_refusals_it_can_answer_with_in_one_shape(
    dependencies: str,
) -> None:
    """Every 404, 409 and 422 has FastAPI's 422 body and FastAPI's words for it, whether or not the host's
    dependencies read a parameter, since FastAPI declares a 422 itself only for a parameter it reads."""
    content = {"application/json": {"schema": {"$ref": "#/components/schemas/HTTPValidationError"}}}
    no_run, not_resumable, refused = (
        {"description": description, "content": content}
        for description in ("Not Found", "Conflict", "Validation Error")
    )
    reads_a_header = dependencies == "read a header"
    app = _app(
        store=MemoryStore(),
        registry=lambda _: NodeRegistry(),
        **({} if reads_a_header else {"metadata": lambda: {}, "run_filter": _every_run}),
    )
    schema = app.openapi()
    declared = {
        (verb, path): {
            code: response
            for code, response in operation["responses"].items()
            if code in ("404", "409", "422")
        }
        for path, verbs in schema["paths"].items()
        for verb, operation in verbs.items()
    }

    on_a_run = {"404": no_run, "422": refused}
    assert declared == {
        # FastAPI's own, for the header the metadata reads
        ("get", "/nodes"): {"422": refused} if reads_a_header else {},
        ("post", "/compile"): {"422": refused},
        ("post", "/runs"): on_a_run,
        ("post", "/estimate"): on_a_run,
        ("get", "/runs/{run_id}"): on_a_run,
        ("get", "/runs/{run_id}/events"): on_a_run,
        ("post", "/runs/{run_id}/answers"): on_a_run,
        ("post", "/runs/{run_id}/declines"): on_a_run,
        ("post", "/runs/{run_id}/resume"): {**on_a_run, "409": not_resumable},
        ("post", "/runs/{run_id}/again"): on_a_run,
        ("post", "/runs/{run_id}/cancel"): on_a_run,
    }
    assert "HTTPValidationError" in schema["components"]["schemas"]


# -- a follower -----------------------------------------------------------------------------------


async def test_a_follower_gets_each_event_as_a_frame_with_its_number_and_last_event_id_goes_on_after_it() -> (
    None
):
    registry, graph, gate, inputs, _ = asking()
    gate.set()
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: registry, poll=0.01)) as client:
        run_id = (
            await client.post(
                "/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs}
            )
        ).json()["id"]
        await store.ended(run_id)
        whole = await client.get(f"/runs/{run_id}/events")
        frames = _frames(whole)
        assert frames == await store.stored_events(run_id) and (
            frames[-1]["type"],
            frames[-1]["status"],
        ) == ("run_end", "waiting")
        assert [line for line in whole.text.splitlines() if line.startswith("id: ")] == [
            f"id: {number}" for number in range(len(frames))
        ]
        rest = await client.get(
            f"/runs/{run_id}/events", headers={"Last-Event-ID": str(len(frames) - 3)}
        )
        assert _frames(rest) == frames[-2:]
        assert (
            "data: "
            not in (
                await client.get(
                    f"/runs/{run_id}/events", headers={"Last-Event-ID": str(len(frames) - 1)}
                )
            ).text
        )


# -- progress -------------------------------------------------------------------------------------


def _uploads() -> tuple[NodeRegistry, dict[str, Any], list[str]]:
    """Builds a graph whose first node, ``upload``, costs nothing and hands on the files it is given.

    Then ``review`` is called once for each file.

    Returns:
        The registry, the graph, and ``calls``, the nodes called, in order.
    """
    registry, calls = NodeRegistry(), []

    @registry.node(
        "upload", title="Upload", description="For a test.", policy=Policy(costs_nothing=True)
    )
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

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        estimated = await client.post(
            "/estimate", json={"graph": graph, "inputs": {"files": ["a", "b", "c"]}}
        )

    assert estimated.status_code == 200
    assert estimated.json()["nodes"]["review"] == {
        "finished": 0,
        "waiting": 0,
        "skipped": 0,
        "none_input": 0,
        "known": 3,
        "total": 3,
    }
    assert calls == ["upload"] and store.rows == {}


async def test_estimate_refuses_inputs_the_graph_refuses() -> None:
    registry, graph, _ = _uploads()

    async with _serving(_app(store=MemoryStore(), registry=lambda _: registry)) as client:
        refused = await client.post("/estimate", json={"graph": graph, "inputs": {}})

    assert refused.status_code == 422 and [
        problem["type"] for problem in refused.json()["detail"]
    ] == ["missing_input"]


async def test_a_graph_that_writes_one_node_id_twice_is_refused_by_every_route_that_reads_a_graph() -> (
    None
):
    registry, _, _ = _uploads()
    store = MemoryStore()
    node = '{"node": "review", "inputs": {"file": {"graph_input": "file"}}}'
    graph = f'{{"nodes": {{"review": {node}, "review": {node}}}}}'
    as_json = {"content-type": "application/json"}

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        answers = [
            await client.post("/compile", content=graph, headers=as_json),
            await client.post(
                "/runs", content=f'{{"graph": {graph}, "inputs": {{"file": "a"}}}}', headers=as_json
            ),
            await client.post(
                "/estimate",
                content=f'{{"graph": {graph}, "inputs": {{"file": "a"}}}}',
                headers=as_json,
            ),
        ]

    for answer in answers:
        assert answer.status_code == 422, answer.text
        assert [
            (problem["type"], problem["ctx"]["error"]) for problem in answer.json()["detail"]
        ] == [("invalid_record", "repeated_key")]
    assert store.rows == {}


async def test_a_run_body_that_writes_graph_twice_is_refused_and_nothing_is_stored() -> None:
    registry, good, _ = _uploads()
    store = MemoryStore()
    bad = '{"nodes": {"review": {"inputs": {}}}}'
    body = f'{{"graph": {bad}, "graph": {json.dumps(good)}, "inputs": {{"files": ["a"]}}}}'

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        answer = await client.post(
            "/runs", content=body, headers={"content-type": "application/json"}
        )

    assert answer.status_code == 422, answer.text
    assert [(problem["type"], problem["ctx"]["error"]) for problem in answer.json()["detail"]] == [
        ("invalid_record", "repeated_key")
    ]
    assert store.rows == {}


@pytest.mark.parametrize(
    ("graph", "error"),
    [
        ('{"nodes": {"review": {"inputs": {"file": {"graph_input": "file"}}}}}', "missing"),
        ('{"nodes": {"review": {"node": "review"}, "review": {"node": "review"}}}', "repeated_key"),
    ],
    ids=["a_node_with_no_node", "a_node_id_written_twice"],
)
async def test_a_run_of_a_graph_record_of_the_wrong_shape_is_refused_with_conductors_problems_never_fastapis(
    graph: str, error: str
) -> None:
    registry, _, _ = _uploads()
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        answer = await client.post(
            "/runs",
            content=f'{{"graph": {graph}, "inputs": {{}}}}',
            headers={"content-type": "application/json"},
        )

    assert answer.status_code == 422, answer.text
    detail = answer.json()["detail"]
    assert [(problem["type"], problem["ctx"]["error"]) for problem in detail] == [
        ("invalid_record", error)
    ]
    assert all(problem["loc"][0] == "record" for problem in detail)
    assert store.rows == {}


async def test_a_run_read_once_it_stopped_has_its_progress() -> None:
    registry, graph, _ = _uploads()
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        run_id = (
            await client.post("/runs", json={"graph": graph, "inputs": {"files": ["a", "b"]}})
        ).json()["id"]
        await store.ended(run_id)
        view = (await client.get(f"/runs/{run_id}")).json()

    assert (view["progress"]["finished"], view["progress"]["total"]) == (3, 3)
    assert view["progress"]["nodes"]["review"] == {
        "finished": 2,
        "waiting": 0,
        "skipped": 0,
        "none_input": 0,
        "known": 2,
        "total": 2,
    }


# -- the schema, the surface, the page ------------------------------------------------------------


def test_an_operations_id_is_its_route_functions_name_without_fastapis_path_and_verb() -> None:
    schema = _app(store=MemoryStore(), registry=lambda _: NodeRegistry()).openapi()

    assert schema["paths"]["/nodes"]["get"]["operationId"] == "list_nodes"
    assert schema["paths"]["/runs/{run_id}/answers"]["post"]["operationId"] == "answer_run"


def test_the_schema_names_each_operation_and_body_as_a_generated_client_reads_them() -> None:
    schema = _app(store=MemoryStore(), registry=lambda _: NodeRegistry()).openapi()
    operations = {
        (verb, path): operation
        for path, verbs in schema["paths"].items()
        for verb, operation in verbs.items()
    }

    assert {key: operation["summary"] for key, operation in operations.items()} == {
        ("get", "/nodes"): "List Nodes",
        ("post", "/compile"): "Compile Graph",
        ("post", "/runs"): "Start Run",
        ("post", "/estimate"): "Estimate Run",
        ("get", "/runs/{run_id}"): "Read Run",
        ("get", "/runs/{run_id}/events"): "Follow Run",
        ("post", "/runs/{run_id}/answers"): "Answer Run",
        ("post", "/runs/{run_id}/declines"): "Decline Run",
        ("post", "/runs/{run_id}/resume"): "Resume Run",
        ("post", "/runs/{run_id}/again"): "Run Again",
        ("post", "/runs/{run_id}/cancel"): "Cancel Run",
    }
    assert all(operation.get("description") for operation in operations.values())
    assert "Resume" in schema["components"]["schemas"] and not [
        name for name in schema["components"]["schemas"] if name.startswith("Body_")
    ]
    assert operations["post", "/runs/{run_id}/resume"]["requestBody"]["content"][
        "application/json"
    ]["schema"] == {"$ref": "#/components/schemas/Resume", "default": {}}
    for key in [
        ("post", "/runs/{run_id}/answers"),
        ("post", "/runs/{run_id}/again"),
    ]:
        assert operations[key]["requestBody"]["content"]["application/json"]["schema"][
            "description"
        ], key


def _refs(value: Any) -> list[str]:
    """Returns every ``$ref`` in a JSON schema, at any depth."""
    if isinstance(value, dict):
        own = [value["$ref"]] if "$ref" in value else []
        return own + [ref for item in value.values() for ref in _refs(item)]
    if isinstance(value, list):
        return [ref for item in value for ref in _refs(item)]
    return []


def test_every_ref_in_the_schema_resolves_and_the_start_body_declares_a_graph_records_shape() -> (
    None
):
    schema = _app(store=MemoryStore(), registry=lambda _: NodeRegistry()).openapi()

    def resolves(ref: str) -> bool:
        found: Any = schema
        for step in ref.removeprefix("#/").split("/"):
            if step not in found:
                return False
            found = found[step]
        return True

    assert [ref for ref in _refs(schema) if not resolves(ref)] == []
    for path in ("/runs", "/estimate"):
        body = schema["paths"][path]["post"]["requestBody"]["content"]["application/json"]["schema"]
        node = body["properties"]["graph"]["properties"]["nodes"]["additionalProperties"]
        assert set(node["properties"]) == {"node", "version", "inputs", "display"}, path
    compiled = schema["paths"]["/compile"]["post"]["requestBody"]["content"]["application/json"]
    assert "nodes" in compiled["schema"]["properties"]


async def test_a_body_whose_inputs_are_not_an_object_is_refused_as_fastapi_refuses_a_body() -> None:
    registry, graph, _ = _uploads()

    async with _serving(_app(store=MemoryStore(), registry=lambda _: registry)) as client:
        for path in ("/runs", "/estimate"):
            refused = await client.post(path, json={"graph": graph, "inputs": 3})

            assert refused.status_code == 422, path
            assert [(problem["type"], problem["loc"]) for problem in refused.json()["detail"]] == [
                ("dict_type", ["body", "inputs"])
            ]


async def test_text_that_isnt_json_is_refused_as_conductor_refuses_a_record() -> None:
    registry, _, _ = _uploads()
    as_json = {"content-type": "application/json"}

    async with _serving(_app(store=MemoryStore(), registry=lambda _: registry)) as client:
        for path in ("/compile", "/runs", "/estimate"):
            refused = await client.post(path, content='{"graph": ', headers=as_json)

            assert refused.status_code == 422, path
            assert [
                (problem["type"], problem["ctx"]["error"]) for problem in refused.json()["detail"]
            ] == [("invalid_record", "json_invalid")]


async def test_a_run_started_from_a_previous_one_reuses_its_work_and_one_the_filter_hides_is_a_404() -> (
    None
):
    registry, graph, calls = _uploads()
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        first = (
            await client.post("/runs", json={"graph": graph, "inputs": {"files": ["a", "b"]}})
        ).json()["id"]
        await store.ended(first)
        body = {"graph": graph, "inputs": {"files": ["a", "b", "c"]}, "previous": first}
        estimated = await client.post("/estimate", json=body)
        hidden = await client.post("/runs", json=body, headers={"x-user": "bo"})
        unknown = await client.post("/runs", json={**body, "previous": "nope"})
        second = (await client.post("/runs", json=body)).json()["id"]
        ended = await store.ended(second)

    # an estimate checks the previous run just as a start does, and counts a reused call as a call
    assert estimated.json()["nodes"]["review"]["total"] == 3
    # the 404 names the body's field, so a client doesn't read it as a route that isn't there
    for refused in (hidden, unknown):
        assert (
            refused.status_code,
            [(problem["type"], problem["loc"]) for problem in refused.json()["detail"]],
        ) == (404, [("no_such_run", ["previous"])])
    assert ended.outputs == {"result": ["a", "b", "c"]}
    assert calls.count("review") == 3  # two for the first run, one for the file the second added


def test_an_app_with_a_typed_router_and_a_route_of_its_own_has_one_runview_component() -> None:
    app = _app(
        store=MemoryStore(),
        registry=lambda _: NodeRegistry(),
        metadata=_member,
        run_filter=_team_runs,
        metadata_type=_Member,
    )

    @app.get("/mine/{run_id}")
    def mine(run_id: str) -> RunView:
        raise NotImplementedError

    schema = app.openapi()
    assert [name for name in schema["components"]["schemas"] if "RunView" in name] == ["RunView"]
    for path in ("/runs/{run_id}", "/mine/{run_id}"):
        answer = schema["paths"][path]["get"]["responses"]["200"]["content"]["application/json"]
        assert answer["schema"] == {"$ref": "#/components/schemas/RunView"}, path


class _Camel(BaseModel):
    """Metadata whose field has a camelCase alias."""

    user_id: str = Field(alias="userId")


async def test_metadata_with_an_alias_is_stored_so_it_reads_back_and_a_resume_gets_it() -> None:
    registry, _, graph = _doubling()
    store = MemoryStore()
    app = _app(
        store=store,
        registry=lambda _: registry,
        metadata=lambda: _Camel(userId="anna"),
        run_filter=lambda: lambda metadata: metadata.user_id == "anna",
        metadata_type=_Camel,
    )

    async with _serving(app) as client:
        started = await client.post("/runs", json={"graph": graph, "inputs": {"x": 1}})
        assert started.status_code == 202, started.text
        run_id = started.json()["id"]
        assert (await store.ended(run_id)).outputs == {"result": 2}
        assert (await client.get(f"/runs/{run_id}")).status_code == 200


async def test_metadata_not_of_the_workers_type_is_a_bug_of_the_host_that_raises_before_anything_is_stored() -> (
    None
):
    """Metadata of the wrong type is the host's bug, so it raises and nothing is stored.

    The metadata comes from the host's own dependency, not from the caller. So pydantic's error is raised as it is,
    the caller gets a 500, and no run is stored.
    """
    registry, _, graph = _doubling()
    store = MemoryStore()
    app = _app(
        store=store,
        registry=lambda _: registry,
        metadata=lambda: {"user": 7},
        run_filter=_every_run,
    )

    async with _serving(app) as client:
        for path in ("/runs", "/estimate"):
            with pytest.raises(ValidationError):
                await client.post(path, json={"graph": graph, "inputs": {"x": 1}})

    assert store.rows == {}


async def test_a_key_a_body_doesnt_know_is_refused_never_dropped() -> None:
    registry, graph, _ = _uploads()
    store = MemoryStore()

    async with _serving(_app(store=store, registry=lambda _: registry)) as client:
        typo = await client.post("/runs", json={"graph": graph, "previus": "x"})
        run_id = (
            await client.post("/runs", json={"graph": graph, "inputs": {"files": ["a"]}})
        ).json()["id"]
        await store.ended(run_id)
        resumed = await client.post(f"/runs/{run_id}/resume", json={"max_item": 5})

    for refused, loc in ((typo, ["body", "previus"]), (resumed, ["body", "max_item"])):
        assert refused.status_code == 422
        assert [(problem["type"], problem["loc"]) for problem in refused.json()["detail"]] == [
            ("extra_forbidden", loc)
        ]


async def test_a_route_reads_the_run_once_to_check_the_filter_and_answer() -> None:
    registry, graph, gate, inputs, _ = asking()
    gate.set()

    class CountsReads(MemoryStore):
        reads = 0

        async def get(self, run_id: str) -> StoredRun | None:
            self.reads += 1
            return await super().get(run_id)

    store = CountsReads()
    async with _serving(_app(store=store, registry=lambda _: registry, poll=0.01)) as client:
        run_id = (
            await client.post(
                "/runs", json={"graph": graph.model_dump(mode="json"), "inputs": inputs}
            )
        ).json()["id"]
        await store.ended(run_id)
        reads = {}
        for name, call in (
            ("read", lambda: client.get(f"/runs/{run_id}")),
            ("events", lambda: client.get(f"/runs/{run_id}/events")),
            ("hidden", lambda: client.get(f"/runs/{run_id}", headers={"x-user": "bo"})),
        ):
            store.reads = 0
            assert (await call()).status_code in (200, 404)
            reads[name] = store.reads

    assert reads == {"read": 1, "events": 1, "hidden": 1}
