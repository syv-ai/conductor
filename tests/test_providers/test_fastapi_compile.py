"""``POST /compile`` takes a graph and answers ``compiled.describe()``; ``GET /nodes`` answers ``registry.describe()``."""

from __future__ import annotations

from typing import Annotated

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from conductor import NodeRegistry, Param, Result, node  # noqa: E402
from conductor.widgets import Textarea  # noqa: E402
from conductor_providers.fastapi import conductor_router  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    @node("shout", title="Shout", description="uppercases input", category="test")
    def shout(text: Annotated[str, Param(title="Text", widget=Textarea())]) -> Annotated[str, Result(title="Loud")]:
        return text.upper()

    registry = NodeRegistry([shout])
    app = FastAPI()
    app.include_router(conductor_router(registry))
    return TestClient(app)


def test_a_graph_that_runs_answers_an_empty_list(client):
    resp = client.post("/compile", json={"nodes": [
        {"id": "n1", "type": "shout", "version": 1, "bindings": {"text": {"value": "hi"}}},
    ]})

    assert resp.status_code == 200
    assert resp.json()["is_runnable"] is True
    assert resp.json()["problems"] == []
    assert resp.json()["run_plan"]["nodes"]["n1"]["runs_per_row_of"] is None


def test_every_problem_comes_back_anchored(client):
    resp = client.post("/compile", json={"nodes": [
        {"id": "n1", "type": "gone", "version": 1},
        {"id": "n2", "type": "shout", "version": 1, "bindings": {"text": {"refs": ["ghost.result"]}}},
    ]})

    assert resp.status_code == 200
    assert [(p["type"], p["fatal"], *p["loc"]) for p in resp.json()["problems"]] == [
        ("unknown_node_type", True, "n1"),
        ("unknown_ref_node", True, "n2", "text"),
    ]
    assert resp.json()["problems"][1]["ctx"] == {"source_node": "ghost"}


# --- a bad request is a 4xx, not a 500 --------------------------------------------------


def test_executing_a_graph_that_cannot_run_is_a_422_with_its_problems(client):
    for route in ("/run", "/run-stream"):
        resp = client.post(route, json={"graph": {"nodes": [{"id": "n1", "type": "gone", "version": 1}]}})

        assert resp.status_code == 422, route
        assert [d["type"] for d in resp.json()["detail"]] == ["unknown_node_type"]


def test_a_state_naming_a_node_the_graph_does_not_have_leaves_it_out(client):
    graph = {"nodes": [{"id": "n1", "type": "shout", "version": 1, "bindings": {"text": {"value": "hi"}}}]}
    state = {"status": "complete", "output_columns": {"ghost.result": {"rows": [[]], "values": ["x"]}}, "finished_rows": {"ghost": [[]]}}
    resp = client.post("/run", json={"graph": graph, "state": state})

    assert resp.status_code == 200
    assert resp.json()["state"]["status"] == "complete"


def test_a_malformed_state_is_the_callers_fault(client):
    graph = {"nodes": [{"id": "n1", "type": "shout", "version": 1, "bindings": {"text": {"value": "hi"}}}]}
    state = {"status": "complete", "output_columns": {"bad": {"rows": [[]], "values": ["x"]}}}
    resp = TestClient(client.app, raise_server_exceptions=False).post("/run", json={"graph": graph, "state": state})

    assert resp.status_code == 422


def test_an_error_that_is_not_a_refusal_stays_the_servers(client, monkeypatch):
    """Only a refused graph, answer or state is the caller's
    fault; a ``ValueError`` from anywhere else is a bug and a 500."""
    import conductor._internal.execution.run_until_idle as leg

    def broken(*args, **kwargs):
        raise ValueError("a bug")

    monkeypatch.setattr(leg.RunningLeg, "__init__", broken)
    graph = {"nodes": [{"id": "n1", "type": "shout", "version": 1, "bindings": {"text": {"value": "hi"}}}]}
    resp = TestClient(client.app, raise_server_exceptions=False).post("/run", json={"graph": graph})

    assert resp.status_code == 500


def test_a_described_graph_comes_back_with_its_nodes_and_fields_and_the_schema_is_published(client):
    resp = client.post("/compile", json={"nodes": [
        {"id": "n1", "type": "shout", "version": 1, "bindings": {"text": {"value": "hi"}}},
    ]})

    assert resp.json()["nodes"]["n1"]["state"] == "ready"
    assert resp.json()["fields"]["n1.result"] == {"dtype": {"id": "str"}, "many": False}
    assert client.get("/openapi.json").status_code == 200


def test_the_palette_has_the_registrys_types(client):
    described = client.get("/nodes").json()

    assert [entry["id"] for entry in described["nodes"]] == ["shout"]
    assert "str" in {entry["id"] for entry in described["types"]}
