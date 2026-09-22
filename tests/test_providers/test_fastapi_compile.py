"""``POST /compile`` answers with the graph's problems, anchored, as a list."""

from __future__ import annotations

from typing import Annotated

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from conductor import NodeRegistry, Param  # noqa: E402
from conductor.metadata import Result  # noqa: E402
from conductor.node import NodeDefinition  # noqa: E402
from conductor.widgets import Textarea  # noqa: E402
from conductor_nodes.types import Text  # noqa: E402
from conductor_providers.fastapi import conductor_router  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client() -> TestClient:
    class Shout(NodeDefinition):
        id = "shout"
        title = "Shout"
        description = "uppercases input"
        category = "test"

        def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())]) -> Annotated[Text, Result(title="Loud")]:
            return Text(text.upper())

    registry = NodeRegistry()
    registry.register(Shout)
    app = FastAPI()
    app.include_router(conductor_router(registry))
    return TestClient(app)


def test_a_graph_that_runs_answers_an_empty_list(client):
    resp = client.post("/compile", json={"graph": {"nodes": [
        {"id": "n1", "type": "shout", "version": 1, "bindings": {"text": {"value": "hi"}}},
    ]}})

    assert resp.status_code == 200
    assert resp.json() == []


def test_every_problem_comes_back_anchored(client):
    resp = client.post("/compile", json={"graph": {"nodes": [
        {"id": "n1", "type": "gone", "version": 1},
        {"id": "n2", "type": "shout", "version": 1, "bindings": {"text": {"refs": ["ghost.result"]}}},
    ]}})

    assert resp.status_code == 200
    assert [(p["code"], p["fatal"], p["node_id"], p["field"]) for p in resp.json()] == [
        ("unknown_node_type", True, "n1", None),
        ("unknown_ref_node", True, "n2", "text"),
    ]
    assert resp.json()[1]["details"] == {"source_node": "ghost"}


# --- a bad request is a 4xx, not a 500 --------------------------------------------------


def test_executing_a_graph_that_cannot_run_is_a_422_with_its_problems(client):
    for route in ("/execute", "/execute-stream"):
        resp = client.post(route, json={"graph": {"nodes": [{"id": "n1", "type": "gone", "version": 1}]}})

        assert resp.status_code == 422, route
        assert [p["code"] for p in resp.json()["detail"]["problems"]] == ["unknown_node_type"]


def test_a_cache_the_run_cannot_take_is_a_422_naming_why(client):
    graph = {"nodes": [{"id": "n1", "type": "shout", "version": 1, "bindings": {"text": {"value": "hi"}}}]}
    resp = client.post("/execute", json={"graph": graph, "cache": {"n1": {"nope": "x"}}})

    assert resp.status_code == 422
    assert resp.json()["detail"] == "'n1' has no output 'nope'"


def test_a_cache_for_a_node_the_graph_does_not_have_is_a_422(client):
    graph = {"nodes": [{"id": "n1", "type": "shout", "version": 1, "bindings": {"text": {"value": "hi"}}}]}
    resp = client.post("/execute", json={"graph": graph, "cache": {"ghost": {"result": "x"}}})

    assert resp.status_code == 422
    assert "'ghost' is not a node of this graph" in resp.json()["detail"]


def test_a_record_naming_a_node_the_graph_does_not_have_leaves_it_out(client):
    graph = {"nodes": [{"id": "n1", "type": "shout", "version": 1, "bindings": {"text": {"value": "hi"}}}]}
    record = {"cells": [{"ref": ["ghost", "result"], "row": None, "value": "x"}], "done_units": [["ghost", None]]}
    resp = client.post("/execute", json={"graph": graph, "record": record})

    assert resp.status_code == 200
    assert resp.json()["type"] == "graph_complete"


def test_an_error_that_is_not_a_refusal_stays_the_servers(client, monkeypatch):
    """Only a refused graph or a refused cache or record is the caller's
    fault; a ``ValueError`` from anywhere else is a bug and a 500."""
    import conductor.execution.leg as leg

    def broken(*args, **kwargs):
        raise ValueError("a bug")

    monkeypatch.setattr(leg.Leg, "__init__", broken)
    graph = {"nodes": [{"id": "n1", "type": "shout", "version": 1, "bindings": {"text": {"value": "hi"}}}]}
    resp = TestClient(client.app, raise_server_exceptions=False).post("/execute", json={"graph": graph})

    assert resp.status_code == 500


def test_the_entities_route_exists_only_with_a_resolver(client):
    assert client.get("/entities/document").status_code == 404


def test_the_dead_provider_bits_are_gone():
    import conductor_providers
    import conductor_providers.react as react

    assert not hasattr(conductor_providers, "PROVIDERS")
    assert not hasattr(react, "palette_from_registry")
