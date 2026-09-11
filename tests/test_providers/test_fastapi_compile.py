"""``POST /compile`` answers with the graph's problems, anchored, as a list."""

from __future__ import annotations

from typing import Annotated

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from conductor import NodeRegistry  # noqa: E402
from conductor.node import NodeDefinition  # noqa: E402
from conductor.returns import Result  # noqa: E402
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

        def run(self, text: Annotated[Text, Textarea(title="Text")]) -> Annotated[Text, Result(title="Loud")]:
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
