"""Request-``cache`` forwarding through the FastAPI provider router.

A node listed in the request ``cache`` is seeded as already completed: the
engine emits ``node_complete`` with ``cached=True`` and never runs the node,
while its cached result still flows to downstream nodes. A host reuses
outputs from a previous run instead of recomputing the whole graph.
"""

from __future__ import annotations

import json
from typing import Annotated

import pytest

# The FastAPI provider + TestClient are optional extras; skip if unavailable.
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
def calls() -> list[str]:
    """Records which nodes actually ran."""
    return []


@pytest.fixture
def client(calls: list[str]) -> TestClient:
    class Seed(NodeDefinition):
        id = "seed"
        title = "Seed"
        description = "emits a value"
        category = "test"

        def run(self) -> Annotated[Text, Result(title="Value")]:
            calls.append("seed")
            return Text("fresh")

    class Shout(NodeDefinition):
        id = "shout"
        title = "Shout"
        description = "uppercases input"
        category = "test"

        def run(
            self, text: Annotated[Text, Textarea(title="Text")]
        ) -> Annotated[Text, Result(title="Loud")]:
            calls.append("shout")
            return Text(text.upper())

    reg = NodeRegistry()
    reg.register(Seed)
    reg.register(Shout)

    app = FastAPI()
    app.include_router(conductor_router(reg))
    return TestClient(app)


def _graph() -> dict:
    # seed (n1) -> shout (n2): n2 uppercases whatever n1 produced.
    return {
        "flow": {
            "nodes": [
                {"id": "n1", "type": "seed", "version": 1},
                {"id": "n2", "type": "shout", "version": 1, "bindings": {"text": {"refs": ["n1.result"]}}},
            ],
        },
    }


def test_execute_without_cache_runs_every_node(client, calls):
    resp = client.post("/execute", json=_graph())
    assert resp.status_code == 200
    results = resp.json()["results"]
    # No cache -> both nodes run on fresh input.
    assert calls == ["seed", "shout"]
    assert results["n2"]["result"] == "FRESH"


def test_execute_skips_cached_node_and_feeds_downstream(client, calls):
    body = _graph()
    body["cache"] = {"n1": {"result": "cached"}}
    resp = client.post("/execute", json=body)
    assert resp.status_code == 200
    results = resp.json()["results"]
    # n1 served from cache -> it never ran; n2 ran on the cached value.
    assert "seed" not in calls
    assert "shout" in calls
    assert results["n2"]["result"] == "CACHED"


def test_execute_stream_marks_cached_node(client, calls):
    body = _graph()
    body["cache"] = {"n1": {"result": "cached"}}
    with client.stream("POST", "/execute-stream", json=body) as resp:
        assert resp.status_code == 200
        events = [
            json.loads(line[len("data: ") :])
            for line in resp.iter_lines()
            if line.startswith("data: ")
        ]

    completes = {
        e["node_id"]: e for e in events if e.get("type") == "node_complete"
    }
    # n1's completion is flagged cached; n2's is a real run.
    assert completes["n1"].get("cached") is True
    assert completes["n2"].get("cached") is not True
    assert "seed" not in calls
