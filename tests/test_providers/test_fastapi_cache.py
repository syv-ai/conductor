"""Tests for request-``cache`` forwarding through the FastAPI provider router.

A node listed in the request ``cache`` must be seeded as already-completed:
the engine emits ``node_complete`` with ``cached=True`` and skips running the
node, while its cached result still flows to downstream nodes. This lets a host
(the conductor lab) reuse outputs from a previous run instead of recomputing
the whole graph.
"""

from __future__ import annotations

import json
from typing import Annotated

import pytest

# The FastAPI provider + TestClient are optional extras; skip if unavailable.
pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from conductor import NodeRegistry  # noqa: E402
from conductor.widgets import Output  # noqa: E402
from conductor_providers.fastapi import conductor_router  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def calls() -> list[str]:
    """Records which node functions actually executed."""
    return []


@pytest.fixture
def client(calls: list[str]) -> TestClient:
    reg = NodeRegistry()

    @reg.node("seed", version=1, name="Seed", description="emits a value")
    def seed() -> Annotated[str, Output(label="Value")]:
        calls.append("seed")
        return "fresh"

    @reg.node("shout", version=1, name="Shout", description="uppercases input")
    def shout(text: str) -> Annotated[str, Output(label="Loud")]:
        calls.append("shout")
        return text.upper()

    app = FastAPI()
    app.include_router(conductor_router(reg))
    return TestClient(app)


def _graph() -> dict:
    # seed (n1) -> shout (n2): n2 uppercases whatever n1 produced.
    return {
        "nodes": [
            {"id": "n1", "type": "seed@1", "data": {}},
            {"id": "n2", "type": "shout@1", "data": {}},
        ],
        "edges": [
            {
                "id": "e1",
                "source": "n1",
                "target": "n2",
                "source_handle": "result",
                "target_handle": "text",
            },
        ],
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
    # n1 served from cache -> its function never ran; n2 ran on the cached value.
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
