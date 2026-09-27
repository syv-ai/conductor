"""A run in legs, through the FastAPI provider router.

A node that returns ``Asks`` ends the leg pending. ``/execute`` answers
with that ending — the questions, what ran, and the state — the same frame
``/execute-stream`` ends on. The next request sends the state back with the
answers in ``cache`` and the run goes on from where it stopped: nothing that
ran runs again. For a node that runs per row, an answer is a series in the
form one is dumped in, ``{"rows": [...], "values": [...]}``, and names only
the rows it answers.
"""

from __future__ import annotations

import json
from typing import Annotated

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from conductor import Asks, NodeRegistry  # noqa: E402
from conductor.metadata import Input, Param, Result  # noqa: E402
from conductor.node import NodeDefinition  # noqa: E402
from conductor.series import Series  # noqa: E402
from conductor.widgets import Switch, Textarea  # noqa: E402
from conductor_nodes.types import Flag, Text  # noqa: E402
from conductor_providers.fastapi import conductor_router  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def calls() -> list[str]:
    return []


@pytest.fixture
def client(calls: list[str]) -> TestClient:
    class Words(NodeDefinition):
        id = "words"
        title = "Words"
        description = "splits a text into words"
        category = "test"

        def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())]) -> Annotated[Series[Text], Result(title="Words")]:
            calls.append("words")
            return [Text(word) for word in text.split()]

    class Approve(NodeDefinition):
        id = "approve"
        title = "Approve"
        description = "asks a person"
        category = "test"

        def run(self, text: Annotated[Text, Param(title="Text", widget=Textarea())]) -> Annotated[Flag, Result(title="Approved")] | Asks:
            calls.append("approve")
            return Asks(questions=(Input(name="result", dtype=Flag, title="Approved", widget=Switch()),))

    registry = NodeRegistry()
    registry.register(Words)
    registry.register(Approve)
    app = FastAPI()
    app.include_router(conductor_router(registry))
    return TestClient(app)


ONCE = {"nodes": [{"id": "ok", "type": "approve", "version": 1, "bindings": {"text": {"value": "ship it"}}}]}

PER_ROW = {
    "nodes": [
        {"id": "w", "type": "words", "version": 1, "bindings": {"text": {"value": "red green"}}},
        {"id": "ok", "type": "approve", "version": 1, "bindings": {"text": {"refs": ["w.result"]}}},
    ],
}


def _stream(client: TestClient, body: dict) -> list[dict]:
    with client.stream("POST", "/execute-stream", json=body) as resp:
        assert resp.status_code == 200
        return [json.loads(line[len("data: "):]) for line in resp.iter_lines() if line.startswith("data: ")]


def test_a_completed_leg_answers_with_its_ending(client):
    ending = client.post("/execute", json={"graph": {"nodes": PER_ROW["nodes"][:1]}}).json()

    assert ending["type"] == "graph_complete"
    assert ending["results"]["w"]["result"]["values"] == ["red", "green"]
    assert "state" in ending


def test_a_leg_that_asks_answers_pending_and_the_next_leg_goes_on(client, calls):
    pending = client.post("/execute", json={"graph": ONCE})

    assert pending.status_code == 200
    body = pending.json()
    assert body["type"] == "graph_pending"
    [unit] = body["pending"]
    assert unit["node_id"] == "ok" and unit["row"] is None
    assert unit["questions"][0]["name"] == "ok.result"

    done = client.post("/execute", json={"graph": ONCE, "state": body["state"], "cache": {"ok": {"result": True}}}).json()

    assert done["type"] == "graph_complete"
    assert done["results"]["ok"]["result"] == 1  # a Flag, written as its own type writes it
    assert calls == ["approve"]


def test_an_answer_for_a_node_that_runs_per_row_names_the_rows_it_answers(client, calls):
    first = client.post("/execute", json={"graph": PER_ROW}).json()
    assert [unit["row"] for unit in first["pending"]] == [[0], [1]]

    second = client.post(
        "/execute",
        json={"graph": PER_ROW, "state": first["state"], "cache": {"ok": {"result": {"rows": [[1]], "values": [False]}}}},
    ).json()

    assert second["type"] == "graph_pending"
    assert [unit["row"] for unit in second["pending"]] == [[0]]

    third = client.post(
        "/execute",
        json={"graph": PER_ROW, "state": second["state"], "cache": {"ok": {"result": {"rows": [[0]], "values": [True]}}}},
    ).json()

    assert third["type"] == "graph_complete"
    assert third["results"]["ok"]["result"]["values"] == [True, False]
    assert calls.count("words") == 1


def test_the_stream_takes_the_state_too(client, calls):
    first = _stream(client, {"graph": PER_ROW})
    assert first[-1]["type"] == "graph_pending"

    answer = {"ok": {"result": {"rows": [[0], [1]], "values": [True, True]}}}
    second = _stream(client, {"graph": PER_ROW, "state": first[-1]["state"], "cache": answer})

    assert second[-1]["type"] == "graph_complete"
    assert second[-1]["results"]["ok"]["result"]["values"] == [True, True]
    assert calls.count("words") == 1
