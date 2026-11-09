"""A run in legs, through the FastAPI provider router.

A node that returns ``AskPerson`` ends the leg waiting. ``/run`` answers
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

from conductor import AskPerson, NodeRegistry, Param, Result, Series, node  # noqa: E402
from conductor.widgets import Switch, Textarea  # noqa: E402
from conductor_providers.fastapi import conductor_router  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def calls() -> list[str]:
    return []


@pytest.fixture
def client(calls: list[str]) -> TestClient:
    @node("words", title="Words", description="splits a text into words", category="test")
    def words(text: Annotated[str, Param(title="Text", widget=Textarea())]) -> Annotated[Series[str], Result(title="Words")]:
        calls.append("words")
        return [word for word in text.split()]

    @node("approve", title="Approve", description="asks a person", category="test")
    def approve(text: Annotated[str, Param(title="Text", widget=Textarea())]) -> Annotated[bool, Result(title="Approved")] | AskPerson:
        calls.append("approve")
        return AskPerson(widgets={"result": Switch()})

    registry = NodeRegistry([words, approve])
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
    with client.stream("POST", "/run-stream", json=body) as resp:
        assert resp.status_code == 200
        return [json.loads(line[len("data: "):]) for line in resp.iter_lines() if line.startswith("data: ")]


def test_a_completed_leg_answers_with_its_state_and_results(client):
    ended = client.post("/run", json={"graph": {"nodes": PER_ROW["nodes"][:1]}}).json()

    assert ended["state"]["status"] == "complete"
    assert ended["results"]["w"]["result"]["values"] == ["red", "green"]


def test_a_leg_that_asks_answers_waiting_and_the_next_leg_goes_on(client, calls):
    pending = client.post("/run", json={"graph": ONCE})

    assert pending.status_code == 200
    body = pending.json()
    assert body["state"]["status"] == "waiting"
    [unit] = body["state"]["waiting"]
    assert unit["id"] == "ok" and unit["row"] == []
    assert unit["questions"][0]["name"] == "result"

    done = client.post("/run", json={"graph": ONCE, "state": body["state"], "answers": {"ok": {"result": True}}}).json()

    assert done["state"]["status"] == "complete"
    assert done["results"]["ok"]["result"] == 1  # a bool, written as its own type writes it
    assert calls == ["approve"]


def test_each_row_of_a_per_row_node_is_answered_by_its_own_id(client, calls):
    first = client.post("/run", json={"graph": PER_ROW}).json()
    assert [unit["id"] for unit in first["state"]["waiting"]] == ["ok@0", "ok@1"]

    second = client.post("/run", json={"graph": PER_ROW, "state": first["state"], "answers": {"ok@1": {"result": False}}}).json()

    assert second["state"]["status"] == "waiting"
    assert [unit["row"] for unit in second["state"]["waiting"]] == [[0]]

    third = client.post("/run", json={"graph": PER_ROW, "state": second["state"], "answers": {"ok@0": {"result": True}}}).json()

    assert third["state"]["status"] == "complete"
    assert third["results"]["ok"]["result"]["values"] == [True, False]
    assert calls.count("words") == 1


def test_the_stream_takes_the_state_and_the_answers_too(client, calls):
    first = _stream(client, {"graph": PER_ROW})
    assert first[-1]["type"] == "run_end" and first[-1]["state"]["status"] == "waiting"

    answers = {"ok@0": {"result": True}, "ok@1": {"result": True}}
    second = _stream(client, {"graph": PER_ROW, "state": first[-1]["state"], "answers": answers})

    assert second[-1]["state"]["status"] == "complete"
    assert second[-1]["results"]["ok"]["result"]["values"] == [True, True]
    assert calls.count("words") == 1


def test_a_wrong_answer_is_a_422_listing_every_fault(client):
    first = client.post("/run", json={"graph": ONCE}).json()

    refused = client.post("/run", json={"graph": ONCE, "state": first["state"], "answers": {"ghost": {}}})
    assert refused.status_code == 422
    assert [(d["type"], d["loc"]) for d in refused.json()["detail"]] == [("not_waiting", ["answers", "ghost"])]


def test_answers_with_no_state_are_refused(client):
    refused = client.post("/run", json={"graph": ONCE, "answers": {"ok": {"result": True}}})

    assert refused.status_code == 422 and refused.json()["detail"][0]["type"] == "not_waiting"
