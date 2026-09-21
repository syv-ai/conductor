"""What a run supplies, through the FastAPI provider router.

``conductor_router(from_run=...)`` is called per request and returns what
``execute(from_run=...)`` takes, keyed by type. A run whose nodes need a
type the hook did not supply is refused before anything runs, and both
endpoints refuse it the same way: the request fails.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Annotated

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from conductor import NodeRegistry  # noqa: E402
from conductor.interface import FromRun  # noqa: E402
from conductor.metadata import Result  # noqa: E402
from conductor.node import NodeDefinition  # noqa: E402
from conductor_nodes.types import Text  # noqa: E402
from conductor_providers.fastapi import conductor_router  # noqa: E402
from fastapi import FastAPI, Request  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


@dataclass(frozen=True)
class Caller:
    name: str


class Greet(NodeDefinition):
    id = "greet"
    title = "Greet"
    description = "greets whoever made the request"
    category = "test"

    def run(self, caller: Annotated[Caller, FromRun()]) -> Annotated[Text, Result(title="Greeting")]:
        return Text(f"hi {caller.name}")


GRAPH = {"graph": {"nodes": [{"id": "g", "type": "greet", "version": 1}]}}


def _client(**router_kwargs) -> TestClient:
    registry = NodeRegistry()
    registry.register(Greet)
    app = FastAPI()
    app.include_router(conductor_router(registry, **router_kwargs))
    return TestClient(app, raise_server_exceptions=False)


def _from_header(request: Request) -> dict[type, object]:
    return {Caller: Caller(request.headers["x-caller"])}


def test_the_hook_supplies_each_request_on_both_endpoints():
    client = _client(from_run=_from_header)

    ran = client.post("/execute", json=GRAPH, headers={"x-caller": "Ida"})
    with client.stream("POST", "/execute-stream", json=GRAPH, headers={"x-caller": "Bo"}) as streamed:
        events = [json.loads(line[len("data: "):]) for line in streamed.iter_lines() if line.startswith("data: ")]

    assert ran.json()["results"]["g"]["result"] == "hi Ida"
    assert events[-1]["type"] == "graph_complete"
    assert events[-1]["results"]["g"]["result"] == "hi Bo"


def test_a_run_missing_what_it_needs_fails_the_request_on_both_endpoints():
    client = _client()

    assert client.post("/execute", json=GRAPH).status_code == 500
    assert client.post("/execute-stream", json=GRAPH).status_code == 500
