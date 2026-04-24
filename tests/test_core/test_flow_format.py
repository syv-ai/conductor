"""Tests for the YAML / JSON flow format round-trip."""

from __future__ import annotations

from pathlib import Path

import pytest
from conductor import Flow, FlowDependency, FlowTrigger, GraphEdge, GraphNode
from conductor.flow_format import (
    dump_flow,
    flow_to_dict,
    flow_to_yaml,
    load_flow,
    load_flow_from_path,
    yaml_to_flow,
)


def _sample_flow() -> Flow:
    return Flow(
        id="order-fulfillment",
        version=1,
        name="Order fulfillment",
        description="Charges card, saves order, sends receipt.",
        on_error_default="compensate",
        dependencies=(
            FlowDependency(id="stripe", kind="api",
                           config={"endpoint": "https://api.stripe.com"}),
            FlowDependency(id="orders_db", kind="db"),
        ),
        triggers=(
            FlowTrigger(id="api", kind="manual", config={}),
            FlowTrigger(id="nightly", kind="schedule",
                        config={"cron": "0 9 * * *", "timezone": "UTC"}),
        ),
        nodes=[
            GraphNode("d", "decision@1", {"value": 100}),
            GraphNode("a", "echo@1", {"text": "A"},
                      compensation="compA", on_error="compensate"),
            GraphNode("compA", "undo@1", {}),
        ],
        edges=[
            GraphEdge("e1", "d", "a", "result", "text",
                      when="value > 10", priority=5),
            GraphEdge("e2", "d", "compA", "result", "_"),
        ],
    )


def test_flow_to_dict_roundtrip() -> None:
    original = _sample_flow()
    data = flow_to_dict(original)
    reloaded = load_flow(data)

    assert reloaded.id == original.id
    assert reloaded.version == original.version
    assert reloaded.name == original.name
    assert reloaded.on_error_default == "compensate"
    assert len(reloaded.nodes) == len(original.nodes)
    assert len(reloaded.edges) == len(original.edges)
    assert reloaded.dependencies[0].id == "stripe"
    assert reloaded.triggers[1].kind == "schedule"


def test_yaml_roundtrip() -> None:
    original = _sample_flow()
    yaml_text = flow_to_yaml(original)
    reloaded = yaml_to_flow(yaml_text)
    assert reloaded.id == original.id
    assert len(reloaded.edges) == 2
    # Check edge fields preserved
    e1 = next(e for e in reloaded.edges if e.id == "e1")
    assert e1.when == "value > 10"
    assert e1.priority == 5


def test_json_roundtrip(tmp_path: Path) -> None:
    original = _sample_flow()
    path = tmp_path / "flow.json"
    dump_flow(original, path)
    reloaded = load_flow_from_path(path)
    assert reloaded.name == original.name


def test_yaml_file_roundtrip(tmp_path: Path) -> None:
    original = _sample_flow()
    path = tmp_path / "flow.yaml"
    dump_flow(original, path)
    reloaded = load_flow_from_path(path)
    assert reloaded.id == original.id


def test_minimal_yaml() -> None:
    yaml_text = """
nodes:
  - id: a
    type: echo@1
    data: {text: hi}
edges: []
"""
    flow = yaml_to_flow(yaml_text)
    assert len(flow.nodes) == 1
    assert flow.nodes[0].type == "echo@1"
    assert flow.nodes[0].data == {"text": "hi"}


def test_compensation_preserved_in_yaml() -> None:
    original = _sample_flow()
    yaml_text = flow_to_yaml(original)
    reloaded = yaml_to_flow(yaml_text)
    a = next(n for n in reloaded.nodes if n.id == "a")
    assert a.compensation == "compA"
    assert a.on_error == "compensate"


def test_missing_type_raises() -> None:
    with pytest.raises(ValueError, match="missing `type`"):
        load_flow({"nodes": [{"id": "x"}], "edges": []})
