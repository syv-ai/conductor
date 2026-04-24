"""Tests for actor metadata, top-level dependencies, and triggers."""

from __future__ import annotations

from typing import Annotated

import pytest
from conductor import (
    Actor,
    CompilationError,
    Flow,
    FlowDependency,
    FlowTrigger,
    GraphNode,
    NodeRegistry,
    compile,
)
from conductor.widgets import Output

# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------


def test_actor_coerce_from_string() -> None:
    a = Actor.coerce("human")
    assert a.kind == "human"
    assert a.role is None


def test_actor_coerce_from_dict() -> None:
    a = Actor.coerce({"kind": "human", "role": "finance_manager"})
    assert a.kind == "human"
    assert a.role == "finance_manager"


def test_actor_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError, match="Invalid actor kind"):
        Actor.coerce("robot")


def test_actor_on_node_definition() -> None:
    reg = NodeRegistry()

    @reg.node(
        "approve", version=1, name="Approve", description="x",
        actor={"kind": "human", "role": "manager"},
    )
    def approve() -> Annotated[str, Output(label="r")]:
        return "approved"

    node_def = reg.get("approve@1")
    assert node_def.actor is not None
    assert node_def.actor.kind == "human"
    assert node_def.actor.role == "manager"


def test_actor_as_bare_string() -> None:
    reg = NodeRegistry()

    @reg.node("svc", version=1, name="Svc", description="x", actor="system")
    def svc() -> Annotated[str, Output(label="r")]:
        return "ok"

    assert reg.get("svc@1").actor.kind == "system"


# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------


def test_uses_must_be_declared() -> None:
    reg = NodeRegistry()

    @reg.node("hit", version=1, name="Hit", description="x", uses=["stripe"])
    def hit() -> Annotated[str, Output(label="r")]:
        return "ok"

    flow = Flow(
        nodes=[GraphNode("n1", "hit@1", {})], edges=[],
        dependencies=(),
    )
    with pytest.raises(CompilationError, match="dependency 'stripe'"):
        compile(flow=flow, registry=reg)


def test_uses_validated_against_declared() -> None:
    reg = NodeRegistry()

    @reg.node("hit", version=1, name="Hit", description="x", uses=["stripe"])
    def hit() -> Annotated[str, Output(label="r")]:
        return "ok"

    flow = Flow(
        nodes=[GraphNode("n1", "hit@1", {})], edges=[],
        dependencies=(FlowDependency(id="stripe", kind="api"),),
    )
    # Should not raise
    compiled = compile(flow=flow, registry=reg)
    assert compiled.flow is flow


def test_deps_surface_on_compiled_flow() -> None:
    reg = NodeRegistry()

    @reg.node("x", version=1, name="X", description="x")
    def x() -> Annotated[str, Output(label="r")]:
        return "ok"

    flow = Flow(
        nodes=[GraphNode("n1", "x@1", {})], edges=[],
        dependencies=(
            FlowDependency(id="db", kind="db", config={"url": "postgres://x"}),
            FlowDependency(id="stripe", kind="api"),
        ),
    )
    compiled = compile(flow=flow, registry=reg)
    assert compiled.flow.dependencies[0].id == "db"
    assert compiled.flow.dependencies[1].kind == "api"


# ---------------------------------------------------------------------------
# Triggers
# ---------------------------------------------------------------------------


def test_triggers_are_metadata_only() -> None:
    reg = NodeRegistry()

    @reg.node("x", version=1, name="X", description="x")
    def x() -> Annotated[str, Output(label="r")]:
        return "ok"

    flow = Flow(
        nodes=[GraphNode("n1", "x@1", {})], edges=[],
        triggers=(
            FlowTrigger(id="nightly", kind="schedule",
                        config={"cron": "0 9 * * 1", "timezone": "UTC"}),
            FlowTrigger(id="webhook", kind="webhook",
                        config={"path": "/hooks/stripe"},
                        input_map="$.body"),
        ),
    )
    compiled = compile(flow=flow, registry=reg)
    assert compiled.flow.triggers[0].kind == "schedule"
    assert compiled.flow.triggers[1].input_map == "$.body"
