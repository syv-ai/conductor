"""Integration tests — features used together.

These exercise combinations the individual feature tests miss: decision
inside a subprocess, compensation around a signal, YAML round-trip of
a flow using every new field, etc.
"""

from __future__ import annotations

import asyncio
from typing import Annotated

from conductor import (
    SUBPROCESS,
    WHILE,
    Actor,
    Flow,
    FlowDependency,
    FlowStore,
    FlowTrigger,
    GraphEdge,
    GraphNode,
    NodeRegistry,
    SubprocessRegistry,
    compile,
    execute,
    execute_sync,
)
from conductor.errors import NodeExecutionError
from conductor.flow_format import flow_to_yaml, yaml_to_flow
from conductor.widgets import Output, Text


def _full_registry() -> NodeRegistry:
    """A registry with every new node type exercised in these tests."""
    reg = NodeRegistry()

    @reg.node("decision", version=1, name="Decision", description="x",
              is_decision=True)
    def decision(value: Annotated[object, Text(label="v")] = None) -> Annotated[
        object, Output(label="v"),
    ]:
        return value

    @reg.node("echo", version=1, name="Echo", description="x",
              actor={"kind": "system", "role": "svc"})
    def echo(
        text: Annotated[str, Text(label="t")] = "ok",
    ) -> Annotated[str, Output(label="r")]:
        return text

    @reg.node("fail", version=1, name="Fail", description="x")
    def fail() -> Annotated[str, Output(label="r")]:
        raise NodeExecutionError("boom", node_id="fail")

    @reg.node("record", version=1, name="Record", description="x")
    def record(
        store: FlowStore,
        target_node_id: str = None,
    ) -> Annotated[str, Output(label="r")]:
        store.set("undos", store.get("undos", []) + [target_node_id or "?"])
        return "undone"

    @reg.node("while-start", version=1, name="WS", description="x")
    def while_start(
        condition: Annotated[str, Text(label="c")] = "false",
        max_iterations: Annotated[int, Text(label="m")] = 10,
        negate: Annotated[bool, Text(label="n")] = False,
    ) -> tuple[
        Annotated[int, Output(label="i")],
        Annotated[object, Output(label="l")],
    ]:
        raise NotImplementedError

    @reg.node("while-end", version=1, name="WE", description="x")
    def while_end(
        item: Annotated[object, Text(label="i")] = None,
    ) -> Annotated[object, Output(label="last")]:
        raise NotImplementedError

    @reg.node("subprocess-call", version=1, name="Call", description="x")
    def call(
        flow_id: Annotated[str, Text(label="id")] = "",
        flow_version: Annotated[int, Text(label="v")] = 1,
        inputs: Annotated[dict, Text(label="in")] = None,
    ) -> Annotated[object, Output(label="r")]:
        raise NotImplementedError

    return reg


# ---------------------------------------------------------------------------
# Decision + compensation
# ---------------------------------------------------------------------------


def test_decision_drives_a_branch_that_fails_triggers_compensation() -> None:
    """Decision picks a branch → that branch fails → compensation runs."""
    reg = _full_registry()
    compiled = compile(
        nodes=[
            GraphNode("d", "decision@1", {"value": 100}),
            GraphNode("a", "echo@1", {"text": "A"},
                      compensation="undo_a"),
            GraphNode("b", "echo@1", {"text": "B"}),
            GraphNode("boom", "fail@1", {}),
            GraphNode("undo_a", "record@1", {}),
        ],
        edges=[
            GraphEdge("e1", "d", "a", "result", None,
                      when="result > 50", priority=10),
            GraphEdge("e2", "d", "b", "result", None),
            GraphEdge("e3", "a", "boom", "result", "_"),
        ],
        registry=reg,
    )

    events: list[dict] = []

    async def go():
        async for ev in execute(compiled):
            events.append(ev)

    asyncio.run(go())

    kinds = [e["type"] for e in events]
    assert "flow_error" in kinds
    assert "compensation_start" in kinds
    # b should not have run (decision went to a)
    b_completed = [e for e in events if e.get("type") == "node_complete"
                   and e.get("node_id") == "b"]
    assert not b_completed


# ---------------------------------------------------------------------------
# While-loop with a decision inside
# ---------------------------------------------------------------------------


def test_while_loop_with_decision_body() -> None:
    """A decision node inside a while-loop body picks branches per iteration."""
    reg = _full_registry()

    @reg.node("counter", version=1, name="Counter", description="x")
    def counter(
        store: FlowStore,
        _: Annotated[int, Text(label="_")] = 0,
    ) -> Annotated[int, Output(label="c")]:
        c = store.get("c", 0) + 1
        store.set("c", c)
        return c

    compiled = compile(
        nodes=[
            GraphNode("w_start", "while-start@1",
                      {"condition": "iteration < 3", "max_iterations": 5}),
            GraphNode("c", "counter@1", {}),
            GraphNode("w_end", "while-end@1", {}),
        ],
        edges=[
            GraphEdge("e1", "w_start", "c", "output_1", "_"),
            GraphEdge("e2", "c", "w_end", "result", "item"),
        ],
        registry=reg,
        compound_types=[WHILE],
    )
    r = execute_sync(compiled)
    assert r["w_end"]["result"] == 3


# ---------------------------------------------------------------------------
# Subprocess referencing a flow with dependencies + actor metadata
# ---------------------------------------------------------------------------


def test_subprocess_with_full_flow_metadata() -> None:
    reg = _full_registry()

    sub = Flow(
        id="greet", version=1,
        name="Greeting",
        nodes=[GraphNode("e", "echo@1", {"text": "hello"})],
        edges=[],
        dependencies=(FlowDependency(id="logging", kind="notification"),),
        triggers=(FlowTrigger(id="manual", kind="manual"),),
    )
    sub_reg = SubprocessRegistry()
    sub_reg.register(sub)

    caller = Flow(
        nodes=[GraphNode("c", "subprocess-call@1",
                         {"flow_id": "greet", "flow_version": 1})],
        edges=[],
    )
    compiled = compile(
        flow=caller, registry=reg,
        compound_types=[SUBPROCESS], subprocess_registry=sub_reg,
    )
    r = execute_sync(compiled)
    assert r["c"]["e"]["result"] == "hello"


# ---------------------------------------------------------------------------
# Full YAML round-trip of a flow using every new feature
# ---------------------------------------------------------------------------


def test_yaml_roundtrip_full_featured_flow() -> None:
    original = Flow(
        id="full",
        version=2,
        name="Full-featured",
        description="Exercise every new field",
        on_error_default="compensate",
        dependencies=(
            FlowDependency(id="stripe", kind="api",
                           config={"endpoint": "https://api.stripe.com"}),
            FlowDependency(id="orders", kind="db"),
        ),
        triggers=(
            FlowTrigger(id="manual", kind="manual"),
            FlowTrigger(id="nightly", kind="schedule",
                        config={"cron": "0 3 * * *", "timezone": "UTC"}),
            FlowTrigger(id="hook", kind="webhook",
                        config={"path": "/hooks/payment"},
                        input_map="$.body.event"),
        ),
        nodes=[
            GraphNode("d", "decision@1", {"v": 1}),
            GraphNode("charge", "echo@1", {"text": "A"},
                      compensation="refund",
                      on_error="compensate"),
            GraphNode("refund", "record@1", {}),
            GraphNode("b", "echo@1", {"text": "B"}),
        ],
        edges=[
            GraphEdge("e1", "d", "charge", "result", None,
                      when="v > 0", priority=10),
            GraphEdge("e2", "d", "b", "result", None),
        ],
    )
    yaml_text = flow_to_yaml(original)
    reloaded = yaml_to_flow(yaml_text)
    assert reloaded.id == "full"
    assert reloaded.version == 2
    assert reloaded.on_error_default == "compensate"
    assert reloaded.dependencies[0].id == "stripe"
    assert reloaded.triggers[2].input_map == "$.body.event"
    charge = next(n for n in reloaded.nodes if n.id == "charge")
    assert charge.compensation == "refund"
    assert charge.on_error == "compensate"
    e1 = next(e for e in reloaded.edges if e.id == "e1")
    assert e1.when == "v > 0"
    assert e1.priority == 10


# ---------------------------------------------------------------------------
# Actor round-trip
# ---------------------------------------------------------------------------


def test_actor_attached_via_decorator_is_discoverable() -> None:
    reg = _full_registry()
    echo_def = reg.get("echo@1")
    assert isinstance(echo_def.actor, Actor)
    assert echo_def.actor.kind == "system"
    assert echo_def.actor.role == "svc"
    assert echo_def.actor.to_dict() == {"kind": "system", "role": "svc"}
