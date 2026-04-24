"""Tests for the subprocess compound."""

from __future__ import annotations

from typing import Annotated

import pytest
from conductor import (
    SUBPROCESS,
    Flow,
    GraphNode,
    NodeRegistry,
    SubprocessRegistry,
    compile,
    execute_sync,
)
from conductor.widgets import Output, Text


def _reg() -> NodeRegistry:
    reg = NodeRegistry()

    @reg.node("subprocess-call", version=1, name="Call", description="x")
    def call(
        flow_id: Annotated[str, Text(label="id")] = "",
        flow_version: Annotated[int, Text(label="ver")] = 1,
        inputs: Annotated[dict, Text(label="in")] = None,
    ) -> Annotated[object, Output(label="r")]:
        raise NotImplementedError

    @reg.node("echo", version=1, name="Echo", description="x")
    def echo(
        text: Annotated[str, Text(label="t")] = "ok",
    ) -> Annotated[str, Output(label="r")]:
        return text.upper()

    return reg


def test_subprocess_runs_sub_flow() -> None:
    reg = _reg()
    sub = Flow(
        id="hello", version=1,
        nodes=[GraphNode("x", "echo@1", {"text": "world"})],
        edges=[],
    )
    sub_reg = SubprocessRegistry()
    sub_reg.register(sub)

    caller = Flow(
        nodes=[GraphNode("c", "subprocess-call@1",
                         {"flow_id": "hello", "flow_version": 1})],
        edges=[],
    )
    compiled = compile(
        flow=caller, registry=reg,
        compound_types=[SUBPROCESS], subprocess_registry=sub_reg,
    )
    r = execute_sync(compiled)
    # call node's result contains the sub-flow's results keyed by node id
    assert r["c"]["x"]["result"] == "WORLD"


def test_missing_sub_flow_raises() -> None:
    reg = _reg()
    sub_reg = SubprocessRegistry()
    caller = Flow(
        nodes=[GraphNode("c", "subprocess-call@1",
                         {"flow_id": "missing", "flow_version": 1})],
        edges=[],
    )
    compiled = compile(
        flow=caller, registry=reg,
        compound_types=[SUBPROCESS], subprocess_registry=sub_reg,
    )
    with pytest.raises(Exception, match="not found in subprocess registry"):
        execute_sync(compiled)


def test_subprocess_registry_needs_id() -> None:
    sub = Flow(id=None, nodes=[], edges=[])
    sub_reg = SubprocessRegistry()
    with pytest.raises(ValueError, match="must have an `id`"):
        sub_reg.register(sub)


def test_subprocess_nested_call_chain() -> None:
    """A subprocess can call another subprocess."""
    reg = _reg()
    inner = Flow(
        id="inner", version=1,
        nodes=[GraphNode("i", "echo@1", {"text": "inner"})],
        edges=[],
    )
    middle = Flow(
        id="middle", version=1,
        nodes=[
            GraphNode("m", "subprocess-call@1",
                      {"flow_id": "inner", "flow_version": 1}),
        ],
        edges=[],
    )
    sub_reg = SubprocessRegistry()
    sub_reg.register(inner)
    sub_reg.register(middle)

    caller = Flow(
        nodes=[GraphNode("top", "subprocess-call@1",
                         {"flow_id": "middle", "flow_version": 1})],
        edges=[],
    )
    compiled = compile(
        flow=caller, registry=reg,
        compound_types=[SUBPROCESS], subprocess_registry=sub_reg,
    )
    r = execute_sync(compiled)
    # top -> middle -> inner. The innermost "inner" echo returns "INNER".
    assert r["top"]["m"]["i"]["result"] == "INNER"


def test_subprocess_infinite_recursion_caught() -> None:
    """A subprocess calling itself hits the runtime depth cap."""
    reg = _reg()
    # A flow that calls itself
    self_call = Flow(
        id="self", version=1,
        nodes=[GraphNode("c", "subprocess-call@1",
                         {"flow_id": "self", "flow_version": 1})],
        edges=[],
    )
    sub_reg = SubprocessRegistry()
    sub_reg.register(self_call)

    caller = Flow(
        nodes=[GraphNode("top", "subprocess-call@1",
                         {"flow_id": "self", "flow_version": 1})],
        edges=[],
    )
    compiled = compile(
        flow=caller, registry=reg,
        compound_types=[SUBPROCESS], subprocess_registry=sub_reg,
    )
    with pytest.raises(Exception, match="depth exceeded"):
        execute_sync(compiled)


def test_subprocess_mutual_recursion_caught() -> None:
    """A calls B calls A — the cycle must be caught by the depth cap.

    Currently detection is depth-based, so mutual recursion is caught only
    after N bounces. This test pins down that the mechanism still triggers
    (so a future cycle-based detector can replace it without regressing).
    """
    reg = _reg()
    # A calls B
    flow_a = Flow(
        id="A", version=1,
        nodes=[GraphNode("ca", "subprocess-call@1",
                         {"flow_id": "B", "flow_version": 1})],
        edges=[],
    )
    # B calls A
    flow_b = Flow(
        id="B", version=1,
        nodes=[GraphNode("cb", "subprocess-call@1",
                         {"flow_id": "A", "flow_version": 1})],
        edges=[],
    )
    sub_reg = SubprocessRegistry()
    sub_reg.register(flow_a)
    sub_reg.register(flow_b)

    caller = Flow(
        nodes=[GraphNode("top", "subprocess-call@1",
                         {"flow_id": "A", "flow_version": 1})],
        edges=[],
    )
    compiled = compile(
        flow=caller, registry=reg,
        compound_types=[SUBPROCESS], subprocess_registry=sub_reg,
    )
    with pytest.raises(Exception, match="depth exceeded"):
        execute_sync(compiled)
