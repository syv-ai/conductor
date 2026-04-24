"""Tests for the while / until compound region."""

from __future__ import annotations

from typing import Annotated

import pytest
from conductor import (
    WHILE,
    FlowStore,
    GraphEdge,
    GraphNode,
    NodeRegistry,
    compile,
    execute_sync,
)
from conductor.widgets import Output, Text


def _registry() -> NodeRegistry:
    reg = NodeRegistry()

    @reg.node("while-start", version=1, name="WS", description="x")
    def while_start(
        condition: Annotated[str, Text(label="c")] = "false",
        max_iterations: Annotated[int, Text(label="m")] = 1000,
        negate: Annotated[bool, Text(label="n")] = False,
    ) -> tuple[
        Annotated[int, Output(label="iter")],
        Annotated[object, Output(label="last")],
    ]:
        raise NotImplementedError

    @reg.node("while-end", version=1, name="WE", description="x")
    def while_end(
        item: Annotated[object, Text(label="item")] = None,
    ) -> Annotated[object, Output(label="last")]:
        raise NotImplementedError

    @reg.node("bump", version=1, name="Bump", description="x")
    def bump(
        store: FlowStore,
        _: Annotated[int, Text(label="_")] = 0,
    ) -> Annotated[int, Output(label="count")]:
        val = store.get("c", 0) + 1
        store.set("c", val)
        return val

    return reg


def test_while_counts_to_three() -> None:
    reg = _registry()
    compiled = compile(
        nodes=[
            GraphNode("w_start", "while-start@1",
                      {"condition": "iteration < 3", "max_iterations": 10}),
            GraphNode("b", "bump@1", {}),
            GraphNode("w_end", "while-end@1", {}),
        ],
        edges=[
            GraphEdge("e1", "w_start", "b", "output_1", "_"),
            GraphEdge("e2", "b", "w_end", "result", "item"),
        ],
        registry=reg,
        compound_types=[WHILE],
    )
    r = execute_sync(compiled)
    # Final value is whatever bump returned on the last iteration
    assert r["w_end"]["result"] == 3


def test_while_zero_iterations() -> None:
    reg = _registry()
    compiled = compile(
        nodes=[
            GraphNode("w_start", "while-start@1",
                      {"condition": "false", "max_iterations": 10}),
            GraphNode("b", "bump@1", {}),
            GraphNode("w_end", "while-end@1", {}),
        ],
        edges=[
            GraphEdge("e1", "w_start", "b", "output_1", "_"),
            GraphEdge("e2", "b", "w_end", "result", "item"),
        ],
        registry=reg,
        compound_types=[WHILE],
    )
    r = execute_sync(compiled)
    # End result is None (no iterations)
    assert r["w_end"] == {} or r["w_end"].get("result") is None


def test_while_runaway_raises() -> None:
    reg = _registry()
    compiled = compile(
        nodes=[
            GraphNode("w_start", "while-start@1",
                      {"condition": "true", "max_iterations": 3}),
            GraphNode("b", "bump@1", {}),
            GraphNode("w_end", "while-end@1", {}),
        ],
        edges=[
            GraphEdge("e1", "w_start", "b", "output_1", "_"),
            GraphEdge("e2", "b", "w_end", "result", "item"),
        ],
        registry=reg,
        compound_types=[WHILE],
    )
    with pytest.raises(Exception) as exc_info:
        execute_sync(compiled)
    # Either FlowExecutionError wrapping LoopRunawayError, or direct.
    # The engine wraps the raise; LoopRunawayError surfaces via str.
    assert "max_iterations" in str(exc_info.value) or "runaway" in str(
        exc_info.value
    ).lower() or "exceeded" in str(exc_info.value)


def test_while_bad_condition_is_node_execution_error() -> None:
    """Runtime condition failures must raise NodeExecutionError, not CompilationError.

    CompilationError at runtime would falsely imply the graph itself is broken,
    misleading hosts/UIs. A bad expression at execute time is a node error.
    """
    from conductor.compound.protocol import Region
    from conductor.compound.while_loop import WhileNode
    from conductor.errors import NodeExecutionError
    from conductor.execution.request import NodeExecRequest

    region = Region(start_id="w_start", end_id="w_end", body_ids=frozenset())
    node = WhileNode(region, execution_order=())

    # Parse-time failure: malformed CEL.
    req = NodeExecRequest(
        node_id="w_start",
        node_type="while-start@1",
        inputs={"condition": "not a valid ))) expression"},
        data={},
        state=None,  # never reached — error raised before state is touched
    )
    with pytest.raises(NodeExecutionError) as exc_info:
        node.execute(req)
    assert exc_info.value.node_id == "w_start"
    assert exc_info.value.node_type == "while-start@1"
    assert "Invalid while-start condition" in str(exc_info.value)

    # Missing condition entirely.
    req_missing = NodeExecRequest(
        node_id="w_start",
        node_type="while-start@1",
        inputs={},
        data={},
        state=None,
    )
    with pytest.raises(NodeExecutionError) as exc_info:
        node.execute(req_missing)
    assert "no `condition`" in str(exc_info.value)


def test_until_negates_condition() -> None:
    """`negate=True` flips while's semantics to until."""
    reg = _registry()
    compiled = compile(
        nodes=[
            # Run until iteration >= 2 → while (iteration < 2) in negated form is
            # actually equivalent; just verify negate flips the sign.
            GraphNode("w_start", "while-start@1",
                      {"condition": "iteration >= 2",  # i.e. stop until this is true
                       "max_iterations": 10,
                       "negate": True}),
            GraphNode("b", "bump@1", {}),
            GraphNode("w_end", "while-end@1", {}),
        ],
        edges=[
            GraphEdge("e1", "w_start", "b", "output_1", "_"),
            GraphEdge("e2", "b", "w_end", "result", "item"),
        ],
        registry=reg,
        compound_types=[WHILE],
    )
    r = execute_sync(compiled)
    # Runs while NOT (iteration >= 2) → 2 iterations
    assert r["w_end"]["result"] == 2
