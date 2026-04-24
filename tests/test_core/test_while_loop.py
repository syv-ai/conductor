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
