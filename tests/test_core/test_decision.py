"""Tests for the decision node + edge guards feature."""

from __future__ import annotations

from typing import Annotated

import pytest
from conductor import (
    CompilationError,
    GraphEdge,
    GraphNode,
    NodeRegistry,
    compile,
    execute_sync,
)
from conductor.widgets import Output, Text

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _registry() -> NodeRegistry:
    reg = NodeRegistry()

    @reg.node("decision", version=1, name="Decision", description="x", is_decision=True)
    def decision(value: Annotated[object, Text(label="value")] = None) -> Annotated[
        object, Output(label="v"),
    ]:
        return value

    @reg.node("echo", version=1, name="Echo", description="x")
    def echo(
        text: Annotated[str, Text(label="text")] = "default",
    ) -> Annotated[str, Output(label="out")]:
        return text

    return reg


# ---------------------------------------------------------------------------
# Compile-time validation
# ---------------------------------------------------------------------------


def test_decision_must_have_else_edge() -> None:
    reg = _registry()
    with pytest.raises(CompilationError, match="exactly one else edge"):
        compile(
            nodes=[
                GraphNode("d", "decision@1", {"value": 1}),
                GraphNode("a", "echo@1", {}),
            ],
            edges=[
                GraphEdge("e1", "d", "a", "result", "text", when="value > 0"),
            ],
            registry=reg,
        )


def test_decision_must_have_guarded_edge() -> None:
    reg = _registry()
    with pytest.raises(CompilationError, match="at least one guarded edge"):
        compile(
            nodes=[
                GraphNode("d", "decision@1", {"value": 1}),
                GraphNode("a", "echo@1", {}),
            ],
            edges=[
                GraphEdge("e1", "d", "a", "result", "text"),
            ],
            registry=reg,
        )


def test_two_else_edges_raises() -> None:
    reg = _registry()
    with pytest.raises(CompilationError, match="exactly one else edge"):
        compile(
            nodes=[
                GraphNode("d", "decision@1", {"value": 1}),
                GraphNode("a", "echo@1", {}),
                GraphNode("b", "echo@1", {}),
            ],
            edges=[
                GraphEdge("e1", "d", "a", "result", "text", when="value > 0"),
                GraphEdge("e2", "d", "a", "result", "text"),
                GraphEdge("e3", "d", "b", "result", "text"),
            ],
            registry=reg,
        )


def test_when_on_non_decision_is_rejected() -> None:
    reg = _registry()
    with pytest.raises(CompilationError, match="not a decision node"):
        compile(
            nodes=[
                GraphNode("a", "echo@1", {"text": "hi"}),
                GraphNode("b", "echo@1", {}),
            ],
            edges=[
                GraphEdge("e1", "a", "b", "result", "text", when="true"),
            ],
            registry=reg,
        )


def test_invalid_cel_on_edge() -> None:
    reg = _registry()
    with pytest.raises(CompilationError, match="invalid `when` expression"):
        compile(
            nodes=[
                GraphNode("d", "decision@1", {"value": 1}),
                GraphNode("a", "echo@1", {}),
                GraphNode("b", "echo@1", {}),
            ],
            edges=[
                GraphEdge("e1", "d", "a", "result", "text", when="value >"),
                GraphEdge("e2", "d", "b", "result", "text"),
            ],
            registry=reg,
        )


# ---------------------------------------------------------------------------
# Runtime branching
# ---------------------------------------------------------------------------


def test_guarded_branch_taken() -> None:
    reg = _registry()
    compiled = compile(
        nodes=[
            GraphNode("d", "decision@1", {"value": 500}),
            GraphNode("high", "echo@1", {"text": "HIGH"}),
            GraphNode("low", "echo@1", {"text": "LOW"}),
        ],
        edges=[
            GraphEdge("e1", "d", "high", "result", None,
                      when="result > 1000", priority=10),
            GraphEdge("e2", "d", "low", "result", None),
        ],
        registry=reg,
    )
    r = execute_sync(compiled)
    assert r["low"] == {"result": "LOW"}
    assert "high" not in r


def test_else_branch_taken() -> None:
    reg = _registry()
    compiled = compile(
        nodes=[
            GraphNode("d", "decision@1", {"value": 2000}),
            GraphNode("high", "echo@1", {"text": "HIGH"}),
            GraphNode("low", "echo@1", {"text": "LOW"}),
        ],
        edges=[
            GraphEdge("e1", "d", "high", "result", None,
                      when="result > 1000"),
            GraphEdge("e2", "d", "low", "result", None),
        ],
        registry=reg,
    )
    r = execute_sync(compiled)
    assert r["high"] == {"result": "HIGH"}
    assert "low" not in r


def test_priority_orders_guards() -> None:
    reg = _registry()
    compiled = compile(
        nodes=[
            GraphNode("d", "decision@1", {"value": 100}),
            GraphNode("a", "echo@1", {"text": "A"}),
            GraphNode("b", "echo@1", {"text": "B"}),
            GraphNode("c", "echo@1", {"text": "C"}),
        ],
        edges=[
            # Lower priority; should not be picked first
            GraphEdge("e1", "d", "a", "result", None,
                      when="result > 50", priority=1),
            # Both match; higher priority wins
            GraphEdge("e2", "d", "b", "result", None,
                      when="result > 10", priority=10),
            GraphEdge("e3", "d", "c", "result", None),
        ],
        registry=reg,
    )
    # e2 has higher priority and matches first
    r = execute_sync(compiled)
    # Since priority=10 > priority=1, e2 is evaluated first and taken.
    assert r["b"] == {"result": "B"}
    assert "a" not in r
    assert "c" not in r
