"""Compensation: ``compensation=`` and ``on_error=`` on a placement.

Compile refuses a dangling or self-referential compensation target and an
unknown ``on_error`` policy. At run time a failure compensates every
completed node in reverse order, best-effort, with a ``compensation_*``
event per step; a compensation node never runs on the happy path.
"""

from __future__ import annotations

from typing import Annotated

import pytest
from conductor import (
    CompilationError,
    FlowExecutionError,
    GraphEdge,
    GraphNode,
    NodeRegistry,
    compile,
    execute,
    execute_sync,
)
from conductor.dtype import DType
from conductor.errors import NodeExecutionError
from conductor.node import NodeDefinition
from conductor.returns import Result
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "compensation-test-text"
    title = "Text"


Out = Annotated[Txt, Result(title="Out")]
#: The compensated node's id, handed to a compensation node by the engine.
Target = Annotated[Txt, Textarea(title="Target node")]

#: What ran, in order. Cleared before every test.
log: list[str] = []


@pytest.fixture(autouse=True)
def _clear_log():
    log.clear()


class Charge(NodeDefinition):
    id = "charge"
    title = "Charge"
    description = "Charges, and records that it did"
    category = "test"

    def run(self) -> Out:
        log.append("charge")
        return Txt("ch_1")


class Refund(NodeDefinition):
    id = "refund"
    title = "Refund"
    description = "Undoes a charge; records which node it compensated"
    category = "test"

    def run(self, target_node_id: Target = Txt("")) -> Out:
        log.append(f"refund:{target_node_id}")
        return Txt("rf_1")


class SaveFail(NodeDefinition):
    id = "save-fail"
    title = "Save fail"
    description = "Fails after recording that it ran"
    category = "test"

    def run(self) -> Out:
        log.append("save_fail")
        raise NodeExecutionError("db down", node_id="save_fail")


def _registry(*extra: type[NodeDefinition]) -> NodeRegistry:
    reg = NodeRegistry()
    for node_cls in (Charge, Refund, SaveFail, *extra):
        reg.register(node_cls)
    return reg


async def _events(compiled) -> list[dict]:
    """Every event of one run, including those after a ``flow_error``."""
    return [event async for event in execute(compiled)]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_missing_compensation_target_raises() -> None:
    with pytest.raises(CompilationError, match="no such node"):
        compile(
            nodes=[GraphNode("n1", "charge", 1, {}, compensation="ghost")],
            edges=[], registry=_registry(),
        )


def test_self_compensation_raises() -> None:
    with pytest.raises(CompilationError, match="own compensation"):
        compile(
            nodes=[GraphNode("n1", "charge", 1, {}, compensation="n1")],
            edges=[], registry=_registry(),
        )


def test_invalid_on_error_policy() -> None:
    with pytest.raises(CompilationError, match="invalid on_error"):
        compile(
            nodes=[GraphNode("n1", "charge", 1, {}, on_error="bogus")],
            edges=[], registry=_registry(),
        )


# ---------------------------------------------------------------------------
# Runtime cascade
# ---------------------------------------------------------------------------


def test_cascade_runs_in_reverse() -> None:
    compiled = compile(
        nodes=[
            GraphNode("n1", "charge", 1, {}, compensation="c1"),
            GraphNode("c1", "refund", 1, {}),
            GraphNode("n2", "save-fail", 1, {}),
        ],
        edges=[GraphEdge("e1", "n1", "n2", "result", "_")],
        registry=_registry(),
    )
    with pytest.raises(FlowExecutionError):
        execute_sync(compiled)
    # charge ran, save_fail ran, refund ran as n1's compensation
    assert "charge" in log
    assert "save_fail" in log
    assert "refund:n1" in log


def test_compensation_nodes_not_run_as_regular_nodes() -> None:
    """A ``compensation=`` target does not run on the happy path."""
    compiled = compile(
        nodes=[
            GraphNode("n1", "charge", 1, {}, compensation="c1"),
            GraphNode("c1", "refund", 1, {}),
        ],
        edges=[],
        registry=_registry(),
    )
    results = execute_sync(compiled)
    assert log == ["charge"]
    assert "n1" in results
    assert "c1" not in results


def test_on_error_continue_skips_compensation() -> None:
    class Oops(NodeDefinition):
        id = "oops"
        title = "Oops"
        description = "Fails"
        category = "test"

        def run(self) -> Out:
            raise NodeExecutionError("oops", node_id="oops")

    compiled = compile(
        nodes=[
            GraphNode("n1", "oops", 1, {}, on_error="continue"),
            GraphNode("n2", "charge", 1, {}),
        ],
        edges=[GraphEdge("e1", "n1", "n2", "result", "_")],
        registry=_registry(Oops),
    )
    execute_sync(compiled)
    # n2 still ran even though n1 failed
    assert "charge" in log


async def test_cascade_continues_when_a_compensation_fails() -> None:
    """A compensation that itself raises does not abort the cascade.

    The cascade is best-effort: a failing compensation emits
    ``compensation_failed`` and the remaining compensations still run.
    """

    class BadRefund(NodeDefinition):
        id = "bad-refund"
        title = "Bad refund"
        description = "Records the attempt, then fails"
        category = "test"

        def run(self, target_node_id: Target = Txt("")) -> Out:
            log.append(f"bad_refund:{target_node_id}")
            raise NodeExecutionError("refund gateway down", node_id="bad_refund")

    compiled = compile(
        nodes=[
            # n1 completes; its compensation (bad) will fail.
            GraphNode("n1", "charge", 1, {}, compensation="bad"),
            GraphNode("bad", "bad-refund", 1, {}),
            # n2 completes; its compensation (good) must still run.
            GraphNode("n2", "charge", 1, {}, compensation="good"),
            GraphNode("good", "refund", 1, {}),
            # n3 fails, triggering the cascade: n2's compensation, then n1's.
            GraphNode("n3", "save-fail", 1, {}),
        ],
        edges=[
            GraphEdge("e1", "n1", "n2", "result", "_"),
            GraphEdge("e2", "n2", "n3", "result", "_"),
        ],
        registry=_registry(BadRefund),
    )

    kinds = [e["type"] for e in await _events(compiled)]

    assert "compensation_failed" in kinds
    # The good compensation still ran after the bad one failed.
    assert "refund:n2" in log
    # Both compensation attempts happened.
    assert "bad_refund:n1" in log


async def test_cascade_events_emitted() -> None:
    compiled = compile(
        nodes=[
            GraphNode("n1", "charge", 1, {}, compensation="c1"),
            GraphNode("c1", "refund", 1, {}),
            GraphNode("n2", "save-fail", 1, {}),
        ],
        edges=[GraphEdge("e1", "n1", "n2", "result", "_")],
        registry=_registry(),
    )

    kinds = [e["type"] for e in await _events(compiled)]
    assert "compensation_start" in kinds
    assert "compensation_complete" in kinds
