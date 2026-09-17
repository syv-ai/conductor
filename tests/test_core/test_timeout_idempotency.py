"""Per-node timeout, declared as ``Policy(timeout=...)`` on a version."""

import asyncio
import time
from typing import Annotated

import pytest
from conductor import CompiledGraph, GraphNode, NodeRegistry
from conductor.dtype import DType
from conductor.errors import GraphExecutionError
from conductor.execution.engine import execute, execute_sync
from conductor.graph.model import Graph
from conductor.node import NodeDefinition, Policy, version
from conductor.returns import Result


class Txt(DType, str):
    id = "timeout-test-text"
    title = "Text"


Out = Annotated[Txt, Result(title="Out")]


def test_per_node_timeout_triggers() -> None:
    reg = NodeRegistry()

    class Slow(NodeDefinition):
        id = "slow"
        title = "Slow"
        description = "x"
        category = "test"

        @version(1, policy=Policy(timeout=0.2))
        def run(self) -> Out:
            time.sleep(1.0)
            return Txt("done")

    reg.register(Slow)
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n1", type="slow", version=1)]), reg)
    with pytest.raises(GraphExecutionError):
        execute_sync(compiled)


def test_a_timed_out_attempt_is_retried_and_the_retry_can_complete() -> None:
    """A timeout is a failure like any other: the version's retries apply, and
    the thread the timed-out attempt ran on is left to finish on its own."""
    reg = NodeRegistry()
    attempts: list[float] = []

    class SlowOnce(NodeDefinition):
        id = "slow-once"
        title = "Slow once"
        description = "x"
        category = "test"

        @version(1, policy=Policy(timeout=0.2, retries=1, delay=0.01))
        def run(self) -> Out:
            attempts.append(time.monotonic())
            if len(attempts) == 1:
                time.sleep(0.6)
            return Txt(f"attempt {len(attempts)}")

    reg.register(SlowOnce)
    compiled = CompiledGraph.from_graph(Graph(nodes=[GraphNode(id="n1", type="slow-once", version=1)]), reg)

    events = asyncio.run(_events(compiled))

    assert [e["type"] for e in events if e["type"] in ("node_retry", "graph_complete")] == ["node_retry", "graph_complete"]
    assert events[-1]["results"]["n1"]["result"] == "attempt 2"


async def _events(compiled: CompiledGraph) -> list[dict]:
    return [event async for event in execute(compiled)]
