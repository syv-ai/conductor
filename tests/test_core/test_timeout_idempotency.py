"""Per-node timeout, declared as ``Policy(timeout=...)`` on a version."""

import time
from typing import Annotated

import pytest
from conductor import (
    FlowExecutionError,
    GraphNode,
    NodeRegistry,
    compile,
    execute_sync,
)
from conductor.dtype import DType
from conductor.graph.model import Flow
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
    compiled = compile(Flow(nodes=[GraphNode("n1", "slow", 1)]), reg)
    with pytest.raises(FlowExecutionError):
        execute_sync(compiled)
