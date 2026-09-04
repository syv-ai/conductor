"""Features used together.

A deciding node routes by ``SKIPPED`` into a branch whose failure triggers
compensation, and a ``Flow`` carrying every top-level field (id, version,
name, description, dependencies, triggers, ``on_error_default``,
``compensation=``, ``on_error=``) survives a YAML round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from conductor import (
    Flow,
    FlowDependency,
    FlowTrigger,
    GraphEdge,
    GraphNode,
    NodeRegistry,
    compile,
    execute,
)
from conductor._sentinel import SKIPPED
from conductor.dtype import DType
from conductor.errors import NodeExecutionError
from conductor.flow_format import flow_to_yaml, yaml_to_flow
from conductor.node import NodeDefinition
from conductor.returns import Result
from conductor.widgets import Number as NumberWidget
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "integration-test-text"
    title = "Text"


class Num(DType, float):
    id = "integration-test-number"
    title = "Number"


Out = Annotated[Txt, Result(title="Out")]


@dataclass(frozen=True)
class Branches:
    """What ``Decide.run`` returns: the value on one output, ``SKIPPED`` on the other."""

    high: Annotated[Num, Result(title="High", choice="branch")]
    low: Annotated[Num, Result(title="Low", choice="branch")]


class Decide(NodeDefinition):
    id = "decide"
    title = "Decide"
    description = "Routes a number to `high` or `low` around a threshold"
    category = "test"

    def run(
        self,
        value: Annotated[Num, NumberWidget(title="Value")] = Num(0),
        threshold: Annotated[Num, NumberWidget(title="Threshold")] = Num(50),
    ) -> Branches:
        if value > threshold:
            return Branches(high=value, low=SKIPPED)
        return Branches(high=SKIPPED, low=value)


class Echo(NodeDefinition):
    id = "echo"
    title = "Echo"
    description = "Returns its text"
    category = "test"

    def run(self, text: Annotated[Txt, Textarea(title="Text")] = Txt("ok")) -> Out:
        return text


class Fail(NodeDefinition):
    id = "fail"
    title = "Fail"
    description = "Fails"
    category = "test"

    def run(self) -> Out:
        raise NodeExecutionError("boom", node_id="fail")


class Record(NodeDefinition):
    id = "record"
    title = "Record"
    description = "Undoes what its target did"
    category = "test"

    def run(self) -> Out:
        return Txt("undone")


def _registry() -> NodeRegistry:
    reg = NodeRegistry()
    for node_cls in (Decide, Echo, Fail, Record):
        reg.register(node_cls)
    return reg


# ---------------------------------------------------------------------------
# Decision + compensation
# ---------------------------------------------------------------------------


async def test_decision_drives_a_branch_that_fails_triggers_compensation() -> None:
    """The deciding node picks a branch; that branch fails; compensation runs."""
    compiled = compile(
        nodes=[
            GraphNode("d", "decide", 1, {"value": 100}),
            GraphNode("a", "echo", 1, {"text": "A"}, compensation="undo_a"),
            GraphNode("b", "echo", 1, {"text": "B"}),
            GraphNode("boom", "fail", 1, {}),
            GraphNode("undo_a", "record", 1, {}),
        ],
        edges=[
            GraphEdge("e1", "d", "a", "high", "_"),
            GraphEdge("e2", "d", "b", "low", "_"),
            GraphEdge("e3", "a", "boom", "result", "_"),
        ],
        registry=_registry(),
    )

    events = [event async for event in execute(compiled)]

    kinds = [e["type"] for e in events]
    assert "flow_error" in kinds
    assert "compensation_start" in kinds
    # b did not run: the decision went to a
    b_completed = [
        e for e in events if e.get("type") == "node_complete" and e.get("node_id") == "b"
    ]
    assert not b_completed


# ---------------------------------------------------------------------------
# YAML round-trip of a flow using every top-level field
# ---------------------------------------------------------------------------


def test_yaml_roundtrip_full_featured_flow() -> None:
    original = Flow(
        id="full",
        version=2,
        name="Full-featured",
        description="Exercise every top-level field",
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
            GraphNode("charge", "echo", 1, {"text": "A"},
                      compensation="refund",
                      on_error="compensate"),
            GraphNode("refund", "record", 1, {}),
            GraphNode("b", "echo", 1, {"text": "B"}),
        ],
        edges=[
            GraphEdge("e1", "charge", "b", "result", "text"),
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
    assert e1.source_handle == "result"
    assert e1.target_handle == "text"
