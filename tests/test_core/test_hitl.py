"""Human-in-the-loop: pause, checkpoint, and resume."""

from typing import Annotated, Any

import pytest

from conductor import GraphEdge, GraphNode, NodeRegistry, compile
from conductor.errors import FlowPausedException, HumanInputRequired
from conductor.execution.checkpoint import FlowCheckpoint
from conductor.execution.engine import execute, execute_sync, resume, resume_sync, collect
from conductor.execution.store import FlowStore
from conductor.node import BaseNode
from conductor.execution.request import NodeExecRequest
from conductor.widgets import Text, Output


@pytest.fixture
def hitl_registry():
    reg = NodeRegistry()

    @reg.node("echo", version=1, name="Echo", description="Echo")
    def echo(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
        return text

    @reg.node("upper", version=1, name="Upper", description="Upper")
    def upper(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
        return text.upper()

    @reg.node("approve", version=1, name="Approve", description="Needs approval")
    def approve(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
        raise HumanInputRequired(
            f"Please approve: {text}",
            schema={"approved": "bool", "comment": "str"},
        )

    return reg


class TestStreamingPauseResume:
    async def test_pause_yields_flow_paused_event(self, hitl_registry):
        """When a node raises HumanInputRequired, execute yields flow_paused."""
        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "hello"}),
                GraphNode("n2", "approve@1", None),
                GraphNode("n3", "upper@1", None),
            ],
            edges=[
                GraphEdge("e1", "n1", "n2", "result", "text"),
                GraphEdge("e2", "n2", "n3", "result", "text"),
            ],
            registry=hitl_registry,
        )

        events = []
        async for event in execute(compiled):
            events.append(event)

        types = [e["type"] for e in events]
        assert "node_start" in types
        assert "node_complete" in types  # n1 completes
        assert "flow_paused" in types
        assert "flow_complete" not in types  # did NOT finish

        paused = next(e for e in events if e["type"] == "flow_paused")
        assert paused["node_id"] == "n2"
        assert "approve" in paused["prompt"]
        assert paused["schema"] == {"approved": "bool", "comment": "str"}
        assert "checkpoint" in paused

    async def test_resume_continues_from_checkpoint(self, hitl_registry):
        """resume() picks up from the paused node and finishes the flow."""
        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "hello"}),
                GraphNode("n2", "approve@1", None),
                GraphNode("n3", "upper@1", None),
            ],
            edges=[
                GraphEdge("e1", "n1", "n2", "result", "text"),
                GraphEdge("e2", "n2", "n3", "result", "text"),
            ],
            registry=hitl_registry,
        )

        # Phase 1: Execute until pause
        checkpoint = None
        async for event in execute(compiled):
            if event["type"] == "flow_paused":
                checkpoint = event["checkpoint"]
                break

        assert checkpoint is not None

        # Phase 2: Resume with human response
        events = []
        async for event in resume(compiled, checkpoint, response="approved_text"):
            events.append(event)

        types = [e["type"] for e in events]
        assert "node_complete" in types  # n2 completes with response
        assert "flow_complete" in types

        complete = next(e for e in events if e["type"] == "flow_complete")
        # n3 should have uppercased the human's response
        assert complete["results"]["n3"]["result"] == "APPROVED_TEXT"

    async def test_checkpoint_is_json_serializable(self, hitl_registry):
        """The checkpoint can be serialized to dict and restored."""
        import json

        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "hello"}),
                GraphNode("n2", "approve@1", None),
            ],
            edges=[GraphEdge("e1", "n1", "n2", "result", "text")],
            registry=hitl_registry,
        )

        checkpoint_dict = None
        async for event in execute(compiled):
            if event["type"] == "flow_paused":
                checkpoint_dict = event["checkpoint"]
                break

        # Must be JSON-serializable
        json_str = json.dumps(checkpoint_dict)
        restored = json.loads(json_str)

        cp = FlowCheckpoint.from_dict(restored)
        assert cp.waiting_node_id == "n2"
        assert cp.results["n1"] == {"result": "hello"}

    async def test_store_preserved_across_pause(self, hitl_registry):
        """FlowStore data survives the checkpoint/resume cycle."""
        reg = NodeRegistry()

        @reg.node("store-writer", version=1, name="Writer", description="Writes to store")
        def store_writer(
            text: Annotated[str, Text(label="In")],
            store: FlowStore,
        ) -> Annotated[str, Output(label="Out")]:
            store.set("cached", text.upper())
            return text

        @reg.node("approve", version=1, name="Approve", description="Pause")
        def approve(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            raise HumanInputRequired("Approve?")

        @reg.node("store-reader", version=1, name="Reader", description="Reads store")
        def store_reader(
            text: Annotated[str, Text(label="In")],
            store: FlowStore,
        ) -> Annotated[str, Output(label="Out")]:
            cached = store.get("cached", "MISSING")
            return f"{text}:{cached}"

        compiled = compile(
            nodes=[
                GraphNode("n1", "store-writer@1", {"text": "hello"}),
                GraphNode("n2", "approve@1", None),
                GraphNode("n3", "store-reader@1", None),
            ],
            edges=[
                GraphEdge("e1", "n1", "n2", "result", "text"),
                GraphEdge("e2", "n2", "n3", "result", "text"),
            ],
            registry=reg,
        )

        checkpoint = None
        async for event in execute(compiled):
            if event["type"] == "flow_paused":
                checkpoint = event["checkpoint"]
                break

        # Resume — store should still have "cached" -> "HELLO"
        results = {}
        async for event in resume(compiled, checkpoint, response="ok"):
            if event["type"] == "flow_complete":
                results = event["results"]

        assert results["n3"]["result"] == "ok:HELLO"


class TestSyncPauseResume:
    def test_execute_sync_raises_paused(self, hitl_registry):
        """execute_sync raises FlowPausedException with checkpoint."""
        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "hello"}),
                GraphNode("n2", "approve@1", None),
            ],
            edges=[GraphEdge("e1", "n1", "n2", "result", "text")],
            registry=hitl_registry,
        )

        with pytest.raises(FlowPausedException) as exc_info:
            execute_sync(compiled)

        cp = exc_info.value.checkpoint
        assert cp["waiting_node_id"] == "n2"

    def test_resume_sync_completes(self, hitl_registry):
        """resume_sync continues and returns final results."""
        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "hello"}),
                GraphNode("n2", "approve@1", None),
                GraphNode("n3", "upper@1", None),
            ],
            edges=[
                GraphEdge("e1", "n1", "n2", "result", "text"),
                GraphEdge("e2", "n2", "n3", "result", "text"),
            ],
            registry=hitl_registry,
        )

        try:
            execute_sync(compiled)
            pytest.fail("Should have raised FlowPausedException")
        except FlowPausedException as e:
            checkpoint = e.checkpoint

        results = resume_sync(compiled, checkpoint, response="go ahead")
        assert results["n3"]["result"] == "GO AHEAD"


class TestClassNodeHITL:
    def test_class_node_can_pause(self):
        """Class-based nodes can raise HumanInputRequired too."""
        reg = NodeRegistry()

        class ReviewNode(BaseNode):
            node_id = "review"
            node_name = "Review"
            node_description = "Needs review"

            def execute(self, req: NodeExecRequest) -> Any:
                raise HumanInputRequired(
                    f"Review: {req.inputs.get('text', '')}",
                    schema={"ok": "bool"},
                )

        reg.register_class(ReviewNode)

        @reg.node("echo", version=1, name="Echo", description="Echo")
        def echo(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            return text

        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "check this"}),
                GraphNode("n2", "review@1", None),
            ],
            edges=[GraphEdge("e1", "n1", "n2", "result", "text")],
            registry=reg,
        )

        with pytest.raises(FlowPausedException) as exc_info:
            execute_sync(compiled)

        cp = exc_info.value.checkpoint
        assert cp["waiting_node_id"] == "n2"
        assert "check this" in cp["prompt"]


class TestMultiplePauses:
    def test_flow_can_pause_multiple_times(self):
        """A flow with multiple approval nodes pauses and resumes sequentially."""
        reg = NodeRegistry()

        @reg.node("echo", version=1, name="Echo", description="Echo")
        def echo(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            return text

        @reg.node("approve", version=1, name="Approve", description="Pause")
        def approve(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            raise HumanInputRequired(f"Approve: {text}")

        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "start"}),
                GraphNode("n2", "approve@1", None),
                GraphNode("n3", "approve@1", None),  # Second approval
                GraphNode("n4", "echo@1", None),
            ],
            edges=[
                GraphEdge("e1", "n1", "n2", "result", "text"),
                GraphEdge("e2", "n2", "n3", "result", "text"),
                GraphEdge("e3", "n3", "n4", "result", "text"),
            ],
            registry=reg,
        )

        # First pause
        try:
            execute_sync(compiled)
        except FlowPausedException as e:
            cp1 = e.checkpoint
            assert cp1["waiting_node_id"] == "n2"

        # Resume -> hits second approval
        try:
            resume_sync(compiled, cp1, response="first_ok")
        except FlowPausedException as e:
            cp2 = e.checkpoint
            assert cp2["waiting_node_id"] == "n3"

        # Resume again -> completes
        results = resume_sync(compiled, cp2, response="second_ok")
        assert results["n4"]["result"] == "second_ok"
