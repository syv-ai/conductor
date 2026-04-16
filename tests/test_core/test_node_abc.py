"""Phase 2: BaseNode ABC and class-based node registration."""

from typing import Annotated, Any

import pytest

from flowengine.node import BaseNode
from flowengine.types import NodeCategory
from flowengine.execution.request import NodeExecRequest
from flowengine.graph.model import GraphNode, GraphEdge
from flowengine.graph.compiler import compile
from flowengine.execution.engine import execute_sync
from flowengine.widgets import Text, Output


class TestBaseNodeABC:
    def test_cannot_instantiate_without_execute(self):
        """BaseNode requires execute() to be implemented."""
        with pytest.raises(TypeError):

            class Incomplete(BaseNode):
                node_id = "incomplete"
                node_name = "Incomplete"
                node_description = "Missing execute"

            Incomplete()

    def test_concrete_subclass_instantiable(self):
        class MyNode(BaseNode):
            node_id = "my-node"
            node_name = "My Node"
            node_description = "A test node"

            def execute(self, req: NodeExecRequest) -> Any:
                return req.inputs.get("text", "default")

        node = MyNode()
        assert node.node_id == "my-node"

    def test_category_defaults_to_io(self):
        class MyNode(BaseNode):
            node_id = "my-node"
            node_name = "My Node"
            node_description = "A test node"

            def execute(self, req: NodeExecRequest) -> Any:
                return "ok"

        assert MyNode.node_category == NodeCategory.IO


class TestClassBasedRegistration:
    def test_register_class_node(self, registry):
        class Reverser(BaseNode):
            node_id = "reverser"
            node_name = "Reverser"
            node_description = "Reverses text"

            def execute(self, req: NodeExecRequest) -> Any:
                return req.inputs["text"][::-1]

        registry.register_class(Reverser)
        node_def = registry.get("reverser@1")
        assert node_def is not None
        assert node_def.name == "Reverser"

    def test_class_node_executes_in_flow(self, registry):
        class Reverser(BaseNode):
            node_id = "reverser"
            node_name = "Reverser"
            node_description = "Reverses text"

            def execute(self, req: NodeExecRequest) -> Any:
                return req.inputs["text"][::-1]

        registry.register_class(Reverser)

        # Also register a function node to wire into
        @registry.node("echo", version=1, name="Echo", description="Echo")
        def echo(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            return text

        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "hello"}),
                GraphNode("n2", "reverser@1", None),
            ],
            edges=[GraphEdge("e1", "n1", "n2", "result", "text")],
            registry=registry,
        )

        results = execute_sync(compiled)
        assert results["n2"]["result"] == "olleh"

    def test_class_node_with_category(self, registry):
        class MyControl(BaseNode):
            node_id = "my-control"
            node_name = "My Control"
            node_description = "A control node"
            node_category = NodeCategory.CONTROL

            def execute(self, req: NodeExecRequest) -> Any:
                return "ok"

        registry.register_class(MyControl)
        node_def = registry.get("my-control@1")
        assert node_def.category == NodeCategory.CONTROL
