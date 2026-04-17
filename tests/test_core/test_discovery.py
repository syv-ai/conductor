"""Phase 4: Auto-discovery, schema serialization, and extension resolver."""

from typing import Annotated, Any

from conductor.execution.engine import execute_sync
from conductor.graph.compiler import compile
from conductor.graph.model import GraphEdge, GraphNode
from conductor.registry import NodeRegistry
from conductor.registry.schema import serialize_registry
from conductor.widgets import Output, Text

# ---------------------------------------------------------------------------
# Auto-discovery
# ---------------------------------------------------------------------------

class TestAutoDiscovery:
    def test_discover_from_package(self, tmp_path):
        """discover() imports all modules in a package, triggering @node decorators."""
        import sys

        # Create a registry and stash it where the temp module can find it
        reg = NodeRegistry()
        import conductor
        conductor._test_discovery_registry = reg  # type: ignore[attr-defined]

        pkg_dir = tmp_path / "fake_nodes"
        pkg_dir.mkdir()
        (pkg_dir / "__init__.py").write_text("")
        (pkg_dir / "text_nodes.py").write_text(
            "from typing import Annotated\n"
            "from conductor.widgets import Text, Output\n"
            "import conductor\n"
            "registry = conductor._test_discovery_registry\n"
            "\n"
            "@registry.node('discovered-echo', version=1, name='Echo', description='Discovered')\n"
            "def echo(text: Annotated[str, Text(label='In')]) -> Annotated[str, Output(label='Out')]:\n"
            "    return text\n"
        )

        sys.path.insert(0, str(tmp_path))
        try:
            count = reg.discover("fake_nodes")
            assert count >= 1
            assert reg.get("discovered-echo@1") is not None
        finally:
            sys.path.pop(0)
            if "fake_nodes" in sys.modules:
                del sys.modules["fake_nodes"]
            if "fake_nodes.text_nodes" in sys.modules:
                del sys.modules["fake_nodes.text_nodes"]
            delattr(conductor, "_test_discovery_registry")


# ---------------------------------------------------------------------------
# Schema serialization
# ---------------------------------------------------------------------------

class TestSchemaSerialisation:
    def test_serialize_registry_produces_list(self, registry):
        @registry.node("echo", version=1, name="Echo", description="Echo node")
        def echo(
            text: Annotated[str, Text(label="Input")],
        ) -> Annotated[str, Output(label="Output")]:
            return text

        result = serialize_registry(registry)
        assert isinstance(result, list)
        assert len(result) >= 1

    def test_serialized_node_has_expected_fields(self, registry):
        @registry.node("echo", version=1, name="Echo", description="Echo node")
        def echo(
            text: Annotated[str, Text(label="Input")],
        ) -> Annotated[str, Output(label="Output")]:
            return text

        result = serialize_registry(registry)
        node_json = result[0]

        assert node_json["id"] == "echo@1"
        assert node_json["base_id"] == "echo"
        assert node_json["version"] == 1
        assert node_json["name"] == "Echo"
        assert node_json["description"] == "Echo node"
        assert isinstance(node_json["inputs"], list)
        assert isinstance(node_json["outputs"], list)

    def test_serialized_input_has_widget_info(self, registry):
        @registry.node("echo", version=1, name="Echo", description="Echo node")
        def echo(
            text: Annotated[str, Text(label="My Input", description="Enter text")],
        ) -> Annotated[str, Output(label="Output")]:
            return text

        result = serialize_registry(registry)
        inputs = result[0]["inputs"]
        assert len(inputs) == 1
        assert inputs[0]["name"] == "text"
        assert inputs[0]["label"] == "My Input"
        assert inputs[0]["widget"] == "text"

    def test_serialized_output_has_label(self, registry):
        @registry.node("echo", version=1, name="Echo", description="Echo node")
        def echo(
            text: Annotated[str, Text(label="Input")],
        ) -> Annotated[str, Output(label="Result")]:
            return text

        result = serialize_registry(registry)
        outputs = result[0]["outputs"]
        assert len(outputs) >= 1
        assert outputs[0]["label"] == "Result"

    def test_deprecated_flag(self, registry):
        @registry.node("echo", version=1, name="Echo v1", description="Old")
        def echo_v1(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            return text

        @registry.node("echo", version=2, name="Echo v2", description="New")
        def echo_v2(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            return text

        result = serialize_registry(registry)
        v1 = next(n for n in result if n["id"] == "echo@1")
        v2 = next(n for n in result if n["id"] == "echo@2")
        assert v1["deprecated"] is True
        assert v2["deprecated"] is False


# ---------------------------------------------------------------------------
# Extension resolver
# ---------------------------------------------------------------------------

class TestExtensionResolver:
    def test_extension_node_dispatched(self, registry):
        """Nodes with types not in the registry can be handled by an extension."""
        from conductor.execution.request import NodeExecRequest

        @registry.node("echo", version=1, name="Echo", description="Echo")
        def echo(text: Annotated[str, Text(label="In")]) -> Annotated[str, Output(label="Out")]:
            return text

        class MockExtensionExecutor:
            def execute(self, req: NodeExecRequest) -> Any:
                return f"extension:{req.inputs.get('text', '')}"

        class MockExtensionResolver:
            def is_known_type(self, node_type: str) -> bool:
                return node_type.startswith("ext:")

            def create_executor(self, node_type: str):
                return MockExtensionExecutor()

        compiled = compile(
            nodes=[
                GraphNode("n1", "echo@1", {"text": "hello"}),
                GraphNode("n2", "ext:custom@1", None),
            ],
            edges=[GraphEdge("e1", "n1", "n2", "result", "text")],
            registry=registry,
            extension_resolver=MockExtensionResolver(),
        )

        results = execute_sync(compiled)
        assert results["n2"]["result"] == "extension:hello"
