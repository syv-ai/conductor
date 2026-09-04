"""Finding nodes: ``discover_nodes`` imports a package so its modules register
their classes, and an extension resolver answers for node types the registry
does not hold."""

from typing import Annotated, Any

from conductor.dtype import DType
from conductor.execution.engine import execute_sync
from conductor.graph.compiler import compile
from conductor.graph.model import GraphEdge, GraphNode
from conductor.node import NodeDefinition
from conductor.registry import NodeRegistry
from conductor.registry.discovery import discover_nodes
from conductor.returns import Result
from conductor.widgets import Textarea


class Txt(DType, str):
    id = "discovery-test-text"
    title = "Text"


Out = Annotated[Txt, Result(title="Out")]


def test_discover_from_package(tmp_path):
    """Importing every module of a package runs the ``register`` calls in them;
    the count is how many definitions the registry gained."""
    import sys

    import conductor

    reg = NodeRegistry()
    # The temp module reaches the registry through this attribute.
    conductor._test_discovery_registry = reg  # type: ignore[attr-defined]

    pkg_dir = tmp_path / "fake_nodes"
    pkg_dir.mkdir()
    (pkg_dir / "__init__.py").write_text("")
    (pkg_dir / "text_nodes.py").write_text(
        "from typing import Annotated\n"
        "import conductor\n"
        "from conductor.dtype import DType\n"
        "from conductor.node import NodeDefinition\n"
        "from conductor.returns import Result\n"
        "from conductor.widgets import Textarea\n"
        "\n"
        "class DiscoveredText(DType, str):\n"
        "    id = 'discovery-test-discovered-text'\n"
        "    title = 'Text'\n"
        "\n"
        "class Echo(NodeDefinition):\n"
        "    id = 'discovered-echo'\n"
        "    title = 'Echo'\n"
        "    description = 'Discovered'\n"
        "    category = 'test'\n"
        "\n"
        "    def run(self, text: Annotated[DiscoveredText, Textarea(title='In')])"
        " -> Annotated[DiscoveredText, Result(title='Out')]:\n"
        "        return text\n"
        "\n"
        "conductor._test_discovery_registry.register(Echo)\n"
    )

    sys.path.insert(0, str(tmp_path))
    try:
        count = discover_nodes("fake_nodes", reg)
        assert count == 1
        assert reg.get("discovered-echo") is not None
    finally:
        sys.path.pop(0)
        sys.modules.pop("fake_nodes", None)
        sys.modules.pop("fake_nodes.text_nodes", None)
        delattr(conductor, "_test_discovery_registry")


def test_extension_node_dispatched(registry):
    """A node type the registry does not hold runs through the extension
    resolver's executor, fed by its wired inputs like any other node."""
    from conductor.execution.request import NodeExecRequest

    class Echo(NodeDefinition):
        id = "echo"
        title = "Echo"
        description = "Echo"
        category = "test"

        def run(self, text: Annotated[Txt, Textarea(title="In")]) -> Out:
            return text

    registry.register(Echo)

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
            GraphNode("n1", "echo", 1, {"text": "hello"}),
            GraphNode("n2", "ext:custom", 1, None),
        ],
        edges=[GraphEdge("e1", "n1", "n2", "result", "text")],
        registry=registry,
        extension_resolver=MockExtensionResolver(),
    )

    results = execute_sync(compiled)
    assert results["n2"]["result"] == "extension:hello"
