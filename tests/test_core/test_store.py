"""Phase 2: FlowStore — cross-node data sharing."""

from typing import Annotated

import pytest

from flowengine.execution.store import FlowStore
from flowengine.graph.model import GraphNode, GraphEdge
from flowengine.graph.compiler import compile
from flowengine.execution.engine import execute_sync
from flowengine.widgets import Text, Output


class TestFlowStore:
    def test_set_and_get(self):
        store = FlowStore()
        store.set("key", "value")
        assert store.get("key") == "value"

    def test_get_default(self):
        store = FlowStore()
        assert store.get("missing") is None
        assert store.get("missing", "fallback") == "fallback"

    def test_has(self):
        store = FlowStore()
        assert store.has("key") is False
        store.set("key", "value")
        assert store.has("key") is True

    def test_keys(self):
        store = FlowStore()
        store.set("a", 1)
        store.set("b", 2)
        assert set(store.keys()) == {"a", "b"}

    def test_clear(self):
        store = FlowStore()
        store.set("a", 1)
        store.clear()
        assert store.has("a") is False

    def test_stores_any_type(self):
        store = FlowStore()
        store.set("dict", {"nested": True})
        store.set("list", [1, 2, 3])
        store.set("none", None)
        assert store.get("dict") == {"nested": True}
        assert store.get("list") == [1, 2, 3]
        assert store.get("none") is None


class TestFlowStoreInjection:
    """FlowStore is auto-injected into function nodes that declare it."""

    def test_function_node_receives_store(self, registry):
        @registry.node("producer", version=1, name="Producer", description="Stores data")
        def producer(
            text: Annotated[str, Text(label="Input")],
            store: FlowStore,
        ) -> Annotated[str, Output(label="Output")]:
            store.set("cached_text", text.upper())
            return text

        @registry.node("consumer", version=1, name="Consumer", description="Reads store")
        def consumer(
            text: Annotated[str, Text(label="Input")],
            store: FlowStore,
        ) -> Annotated[str, Output(label="Output")]:
            cached = store.get("cached_text", "")
            return f"{text}:{cached}"

        compiled = compile(
            nodes=[
                GraphNode("n1", "producer@1", {"text": "hello"}),
                GraphNode("n2", "consumer@1", None),
            ],
            edges=[GraphEdge("e1", "n1", "n2", "result", "text")],
            registry=registry,
        )

        results = execute_sync(compiled)
        # Consumer gets "hello" via edge AND "HELLO" via store
        assert results["n2"]["result"] == "hello:HELLO"

    def test_store_not_treated_as_node_input(self, registry):
        """FlowStore parameter should NOT appear as a node input in metadata."""

        @registry.node("with_store", version=1, name="WithStore", description="Has store")
        def with_store(
            text: Annotated[str, Text(label="Input")],
            store: FlowStore,
        ) -> Annotated[str, Output(label="Output")]:
            return text

        node_def = registry.get("with_store@1")
        input_names = [inp.name for inp in node_def.inputs]
        assert "text" in input_names
        assert "store" not in input_names  # store is injected, not an input

    def test_node_without_store_works_fine(self, registry):
        """Nodes that don't declare FlowStore still work."""

        @registry.node("plain", version=1, name="Plain", description="No store")
        def plain(
            text: Annotated[str, Text(label="Input")],
        ) -> Annotated[str, Output(label="Output")]:
            return text

        compiled = compile(
            nodes=[GraphNode("n1", "plain@1", {"text": "hello"})],
            edges=[],
            registry=registry,
        )

        results = execute_sync(compiled)
        assert results["n1"]["result"] == "hello"
