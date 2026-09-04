"""FlowStore — the key-value store a flow run carries."""

from conductor.execution.store import FlowStore


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
