"""FlowStore — key-value store for cross-node data sharing."""

from typing import Any


class FlowStore:
    """Side-channel key-value store within a flow run.

    Nodes can store and retrieve data here for cross-node sharing
    (e.g., parsed documents, cached API responses). This complements
    edge-based data flow rather than replacing it.

    Function nodes declare `store: FlowStore` in their signature
    to receive it via auto-injection.
    """

    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self._data: dict[str, Any] = data or {}

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def has(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> list[str]:
        return list(self._data.keys())

    def clear(self) -> None:
        self._data.clear()
