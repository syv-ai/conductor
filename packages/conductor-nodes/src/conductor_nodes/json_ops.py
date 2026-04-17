"""JSON nodes (``json-parse``, ``json-stringify``, ``json-get``)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any

from conductor.widgets import Checkbox, Output, Range, Text, Textarea

if TYPE_CHECKING:
    from conductor import NodeRegistry


def register(registry: "NodeRegistry") -> None:
    """Register every JSON node on the supplied registry."""

    @registry.node("json-parse", version=1, name="JSON Parse", description="Parses a JSON string into an object")
    def parse(
        text: Annotated[str, Textarea(label="JSON text")],
    ) -> Annotated[object, Output(label="Parsed")]:
        return json.loads(text)

    @registry.node(
        "json-stringify", version=1, name="JSON Stringify",
        description="Serializes a value to a JSON string",
    )
    def stringify(
        value: Annotated[object, Text(label="Value")],
        indent: Annotated[int, Range(label="Indent", min_val=0, max_val=8, step=1)] = 0,
        sort_keys: Annotated[bool, Checkbox(label="Sort keys")] = False,
    ) -> Annotated[str, Output(label="JSON")]:
        return json.dumps(value, indent=indent or None, sort_keys=sort_keys)

    @registry.node(
        "json-get", version=1, name="JSON Get",
        description="Reads a dotted path from a JSON-like value (e.g. 'user.name', 'items.0.id')",
    )
    def get_path(
        value: Annotated[object, Text(label="Value")],
        path: Annotated[str, Text(label="Path")],
    ) -> Annotated[object, Output(label="Extracted")]:
        return _get_path(value, path)


def _get_path(value: Any, path: str) -> Any:
    """Walk a dotted path — each segment indexes a dict by key or a list by integer."""
    if not path:
        return value
    current = value
    for raw in path.split("."):
        segment = raw.strip()
        if isinstance(current, dict):
            current = current.get(segment)
        elif isinstance(current, list):
            try:
                current = current[int(segment)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current
