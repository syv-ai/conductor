"""JSON nodes (``json-parse``, ``json-stringify``, ``json-get``)."""

from __future__ import annotations

import json
from typing import Annotated, Any

from conductor import Param, Result
from conductor.widgets import Range, Textarea

from conductor_nodes.types import Flag, Json, Line, stdlib_node


@stdlib_node("json-parse", title="JSON Parse", description="Parses a JSON string into a value", category="json")
def parse(text: Annotated[str, Param(title="JSON text", widget=Textarea())]
) -> Annotated[Json, Result(title="Parsed")]:
    return Json(json.loads(text))


@stdlib_node("json-stringify", title="JSON Stringify", description="Serializes a value to a JSON string", category="json")
def stringify(
    value: Annotated[Json, Param(widget=Textarea())],
    indent: Annotated[
        float, Param(ge=0, le=8, widget=Range(step=1))
    ] = 0.0,
    sort_keys: Flag = False,
) -> Annotated[str, Result(title="JSON")]:
    return json.dumps(value.value, indent=int(indent) or None, sort_keys=bool(sort_keys))


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

@stdlib_node("json-get", title="JSON Get", description="Reads a dotted path from a JSON value (e.g. 'user.name', 'items.0.id')", category="json")
def get_path(
    value: Annotated[Json, Param(widget=Textarea())],
    path: Line,
) -> Annotated[Json, Result(title="Extracted")]:
    return Json(_get_path(value.value, path))


#: Every node this module defines, in palette order.
NODES = (parse, stringify, get_path)
