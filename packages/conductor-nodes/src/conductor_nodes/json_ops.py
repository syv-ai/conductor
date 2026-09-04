"""JSON nodes (``json-parse``, ``json-stringify``, ``json-get``)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Annotated, Any

from conductor.returns import Result
from conductor.widgets import Range, Switch, Textarea
from conductor.widgets import Text as TextWidget

from conductor_nodes.types import Flag, Json, Number, StdlibNode, Text

if TYPE_CHECKING:
    from conductor import NodeRegistry


class Parse(StdlibNode):
    id = "json-parse"
    title = "JSON Parse"
    description = "Parses a JSON string into a value"
    category = "json"

    def run(
        self, text: Annotated[Text, Textarea(title="JSON text")]
    ) -> Annotated[Json, Result(title="Parsed")]:
        return Json(json.loads(text))


class Stringify(StdlibNode):
    id = "json-stringify"
    title = "JSON Stringify"
    description = "Serializes a value to a JSON string"
    category = "json"

    def run(
        self,
        value: Annotated[Json, Textarea(title="Value")],
        indent: Annotated[
            Number, Range(title="Indent", min_val=0, max_val=8, step=1)
        ] = Number(0),
        sort_keys: Annotated[Flag, Switch(title="Sort keys")] = Flag(False),
    ) -> Annotated[Text, Result(title="JSON")]:
        return Text(json.dumps(value.value, indent=int(indent) or None, sort_keys=bool(sort_keys)))


class GetPath(StdlibNode):
    id = "json-get"
    title = "JSON Get"
    description = "Reads a dotted path from a JSON value (e.g. 'user.name', 'items.0.id')"
    category = "json"

    def run(
        self,
        value: Annotated[Json, Textarea(title="Value")],
        path: Annotated[Text, TextWidget(title="Path")],
    ) -> Annotated[Json, Result(title="Extracted")]:
        return Json(_get_path(value.value, path))


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


NODES = (Parse, Stringify, GetPath)


def register(registry: "NodeRegistry") -> None:
    """Register every JSON node on the supplied registry."""
    for node_cls in NODES:
        registry.register(node_cls)
