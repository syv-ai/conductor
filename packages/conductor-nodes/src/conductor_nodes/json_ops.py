"""JSON nodes: ``json-parse``, ``json-stringify`` and ``json-get``. A JSON value is pydantic's ``JsonValue``."""

import json
import re
from typing import Annotated

from conductor import NodeError, NodeRegistry, Param
from conductor.widgets import NumberInput, Textarea
from pydantic import JsonValue

from conductor_nodes._inputs import Flag, Line

registry = NodeRegistry(category="json")

JsonText = Annotated[str, Param(title="JSON text", widget=Textarea())]


@registry.node("json-parse", title="Read JSON", description="Reads JSON text into a value. Text that isn't JSON is an error.")
def parse(text: JsonText) -> Annotated[JsonValue, Param(title="Parsed")]:
    """Reads a JSON text into a value; fails on text that isn't JSON."""
    try:
        parsed: JsonValue = json.loads(text)
    except json.JSONDecodeError as error:
        raise NodeError(f"The text is not JSON: {error.msg}, at character {error.pos}.", code="not_json") from None
    return parsed


@registry.node("json-stringify", title="Write JSON", description="Writes a value out as JSON text.")
def stringify(
    value: JsonValue,
    indent: Annotated[int, Param(ge=0, le=8, widget=NumberInput())] = 0,
    sort_keys: Flag = False,
) -> Annotated[str, Param(title="JSON")]:
    """Writes a value as JSON text."""
    return json.dumps(value, indent=indent or None, sort_keys=sort_keys)


@registry.node("json-get", title="Get from JSON", description="Picks one value out of JSON by its path, such as user.name. It gives null when nothing is there.")
def get_path(value: JsonValue, path: Line) -> Annotated[JsonValue, Param(title="Extracted")]:
    """Reads a dotted path from a JSON value, such as `user.name` or `items.0.id`; JSON null where there is nothing."""
    current = value
    for segment in path.split(".") if path else ():
        if isinstance(current, dict):
            current = current.get(segment.strip())
        elif isinstance(current, list) and re.fullmatch(r"-?[0-9]+", segment.strip()) and -len(current) <= int(segment) < len(current):
            current = current[int(segment)]
        else:
            return None
    return current
