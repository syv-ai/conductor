"""Nodes for JSON: ``json-parse``, ``json-stringify`` and ``json-get``.

A JSON value here is pydantic's ``JsonValue``: a text, number, yes or no, ``None``, or a list or dict of those.

A float that isn't finite has no JSON form. Python's ``json`` reads ``NaN``, ``Infinity`` and ``-Infinity``, reads
a number too large for a float, such as ``1e999``, as ``inf``, and writes ``inf`` and ``nan`` back. These nodes
refuse all of them with ``not_json``.
"""

import json
import math
import re
from json import JSONDecodeError
from typing import Annotated, NoReturn

from conductor import NodeError, NodeRegistry, Param, Policy
from conductor.widgets import Textarea
from pydantic import JsonValue

registry = NodeRegistry(category="json")


@registry.node(
    "json-parse",
    title="Read JSON",
    description="Reads JSON text into a value. Text that isn't JSON is an error.",
    policy=Policy(costs_nothing=True),
)
def parse(
    text: Annotated[str, Param(title="JSON text", widget=Textarea())],
) -> Annotated[JsonValue, Param(title="Parsed")]:
    """Turns JSON text into a value.

    Raises:
        NodeError: ``not_json`` if the text isn't JSON, which ``NaN``, ``Infinity`` and ``-Infinity`` aren't, or
            if it has a number too large for a float, such as ``1e999``.
    """
    try:
        parsed: JsonValue = json.loads(text, parse_constant=_refuse_constant, parse_float=_finite)
    except JSONDecodeError as error:
        raise NodeError(
            f"The text is not JSON: {error.msg}, at character {error.pos}.", code="not_json"
        ) from None
    # json.loads raises a plain ValueError, not a JSONDecodeError, for a number of more than 4,300 digits
    except ValueError as error:
        raise NodeError(f"The text is not JSON: {error}.", code="not_json") from None
    return parsed


def _refuse_constant(constant: str) -> NoReturn:
    """Refuses ``NaN``, ``Infinity`` or ``-Infinity``, which ``json.loads`` calls this with.

    Raises:
        NodeError: ``not_json``, always.
    """
    raise NodeError(
        f'The text is not JSON: {constant} is not a number JSON has. Write it as a text instead: "{constant}".',
        code="not_json",
    )


def _finite(number: str) -> float:
    """Reads a number with a fraction or an exponent, which ``json.loads`` calls this with, as a float.

    Raises:
        NodeError: ``not_json`` if the number is too large for a float, such as ``1e999``.
    """
    read = float(number)
    if not math.isfinite(read):
        raise NodeError(
            f'The text has the number {number}, which is too large for a float. Write it as a text instead: "{number}".',
            code="not_json",
        )
    return read


@registry.node(
    "json-stringify",
    title="Write JSON",
    description="Writes a value out as JSON text.",
    policy=Policy(costs_nothing=True),
)
def stringify(
    value: JsonValue, indent: Annotated[int, Param(ge=0, le=8)] = 0, sort_keys: bool = False
) -> Annotated[str, Param(title="JSON")]:
    """Writes a value out as JSON text, all on one line unless ``indent`` is set.

    Raises:
        NodeError: ``not_json`` if the value holds a number that isn't finite, such as a ``math-multiply`` result
            too large for a float.
    """
    try:
        return json.dumps(value, indent=indent or None, sort_keys=sort_keys, allow_nan=False)
    except ValueError:
        raise NodeError(
            "The value is not JSON: it holds NaN or Infinity, a number JSON can't write. Change the node that made it.",
            code="not_json",
        ) from None


@registry.node(
    "json-get",
    title="Get from JSON",
    description="Picks one value out of JSON by its path, such as user.name. It gives null when nothing is there.",
    policy=Policy(costs_nothing=True),
)
def get_path(value: JsonValue, path: str) -> Annotated[JsonValue, Param(title="Extracted")]:
    """Picks one value out of a JSON value by a dotted path, such as ``user.name`` or ``items.0.id``.

    A number picks an item from a list, and ``-1`` is the last one. An empty path gives the whole value. If
    nothing is at the path, the node returns ``None``, which is JSON's null.
    """
    current = value
    for segment in path.split(".") if path else ():
        if isinstance(current, dict):
            current = current.get(segment.strip())
        elif (
            isinstance(current, list)
            and re.fullmatch(r"-?[0-9]+", segment.strip())
            and -len(current) <= int(segment) < len(current)
        ):
            current = current[int(segment)]
        else:
            return None
    return current
