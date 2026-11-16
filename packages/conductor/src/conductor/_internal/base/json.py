"""How conductor writes values as JSON, whatever their type.

Every value conductor saves or sends follows the same few rules, and they are written once, here. JSON has no
number for ``inf`` or ``nan``, so conductor writes them as text. Two values are compared and hashed by one fixed
JSON text. And a value given from Python is checked exactly as if it had come in as JSON, so it gives the same
result as the same value sent over HTTP.
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable
from typing import Any, Final

from pydantic import JsonValue, TypeAdapter
from pydantic_core import (
    PydanticCustomError,
    PydanticSerializationError,
    SchemaValidator,
    from_json,
    to_json,
)

#: How conductor writes ``inf``, ``-inf`` and ``nan`` in JSON, which has no number for them. They are written as
#: the texts ``"Infinity"``, ``"-Infinity"`` and ``"NaN"``. A ``float`` reads each one back as the float, and a
#: ``str`` keeps the text. Every pydantic config in conductor that writes JSON uses it, as
#: ``ser_json_inf_nan=NON_FINITE_FLOATS``.
NON_FINITE_FLOATS: Final = "strings"


def plain_json(value: Any, fallback: Callable[[Any], Any] | None = None) -> JsonValue:
    """Returns ``value`` as plain JSON data: what you get by writing it as JSON text and reading that back.

    The result has only dicts, lists, strings, numbers, booleans and ``None``. For example,
    ``plain_json({"day": date(2026, 1, 2), "sizes": (1, float("inf"))})`` returns
    ``{"day": "2026-01-02", "sizes": [1, "Infinity"]}``.

    Pydantic's ``to_jsonable_python`` would leave ``inf`` as a float, which JSON has no number for. So this writes
    the JSON text and reads it back, and ``inf`` comes out as the text ``"Infinity"`` (``NON_FINITE_FLOATS``).

    Args:
        value: The value to convert.
        fallback: Called with any value pydantic can't write as JSON. It returns what to write instead.

    Raises:
        PydanticSerializationError: A value can't be written as JSON and there is no ``fallback``, or ``fallback``
            raised.
    """
    plain: JsonValue = from_json(to_json(value, inf_nan_mode=NON_FINITE_FLOATS, fallback=fallback))
    return plain


def canonical_json(value: Any) -> str:
    """Returns one fixed JSON text for ``value``, so that two values can be compared and hashed by their text.

    The keys of every dict are sorted, and there are no spaces. So ``{"b": 1, "a": [2]}`` and
    ``{"a": [2], "b": 1}`` both give ``'{"a":[2],"b":1}'``. Two values are the same JSON when their texts are equal.

    ``value`` should already be plain data, as ``plain_json`` or pydantic's ``to_python(mode="json")`` returns it.
    The second leaves ``inf`` as a float, so a float that isn't finite is written as its text here too
    (``NON_FINITE_FLOATS``). That way ``float("inf")`` and ``"Infinity"`` give the same text.
    """
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        )
    except ValueError:  # inf or nan, which JSON has no number for
        return canonical_json(plain_json(value))


def validate_as_json(
    validator: SchemaValidator | TypeAdapter[Any],
    value: Any,
    fallback: Callable[[Any], Any] | None = None,
) -> Any:
    """Checks a Python value strictly, the way ``validator`` checks the same value sent as JSON text.

    This is what makes a value given from Python and the same value sent over HTTP come out the same. A ``bool``
    takes ``True`` but refuses ``"yes"``. A ``date`` takes both ``date(2026, 1, 2)`` and ``"2026-01-02"``. A
    ``tuple`` takes a list.

    Pydantic has no call that checks a Python value by the rules for JSON input. ``validate_python(strict=True)``
    refuses a date given as text, and ``validate_strings`` takes ``"yes"`` as ``True``. So the value is written as
    JSON text first, and then the text is checked. ``inf`` and ``nan`` are written as the bare words ``Infinity``
    and ``NaN``, which pydantic reads back as floats.

    Args:
        validator: Pydantic's validator for the type.
        value: The value to check.
        fallback: Called with any value pydantic can't write as JSON. It returns what to write instead, or raises
            ``PydanticCustomError`` to refuse the value. That error is raised from here unchanged.

    Returns:
        The value as ``validator`` returns it.

    Raises:
        PydanticSerializationError: The value can't be written as JSON, and ``fallback`` gave nothing to write
            instead.
        PydanticCustomError: ``fallback`` refused the value.
        ValidationError: The type refuses the value.
    """
    refused: list[PydanticCustomError] = []

    def written(part: Any, write: Callable[[Any], Any]) -> Any:
        try:
            return write(part)
        # Keep the refusal: to_json would turn it into a PydanticSerializationError.
        except PydanticCustomError as refusal:
            refused.append(refusal)
            raise

    try:
        text = to_json(
            value,
            inf_nan_mode="constants",
            fallback=None if fallback is None else functools.partial(written, write=fallback),
        )
    except PydanticSerializationError:
        if refused:
            raise refused[0] from None
        raise
    return validator.validate_json(text, strict=True)
