"""How conductor writes values as JSON, whatever their type.

Every value conductor saves or sends follows the same few rules, and they are written once, here. JSON has no
number for ``inf`` or ``nan``, so conductor writes them as text. Two values are compared and hashed by one fixed
JSON text. And a value given from Python is checked exactly as if it had come in as JSON, so it gives the same
result as the same value sent over HTTP.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Final, Literal

from pydantic import JsonValue, TypeAdapter
from pydantic_core import (
    PydanticCustomError,
    PydanticSerializationError,
    SchemaValidator,
    from_json,
    to_json,
    to_jsonable_python,
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
    the JSON text and reads it back, and ``inf`` comes out as the text ``"Infinity"`` (``NON_FINITE_FLOATS``), in a
    field of a host's own model too (``json_text``).

    Args:
        value: The value to convert.
        fallback: Called with any value pydantic can't write as JSON. It returns what to write instead.

    Raises:
        PydanticSerializationError: A value can't be written as JSON and there is no ``fallback``, or ``fallback``
            raised.
    """
    plain: JsonValue = from_json(_written(value, NON_FINITE_FLOATS, fallback))
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
    JSON text first (``json_text``), and then the text is checked.

    Args:
        validator: Pydantic's validator for the type.
        value: The value to check.
        fallback: As ``json_text`` takes it.

    Returns:
        The value as ``validator`` returns it.

    Raises:
        PydanticSerializationError: As ``json_text`` raises it.
        PydanticCustomError: ``fallback`` refused the value.
        ValidationError: The type refuses the value.
    """
    return validator.validate_json(json_text(value, fallback), strict=True)


def json_text(value: Any, fallback: Callable[[Any], Any] | None = None) -> str:
    """Writes a Python value as the JSON text it is checked from (``validate_as_json``).

    ``inf`` and ``nan`` are written as the bare words ``Infinity`` and ``NaN``, which pydantic reads back as floats.

    Args:
        value: The value to write.
        fallback: Called with any value pydantic can't write as JSON. It returns what to write instead, or raises
            ``PydanticCustomError`` to refuse the value. That error is raised from here unchanged.

    Raises:
        PydanticSerializationError: The value can't be written as JSON, and ``fallback`` gave nothing to write
            instead.
        PydanticCustomError: ``fallback`` refused the value.
    """
    return _written(value, "constants", fallback).decode()


def _written(
    value: Any, non_finite: Literal["constants", "strings"], fallback: Callable[[Any], Any] | None
) -> bytes:
    """Writes ``value`` as JSON text, with ``inf`` and ``nan`` written as ``non_finite`` says, wherever they are.

    A pydantic model writes its own floats by its own config, whatever ``to_json`` is told, and its config writes
    ``inf`` as ``null`` by default. So a host's model with ``x=inf`` would be written as
    ``{"x": null}``, and a ``float | None`` field would read back as ``None``. The value is turned into Python's
    JSON values first (``to_jsonable_python``), which keep a float as it is, each model by its own serializer, and
    then written as text by one rule.

    What can't be written is refused as ``to_json`` refuses it, with ``PydanticSerializationError``: a cycle, bytes
    that aren't text, or a value ``fallback`` raises ``TypeError`` for. A ``PydanticCustomError`` that ``fallback``
    raises to refuse a value is raised unchanged.
    """
    try:
        data = to_jsonable_python(value, fallback=fallback)
    except (PydanticCustomError, PydanticSerializationError):
        raise
    # what to_json raises as PydanticSerializationError, to_jsonable_python raises as it is
    except (TypeError, ValueError) as error:
        raise PydanticSerializationError(str(error)) from None
    return to_json(data, inf_nan_mode=non_finite)
