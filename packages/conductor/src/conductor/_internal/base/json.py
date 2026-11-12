"""The rules of JSON that hold for every value conductor writes or reads, whatever its type.

``NON_FINITE_FLOATS`` is how all JSON conductor writes holds a float that is not finite; ``plain_json`` writes a
Python value as the plain data its JSON text reads back as; ``validate_as_json`` validates a Python value as
pydantic validates its JSON text; ``canonical_json`` writes a value as JSON text with its keys sorted, to hash.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Final

import pydantic_core
from pydantic import JsonValue, TypeAdapter
from pydantic_core import SchemaValidator, from_json, to_json

#: How all JSON conductor writes holds ``inf``, ``-inf`` and ``nan``, which JSON has no number for: as the texts
#: ``"Infinity"``, ``"-Infinity"`` and ``"NaN"``. A ``float`` reads each back as the float; a ``str`` keeps the text.
#: Every pydantic config conductor writes JSON by reads it from here (``ser_json_inf_nan=NON_FINITE_FLOATS``).
NON_FINITE_FLOATS: Final = "strings"


def plain_json(value: Any, fallback: Callable[[Any], Any] | None = None) -> JsonValue:
    """``value`` as what reading its JSON text gives: dicts, lists, texts, numbers, booleans and ``None``.

    ``plain_json({"day": date(2026, 1, 2), "sizes": (1, float("inf"))})`` returns
    ``{"day": "2026-01-02", "sizes": [1, "Infinity"]}``. A float that is not finite is written as its text
    (``NON_FINITE_FLOATS``). Pydantic's ``to_jsonable_python`` keeps such a float as a float, so the value is
    written as JSON text and the text read back.

    Args:
        value: What to write.
        fallback: Called for a value pydantic cannot write as JSON; it returns what to write in its place.

    Raises:
        PydanticSerializationError: A value has no JSON form and there is no ``fallback``, or ``fallback`` raised.
    """
    plain: JsonValue = from_json(to_json(value, inf_nan_mode=NON_FINITE_FLOATS, fallback=fallback))
    return plain


def validate_as_json(validator: SchemaValidator | TypeAdapter[Any], value: Any, fallback: Callable[[Any], Any] | None = None) -> Any:
    """Validates the Python value ``value`` strictly, as ``validator`` validates the same value sent as JSON text.

    For a ``bool``, ``True`` is read and ``"yes"`` refused; for a ``date``, both ``date(2026, 1, 2)`` and
    ``"2026-01-02"`` are read; for a ``tuple``, a list is read. So a value a host gives from Python reads the same
    as that value sent over HTTP. Pydantic has no call that validates a Python value by the rules of JSON input:
    ``validate_python(strict=True)`` refuses a date's text, and ``validate_strings`` reads ``"yes"`` as ``True``.
    So the value is written as JSON text and the text validated. A float that is not finite is written as the
    bare word ``Infinity`` or ``NaN``, which pydantic reads back as the float.

    Args:
        validator: Pydantic's validator of the type.
        value: The value to validate.
        fallback: Called for a value pydantic cannot write as JSON; it returns what to write in its place.

    Raises:
        PydanticSerializationError: The value has no JSON form, and ``fallback`` gave none.
        ValidationError: The type refuses the value.
    """
    return validator.validate_json(to_json(value, inf_nan_mode="constants", fallback=fallback), strict=True)


def canonical_json(value: Any, fallback: Callable[[Any], Any] | None = None) -> bytes:
    """``value`` as JSON with the keys of every object sorted.

    ``canonical_json({"b": 1, "a": 2})`` returns ``b'{"a":2,"b":1}'``. A database may give a JSON object back with
    its keys in another order; hashing this text gives the same hash either way. For a value with no dict in it,
    callers hash pydantic's own JSON instead, which is faster.

    Args:
        value: What to write.
        fallback: Called for a value pydantic cannot write as JSON; it returns what to write in its place.
    """
    plain = pydantic_core.to_jsonable_python(value, fallback=fallback)
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
