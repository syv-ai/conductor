"""A Python value as plain JSON data, and as JSON text that hashes the same whatever order its keys are in."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pydantic_core
from pydantic import JsonValue
from pydantic_core import from_json, to_json


def plain_json(value: Any, fallback: Callable[[Any], Any] | None = None) -> JsonValue:
    """``value`` as what reading its JSON text gives: dicts, lists, texts, numbers, booleans and ``None``.

    ``plain_json({"day": date(2026, 1, 2), "sizes": (1, float("inf"))})`` returns
    ``{"day": "2026-01-02", "sizes": [1, "Infinity"]}``. A float that is not finite is written as its text
    (``"NaN"``, ``"Infinity"``, ``"-Infinity"``), since JSON has no number for it.

    Args:
        value: What to write.
        fallback: Called for a value pydantic cannot write as JSON; it returns what to write in its place.

    Raises:
        PydanticSerializationError: A value has no JSON form and there is no ``fallback``, or ``fallback`` raised.
    """
    plain: JsonValue = from_json(to_json(value, inf_nan_mode="strings", fallback=fallback))
    return plain


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
