"""A Python value as plain JSON data."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

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
