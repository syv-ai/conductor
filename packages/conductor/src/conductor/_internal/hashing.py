"""Short hashes, and JSON that hashes the same whatever order its keys are in."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import pydantic_core


def short_hash(data: bytes, size: int) -> str:
    """The BLAKE2b hash of ``data`` in ``size`` bytes, as ``2 * size`` hex digits.

    ``short_hash(b"shout", 4)`` returns ``"af148f0d"``. Each caller picks a size for how many values it tells
    apart; a saved run contains these hashes, so changing a size changes what a saved run is compared with.
    """
    return hashlib.blake2b(data, digest_size=size).hexdigest()


def canonical_json(value: Any, fallback: Callable[[Any], Any] | None = None) -> bytes:
    """``value`` as JSON with the keys of every object sorted.

    ``canonical_json({"b": 1, "a": 2})`` returns ``b'{"a":2,"b":1}'``. A JSON store may give an object back with
    its keys in another order; hashing this text gives the same hash either way. A value with no dict in it has no
    keys to sort, and its callers hash pydantic's own JSON, which is faster.

    Args:
        value: What to write.
        fallback: Called for a value pydantic can't write as JSON; it returns what to write in its place.
    """
    plain = pydantic_core.to_jsonable_python(value, fallback=fallback)
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
