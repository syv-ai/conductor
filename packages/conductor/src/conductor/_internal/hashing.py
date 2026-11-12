"""Short hashes, and JSON that hashes the same whatever order its keys are in."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import pydantic_core


def short_hash(data: bytes, size: int) -> str:
    """The BLAKE2b hash of ``data`` in ``size`` bytes, as ``2 * size`` hex digits.

    ``short_hash(b"shout", 4)`` returns ``"af148f0d"``. The caller picks the size: more bytes make it less likely
    that two different values get the same hash. Saved runs contain these hashes, so a caller that changes its size
    no longer matches the hashes in runs saved before.
    """
    return hashlib.blake2b(data, digest_size=size).hexdigest()


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
