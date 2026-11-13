"""Short hashes, of bytes and of a value's JSON text."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from pydantic_core import to_jsonable_python


def short_hash(data: bytes, size: int) -> str:
    """The BLAKE2b hash of ``data`` in ``size`` bytes, as ``2 * size`` hex digits.

    ``short_hash(b"shout", 4)`` returns ``"af148f0d"``. The caller picks the size: more bytes make it less likely
    that two different values get the same hash. Saved runs contain these hashes, so a caller that changes its size
    no longer matches the hashes in runs saved before.
    """
    return hashlib.blake2b(data, digest_size=size).hexdigest()


def json_hash(value: Any, size: int, *, fallback: Callable[[Any], Any] | None = None) -> str:
    """The ``short_hash`` of ``value``'s JSON text, in ``size`` bytes, with the keys of every dict sorted.

    ``json_hash({"b": 1, "a": 2}, 4)`` hashes ``b'{"a":2,"b":1}'``. A database may give a JSON object back with its
    keys in another order, so the keys are sorted, by ``json.dumps(sort_keys=True)``, and a value read back from
    a store hashes as it did before it was saved.

    Args:
        value: What to hash.
        size: The size of the hash in bytes.
        fallback: Called for a value pydantic cannot write as JSON; it returns what to write in its place.

    Raises:
        PydanticSerializationError: A value has no JSON form, and ``fallback`` gave none.
    """
    text = json.dumps(to_jsonable_python(value, fallback=fallback), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return short_hash(text.encode(), size)
