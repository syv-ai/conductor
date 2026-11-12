"""Short hashes, of bytes and of a value's JSON text."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from pydantic_core import to_json, to_jsonable_python


def short_hash(data: bytes, size: int) -> str:
    """The BLAKE2b hash of ``data`` in ``size`` bytes, as ``2 * size`` hex digits.

    ``short_hash(b"shout", 4)`` returns ``"af148f0d"``. The caller picks the size: more bytes make it less likely
    that two different values get the same hash. Saved runs contain these hashes, so a caller that changes its size
    no longer matches the hashes in runs saved before.
    """
    return hashlib.blake2b(data, digest_size=size).hexdigest()


def json_hash(value: Any, size: int, *, keys_sorted: bool, fallback: Callable[[Any], Any] | None = None) -> str:
    """The ``short_hash`` of ``value``'s JSON text, in ``size`` bytes.

    ``json_hash({"b": 1, "a": 2}, 4, keys_sorted=True)`` hashes ``b'{"a":2,"b":1}'``. A database may give a JSON
    object back with its keys in another order, so a value that can contain a dict is hashed with the keys of
    every dict sorted, by ``json.dumps(sort_keys=True)``: pydantic sorts no keys. A caller that knows the value
    contains no dict passes ``keys_sorted=False``, and pydantic writes the text, which is faster. The two write some
    numbers differently (``1e+16`` and ``1e16``), so a value hashed again later is hashed the same way each time.

    Args:
        value: What to hash.
        size: The size of the hash in bytes.
        keys_sorted: Whether the keys of every dict are sorted before hashing.
        fallback: Called for a value pydantic cannot write as JSON; it returns what to write in its place.

    Raises:
        PydanticSerializationError: A value has no JSON form, and ``fallback`` gave none.
    """
    if keys_sorted:
        text = json.dumps(to_jsonable_python(value, fallback=fallback), sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    else:
        text = to_json(value, fallback=fallback)
    return hashlib.blake2b(text, digest_size=size).hexdigest()  # short_hash's one line, written out: a call less on every waiting question
