"""Short hashes, of bytes and of a value's JSON text.

Saved runs store some of these hashes, such as the fingerprint in a waiting call's id. So a value must get the same
hash every time, in every process.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from pydantic_core import to_jsonable_python

from .json import JSON_CONFIG, WRITE_BY_NAME, canonical_json


def short_hash(data: bytes, size: int) -> str:
    """Returns a BLAKE2b hash of ``data`` that is ``size`` bytes long, written as hex.

    ``short_hash(b"shout", 4)`` returns ``"af148f0d"``, eight hex digits. A bigger size makes it less likely that
    two different values share a hash.

    Saved runs contain these hashes. If a caller changes the size it asks for, its new hashes won't match the ones
    in runs saved before.
    """
    return hashlib.blake2b(data, digest_size=size).hexdigest()


def json_hash(value: Any, size: int, *, fallback: Callable[[Any], Any] | None = None) -> str:
    """Returns the ``short_hash`` of a value's JSON text.

    The text has the keys of every dict sorted (``canonical_json``), and each field of a host's model or dataclass
    under its name (``WRITE_BY_NAME``), so an alias never changes a hash. That matters because a database may give a
    JSON object back with its keys in another order. With the keys sorted, a value read back from a store hashes
    the same as it did before it was saved. So ``json_hash({"b": 1, "a": 2}, 4)`` hashes ``b'{"a":2,"b":1}'``.

    Args:
        value: The value to hash.
        size: The length of the hash, in bytes.
        fallback: Called with any value pydantic can't write as JSON. It returns what to write instead.

    Raises:
        PydanticSerializationError: A value can't be written as JSON, and ``fallback`` gave nothing to write instead.
    """
    written = to_jsonable_python(
        value, fallback=fallback, bytes_mode=JSON_CONFIG["ser_json_bytes"], **WRITE_BY_NAME
    )
    return short_hash(canonical_json(written).encode(), size)
