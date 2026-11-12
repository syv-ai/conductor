"""Short hashes."""

from __future__ import annotations

import hashlib


def short_hash(data: bytes, size: int) -> str:
    """The BLAKE2b hash of ``data`` in ``size`` bytes, as ``2 * size`` hex digits.

    ``short_hash(b"shout", 4)`` returns ``"af148f0d"``. The caller picks the size: more bytes make it less likely
    that two different values get the same hash. Saved runs contain these hashes, so a caller that changes its size
    no longer matches the hashes in runs saved before.
    """
    return hashlib.blake2b(data, digest_size=size).hexdigest()
