"""The hash of what one unit is about to receive: what says whether two units are the same work.

A node that changed since the state a leg starts from runs its units again,
except a unit whose inputs match one its earlier units received: that unit
takes the earlier result and is not called. The inputs are exactly the mapping
the call gets, statics and wired values alike, after skips and blanks are
settled. A ``FromRun`` value is not one.

A table is hashed by what it holds, without ``__row``: where its rows sit is
not what a node received. A row inserted above leaves a whole-table reader's
hash as it was, and a reused unit's ids follow its inputs' rows by position.
"""

from __future__ import annotations

import json
import secrets
from collections.abc import Iterable
from hashlib import blake2b
from typing import Any

from conductor._internal.codec import to_jsonable
from conductor._internal.table import content_digest_of_table, is_frame

#: Digests of the values that aren't plain data, by ``id``; each kept with the value, so an id can't be reused while it is here.
DigestsOfValues = dict[int, tuple[Any, bytes]]


def hash_of_inputs(seed: bytes, values: Iterable[Any], digests_of_values: DigestsOfValues) -> str:
    """The hash of a unit receiving ``values``, on a node whose ``_input_hash_seed`` is ``seed``.

    A digest of a table, of a list or of any value with a JSON form is worked
    out once per object per leg, so a value that reaches every row is read once."""
    parts = [seed]
    for value in values:
        if isinstance(value, str):
            encoded = value.encode("utf-8", "surrogatepass")
            parts += (b"s%d:" % len(encoded), encoded)
        elif type(value) is int:
            parts.append(b"i%d;" % value)
        else:
            _add_value(parts, value, digests_of_values)
    return blake2b(b"".join(parts), digest_size=16).hexdigest()


def _add_value(parts: list[bytes], value: Any, digests_of_values: DigestsOfValues) -> None:
    """Append what identifies ``value`` to ``parts``, tagged so that no two values read the same."""
    if value is None:
        parts.append(b"n;")
    elif isinstance(value, str):
        encoded = value.encode("utf-8", "surrogatepass")
        parts += (b"s%d:" % len(encoded), encoded)
    elif isinstance(value, bool):
        parts.append(b"b1;" if value else b"b0;")
    elif isinstance(value, int):
        parts.append(b"i%d;" % value)
    elif isinstance(value, float):
        parts.append(b"f%a;" % value)
    elif isinstance(value, list):
        parts.append(b"[%d:" % len(value))
        for item in value:
            _add_value(parts, item, digests_of_values)
    else:
        parts.append(_digest_of_value(value, digests_of_values))


def _digest_of_value(value: Any, digests_of_values: DigestsOfValues) -> bytes:
    """A table by the digest a state files it under; any other value by its JSON form.

    A value no type can write has no form to compare, so it gets a hash nothing
    else will ever have and the unit always runs."""
    known = digests_of_values.get(id(value))
    if known is not None:
        return known[1]
    try:
        digest = b"t" + content_digest_of_table(value).encode() if is_frame(value) else b"j" + blake2b(_json_form(value), digest_size=16).digest()
    except Exception:  # whatever a value's own writer raises means only one thing here: nothing to compare
        digest = b"x" + secrets.token_bytes(16)
    digests_of_values[id(value)] = (value, digest)
    return digest


def _json_form(value: Any) -> bytes:
    return json.dumps(to_jsonable(value), sort_keys=True, ensure_ascii=False).encode("utf-8", "surrogatepass")
