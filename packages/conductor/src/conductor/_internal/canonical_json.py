"""JSON to hash: one way of writing a value whatever order its keys came back from a store in."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pydantic_core


def canonical_json(value: Any, fallback: Callable[[Any], Any] | None = None) -> bytes:
    """``value`` as JSON to hash, with every object's keys sorted, so a value a JSON store gave back in another key order is written alike.

    Every hash whose keys may come back from a store is written here. A hash of values that hold no mapping (a reuse
    fingerprint of a node whose inputs can't, a question whose inputs are plain) is their plain JSON, hashed as it
    is: there are no keys to sort, and those hashes are made on every call.
    """
    plain = pydantic_core.to_jsonable_python(value, fallback=fallback)
    return json.dumps(plain, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
