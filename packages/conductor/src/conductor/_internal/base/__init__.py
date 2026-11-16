"""The small building blocks the rest of conductor is made from.

None of them knows how a graph runs. They are the base class for records a host stores as JSON, the names of
places in a graph, how values are written and hashed as JSON, and two special values. Every package may use them,
so this one imports nothing from conductor except ``utils``.

- ``model.py``: ``ConductorModel``, the base class of every record a host saves or sends as JSON.
- ``place.py``: how a node, an output or one call of a node is named, as a tuple and as text.
- ``json.py``: how any value is written as JSON, compared by its JSON, and checked as if it came in as JSON.
- ``hashing.py``: short hashes, of bytes and of a value's JSON text.
- ``sentinels.py``: ``REQUIRED`` and ``SKIPPED``, two values that each mean a different kind of "no value".
"""

from .hashing import json_hash, short_hash
from .json import NON_FINITE_FLOATS, canonical_json, plain_json, validate_as_json
from .model import ConductorModel
from .place import (
    ItemIndexes,
    Loc,
    Name,
    NodePath,
    Source,
    is_name,
    parse_node_path,
    place_text,
)
from .sentinels import REQUIRED, SKIPPED

__all__ = [
    "NON_FINITE_FLOATS",
    "REQUIRED",
    "SKIPPED",
    "ConductorModel",
    "ItemIndexes",
    "Loc",
    "Name",
    "NodePath",
    "Source",
    "canonical_json",
    "is_name",
    "json_hash",
    "parse_node_path",
    "place_text",
    "plain_json",
    "short_hash",
    "validate_as_json",
]
