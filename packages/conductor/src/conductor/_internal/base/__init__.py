"""The small building blocks the rest of conductor is made from.

None of them knows how a graph runs. They are the base class for records a host stores as JSON, the names of
places in a graph, how values are written and hashed as JSON, and two special values. Every package may use them,
so this one imports nothing from conductor except ``utils``.

- ``model.py``: ``ConductorModel``, the base class of every record a host saves or sends as JSON.
- ``place.py``: how a node, an output or one call of a node is named, as a tuple and as text, and the id of
  a waiting call.
- ``json.py``: how any value is written as JSON, compared by its JSON, and checked as if it came in as JSON.
- ``hashing.py``: short hashes, of bytes and of a value's JSON text.
- ``sentinels.py``: ``REQUIRED`` and ``SKIPPED``, two values that each mean a different kind of "no value".
"""

from .hashing import json_hash, short_hash
from .json import (
    NON_FINITE_FLOATS,
    canonical_json,
    finite_json_text,
    json_text,
    plain_json,
    validate_as_json,
)
from .model import ConductorModel
from .place import (
    FIELD_SEPARATOR,
    SEPARATORS,
    ItemIndexes,
    Loc,
    Name,
    NodePath,
    Source,
    is_field,
    is_name,
    parse_loc,
    parse_node_path,
    parse_waiting_id,
    place_text,
    write_waiting_id,
)
from .sentinels import REQUIRED, SKIPPED

__all__ = [
    "FIELD_SEPARATOR",
    "NON_FINITE_FLOATS",
    "REQUIRED",
    "SEPARATORS",
    "SKIPPED",
    "ConductorModel",
    "ItemIndexes",
    "Loc",
    "Name",
    "NodePath",
    "Source",
    "canonical_json",
    "finite_json_text",
    "is_field",
    "is_name",
    "json_hash",
    "json_text",
    "parse_loc",
    "parse_node_path",
    "parse_waiting_id",
    "place_text",
    "write_waiting_id",
    "plain_json",
    "short_hash",
    "validate_as_json",
]
