"""The small building blocks the rest of conductor is made from.

None of them knows how a graph runs. They are the base class for models a host stores as JSON, the names of
places in a graph, how values are written and hashed as JSON, two special values and the base class of every
validator. Every package may use them, so this one imports nothing from conductor except ``utils``.

- ``model.py``: ``ConductorModel``, the base class of every model a host saves or sends as JSON.
- ``place.py``: how a node, an output or one call of a node is named, as a tuple and as text, and the id of
  a waiting call.
- ``core_schema.py``: which parts of a pydantic core schema are the schemas inside it, and its ``definitions``.
- ``json.py``: how any value is written as JSON, compared by its JSON, and checked as if it came in as JSON, and
  the one walk over a value and everything inside it.
- ``hashing.py``: short hashes, of bytes and of a value's JSON text.
- ``sentinels.py``: ``REQUIRED`` and ``SKIPPED``, two values that each mean a different kind of "no value".
- ``validator.py``: ``Validator``, the base class of every check on a value that comes in from outside.
"""

from .core_schema import (
    HOSTS_OWN_CLASSES,
    looked_up,
    schemas_in,
    under_definitions,
    with_definitions,
    with_schemas_in,
)
from .hashing import json_hash, short_hash
from .json import (
    JSON_CONFIG,
    JSON_CORE_CONFIG,
    READ_BY_NAME,
    WRITE_BY_NAME,
    canonical_json,
    finite_json_text,
    holds_a_set,
    json_text,
    json_typed_dict_schema,
    plain_json,
    validate_as_json,
    value_parts,
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
from .validator import Validator

__all__ = [
    "FIELD_SEPARATOR",
    "HOSTS_OWN_CLASSES",
    "JSON_CONFIG",
    "JSON_CORE_CONFIG",
    "READ_BY_NAME",
    "REQUIRED",
    "SEPARATORS",
    "SKIPPED",
    "ConductorModel",
    "ItemIndexes",
    "Loc",
    "Name",
    "NodePath",
    "Source",
    "Validator",
    "WRITE_BY_NAME",
    "canonical_json",
    "finite_json_text",
    "holds_a_set",
    "is_field",
    "is_name",
    "json_hash",
    "json_text",
    "json_typed_dict_schema",
    "looked_up",
    "parse_loc",
    "parse_node_path",
    "parse_waiting_id",
    "place_text",
    "write_waiting_id",
    "plain_json",
    "schemas_in",
    "short_hash",
    "under_definitions",
    "validate_as_json",
    "value_parts",
    "with_definitions",
    "with_schemas_in",
]
