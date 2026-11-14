"""The building blocks every package of conductor uses; nothing here imports anything of conductor's but ``utils``.

``model.py`` is ``ConductorModel``, the base class of every record a host saves or sends as JSON; ``place.py``
names a place in a graph or a run, as tuples and as text; ``json.py`` the rules of JSON for every value (how a
float that is not finite is written, a value as plain JSON, the JSON text two values are compared by, a Python value
validated as its JSON text is); ``hashing.py`` is ``short_hash`` and ``json_hash``, the one way a value is hashed;
``sentinels.py`` is ``REQUIRED`` and ``SKIPPED``.
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
