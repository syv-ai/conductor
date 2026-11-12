"""The building blocks every package of conductor uses; nothing here imports anything of conductor's.

``model.py`` is ``ConductorModel``, the base class of every record a host saves or sends as JSON; ``address.py``
names a place in a graph or a run, as tuples and as text; ``json.py`` the rules of JSON for every value (how a
float that is not finite is written, a value as plain JSON, a Python value validated as its JSON text is, JSON
text to hash); ``hashing.py`` is ``short_hash``; ``text.py`` the English messages are written with; ``sentinels.py``
is ``REQUIRED`` and ``SKIPPED``.
"""

from .address import (
    NAME_PATTERN,
    ItemIndexes,
    Loc,
    NodePath,
    is_name,
    loc_text,
    node_path_text,
    parse_loc_text,
    parse_node_path,
    split_address,
)
from .hashing import short_hash
from .json import NON_FINITE_FLOATS, canonical_json, plain_json, validate_as_json
from .model import ConductorModel
from .sentinels import REQUIRED, SKIPPED
from .text import comma_separated, plural, qualified_name
