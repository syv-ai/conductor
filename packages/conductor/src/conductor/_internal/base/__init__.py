"""The building blocks every package of conductor uses; nothing here imports anything of conductor's.

``model.py`` is ``ConductorModel``, the base class of every record a host saves or sends as JSON; ``address.py``
names a place in a graph or a run, as tuples and as text; ``json.py`` the rules of JSON for every value (how a
float that is not finite is written, a value as plain JSON, a Python value validated as its JSON text is);
``hashing.py`` is ``short_hash`` and ``json_hash``, the one way a value is hashed; ``text.py`` the English messages are written with; ``sentinels.py``
is ``REQUIRED`` and ``SKIPPED``.
"""

from .address import (
    Address,
    ItemIndexes,
    Loc,
    Name,
    Path,
    is_name,
    loc_text,
    node_path_text,
    parse_loc_text,
    parse_node_path,
    place_text,
)
from .hashing import json_hash, short_hash
from .json import NON_FINITE_FLOATS, plain_json, validate_as_json
from .model import ConductorModel
from .sentinels import REQUIRED, SKIPPED
from .text import comma_separated, plural, qualified_name
