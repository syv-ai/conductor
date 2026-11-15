"""``REQUIRED`` and ``SKIPPED``, two special values that each mean a different kind of "no value".

``REQUIRED`` is the default of an input that has no default, so its value has to come from somewhere else.
``SKIPPED`` is the value of an output on a branch that wasn't chosen. Say a node ``route(line)`` has two branches,
``short`` and ``long``. For the line ``"a"`` it returns ``long=SKIPPED``, and the nodes wired to ``long`` are
skipped for that line.

Both are made with ``typing_extensions.sentinel`` (PEP 661), like pydantic's ``MISSING``. Each prints as its name,
stays the same object when it is copied or pickled, and is tested with ``is``.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import sentinel

#: The default of an input whose parameter has no default. Its value must come from an edge, a constant or the caller.
REQUIRED = sentinel("REQUIRED")

#: The value of an output on a branch that wasn't chosen. It is typed ``Any`` so that a type checker accepts it as
#: the value of an output of any type.
SKIPPED: Any = sentinel("SKIPPED")
