"""``REQUIRED`` and ``SKIPPED``: two values that stand for "no value", each with its own meaning.

``REQUIRED`` is the default of an input whose parameter has none. ``SKIPPED`` is the value of an output on a branch
that was not chosen: a node ``route(line)`` with the branches ``short`` and ``long`` returns ``long=SKIPPED`` for
the line ``"a"``.

Each is a ``typing_extensions.sentinel`` (PEP 661), as pydantic's ``MISSING`` is: it prints as its name, a copy or
a pickle of it is the same object, and code tests for it with ``is``.
"""

from __future__ import annotations

from typing import Any

from typing_extensions import sentinel

#: The default of an input whose parameter has no default. Its value must come from an edge, a constant or the caller.
REQUIRED = sentinel("REQUIRED")

#: The value of an output on a branch that was not chosen. Typed ``Any`` so that a type checker accepts it as the
#: value of an output of any type.
SKIPPED: Any = sentinel("SKIPPED")
