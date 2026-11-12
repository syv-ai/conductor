"""``REQUIRED`` and ``SKIPPED``: two values that stand for "no value", each with its own meaning.

``REQUIRED`` is the default of an input whose parameter has none. ``SKIPPED`` is the value of an output on a branch
that was not chosen: a node ``route(line)`` with the branches ``short`` and ``long`` returns ``long=SKIPPED`` for
the line ``"a"``.
"""

from __future__ import annotations

from typing import Any


class _Marker:
    """One named value: ``REQUIRED`` or ``SKIPPED``. It is falsy, and a copy or a pickle of it is the same object.

    Attributes:
        name: The name of the module-level constant, which is also its repr.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __reduce__(self) -> str:
        return self.name

    def __bool__(self) -> bool:
        return False


#: The default of an input whose parameter has no default. Its value must come from an edge, a constant or the caller.
REQUIRED: Any = _Marker("REQUIRED")

#: The value of an output on a branch that was not chosen. Typed ``Any`` so that a type checker accepts it as the
#: value of an output of any type.
SKIPPED: Any = _Marker("SKIPPED")
