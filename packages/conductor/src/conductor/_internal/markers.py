"""``REQUIRED`` and ``SKIPPED``, the two marker values: ``REQUIRED`` is the default of an input that has no default,
and ``SKIPPED`` is what a node returns for an output on a branch it did not take.
"""

from __future__ import annotations

from typing import Any


class _Marker:
    """One named value that is falsy and stays itself through a copy or a pickle: ``REQUIRED`` or ``SKIPPED``.

    Attributes:
        name: The module-level name the value is kept under, which is also its repr.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __reduce__(self) -> str:
        return self.name

    def __bool__(self) -> bool:
        return False


#: The default of an input nothing defaults.
REQUIRED: Any = _Marker("REQUIRED")

#: The value of an output on a branch its node did not take. Typed ``Any``
#: for a checker: it goes where a value would, on any output.
SKIPPED: Any = _Marker("SKIPPED")
