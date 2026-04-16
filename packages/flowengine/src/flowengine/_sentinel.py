"""SKIPPED sentinel for conditional branch propagation."""

from typing import Any


class _SkippedType:
    """Sentinel value indicating an output branch was not taken.

    Used by conditional nodes (If, Switch) to mark inactive branches.
    When a node receives only SKIPPED inputs, it is also skipped.
    """

    _instance: "_SkippedType | None" = None

    def __new__(cls) -> "_SkippedType":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "SKIPPED"

    def __bool__(self) -> bool:
        return False


SKIPPED = _SkippedType()


def is_skipped(value: Any) -> bool:
    """Check if a value is the SKIPPED sentinel."""
    return value is SKIPPED
