"""``FieldValueError``: raised when a value is not valid for its type."""

from __future__ import annotations

from typing import Any


class FieldValueError(Exception):
    """A value its type refuses.

    Attributes:
        reason: Why, quoting nothing of the value.
        errors: pydantic's account of the refusal without the value; ``None`` when pydantic didn't refuse it.
        column: The column of a table the first fault is in, else ``None``.
    """

    def __init__(self, reason: str, *, errors: list[dict[str, Any]] | None = None, column: str | None = None) -> None:
        super().__init__(reason)
        self.reason, self.errors, self.column = reason, errors, column
