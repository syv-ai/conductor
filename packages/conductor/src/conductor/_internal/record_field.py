"""Reading one named field of a record, as an edge such as ``split.result.name`` does."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def field_of(value: Any, name: str) -> Any:
    """One field of a record: a mapping's key, else an attribute; of records held by column, that column.

    Raises:
        NoSuchFieldError: The record has no such field. Only a record whose type declares no fields can lack
            one: a field a type declares is checked when the graph loads.
    """
    try:
        return value[name] if type(value) is dict or isinstance(value, Mapping) else getattr(value, name)
    except (KeyError, AttributeError):
        raise NoSuchFieldError(name) from None


class NoSuchFieldError(Exception):
    """A record an edge brings has no field by the name the edge reads.

    Attributes:
        field: The name.
    """

    def __init__(self, field: str) -> None:
        super().__init__(field)
        self.field = field
