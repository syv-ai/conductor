"""Reading one field of a record, as an edge from ``read.result.name`` does: the field ``name`` of each invoice."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def field_of(value: Any, name: str) -> Any:
    """One field of a record: the key ``name`` of a mapping, else the attribute ``name``.

    ``field_of({"name": "Ida", "amount": 100}, "name")`` returns ``"Ida"``.

    Raises:
        NoSuchFieldError: The record has no such field. This can happen only for a record whose type declares no
            fields: a field that a type declares is checked when the graph is compiled.
    """
    try:
        return value[name] if type(value) is dict or isinstance(value, Mapping) else getattr(value, name)
    except (KeyError, AttributeError):
        raise NoSuchFieldError(name) from None


class NoSuchFieldError(Exception):
    """A record has no field with the name an edge reads.

    Attributes:
        field: The name of the field.
    """

    def __init__(self, field: str) -> None:
        super().__init__(field)
        self.field = field
