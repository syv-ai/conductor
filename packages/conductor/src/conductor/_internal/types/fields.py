"""Getting one field out of a record.

An edge can pick one field of the record an output returns. Say a node ``read`` returns invoices, each with a
field ``name``. An edge from ``read.result.name`` passes on just the name of each invoice. ``field_of`` is how
that field is fetched.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def field_of(value: Any, name: str) -> Any:
    """Returns one field of a record: the key ``name`` of a mapping, or else the attribute ``name``.

    For example, ``field_of({"name": "Ida", "amount": 100}, "name")`` returns ``"Ida"``.

    Raises:
        NoSuchFieldError: The record has no such field. This can only happen when the record's type declares no
            fields, like ``dict``. A field a type declares is checked when the graph is compiled.
    """
    try:
        return (
            value[name]
            if type(value) is dict or isinstance(value, Mapping)
            else getattr(value, name)
        )
    except (KeyError, AttributeError):
        raise NoSuchFieldError(name) from None


class NoSuchFieldError(Exception):
    """A record has no field with the name an edge picks.

    Attributes:
        field: The name of the field.
    """

    def __init__(self, field: str) -> None:
        super().__init__(field)
        self.field = field
