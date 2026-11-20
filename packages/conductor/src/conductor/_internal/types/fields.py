"""Getting one field out of a value.

An edge can pick one field of the value an output returns. Say a node ``read`` returns invoices, each with a
field ``name``. An edge from ``read.result.name`` passes on just the name of each invoice. ``field_of`` is how
that field is fetched.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from conductor._internal.base import SKIPPED
from conductor._internal.problem import conductor_error


def field_of(value: Any, name: str) -> Any:
    """Returns one field of a value: the key ``name`` of a mapping, or else the attribute ``name``.

    For example, ``field_of({"name": "Ida", "amount": 100}, "name")`` returns ``"Ida"``.

    A field that is ``SKIPPED`` is refused. Only a branch's whole value, or an item of a ``Series``, may be
    ``SKIPPED``, so an edge never skips the node it feeds because a field inside a value is ``SKIPPED``. Only the
    field read is looked at; the value isn't searched.

    Raises:
        UnreadableFieldError: The value has no such field, or the field is ``SKIPPED``. A value can lack a field
            only when its type declares no fields, like ``dict``: a field a type declares is checked when the graph is
            compiled.
    """
    try:
        field = (
            value[name]
            if type(value) is dict or isinstance(value, Mapping)
            else getattr(value, name)
        )
    except (KeyError, AttributeError):
        raise UnreadableFieldError(name) from None
    if field is SKIPPED:
        raise UnreadableFieldError(name, skipped=True)
    return field


class UnreadableFieldError(Exception):
    """The field an edge picks from a value can't be read: the value has no such field, or the field is ``SKIPPED``.

    Attributes:
        field: The name of the field.
        error: Why, as the one error of the problem the call fails with: ``no_such_field`` or ``skipped_inside``.
    """

    def __init__(self, field: str, *, skipped: bool = False) -> None:
        """Makes the error for the field ``field``, which is missing, or ``SKIPPED`` if ``skipped``."""
        super().__init__(field)
        self.field = field
        if skipped:
            self.error = conductor_error(
                "skipped_inside",
                "its field '{field}' is SKIPPED, which has no JSON form inside a value",
                {"field": field},
            )
        else:
            self.error = conductor_error(
                "no_such_field", "it has no field '{field}'", {"field": field}
            )
