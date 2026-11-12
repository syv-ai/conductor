"""``ConductorModel``: the base class of every record a host saves or sends as JSON.

A graph record, a problem, an event, a question waiting for a person and a widget are each a frozen pydantic model
on this base. Each is written with pydantic's ``model_dump_json`` and read back with ``model_validate_json``. What
only the engine uses, such as a compiled graph or an outcome, is a dataclass instead: those are made far more often
than they are read from JSON, and a dataclass is faster to make.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict


class ConductorModel(BaseModel):
    """A frozen pydantic model that a host writes as JSON and reads back. A field it does not declare is refused."""

    #: A float that is not finite is written as ``"NaN"``, ``"Infinity"`` or ``"-Infinity"``, as a saved run writes one.
    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="strings")

    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        """The fields pydantic's repr shows, without those that have their default, as scikit-learn prints an estimator.

        A repr reads ``Textarea(rows=8)``, and ``Textarea()`` when ``rows`` has its default, 4.
        """
        fields = type(self).model_fields
        for name, value in super().__repr_args__():
            field = fields.get(name) if name is not None else None
            if field is not None and not field.is_required():
                default = field.get_default(call_default_factory=True)
                if value is default or (default is not None and value == default):  # a frame's == compares cell by cell, so ``is`` is tried first
                    continue
            yield name, value
