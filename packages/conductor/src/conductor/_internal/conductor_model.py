"""``ConductorModel``: the base of every record a host saves or sends.

A graph a host stores, a problem it shows, an event it streams, a row
waiting for a person: each crosses into JSON, so each is a frozen pydantic
model on this one base, and reads back what it wrote with pydantic's own
``model_dump_json`` and ``model_validate_json``. A widget reads back as
written too. What loading and the engine build and keep to themselves (a
loaded graph, what a run holds of each unit) is not a model: those are dataclasses, built
far more often than they are ever parsed.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Self

from pydantic import BaseModel, ConfigDict


class ConductorModel(BaseModel):
    """A frozen record that writes itself as JSON and reads back what a host saves."""

    #: A float that isn't finite is written as ``"NaN"``, ``"Infinity"`` or ``"-Infinity"``, as a run writes one.
    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="strings")

    @classmethod
    def _built_from_checked_fields(cls, **fields: Any) -> Self:
        """A record from values that already passed, every field given: what ``model_construct``
        does without its walk over the fields. A field left out is absent, not defaulted."""
        record = cls.__new__(cls)
        object.__setattr__(record, "__dict__", fields)
        object.__setattr__(record, "__pydantic_fields_set__", set(fields))
        object.__setattr__(record, "__pydantic_extra__", None)
        object.__setattr__(record, "__pydantic_private__", None)
        return record

    def __repr_args__(self) -> Iterator[tuple[str | None, Any]]:
        """Pydantic's repr arguments without the fields at their default, as scikit-learn prints an estimator.

        ``Input(name='text', dtype=Text, title='Text', widget=Textarea())``
        rather than every ``None`` and ``True`` the record carries.
        """
        fields = type(self).model_fields
        for name, value in super().__repr_args__():
            field = fields.get(name) if name is not None else None
            if field is not None and not field.is_required():
                default = field.get_default(call_default_factory=True)
                if value is default or (default is not None and value == default):  # a frame compares by cell
                    continue
            yield name, value
