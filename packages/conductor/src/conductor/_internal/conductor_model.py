"""``ConductorModel`` — the base of every record a host saves or sends.

A graph a host stores, a problem it shows, an input it renders, the cause
of a failure it reports: each crosses into JSON, so each is a frozen
pydantic model on this one base.

What a host saves reads back what it wrote — a ``Graph`` and its parts, a
``Problem``, a ``Policy``, an ``Index``: ``model_dump_json`` and
``model_validate_json``, pydantic's own.

What describes a node — ``NodeDescription``, ``VersionDescription``,
``Input``, ``Output`` and the widgets — is written for an editor. It is
built from a function's signature, so its ``dtype`` dumps as the type's
description, and a dump does not read back: a description is not a type,
and reading a record holding one raises. A widget reads back as written. The
class is still there to ``describe()``.

The events a leg yields are models too, since a host sends every one of
them on. What compile and the engine build and keep to themselves —
``CompiledGraph`` and its node and field views, a version,
a run's ledger — is not a model: those are frozen dataclasses,
built far more often than they are ever parsed. A node's return declaration is not one either: a
function that returns a model returns one value, not one output per field.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Self

from pydantic import BaseModel, ConfigDict


class ConductorModel(BaseModel):
    """A frozen record that writes itself as JSON and reads back what a host saves."""

    model_config = ConfigDict(frozen=True, extra="forbid")

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
