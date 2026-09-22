"""``DTypeRef`` — how a record carries a declared type on the wire.

A record such as ``Field`` holds the *class* an author declared — ``Text``,
``Series[Text]``, ``Any`` — and the frontend needs that class as JSON.
``DTypeRef`` is the annotation that makes pydantic do the crossing: in
Python the field holds the class untouched, dumped it is ``description_of``
the class, and its JSON schema is published so a record holding one can be
a response model::

    class Field(ConductorModel):
        name: str
        dtype: DTypeRef

    Field(name="text", dtype=Text).model_dump(mode="json")
    # {"name": "text", "dtype": {"id": "text"}}

Where a value of the type may land is not on the field's record: that is
``accepted_as`` on the registry's ``TypeDescription``, served once per
type in ``NodeRegistry.describe()``.

``description_of`` is the one function behind it, and is also what a
``Series`` uses to nest its element type.
"""

from __future__ import annotations

from typing import Annotated, Any, get_origin

from pydantic import PlainSerializer, PlainValidator, WithJsonSchema

from conductor.dtype import DType


def description_of(declared: Any) -> dict[str, Any] | None:
    """The JSON form of a declared type.

    A ``DType`` gives its ``describe()``; ``Any`` (an input that routes a
    value it does not read) gives ``{"id": "any"}``; any other type gives
    ``None``, because no value of it travels on an edge — it is the static
    type of an input no edge can reach.
    """
    if declared is Any:
        return {"id": "any"}
    if isinstance(declared, type) and issubclass(declared, DType):
        return declared.describe()
    return None


def name_of(declared: Any) -> str:
    """A declared type named for an English sentence: ``text``, ``a series of
    text``, ``any`` — by its id, which is what a host also keys on."""
    if declared is Any:
        return "any"
    if isinstance(declared, type) and issubclass(declared, DType):
        return f"a series of {declared.element.id}" if declared.element is not None else declared.id
    return getattr(declared, "__name__", str(declared))


def _declared(value: Any) -> Any:
    """Validator for a ``DTypeRef`` field: keep the declared type as it is, and refuse anything that is not one.

    A record's ``dtype`` is read off a signature, never off the wire, so
    there is nothing to coerce: a ``DType`` class, ``Any``, or the static
    type of an input no edge reaches — a class or a typing form such as
    ``list[str]``. A value (``42``, ``"text"``, a dumped ``{"id": ...}``)
    is refused rather than carried as a type nobody can describe.
    """
    if value is Any or isinstance(value, type) or get_origin(value) is not None:
        return value
    raise ValueError(f"dtype must be a DType class, Any or a type, not {value!r}")


#: The JSON schema of ``describe()``'s record.
_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"id": {"type": "string"}},
    "required": ["id"],
}

#: The type of a record's ``dtype`` field. In Python it holds the declared
#: class (or ``Any``); serialised it is ``description_of`` — the
#: ``describe()`` record, ``{"id": "any"}``, or ``null`` for a static type
#: nothing travels on. The published JSON schema is two levels deep (a
#: series' ``of``) and no deeper, since ``Series[Series[...]]`` does not exist.
DTypeRef = Annotated[
    Any,
    PlainValidator(_declared),
    PlainSerializer(description_of, return_type=dict | None),
    WithJsonSchema(
        {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {**_DESCRIPTION_SCHEMA["properties"], "of": _DESCRIPTION_SCHEMA},
                    "required": ["id"],
                },
                {"type": "null"},
            ]
        }
    ),
]
