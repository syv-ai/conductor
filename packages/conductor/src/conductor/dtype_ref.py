"""``DTypeRef`` — how a record carries a declared type on the wire.

A record such as ``Field`` holds the *class* an author declared — ``Text``,
``Series[Text]``, ``Any`` — and the frontend needs that class as JSON.
``DTypeRef`` is the annotation that makes pydantic do the crossing: in
Python the field holds the class untouched, dumped it is ``description_of``
the class, and its JSON schema is published so a record holding one can be
a response model::

    @dataclass(frozen=True)
    class Field:
        name: str
        dtype: DTypeRef

    TypeAdapter(Field).dump_python(Field("text", Text), mode="json")
    # {"name": "text", "dtype": {"id": "text", "accepted_as": ["text"]}}

``description_of`` is the one function behind it, and is also what a
``Series`` uses to nest its element type.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import PlainSerializer, PlainValidator, WithJsonSchema

from conductor.dtype import DType


def description_of(declared: Any) -> dict[str, Any] | None:
    """The JSON form of a declared type.

    A ``DType`` gives its ``describe()``; ``Any`` (an input that routes a
    value it does not read) gives ``{"id": "any"}``; any other type gives
    ``None``, because no value of it travels on a wire — it is the static
    type of an input no cable can reach.
    """
    if declared is Any:
        return {"id": "any"}
    if isinstance(declared, type) and issubclass(declared, DType):
        return declared.describe()
    return None


def _declared(value: Any) -> Any:
    """Validator for a ``DTypeRef`` field: keep the declared class as it is.

    A record's ``dtype`` is read off a signature, never off the wire, so
    there is nothing to coerce.
    """
    return value


#: The JSON schema of ``describe()``'s record.
_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "accepted_as": {"type": "array", "items": {"type": "string"}},
    },
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
