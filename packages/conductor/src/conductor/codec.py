"""One codec: a value goes to JSON and back by its declared type.

A value lives in memory as its type — a ``Text`` is a ``str`` subclass, a
``Series[X]`` an object with an index and rows, a host's file a name and
bytes. Whenever such a value crosses into JSON and back — the state of
a run, an answer a host hands in, a result a host stores
— this module is the one crossing, and the declared type decides the
form: ``to_wire(value, dtype)`` writes it, ``from_wire(data, dtype)``
reads it back as the same type. Nothing else in the library turns a
value into JSON, so every crossing agrees.

The form is the type's own pydantic schema, which every ``DType`` has: a
type built on ``str`` or ``float`` is its plain value, a type with a
schema of its own (a file, a table) is whatever that schema writes, and
``Any`` is the value as JSON-ready data, a typed value inside it written
through its own type. A series is the one form this
module owns outright — ``{"index", "rows", "values"}``, the values each
through the element type — because a series is the engine's own idea:
``Series`` points its pydantic schema at ``series_to_wire`` and
``series_from_wire`` here, so both directions live in one place.

What this is not: the skip marker. A skip has no type to cross by; a run's
state lists the rows skipped beside the values, instead of a value.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from functools import cache
from typing import Any

from pydantic import PydanticUserError, TypeAdapter

from conductor.series import Index, Series


def to_wire(value: Any, dtype: Any) -> Any:
    """``value``, declared as ``dtype``, as JSON-ready data.

    ``dtype`` is what the field declares: a ``DType`` class, ``Series[X]``
    or ``Any``. A value pydantic cannot write raises its own error; the
    caller, who knows which field the value sits on, names it.
    """
    return _adapter(dtype).dump_python(value, mode="json", fallback=_by_own_type)


def from_wire(data: Any, dtype: Any) -> Any:
    """``data``, the JSON form of a value declared as ``dtype``, as that type again.

    A series comes back on the index it was written with, rows and all;
    under ``Any`` the data is the value.
    """
    return _adapter(dtype).validate_python(data)


def reader_of(dtype: Any) -> Callable[[Any], Any]:
    """The one check a value of ``dtype`` passes to enter a run's state.

    A node's return, a person's answer and a stored value all go through it.
    What it takes is the value or its JSON form; what comes back is an instance
    of ``dtype``. A value that already is one passes untouched, so a typed
    value costs one comparison however often it is read. It raises pydantic's
    ``ValidationError``. A type whose validator hands back something other than
    the type is a bug in the type, and a ``TypeError`` naming it.
    """
    if dtype.id == "any":
        return _unchanged
    validate = _adapter(dtype).validate_python

    def read(value: Any) -> Any:
        if type(value) is dtype:
            return value
        made = validate(value)
        if not isinstance(made, dtype):
            raise TypeError(f"{dtype.__name__} read a {type(value).__name__} as a {type(made).__name__}; a type's validator must return the type")
        return made

    return read


def _unchanged(value: Any) -> Any:
    return value


def values_to_wire(values: list[Any]) -> list[Any]:
    """Each value as JSON-ready data, written by its own type. Values all of
    one type, the usual case, are written through one adapter in one call."""
    kinds = {type(value) for value in values}
    if len(kinds) == 1:
        return _adapter(list[kinds.pop()]).dump_python(values, mode="json", fallback=_by_own_type)  # type: ignore[misc]
    return [to_wire(value, type(value)) for value in values]


def json_schema_of(dtype: Any) -> dict[str, Any]:
    """The JSON schema of what a person may send for one value of ``dtype``:
    the type's own, the elements' as an array for a series, ``{}`` when a type
    publishes none."""
    if dtype.element is not None:
        return {"type": "array", "items": json_schema_of(dtype.element)}
    try:
        return _adapter(dtype).json_schema()
    except PydanticUserError:
        return {}


def _by_own_type(value: Any) -> Any:
    """A typed value where the declared type says nothing about it — a slot
    typed ``Any``, the contents of a bare ``Series`` — written through its
    own type. A value its own type cannot write raises."""

    def inside(unknown: Any) -> Any:
        if unknown is value:
            raise TypeError(f"a {type(value).__name__} has no JSON form")
        return _by_own_type(unknown)

    return _adapter(type(value)).dump_python(value, mode="json", fallback=inside)


@cache
def _adapter(dtype: Any) -> TypeAdapter[Any]:
    """One adapter per declared type: building one reads the type's schema, which is the slow part."""
    return TypeAdapter(dtype)


# -- a series ----------------------------------------------------------------------------


def series_to_wire(series: Series[Any]) -> dict[str, Any]:
    """The JSON form of a series: ``{"index": {...}, "rows": [...], "values": [...]}``.

    The index travels whole (id and parent chain), so a reader can tell
    which series share one and which rows belong to which parent. The
    values are left to the caller's schema, which writes each through the
    element type. What ``Series``'s pydantic schema calls to serialise.
    """
    return {
        "index": series.index.model_dump(),
        "rows": series.rows,
        "values": list(series.values),
    }


def series_from_wire(value: Any, validate_values: Callable[[list[Any]], list[Any]]) -> Series[Any]:
    """A series from what arrives: a series passes through, its values
    validated; the wire form comes back on its own index and rows; a plain
    list is dense rows on a fresh root index; a scalar is refused rather
    than wrapped. What ``Series``'s pydantic schema calls to validate."""
    if isinstance(value, Series):
        return Series(value.index, validate_values(list(value.values)), rows=value.rows)
    if isinstance(value, Mapping) and set(value) == {"index", "rows", "values"}:
        return Series(
            Index.model_validate(value["index"]),
            validate_values(list(value["values"])),
            rows=[tuple(row) for row in value["rows"]],
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return Series(Index.fresh(), validate_values(list(value)))
    raise TypeError(f"expected a series, got {type(value).__name__}")


#: The JSON schema of an index. ``parent`` has this same shape, as deep
#: as the lineage goes; it is published as a plain object because a
#: schema built inside a pydantic schema function cannot reference itself.
INDEX_WIRE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "parent": {
            "anyOf": [{"type": "object"}, {"type": "null"}],
            "description": "The parent index, in this same shape; null for a root.",
        },
    },
    "required": ["id", "parent"],
}

#: The JSON schema of ``series_to_wire``'s form, published for a record that holds a series.
SERIES_WIRE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "index": INDEX_WIRE_SCHEMA,
        "rows": {"type": "array", "items": {"type": "array", "items": {"type": "integer"}}},
        "values": {"type": "array"},
    },
    "required": ["index", "rows", "values"],
}
