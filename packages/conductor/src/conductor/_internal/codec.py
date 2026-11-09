"""One codec: a value goes to JSON and back by its declared type.

A value lives in memory as its type — a ``Text`` is a ``str`` subclass, a
``Series[X]`` an object with an index and rows, a table a Polars frame, a
host's file a name and bytes. Whenever such a value crosses into JSON and back — the state of
a run, an answer a host hands in, a result a host stores
— this module is the one place that happens, and the declared type decides the
form: ``to_wire(value, dtype)`` writes it, ``cell_validator_for(dtype)`` reads it
back as the same type. Nothing else in the library turns a
value into JSON, so every such value agrees.

The form is the type's own pydantic schema, which every ``DType`` has: a
type built on ``str`` or ``float`` is its plain value, a type with a
schema of its own (a file, a table) is whatever that schema writes, and
``Any`` is the value as JSON-ready data, a typed value inside it written
through its own type. A series is the one form this
module owns outright — ``{"index", "rows", "values"}``, the values each
through the element type — because a series is the engine's own idea:
``Series`` points its pydantic schema at ``series_to_wire`` and
``series_from_wire`` here, so both directions live in one place.

A skip has no type to cross by, so a run's state lists the skipped rows beside the values.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from functools import cache, partial
from math import isfinite
from types import NoneType
from typing import Any

from pydantic import ConfigDict, PydanticUserError, TypeAdapter

from conductor._internal.dtype import _FloatThatIsNotFinite
from conductor._internal.series import Index, Series
from conductor._internal.table import Table, is_frame_type, is_table_or_series_of_tables

#: Runtime types whose JSON form can hold no float a dump would have to look for.
#: A table writes its own columns and raises for what it can't keep.
_NEVER_HOLDS_A_FLOAT = (str, int, NoneType)


def to_wire(value: Any, dtype: Any) -> Any:
    """``value``, declared as ``dtype``, as JSON-ready data.

    ``dtype`` is what the field declares: a ``DType`` class, ``Series[X]``
    or ``Any``. A value pydantic cannot write raises its own error; the
    caller, who knows which field the value sits on, names it.
    """
    return cached_type_adapter(dtype).dump_python(value, mode="json", fallback=_dump_by_runtime_type)


def cell_validator_for(dtype: Any) -> Callable[[Any], Any]:
    """The one check a value of ``dtype`` passes to enter a run's state.

    A node's return, a person's answer and a stored value all go through it.
    What it takes is the value or its JSON form; what comes back is an instance
    of ``dtype``. A value that already is one passes untouched, so a typed
    value costs one comparison however often it is read. It raises pydantic's
    ``ValidationError``. A type whose validator hands back something other than
    the type is a bug in the type, and a ``TypeError`` naming it.
    """
    if dtype.id == "any":
        return _return_as_is
    validate = cached_type_adapter(dtype).validate_python
    if is_table_or_series_of_tables(dtype):  # a frame says nothing about its columns, so every one is read
        return validate
    instance_type = dtype._instance_type()
    if instance_type is float:
        return partial(_read_float, validate)

    def validate_cell(value: Any) -> Any:
        if type(value) is instance_type:
            return value
        validated = validate(value)
        if not isinstance(validated, instance_type):
            raise TypeError(f"{dtype.__name__} read a {type(value).__name__} as a {type(validated).__name__}; a type's validator must return the type")
        return validated

    return validate_cell


def _read_float(validate: Callable[[Any], float], value: Any) -> float:
    """One value of an output declared ``float``; one that isn't finite is held so that a state can write it."""
    validated = value if type(value) is float else validate(value)
    return validated if isfinite(validated) else _FloatThatIsNotFinite(validated)


def _return_as_is(value: Any) -> Any:
    return value


def values_to_wire(values: list[Any]) -> list[Any]:
    """Each value as JSON-ready data, written by its own type. Values all of
    one type, the usual case, are written through one adapter in one call.

    A float that isn't finite comes out as JSON can hold it where its type
    reads it back (a type built on ``float``, an output declared ``float``, a
    table's column). Anywhere else, a plain ``float`` in a field typed ``Any`` or
    one inside a record, nothing would, and this raises rather than let it come
    back as something else.
    """
    runtime_types = {type(value) for value in values}
    if len(runtime_types) == 1:
        (runtime_type,) = runtime_types
        written: list[Any] = cached_type_adapter(list[_written_as(runtime_type)]).dump_python(values, mode="json", fallback=_dump_by_runtime_type)  # type: ignore[misc]
    else:
        written = [to_wire(value, type(value)) for value in values]
    if not all(issubclass(runtime_type, _NEVER_HOLDS_A_FLOAT) or is_frame_type(runtime_type) for runtime_type in runtime_types):
        json.dumps(written, allow_nan=False)
    return written


def json_schema_of(dtype: Any) -> dict[str, Any]:
    """The JSON schema of what a person may send for one value of ``dtype``:
    the type's own, the elements' as an array for a series, ``{}`` when a type
    publishes none."""
    if dtype.element is not None:
        return {"type": "array", "items": json_schema_of(dtype.element)}
    try:
        return cached_type_adapter(dtype).json_schema()
    except PydanticUserError:
        return {}


def _dump_by_runtime_type(value: Any) -> Any:
    """A typed value where the declared type says nothing about it — a slot
    typed ``Any``, the contents of a bare ``Series`` — written through its
    own type. A value its own type cannot write raises."""

    def dump_nested(nested_value: Any) -> Any:
        if nested_value is value:
            raise TypeError(f"a {type(value).__name__} has no JSON form")
        return _dump_by_runtime_type(nested_value)

    return cached_type_adapter(type(value)).dump_python(value, mode="json", fallback=dump_nested)


@cache
def cached_type_adapter(dtype: Any) -> TypeAdapter[Any]:
    """One adapter per declared type: building one reads the type's schema, which is the
    slow part. It leaves a float that isn't finite as it is, for ``values_to_wire`` to see;
    pydantic's default would write it as ``null``. A type with a config of its own keeps that."""
    dtype = _written_as(dtype)
    try:
        return TypeAdapter(dtype, config=ConfigDict(ser_json_inf_nan="constants"))
    except PydanticUserError:
        return TypeAdapter(dtype)


def _written_as(runtime_type: Any) -> Any:
    """The type a runtime type writes through: a frame has no schema of its own, so a table's."""
    return Table if isinstance(runtime_type, type) and is_frame_type(runtime_type) else runtime_type


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
    list is dense rows on a fresh root index; a scalar raises rather
    than being wrapped. What ``Series``'s pydantic schema calls to validate."""
    if isinstance(value, Series):
        return Series(validate_values(list(value.values)), index=value.index, rows=value.rows)
    if isinstance(value, Mapping) and set(value) == {"index", "rows", "values"}:
        return Series(
            validate_values(list(value["values"])),
            index=Index.model_validate(value["index"]),
            rows=[tuple(row) for row in value["rows"]],
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return Series(validate_values(list(value)))
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
