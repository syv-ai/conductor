"""One codec: a value goes to JSON and back by its declared type.

A value lives in memory as its type — a ``Text`` is a ``str`` subclass, a
table a Polars frame, a host's file a name and bytes. Whenever such a value crosses into JSON and back — the state of
a run, an answer a host hands in, a result a host stores
— this module is the one place that happens, and the declared type decides the
form: ``to_jsonable(value)`` writes it, ``cell_validator_for(dtype)`` reads it
back as the same type. Nothing else in the library turns a
value into JSON, so every such value agrees, and ``TypedValue`` is how a
record's field says it holds one.

The form is the type's own pydantic schema, which every ``DType`` has: a
type built on ``str`` or ``float`` is its plain value, a type with a
schema of its own (a file, a table) is whatever that schema writes, and
a plain container is the value as JSON-ready data, a typed value inside it
written through its own type.

A float that isn't finite is the text ``"NaN"``, ``"Infinity"`` or
``"-Infinity"`` wherever it stands, since JSON has no literal for one and a
``jsonb`` column refuses the bare token. A float's validator reads the text
back; a value typed ``Any`` comes back as this text, as it comes back as the
JSON form of everything else it held.

A skip has no type to cross by, so a run's state lists the skipped rows beside the values.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import cache
from math import isfinite, isnan
from types import NoneType
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, PlainSerializer, PydanticUserError, TypeAdapter
from pydantic_core import from_json, to_json

from conductor._internal.table import Table, is_frame_type, is_table_type

#: Runtime types whose JSON form can hold no float a dump would have to look for.
#: A table writes its own columns and raises for what it can't keep.
_NEVER_HOLDS_A_FLOAT = (str, int, NoneType)


def to_jsonable(value: Any) -> Any:
    """``value`` as JSON-ready data, every value in it written by its own type.

    ``value`` is a typed value (a table, a ``Text``) or plain data holding some,
    as ``compiled.results(state)`` is. What comes back goes through ``json.dumps``
    as it is. A value with no JSON form raises."""
    return each_to_jsonable([value])[0]


#: A field that holds a typed value: as JSON it is what ``to_jsonable`` writes, in Python the value as it is.
TypedValue = Annotated[Any, PlainSerializer(to_jsonable, return_type=Any, when_used="json")]


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
    if is_table_type(dtype):  # a frame says nothing about its columns, so every one is read
        return validate
    instance_type = dtype._instance_type()

    def validate_cell(value: Any) -> Any:
        if type(value) is instance_type:
            return value
        validated = validate(value)
        if not isinstance(validated, instance_type):
            raise TypeError(f"{dtype.__name__} read a {type(value).__name__} as a {type(validated).__name__}; a type's validator must return the type")
        return validated

    return validate_cell


def _return_as_is(value: Any) -> Any:
    return value


def each_to_jsonable(values: list[Any]) -> list[Any]:
    """Each value as ``to_jsonable`` writes it. Values all of one type, the usual
    case, are written through one adapter in one call."""
    runtime_types = {type(value) for value in values}
    if len(runtime_types) != 1:
        return [to_jsonable(value) for value in values]
    (runtime_type,) = runtime_types
    adapter = cached_type_adapter(list[_written_as(runtime_type)])  # type: ignore[misc]
    if issubclass(runtime_type, _NEVER_HOLDS_A_FLOAT) or is_frame_type(runtime_type):
        written: list[Any] = adapter.dump_python(values, mode="json", fallback=_dump_by_runtime_type)
        return written
    if issubclass(runtime_type, BaseModel):  # a record's config spells its own floats, so they are read out of it first
        return from_json(to_json(adapter.dump_python(values, mode="json", fallback=_dump_by_runtime_type), inf_nan_mode="strings"), allow_inf_nan=False)  # type: ignore[no-any-return]
    return from_json(adapter.dump_json(values, fallback=_dump_by_runtime_type), allow_inf_nan=False)  # type: ignore[no-any-return]


def json_schema_of(dtype: Any, many: bool = False) -> dict[str, Any]:
    """The JSON schema of what a person may send for ``dtype``: the type's own, as
    an array for many values, ``{}`` when a type publishes none."""
    if many:
        return {"type": "array", "items": json_schema_of(dtype)}
    try:
        return cached_type_adapter(dtype).json_schema()
    except PydanticUserError:
        return {}


def _dump_by_runtime_type(value: Any) -> Any:
    """A typed value inside plain data, written through its
    own type. A value its own type cannot write raises."""

    def dump_nested(nested_value: Any) -> Any:
        if nested_value is value:
            raise TypeError(f"a {type(value).__name__} has no JSON form")
        return _dump_by_runtime_type(nested_value)

    return cached_type_adapter(type(value)).dump_python(value, mode="json", fallback=dump_nested)


@cache
def cached_type_adapter(dtype: Any) -> TypeAdapter[Any]:
    """One adapter per declared type: building one reads the type's schema, which is the slow part.
    A float it writes itself that isn't finite is spelled the way ``float_as_json`` does; a pydantic record's own floats are spelled by ``each_to_jsonable``."""
    dtype = _written_as(dtype)
    try:
        return TypeAdapter(dtype, config=ConfigDict(ser_json_inf_nan="strings"))
    except PydanticUserError:
        return TypeAdapter(dtype)


def _written_as(runtime_type: Any) -> Any:
    """The type a runtime type writes through: a frame has no schema of its own, so a table's."""
    return Table if isinstance(runtime_type, type) and is_frame_type(runtime_type) else runtime_type


def float_as_json(value: float) -> float | str:
    """A float as JSON: a finite one as it is, the others as the module docstring says.
    What a ``DType`` built on ``float`` writes wherever a pydantic record holds one."""
    if isfinite(value):
        return value
    return "NaN" if isnan(value) else ("Infinity" if value > 0 else "-Infinity")
