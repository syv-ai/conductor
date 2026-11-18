"""Reading a value given from Python where JSON could have come instead.

A caller's graph inputs and a person's answers may come from Python or over HTTP, and both must read the same.
So such a value is read strictly, the way its JSON text would be (``read_as_json``): ``True`` is a ``bool`` and
``"yes"`` is refused, ``3`` is an ``int`` and ``"3"`` is refused, and a ``date`` may be given as a ``date`` or as
its ISO text. What reads that way is also what a saved run reads back.

Two things have no JSON form here. A set has no order (``set_has_no_order``), and a frame is a value only where
its type is a table, which reads it as it is. Anything else pydantic can't write, such as an object held in
memory, is ``no_json_form``.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import polars as pl
from pydantic import BaseModel
from pydantic_core import PydanticCustomError, PydanticSerializationError, SchemaValidator

from conductor._internal.base import validate_as_json
from conductor._internal.problem import NodeDeclarationError, conductor_error
from conductor._internal.table import TableType, frame_as_json
from conductor._internal.types import Type


def read_as_json(value_type: Type, validator: SchemaValidator, value: Any) -> Any:
    """Reads ``value`` by ``value_type`` strictly, the way ``validator`` reads the same value sent as JSON text.

    The value is written as JSON text and read back (``validate_as_json``). A host's object is written by its own
    class's schema, as a value of its type is. A frame given where the type is a table is read as it is, strictly,
    as a node's return is: the table's own rules make it read as its JSON would, and it keeps only the columns it
    names, so a column it doesn't name is never written.

    Args:
        value_type: The type the value is read by.
        validator: One of ``value_type``'s checks: ``argument_validator`` for an input, ``output_validator`` for
            an answer.
        value: The value given.

    Raises:
        PydanticCustomError: ``set_has_no_order`` if the value has a set in it, and ``no_json_form`` if it can't
            be written as JSON, a frame where the type is no table among them.
        ValidationError: ``validator`` refuses the value.
    """
    if isinstance(value, pl.DataFrame) and any(
        isinstance(Type(option), TableType) for option in value_type.options
    ):
        return validator.validate_python(value, strict=True)
    if holds_a_set(value):
        raise set_has_no_order()
    try:
        return validate_as_json(validator, value, fallback=_written)
    except PydanticSerializationError:
        raise no_json_form() from None


def json_data(value: Any) -> Any:
    """Returns ``value`` as the JSON data it is read from, where no type says what it is yet.

    A host keeps a person's answer this way until the run is given it (``Run.queued_as_json``). A ``date``
    becomes its ISO text, a dataclass or a model a mapping of its fields, and a frame its columns, since the
    answer's type isn't known to read the frame as it is. ``inf`` stays a float, so the data reads back as the
    value did; ``base.plain_json`` writes it as its text instead, for JSON text a standard parser reads.

    Raises:
        PydanticCustomError: ``set_has_no_order`` or ``no_json_form``, as ``read_as_json`` raises them, or a
            frame column's own refusal when a saved run can't keep it as it is.
    """
    if holds_a_set(value):
        raise set_has_no_order()
    try:
        return validate_as_json(
            Type(Any).output_validator,
            value,
            fallback=lambda part: (
                frame_as_json(part) if isinstance(part, pl.DataFrame) else _written(part)
            ),
        )
    except PydanticSerializationError:
        raise no_json_form() from None


def no_json_form() -> PydanticCustomError:
    """The refusal of a value that can't be written as JSON, such as a ``str`` with a lone surrogate, or an object
    held in memory that pydantic only checks with ``isinstance``."""
    return conductor_error("no_json_form", "the value can't be written as JSON")


def set_has_no_order() -> PydanticCustomError:
    """The refusal of a set where a value is read from its JSON form, written into a graph record, or given for a
    ``Series``.

    A set's order changes from one process to the next, so its items would come in a different order each time,
    and the same set would be written as a different JSON text.
    """
    return conductor_error(
        "set_has_no_order",
        "a set has no order, so its items could come in a different order each time",
    )


def holds_a_set(value: Any, walked: frozenset[int] = frozenset()) -> bool:
    """Whether ``value`` is a set, or has one somewhere inside: in a list, tuple or dict, or in a field of a
    dataclass or a pydantic model.

    Pydantic writes a set as an array by itself and never hands it to a ``fallback``, so the set is looked for
    before the value is written. A container already walked on the way down (``walked``, by ``id``) is a cycle,
    which isn't walked again: pydantic refuses it when it writes the value.
    """
    if isinstance(value, (set, frozenset)):
        return True
    if id(value) in walked:
        return False
    inside = walked | {id(value)}
    if isinstance(value, dict):
        parts: Any = value.values()
    elif isinstance(value, (list, tuple)):
        parts = value
    elif isinstance(value, BaseModel):
        parts = (field for _, field in value)
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        parts = (getattr(value, field.name) for field in dataclasses.fields(value))
    else:
        return False
    return any(holds_a_set(part, inside) for part in parts)


def _written(value: Any) -> Any:
    """Writes an object pydantic can't write by itself, by its own class's schema: what ``to_json`` calls as its
    ``fallback`` here.

    Raises:
        PydanticCustomError: ``no_json_form`` for a frame, which is a value only where the type is a table.
        TypeError: The object's class has no schema that writes it.
    """
    if isinstance(value, pl.DataFrame):
        raise conductor_error(
            "no_json_form",
            "a frame is read only where the value's own type is a table, or a union with one: give its rows as a list of records",
        )
    try:
        return Type(type(value)).to_json(value)
    # a class pydantic has no schema for, or whose schema can't write it
    except (NodeDeclarationError, PydanticSerializationError):
        raise TypeError(type(value).__name__) from None
