"""Reading a value that enters a graph, where JSON could have come instead.

A caller's graph inputs and a person's answers may come from Python or over HTTP, and a graph record's constants
and defaults are JSON, and all must read the same. So such a value is read strictly, the way its JSON text would be
(``read_as_json``): ``True`` is a ``bool`` and ``"yes"`` is refused, ``3`` is an ``int`` and ``"3"`` is refused,
and a ``date`` may be given as a ``date`` or as its ISO text. A saved run is read laxly, since conductor wrote it,
and what reads strictly here reads the same there.

Two things have no JSON form here. A set has no order (``set_has_no_order``), and a frame is a value only where
its type is a table, which reads it as it is. Anything else pydantic can't write, such as an object held in
memory, is ``no_json_form``. And a number that enters is finite: ``inf`` and ``nan`` are refused here with
pydantic's ``finite_number``, though what a node computes may hold them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from typing import Any

import polars as pl
from pydantic import BaseModel, ValidationError
from pydantic_core import (
    PydanticCustomError,
    PydanticSerializationError,
    SchemaValidator,
)

from conductor._internal.base import finite_json_text, validate_as_json
from conductor._internal.problem import NodeDeclarationError, conductor_error
from conductor._internal.table import TableType, frame_as_json
from conductor._internal.types import Type


def read_as_json(value_type: Type, validator: SchemaValidator, value: Any) -> Any:
    """Reads ``value`` by ``value_type`` strictly, the way ``validator`` reads the same value sent as JSON text.

    The value is written as JSON text (``json_text``) and read back strictly, a host's object written by its own
    class's schema, as a value of its type is. A number in it must be finite: pydantic's parser reads the text with
    ``allow_inf_nan=False``, so ``inf`` and ``nan`` are refused at any depth, whatever the type, ``Any`` and a
    host's model included, with pydantic's own ``finite_number``. Their texts, ``"Infinity"`` and ``"NaN"``, are
    text, which a ``float`` refuses.

    A frame given where the type is a table is read as it is, strictly, as a node's return is: the table's own
    rules make it read as its JSON would, and it keeps only the columns it names, so a column it doesn't name is
    never written. A float column of it with a ``NaN`` or an infinite cell is refused with ``finite_number`` too,
    at the column.
    The table type itself takes them, since it also reads what a node returns.

    A host's own check may raise something other than pydantic's error, such as ``float(None)``'s ``TypeError``.
    That is a refusal too, as ``Type.takes_value`` reads it: a ``ValidationError`` with ``value_error``.

    Args:
        value_type: The type the value is read by.
        validator: One of ``value_type``'s checks: ``argument_validator`` for an input, ``output_validator`` for
            an answer.
        value: The value given.

    Raises:
        PydanticCustomError: ``set_has_no_order`` if the value has a set in it, and ``no_json_form`` if it can't
            be written as JSON, a frame where the type is no table among them.
        ValidationError: ``validator`` refuses the value, or ``finite_number`` for a number in it that isn't
            finite.
    """
    if isinstance(value, pl.DataFrame) and any(
        isinstance(Type(option), TableType) for option in value_type.options
    ):
        if (column := _column_not_finite(value)) is not None:
            raise ValidationError.from_exception_data(
                value_type.id, [{"type": "finite_number", "loc": (column,), "input": value}]
            )
        return _checked(value_type, value, lambda: validator.validate_python(value, strict=True))
    if holds_a_set(value):
        raise set_has_no_order()
    try:
        text = finite_json_text(value, value_type.id, fallback=_written)
    except PydanticSerializationError:
        raise no_json_form() from None
    read = _checked(value_type, value, lambda: validator.validate_json(text, strict=True))
    # A dict's key is text in the JSON, so the text can't show a key that isn't finite, like dict[float, str]'s
    # "Infinity", until the type has read it. A saved run is read by its own reader, so it may keep such keys.
    finite_json_text(
        [key for part in _parts(read) if isinstance(part, dict) for key in part], value_type.id
    )
    return read


def _checked(value_type: Type, value: Any, check: Callable[[], Any]) -> Any:
    """Returns what ``check``, a call of the type's validator on ``value``, gives.

    A host's own check may raise something other than pydantic's error where pydantic would refuse, such as
    ``float(None)``'s ``TypeError``, as ``Type.takes_value`` says. That is a refusal too, so it is raised as a
    ``ValidationError`` with ``value_error``.
    """
    try:
        return check()
    except ValidationError:
        raise
    except Exception as error:  # noqa: BLE001  a host's own check, which may raise anything
        raise ValidationError.from_exception_data(
            value_type.id,
            [{"type": "value_error", "loc": (), "input": value, "ctx": {"error": error}}],
        ) from None


def _column_not_finite(frame: pl.DataFrame) -> str | None:
    """The first float column of ``frame`` with a ``NaN`` or an infinite cell, or ``None`` when it has none."""
    floats = frame.select(pl.col(pl.Float32, pl.Float64))
    held = floats.select((~pl.all().is_finite()).any())
    return next((column for column in held.columns if held[column].item()), None)


def json_data(value: Any) -> Any:
    """Returns ``value`` as the JSON data it is read from, where no type says what it is yet.

    A host keeps a person's answer this way until the run is given it (``Run.queued_as_json``). A ``date``
    becomes its ISO text, a dataclass or a model a mapping of its fields, and a frame its columns, since the
    answer's type isn't known to read the frame as it is. ``inf`` stays a float, so the data is refused where it is
    read (``finite_number``), as the value itself is.

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


def holds_a_set(value: Any) -> bool:
    """Whether ``value`` is a set, or has one somewhere inside (``_parts``).

    Pydantic writes a set as an array by itself and never hands it to a ``fallback``, so the set is looked for
    before the value is written.
    """
    return any(isinstance(part, (set, frozenset)) for part in _parts(value))


def _parts(value: Any, walked: frozenset[int] = frozenset()) -> Iterator[Any]:
    """Yields ``value`` and every value inside it: in a list, tuple or dict, or in a field of a dataclass or a
    pydantic model.

    A container already walked on the way down (``walked``, by ``id``) is a cycle, which isn't walked again: pydantic
    refuses it when it writes the value.
    """
    yield value
    if id(value) in walked:
        return
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
        return
    for part in parts:
        yield from _parts(part, inside)


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
            "a frame is read only where a run is given it for a table, or a union with one: give its rows as a list of records",
        )
    try:
        return Type(type(value)).to_json(value)
    # a class pydantic has no schema for, or whose schema can't write it
    except (NodeDeclarationError, PydanticSerializationError):
        raise TypeError(type(value).__name__) from None
