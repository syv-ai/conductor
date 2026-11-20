"""Reading a value that enters a graph, where JSON could have come instead.

A caller's graph inputs and a person's answers may come from Python or over HTTP, a graph record's constants
and defaults are JSON, and a node's own default is written into a record, and all must read the same. So such a
value is read strictly, the way its JSON text would be (``read_as_json``): ``True`` is a ``bool`` and ``"yes"`` is
refused, ``3`` is an ``int`` and ``"3"`` is refused, and a ``date`` may be given as a ``date`` or as its ISO text. A saved run is read laxly, since conductor wrote it,
and what reads strictly here reads the same there.

Two things have no JSON form here. A set has no order (``set_has_no_order``), and a frame is a value only where
its type is a table, which reads it as it is. Anything else pydantic can't write, such as an object held in
memory, is ``no_json_form``. And a number that enters is finite: ``inf`` and ``nan`` are refused here with
pydantic's ``finite_number``, though what a node computes may hold them. Every refusal here is pydantic's
``ValidationError``, as a type's own is, so a caller handles one kind of error.

The reader walks a value with ``value_parts``, the one walk over a value and everything inside it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import polars as pl
from pydantic import ValidationError

from conductor._internal.base import finite_json_text, value_parts
from conductor._internal.problem import json_form, no_json_form

from .type import Type, json_by_its_class


def read_as_json(value_type: Type, value: Any, *, as_output: bool = False) -> Any:
    """Reads ``value`` by ``value_type`` strictly, the way the type reads the same value sent as JSON text.

    The value is written as JSON text (``json_text``) and read back strictly, a host's object written by its own
    class's schema, as a value of its type is. A number in it must be finite: pydantic's parser reads the text with
    ``allow_inf_nan=False``, so ``inf`` and ``nan`` are refused at any depth, whatever the type, ``Any`` and a
    host's model included, with pydantic's own ``finite_number``. Their texts, ``"Infinity"`` and ``"NaN"``, are
    text, which a ``float`` refuses.

    A frame given where the type is a table, or a union with one (a type with a ``row``), is read as it is, strictly, as a node's return is: the table's own
    rules make it read as its JSON would, and it keeps only the columns it names, so a column it doesn't name is
    never written. A float column of it with a ``NaN`` or an infinite cell is refused with ``finite_number`` too,
    at the column.
    The table type itself takes them, since it also reads what a node returns.

    A host's own check may raise something other than pydantic's error, such as ``float(None)``'s ``TypeError``.
    That is a refusal too, as ``Type.takes_value`` reads it: a ``ValidationError`` with ``value_error``.

    Args:
        value_type: The type the value is read by.
        value: The value given.
        as_output: Read the value by the type's ``output_validator``, as an answer is, and not by its
            ``argument_validator``, as an input is.

    Raises:
        ValidationError: The type's check refuses the value, ``finite_number`` for a number in it that isn't finite,
            ``set_has_no_order`` if the value has a set in it, or ``no_json_form`` if it can't be written as JSON,
            a frame where the type is no table among them.
    """
    validator = value_type.output_validator if as_output else value_type.argument_validator
    if isinstance(value, pl.DataFrame) and any(
        Type(option).row is not None for option in value_type.options
    ):
        if (column := _column_not_finite(value)) is not None:
            raise ValidationError.from_exception_data(
                value_type.id, [{"type": "finite_number", "loc": (column,), "input": value}]
            )
        return _checked(value_type, value, lambda: validator.validate_python(value, strict=True))
    text = json_form(
        value_type.id, value, lambda: finite_json_text(value, value_type.id, fallback=_written)
    )
    read = _checked(value_type, value, lambda: validator.validate_json(text, strict=True))
    # A dict's key is text in the JSON, so the text can't show a key that isn't finite, like dict[float, str]'s
    # "Infinity", until the type has read it. A saved run is read by its own reader, so it may keep such keys.
    finite_json_text(
        [key for part in value_parts(read) if isinstance(part, dict) for key in part], value_type.id
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


def _written(value: Any) -> Any:
    """Writes an object pydantic can't write by itself, by its own class's schema (``json_by_its_class``): what
    ``to_json`` calls as its ``fallback`` here.

    Raises:
        PydanticCustomError: ``no_json_form`` for a frame, which is a value only where the type is a table.
        TypeError: The object's class has no schema that writes it.
    """
    if isinstance(value, pl.DataFrame):
        raise no_json_form(
            "a frame is read only where a run is given it for a table, or a union with one: give its rows as a list with one object per row"
        )
    return json_by_its_class(value)
