"""Reading a value that enters a graph, where JSON could have come instead.

A caller's graph inputs and a person's answers may come from Python or over HTTP, a graph record's constants
and defaults are JSON, and a node's own default is written into a record, and all must read the same. So such a
value is read strictly, the way its JSON text would be: ``True`` is a ``bool`` and ``"yes"`` is refused, ``3`` is an
``int`` and ``"3"`` is refused, and a ``date`` may be given as a ``date`` or as its ISO text. A saved run is read
laxly, since conductor wrote it, and what reads strictly here reads the same there.

A value is read by one of its type's two checks. ``read_as_argument`` reads it as an input is read, by the type's
``argument_validator``: a caller's graph input, a constant, a default, a decline's reason. ``read_as_output`` reads it
as what a node gives is read, by its ``output_validator``: a person's answer, an ``Ask``'s choices and default, and a
value ``make_graph`` writes into a record.

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
from pydantic_core import PydanticCustomError, PydanticSerializationError, SchemaValidator

from conductor._internal.base import finite_json_text, holds_a_set, value_parts
from conductor._internal.problem import no_json_form, set_has_no_order

from .type import Type, json_by_its_class


def read_as_argument(value_type: Type, value: Any) -> Any:
    """Reads ``value`` strictly by ``value_type``'s ``argument_validator``, as the same value sent as JSON text is
    read: a caller's graph input, a constant or a default.

    Raises:
        ValidationError: See ``_read``.
    """
    return _read(value_type, value_type.argument_validator, value)


def read_as_output(value_type: Type, value: Any) -> Any:
    """Reads ``value`` strictly by ``value_type``'s ``output_validator``, as the same value sent as JSON text is
    read: a person's answer, or a value written into a record.

    Raises:
        ValidationError: See ``_read``.
    """
    return _read(value_type, value_type.output_validator, value)


def _read(value_type: Type, validator: SchemaValidator, value: Any) -> Any:
    """Reads ``value`` by ``value_type`` strictly, the way ``validator``, one of the type's two checks, reads the
    same value sent as JSON text (``read_entering_value``).

    A frame given where the type is a table, or a union with one (a type with a ``row``), is read as it is,
    strictly, as a node's return is: the table's own rules make it read as its JSON would, and it keeps only the
    columns it names, so a column it doesn't name is never written. A float column of it with a ``NaN`` or an
    infinite cell is refused with ``finite_number`` too, at the column. The table type itself takes them, since it
    also reads what a node returns.

    Args:
        value_type: The type the value is read by.
        validator: The type's ``argument_validator`` or its ``output_validator``.
        value: The value given.

    Raises:
        ValidationError: As ``read_entering_value`` raises it, with the type's id as its title, or ``finite_number``
            for a float cell of a frame that isn't finite.
    """
    if isinstance(value, pl.DataFrame) and any(
        Type(option).row is not None for option in value_type.options
    ):
        if (column := _column_not_finite(value)) is not None:
            raise ValidationError.from_exception_data(
                value_type.id, [{"type": "finite_number", "loc": (column,), "input": value}]
            )
        return _checked(value_type.id, value, lambda: validator.validate_python(value, strict=True))
    return read_entering_value(validator, value, title=value_type.id)


def read_entering_value(validator: SchemaValidator, value: Any, *, title: str) -> Any:
    """Reads a value that enters a graph strictly by a bare validator, held to the rules of such a value.

    It reads ``value`` the way ``validator`` reads the same value sent as JSON text, with every rule this module
    names: a number that isn't finite, a set and a value with no JSON form are refused, and a host's object is
    written by its own class's schema. ``read_as_argument`` and ``read_as_output`` read a value by a type through
    it. A type's choices are read by it too (``declared_choices``), since a choice is given as an answer is, by a
    validator that may hold more checks than the type's own.

    The value is written as JSON text (``json_text``) and read back strictly, a host's object written by its own
    class's schema (``json_by_its_class``), as a value of its type is. A number in it must be finite: pydantic's
    parser reads the text with ``allow_inf_nan=False``, so ``inf`` and ``nan`` are refused at any depth, whatever
    the type, ``Any`` and a host's model included, with pydantic's own ``finite_number``. Their texts,
    ``"Infinity"`` and ``"NaN"``, are text, which a ``float`` refuses. A dict's key is held to it too, once the
    validator has read it.

    A host's own check may raise something other than pydantic's error, such as ``float(None)``'s ``TypeError``.
    That is a refusal too, as ``Type.takes_value`` reads it: a ``ValidationError`` with ``value_error``.

    Args:
        validator: The check the value is read by.
        value: The value given.
        title: The title of a refusal, such as the type's id.

    Raises:
        ValidationError: The validator refuses the value, ``finite_number`` for a number in it that isn't finite,
            ``set_has_no_order`` if the value has a set in it, or ``no_json_form`` if it can't be written as JSON,
            a frame among them.
    """
    text = written_as_json(title, value, lambda: finite_json_text(value, title, fallback=_written))
    read = _checked(title, value, lambda: validator.validate_json(text, strict=True))
    # A dict's key is text in the JSON, so the text can't show a key that isn't finite, like dict[float, str]'s
    # "Infinity", until the validator has read it. A saved run is read by its own reader, so it may keep such keys.
    finite_json_text(
        [key for part in value_parts(read) if isinstance(part, dict) for key in part], title
    )
    return read


def _checked(title: str, value: Any, check: Callable[[], Any]) -> Any:
    """Returns what ``check``, a call of a validator on ``value``, gives.

    A host's own check may raise something other than pydantic's error where pydantic would refuse, such as
    ``float(None)``'s ``TypeError``, as ``Type.takes_value`` says. That is a refusal too, so it is raised as a
    ``ValidationError`` with ``value_error``, titled ``title``.
    """
    try:
        return check()
    except ValidationError:
        raise
    except Exception as error:  # noqa: BLE001  a host's own check, which may raise anything
        raise ValidationError.from_exception_data(
            title, [{"type": "value_error", "loc": (), "input": value, "ctx": {"error": error}}]
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
            "a frame is read as it is only where a run is given it for a table, or a union with one; anywhere else,"
            " such as a constant or a choice, give its rows as a list with one object per row"
        )
    return json_by_its_class(value)


def written_as_json[T](title: str, value: Any, write: Callable[[], T]) -> T:
    """Calls ``write``, which writes ``value`` as JSON, and returns what it gives, or refuses a value with no JSON
    form.

    A value JSON can't hold is refused with pydantic's ``ValidationError``, as a type refuses a value, so a caller
    handles one kind of error. The two readers here write a value through it, and so does the validator of an
    answer a host keeps. A set is looked for first, since pydantic writes it as an array by itself (``holds_a_set``).

    Args:
        title: The title of the error, such as the type's id.
        value: The value written.
        write: Writes ``value``. It may raise ``PydanticSerializationError``, which is ``no_json_form``, or a
            refusal of its ``fallback``, such as a frame where the type is no table.

    Raises:
        ValidationError: ``set_has_no_order``, ``no_json_form`` or the fallback's refusal, with an empty ``loc``,
            or whatever ``write`` raises as one.
    """
    try:
        if holds_a_set(value):
            raise set_has_no_order()
        return write()
    except PydanticSerializationError:
        refused = no_json_form()
    except PydanticCustomError as error:
        refused = error
    raise ValidationError.from_exception_data(title, [{"type": refused, "loc": (), "input": value}])
