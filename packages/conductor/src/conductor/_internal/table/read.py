"""How a ``Table[R]`` value is checked and converted, as a pydantic schema.

A table can be given in three forms: a ``polars.DataFrame``, a list of rows (one dict per row), or the JSON
conductor writes for a table (``write.py``). Whatever the form, the schema gives back a frame with exactly the
columns ``R`` names, each of the right type.

There is one more step. A run saves its values as JSON and reads them back later, and some polars types don't
come back the same. An ``Int32`` column comes back as ``Int64``, and a ``Categorical`` as ``String``. So the
schema converts each column to the type it will come back as (``as_saved``). A column that can't be converted
without changing a cell is refused instead. That way a table looks the same before and after a run is saved.

Polars has no type for a host's model, dataclass or enum. A frame keeps such a cell as a struct of its fields or as
its value, and the cell is checked by the row type from that (``cells_by_row``). A Series of the rows reads the
cell back the same way, so a node that gets a row gets the model, wherever the row came from.
"""

from __future__ import annotations

import dataclasses
import datetime
import functools
from collections.abc import Callable, Iterable, Mapping
from decimal import Decimal
from enum import Enum
from typing import Annotated, Any, NotRequired, TypeVar, get_args, get_origin

import polars as pl
from polars.exceptions import InvalidOperationError, PolarsError
from pydantic import BaseModel, GetCoreSchemaHandler, GetJsonSchemaHandler, GetPydanticSchema
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, PydanticCustomError, PydanticSerializationError, core_schema

from conductor._internal.base import json_text, json_typed_dict_schema, validate_as_json
from conductor._internal.problem import (
    OWN_MESSAGE,
    NodeDeclarationError,
    conductor_error,
    declaration_problems,
)
from conductor._internal.types import SIZED, Type, field_hints, is_typed_dict, json_by_its_class

from .write import CELLS, DTYPES, WRITTEN_TABLE, written_table


def column_missing(column: str) -> PydanticCustomError:
    """The error for a table that lacks a column its row type declares.

    A frame without the column raises it here. Rows without the field get pydantic's ``missing`` error, and
    ``TableType`` turns that into this one, so a host sees one code for both.
    """
    return conductor_error(
        "column_missing",
        "the column '{column}' is missing, and the row type needs it",
        {"column": column},
    )


def _frame_of(
    columns: Iterable[tuple[str, list[Any] | pl.Series, Any]], *, strict: bool
) -> pl.DataFrame:
    """Makes a frame from a list of columns.

    Each column is a name, its cells and a polars dtype. With ``strict``, a cell that doesn't fit its column's dtype
    is refused. Without it, the cell is converted to the dtype.

    A dtype of ``None`` lets polars pick one type for the whole column, and every cell is converted to it. So ``1``
    and ``2.5`` become floats, ``True`` and ``2`` become ``1`` and ``2``, and ``"x"`` and ``2`` become text. This
    is never strict. Strict, polars would guess the shape of a nested column from its first cells, and lose a field
    that is null at first and ``5`` later.

    Raises:
        PydanticCustomError: ``not_a_table`` if polars can't make a frame of the columns. That happens for a name
            used twice, columns of different lengths, or cells with no one type in common.
    """
    try:
        return pl.DataFrame(
            [
                pl.Series(name, cells, dtype=dtype, strict=strict and dtype is not None)
                for name, cells, dtype in columns
            ]
        )
    except (PolarsError, TypeError):  # polars' own message may quote a cell, so it isn't passed on
        raise conductor_error(
            "not_a_table",
            "the columns don't make a table: a name may be used twice, the columns may differ in length, or a column's cells may not share one type",
        ) from None


def _frame_written(table: Mapping[str, list[dict[str, Any]]]) -> pl.DataFrame:
    """Makes a frame from a table's JSON, once pydantic has checked its cells.

    Each column gets the polars type its ``type`` word names, strictly. A column written as ``json`` gets the type
    polars picks for its cells.
    """
    return _frame_of(
        (
            (column["name"], column["values"], DTYPES.get(column["type"]))
            for column in table["columns"]
        ),
        strict=True,
    )


def _found_in_json(dtype: Any) -> Any:
    """Returns the polars type a column of ``dtype`` comes back as after it is written as plain JSON.

    This is for a list or struct column, which is written with the word ``json``. Polars picks the type again
    from the JSON. A whole number comes back as ``Int64`` and a float as ``Float64``. A date, a time or a decimal
    comes back as a ``String``, since that is what its JSON is. A list and a struct come back as a ``List`` and a
    ``Struct`` of what is inside them. Cells with no JSON form, like Python objects, give ``None``.
    """
    if isinstance(dtype, pl.List | pl.Array):
        inner = _found_in_json(dtype.inner)
        return None if inner is None else pl.List(inner)
    if isinstance(dtype, pl.Struct):
        fields = {field.name: _found_in_json(field.dtype) for field in dtype.fields}
        return None if None in fields.values() else pl.Struct(fields)
    if dtype == pl.Object:
        return None
    kind = dtype.to_python()
    if kind is int:
        return pl.Int64
    if kind is float:
        return pl.Float64
    if kind is bool:
        return pl.Boolean
    if dtype == pl.Null:
        return pl.Null
    return pl.String


@functools.lru_cache(maxsize=1024)  # a server meets many kinds of Enum and Struct
def _saved_dtype(dtype: Any) -> Any:
    """Returns the polars type a column of ``dtype`` comes back as from a saved run.

    A column is written with the Python class of its cells (``CELLS``), and read back as that class's polars type.
    So an ``Int32`` comes back as ``Int64``, and a ``Categorical`` as ``String``. A ``Datetime`` comes back in
    microseconds, and in UTC if it had a time zone. A ``Decimal`` comes back with a precision of 38 and its own
    scale. A list or a struct is written as plain JSON (``_found_in_json``).

    Returns ``None`` when the column has no JSON form at all.
    """
    kind = dtype.to_python()
    if kind not in CELLS:
        return _found_in_json(dtype)
    if kind is datetime.datetime:
        return pl.Datetime("us", "UTC" if dtype.time_zone else None)
    if kind is datetime.timedelta:
        return pl.Duration("us")
    if kind is Decimal:
        return pl.Decimal(38, dtype.scale)
    return CELLS[kind][0]


def _converts_exactly(dtype: Any, saved: Any) -> bool:
    """Says whether a column of ``dtype`` can be converted to ``saved``, the type it comes back as, with no cell
    changed.

    These convert exactly:

    - a narrower whole number (``Int32``, ``UInt32``) to ``Int64``. A ``UInt64`` too, as long as every cell fits,
      which the cast in ``as_saved`` checks.
    - ``Float32`` to ``Float64``, and ``Categorical`` or ``Enum`` to ``String``.
    - a ``Datetime`` or ``Duration`` in another unit than microseconds. Its cells are checked too (``_kept_as``).
    - a ``Decimal`` of a smaller precision, and a fixed-size ``Array`` to a list.
    - a list or struct whose cells all convert exactly.

    A ``Datetime`` in a time zone other than UTC would change, since it comes back in UTC. So would a date, time
    or decimal inside a list, since it comes back as text.
    """
    if isinstance(dtype, pl.List | pl.Array):
        return isinstance(saved, pl.List) and _converts_exactly(dtype.inner, saved.inner)
    if isinstance(dtype, pl.Struct):
        return isinstance(saved, pl.Struct) and all(
            _converts_exactly(field.dtype, kept.dtype)
            for field, kept in zip(dtype.fields, saved.fields, strict=True)
        )
    if dtype == saved:
        return True
    if isinstance(dtype, pl.Datetime):
        return isinstance(saved, pl.Datetime) and dtype.time_zone in (None, "UTC")
    if isinstance(dtype, pl.Duration | pl.Decimal):
        return isinstance(saved, type(dtype))
    if dtype.is_integer():
        return saved == pl.Int64
    if isinstance(dtype, pl.Categorical | pl.Enum):
        return saved == pl.String
    return dtype == pl.Float32 and saved == pl.Float64


#: The largest number of milliseconds that still fits in an ``Int64`` once it is counted in microseconds.
_MILLISECONDS_IN_RANGE = (2**63 - 1) // 1000


def _finer_than_a_microsecond(column: pl.Series) -> str | None:
    """Checks that every cell of a nanosecond column is a whole number of microseconds.

    This is for a time, or a ``Datetime`` or ``Duration`` in nanoseconds. A saved run counts in microseconds, as
    Python's ``time``, ``datetime`` and ``timedelta`` do, so anything finer would be lost.

    Returns:
        What is wrong with the column, or ``None`` if nothing is.
    """
    return (
        "has a time finer than a microsecond, which a saved run does not keep"
        if (column.cast(pl.Int64) % 1000 != 0).any()
        else None
    )


def _out_of_microseconds(column: pl.Series) -> str | None:
    """Checks that every cell of a millisecond column still fits when it is counted in microseconds.

    This is for a ``Datetime`` or ``Duration`` in milliseconds. A date very far from 1970, or a very long duration,
    would overflow an ``Int64`` once it is multiplied by 1,000.

    Returns:
        What is wrong with the column, or ``None`` if nothing is.
    """
    counts = column.cast(pl.Int64)
    return (
        "has a time too far out to be held in microseconds, which a saved run keeps"
        if ((counts > _MILLISECONDS_IN_RANGE) | (counts < -_MILLISECONDS_IN_RANGE)).any()
        else None
    )


def _floats_inside(column: pl.Series) -> list[pl.Series]:
    """Returns the float cells inside a list or struct column.

    There is one series for each place a float can be. A struct with two float fields gives two series.
    """
    dtype = column.dtype
    if isinstance(dtype, pl.List):
        return _floats_inside(column.explode())
    if isinstance(dtype, pl.Array):
        return _floats_inside(column.arr.explode())
    if isinstance(dtype, pl.Struct):
        return [
            floats
            for field in dtype.fields
            for floats in _floats_inside(column.struct.field(field.name))
        ]
    return [column] if dtype.is_float() else []


def _not_finite_inside(column: pl.Series) -> str | None:
    """Checks that no float inside a list or struct is ``nan`` or ``inf``.

    A list or struct is written as plain JSON, which has no number for ``nan`` or ``inf``. Such a float would be
    written as text and come back as text, not as a float.

    Returns:
        What is wrong with the column, or ``None`` if nothing is.
    """
    if any((floats.is_nan() | floats.is_infinite()).any() for floats in _floats_inside(column)):
        return "has a float that isn't finite (nan or inf) inside a list or struct, and a saved run can't keep it"
    return None


def _casts(column: pl.Series, dtype: Any) -> bool:
    """Says whether ``column`` can be cast to ``dtype`` with every cell in range."""
    try:
        column.cast(dtype, strict=True)
    except InvalidOperationError:
        return False
    return True


def _has_float_inside(dtype: Any) -> bool:
    """Says whether a list or struct of ``dtype`` has floats anywhere inside it."""
    if isinstance(dtype, pl.List | pl.Array):
        return _has_float_inside(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(_has_float_inside(field.dtype) for field in dtype.fields)
    return bool(dtype.is_float())


@functools.lru_cache(maxsize=1024)  # a server meets many kinds of Enum and Struct
def _kept_as(dtype: Any) -> tuple[Any, tuple[Callable[[pl.Series], str | None], ...]]:
    """Returns the type a column of ``dtype`` is converted to, and the checks its cells must pass.

    The type is ``dtype`` itself when the column comes back from a saved run unchanged. It is the type the column
    comes back as when converting changes no cell, so an ``Int32`` column becomes ``Int64``. It is ``None`` when the
    column would come back changed, or can't be written as JSON at all.

    Some columns depend on their cells too, such as a ``Datetime`` in nanoseconds. Those get checks: each takes the
    column and returns what is wrong with it, or ``None``.
    """
    saved = _saved_dtype(dtype)
    if saved is None or not _converts_exactly(dtype, saved):
        return None, ()
    checks: list[Callable[[pl.Series], str | None]] = []
    unit = getattr(dtype, "time_unit", None)
    if dtype == pl.Time or unit == "ns":
        checks.append(_finer_than_a_microsecond)
    if unit == "ms":
        checks.append(_out_of_microseconds)
    if dtype.to_python() not in CELLS and _has_float_inside(dtype):
        checks.append(_not_finite_inside)
    return saved, tuple(checks)


def as_saved(frame: pl.DataFrame) -> pl.DataFrame:
    """Converts each column of ``frame`` to the type it will come back as from a saved run.

    After this, every cell comes back from ``run.to_json()`` and ``Run.from_json`` exactly as it is. A column is
    converted when that changes no cell, and refused otherwise. Mostly the column's type decides, but for a few
    types the cells are checked too.

    Some examples:

    - an ``Int32`` or ``UInt32`` column becomes ``Int64``, and a ``Categorical`` becomes ``String``.
    - a ``UInt64`` column becomes ``Int64`` if every cell fits.
    - a ``Datetime`` in nanoseconds becomes microseconds if every cell is a whole microsecond.
    - a ``Datetime`` in ``"Europe/Copenhagen"`` is refused, since it would come back in UTC.
    - a ``nan`` inside a list is refused.

    A column with no values, empty or all null, may still come back with another type. There is no cell to tell
    what it was.

    Raises:
        PydanticCustomError: ``column_not_saved`` if a column would come back changed.
    """
    casts: dict[Any, Any] = {}
    for name, dtype in frame.schema.items():
        kept, checks = _kept_as(dtype)
        if kept is None:
            saved = _saved_dtype(dtype)
            if saved is None:
                raise conductor_error(
                    "column_not_saved",
                    "column '{column}' is {type}, which can't be written as JSON",
                    {"column": name, "type": str(dtype)},
                )
            raise conductor_error(
                "column_not_saved",
                "column '{column}' is {type}, and a saved run would read it back as {saved}",
                {"column": name, "type": str(dtype), "saved": str(saved)},
            )
        for check in checks:
            if (fault := check(frame[name])) is not None:
                raise conductor_error(
                    "column_not_saved", "column '{column}' " + fault, {"column": name}
                )
        if kept != dtype:
            casts[name] = kept
    if not casts:
        return frame
    try:
        return frame.cast(casts, strict=True)
    # a UInt64 cell too big for an Int64, in the column itself or inside a list or struct
    except InvalidOperationError:
        failed = next(name for name, kept in casts.items() if not _casts(frame[name], kept))
        raise conductor_error(
            "column_not_saved",
            "column '{column}' is {type}, and has a whole number too big for an Int64, which is what a saved run reads its whole numbers back as",
            {"column": failed, "type": str(frame.schema[failed])},
        ) from None


def _table_form(value: Any) -> str | None:
    """Says which of a table's three forms ``value`` is in: a frame, a list of rows, or conductor's JSON."""
    if isinstance(value, pl.DataFrame):
        return "frame"
    if isinstance(value, Mapping) and "columns" in value:
        return "written"  # a dict without it, such as one row, is no table
    if isinstance(value, list):
        return "rows"
    return None


def is_bare_table_row(hint: Any) -> bool:
    """Whether ``hint`` is the type of one row of a bare ``Table``: ``Any``, ``dict`` or ``dict[str, Any]``.

    A frame fills each of these without a check: a row is a dict with a text key per column and a cell of any type.
    A dict with one type of value, such as ``dict[str, int]``, isn't one, since a bare table checks no cell.
    """
    return hint is Any or hint is dict or hint == dict[str, Any]


def check_table_row(row: Any) -> None:
    """Refuses ``row``, the row type ``R`` of a ``Table[R]`` a node declares, if it names no columns.

    A ``TypedDict`` names a table's columns, a generic one's too, and ``Any``, ``dict`` and ``dict[str, Any]`` are a
    bare table's row (``is_bare_table_row``). Any other row type, such as a dataclass or ``int``, names none, so the
    table would keep every column it was given and check none of them. A type variable is filled in by a graph, so
    it is let through. A graph that fills in ``T`` in ``Table[T] | list[T]`` with ``str`` is no declaration, so its
    ``Table[str]`` is read as a bare table.

    Raises:
        NodeDeclarationError: ``table_row_not_a_typed_dict``.
    """
    if isinstance(row, TypeVar) or is_bare_table_row(row) or is_typed_dict(row):
        return
    raise NodeDeclarationError(
        declaration_problems.table_row_not_a_typed_dict(
            row=row.__qualname__ if isinstance(row, type) else str(row)
        )
    )


@functools.lru_cache(maxsize=1024)  # a server meets many row types
def _column_types(row: Any) -> dict[str, Type]:
    """The type of each column of a table of rows of type ``row``, by name. A row that isn't a ``TypedDict``, the row
    of a bare table, names no columns."""
    return (
        {name: Type(hint) for name, hint in field_hints(row).items()} if is_typed_dict(row) else {}
    )


@functools.lru_cache(maxsize=1024)  # a server meets many row types
def _columns_type(row: Any) -> Type:
    """The type of some of a table's columns, each a list of cells, by name: ``{"money": [{"amount": 1.0}]}``.

    Each column's cells are checked by the row's own schema of its field, as one item of a Series of the rows is. So a
    cell of a class that takes the config it sits in, like a plain dataclass, is read by the row's config. A column
    may be left out, so only the columns that need it are read.
    """

    def columns(_: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        fields = handler.resolve_ref_schema(handler.generate_schema(row))["fields"]
        return json_typed_dict_schema(
            {
                name: core_schema.typed_dict_field(
                    core_schema.list_schema(field["schema"]), required=False
                )
                for name, field in fields.items()
            },
            cls=get_origin(row) or row,
        )

    return Type(Annotated[Any, GetPydanticSchema(columns)])


def _read_in_python(column: Type, dtype: Any) -> bool:
    """Whether a column of ``dtype`` must be read in Python to give values of ``column``, its declared type.

    Polars gives each cell as a plain Python value: a ``String`` column's cell is a ``str``, and a ``Struct``
    column's a ``dict``. That is already a value of a ``str`` or a ``float`` column, which most of a table is. It
    isn't a value of a host's model, an enum or a list, so such a column is read.
    """
    return not column.holds(Type(dtype.to_python()))


def cells_by_row(
    row: Any, dtypes: Mapping[str, Any], columns: dict[str, list[Any]]
) -> dict[str, list[Any]]:
    """Returns the cells of a frame's ``columns``, each a value of its column's type in the row type ``row``.

    ``columns`` is the cells of each column as polars gives them (``frame[name].to_list()``), and ``dtypes`` the
    frame's ``schema``. A column polars gives values of its type for, like a ``str`` or ``float`` column, is
    returned as it is, the same list. Any other column is read by the row's own schema of its field, from the JSON
    its cells are written as, the way a saved table is read. So a struct cell becomes the host's model or
    dataclass, a text cell the enum's member, and a list of structs a list of models. Say ``Row`` has a field
    ``money: Money``: the column ``{"money": [{"amount": 1.0}]}`` gives ``{"money": [Money(amount=1.0)]}``.

    A table's check reads its frame's cells here, and so does a Series of rows a node returned as a frame.

    Raises:
        ValidationError: A cell its column's type refuses. The ``loc`` starts with the column and the row's index.
        PydanticCustomError: ``column_not_saved`` for a column of Python objects with no JSON form.
    """
    types = _column_types(row)
    read = {
        name: cells
        for name, cells in columns.items()
        if name in types and _read_in_python(types[name], dtypes[name])
    }
    if not read:
        return columns
    try:
        return columns | validate_as_json(
            _columns_type(row).argument_validator, read, fallback=json_by_its_class
        )
    # a column of Python objects, such as a host's class with no schema that writes it
    except PydanticSerializationError:
        failed = next(name for name, cells in read.items() if not _has_json_form(cells))
        raise conductor_error(
            "column_not_saved",
            "column '{column}' is {type}, which can't be written as JSON",
            {"column": failed, "type": str(dtypes[failed])},
        ) from None


def _has_json_form(cells: list[Any]) -> bool:
    """Whether the cells of one column can be written as JSON, each object by its own class's schema."""
    try:
        json_text(cells, fallback=json_by_its_class)
    except PydanticSerializationError:
        return False
    return True


def _plain_column(cells: list[Any]) -> pl.Series | list[Any]:
    """Returns the cells of a column of rows whose type polars picks, as polars can keep them (``_as_polars_keeps``).

    Polars makes a column of models or of enum members plain by itself, and that is tried first, since it is fast.
    Only a column it can't make plain, such as one of lists of models, is made plain here, cell by cell.
    """
    try:
        column = pl.Series(cells, strict=False)
    # polars refuses some objects inside a struct, such as an enum member
    except (PolarsError, TypeError):
        return [_as_polars_keeps(cell) for cell in cells]
    return [_as_polars_keeps(cell) for cell in cells] if column.dtype == pl.Object else column


def _as_polars_keeps(cell: Any) -> Any:
    """Returns ``cell`` as a frame keeps it: a value polars has a type for as it is, and a host's model, dataclass or
    enum member as its fields or its value.

    Polars has no type for a host's model, dataclass or enum member. It turns one into a struct or a text by itself,
    but keeps one inside a list or a dict as a Python object, which a saved run can't write. So each is made plain
    here, wherever it is: ``Money(amount=1.0)`` becomes the dict of its fields ``{"amount": 1.0}``, ``Colour.RED``
    its value ``"red"``, and a list of models a list of such dicts. This is what polars makes of a model outside a
    list too, so a frame keeps one the same way whether a node gave it in a list of rows or in a frame. Reading the
    frame by its row type (``cells_by_row``) gives the model back.
    """
    kind = type(cell)
    if cell is None or kind in CELLS:
        return cell
    if kind is list or kind is tuple:
        return [_as_polars_keeps(one) for one in cell]
    if kind is dict:
        return {key: _as_polars_keeps(value) for key, value in cell.items()}
    if isinstance(cell, Enum):
        return _as_polars_keeps(cell.value)
    if isinstance(cell, BaseModel) or dataclasses.is_dataclass(cell):
        return _as_polars_keeps(Type(kind).serializer.to_python(cell))
    # any other object stays one, and a saved run refuses its column (``as_saved``)
    return cell


def table_schema(source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
    """Returns the pydantic schema of ``Table[R]``.

    The schema accepts a table in any of its three forms: a frame, a list of rows like ``[{"name": "Ann"}]``,
    or conductor's JSON, ``{"columns": [...]}``. It always gives back a frame with exactly the columns ``R`` names,
    converted by ``as_saved``. It writes a table as conductor's JSON.

    Args:
        source: The hint, such as ``Table[Invoice]``.
        handler: Pydantic's handler, which builds the schema of each column's type.

    Raises:
        NodeDeclarationError: ``row_field_optional`` if the row type lets a row leave a column out.
    """
    row = (get_args(source) or (Any,))[0]
    origin = get_origin(row) or row
    # only a TypedDict names a table's columns
    declared = is_typed_dict(row)
    hints = field_hints(row) if declared else {}
    # a column some rows could leave out: a NotRequired field, or any field of a total=False row
    left_out = [name for name, hint in hints.items() if get_origin(hint) is NotRequired]
    if left_out:
        raise NodeDeclarationError(
            declaration_problems.row_field_optional(row=origin.__qualname__, field=left_out[0])
        )
    columns = _column_types(row)
    # each cell is checked by the row's own schema of its field, as one item of a Series of the rows is, so a cell
    # of a class that takes the config it sits in, like a plain dataclass, is built under the row's config
    fields = handler.resolve_ref_schema(handler.generate_schema(row))["fields"] if declared else {}
    cells = {name: fields[name]["schema"] for name in columns}
    dtypes = {
        name: CELLS[column.options[0]][0]
        if column.options[0] in CELLS and len(column.options) == 1
        else None
        for name, column in columns.items()
    }

    def from_rows(rows: list[dict[str, Any]]) -> pl.DataFrame:
        names = columns if declared else dict.fromkeys(key for one in rows for key in one)
        return _frame_of(
            (
                (
                    name,
                    _plain_column([one.get(name) for one in rows])
                    if declared and dtypes[name] is None
                    else [one.get(name) for one in rows],
                    dtypes.get(name),
                )
                for name in names
            ),
            strict=False,
        )

    def from_frame(frame: pl.DataFrame) -> pl.DataFrame:
        # a table with no rows still has every column, empty and of its declared type
        if not frame.height:
            kept = {name for name, dtype in frame.schema.items() if dtype != pl.Null}
            return pl.DataFrame(
                [
                    frame[name] if name in kept else pl.Series(name, [], dtype=dtypes[name])
                    for name in columns
                ]
            )
        for name, column in columns.items():
            if name not in frame.columns:
                raise column_missing(name)
            if not column.optional and frame[name].null_count():
                raise conductor_error(
                    "column_blank",
                    "column '{column}' has a blank cell, and its type doesn't allow blanks",
                    {"column": name},
                )
        read = cells_by_row(
            row,
            frame.schema,
            {
                name: frame[name].to_list()
                for name, column in columns.items()
                if _read_in_python(column, frame.schema[name])
            },
        )
        # A cell of one of polars' own types, like a date given as text, is kept as the value it was read as. Any
        # other cell, like a host's model, stays as polars holds it, its fields or its value, which a saved run
        # writes; a Series of the rows reads it again by the row type when it makes a row
        converted = {name: cells for name, cells in read.items() if dtypes[name] is not None}
        if converted:
            frame = frame.with_columns(
                pl.Series(name, cells, strict=False) for name, cells in converted.items()
            )
        return frame if frame.columns == list(columns) else frame.select(list(columns))

    def json_schema(_: CoreSchema, write: GetJsonSchemaHandler) -> JsonSchemaValue:
        # pydantic's generator writes a cell as it writes any schema, with what a Param adds to it, like the enum
        # of its choices; calling ``write`` itself would write only the bare type
        generator = write.generate_json_schema  # type: ignore[attr-defined]
        column = [
            {
                "type": "object",
                "properties": {
                    "name": {"const": name},
                    "values": {"type": "array", "items": generator.generate_inner(cell)},
                },
            }
            for name, cell in cells.items()
        ]
        return {
            "type": "object",
            "properties": {
                "columns": {"type": "array", "prefixItems": column}
                if declared
                else {"type": "array"}
            },
        }

    # each written with its row class, so a fault in a column names the row's field, as in a bare TypedDict; a row
    # is read by the row class's own config when it has one, else by conductor's, so a bytes cell reads base64
    row_class = origin if declared else None
    one_row = (
        json_typed_dict_schema(
            {name: core_schema.typed_dict_field(cell) for name, cell in cells.items()},
            cls=row_class,
        )
        if declared
        else core_schema.dict_schema(core_schema.str_schema())
    )
    as_frame = (
        core_schema.no_info_plain_validator_function(from_frame)
        if declared
        else core_schema.is_instance_schema(pl.DataFrame)
    )
    written = core_schema.no_info_after_validator_function(_frame_written, WRITTEN_TABLE)
    any_form = core_schema.tagged_union_schema(
        {
            "frame": as_frame,
            "rows": core_schema.no_info_after_validator_function(
                from_rows, core_schema.list_schema(one_row)
            ),
            "written": core_schema.chain_schema([written, as_frame]),
        },
        discriminator=_table_form,
        custom_error_type="not_a_table",
        custom_error_message="a table must be a frame, or a list with one object per row",
        custom_error_context=dict(OWN_MESSAGE),
    )
    kept = core_schema.no_info_after_validator_function(as_saved, any_form)
    # The same schema for JSON and for Python. It is deliberately not a function schema at the top: that way, a
    # limit pydantic has no rule for on a table, like ``Param(ge=1)``, is reported when the node is declared. It says
    # its values are sized (SIZED), so a length limit, like ``Param(min_length=1)``, counts the table's rows.
    return core_schema.json_or_python_schema(
        json_schema=kept,
        python_schema=kept,
        serialization=core_schema.plain_serializer_function_ser_schema(
            written_table, when_used="json"
        ),
        metadata={"pydantic_js_functions": [json_schema], SIZED: True},
    )
