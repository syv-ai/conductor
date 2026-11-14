"""The pydantic schema of ``Table[R]``: how a table is read from a frame, from a list of records or from its JSON form,
and checked against the columns ``R`` names.
"""

from __future__ import annotations

import datetime
import functools
import typing
from collections.abc import Callable, Iterable, Mapping
from decimal import Decimal
from typing import Any, NotRequired, get_args

import polars as pl
from polars.exceptions import InvalidOperationError, PolarsError
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, PydanticCustomError, core_schema

from conductor._internal.problem import NodeDeclarationError, declaration_problems
from conductor._internal.types import Type

from .write import CELLS, DTYPES, WRITTEN_TABLE, written_table


def _frame_of(columns: Iterable[tuple[str, list[Any], Any]], *, strict: bool) -> pl.DataFrame:
    """Makes a frame of ``columns``, each a name, its cells and its polars dtype (``None`` to let polars find one).
    ``strict`` refuses a cell that doesn't fit its column's dtype; otherwise the cell is converted to it. A column
    with no dtype is read laxly either way: polars finds one type for the column and converts every cell to it
    (``1`` and ``2.5`` as floats, ``True`` and ``2`` as ``1`` and ``2``, ``"x"`` and ``2`` as text). Read strictly,
    polars would take a nested column's shape from its first cells and lose what later cells hold (a field that is
    null at first and ``5`` later).

    Raises:
        PydanticCustomError: Polars can't make a table of them (``not_a_table``): a name twice, lengths that
            differ, or cells of no one type.
    """
    try:
        return pl.DataFrame([pl.Series(name, cells, dtype=dtype, strict=strict and dtype is not None) for name, cells, dtype in columns])
    except (PolarsError, TypeError):  # polars' own account may quote a cell
        raise PydanticCustomError("not_a_table", "the columns don't make a table: one name twice, lengths that differ, or cells of no one type") from None


def _frame_written(table: Mapping[str, list[dict[str, Any]]]) -> pl.DataFrame:
    """Makes the frame a table's JSON form, its cells already read, says: each column of the polars type its word
    names, read strictly, or of the type polars finds for a column written as ``json``."""
    return _frame_of(((column["name"], column["values"], DTYPES.get(column["type"])) for column in table["columns"]), strict=True)


def _found_in_json(dtype: Any) -> Any:
    """Returns the polars type polars finds for cells of ``dtype`` inside a column written as ``json``: a number
    as ``Int64`` or ``Float64``, a date, a time or a decimal as the ``String`` its JSON text is, a list as a
    ``List`` and a struct as a ``Struct`` of what it holds; ``None`` for cells that have no JSON form."""
    if isinstance(dtype, pl.List | pl.Array):
        inner = _found_in_json(dtype.inner)
        return None if inner is None else pl.List(inner)
    if isinstance(dtype, pl.Struct):
        fields = {field.name: _found_in_json(field.dtype) for field in dtype.fields}
        return None if None in fields.values() else pl.Struct(fields)
    if dtype == pl.Object:
        return None
    kind = dtype.to_python()
    return pl.Int64 if kind is int else pl.Float64 if kind is float else pl.Boolean if kind is bool else pl.Null if dtype == pl.Null else pl.String


@functools.lru_cache(maxsize=1024)  # a server sees many shapes of Enum and Struct
def _saved_dtype(dtype: Any) -> Any:
    """Returns the polars type a saved run reads a column of ``dtype`` back as, or ``None`` when the column has no
    JSON form. The write names a column's cells by their Python class (``CELLS``), and the read makes a column of
    that word's polars type, so ``Int32`` is read back as ``Int64`` and ``Categorical`` as ``String``; a
    ``Datetime`` comes back in microseconds and, when it has a time zone, in UTC, a ``Decimal`` with the precision
    38 and its own scale. A list or a struct is written as JSON, and polars finds its type in that."""
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
    """Says whether a column of ``dtype`` becomes the ``saved`` type it reads back as with no cell changed: a
    narrower whole number (``Int32``, ``UInt32``; a ``UInt64`` too, whose cells the cast checks) as ``Int64``,
    ``Float32`` as ``Float64``, ``Categorical`` and ``Enum`` as ``String``, a ``Datetime`` or ``Duration`` in
    another unit than microseconds (whose cells ``_cell_checks`` checks), a ``Decimal`` of a smaller precision, a
    fixed-size ``Array`` as a list, and a list or struct whose cells convert so. A ``Datetime`` in a time zone other
    than UTC, and a date, time or decimal inside a list (written as text), would change."""
    if isinstance(dtype, pl.List | pl.Array):
        return isinstance(saved, pl.List) and _converts_exactly(dtype.inner, saved.inner)
    if isinstance(dtype, pl.Struct):
        return isinstance(saved, pl.Struct) and all(_converts_exactly(field.dtype, kept.dtype) for field, kept in zip(dtype.fields, saved.fields, strict=True))
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


#: The largest number of milliseconds that is a whole number of microseconds an ``Int64`` holds.
_MILLISECONDS_IN_RANGE = (2**63 - 1) // 1000


def _finer_than_a_microsecond(column: pl.Series) -> str | None:
    """Says so when a cell of a time, a ``Datetime`` or a ``Duration`` in nanoseconds is not a whole number of
    microseconds, which a saved run (and Python's ``time``, ``datetime`` and ``timedelta``) holds."""
    return "has a time finer than a microsecond, which a saved run does not keep" if (column.cast(pl.Int64) % 1000 != 0).any() else None


def _out_of_microseconds(column: pl.Series) -> str | None:
    """Says so when a cell of a ``Datetime`` or ``Duration`` in milliseconds is too far from 1970, or too long, to
    be held in microseconds."""
    counts = column.cast(pl.Int64)
    return "has a time too far out to be held in microseconds, which a saved run keeps" if ((counts > _MILLISECONDS_IN_RANGE) | (counts < -_MILLISECONDS_IN_RANGE)).any() else None


def _floats_inside(column: pl.Series) -> list[pl.Series]:
    """Returns the float cells inside a list or struct column, as one series per place they are at."""
    dtype = column.dtype
    if isinstance(dtype, pl.List):
        return _floats_inside(column.explode())
    if isinstance(dtype, pl.Array):
        return _floats_inside(column.arr.explode())
    if isinstance(dtype, pl.Struct):
        return [floats for field in dtype.fields for floats in _floats_inside(column.struct.field(field.name))]
    return [column] if dtype.is_float() else []


def _not_finite_inside(column: pl.Series) -> str | None:
    """Says so when a float inside a list or struct is not finite: a list or struct is written as plain JSON, which
    has no number for ``nan`` or ``inf``, so it would be written as text and not read back as a float."""
    if any((floats.is_nan() | floats.is_infinite()).any() for floats in _floats_inside(column)):
        return "has a float that is not finite inside a list or struct, which a saved run does not keep"
    return None


def _casts(column: pl.Series, dtype: Any) -> bool:
    """Says whether ``column`` casts to ``dtype`` with no cell out of its range."""
    try:
        column.cast(dtype, strict=True)
    except InvalidOperationError:
        return False
    return True


def _has_float_inside(dtype: Any) -> bool:
    """Says whether a list or struct of ``dtype`` holds floats, at any depth."""
    if isinstance(dtype, pl.List | pl.Array):
        return _has_float_inside(dtype.inner)
    if isinstance(dtype, pl.Struct):
        return any(_has_float_inside(field.dtype) for field in dtype.fields)
    return bool(dtype.is_float())


@functools.lru_cache(maxsize=1024)  # a server sees many shapes of Enum and Struct
def _kept_as(dtype: Any) -> tuple[Any, tuple[Callable[[pl.Series], str | None], ...]]:
    """Returns the polars type a table keeps a column of ``dtype`` as, and the checks of its cells that decide
    whether it is kept. The type is ``dtype`` itself when a saved run reads it back as it is, the type it reads back
    as when converting to it changes no cell (``Int32`` is kept as ``Int64``), and ``None`` when the column would
    read back changed or has no JSON form."""
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
    """Returns ``frame`` as a run keeps it: each column of the type a saved run reads it back as, so every cell reads
    back from a save as it is. A column is converted when no cell changes, and refused otherwise. It is decided by
    each column's type, and for a few types by their cells too. A column with no value at all may still read back
    with another type (an empty or all-null column has no cell to say what it is).

    Example: an ``Int32`` or ``UInt32`` column is kept as ``Int64`` and a ``Categorical`` as ``String``; a ``UInt64``
    is kept as ``Int64`` when every cell fits, and a ``Datetime`` in nanoseconds in microseconds when every cell is
    a whole microsecond. A ``Datetime`` in ``"Europe/Copenhagen"`` is refused (a saved run reads it back in UTC), and
    so is a ``nan`` inside a list.

    Raises:
        PydanticCustomError: A column would read back changed (``column_not_saved``).
    """
    casts: dict[Any, Any] = {}
    for name, dtype in frame.schema.items():
        kept, checks = _kept_as(dtype)
        if kept is None:
            saved = _saved_dtype(dtype)
            if saved is None:
                raise PydanticCustomError("column_not_saved", "column '{column}' is {type}, which has no JSON form", {"column": name, "type": str(dtype)})
            raise PydanticCustomError(
                "column_not_saved", "column '{column}' is {type}, which a saved run reads back as {saved}; cast it to {saved}",
                {"column": name, "type": str(dtype), "saved": str(saved)},
            )
        for check in checks:
            if (fault := check(frame[name])) is not None:
                raise PydanticCustomError("column_not_saved", "column '{column}' " + fault, {"column": name})
        if kept != dtype:
            casts[name] = kept
    if not casts:
        return frame
    try:
        return frame.cast(casts, strict=True)
    except InvalidOperationError:  # a UInt64 past the largest Int64, at the top or inside a list or struct
        failed = next(name for name, kept in casts.items() if not _casts(frame[name], kept))
        raise PydanticCustomError(
            "column_not_saved", "column '{column}' is {type} and has a whole number past the largest Int64, which a saved run reads it back as",
            {"column": failed, "type": str(frame.schema[failed])},
        ) from None


def _table_form(value: Any) -> str | None:
    """Says which of a table's three forms ``value`` is: a frame, a list of records, or the columns conductor writes."""
    if isinstance(value, pl.DataFrame):
        return "frame"
    return "written" if isinstance(value, Mapping) else "records" if isinstance(value, list) else None


def table_schema(source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
    """Returns the pydantic schema of ``Table[R]``, where ``source`` is that hint.

    The schema validates a table given as a frame, as a list of records (``[{"name": "Ann"}]``) or as the JSON form
    (``{"columns": [...]}``), and always returns a frame with exactly the columns ``R`` names. It writes a table as
    the JSON form. Whatever form it is given in, a table is returned with each column converted to the type a saved
    run reads it back as, or refused when a cell would change (``as_saved``), so every cell a run keeps reads back
    from ``run.to_json()`` by ``Run.from_json`` as it is.
    """
    row = (get_args(source) or (Any,))[0]
    declared = typing.is_typeddict(row)
    hints = typing.get_type_hints(row, include_extras=True) if declared else {}
    # a NotRequired field, or total=False, is a column some rows would leave out; a NotRequired written as text is seen only in the hints
    left_out = [name for name, hint in hints.items() if name in row.__optional_keys__ or typing.get_origin(hint) is NotRequired]
    if left_out:
        raise NodeDeclarationError(declaration_problems.row_field_optional(row=row.__qualname__, field=left_out[0]))
    columns = {name: Type(hint) for name, hint in hints.items()}
    cells = {name: handler.generate_schema(column.carried) for name, column in columns.items()}
    dtypes = {name: CELLS[column.options[0]][0] if column.options[0] in CELLS and len(column.options) == 1 else None for name, column in columns.items()}

    def from_records(records: list[dict[str, Any]]) -> pl.DataFrame:
        names = columns if declared else dict.fromkeys(key for record in records for key in record)
        return _frame_of(((name, [record.get(name) for record in records], dtypes.get(name)) for name in names), strict=False)

    def from_frame(frame: pl.DataFrame, read_columns: Callable[[dict[str, list[Any]]], dict[str, list[Any]]]) -> pl.DataFrame:
        if not frame.height:  # a table of no rows holds every column, empty and of its declared type
            kept = {name for name, dtype in frame.schema.items() if dtype != pl.Null}
            return pl.DataFrame([frame[name] if name in kept else pl.Series(name, [], dtype=dtypes[name]) for name in columns])
        for name, column in columns.items():
            if name not in frame.columns:
                raise PydanticCustomError("column_missing", "the column '{column}' its row type declares is missing", {"column": name})
            if not column.optional and frame[name].null_count():
                raise PydanticCustomError("column_blank", "column '{column}': a cell is blank, and the column's type takes no blank", {"column": name})
        unread = [name for name, column in columns.items() if not column.holds(Type(frame.schema[name].to_python()))]
        if unread:
            read = read_columns({name: frame[name].to_list() for name in unread})
            frame = frame.with_columns(pl.Series(name, values, strict=False) for name, values in read.items())
        return frame if frame.columns == list(columns) else frame.select(list(columns))

    def json_schema(_: CoreSchema, write: GetJsonSchemaHandler) -> JsonSchemaValue:
        column = [
            {"type": "object", "properties": {"name": {"const": name}, "values": {"type": "array", "items": write(cell)}}}
            for name, cell in cells.items()
        ]
        return {"type": "object", "properties": {"columns": {"type": "array", "prefixItems": column} if declared else {"type": "array"}}}

    record = core_schema.typed_dict_schema({name: core_schema.typed_dict_field(cell) for name, cell in cells.items()}) if declared else core_schema.dict_schema(core_schema.str_schema())
    by_column = core_schema.typed_dict_schema({name: core_schema.typed_dict_field(core_schema.list_schema(cell), required=False) for name, cell in cells.items()})
    as_frame = core_schema.no_info_wrap_validator_function(from_frame, by_column) if declared else core_schema.is_instance_schema(pl.DataFrame)
    written = core_schema.no_info_after_validator_function(_frame_written, WRITTEN_TABLE)
    any_form = core_schema.tagged_union_schema(
        {
            "frame": as_frame,
            "records": core_schema.no_info_after_validator_function(from_records, core_schema.list_schema(record)),
            "written": core_schema.chain_schema([written, as_frame]),
        },
        discriminator=_table_form,
        custom_error_type="not_a_table", custom_error_message="a table is a frame or a list of records, one object per row",
    )
    kept = core_schema.no_info_after_validator_function(as_saved, any_form)
    # The same in JSON and in Python. Not a function schema itself, so a limit pydantic holds no rule for on a table
    # (``Param(min_length=1)``) is seen as one when the node is declared.
    return core_schema.json_or_python_schema(
        json_schema=kept, python_schema=kept,
        serialization=core_schema.plain_serializer_function_ser_schema(written_table, when_used="json"),
        metadata={"pydantic_js_functions": [json_schema]},
    )
