"""How a ``Table[R]`` value is checked and converted, as a pydantic schema.

A table can be given in three forms: a ``polars.DataFrame``, a list of rows (one dict per row), or the JSON
conductor writes for a table (``write.py``). Whatever the form, the schema gives back a frame with exactly the
columns ``R`` names, each of the right type.

There is one more step. A run saves its values as JSON and reads them back later, and some polars types don't
come back the same. An ``Int32`` column comes back as ``Int64``, and a ``Categorical`` as ``String``. So the
schema converts each column to the type it will come back as (``as_saved``). A column that can't be converted
without changing a cell is refused instead. That way a table looks the same before and after a run is saved. A
list or struct is saved as plain JSON, and a column its row type declares is read back by its type, so a date
inside it comes back as a date.

Polars has no type for a host's model, dataclass or enum. A frame keeps such a cell as a struct of its fields or as
its value. A frame's cell is checked by the row type from that, and kept as polars holds what the check gave. A
Series of the rows keeps the cells the check read (``table_and_cells_schema``), so a node that gets a row gets the
model, wherever the row came from.
"""

from __future__ import annotations

import datetime
import functools
from collections.abc import Callable, Iterable, Mapping
from decimal import Decimal
from typing import Any, NotRequired, TypeVar, get_args, get_origin

import polars as pl
from polars.exceptions import InvalidOperationError, PolarsError
from pydantic import (
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    ValidatorFunctionWrapHandler,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, PydanticCustomError, PydanticSerializationError, core_schema

from conductor._internal.base import json_text, json_typed_dict_schema
from conductor._internal.problem import (
    OWN_MESSAGE,
    NodeDeclarationError,
    conductor_error,
    declaration_problems,
)
from conductor._internal.types import (
    SIZED,
    Type,
    field_hints,
    is_typed_dict,
    json_by_its_class,
    type_fields,
)

from .cells import cell_not_kept, cell_type, kept_column
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
def _saved_dtype(dtype: Any, typed: bool) -> Any:
    """Returns the polars type a column of ``dtype`` comes back as from a saved run.

    A column is written with the Python class of its cells (``CELLS``), and read back as that class's polars type.
    So an ``Int32`` comes back as ``Int64``, and a ``Categorical`` as ``String``. A ``Datetime`` comes back in
    microseconds, and in UTC if it had a time zone. A ``Decimal`` comes back with a precision of 38 and its own
    scale.

    A list or a struct is written as plain JSON. A column a row type declares (``typed``) is read back by its type,
    so each value inside it comes back as its own class's polars type, as it would in a column of its own: a
    ``List(Date)`` comes back as a ``List(Date)``. Any other list or struct comes back as polars reads the JSON
    (``_found_in_json``), so the same column would come back as a ``List(String)``.

    Returns ``None`` when the column has no JSON form at all.
    """
    if typed and isinstance(dtype, pl.List | pl.Array):
        inner = _saved_dtype(dtype.inner, typed)
        return None if inner is None else pl.List(inner)
    if typed and isinstance(dtype, pl.Struct):
        fields = {field.name: _saved_dtype(field.dtype, typed) for field in dtype.fields}
        return None if None in fields.values() else pl.Struct(fields)
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
    or decimal inside a list no row type declares, since it comes back as text.
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
    """Checks that every time, and every ``Datetime`` or ``Duration`` in nanoseconds, in the column or inside its
    lists and structs, is a whole number of microseconds.

    A saved run counts in microseconds, as Python's ``time``, ``datetime`` and ``timedelta`` do, so anything finer
    would be lost.

    Returns:
        What is wrong with the column, or ``None`` if nothing is.
    """
    if any((times.cast(pl.Int64) % 1000 != 0).any() for times in _inside(column, _in_nanoseconds)):
        return "has a time finer than a microsecond, which a saved run does not keep"
    return None


def _out_of_microseconds(column: pl.Series) -> str | None:
    """Checks that every ``Datetime`` or ``Duration`` in milliseconds, in the column or inside its lists and
    structs, still fits when it is counted in microseconds.

    A date very far from 1970, or a very long duration, would overflow an ``Int64`` once it is multiplied by 1,000.

    Returns:
        What is wrong with the column, or ``None`` if nothing is.
    """
    for times in _inside(column, _in_milliseconds):
        counts = times.cast(pl.Int64)
        if ((counts > _MILLISECONDS_IN_RANGE) | (counts < -_MILLISECONDS_IN_RANGE)).any():
            return "has a time too far out to be held in microseconds, which a saved run keeps"
    return None


def _in_nanoseconds(dtype: Any) -> bool:
    """Says whether a column of ``dtype`` counts in nanoseconds: a time, or a ``Datetime`` or ``Duration`` in ns."""
    return bool(dtype == pl.Time) or getattr(dtype, "time_unit", None) == "ns"


def _in_milliseconds(dtype: Any) -> bool:
    """Says whether a column of ``dtype`` is a ``Datetime`` or ``Duration`` in milliseconds."""
    return getattr(dtype, "time_unit", None) == "ms"


def _is_float(dtype: Any) -> bool:
    """Says whether a column of ``dtype`` holds floats."""
    return bool(dtype.is_float())


def _inside(column: pl.Series, kind: Callable[[Any], bool]) -> list[pl.Series]:
    """Returns the cells of the column, and those inside its lists and structs, of a polars type ``kind`` takes.

    There is one series for each place such a cell can be. A struct with two float fields gives two series for
    ``_is_float``.
    """
    dtype = column.dtype
    if isinstance(dtype, pl.List):
        return _inside(column.explode(), kind)
    if isinstance(dtype, pl.Array):
        return _inside(column.arr.explode(), kind)
    if isinstance(dtype, pl.Struct):
        return [
            cells
            for field in dtype.fields
            for cells in _inside(column.struct.field(field.name), kind)
        ]
    return [column] if kind(dtype) else []


def _has_inside(dtype: Any, kind: Callable[[Any], bool]) -> bool:
    """Says whether a column of ``dtype`` has cells of a polars type ``kind`` takes, itself or inside its lists and
    structs."""
    if isinstance(dtype, pl.List | pl.Array):
        return _has_inside(dtype.inner, kind)
    if isinstance(dtype, pl.Struct):
        return any(_has_inside(field.dtype, kind) for field in dtype.fields)
    return kind(dtype)


def _not_finite_inside(column: pl.Series) -> str | None:
    """Checks that no float inside a list or struct is ``nan`` or ``inf``.

    A list or struct is written as plain JSON, which has no number for ``nan`` or ``inf``. Such a float would be
    written as text and come back as text, not as a float.

    Returns:
        What is wrong with the column, or ``None`` if nothing is.
    """
    if any((floats.is_nan() | floats.is_infinite()).any() for floats in _inside(column, _is_float)):
        return "has a float that isn't finite (nan or inf) inside a list or struct, and a saved run can't keep it"
    return None


def _casts(column: pl.Series, dtype: Any) -> bool:
    """Says whether ``column`` can be cast to ``dtype`` with every cell in range."""
    try:
        column.cast(dtype, strict=True)
    except InvalidOperationError:
        return False
    return True


@functools.lru_cache(maxsize=1024)  # a server meets many kinds of Enum and Struct
def _kept_as(dtype: Any, typed: bool) -> tuple[Any, tuple[Callable[[pl.Series], str | None], ...]]:
    """Returns the type a column of ``dtype`` is converted to, and the checks its cells must pass.

    The type is ``dtype`` itself when the column comes back from a saved run unchanged. It is the type the column
    comes back as when converting changes no cell, so an ``Int32`` column becomes ``Int64``. It is ``None`` when the
    column would come back changed, or can't be written as JSON at all. ``typed`` says whether a row type declares
    the column, so a saved table reads it back by its type (``_saved_dtype``).

    Some columns depend on their cells too, such as a ``Datetime`` in nanoseconds, in the column or inside a list.
    Those get checks: each takes the column and returns what is wrong with it, or ``None``.
    """
    saved = _saved_dtype(dtype, typed)
    if saved is None or not _converts_exactly(dtype, saved):
        return None, ()
    checks: list[Callable[[pl.Series], str | None]] = []
    if _has_inside(dtype, _in_nanoseconds):
        checks.append(_finer_than_a_microsecond)
    if _has_inside(dtype, _in_milliseconds):
        checks.append(_out_of_microseconds)
    if dtype.to_python() not in CELLS and _has_inside(dtype, _is_float):
        checks.append(_not_finite_inside)
    return saved, tuple(checks)


def as_saved(frame: pl.DataFrame, typed: frozenset[str] = frozenset()) -> pl.DataFrame:
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
    - a date inside a list is kept in a column ``typed`` names, which a saved table reads back by its row type, and
      refused in any other, since it would come back as text.

    A column with no values, empty or all null, may still come back with another type. There is no cell to tell
    what it was.

    Args:
        frame: The table.
        typed: The columns a row type declares, each of a type that says its polars type (``CellType.dtype``).

    Raises:
        PydanticCustomError: ``column_not_saved`` if a column would come back changed.
    """
    casts: dict[Any, Any] = {}
    for name, dtype in frame.schema.items():
        kept, checks = _kept_as(dtype, name in typed)
        if kept is None:
            saved = _saved_dtype(dtype, name in typed)
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

    A column of a type a table doesn't keep as it is is refused too (``CellType.refusal``).

    Raises:
        NodeDeclarationError: ``table_row_not_a_typed_dict``, or ``table_cell_not_kept``.
    """
    if isinstance(row, TypeVar) or is_bare_table_row(row):
        return
    if is_typed_dict(row):
        for name, hint in type_fields(row).items():
            if (refusal := cell_type(hint).refusal) is not None:
                cell, reason = refusal
                raise NodeDeclarationError(
                    declaration_problems.table_cell_not_kept(
                        row=(get_origin(row) or row).__qualname__,
                        column=name,
                        cell=cell,
                        reason=reason,
                    )
                )
        return
    raise NodeDeclarationError(
        declaration_problems.table_row_not_a_typed_dict(
            row=row.__qualname__ if isinstance(row, type) else str(row)
        )
    )


def _read_from_json(column: Type, dtype: Any) -> bool:
    """Says whether a frame's column of ``dtype`` must be read from its JSON to give values of ``column``, its
    declared type.

    Polars gives each cell as a plain Python value: a ``String`` column's cell is a ``str``, and a ``Struct``
    column's a ``dict``. Where that already is a value of the type, as a ``str`` is of a ``str`` column, the column
    isn't read, and most of a table is like that. Any other column is read from its JSON, the way a saved table is
    read: a struct as the host's model or dataclass, a text as the enum's member or a date, an ``int`` against a
    limit like ``Param(ge=0)``, and a column of Python objects (``Object``) by each object's own class.
    """
    return not column.holds(Type(dtype.to_python()))


def _columns_as_json(frame: pl.DataFrame, names: list[str]) -> str:
    """Writes the columns ``names`` of ``frame`` as the JSON text of a dict of lists, each object by its own class.

    A cell is written as the Python value polars gives for it, and Python's ``time``, ``datetime`` and ``timedelta``
    stop at microseconds. So a column with a time finer than that, itself or inside a list or struct, is refused
    rather than cut.

    Raises:
        PydanticCustomError: ``column_not_saved`` for a column of Python objects with no JSON form, or with a time
            finer than a microsecond.
    """
    for name in names:
        if (fault := _finer_than_a_microsecond(frame[name])) is not None:
            raise conductor_error(
                "column_not_saved", "column '{column}' " + fault, {"column": name}
            )
    try:
        return json_text(
            {name: frame[name].to_list() for name in names}, fallback=json_by_its_class
        )
    except PydanticSerializationError:
        failed = next(name for name in names if not _has_json_form(frame[name].to_list()))
        raise conductor_error(
            "column_not_saved",
            "column '{column}' is {type}, which can't be written as JSON",
            {"column": failed, "type": str(frame.schema[failed])},
        ) from None


def _has_json_form(cells: list[Any]) -> bool:
    """Whether the cells of one column can be written as JSON, each object by its own class's schema."""
    try:
        json_text(cells, fallback=json_by_its_class)
    except PydanticSerializationError:
        return False
    return True


def table_and_cells_schema(source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
    """Returns the schema that checks a table as ``Table[R]`` does, and gives the frame and the cells it read.

    It gives ``(frame, cells)``. ``frame`` is what a ``Table[R]`` keeps. ``cells`` is each column the check read,
    by name, as values of its field's type, such as ``{"money": [Money(amount=1.0)]}``; a column polars already gives
    values of its type for, like a ``str`` column, isn't there. A Series of rows keeps both, so a node gets the cells
    the check read, and no cell is read twice.
    """
    return _table_schemas(source, handler)[0]


def table_schema(
    source: Any, handler: GetCoreSchemaHandler, *, rows_checked: bool = False
) -> CoreSchema:
    """Returns the pydantic schema of ``Table[R]``.

    The schema accepts a table in any of its three forms: a frame, a list of rows like ``[{"name": "Ann"}]``,
    or conductor's JSON, ``{"columns": [...]}``. It always gives back a frame with exactly the columns ``R`` names,
    converted by ``as_saved``. It writes a table as conductor's JSON.

    Args:
        source: The hint, such as ``Table[Invoice]``.
        handler: Pydantic's handler, which builds the schema of each column's type.
        rows_checked: Whether a table given in Python is already a table of ``R``'s values, as a run knows for one
            made of rows from outputs that declare ``R`` (``RowsChecked``). A list of rows is then made into a frame
            without its cells being checked again, and a frame with exactly the polars types ``R``'s own frame has
            is kept as it is. The table's own checks still hold: its limits, such as ``Param(min_length=1)``, and
            ``as_saved``.

    Raises:
        NodeDeclarationError: ``row_field_optional`` if the row type lets a row leave a column out.
    """
    with_cells, json_schema = _table_schemas(source, handler, rows_checked=rows_checked)
    kept = core_schema.no_info_after_validator_function(lambda read: read[0], with_cells)
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


def _table_schemas(
    source: Any, handler: GetCoreSchemaHandler, *, rows_checked: bool = False
) -> tuple[CoreSchema, Callable[[CoreSchema, GetJsonSchemaHandler], JsonSchemaValue]]:
    """Returns the schema that checks a ``Table[R]`` and gives the frame and the cells it read, and the function
    that writes its JSON schema.

    Each cell is checked by the row's own schema of its field, built once here under the config the table sits in.
    So rows and a frame are checked alike, and a cell of a class that takes the config it sits in, like a plain
    dataclass, is built under the row's config, as one item of a Series of the rows is.
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
    columns = {name: Type(hint) for name, hint in hints.items()}
    fields = handler.resolve_ref_schema(handler.generate_schema(row))["fields"] if declared else {}
    cells = {name: fields[name]["schema"] for name in columns}
    # the polars type of each column a frame of R's own values has, where its type says it
    kept_schema = {name: cell_type(column.hint).dtype for name, column in columns.items()}
    # the columns a saved table is read back by their type, so a date in a list comes back as a date
    typed = frozenset(name for name, dtype in kept_schema.items() if dtype is not None)
    # the columns of a type a table doesn't keep as it is, which a Series of rows may declare, since a list keeps it
    not_kept = {
        name: refusal
        for name, column in columns.items()
        if (refusal := cell_type(column.hint).refusal) is not None
    }

    def refuse_not_kept() -> None:
        if not_kept:
            name, (cell, reason) = next(iter(not_kept.items()))
            raise cell_not_kept(name, cell, reason)

    dtypes = {
        name: CELLS[column.options[0]][0]
        if column.options[0] in CELLS and len(column.options) == 1
        else None
        for name, column in columns.items()
    }

    def from_rows(rows: list[dict[str, Any]]) -> tuple[pl.DataFrame, dict[str, list[Any]]]:
        refuse_not_kept()
        names = columns if declared else dict.fromkeys(key for one in rows for key in one)
        frame = _frame_of(
            (
                (
                    name,
                    kept_column(
                        name, [one.get(name) for one in rows], cell_type(columns[name].hint)
                    )
                    if declared and dtypes[name] is None
                    else [one.get(name) for one in rows],
                    dtypes.get(name),
                )
                for name in names
            ),
            strict=False,
        )
        return frame, {}

    def from_frame(
        frame: pl.DataFrame, read_columns: ValidatorFunctionWrapHandler
    ) -> tuple[pl.DataFrame, dict[str, list[Any]]]:
        # a table with no rows still has every column, empty and of its declared type
        if not frame.height:
            kept = {name for name, dtype in frame.schema.items() if dtype != pl.Null}
            return pl.DataFrame(
                [
                    frame[name] if name in kept else pl.Series(name, [], dtype=dtypes[name])
                    for name in columns
                ]
            ), {}
        refuse_not_kept()
        for name, column in columns.items():
            if name not in frame.columns:
                raise column_missing(name)
            if not column.optional and frame[name].null_count():
                raise conductor_error(
                    "column_blank",
                    "column '{column}' has a blank cell, and its type doesn't allow blanks",
                    {"column": name},
                )
        # a frame of R's own values, with exactly the polars types R's own frame has, is kept as it is
        if rows_checked and frame.schema == kept_schema:
            return frame, {}
        read = [
            name for name, column in columns.items() if _read_from_json(column, frame.schema[name])
        ]
        cells: dict[str, list[Any]] = read_columns(_columns_as_json(frame, read)) if read else {}
        # the frame keeps each cell as it was read, so a value its type changes, like a text a model strips, is
        # the same in the table, in a Series of its rows and in a saved run
        if cells:
            frame = frame.with_columns(
                kept_column(name, values, cell_type(columns[name].hint))
                for name, values in cells.items()
            )
        return frame if frame.columns == list(columns) else frame.select(list(columns)), cells

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
    # some of a frame's columns, each a list of cells by name, checked by the row's own schema of each field, and
    # read by the row's config as a row is, so bytes inside a list read from the base64 they are written as
    by_column = json_typed_dict_schema(
        {
            name: core_schema.typed_dict_field(core_schema.list_schema(cell), required=False)
            for name, cell in cells.items()
        },
        cls=row_class,
    )
    as_frame = (
        core_schema.no_info_wrap_validator_function(from_frame, core_schema.json_schema(by_column))
        if declared
        else core_schema.no_info_after_validator_function(
            lambda frame: (frame, {}), core_schema.is_instance_schema(pl.DataFrame)
        )
    )
    written = core_schema.no_info_after_validator_function(_frame_written, WRITTEN_TABLE)
    rows = core_schema.list_schema(one_row)

    def with_cells(rows_form: CoreSchema) -> CoreSchema:
        any_form = core_schema.tagged_union_schema(
            {
                "frame": as_frame,
                "rows": rows_form,
                "written": core_schema.chain_schema([written, as_frame]),
            },
            discriminator=_table_form,
            custom_error_type="not_a_table",
            custom_error_message="a table must be a frame, or a list with one object per row",
            custom_error_context=dict(OWN_MESSAGE),
        )
        return core_schema.no_info_after_validator_function(
            lambda read: (as_saved(read[0], typed), read[1]), any_form
        )

    read_rows = core_schema.no_info_after_validator_function(from_rows, rows)
    # Rows already of R's values are only ever given in Python. A list of them is taken as it is, with no check of
    # its rows; JSON keeps the plain schema
    taken_rows = core_schema.no_info_after_validator_function(from_rows, core_schema.list_schema())
    return (
        core_schema.json_or_python_schema(
            json_schema=with_cells(read_rows),
            python_schema=with_cells(taken_rows if rows_checked else read_rows),
        ),
        json_schema,
    )
