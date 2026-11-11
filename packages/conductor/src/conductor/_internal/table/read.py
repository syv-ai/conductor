"""The pydantic schema of ``Table[R]``: how a table is read from a frame, from a list of records or from its JSON form,
and checked against the columns ``R`` names.
"""

from __future__ import annotations

import typing
from collections.abc import Callable, Iterable, Mapping
from typing import Any, get_args

import polars as pl
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, PydanticCustomError, core_schema

from conductor._internal.table.write import CELLS, DTYPES, WRITTEN_TABLE, written_table
from conductor._internal.value_type import Type


def _frame_of(columns: Iterable[tuple[str, list[Any], Any]]) -> pl.DataFrame:
    """Makes a frame of ``columns``, each a name, its cells and its polars dtype (``None`` to let polars find one).

    Raises:
        PydanticCustomError: Polars can't make a table of them (``not_a_table``): a name twice, lengths that
            differ, or cells of no one type.
    """
    try:
        return pl.DataFrame([pl.Series(name, cells, dtype=dtype, strict=False) for name, cells, dtype in columns])
    except (pl.exceptions.PolarsError, TypeError):  # polars' own account may quote a cell
        raise PydanticCustomError("not_a_table", "the columns don't make a table: one name twice, lengths that differ, or cells of no one type") from None


def _table_form(value: Any) -> str | None:
    """Says which of a table's three forms ``value`` is: a frame, a list of records, or the columns conductor writes."""
    if isinstance(value, pl.DataFrame):
        return "frame"
    return "written" if isinstance(value, Mapping) else "records" if isinstance(value, list) else None


def table_schema(source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
    """Returns the pydantic schema of ``Table[R]``, where ``source`` is that hint.

    The schema validates a table given as a frame, as a list of records (``[{"name": "Ann"}]``) or as the JSON form
    (``{"columns": [...]}``), and always returns a frame with exactly the columns ``R`` names. It writes a table as
    the JSON form.
    """
    row = (get_args(source) or (Any,))[0]
    declared = typing.is_typeddict(row)
    columns = {name: Type(hint) for name, hint in typing.get_type_hints(row, include_extras=True).items()} if declared else {}
    cells = {name: handler.generate_schema(column.carried) for name, column in columns.items()}
    dtypes = {name: CELLS[column.options[0]][0] if column.options[0] in CELLS and len(column.options) == 1 else None for name, column in columns.items()}

    def from_records(records: list[dict[str, Any]]) -> pl.DataFrame:
        names = columns if declared else dict.fromkeys(key for record in records for key in record)
        return _frame_of((name, [record.get(name) for record in records], dtypes.get(name)) for name in names)

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
    written = core_schema.no_info_after_validator_function(lambda table: _frame_of((c["name"], c["values"], DTYPES.get(c["type"])) for c in table["columns"]), WRITTEN_TABLE)
    return core_schema.tagged_union_schema(
        {
            "frame": as_frame,
            "records": core_schema.no_info_after_validator_function(from_records, core_schema.list_schema(record)),
            "written": core_schema.chain_schema([written, as_frame]),
        },
        discriminator=_table_form,
        custom_error_type="not_a_table", custom_error_message="a table is a frame or a list of records, one object per row",
        serialization=core_schema.plain_serializer_function_ser_schema(written_table, when_used="json"),
        metadata={"pydantic_js_functions": [json_schema]},
    )
