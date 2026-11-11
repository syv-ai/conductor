"""``Table[R]``, the hint a node author writes, and ``TableType``, the ``Type`` of such a hint."""

from __future__ import annotations

import functools
import typing
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, get_args

import polars as pl
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, PydanticCustomError, core_schema

from conductor._internal.series import read_as_list
from conductor._internal.table.read import _frame_of, _table_form
from conductor._internal.table.write import (
    _CELLS,
    _DTYPES,
    _WRITTEN_TABLE,
    table_digest,
    written_table,
)
from conductor._internal.value_type import Type

_T = TypeVar("_T")


class _Table(Generic[_T]):
    """``Table[R]``: a whole table whose columns the ``TypedDict`` ``R`` names; a ``polars.DataFrame`` to the node
    and to a type checker.

    A bare ``Table`` has any columns. A table is read from a frame, from a list of records (how gathered records
    and a person's answer arrive) or from the JSON conductor writes one as. A ``Table[R]`` read holds exactly
    ``R``'s columns, as a ``TypedDict`` read drops other keys: each present, each cell a value of its column's
    type, and blank only where that type takes a blank.
    """

    # The class is named as a host imports it, so the text a reuse digest is made of names no internal file.
    __module__, __qualname__ = "conductor", "Table"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Gives the schema that reads a table in any of its three forms and writes it as its columns."""
        row = (get_args(source) or (Any,))[0]
        declared = typing.is_typeddict(row)
        columns = {name: Type(hint) for name, hint in typing.get_type_hints(row, include_extras=True).items()} if declared else {}
        cells = {name: handler.generate_schema(column.carried) for name, column in columns.items()}
        dtypes = {name: _CELLS[column.options[0]][0] if column.options[0] in _CELLS and len(column.options) == 1 else None for name, column in columns.items()}

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
        written = core_schema.no_info_after_validator_function(lambda table: _frame_of((c["name"], c["values"], _DTYPES.get(c["type"])) for c in table["columns"]), _WRITTEN_TABLE)
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


if TYPE_CHECKING:  # to a type checker a Table is the frame a node gets, which keeps its row type
    from conductor._internal.table.table_as_a_type_checker_reads_it import Table
else:
    Table = _Table


#: The words pydantic's ``loc`` names the form a table came in with, which are no place in the table.
_TABLE_FORMS = ("frame", "records", "written", "columns")


class TableType(Type, of=_Table, value_class=pl.DataFrame):
    """The type of a ``Table[R]``, with or without ``| None``. A ``polars.DataFrame`` written as a hint is the
    table of any columns it is."""

    keys_in_order: ClassVar[bool] = True  # a table is written by column, in order
    typed_in_advice: ClassVar[str] = "; give a table as a list of records"

    @functools.cached_property
    def _row(self) -> Any:
        """The row type ``R``; ``Any`` for a table of any columns."""
        return (get_args(self.options[0]) or (Any,))[0]

    @functools.cached_property
    def fields(self) -> Mapping[str, Any]:
        """The hint of each column the row type declares, by name; none for a table of any columns."""
        return Type(self._row).fields

    @functools.cached_property
    def _named(self) -> tuple[str, str, tuple[type, ...]]:
        """The table's id, which names its columns, its title, and no class."""
        columns = typing.get_type_hints(self._row) if typing.is_typeddict(self._row) else {}
        return f"Table[{', '.join(f'{column}: {Type(hint).text}' for column, hint in columns.items())}]" if columns else "Table", "Table", ()

    @functools.cached_property
    def record(self) -> Type:
        """The type of one record, which is what a table input gathers: the row type, and ``dict`` for a table of
        any columns, whose records are mappings of any keys."""
        return Type(dict if self._row is Any else self._row)

    def _walked(self, step: int | str) -> tuple[Type, bool, str | None] | None:
        """Follows one step of pydantic's ``loc`` into the table: the form it came in is no place, a record's
        place keeps the table's columns, and a column is named."""
        if step in _TABLE_FORMS:
            return self, False, None
        if isinstance(step, int):
            return self, True, None
        return (Type(self.fields[step]), True, step) if step in self.fields else None

    def _fault_in(self, part: str, kind: str, msg: str) -> str:
        """Says a fault in the column ``part``."""
        return f"the column '{part}' its row type declares is missing" if kind == "missing" else f"column '{part}': {msg}"

    @staticmethod
    def untyped_as_json(value: Any) -> Any:
        """Gives a frame met among values of no declared type as conductor writes a table."""
        return written_table(value)

    @staticmethod
    def untyped_to_hash(value: Any) -> Any:
        """Returns ``{"table": digest}`` for a frame in what a call received, where the digest is computed once
        per frame."""
        return {"table": table_digest(value)}

    @staticmethod
    def untyped_equal(one: Any, other: Any) -> bool:
        """Says whether two frames hold the same cells; a frame has no plain ``==``."""
        return isinstance(other, pl.DataFrame) and one.equals(other)


read_as_list(pl.Series, pl.Series.to_list)  # a column of a frame given for a Series is the list of its cells
