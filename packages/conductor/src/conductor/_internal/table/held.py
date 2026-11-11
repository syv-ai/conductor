"""``Records``: a ``Series`` of records stored as the table a node returned for it."""

from __future__ import annotations

import typing
from typing import Any, Generic, TypeVar, get_args

import polars as pl
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from conductor._internal.series import HeldSeries
from conductor._internal.table.type import _Table
from conductor._internal.table.write import _written_columns
from conductor._internal.value_type import Type

_T = TypeVar("_T")


class Records(HeldSeries, Generic[_T], returned_as=pl.DataFrame):
    """A ``Series`` of records held by column: the table a node returned for it, read one record at a time.

    It is internal to a run. A node that takes a record, or the whole ``Series``, receives plain records, and
    an edge that names a field reads the column's cell with no record built. ``Records[R]`` is the type a saved
    run reads and writes one by: the table's own JSON form, where a ``Series`` held as a list is an array.

    Attributes:
        frame: The table whose rows are the records.
        columns: The cells of each column as a list, by name.
    """

    __slots__ = ("columns", "frame")

    def __init__(self, frame: pl.DataFrame) -> None:
        """Holds ``frame``, a table already read as the type of the records' table."""
        self.frame, self.columns = frame, frame.to_dict(as_series=False)

    @classmethod
    def hint_of(cls, item: Type) -> Any | None:
        """Gives ``Records[R]`` for items of a ``TypedDict`` ``R`` or of ``Any``; ``None`` for any other items,
        those that carry a limit included."""
        if item.limited or not (item.bare is Any or typing.is_typeddict(item.bare)):
            return None
        records: Any = cls
        return records[item.bare]

    @classmethod
    def holding(cls, returned: Any, item: Type) -> Records[Any]:
        """Holds the frame a node ``returned`` for a ``Series`` of ``item``, read as the table of such records."""
        table: Any = _Table
        return cls(Type(table[item.bare]).read(returned, strict=True))

    def __len__(self) -> int:
        return self.frame.height

    def __getitem__(self, place: int) -> dict[str, Any]:
        """Gives the record at ``place``."""
        return {name: cells[place] for name, cells in self.columns.items()}

    def records(self) -> list[dict[str, Any]]:
        """Gives every record, in order, as a node receives a ``Series`` of them."""
        names = list(self.columns)
        return [dict(zip(names, cells, strict=True)) for cells in zip(*self.columns.values(), strict=True)]

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Gives the schema a saved run reads and writes the records by: their table's."""
        table: Any = _Table
        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(cls, handler.generate_schema(table[get_args(source)[0]])),
            python_schema=core_schema.is_instance_schema(cls),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda held: _written_columns(held.frame, held.columns)),
        )
