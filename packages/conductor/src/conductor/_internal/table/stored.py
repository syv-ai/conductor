"""``Records``: a ``Series`` of records stored as the table a node returned for it."""

from __future__ import annotations

import typing
from typing import Any, get_args

import polars as pl
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from conductor._internal.record_field import NoSuchFieldError
from conductor._internal.series import ColumnsOnDemand, StoredSeries
from conductor._internal.table.type import _Table
from conductor._internal.table.write import written_columns
from conductor._internal.value_type import Type


class Records[T](StoredSeries, returned_as=pl.DataFrame):
    """A ``Series`` of records stored as the table (a ``polars.DataFrame``) a node returned for it, one row per
    record. Only the frame is kept. A column is read out of it as a Python list the first time an edge reads a
    field of a record or the run is saved, and that list is then reused, so each column exists as a Python list at
    most once and a column nothing reads never does.

    Example: for a frame with the columns ``name`` and ``amount`` and two rows, ``records[0]`` is
    ``{"name": "Ann", "amount": 10}`` and ``records.columns["amount"]`` is ``[10, 20]``. A node with a ``Table``
    input wired to the Series is called with ``records.frame`` itself, so no record is built for it.

    ``Records[R]`` is also the type hint a saved run reads and writes one by, as the table's own JSON.

    Attributes:
        frame: The table whose rows are the records.
        columns: ``columns[name]`` is the list of the cells of the column ``name``; for a name the frame has no
            column of, a list of ``NoSuchFieldError``.
    """

    __slots__ = ("columns", "frame")

    def __init__(self, frame: pl.DataFrame) -> None:
        """Stores ``frame``, which is already validated as the table of the records' type."""
        self.frame = frame
        self.columns = ColumnsOnDemand(self._cells_of)

    def _cells_of(self, name: str) -> list[Any]:
        if name not in self.frame.columns:
            return [NoSuchFieldError(name)] * self.frame.height
        cells: list[Any] = self.frame[name].to_list()
        return cells

    @classmethod
    def stored_hint(cls, item: Type) -> Any | None:
        """Returns ``Records[R]`` for items of a ``TypedDict`` ``R`` or of ``Any``; ``None`` for any other item
        type, and for one with a limit or choices."""
        if item.limited or not (item.bare is Any or typing.is_typeddict(item.bare)):
            return None
        records: Any = cls
        return records[item.bare]

    @classmethod
    def store(cls, returned: Any, item: Type) -> Records[Any]:
        """Validates the frame a node ``returned`` as a table of ``item`` records and returns it as ``Records``."""
        table: Any = _Table
        return cls(Type(table[item.bare]).read(returned, strict=True))

    def __len__(self) -> int:
        return self.frame.height

    def __getitem__(self, place: int) -> dict[str, Any]:
        """Returns the record at the index ``place`` as a dict."""
        columns = self.columns
        return {name: columns[name][place] for name in self.frame.columns}

    def as_list(self) -> list[dict[str, Any]]:
        names = self.frame.columns
        return [dict(zip(names, cells, strict=True)) for cells in zip(*(self.columns[name] for name in names), strict=True)]

    def as_table_input(self) -> pl.DataFrame:
        return self.frame

    def written_once(self) -> pl.DataFrame:
        return self.frame

    @staticmethod
    def without_rows(value: pl.DataFrame, places: set[int]) -> pl.DataFrame:
        return value.filter([place not in places for place in range(value.height)])

    def _as_json(self) -> dict[str, Any]:
        # Read through ``columns``, so a run saved at every checkpoint reads each column out of the frame once.
        return written_columns(self.frame, {name: self.columns[name] for name in self.frame.columns})

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Returns the schema of ``Records[R]`` in a saved run: the JSON of ``Table[R]``, read into this class."""
        table: Any = _Table
        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(cls, handler.generate_schema(table[get_args(source)[0]])),
            python_schema=core_schema.is_instance_schema(cls),
            serialization=core_schema.plain_serializer_function_ser_schema(Records._as_json),
        )
