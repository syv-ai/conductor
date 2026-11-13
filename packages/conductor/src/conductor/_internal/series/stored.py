"""How a run stores the items a node returned for a ``Series`` output: as ``SeriesItems`` or as ``Records``.

Example: a node returns ``[{"name": "Ann"}, {"name": "Bo"}]`` for a ``Series`` output. The run stores it as a
``SeriesItems``. A node wired to that output is called once per item, with ``stored[0]`` and then ``stored[1]``.
If its edge reads the field ``name`` (``split.result.name``), it is called with ``stored.columns["name"][0]``,
which is ``"Ann"``. A node that returns a ``polars.DataFrame`` for a ``Series`` of records has it stored as
``Records``, which keeps the frame.

Both classes have ``len(stored)``, ``stored[index]`` (the item at that index), ``columns``, ``as_list()``,
``as_table_input()``, ``field_as_list(name)``, ``for_saving()`` and ``written_once()``, so code that reads a
stored Series never asks which of the two it has.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, assert_never, get_args

import polars as pl
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from conductor._internal.base import SKIPPED
from conductor._internal.table import Table, written_columns
from conductor._internal.types import NoSuchFieldError, field_of


class ColumnsOnDemand(dict[str, list[Any]]):
    """A dict of lists that fills in a missing key by calling ``read(key)`` once, and keeps the result.

    Example: ``columns = ColumnsOnDemand(lambda name: [1, 2])``; ``columns["a"]`` calls the function and returns
    ``[1, 2]``, and a second ``columns["a"]`` returns the same list without a call.
    """

    __slots__ = ("read",)

    def __init__(self, read: Callable[[str], list[Any]]) -> None:
        self.read = read

    def __missing__(self, name: str) -> list[Any]:
        column = self[name] = self.read(name)
        return column


class SeriesItems(list[Any]):
    """A ``Series`` output stored as the list the node returned. An item the node returned ``SKIPPED`` for is
    ``SKIPPED`` in the list.

    Example: ``SeriesItems([{"name": "Ann"}, None]).columns["name"]`` is ``["Ann", None]``.

    Attributes:
        columns: ``columns[name]`` is a list with the field ``name`` of every item, in order. The list is made the
            first time it is asked for. For an item that is ``None`` or ``SKIPPED`` the list has that value, and
            for an item with no field ``name`` it has a ``NoSuchFieldError`` instance.
    """

    __slots__ = ("columns",)

    def __init__(self, items: list[Any]) -> None:
        super().__init__(items)
        self.columns = ColumnsOnDemand(self._field_of_each)

    def _field_of_each(self, name: str) -> list[Any]:
        column = []
        for item in self:
            if item is None or item is SKIPPED:
                column.append(item)
                continue
            try:
                column.append(field_of(item, name))
            except NoSuchFieldError as missing:
                column.append(missing)
        return column

    def as_list(self) -> list[Any]:
        """Returns the items as a new plain list, as a node with a ``Series`` input receives them: ``None`` where
        an item is ``SKIPPED``."""
        return [None if item is SKIPPED else item for item in self]

    def as_table_input(self) -> list[Any]:
        """Returns the items as a node with a ``Table`` input receives them, before the input's type validates
        them: the list of ``as_list``, such as ``[{"name": "Ann"}]``."""
        return self.as_list()

    def field_as_list(self, name: str) -> list[Any]:
        """Returns the field ``name`` of every item as a new list, with ``None`` for an item that is ``None`` or
        ``SKIPPED``. ``SeriesItems([{"n": 1}, None]).field_as_list("n")`` is ``[1, None]``.

        Raises:
            NoSuchFieldError: An item has no field ``name``.
        """
        column = self.columns[name]
        for cell in column:
            if type(cell) is NoSuchFieldError:  # an item of a Series[dict] or Series[Any] may lack the field the edge names: the read fails
                raise NoSuchFieldError(name)
        return [None if cell is SKIPPED else cell for cell in column]

    def for_saving(self) -> tuple[SeriesItems, list[int]]:
        """Returns the Series as a saved run writes it, and the indexes of its ``SKIPPED`` items, which the save
        writes as ``None`` and lists beside the Series. ``SeriesItems(["a", SKIPPED]).for_saving()`` is
        ``(SeriesItems(["a", None]), [1])``."""
        skipped = [index for index, item in enumerate(self) if item is SKIPPED]
        return (SeriesItems(self.as_list()), skipped) if skipped else (self, [])

    def written_once(self) -> None:
        """Returns ``None``: a list is written wherever an output has it, unlike the frame of ``Records``."""
        return None

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Returns the schema of ``SeriesItems[T]`` in a saved run: a JSON array of ``T``, read into this class."""
        return core_schema.no_info_after_validator_function(cls, handler.generate_schema(list[get_args(source)[0]]))  # type: ignore[misc]  # the item type is known only here


class Records[T]:
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

    def __len__(self) -> int:
        return self.frame.height

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Returns the record at ``index`` as a dict."""
        columns = self.columns
        return {name: columns[name][index] for name in self.frame.columns}

    def as_list(self) -> list[dict[str, Any]]:
        """Returns the records as a new list of dicts, as a node with a ``Series`` input receives them."""
        names = self.frame.columns
        return [dict(zip(names, cells, strict=True)) for cells in zip(*(self.columns[name] for name in names), strict=True)]

    def as_table_input(self) -> pl.DataFrame:
        """Returns the frame itself, as a node with a ``Table`` input receives it."""
        return self.frame

    def field_as_list(self, name: str) -> list[Any]:
        """Returns the column ``name`` as a new list.

        Raises:
            NoSuchFieldError: The frame has a row and no column ``name``.
        """
        if self.frame.height and name not in self.frame.columns:  # a frame of a Series[Any] may lack the field the edge names
            raise NoSuchFieldError(name)
        return list(self.columns[name])

    def for_saving(self) -> tuple[Records[T], list[int]]:
        """Returns the Series as a saved run writes it, and no indexes: a frame has no ``SKIPPED`` row."""
        return self, []

    def written_once(self) -> pl.DataFrame:
        """Returns the frame, which a save writes once however many outputs have it."""
        return self.frame

    def _as_json(self) -> dict[str, Any]:
        # Read through ``columns``, so a run saved at every checkpoint reads each column out of the frame once.
        return written_columns(self.frame, {name: self.columns[name] for name in self.frame.columns})

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Returns the schema of ``Records[R]`` in a saved run: the JSON of ``Table[R]``, read into this class."""
        table: Any = Table
        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(cls, handler.generate_schema(table[get_args(source)[0]])),
            python_schema=core_schema.is_instance_schema(cls),
            serialization=core_schema.plain_serializer_function_ser_schema(Records._as_json),
        )


#: The two forms a run stores a ``Series`` output in: the list a node returned, or the frame it returned for a
#: Series of records. ``SeriesType`` chooses one by the class of the value.
StoredSeries = SeriesItems | Records


def indexes_of_none(value: Any) -> list[int]:
    """Returns the indexes at which ``value``, the value of a ``Series`` or ``Table`` input, has ``None``.
    ``indexes_of_none(["a", None])`` is ``[1]``. A value that is not a list (a table) has a row for every item, so
    the result is ``[]``."""
    return [index for index, item in enumerate(value) if item is None] if type(value) is list else []


def without_indexes(value: list[Any] | pl.DataFrame, indexes: set[int]) -> list[Any] | pl.DataFrame:
    """Returns ``value``, the value of a ``Series`` or ``Table`` input, without the items at the indexes
    ``indexes``. ``without_indexes(["a", "b", "c"], {1})`` is ``["a", "c"]``; a three-row frame and ``{1}`` give a
    frame of its first and third row."""
    match value:
        case list():
            return [item for index, item in enumerate(value) if index not in indexes]
        case pl.DataFrame():
            return value.filter([index not in indexes for index in range(value.height)])
        case _:
            assert_never(value)
