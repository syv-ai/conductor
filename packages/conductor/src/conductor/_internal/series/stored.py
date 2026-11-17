"""How a run stores what a node returned for a ``Series`` output.

A node usually returns a list, and the run stores it as a ``SeriesItems``. Say a node ``split`` returns
``[{"name": "Ann"}, {"name": "Bo"}]``. A node wired to it is called once per item, first with ``stored[0]`` and
then with ``stored[1]``. If the edge picks the field ``name`` (``split.result.name``), the first call gets
``stored.columns["name"][0]``, which is ``"Ann"``.

A node with a ``Series`` of records may return a ``polars.DataFrame`` instead. The run stores that as
``Records``, which keeps the frame and makes each row a record only when one is needed.

Both classes have the same methods: ``len(stored)``, ``stored[index]``, ``columns``, ``as_list()``,
``as_table_input()``, ``field_as_list(name)`` and ``for_saving()``. Code that uses a stored Series never has to
check which of the two it has.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, get_args

import polars as pl
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from conductor._internal.base import SKIPPED
from conductor._internal.table import Table, written_columns
from conductor._internal.types import NoSuchFieldError, field_of


class ColumnsOnDemand(dict[str, list[Any]]):
    """A dict of columns that makes each column the first time it is asked for, and keeps it.

    For example, take ``columns = ColumnsOnDemand(lambda name: [1, 2])``. The first ``columns["a"]`` calls the
    function and returns ``[1, 2]``. A second ``columns["a"]`` returns the same list without calling it again.
    """

    __slots__ = ("read",)

    def __init__(self, read: Callable[[str], list[Any]]) -> None:
        self.read = read

    def __missing__(self, name: str) -> list[Any]:
        column = self[name] = self.read(name)
        return column


class SeriesItems(list[Any]):
    """A ``Series`` output stored as the list the node returned.

    An item the node returned as ``SKIPPED`` stays ``SKIPPED`` in the list. For example,
    ``SeriesItems([{"name": "Ann"}, None]).columns["name"]`` is ``["Ann", None]``.

    Attributes:
        columns: ``columns[name]`` is the field ``name`` of every item, in order. Each list is made the first time
            it is asked for. An item that is ``None`` or ``SKIPPED`` gives that value. An item with no field
            ``name`` gives a ``NoSuchFieldError``, which is raised only if something uses that field.
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
        """Returns the items as a new list, the way a node with a ``Series`` input gets them.

        A ``SKIPPED`` item becomes ``None``.
        """
        return [None if item is SKIPPED else item for item in self]

    def as_table_input(self) -> list[Any]:
        """Returns the items for a node with a ``Table`` input, before the input's type checks them.

        It is the same list as ``as_list``, such as ``[{"name": "Ann"}]``. The table type turns it into a frame.
        """
        return self.as_list()

    def field_as_list(self, name: str) -> list[Any]:
        """Returns the field ``name`` of every item as a new list.

        An item that is ``None`` or ``SKIPPED`` gives ``None``. For example,
        ``SeriesItems([{"n": 1}, None]).field_as_list("n")`` is ``[1, None]``.

        Raises:
            NoSuchFieldError: An item has no field ``name``.
        """
        column = self.columns[name]
        for cell in column:
            # an item of a Series[dict] or Series[Any] may not have the field the edge names
            if type(cell) is NoSuchFieldError:
                raise NoSuchFieldError(name)
        return [None if cell is SKIPPED else cell for cell in column]

    def for_saving(self) -> tuple[SeriesItems, list[int]]:
        """Returns the Series ready to be saved, and the indexes of its ``SKIPPED`` items.

        JSON has no ``SKIPPED``, so a saved run writes those items as ``None`` and lists their indexes next to the
        Series. For example, ``SeriesItems(["a", SKIPPED]).for_saving()`` is ``(SeriesItems(["a", None]), [1])``.
        """
        skipped = [index for index, item in enumerate(self) if item is SKIPPED]
        return (SeriesItems(self.as_list()), skipped) if skipped else (self, [])

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Returns the schema of ``SeriesItems[T]`` in a saved run: a JSON array of ``T``, made into this class."""
        items = list[get_args(source)[0]]  # type: ignore[misc, valid-type]  # the item type is known only here
        return core_schema.no_info_after_validator_function(cls, handler.generate_schema(items))


class Records[T]:
    """A ``Series`` of records stored as the ``polars.DataFrame`` a node returned for it, one row per record.

    Only the frame is kept. A column is turned into a Python list the first time something needs it: an edge that
    picks a field, or saving the run. That list is then reused. So each column becomes a Python list at most once,
    and a column nothing uses never does.

    Say the frame has the columns ``name`` and ``amount`` and two rows. Then ``records[0]`` is
    ``{"name": "Ann", "amount": 10}``, and ``records.columns["amount"]`` is ``[10, 20]``. A node with a ``Table``
    input wired to the Series gets ``records.frame`` itself, so no record is made for it.

    ``Records[R]`` is also the type hint a saved run writes and reads one back by. It is saved as a table's JSON.

    Attributes:
        frame: The table whose rows are the records.
        columns: ``columns[name]`` is the cells of the column ``name``. For a name the frame has no column for, it
            is a list of ``NoSuchFieldError``.
    """

    __slots__ = ("columns", "frame")

    def __init__(self, frame: pl.DataFrame) -> None:
        """Stores ``frame``, which has already been checked as a table of the records' type."""
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
        """Returns the records as a new list of dicts, the way a node with a ``Series`` input gets them."""
        names = self.frame.columns
        return [
            dict(zip(names, cells, strict=True))
            for cells in zip(*(self.columns[name] for name in names), strict=True)
        ]

    def as_table_input(self) -> pl.DataFrame:
        """Returns the frame itself, the way a node with a ``Table`` input gets it.

        ``as_list`` would work too, since the table type reads a list of records. But then every call that collects
        the frame would turn it into one dict per row and back into a frame.
        """
        return self.frame

    def field_as_list(self, name: str) -> list[Any]:
        """Returns the column ``name`` as a new list.

        Raises:
            NoSuchFieldError: The frame has a row and no column ``name``.
        """
        # a frame of a Series[Any] may not have the field the edge names
        if self.frame.height and name not in self.frame.columns:
            raise NoSuchFieldError(name)
        return list(self.columns[name])

    def for_saving(self) -> tuple[Records[T], list[int]]:
        """Returns the Series ready to be saved, and no indexes, since a frame has no ``SKIPPED`` row."""
        return self, []

    def _as_json(self) -> dict[str, Any]:
        # Go through ``columns``, so a run saved at every checkpoint turns each column into a list only once.
        return written_columns(
            self.frame, {name: self.columns[name] for name in self.frame.columns}
        )

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Returns the schema of ``Records[R]`` in a saved run: the JSON of a ``Table[R]``, made into this class."""
        table: Any = Table
        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(
                cls, handler.generate_schema(table[get_args(source)[0]])
            ),
            python_schema=core_schema.is_instance_schema(cls),
            serialization=core_schema.plain_serializer_function_ser_schema(Records._as_json),
        )


#: The two ways a run stores a ``Series`` output: the list a node returned, or the frame it returned for a Series
#: of records. ``SeriesType.returned_schema`` picks one by the class of the value.
StoredSeries = SeriesItems | Records
