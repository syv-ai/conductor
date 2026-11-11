"""How a run stores the items a node returned for a ``Series`` output.

Example: a node returns ``[{"name": "Ann"}, {"name": "Bo"}]`` for a ``Series`` output. The run stores it as a
``SeriesItems``. A node wired to that output is called once per item, with ``stored[0]`` and then ``stored[1]``.
If its edge reads the field ``name`` (``split.result.name``), it is called with ``stored.columns["name"][0]``,
which is ``"Ann"``.

A package can store what a node returned in another form by subclassing ``StoredSeries`` with ``returned_as``:
the table package stores a ``polars.DataFrame`` as ``Records``. Code that reads a stored Series uses only what
``StoredSeries`` lists, so it never asks which form it has.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ClassVar, get_args

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from conductor._internal.record_field import NoSuchFieldError, field_of
from conductor._internal.sentinels import SKIPPED
from conductor._internal.value_type import Type


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


class StoredSeries:
    """The base class of every form a run stores a ``Series`` output in.

    Every subclass has ``len(stored)``, ``stored[place]`` (the item at that index) and the members below.
    A subclass declared with ``returned_as=SomeClass`` is used when a node returns a ``SomeClass`` for a
    ``Series`` output, and it then defines ``stored_hint`` and ``store`` as well.

    Attributes:
        columns: ``columns[name]`` is a list with the field ``name`` of every item, in order. The list is made the
            first time it is asked for. For an item that is ``None`` or ``SKIPPED`` the list has that value, and
            for an item with no field ``name`` it has a ``NoSuchFieldError`` instance.
    """

    __slots__ = ()

    columns: ColumnsOnDemand
    by_returned_class: ClassVar[dict[type, type[StoredSeries]]] = {}

    def __init_subclass__(cls, *, returned_as: type | None = None) -> None:
        super().__init_subclass__()
        if returned_as is not None:
            StoredSeries.by_returned_class[returned_as] = cls

    @classmethod
    def stored_hint(cls, item: Type) -> Any | None:
        """Returns the type hint a saved run reads and writes this class by, for a Series whose items have the
        type ``item``; ``None`` if items of that type can't be stored as this class. ``Records`` returns
        ``Records[Row]`` for items of a ``TypedDict`` ``Row`` and ``None`` for items of ``str``."""
        raise NotImplementedError

    @classmethod
    def store(cls, returned: Any, item: Type) -> StoredSeries:
        """Validates what a node ``returned`` for a Series whose items have the type ``item``, and returns it
        stored as this class. ``Records.store(frame, Type(Row))`` checks the frame's columns against ``Row``.

        Raises:
            FieldValueError: ``returned`` is not valid for the items' type.
        """
        raise NotImplementedError

    def as_list(self) -> list[Any]:
        """Returns the items as a new plain list, as a node with a ``Series`` input receives them: ``None`` where
        an item is ``SKIPPED``."""
        raise NotImplementedError

    def as_table_input(self) -> Any:
        """Returns the items as a node with a ``Table`` input receives them, before the input's type validates
        them: the list of ``as_list``, such as ``[{"name": "Ann"}]``, unless the subclass has a table already,
        which ``Records`` returns as it is."""
        return self.as_list()

    def field_as_list(self, name: str) -> list[Any]:
        """Returns the field ``name`` of every item as a new list, with ``None`` for an item that is ``None`` or
        ``SKIPPED``. ``SeriesItems([{"n": 1}, None]).field_as_list("n")`` is ``[1, None]``.

        Raises:
            NoSuchFieldError: An item has no field ``name``.
        """
        column = self.columns[name]
        for cell in column:
            if type(cell) is NoSuchFieldError:
                raise NoSuchFieldError(name)
        return [None if cell is SKIPPED else cell for cell in column]

    def for_saving(self) -> tuple[StoredSeries, list[int]]:
        """Returns the Series as a saved run writes it, and the indexes of its ``SKIPPED`` items, which the save
        writes as ``None`` and lists beside the Series. ``SeriesItems(["a", SKIPPED]).for_saving()`` is
        ``(SeriesItems(["a", None]), [1])``."""
        return self, []

    def written_once(self) -> Any | None:
        """Returns the object this Series is stored as when a save should write that object only once however
        many outputs have it (a table); ``None`` for a list."""
        return None

    @staticmethod
    def without_rows(value: Any, places: set[int]) -> Any:
        """Returns ``value``, a value ``as_table_input`` returned that is not a list, without the rows at the
        indexes ``places``: a three-row frame and ``{1}`` give a frame of its first and third row."""
        raise NotImplementedError


class SeriesItems(list[Any], StoredSeries):
    """A ``Series`` output stored as the list the node returned. An item the node returned ``SKIPPED`` for is
    ``SKIPPED`` in the list.

    Example: ``SeriesItems([{"name": "Ann"}, None]).columns["name"]`` is ``["Ann", None]``.
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
        return [None if item is SKIPPED else item for item in self]

    def for_saving(self) -> tuple[StoredSeries, list[int]]:
        skipped = [place for place, item in enumerate(self) if item is SKIPPED]
        return (SeriesItems(self.as_list()), skipped) if skipped else (self, [])

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Returns the schema of ``SeriesItems[T]`` in a saved run: a JSON array of ``T``, read into this class."""
        return core_schema.no_info_after_validator_function(cls, handler.generate_schema(list[get_args(source)[0]]))  # type: ignore[misc]  # the item type is known only here


def places_of_none(value: Any) -> list[int]:
    """Returns the indexes at which ``value``, the value of a ``Series`` or ``Table`` input, has ``None``.
    ``places_of_none(["a", None])`` is ``[1]``. A value that is not a list (a table) has a row for every item, so
    the result is ``[]``."""
    return [place for place, item in enumerate(value) if item is None] if type(value) is list else []


def without_places(value: Any, places: set[int]) -> Any:
    """Returns ``value``, the value of a ``Series`` or ``Table`` input, without the items at the indexes
    ``places``. ``without_places(["a", "b", "c"], {1})`` is ``["a", "c"]``; a value that is not a list is
    cut by the ``StoredSeries`` subclass registered for its class."""
    if type(value) is list:
        return [item for place, item in enumerate(value) if place not in places]
    return StoredSeries.by_returned_class[type(value)].without_rows(value, places)
