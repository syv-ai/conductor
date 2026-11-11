"""``Series[T]``, the hint a node author writes, and ``SeriesType``, the ``Type`` of such a hint."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Generic, TypeVar, get_args

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema

from conductor._internal.series.held import SeriesItems, StoredSeries
from conductor._internal.series.read import series_schema
from conductor._internal.value_type import Type, union_of

_T = TypeVar("_T")


class _Series(Generic[_T]):
    """``Series[T]``: many values of ``T``, which a graph works through one at a time; a plain ``list[T]`` to the
    node and to a type checker."""

    # The class is named as a host imports it, so the text a reuse digest is made of names no internal file.
    __module__, __qualname__ = "conductor", "Series"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Returns the schema of ``Series[T]``, built in ``read.py``."""
        return series_schema(source, handler)


if TYPE_CHECKING:  # to a type checker a Series is the list a node gets
    type Series[T] = list[T]
else:
    Series = _Series


class SeriesType(Type, of=_Series):
    """The type of a ``Series[T]``, with or without ``| None``."""

    @functools.cached_property
    def item(self) -> Type:
        """The type of one item."""
        return Type((get_args(self.options[0]) or (Any,))[0])

    @functools.cached_property
    def _named(self) -> tuple[str, str, tuple[type, ...]]:
        """The Series' id, which says its items' type unless that is ``Any``, its title, and the classes its
        items name."""
        return ("Series" if self.item.hint is Any else f"Series[{self.item.text}]"), "Series", self.item.classes

    @functools.cached_property
    def stored_hint(self) -> Any:
        """The type hint a saved run reads and writes a stored Series of this type by: ``SeriesItems[T | None]``,
        or the hint of another class its items can be stored as (``Records[R]`` for records)."""
        items: Any = SeriesItems
        other_forms = [hint for form in StoredSeries.by_returned_class.values() if (hint := form.stored_hint(self.item)) is not None]
        return union_of((items[union_of((self.item.hint, type(None)))], *other_forms))

    def read_returned(self, value: Any) -> StoredSeries | None:
        """Validates what a node returned for the Series, strictly, and returns it as the run stores it: a
        ``SeriesItems``, or, for a value of a class registered with ``returned_as`` (a frame), that class's form.

        Raises:
            FieldValueError: ``value`` is not valid for the type.
        """
        for returned_as, form in StoredSeries.by_returned_class.items():
            if isinstance(value, returned_as) and form.stored_hint(self.item) is not None:
                return form.store(value, self.item)
        items = self.read(value, strict=True)
        return items if items is None else SeriesItems(items)  # ``Series[T] | None`` may be returned as None

    @functools.cached_property
    def may_write_once(self) -> bool:  # type: ignore[override]  # decided per item type, where the base has one answer
        """Whether a Series of this type can be stored as a table: its items are records."""
        return any(form.stored_hint(self.item) is not None for form in StoredSeries.by_returned_class.values())

    def written_once(self, stored: Any) -> Any | None:
        """Returns the table a stored Series is kept as, if it is kept as one; else ``None``."""
        return stored.written_once()

    def read_answer(self, value: Any) -> StoredSeries | None:
        """Validates a person's answer for the Series from its JSON form and returns it as the run stores it.

        Raises:
            FieldValueError: ``value`` is not valid for the type.
        """
        items = self.read_json(value)
        return items if items is None else SeriesItems(items)
