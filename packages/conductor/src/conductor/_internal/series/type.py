"""``Series[T]``, the hint a node author writes, and ``SeriesType``, the ``Type`` of such a hint."""

from __future__ import annotations

import functools
import typing
from typing import TYPE_CHECKING, Any, get_args

import polars as pl
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from conductor._internal.table import Table
from conductor._internal.types import Type, union_of

from .read import series_schema
from .stored import Records, SeriesItems


class _Series[T]:
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
        """The id (``Series[int]``, or ``Series`` when the items are ``Any``), the title, and the classes named in
        the items' type."""
        return ("Series" if self.item.hint is Any else f"Series[{self.item.text}]"), "Series", self.item.classes

    @functools.cached_property
    def _table(self) -> Type | None:
        """The type that reads a frame a node returns for the Series, ``Type(Table[R])``, when its items are records
        ``R`` (a ``TypedDict``, or ``Any``) with no limit or choices; ``None`` for any other items, such as ``str``."""
        item = self.item
        if item.limited or not (item.bare is Any or typing.is_typeddict(item.bare)):
            return None
        table: Any = Table
        return Type(table[item.bare])

    @functools.cached_property
    def stored_hint(self) -> Any:
        """The type hint a saved run reads and writes a stored Series of this type by: ``SeriesItems[T | None]``,
        or for a Series of records ``R`` either that or ``Records[R]``."""
        items: Any = SeriesItems
        stored = items[union_of((self.item.hint, type(None)))]
        if self._table is None:
            return stored
        records: Any = Records
        return stored | records[self.item.bare]

    @functools.cached_property
    def returned_schema(self) -> CoreSchema:
        """The schema that reads what a node returned for the Series into the form the run stores it in.

        For ``Series[str]`` it reads ``["a", "b"]`` into ``SeriesItems(["a", "b"])``. For ``Series[Row]``, whose
        items are records, a frame is read by the table schema of ``Row`` into ``Records(frame)``, and anything
        else into a ``SeriesItems``. The form is chosen by the value's class and named in pydantic's ``loc`` of an
        error (``("Records", ...)``), so ``_walked`` reads the rest of that ``loc`` with the form's own type.
        """
        def as_items(items: list[Any] | None) -> SeriesItems | None:
            return items if items is None else SeriesItems(items)  # ``Series[T] | None`` may be returned as None

        items = core_schema.no_info_after_validator_function(as_items, self.schema)
        if self._table is None:
            return items

        def form_of(value: Any) -> str:
            return Records.__name__ if isinstance(value, pl.DataFrame) else SeriesItems.__name__

        records = core_schema.no_info_after_validator_function(Records, self._table.schema)
        return core_schema.tagged_union_schema({SeriesItems.__name__: items, Records.__name__: records}, discriminator=form_of)

    def _walked(self, step: int | str) -> tuple[Type, bool, str | None] | None:
        """Takes one step of a pydantic error's ``loc``, as ``Type._walked`` does. The name of a stored form, which
        ``returned_schema`` puts first for a Series of records, leads to the type that form is read by and is not
        kept: ``"SeriesItems"`` to this type, ``"Records"`` to the table of the items' records."""
        if self._table is not None:
            if step == SeriesItems.__name__:
                return self, False, None
            if step == Records.__name__:
                return self._table, False, None
        return super()._walked(step)

    @functools.cached_property
    def may_write_once(self) -> bool:  # type: ignore[override]  # decided per item type, where the base has one answer
        """Whether a Series of this type can be stored as a table: its items are records."""
        return self._table is not None

    def written_once(self, stored: Any) -> Any | None:
        """Returns the table a stored Series is kept as, if it is kept as one; else ``None``."""
        return stored.written_once()
