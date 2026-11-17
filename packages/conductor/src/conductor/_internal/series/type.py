"""``Series[T]``, the hint a node author writes, and ``SeriesType``, how conductor reads that hint.

``Series`` itself does very little. It exists so pydantic can find the Series' schema, which ``read.py`` builds.
``SeriesType`` is the ``Type`` subclass for a Series: it gives the type of one item, the Series' id, and how a run
stores what a node returned for it.
"""

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
    """A list of values that a graph works through one item at a time.

    A node that returns ``Series[str]`` gives a list of strings. A node wired to it that takes a plain ``str`` is
    then called once for each item. A node that takes ``Series[str]`` gets the whole list at once. Either way, the
    node sees a plain ``list``, and so does a type checker.
    """

    # Named as a host imports it. The hash that decides whether a call can be reused includes this name, so it
    # must not change when an internal file moves.
    __module__, __qualname__ = "conductor", "Series"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Returns the pydantic schema of ``Series[T]``, which ``read.py`` builds."""
        return series_schema(source, handler)


if TYPE_CHECKING:  # a type checker sees a Series as the list a node gets
    type Series[T] = list[T]
else:
    Series = _Series


class SeriesType(Type, of=_Series):
    """The ``Type`` of a ``Series[T]`` hint, with or without ``| None``."""

    @functools.cached_property
    def item(self) -> Type:
        """The type of one item, ``T``."""
        return Type((get_args(self.options[0]) or (Any,))[0])

    @functools.cached_property
    def _named(self) -> tuple[str, str, tuple[type, ...]]:
        """The Series' id, its title and the classes it names.

        The id is ``Series[int]``, or just ``Series`` when the items are ``Any``. The title is always ``Series``. The
        classes are the ones the item type names.
        """
        return (
            ("Series" if self.item.hint is Any else f"Series[{self.item.text}]"),
            "Series",
            self.item.classes,
        )

    @functools.cached_property
    def _table(self) -> Type | None:
        """The table type a node may return a frame as, for a Series of records.

        A node that returns ``Series[Row]``, where ``Row`` is a ``TypedDict``, may return a ``polars.DataFrame``
        instead of a list of dicts. That frame is checked as a ``Table[Row]``. The same goes for ``Series[dict]``
        and ``Series[Any]``, whose frame is checked as a table with any columns. This is ``None`` for any other
        items, such as ``str``, and for records with limits or choices.
        """
        item = self.item
        if item.limited or not (
            item.bare is Any or item.bare is dict or typing.is_typeddict(item.bare)
        ):
            return None
        table: Any = Table
        return Type(table[item.bare])

    @functools.cached_property
    def stored_hint(self) -> Any:
        """The type hint a saved run uses to write and read back a stored Series of this type.

        It is ``SeriesItems[T | None]``. For a Series of records ``R`` it is ``SeriesItems[R | None] | Records[R]``,
        since the node may have returned either a list or a frame.
        """
        items: Any = SeriesItems
        stored = items[union_of((self.item.hint, type(None)))]
        if self._table is None:
            return stored
        records: Any = Records
        return stored | records[self.item.bare]

    @functools.cached_property
    def returned_schema(self) -> CoreSchema:
        """The schema that checks what a node returned for the Series, and converts it to how a run stores it.

        For ``Series[str]``, the return ``["a", "b"]`` becomes ``SeriesItems(["a", "b"])``. For ``Series[Row]``,
        where ``Row`` is a record, a frame is checked as a ``Table[Row]`` and becomes ``Records(frame)``. Anything
        else becomes a ``SeriesItems``.

        The class of the value picks the form. Pydantic puts the form's name first in an error's ``loc``, as in
        ``("Records", ...)``, and ``_walked`` uses that name to follow the rest of the ``loc``.
        """

        def as_items(items: list[Any] | None) -> SeriesItems | None:
            # a node may return None for a ``Series[T] | None``
            return items if items is None else SeriesItems(items)

        items = core_schema.no_info_after_validator_function(as_items, self.schema)
        if self._table is None:
            return items

        def form_of(value: Any) -> str:
            return Records.__name__ if isinstance(value, pl.DataFrame) else SeriesItems.__name__

        records = core_schema.no_info_after_validator_function(Records, self._table.schema)
        return core_schema.tagged_union_schema(
            {SeriesItems.__name__: items, Records.__name__: records}, discriminator=form_of
        )

    def _walked(self, step: int | str) -> tuple[Type, bool, str | None] | None:
        """Follows one step of a pydantic error's ``loc``, as ``Type._walked`` does.

        For a Series of records, the first step can be the name of a stored form, which ``returned_schema`` adds.
        That step is dropped from the ``loc``. ``"SeriesItems"`` leads to this type, and ``"Records"`` to the table
        type of the records.
        """
        if self._table is not None:
            if step == SeriesItems.__name__:
                return self, False, None
            if step == Records.__name__:
                return self._table, False, None
        return super()._walked(step)
