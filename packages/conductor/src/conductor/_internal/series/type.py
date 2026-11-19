"""``Series[T]``, the hint a node author writes, and ``SeriesType``, how conductor reads that hint.

``Series`` itself does very little. It exists so pydantic can find the Series' schema, which ``read.py`` builds.
``SeriesType`` is the ``Type`` subclass for a Series: it gives the type of one item, the Series' id, and how a run
stores what a node returned for it.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Annotated, Any, get_args, get_origin

import polars as pl
import typing_extensions
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from conductor._internal.table import Table, is_bare_table_row
from conductor._internal.types import Param, Type, split_hint, union_members, union_of

from .read import series_schema
from .stored import SeriesItems, SeriesRows


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
    """The ``Type`` of a ``Series[T]`` hint, with or without ``| None``.

    Everything in a Series' ``Annotated`` is for its items: limits, choices, a title and any other metadata, such as
    a validator. So ``Annotated[Series[int], Param(ge=0)]`` is read as ``Series[Annotated[int, Param(ge=0)]]``: each
    item is at least 0.
    """

    @functools.cached_property
    def item(self) -> Type:
        """The type of one item, ``T``, with the Series' own metadata on it.

        The item's hint is read as written, so a type variable in it is kept for a graph to fill in. The item's own
        metadata comes first and the Series' after it, so the Series' wins where both set one thing, as pydantic's
        last one does.
        """
        bare, param, others = split_hint(self.hint)
        on_items = (*others, *([] if param == Param() else [param]))
        (series,) = (option for option in union_members(bare) if option is not type(None))
        item = (get_args(series) or (Any,))[0]
        return Type(Annotated[(item, *on_items)] if on_items else item)

    @functools.cached_property
    def _parts(self) -> tuple[Any, Param, tuple[Any, ...]]:
        """The closed hint taken apart as ``Type._parts`` does, with the Series' own metadata on its items
        (``item``)."""
        bare, param, others = split_hint(self.closed)
        if not others and param == Param():
            return bare, param, others
        lifted = union_of(
            option if option is type(None) else (get_origin(option) or option)[self.item.closed]
            for option in union_members(bare)
        )
        return lifted, Param(), ()

    def _check_own_arguments(self) -> None:
        """Refuses the items' choices if they can't be read, as ``Type._check_own_arguments`` says.

        The items' type is read as a whole, so a check that converts a value is read with them: a choice it moves
        off would never be picked (``choices_invalid``).
        """
        self.item.choices  # noqa: B018

    @functools.cached_property
    def _named(self) -> tuple[str, str, tuple[type, ...]]:
        """The Series' id, its title and the classes it names.

        The id is ``Series[int]``, or just ``Series`` when the items are ``Any``. The title is always ``Series``. The
        classes are the ones the item type names.
        """
        return (
            ("Series" if self.item.closed is Any else f"Series[{self.item.text}]"),
            "Series",
            self.item.classes,
        )

    @functools.cached_property
    def _table(self) -> Type | None:
        """The table type a node may return a frame as, for a Series of rows.

        A node that returns ``Series[Row]``, where ``Row`` is a ``TypedDict``, may return a ``polars.DataFrame``
        instead of a list of dicts. That frame is checked as a ``Table[Row]``. The same goes for a Series whose
        item is the row of a bare ``Table`` (``is_bare_table_row``), such as ``Series[dict]``: its frame is checked
        as a table with any columns. This is ``None`` for any other items, such as ``str`` or ``dict[str, int]``,
        and for rows with limits or choices.
        """
        item = self.item
        if item.limited or not (
            is_bare_table_row(item.bare) or typing_extensions.is_typeddict(item.bare)
        ):
            return None
        table: Any = Table
        return Type(table[item.bare])

    @functools.cached_property
    def stored_hint(self) -> Any:
        """The type hint a saved run uses to write and read back a stored Series of this type.

        It is ``SeriesItems[T | None]``. For a Series of rows ``R`` it is ``SeriesItems[R | None] | SeriesRows[R]``,
        since the node may have returned either a list or a frame.
        """
        items: Any = SeriesItems
        stored = items[union_of((self.item.closed, type(None)))]
        if self._table is None:
            return stored
        rows: Any = SeriesRows
        return stored | rows[self.item.bare]

    @functools.cached_property
    def returned_schema(self) -> CoreSchema:
        """The schema that checks what a node returned for the Series, and converts it to how a run stores it.

        For ``Series[str]``, the return ``["a", "b"]`` becomes ``SeriesItems(["a", "b"])``. For ``Series[Row]``,
        where ``Row`` is a ``TypedDict``, a frame is checked as a ``Table[Row]`` and becomes ``SeriesRows(frame)``. Anything
        else becomes a ``SeriesItems``.

        The class of the value picks the form. Pydantic puts the form's name first in an error's ``loc``, as in
        ``("SeriesRows", ...)``, and ``_walked`` uses that name to follow the rest of the ``loc``.
        """

        def as_items(items: list[Any] | None) -> SeriesItems | None:
            # a node may return None for a ``Series[T] | None``
            return items if items is None else SeriesItems(items)

        items = core_schema.no_info_after_validator_function(as_items, self.schema)
        if self._table is None:
            return items

        def form_of(value: Any) -> str:
            return SeriesRows.__name__ if isinstance(value, pl.DataFrame) else SeriesItems.__name__

        rows = core_schema.no_info_after_validator_function(SeriesRows, self._table.schema)
        return core_schema.tagged_union_schema(
            {SeriesItems.__name__: items, SeriesRows.__name__: rows}, discriminator=form_of
        )

    def _walked(self, step: int | str) -> tuple[Type, bool, str | None] | None:
        """Follows one step of a pydantic error's ``loc``, as ``Type._walked`` does.

        For a Series of rows, the first step can be the name of a stored form, which ``returned_schema`` adds.
        That step is dropped from the ``loc``. ``"SeriesItems"`` leads to this type, and ``"SeriesRows"`` to the table
        type of the rows.
        """
        if self._table is not None:
            if step == SeriesItems.__name__:
                return self, False, None
            if step == SeriesRows.__name__:
                return self._table, False, None
        return super()._walked(step)
