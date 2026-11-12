"""``Series[T]``, the hint a node author writes, and ``SeriesType``, the ``Type`` of such a hint."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, get_args

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from conductor._internal.types import Type, union_of

from .read import series_schema
from .stored import SeriesItems, StoredSeries


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
    def stored_hint(self) -> Any:
        """The type hint a saved run reads and writes a stored Series of this type by: ``SeriesItems[T | None]``,
        or the hint of another class its items can be stored as (``Records[R]`` for records)."""
        items: Any = SeriesItems
        other_forms = [hint for form in StoredSeries.by_returned_class.values() if (hint := form.stored_hint(self.item)) is not None]
        return union_of((items[union_of((self.item.hint, type(None)))], *other_forms))

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
        forms = self._returned_forms
        if not forms:
            return items

        def form_of(value: Any) -> str:
            return next((form.__name__ for returned_as, form in forms if isinstance(value, returned_as)), SeriesItems.__name__)

        read_as = {form.__name__: core_schema.no_info_after_validator_function(form, form.returned_type(self.item).schema) for _, form in forms}
        choices: dict[Any, CoreSchema] = {SeriesItems.__name__: items, **read_as}
        return core_schema.tagged_union_schema(choices, discriminator=form_of)

    @functools.cached_property
    def _returned_forms(self) -> list[tuple[type, type[StoredSeries]]]:
        """Each class a node may return for the Series other than a list, with the form it is stored in: a
        frame and ``Records`` for a Series of records, none for ``Series[str]``."""
        return [(returned_as, form) for returned_as, form in StoredSeries.by_returned_class.items() if form.stored_hint(self.item) is not None]

    def _walked(self, step: int | str) -> tuple[Type, bool, str | None] | None:
        """Takes one step of a pydantic error's ``loc``, as ``Type._walked`` does. The name of a stored form, which
        ``returned_schema`` puts first, leads to the type that form is read by and is not kept: ``"SeriesItems"`` to
        this type, ``"Records"`` to ``Records.returned_type``, the table of the items' records."""
        if step == SeriesItems.__name__ and self._returned_forms:
            return self, False, None
        for _, form in self._returned_forms:
            if step == form.__name__:
                return form.returned_type(self.item), False, None
        return super()._walked(step)

    @functools.cached_property
    def may_write_once(self) -> bool:  # type: ignore[override]  # decided per item type, where the base has one answer
        """Whether a Series of this type can be stored as a table: its items are records."""
        return any(form.stored_hint(self.item) is not None for form in StoredSeries.by_returned_class.values())

    def written_once(self, stored: Any) -> Any | None:
        """Returns the table a stored Series is kept as, if it is kept as one; else ``None``."""
        return stored.written_once()
