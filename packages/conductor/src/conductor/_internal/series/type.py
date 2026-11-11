"""``Series[T]``, the hint a node author writes, and ``SeriesType``, the ``Type`` of such a hint."""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Generic, TypeVar, get_args

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, PydanticCustomError, core_schema

from conductor._internal.sentinels import SKIPPED
from conductor._internal.series.held import HeldSeries
from conductor._internal.series.read import _AS_LIST, SET_HAS_NO_ORDER
from conductor._internal.value_type import Type, list_of, union_of

_T = TypeVar("_T")


class _Series(Generic[_T]):
    """``Series[T]``: many values of ``T``, which a graph works through one at a time; a plain ``list[T]`` to the
    node and to a type checker."""

    # The class is named as a host imports it, so the text a reuse digest is made of names no internal file.
    __module__, __qualname__ = "conductor", "Series"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Gives the schema of a list of items.

        From Python any sequence but a set is taken, as is a value of a class declared with ``read_as_list``, and
        an item that is ``SKIPPED`` is a gap kept where it is.
        """
        items = core_schema.list_schema(handler.generate_schema((get_args(source) or (Any,))[0]))

        def read(value: Any, read_items: Callable[[Any], list[Any]]) -> list[Any]:
            if isinstance(value, (set, frozenset)):
                raise PydanticCustomError("set_has_no_order", SET_HAS_NO_ORDER)
            for cls, as_list in _AS_LIST.items():
                if isinstance(value, cls):
                    value = as_list(value)
                    break
            if isinstance(value, list) and any(item is SKIPPED for item in value):
                kept = iter(read_items([item for item in value if item is not SKIPPED]))
                return [SKIPPED if item is SKIPPED else next(kept) for item in value]
            return read_items(value)

        return core_schema.json_or_python_schema(json_schema=items, python_schema=core_schema.no_info_wrap_validator_function(read, items))


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
    def held_hint(self) -> Any:
        """The hint of a Series as a unit holds it: a list whose item on a branch not taken is blank, or one of
        the other ways a Series of its items may be held."""
        other_ways = [hint for holder in HeldSeries._by_class_returned.values() if (hint := holder.hint_of(self.item)) is not None]
        return union_of((list_of(union_of((self.item.hint, type(None)))), *other_ways))

    def read_returned(self, value: Any) -> Any:
        """Reads what a node returned for the Series, strictly: a list of its items, or, for a value of a class
        held another way, what holds it.

        Raises:
            FieldValueError: The type refuses the value.
        """
        for returned_as, holder in HeldSeries._by_class_returned.items():
            if isinstance(value, returned_as) and holder.hint_of(self.item) is not None:
                return holder.holding(value, self.item)
        return self.read(value, strict=True)
