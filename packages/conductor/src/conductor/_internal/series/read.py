"""How a ``Series[T]`` value is checked, as a pydantic schema.

A Series is a list whose items are each checked as ``T``. From Python, a node may also return a tuple, a range or
a ``polars.Series``, such as a column of a table. A set is refused, since it has no order.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, get_args

import polars as pl
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema

from conductor._internal.base import SKIPPED
from conductor._internal.validation import set_has_no_order


def series_schema(source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
    """Returns the pydantic schema of ``Series[T]``.

    For ``Series[int]``, the schema accepts ``[1, 2]`` and ``(1, 2)`` and returns ``[1, 2]``. An item that is
    ``SKIPPED`` isn't checked, and stays ``SKIPPED`` at its index.

    Args:
        source: The hint, such as ``Series[int]``.
        handler: Pydantic's handler, which builds the schema of the item type.
    """
    items = core_schema.list_schema(handler.generate_schema((get_args(source) or (Any,))[0]))

    def read(value: Any, read_items: Callable[[Any], list[Any]]) -> list[Any]:
        match value:
            case set() | frozenset():
                raise set_has_no_order()
            case tuple() | range():
                value = list(value)
            case pl.Series():
                value = value.to_list()
        if isinstance(value, list) and any(item is SKIPPED for item in value):
            kept = iter(read_items([item for item in value if item is not SKIPPED]))
            return [SKIPPED if item is SKIPPED else next(kept) for item in value]
        return read_items(value)

    return core_schema.json_or_python_schema(
        json_schema=items, python_schema=core_schema.no_info_wrap_validator_function(read, items)
    )
