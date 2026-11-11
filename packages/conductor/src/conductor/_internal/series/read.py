"""The pydantic schema of ``Series[T]``: a list of items, each validated as ``T``. A Series can also be read from a
tuple, a range and a value of any class registered with ``read_as_list``. A set is refused.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, get_args

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, PydanticCustomError, core_schema

from conductor._internal.sentinels import SKIPPED

#: Why a set is refused wherever a list is read or written: its order differs between processes.
SET_HAS_NO_ORDER = "a set has no order, so its items would come in another order in another process; give a list"


#: How a value of a class that is no list is made the list of items it holds, by that class.
_AS_LIST: dict[type, Callable[[Any], list[Any]]] = {tuple: list, range: list}


def read_as_list(cls: type, as_list: Callable[[Any], list[Any]]) -> None:
    """Declares that a value of ``cls`` given for a ``Series`` is the list ``as_list`` makes of it."""
    _AS_LIST[cls] = as_list


def series_schema(source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
    """Returns the pydantic schema of ``Series[T]``, where ``source`` is that hint: a list whose items are each
    validated as ``T``.

    From Python it also takes a tuple, a range and a value of a class registered with ``read_as_list``, and
    refuses a set. An item that is ``SKIPPED`` stays ``SKIPPED`` at its index.
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
