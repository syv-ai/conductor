"""What a ``Series`` can be read from besides a list: a tuple, a range, and a value of any class registered with
``read_as_list``. A set is refused.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

#: Why a set is refused wherever a list is read or written: its order differs between processes.
SET_HAS_NO_ORDER = "a set has no order, so its items would come in another order in another process; give a list"


#: How a value of a class that is no list is made the list of items it holds, by that class.
_AS_LIST: dict[type, Callable[[Any], list[Any]]] = {tuple: list, range: list}


def read_as_list(cls: type, as_list: Callable[[Any], list[Any]]) -> None:
    """Declares that a value of ``cls`` given for a ``Series`` is the list ``as_list`` makes of it."""
    _AS_LIST[cls] = as_list
