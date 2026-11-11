"""``HeldSeries``: the base class for storing a node's ``Series`` output in a run in a form other than a list."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from conductor._internal.value_type import Type


class HeldSeries(ABC):
    """A way a run holds the items of a ``Series`` other than as the list a node returned, declared by subclassing::

        class Records(HeldSeries, returned_as=Frame): ...

    A node that returns a ``returned_as`` for a ``Series`` output has it held as the subclass, where the items'
    type allows it.
    """

    __slots__ = ()

    _by_class_returned: ClassVar[dict[type, type[HeldSeries]]] = {}

    def __init_subclass__(cls, *, returned_as: type) -> None:
        """Keeps the subclass as how a ``returned_as`` returned for a ``Series`` is held."""
        super().__init_subclass__()
        HeldSeries._by_class_returned[returned_as] = cls

    @classmethod
    @abstractmethod
    def hint_of(cls, item: Type) -> Any | None:
        """Gives the hint a saved run reads and writes a ``Series`` of ``item`` held this way by; ``None`` when a
        ``Series`` of such items can't be held this way."""

    @classmethod
    @abstractmethod
    def holding(cls, returned: Any, item: Type) -> HeldSeries:
        """Holds what a node ``returned`` for a ``Series`` of ``item``, read strictly.

        Raises:
            FieldValueError: The items' type refuses what was returned.
        """
