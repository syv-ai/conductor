"""The types this library's nodes are declared with.

Conductor itself ships no concrete ``DType``, so a node library declares
its own. Four cover this catalog: ``Text``, ``Number``, ``Flag`` and
``Json``. An application with its own vocabulary declares its own types
and does not need these.

``Category`` is the closed set of category names this package uses, and
``StdlibNode`` is the base every node here subclasses: it narrows
``category`` from the ``str`` conductor leaves open to that ``Literal``,
so a misspelled category is a type error rather than a new, accidental
palette section.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from conductor.dtype import DType
from conductor.node import NodeDefinition
from pydantic_core import core_schema

Category = Literal["control", "json", "logic", "math", "regex", "text"]


class StdlibNode(NodeDefinition):
    """Base of every node in this package; pins ``category`` to ``Category``."""

    category: ClassVar[Category]


class Text(DType, str):
    id = "text"
    title = "Text"


class Number(DType, float):
    id = "number"
    title = "Number"


class Flag(DType, int):
    """A boolean value.

    Built on ``int`` because ``bool`` cannot be subclassed, and a ``bool``
    is an ``int``: ``Flag(True) == 1`` and ``bool(Flag(0)) is False``.
    """

    id = "flag"
    title = "Flag"

    def __repr__(self) -> str:
        return f"Flag({bool(self)})"


class Json(DType):
    """Any JSON-shaped value — an object, a list, a scalar or ``None`` — held in ``value``.

    The widest type in the library. A node that actually handles any value
    (parses it, walks it) declares ``Json``. A node that only passes a
    value through without looking at it declares ``Any`` instead, as
    ``decision`` does.
    """

    id = "json"
    title = "JSON"

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Json) and other.value == self.value

    def __repr__(self) -> str:
        return f"Json({self.value!r})"

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: Any
    ) -> core_schema.CoreSchema:
        """Wrap whatever arrives; there is no builtin to validate as."""
        return core_schema.no_info_plain_validator_function(
            lambda value: value if isinstance(value, cls) else cls(value)
        )
