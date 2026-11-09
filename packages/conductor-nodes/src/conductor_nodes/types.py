"""The types this library's nodes are declared with.

Conductor itself ships no concrete ``DType``, so a node library declares
its own. Four cover this catalog: ``Text``, ``Number``, ``Flag`` and
``Json``. An application with its own vocabulary declares its own types
and does not need these.

``Category`` is the closed set of category names this package uses, and
``stdlib_node`` is ``@node`` with ``category`` narrowed to it, so a
misspelled category is a type error rather than a new, accidental palette
section.
"""

from __future__ import annotations

from typing import Any, Literal

from conductor import DType, node
from pydantic_core import core_schema

Category = Literal["control", "json", "logic", "math", "regex", "text"]


def stdlib_node(id: str, *, category: Category, **declared: Any) -> Any:
    """``@node`` with ``category`` pinned to ``Category``."""
    return node(id, category=category, **declared)


class Text(DType, str):
    id = "text"
    display_title = "Text"


class Number(DType, float):
    id = "number"
    display_title = "Number"


class Flag(DType, int):
    """A boolean value.

    Built on ``int`` because ``bool`` cannot be subclassed, and a ``bool``
    is an ``int``: ``Flag(True) == 1`` and ``bool(Flag(0)) is False``.
    """

    id = "flag"
    display_title = "Flag"
    column = "Boolean"

    def __repr__(self) -> str:
        return f"Flag({bool(self)})"


class Json(DType):
    """Any JSON-shaped value — an object, a list, a scalar or ``None`` — held in ``value``.

    The widest type in the library. A node that actually handles any value
    (parses it, walks it) declares ``Json``. A node that only passes a
    value through without looking at it declares ``Any`` instead, as
    ``decision`` does.

    Dumped, a ``Json`` is the value it holds, and JSON's limits are its
    own: a tuple comes back a list, a key a string, and a float that is not
    a number ``null``.
    """

    id = "json"
    display_title = "JSON"

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
        """Wrap whatever arrives, since there is no builtin to validate as; dump the value it holds."""
        return core_schema.no_info_plain_validator_function(
            lambda value: value if isinstance(value, cls) else cls(value),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda json: json.value),
        )
