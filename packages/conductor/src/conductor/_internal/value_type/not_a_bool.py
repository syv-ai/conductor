"""The rule that ``True`` and ``False`` are not numbers. Pydantic alone reads ``True`` as 1; this mark, added to a
type's schema, refuses it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, PydanticCustomError, core_schema


def _not_a_bool(value: Any) -> Any:
    """Refuses a ``bool`` before pydantic reads it as the number Python counts it as."""
    if isinstance(value, bool):
        raise PydanticCustomError("bool_is_not_a_number", "a yes or no is not a number; give a number")
    return value


@dataclass(frozen=True)
class _NotABool:
    """The one rule that a yes or no is not a number: refuses a ``bool`` where pydantic alone reads ``True`` as 1.

    A ``Type`` adds it to a hint after asking pydantic what it makes of ``True`` there.

    Attributes:
        strict_too: Whether pydantic takes ``True`` in strict mode as well (a ``Literal`` of numbers). When false,
            a strict read is pydantic's own, untouched.
    """

    strict_too: bool

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Gives the schema of ``source`` behind the refusal."""
        plain = handler(source)
        guarded = core_schema.no_info_before_validator_function(_not_a_bool, plain)
        return guarded if self.strict_too else core_schema.lax_or_strict_schema(lax_schema=guarded, strict_schema=plain)


def _a_number(cls: type) -> bool:
    """Says whether ``cls`` is a class of number that a yes or no must not be read as: ``int`` or ``float`` or a
    class of theirs, never ``bool``, though Python counts ``True`` as 1."""
    return issubclass(cls, (int, float)) and not issubclass(cls, bool)
