"""The rule that ``True`` and ``False`` are not numbers. Pydantic alone reads ``True`` as 1; this mark, added to a
type's schema, refuses it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, PydanticCustomError, core_schema


def _refuse(value: Any) -> Any:
    raise PydanticCustomError("bool_is_not_a_number", "a yes or no is not a number; give a number")


def _bool_or_number(value: Any) -> str:
    return "bool" if isinstance(value, bool) else "number"


#: The schema a ``bool`` is sent to where a number is asked for: it refuses every value.
_REFUSED = core_schema.no_info_plain_validator_function(_refuse)


def _refusing_a_bool(number: Any) -> CoreSchema:
    """Returns a schema that refuses a ``bool`` and validates any other value by the schema ``number``, in strict
    and lax mode and from JSON alike. Its JSON schema is that of ``number``."""
    return core_schema.tagged_union_schema(
        {"bool": _REFUSED, "number": number}, discriminator=_bool_or_number,
        metadata={"pydantic_js_functions": [lambda _, write: write(number)]},
    )


#: The keys of a core schema that hold no schema that validates a value.
_NOT_VALIDATORS = frozenset({"serialization", "metadata"})

#: The core schemas of a host's own model or dataclass, whose fields pydantic validates by the host's rules.
_HOSTS_OWN = frozenset({"model", "dataclass"})


def numbers_refuse_bools(schema: Any, resolve: Callable[[Any], Any], seen: set[str]) -> Any:
    """Returns the pydantic core schema ``schema`` with every ``int`` and ``float`` in it changed to refuse a
    ``bool``. This is the one place the rule is applied.

    Example: for the schema of ``list[int]`` the result validates ``[1, 2]`` and refuses ``[1, True]``, where
    pydantic alone returns ``[1, 1]``. The same holds for a field of a ``TypedDict``, a value of a dict, an
    item of a ``Series`` and a cell of a table. A ``Literal`` of numbers and an enum of numbers refuse a ``bool``
    too. The fields of a host's own pydantic model or dataclass are left to pydantic.

    Args:
        schema: A core schema, or a list, tuple or other value found inside one.
        resolve: Pydantic's ``handler.resolve_ref_schema``, which returns the schema a reference stands for.
        seen: The references already changed, so a type that contains itself is changed once.
    """
    if isinstance(schema, list):
        return [numbers_refuse_bools(part, resolve, seen) for part in schema]
    if isinstance(schema, tuple):
        return tuple(numbers_refuse_bools(part, resolve, seen) for part in schema)
    if not isinstance(schema, dict):
        return schema
    kind = schema.get("type")
    if not isinstance(kind, str):  # a mapping of fields by name, one of which may be named "type"
        return {key: numbers_refuse_bools(part, resolve, seen) for key, part in schema.items()}
    if kind in _HOSTS_OWN or (kind == "tagged-union" and schema["discriminator"] is _bool_or_number):
        return schema
    if kind in ("int", "float"):
        return _refusing_a_bool(schema)
    if kind == "literal" and all(_a_number(type(value)) for value in schema["expected"]):
        return _refusing_a_bool(schema)
    if kind == "enum" and schema.get("sub_type") in ("int", "float"):
        return _refusing_a_bool(schema)
    if kind == "definition-ref":  # pydantic keeps a TypedDict's schema aside and refers to it; that one is changed in place
        if schema["schema_ref"] not in seen:
            seen.add(schema["schema_ref"])
            referred = resolve(schema)
            changed = numbers_refuse_bools(referred, resolve, seen)
            if changed is not referred:
                referred.clear()
                referred.update(changed)
        return schema
    return {key: part if key in _NOT_VALIDATORS else numbers_refuse_bools(part, resolve, seen) for key, part in schema.items()}


class _NumbersRefuseBools:
    """The mark ``Type`` puts on every hint it hands to pydantic, so the schema pydantic builds for the hint is
    passed through ``numbers_refuse_bools``."""

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        refusing: CoreSchema = numbers_refuse_bools(handler(source), handler.resolve_ref_schema, set())
        return refusing


NUMBERS_REFUSE_BOOLS = _NumbersRefuseBools()


def _a_number(cls: type) -> bool:
    """Says whether ``cls`` is a class of number that a yes or no must not be read as: ``int`` or ``float`` or a
    class of theirs, never ``bool``, though Python counts ``True`` as 1."""
    return issubclass(cls, (int, float)) and not issubclass(cls, bool)
