"""The rule that ``True`` and ``False`` are not numbers. Pydantic alone reads ``True`` as 1; this mark, added to a
type's schema, refuses it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, PydanticCustomError, core_schema


def _not_a_bool(value: Any) -> Any:
    """Raises for a ``bool``; returns any other value unchanged, for pydantic to validate as a number."""
    if isinstance(value, bool):
        raise PydanticCustomError("bool_is_not_a_number", "a yes or no is not a number; give a number")
    return value


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
    too. The fields of a host's own pydantic model or dataclass are left to pydantic. The rule applies to a
    Python value; a number read from JSON text is validated by pydantic alone, which in strict mode (a person's
    answer) refuses ``true`` with its own message.

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
    if kind in _HOSTS_OWN or (kind == "function-before" and schema["function"]["function"] is _not_a_bool):
        return schema
    if kind == "json-or-python" and schema["python_schema"].get("type") == "function-before" and schema["python_schema"]["function"]["function"] is _not_a_bool:
        return schema
    if kind in ("int", "float"):
        # Read from JSON text the number is left as it is: in strict mode pydantic refuses ``true`` itself, and the
        # only JSON read in lax mode is a saved run, which conductor wrote. So a save costs no call per number.
        return core_schema.json_or_python_schema(json_schema=schema, python_schema=core_schema.no_info_before_validator_function(_not_a_bool, schema))  # type: ignore[arg-type]  # a core schema, read here as a dict
    if (kind == "literal" and all(a_number(type(value)) for value in schema["expected"])) or (kind == "enum" and schema.get("sub_type") in ("int", "float")):
        return core_schema.no_info_before_validator_function(_not_a_bool, schema)  # type: ignore[arg-type]  # pydantic takes True for 1 here from JSON too
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


def a_number(cls: type) -> bool:
    """Says whether ``cls`` is a class of number that a yes or no must not be read as: ``int`` or ``float`` or a
    class of theirs, never ``bool``, though Python counts ``True`` as 1."""
    return issubclass(cls, (int, float)) and not issubclass(cls, bool)
