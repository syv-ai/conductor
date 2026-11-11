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
    if kind == "definition-ref":
        # Pydantic keeps a TypedDict's and an enum's schema aside and refers to it by name.
        referred = resolve(schema)
        if _a_number_schema(referred):  # the reference is guarded; the schema it names stays as pydantic made it
            return _guarded(schema, json_too=referred["type"] != "int" and referred["type"] != "float")
        if schema["schema_ref"] not in seen:  # a TypedDict: its fields are guarded once, in the schema itself
            seen.add(schema["schema_ref"])
            changed = numbers_refuse_bools(referred, resolve, seen)
            if changed is not referred:
                referred.clear()
                referred.update(changed)
        return schema
    if _a_number_schema(schema):
        return _guarded(schema, json_too=kind not in ("int", "float"))
    return {key: part if key in _NOT_VALIDATORS else numbers_refuse_bools(part, resolve, seen) for key, part in schema.items()}


def _a_number_schema(schema: Any) -> bool:
    """Returns whether the core schema validates one number: an ``int``, a ``float``, a ``Literal`` of numbers or an
    enum whose members are ints or floats."""
    kind = schema.get("type")
    if kind == "literal":
        return all(a_number(type(value)) for value in schema["expected"])
    return kind in ("int", "float") or (kind == "enum" and schema.get("sub_type") in ("int", "float"))


def _guarded(number: Any, *, json_too: bool) -> CoreSchema:
    """Returns a schema that refuses a ``bool`` and validates any other value by the schema ``number``.

    Args:
        number: The schema of one number, or a reference to it.
        json_too: Whether a value read from JSON text is guarded too. False for ``int`` and ``float``: in strict
            mode pydantic refuses ``true`` for them itself, and the only JSON read in lax mode is a saved run,
            which conductor wrote, so a save costs no call per number. True for a ``Literal`` and an enum, where
            pydantic takes ``true`` for 1 from JSON as well.
    """
    refusing = core_schema.no_info_before_validator_function(_not_a_bool, number)
    return refusing if json_too else core_schema.json_or_python_schema(json_schema=number, python_schema=refusing)


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
