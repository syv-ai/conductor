"""The rules conductor adds to every type's own check, beyond pydantic's.

``with_rules`` returns a type's core schema with them, wherever a value sits in it: on the value itself, an item of a
list or a ``Series``, a field of a ``TypedDict``, a value of a dict and a cell of a table. The fields of a host's own
model or dataclass are left to the host's rules.

- A yes or no is not a number. Pydantic takes ``True`` as the number 1, so the schema of ``list[int]`` turns
  ``[1, True]`` into ``[1, 1]``; with the rule it is refused with ``bool_is_not_a_number`` at ``loc`` ``(1,)``.
- A ``Literal``'s values are plain JSON: text, numbers, yes or no, or ``None``, each with a JSON form of its own.
  Pydantic reads a ``Literal``'s value back from JSON only when it is plain JSON, so ``Literal[Color.RED]`` or
  ``Literal[b"a"]`` couldn't be saved, described or asked. The walk refuses such a type, so ``@node`` does
  (``literal_invalid``).

``Type.argument_validator`` is the type's check with these rules: the one an edge's value, a graph input and a
constant get, and the one ``Type.takes_value`` asks.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic_core import CoreSchema, core_schema

from conductor._internal.base import canonical_json
from conductor._internal.problem import NodeDeclarationError, conductor_error, declaration_problems

from .hint import a_number


def _not_a_bool(value: Any) -> Any:
    """Raises for a ``bool``, and returns any other value unchanged for pydantic to check as a number.

    ``bool`` can't be subclassed, so ``True`` and ``False`` are its only values.
    """
    if value is True or value is False:
        raise conductor_error("bool_is_not_a_number", "a yes or no is not a number")
    return value


#: The keys of a core schema whose values are not schemas that check a value, so the walk leaves them alone.
_NOT_VALIDATORS = frozenset({"serialization", "metadata"})

#: The core schema types of a host's own model or dataclass. Pydantic checks their fields by the host's own rules.
_HOSTS_OWN = frozenset({"model", "dataclass"})


def with_rules(schema: CoreSchema) -> CoreSchema:
    """Returns a copy of a pydantic core schema with conductor's rules added, at any depth.

    For the schema of ``list[int]``, the result takes ``[1, 2]`` and refuses ``[1, True]``, where pydantic alone
    returns ``[1, 1]``. A ``Literal`` of numbers and an enum of numbers refuse a ``bool`` too. The rule holds for JSON
    text too: a ``true`` in a saved run or in a person's answer is refused for an ``int``, just like ``True`` from a
    caller, and with the same message. ``schema`` itself is left unchanged.

    Raises:
        NodeDeclarationError: ``literal_invalid``, for a ``Literal`` anywhere in ``schema`` whose values aren't plain
            JSON.
    """
    walked: CoreSchema = _walked(schema, {})
    return walked


def _walked(schema: Any, definitions: Mapping[str, Any]) -> Any:
    """Returns ``schema``, or a part of one, with every number in it set to refuse a ``bool``.

    It walks down through the schema, wraps each number schema it finds with ``_guarded``, and checks each ``literal``
    schema's values (``_check_literal``).

    Args:
        schema: A core schema, or a list, tuple or other value found inside one.
        definitions: The schemas pydantic declared beside the schema, such as a ``TypedDict``'s or an enum's, by
            their ``ref``. The walk looks a reference up here to see what it points to.
    """
    if isinstance(schema, list):
        return [_walked(part, definitions) for part in schema]
    if isinstance(schema, tuple):
        return tuple(_walked(part, definitions) for part in schema)
    if not isinstance(schema, dict):
        return schema
    kind = schema.get("type")
    if not isinstance(kind, str):  # not a schema but fields by name, and one may be named "type"
        return {key: _walked(part, definitions) for key, part in schema.items()}
    if kind in _HOSTS_OWN or _walked_already(schema):
        return schema
    # Walk each declared schema once, here. A declared number is kept as it is: each reference to it is wrapped.
    if kind == "definitions":
        declared = {
            **definitions,
            **{definition["ref"]: definition for definition in schema["definitions"]},
        }
        return {
            **schema,
            "schema": _walked(schema["schema"], declared),
            "definitions": [
                definition if _a_number_schema(definition) else _walked_inside(definition, declared)
                for definition in schema["definitions"]
            ],
        }
    if kind == "definition-ref":
        referred = definitions[schema["schema_ref"]]
        return _guarded(schema) if _a_number_schema(referred) else schema
    if kind == "literal":
        _check_literal(schema["expected"])
    if _a_number_schema(schema):
        return _guarded(schema)
    return _walked_inside(schema, definitions)


def _walked_inside(schema: dict[str, Any], definitions: Mapping[str, Any]) -> dict[str, Any]:
    """Returns ``schema`` with every schema inside it walked, and ``schema``'s own type kept."""
    return {
        key: part if key in _NOT_VALIDATORS else _walked(part, definitions)
        for key, part in schema.items()
    }


def _walked_already(schema: dict[str, Any]) -> bool:
    """Whether ``_guarded`` already made ``schema``, so that a schema walked twice is only wrapped once."""
    return schema["type"] == "function-before" and schema["function"]["function"] is _not_a_bool


def _check_literal(values: list[Any]) -> None:
    """Refuses a ``Literal`` whose values aren't plain JSON, each with a JSON form of its own.

    Raises:
        NodeDeclarationError: ``literal_invalid`` for a value of another class, such as an enum's member; its variant
            ``alike`` for two values written as the same JSON, like ``inf`` and ``"Infinity"``.
    """
    for value in values:
        if type(value) not in (str, int, float, bool, type(None)):  # plain JSON
            instead = (
                f"Take the enum {type(value).__name__} as the type, with Param(choices=[...]) for some of its members."
                if isinstance(value, Enum)
                else f"Take {type(value).__name__} as the type, with Param(choices=[...])."
            )
            raise NodeDeclarationError(
                declaration_problems.literal_invalid(
                    value=repr(value), cls=type(value).__name__, instead=instead
                )
            )
    if len({canonical_json(value) for value in values}) != len(values):
        raise NodeDeclarationError(declaration_problems.literal_invalid_alike(values=list(values)))


def _a_number_schema(schema: Any) -> bool:
    """Whether the core schema checks one number: an ``int``, a ``float``, a ``Literal`` of numbers, or an enum of
    ints or floats."""
    kind = schema.get("type")
    if kind == "literal":
        return all(a_number(type(value)) for value in schema["expected"])
    return kind in ("int", "float") or (
        kind == "enum" and schema.get("sub_type") in ("int", "float")
    )


def _guarded(number: Any) -> CoreSchema:
    """Returns a schema that refuses a ``bool`` and checks any other value with ``number``.

    ``number`` is the schema of one number, or a reference to one.
    """
    return core_schema.no_info_before_validator_function(_not_a_bool, number)
