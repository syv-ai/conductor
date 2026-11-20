"""The rules conductor adds to every type's own check, beyond pydantic's.

``with_rules`` returns a type's core schema with them, wherever a value sits in it: on the value itself, an item of a
list or a ``Series``, a field of a ``TypedDict``, a value of a dict and a cell of a table. A dict's key is left as
pydantic reads it, from its text. The fields of a host's own
model or dataclass are left to the host's rules.

- A yes or no is not a number. Pydantic takes ``True`` as the number 1, so the schema of ``list[int]`` turns
  ``[1, True]`` into ``[1, 1]``; with the rule it is refused with ``bool_is_not_a_number`` at ``loc`` ``(1,)``.
- A named value, a ``Literal``'s value or an enum's member, reads back from its own JSON as itself, so a run of it
  can be saved, described and asked. ``Literal[Color.RED]`` of a plain enum, ``Literal[b"a"]`` and an enum of
  tuples don't, so the walk refuses them, and ``@node`` does (``named_value_invalid``).

``Type.argument_validator`` is the type's check with these rules: the one an edge's value, a graph input and a
constant get, and the one ``Type.takes_value`` asks.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic import ValidationError
from pydantic_core import CoreSchema, PydanticSerializationError, SchemaValidator, core_schema

from conductor._internal.base import (
    HOSTS_OWN_CLASSES,
    JSON_CORE_CONFIG,
    canonical_json,
    plain_json,
    under_definitions,
    validate_as_json,
    with_schemas_in,
)
from conductor._internal.problem import NodeDeclarationError, conductor_error, declaration_problems

from .hint import a_number


def _not_a_bool(value: Any) -> Any:
    """Raises for a ``bool``, and returns any other value unchanged for pydantic to check as a number.

    ``bool`` can't be subclassed, so ``True`` and ``False`` are its only values.
    """
    if value is True or value is False:
        raise conductor_error("bool_is_not_a_number", "a yes or no is not a number")
    return value


def with_rules(schema: CoreSchema) -> CoreSchema:
    """Returns a copy of a pydantic core schema with conductor's rules added, at any depth.

    For the schema of ``list[int]``, the result takes ``[1, 2]`` and refuses ``[1, True]``, where pydantic alone
    returns ``[1, 1]``. A ``Literal`` of numbers and an enum of numbers refuse a ``bool`` too. The rule holds for JSON
    text too: a ``true`` in a saved run or in a person's answer is refused for an ``int``, just like ``True`` from a
    caller, and with the same message. ``schema`` itself is left unchanged.

    Raises:
        NodeDeclarationError: ``named_value_invalid``, for a ``Literal`` or an enum anywhere in ``schema`` whose named
            values don't read back from their own JSON.
    """
    walked: CoreSchema = _walked(schema, {})
    return walked


def _walked(schema: Mapping[str, Any], definitions: Mapping[str, Any]) -> Any:
    """Returns a core schema with every number in it set to refuse a ``bool``.

    It walks down through the schema, wraps each number schema it finds with ``_guarded``, and checks each ``literal``
    and ``enum`` schema's named values (``_check_named_values``).

    Args:
        schema: A core schema.
        definitions: The schemas pydantic declared beside the schema, such as a ``TypedDict``'s or an enum's, by
            their ``ref``. The walk looks a reference up here to see what it points to.

    Raises:
        NodeDeclarationError: ``named_value_invalid``, from ``_check_named_values``.
    """
    kind = schema["type"]
    if kind in HOSTS_OWN_CLASSES or _walked_already(schema):
        return schema
    # Walk each declared schema once, here. A declared number is kept as it is: each reference to it is wrapped.
    if kind == "definitions":
        inner, declared_here = under_definitions(schema)
        declared = {**definitions, **declared_here}
        # a declared enum is checked here, since a number isn't walked into
        for definition in declared_here.values():
            _check_named_values(definition)
        return {
            **schema,
            "schema": _walked(inner, declared),
            "definitions": [
                definition if _a_number_schema(definition) else _walked_inside(definition, declared)
                for definition in declared_here.values()
            ],
        }
    if kind == "definition-ref":
        referred = definitions[schema["schema_ref"]]
        return _guarded(schema) if _a_number_schema(referred) else schema
    _check_named_values(schema)
    if _a_number_schema(schema):
        return _guarded(schema)
    return _walked_inside(schema, definitions)


def _check_named_values(schema: Mapping[str, Any]) -> None:
    """Refuses a ``literal`` or ``enum`` schema whose named values don't each read back from their own JSON.

    Each value is written as JSON and read back by the schema, as a saved run or an answer reads it
    (``validate_as_json``), and must come back as itself, of the same class. ``Literal["a", 1]``,
    ``Literal[IntEnum.ONE]`` and an enum of text do. ``Literal[Color.RED]`` of a plain enum, ``Literal[b"a"]``,
    ``Literal[float("nan")]`` and an enum of tuples don't.

    Raises:
        NodeDeclarationError: ``named_value_invalid`` for a ``Literal``'s value, ``named_value_invalid_enum`` for an enum's
            member, and ``named_value_invalid_alike`` for two of a ``Literal``'s values or an enum's members written as the same JSON, like
            ``inf`` and ``"Infinity"``.
    """
    kind = schema.get("type")
    if kind not in ("literal", "enum"):
        return
    values = schema["expected"] if kind == "literal" else schema["members"]
    reader = SchemaValidator(schema, config=JSON_CORE_CONFIG)
    for value in values:
        try:
            read = validate_as_json(reader, value)
        except (
            ValidationError,
            PydanticSerializationError,
        ):  # it has no JSON form, or that form reads as nothing
            read_back = False
        else:
            read_back = type(read) is type(value) and (read is value or read == value)
        if read_back:
            continue
        if kind == "enum":
            raise NodeDeclarationError(
                declaration_problems.named_value_invalid_enum(
                    enum=schema["cls"].__name__, member=repr(value)
                )
            )
        # what to write instead: the enum as the type, when the enum reads its own members back
        instead = f"Take {type(value).__name__} as the type, with Param(choices=[...]) for some of its values."
        if isinstance(value, Enum):
            from .type import Type  # Type is built on these rules, so it is imported here

            try:
                Type(type(value)).argument_schema  # noqa: B018  fails for an enum whose members don't read back
            except NodeDeclarationError:
                instead = (
                    f"Give the values of the enum {type(value).__name__} as text, numbers or yes or no, and take it"
                    " as the type."
                )
            else:
                instead = (
                    f"Take the enum {type(value).__name__} as the type, with Param(choices=[...]) for some of its"
                    " members."
                )
        raise NodeDeclarationError(
            declaration_problems.named_value_invalid(value=repr(value), instead=instead)
        )
    # two values written as one JSON can't be told apart, in a Literal or an enum, like inf and "Infinity"
    if len({canonical_json(plain_json(value)) for value in values}) != len(values):
        raise NodeDeclarationError(
            declaration_problems.named_value_invalid_alike(values=list(values))
        )


def _walked_inside(schema: Mapping[str, Any], definitions: Mapping[str, Any]) -> dict[str, Any]:
    """Returns ``schema`` with every schema inside it walked, and ``schema``'s own type kept.

    A dict's ``keys_schema`` is left alone: a key is text in JSON, and so never a yes or no. Pydantic reads a key
    like ``"1"`` from its text only when the key's schema is its own, so a guard around it would refuse every key of
    ``dict[int, str]`` given as JSON.
    """
    return with_schemas_in(
        schema, lambda part, key: part if key == "keys_schema" else _walked(part, definitions)
    )


def _walked_already(schema: Mapping[str, Any]) -> bool:
    """Whether ``_guarded`` already made ``schema``, so that a schema walked twice is only wrapped once."""
    return schema["type"] == "function-before" and schema["function"]["function"] is _not_a_bool


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
