"""The rules conductor adds to every type's own check, beyond pydantic's.

Pydantic on its own is a little too forgiving for conductor, and in one place too strict. ``with_rules`` returns a
type's core schema with two rules added, wherever a value sits in it: on the value itself, an item of a list or a
``Series``, a field of a ``TypedDict``, a value of a dict and a cell of a table. The fields of a host's own model or
dataclass are left to the host's rules, since pydantic checks them by the class's own validator.

- A yes or no is not a number. Pydantic takes ``True`` as the number 1, so the schema of ``list[int]`` turns
  ``[1, True]`` into ``[1, 1]``; with the rule it is refused with ``bool_is_not_a_number`` at ``loc`` ``(1,)``.
- An enum member in a ``Literal`` is read from its JSON as the enum reads it. Pydantic writes
  ``Literal[Color.RED]``'s value as ``"red"`` but reads ``"red"`` from JSON as no value of it, so a saved run, an
  answer and the type's own choices couldn't be read back. With the rule, the JSON ``"red"`` is read by ``Color``'s
  own schema, and kept when the member it gives is one of the ``Literal``'s values. A Python value is left as it is,
  so a node returning ``"red"`` is still refused.

``Type.argument_validator`` is the type's check with both rules: the one an edge's value, a graph input and a
constant get, and the one ``Type.takes_value`` asks.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from enum import Enum
from typing import Any

from pydantic import TypeAdapter, ValidationError, ValidationInfo
from pydantic_core import CoreSchema, core_schema, to_json

from conductor._internal.problem import conductor_error

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

#: The core schema types of a host's own model or dataclass. Pydantic checks their fields by the host's own rules: it
#: calls the class's own validator, whatever schema the class sits in, so no rule written here could reach them.
_HOSTS_OWN = frozenset({"model", "dataclass"})


def with_rules(schema: CoreSchema) -> CoreSchema:
    """Returns a copy of a pydantic core schema with conductor's two rules added, at any depth.

    For the schema of ``list[int]``, the result takes ``[1, 2]`` and refuses ``[1, True]``, where pydantic alone
    returns ``[1, 1]``. A ``Literal`` of numbers and an enum of numbers refuse a ``bool`` too. For the schema of
    ``list[Literal[Color.RED, Color.BLUE]]``, the JSON ``["red"]`` is ``[Color.RED]``.

    Both rules hold for JSON text and for Python values alike, except that a member is read only from JSON. A
    ``true`` in a saved run or in a person's answer is refused for an ``int``, just like ``True`` from a caller, and
    with the same message. ``schema`` itself is left unchanged.
    """
    walked: CoreSchema = _walked(schema, {})
    return walked


def _walked(schema: Any, definitions: Mapping[str, Any]) -> Any:
    """Returns ``schema``, or a part of one, with the rules added to every number and every ``Literal`` of members.

    It walks down through the schema, wraps each number schema it finds with ``_guarded``, and each ``literal``
    schema that holds an enum member with ``_MembersOf``.

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
    guarded = _guarded(schema) if _a_number_schema(schema) else None
    if kind == "literal" and any(isinstance(value, Enum) for value in schema["expected"]):
        return core_schema.with_info_before_validator_function(
            _MembersOf(tuple(schema["expected"])), guarded or schema
        )
    return guarded or _walked_inside(schema, definitions)


def _walked_inside(schema: dict[str, Any], definitions: Mapping[str, Any]) -> dict[str, Any]:
    """Returns ``schema`` with every schema inside it walked, and ``schema``'s own type kept."""
    return {
        key: part if key in _NOT_VALIDATORS else _walked(part, definitions)
        for key, part in schema.items()
    }


def _walked_already(schema: dict[str, Any]) -> bool:
    """Whether the walk already made ``schema``, so that a schema walked twice gets each rule once."""
    if schema["type"] != "function-before":
        return False
    function = schema["function"]["function"]
    return function is _not_a_bool or isinstance(function, _MembersOf)


class _MembersOf:
    """Reads a JSON value as one of a ``Literal``'s enum members, by the member's own enum.

    The value is read by each enum among the ``Literal``'s values in turn, from its JSON text, as pydantic reads that
    enum from JSON. The first member read that is one of the ``Literal``'s values is kept. A value no enum reads as
    one of them is passed on as it is, for the ``Literal`` to refuse. A Python value is always passed on as it is.
    """

    __slots__ = ("_enums", "_members")

    def __init__(self, values: tuple[Any, ...]) -> None:
        self._members = values
        self._enums = tuple(
            dict.fromkeys(type(value) for value in values if isinstance(value, Enum))
        )

    def __call__(self, value: Any, info: ValidationInfo) -> Any:
        if info.mode != "json":
            return value
        text = to_json(value)
        for enum in self._enums:
            try:
                read = _enum_reader(enum).validate_json(text, strict=True)
            except ValidationError:
                continue
            if any(read is member for member in self._members):
                return read
        return value


@functools.cache
def _enum_reader(enum: type[Enum]) -> TypeAdapter[Any]:
    """Returns pydantic's reader of ``enum``, built once for each enum."""
    return TypeAdapter(enum)


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
