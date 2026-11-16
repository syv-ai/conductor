"""The rules every value is checked by, whatever its type, and the pydantic validators that add them.

Pydantic on its own is a little too forgiving for conductor. For example, it takes ``True`` as the number 1, so
the schema of ``Type(list[int])``, pydantic's own, turns ``[1, True]`` into ``[1, 1]``. The validators built here
add conductor's rules to a type's schema. So ``schema_validator_of(Type(list[int]))`` refuses ``[1, True]``, with
``bool_is_not_a_number`` at ``loc`` ``(1,)``.

Two rules are added here, to every type:

- A yes or no is not a number (``numbers_refuse_bools``).
- When a node returns a value or a person answers, a host's class must come out as itself. A host's class that
  defines its own check can give back an object of another class, and that is refused with ``not_its_class``
  (``output_schema_of``).

A type's own rules, such as a ``Param``'s limits or a table's columns, are already part of its schema, declared in
the type's own package. Two more rules hold for every value but live elsewhere, where the order of imports needs
them. A name can't contain the characters places are written with (``is_name``, in ``base/place.py``). And a set
is never read as a list, since a set has no order (``set_has_no_order``, in ``series/``).

There are two validators for each type, each built once:

- ``schema_validator_of`` checks an argument a node is called with: a constant, a default, a graph input, or a
  value from an edge. It may be called lax, where ``"3"`` is the ``int`` 3, or strict.
- ``schema_output_validator_of`` checks an output: what a node returns, or a person's answer. It is called
  strictly. A ``Series`` comes out as the type a run stores it as (``SeriesItems`` or ``Records``).

``exact_classes`` says which values are already right, so a validator can skip pydantic for them.
"""

from __future__ import annotations

import functools
import typing
from collections.abc import Callable, Mapping
from typing import Any
from weakref import WeakKeyDictionary

from pydantic_core import CoreSchema, SchemaValidator, core_schema

from conductor._internal.problem import conductor_error
from conductor._internal.types import NAMES, Type, a_number, union_members


def _once_per_type[R](build: Callable[[Type], R]) -> Callable[[Type], R]:
    """Makes ``build`` run only once for each ``Type``, keeping its result for as long as the type lives."""
    built: WeakKeyDictionary[Type, R] = WeakKeyDictionary()

    @functools.wraps(build)
    def once(value_type: Type) -> R:
        try:
            return built[value_type]
        except KeyError:
            made = built[value_type] = build(value_type)
            return made

    return once


@_once_per_type
def schema_of(value_type: Type) -> CoreSchema:
    """Returns the schema for checking an argument of ``value_type``: the type's schema with the rules above added.

    A validator that checks several values with one pydantic validator, such as the graph inputs validator, puts
    these schemas together.
    """
    return numbers_refuse_bools(value_type.schema)


@_once_per_type
def schema_validator_of(value_type: Type) -> SchemaValidator:
    """Returns the pydantic validator for an argument of ``value_type``, built from ``schema_of``.

    For example, ``schema_validator_of(Type(Annotated[int, Param(le=3)])).validate_python(12)`` raises pydantic's
    ``ValidationError`` (``less_than_equal``). ``value_type.refusal`` turns that error into a reason a host is
    shown.
    """
    return SchemaValidator(schema_of(value_type))


@_once_per_type
def output_schema_of(value_type: Type) -> CoreSchema:
    """Returns the schema for checking an output of ``value_type``, strictly.

    It is the type's ``returned_schema`` with the rules above added. A host's class may give its own pydantic
    schema, and that schema may give back an object of another class. So where the type is made of a host's
    classes, a value that comes out of pydantic as another class is refused with ``not_its_class``.
    """
    schema = numbers_refuse_bools(value_type.returned_schema)
    exact = exact_classes(value_type)
    # Conductor's own classes always come out of pydantic as themselves, so there is nothing to check.
    if all(cls in NAMES for cls in exact):
        return schema
    type_id = value_type.id

    def of_its_class(read: Any) -> Any:
        if not isinstance(read, exact):
            raise conductor_error(
                "not_its_class",
                "its type's own check returned a {got}, not a {type}",
                {"got": type(read).__name__, "type": type_id},
            )
        return read

    return core_schema.no_info_after_validator_function(of_its_class, schema)


@_once_per_type
def schema_output_validator_of(value_type: Type) -> SchemaValidator:
    """Returns the pydantic validator for an output of ``value_type``, built from ``output_schema_of``.

    Validators call it with ``strict=True``, so ``"12"`` is never an ``int`` and ``"no"`` is never a ``bool``.
    """
    return SchemaValidator(output_schema_of(value_type))


@_once_per_type
def exact_classes(value_type: Type) -> tuple[type, ...]:
    """Returns the classes whose values a validator can keep as they are, without calling pydantic.

    A value whose class is exactly one of these needs no checking. ``Type(str)`` gives ``(str,)``, and
    ``Type(int | None)`` gives ``(int, type(None))``.

    It returns ``()`` when every value has to be checked. That is when an option of the type isn't a plain class
    (``list[int]``, a ``Literal``, a ``TypedDict``, ``Any``, a ``Series`` or a table), or when the type has a limit,
    choices or other metadata.
    """
    options = union_members(value_type.bare)
    plain = all(
        isinstance(option, type)
        and not typing.is_typeddict(option)
        and option is not Any
        and Type(option).item is None
        and Type(option).record is None
        for option in options
    )
    return options if plain and not value_type.limited else ()


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


def numbers_refuse_bools(schema: CoreSchema) -> CoreSchema:
    """Returns a copy of a pydantic core schema in which every ``int`` and ``float`` refuses a ``bool``.

    For the schema of ``list[int]``, the result takes ``[1, 2]`` and refuses ``[1, True]``, where pydantic alone
    returns ``[1, 1]``. It reaches numbers anywhere inside: a field of a ``TypedDict``, a value of a dict, an item
    of a ``Series`` and a cell of a table. A ``Literal`` of numbers and an enum of numbers refuse a ``bool`` too. The
    fields of a host's own pydantic model or dataclass are left to pydantic.

    The rule holds for JSON text too. A ``true`` in a saved run or in a person's answer is refused for an ``int``,
    just like ``True`` from a caller, and with the same message. ``schema`` itself is left unchanged.
    """
    walked: CoreSchema = _walked(schema, {})
    return walked


def _walked(schema: Any, definitions: Mapping[str, Any]) -> Any:
    """Returns ``schema``, or a part of one, with every number in it set to refuse a ``bool``.

    It walks down through the schema and wraps each number schema it finds with ``_guarded``.

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
    if kind in _HOSTS_OWN or _guarded_already(schema):
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
    if _a_number_schema(schema):
        return _guarded(schema)
    return _walked_inside(schema, definitions)


def _walked_inside(schema: dict[str, Any], definitions: Mapping[str, Any]) -> dict[str, Any]:
    """Returns ``schema`` with every schema inside it walked, and ``schema``'s own type kept."""
    return {
        key: part if key in _NOT_VALIDATORS else _walked(part, definitions)
        for key, part in schema.items()
    }


def _guarded_already(schema: dict[str, Any]) -> bool:
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
