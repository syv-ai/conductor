"""The rules every type's values are held to, and the pydantic validators built from a type's schema with them.

Example: pydantic reads ``True`` as 1 for an ``int``, so ``Type(list[int]).schema``, pydantic's own schema, takes
``[1, True]`` as ``[1, 1]``. ``schema_validator_of(Type(list[int]))`` is the pydantic validator built from that schema,
with the rules below put in, and it refuses ``[1, True]``: ``bool_is_not_a_number`` at ``loc`` ``(1,)``.

A type's own rules (what a ``Series`` is, a table's columns, ``Param``'s limits and choices) are part of its schema,
declared in the type's own package. The rules here hold for a value of any type, so this package puts them into a
type's schema when it builds a validator from it, once per type:

- A yes or no is not a number (``numbers_refuse_bools``).
- A host's class whose own check gives another class is refused when a node returns it or a person answers with it
  (``not_its_class``, in ``output_schema_of``).

Two more rules hold for every value and are declared where the order of imports needs them: a node id and the
name of an input or output hold none of the characters sources are written with (``place.is_name``), and a
set has no order, so it is never read as a list (``series.set_has_no_order``). A run's limits are read by the
limits validator (``LimitsValidator``).

What a validator reads a value as:

- ``schema_validator_of`` reads it as an argument a node is called with: a constant, a default, a graph input, a value
  an edge gives. Called lax (``"3"`` for an ``int`` is 3) or strict.
- ``schema_output_validator_of`` reads it as an output a run keeps: what a node returns and a person's answer, read
  strictly. A ``Series`` is kept in the form a run stores it in (``SeriesItems``, ``Records``).
- ``exact_classes`` says which values need no reading at all, so a validator can skip pydantic for them.
"""

from __future__ import annotations

import functools
import typing
from collections.abc import Callable, Mapping
from typing import Any
from weakref import WeakKeyDictionary

from pydantic_core import CoreSchema, PydanticCustomError, SchemaValidator, core_schema

from conductor._internal.types import NAMES, Type, a_number, union_members


def _once_per_type[R](build: Callable[[Type], R]) -> Callable[[Type], R]:
    """Makes ``build`` run once per ``Type``, keeping its result for as long as the type lives."""
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
    """The schema an argument of ``value_type`` is read by: the type's schema with every rule above put in. A validator
    that reads several values in one pydantic validator (the inputs validator) puts these schemas together."""
    return numbers_refuse_bools(value_type.schema)


@_once_per_type
def schema_validator_of(value_type: Type) -> SchemaValidator:
    """The validator an argument of ``value_type`` is read by, built from ``schema_of``.

    Example: ``schema_validator_of(Type(Annotated[int, Param(le=3)])).validate_python(12)`` raises pydantic's
    ``ValidationError`` (``less_than_equal``), and ``value_type.refusal`` says why in words a host is shown.
    """
    return SchemaValidator(schema_of(value_type))


@_once_per_type
def output_schema_of(value_type: Type) -> CoreSchema:
    """The schema an output of ``value_type`` is read by, strictly, into the form a run keeps.

    It is the type's ``returned_schema`` with the rules above put in. A host's class may give its own schema,
    and so another class than itself; where the type's classes are a host's, a value pydantic reads as another
    class is refused (``not_its_class``).
    """
    schema = numbers_refuse_bools(value_type.returned_schema)
    exact = exact_classes(value_type)
    # conductor's own classes, which pydantic always reads as themselves
    if all(cls in NAMES for cls in exact):
        return schema
    type_id = value_type.id

    def of_its_class(read: Any) -> Any:
        if not isinstance(read, exact):
            raise PydanticCustomError(
                "not_its_class",
                "its type's own check gave a {got}, not a {type}",
                {"got": type(read).__name__, "type": type_id},
            )
        return read

    return core_schema.no_info_after_validator_function(of_its_class, schema)


@_once_per_type
def schema_output_validator_of(value_type: Type) -> SchemaValidator:
    """The validator an output of ``value_type`` is read by, built from ``output_schema_of``. A validator calls it
    with ``strict=True``: ``"12"`` is never an ``int`` and ``"no"`` never a ``bool``."""
    return SchemaValidator(output_schema_of(value_type))


@_once_per_type
def exact_classes(value_type: Type) -> tuple[type, ...]:
    """The classes whose values, of exactly that class, a validator keeps as they are, with no call to pydantic.

    ``(str,)`` for ``Type(str)``, ``(int, type(None))`` for ``Type(int | None)``. Empty when every value must be
    read: an option is no plain class (``list[int]``, a ``Literal``, a ``TypedDict``, ``Any``, a ``Series`` or a
    table), or the type has a limit, choices or other metadata.
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
    """Raises for a ``bool``; returns any other value unchanged, for pydantic to read as a number. ``bool`` can't be
    subclassed, so ``True`` and ``False`` are its only values."""
    if value is True or value is False:
        raise PydanticCustomError(
            "bool_is_not_a_number", "a yes or no is not a number; give a number"
        )
    return value


#: The keys of a core schema that hold no schema that reads a value.
_NOT_VALIDATORS = frozenset({"serialization", "metadata"})

#: The core schemas of a host's own model or dataclass, whose fields pydantic reads by the host's rules.
_HOSTS_OWN = frozenset({"model", "dataclass"})


def numbers_refuse_bools(schema: CoreSchema) -> CoreSchema:
    """Returns a copy of the pydantic core schema ``schema`` in which every ``int`` and ``float`` refuses a ``bool``.

    Example: for the schema of ``list[int]`` the result reads ``[1, 2]`` and refuses ``[1, True]``, where pydantic
    alone returns ``[1, 1]``. The same holds for a field of a ``TypedDict``, a value of a dict, an item of a
    ``Series`` and a cell of a table. A ``Literal`` of numbers and an enum of numbers refuse a ``bool`` too. The
    fields of a host's own pydantic model or dataclass are left to pydantic.

    The rule holds for a value read from JSON text as for a Python value: ``true`` in a saved run or a person's
    answer is refused for an ``int`` as ``True`` from a caller is, with the same message. ``schema`` itself is not
    changed.
    """
    walked: CoreSchema = _walked(schema, {})
    return walked


def _walked(schema: Any, definitions: Mapping[str, Any]) -> Any:
    """``schema``, or a part of one, with every number in it refusing a ``bool``.

    Args:
        schema: A core schema, or a list, tuple or other value found inside one.
        definitions: The schemas pydantic declared beside the schema (a ``TypedDict``'s, an enum's), by their
            ``ref``, so a reference to one is read as what it names.
    """
    if isinstance(schema, list):
        return [_walked(part, definitions) for part in schema]
    if isinstance(schema, tuple):
        return tuple(_walked(part, definitions) for part in schema)
    if not isinstance(schema, dict):
        return schema
    kind = schema.get("type")
    if not isinstance(kind, str):  # a mapping of fields by name, one of which may be named "type"
        return {key: _walked(part, definitions) for key, part in schema.items()}
    if kind in _HOSTS_OWN or _guarded_already(schema):
        return schema
    # each schema declared here is changed once, here; a reference to it below is left as it is
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
    """``schema`` with every schema inside it walked, the schema itself kept."""
    return {
        key: part if key in _NOT_VALIDATORS else _walked(part, definitions)
        for key, part in schema.items()
    }


def _guarded_already(schema: dict[str, Any]) -> bool:
    """Whether ``schema`` is one ``_guarded`` made, so a schema walked twice is changed once."""
    return schema["type"] == "function-before" and schema["function"]["function"] is _not_a_bool


def _a_number_schema(schema: Any) -> bool:
    """Whether the core schema reads one number: an ``int``, a ``float``, a ``Literal`` of numbers, or an enum whose
    members are ints or floats."""
    kind = schema.get("type")
    if kind == "literal":
        return all(a_number(type(value)) for value in schema["expected"])
    return kind in ("int", "float") or (
        kind == "enum" and schema.get("sub_type") in ("int", "float")
    )


def _guarded(number: Any) -> CoreSchema:
    """Returns a schema that refuses a ``bool`` and reads any other value by ``number``, the schema of one number or a
    reference to it."""
    return core_schema.no_info_before_validator_function(_not_a_bool, number)
