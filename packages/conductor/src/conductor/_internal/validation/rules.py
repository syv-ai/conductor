"""The pydantic validators every value is checked by, whatever its type.

Pydantic on its own is a little too forgiving for conductor. For example, it takes ``True`` as the number 1, so
the schema of ``Type(list[int])``, pydantic's own, turns ``[1, True]`` into ``[1, 1]``. A type's own check adds
conductor's rules to its schema (``Type.argument_schema``, written by ``with_rules`` in ``types/rules.py``): a yes
or no is not a number, and a ``Literal``'s values are plain JSON. So
``schema_validator_of(Type(list[int]))`` refuses ``[1, True]``, with ``bool_is_not_a_number`` at ``loc`` ``(1,)``.

One more rule is added here, to an output: when a node returns a value or a person answers, a host's class must
come out as itself. A host's class that defines its own check can give back an object of another class, and that is
refused with ``not_its_class`` (``output_schema_of``).

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
from collections.abc import Callable
from typing import Any
from weakref import WeakKeyDictionary

from pydantic_core import CoreSchema, SchemaValidator, core_schema

from conductor._internal.problem import conductor_error
from conductor._internal.types import NAMES, Type, union_members, with_rules


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


def schema_of(value_type: Type) -> CoreSchema:
    """Returns the schema for checking an argument of ``value_type``: its ``argument_schema``, with conductor's rules.

    A validator that checks several values with one pydantic validator, such as the graph inputs validator, puts
    these schemas together.
    """
    return value_type.argument_schema


def schema_validator_of(value_type: Type) -> SchemaValidator:
    """Returns the pydantic validator for an argument of ``value_type``: its ``argument_validator``.

    For example, ``schema_validator_of(Type(Annotated[int, Param(le=3)])).validate_python(12)`` raises pydantic's
    ``ValidationError`` (``less_than_equal``). ``value_type.refusal`` turns that error into a reason a host is
    shown.
    """
    return value_type.argument_validator


@_once_per_type
def output_schema_of(value_type: Type) -> CoreSchema:
    """Returns the schema for checking an output of ``value_type``, strictly.

    It is the type's ``returned_schema`` with conductor's rules added (``with_rules``). A host's class may give its own pydantic
    schema, and that schema may give back an object of another class. So where the type is made of a host's
    classes, a value that comes out of pydantic as another class is refused with ``not_its_class``.
    """
    schema = with_rules(value_type.returned_schema)
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
