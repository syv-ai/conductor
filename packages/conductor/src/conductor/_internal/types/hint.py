"""Small helpers for taking a type hint apart or building one.

They answer questions like: what are the members of this union? What does this ``type X = ...`` alias stand
for? Does this class declare fields? What is this hint with its type variables filled in?
"""

from __future__ import annotations

import dataclasses
import functools
import operator
import typing
from collections.abc import Iterable, Mapping
from types import UnionType
from typing import Any, Literal, TypeAliasType, TypeGuard, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, JsonValue

#: Classes that mean another hint when written as a hint, such as ``polars.DataFrame``, which means ``Table``.
READ_AS: dict[type, Any] = {}


def union_members(hint: Any) -> tuple[Any, ...]:
    """Returns the members of a union. Any other hint is returned as a union of one member."""
    return get_args(hint) if get_origin(hint) in (Union, UnionType) else (hint,)


def union_of(options: Iterable[Any]) -> Any:
    """Returns the union of ``options``. With a single option, it returns that option."""
    return functools.reduce(operator.or_, options)


def literal_classes(hint: Any) -> Any:
    """Returns ``hint`` with each ``Literal`` in its union read as its values' classes: one class, or several.

    A ``Literal`` is a value of its values' classes. ``Literal["a", "b"]`` gives ``str``,
    ``Literal["a", 1, None]`` gives ``str | int | None``, and ``Literal["a"] | bytes`` gives ``str | bytes``.
    """
    return union_of(
        dict.fromkeys(
            value_class
            for member in union_members(hint)
            for value_class in (
                [type(value) for value in get_args(member)]
                if get_origin(member) is Literal
                else [member]
            )
        )
    )


def is_alias(hint: Any) -> TypeGuard[TypeAliasType]:
    """Says whether ``hint`` is a named type alias, like ``type X = ...`` or pydantic's ``JsonValue``."""
    return isinstance(hint, (TypeAliasType, type(JsonValue)))


def is_record(cls: Any) -> bool:
    """Says whether ``cls`` is a record: a pydantic model, a dataclass or a ``TypedDict``."""
    return isinstance(cls, type) and (
        issubclass(cls, BaseModel) or dataclasses.is_dataclass(cls) or typing.is_typeddict(cls)
    )


def a_number(cls: type) -> bool:
    """Says whether ``cls`` is a number class: ``int``, ``float`` or a subclass of one, but not ``bool``.

    Python counts ``True`` as ``1``, but conductor never takes a yes or no as a number. This is the one place that
    rule says what a number is. ``fit`` uses it for types when a graph is compiled, and ``types/rules.py`` for
    values.
    """
    return issubclass(cls, (int, float)) and not issubclass(cls, bool)


@functools.cache
def aliased(alias: Any) -> Any:
    """Returns what a named type alias stands for.

    Names written as strings inside the alias are resolved in the module where the alias was written.
    """
    holder = type(
        "_", (), {"__annotations__": {"value": alias.__value__}, "__module__": alias.__module__}
    )
    return typing.get_type_hints(holder, include_extras=True)["value"]


def substituted(hint: Any, bound: Mapping[TypeVar, Any]) -> Any:
    """Returns ``hint`` with each type variable filled in.

    A variable is replaced by what ``bound`` binds it to. If ``bound`` doesn't have it, it is replaced by the
    variable's own bound, or by ``Any`` if it has none.
    """
    if isinstance(hint, TypeVar):
        if hint in bound:
            return bound[hint]
        if hint.__bound__ is not None:
            return hint.__bound__
        return Any
    if isinstance(hint, type):
        return hint
    variables = getattr(hint, "__parameters__", ())  # typing sets this on every generic hint
    return (
        hint[tuple(substituted(variable, bound) for variable in variables)] if variables else hint
    )
