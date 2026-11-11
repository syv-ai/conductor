"""Small helpers that take a type hint apart or build one: the members of a union, what a named alias stands for,
whether a class declares fields, a hint with its type variables replaced.
"""

from __future__ import annotations

import dataclasses
import functools
import operator
import types
import typing
from collections.abc import Iterable, Mapping
from typing import Any, TypeGuard, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, JsonValue

#: Maps a run-time class to the hint it means when written as a hint: ``polars.DataFrame`` to ``Table``.
READ_AS: dict[type, Any] = {}


def union_members(hint: Any) -> tuple[Any, ...]:
    """Gives the members of a union, or the hint itself as the one member of any other type."""
    return get_args(hint) if get_origin(hint) in (Union, types.UnionType) else (hint,)


def list_of(item: Any) -> Any:
    """Gives ``list[item]`` for an ``item`` known only when a hint is read."""
    return types.GenericAlias(list, (item,))


def union_of(options: Iterable[Any]) -> Any:
    """Gives the union of ``options``, which is the option itself when there is one."""
    return functools.reduce(operator.or_, options)


def is_alias(hint: Any) -> TypeGuard[typing.TypeAliasType]:
    """Says whether ``hint`` is a named type alias (``type X = ...``): typing's, or the one pydantic's own aliases are."""
    return isinstance(hint, (typing.TypeAliasType, type(JsonValue)))


def is_record(cls: Any) -> bool:
    """Says whether ``cls`` is a class that declares fields: a pydantic model, a dataclass or a ``TypedDict``."""
    return isinstance(cls, type) and (issubclass(cls, BaseModel) or dataclasses.is_dataclass(cls) or typing.is_typeddict(cls))


@functools.cache
def aliased(alias: Any) -> Any:
    """Gives what a named type alias stands for, the names it holds as text resolved where it was written."""
    holder = type("_", (), {"__annotations__": {"value": alias.__value__}, "__module__": alias.__module__})
    return typing.get_type_hints(holder, include_extras=True)["value"]


def substituted(hint: Any, bound: Mapping[TypeVar, Any]) -> Any:
    """Gives ``hint`` with each type variable replaced by what it is bound to, else its own bound, else ``Any``."""
    if isinstance(hint, TypeVar):
        return bound[hint] if hint in bound else Any if hint.__bound__ is None else hint.__bound__
    if isinstance(hint, type):
        return hint
    variables = getattr(hint, "__parameters__", ())  # typing's own, on every generic hint
    return hint[tuple(substituted(variable, bound) for variable in variables)] if variables else hint
