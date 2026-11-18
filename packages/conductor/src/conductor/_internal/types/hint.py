"""Small helpers for taking a type hint apart or building one.

They answer questions like: what are the members of this union? What does this ``type X = ...`` alias stand
for? Does this class declare fields? What is this hint with its type variables filled in?
"""

from __future__ import annotations

import dataclasses
import functools
import operator
import typing
from collections.abc import Iterable, Iterator, Mapping
from types import UnionType
from typing import Any, Literal, TypeAliasType, TypeGuard, TypeVar, Union, get_args, get_origin

from pydantic import BaseModel, JsonValue

from conductor._internal.problem import NodeDeclarationError, declaration_problems

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


def hint_parts(hint: Any, walked: frozenset[int] = frozenset()) -> Iterator[Any]:
    """Yields ``hint`` and every hint inside it, as its author wrote them.

    It walks a union's members, a generic's arguments, ``Annotated``'s type, what a ``type X = ...`` alias stands
    for, a type variable's bound, which a variable no edge fills in becomes, and a ``TypedDict``'s fields, a generic
    one's too. A host's own model or dataclass is its own, so its fields
    are not walked. ``list[Table[Row]] | None`` yields itself, ``list[Table[Row]]``, ``Table[Row]``, ``Row``, each of
    ``Row``'s field hints and ``None``.

    Args:
        hint: The hint.
        walked: The ``TypedDict``s and aliases already walked on the way down (by ``id``), so one that holds
            itself ends.

    Raises:
        NodeDeclarationError: ``type_not_supported`` for a ``TypedDict`` field written as text that names nothing
            Python can find.
    """
    yield hint
    parts: list[Any] = list(get_args(hint))
    if isinstance(hint, TypeVar) and hint.__bound__ is not None:
        parts.append(hint.__bound__)
    record = get_origin(hint) or hint
    if (is_alias(hint) or typing.is_typeddict(record)) and id(record) not in walked:
        walked |= {id(record)}
        if is_alias(hint):
            parts.append(hint.__value__)
        else:
            try:
                parts += typing.get_type_hints(record, include_extras=True).values()
            # a field written as text that names nothing Python can find
            except NameError:
                raise NodeDeclarationError(
                    declaration_problems.type_not_supported(type=record.__qualname__)
                ) from None
    for part in parts:
        yield from hint_parts(part, walked)
