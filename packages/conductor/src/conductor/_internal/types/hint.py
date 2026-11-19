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
from datetime import date, datetime
from ipaddress import (
    IPv4Address,
    IPv4Interface,
    IPv4Network,
    IPv6Address,
    IPv6Interface,
    IPv6Network,
)
from types import UnionType
from typing import (
    Annotated,
    Any,
    Literal,
    TypeAliasType,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

from pydantic import (
    AwareDatetime,
    BaseModel,
    EmailStr,
    FutureDate,
    FutureDatetime,
    IPvAnyAddress,
    IPvAnyInterface,
    IPvAnyNetwork,
    JsonValue,
    NaiveDatetime,
    PastDate,
    PastDatetime,
)

from conductor._internal.problem import NodeDeclarationError, declaration_problems

#: Classes that mean another hint when written as a hint, such as ``polars.DataFrame``, which means ``Table``.
#:
#: Pydantic's marker classes are here. A value of one is never of the marker's own class: ``EmailStr`` checks a
#: ``str`` and gives a ``str``. So each is read as the class of its values with the marker's check on it, the way
#: pydantic's ``PositiveInt`` is an ``int`` with a check: ``EmailStr`` has the id ``str``, and an edge from it lands
#: on a ``str`` input.
READ_AS: dict[Any, Any] = {
    EmailStr: Annotated[str, EmailStr],
    PastDate: Annotated[date, PastDate],
    FutureDate: Annotated[date, FutureDate],
    PastDatetime: Annotated[datetime, PastDatetime],
    FutureDatetime: Annotated[datetime, FutureDatetime],
    AwareDatetime: Annotated[datetime, AwareDatetime],
    NaiveDatetime: Annotated[datetime, NaiveDatetime],
    IPvAnyAddress: Annotated[IPv4Address | IPv6Address, IPvAnyAddress],
    IPvAnyInterface: Annotated[IPv4Interface | IPv6Interface, IPvAnyInterface],
    IPvAnyNetwork: Annotated[IPv4Network | IPv6Network, IPvAnyNetwork],
}


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


def declares_fields(cls: Any) -> bool:
    """Says whether ``cls`` declares its fields: a pydantic model, a dataclass or a ``TypedDict``."""
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
    origin = get_origin(hint) or hint
    if (is_alias(hint) or typing.is_typeddict(origin)) and id(origin) not in walked:
        walked |= {id(origin)}
        if is_alias(hint):
            parts.append(hint.__value__)
        else:
            try:
                parts += typing.get_type_hints(origin, include_extras=True).values()
            # a field written as text that names nothing Python can find
            except NameError:
                raise NodeDeclarationError(
                    declaration_problems.type_not_supported(type=origin.__qualname__)
                ) from None
    for part in parts:
        yield from hint_parts(part, walked)
