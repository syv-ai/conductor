"""Small helpers for taking a type hint apart or building one.

They answer questions like: what are the members of this union? What does this ``type X = ...`` alias stand
for? Which fields does this class declare? What is this hint with its type variables filled in?
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
    ClassVar,
    Literal,
    NotRequired,
    Required,
    TypeAliasType,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import typing_extensions
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
from typing_extensions import ReadOnly

from conductor._internal.problem import NodeDeclarationError, declaration_problems

#: The qualifiers a ``TypedDict``'s field may be written in. They say whether a value may leave the field out, or
#: change it, and not what type the field is.
_QUALIFIERS = frozenset({Required, NotRequired, ReadOnly})

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
        issubclass(cls, BaseModel) or dataclasses.is_dataclass(cls) or is_typed_dict(cls)
    )


def is_typed_dict(hint: Any) -> bool:
    """Says whether ``hint`` is a ``TypedDict``, a generic one with its arguments given too, like ``Row[int]``."""
    return typing_extensions.is_typeddict(get_origin(hint) or hint)


def field_hints(hint: Any) -> dict[str, Any]:
    """Returns the type hint of each field a pydantic model, a dataclass or a ``TypedDict`` declares, by name.

    It is empty for a type that declares no fields, such as ``str`` or ``dict[str, int]``. Conductor reads every
    type's fields with it, and a ``compute_outputs`` hook calls it to read the fields of a type arriving on an
    input: ``field_hints(Contact)`` is ``{"name": str, "email": str | None}``. It is provisional.

    A generic type has its type arguments filled in: for ``class Row[T](TypedDict)``, ``Row[int]``'s field
    ``value: T`` is ``int``, and so is a generic pydantic model's, ``Box[int]``. A type variable left open, as in a
    bare ``Row``, becomes its bound, or ``Any``.

    Each hint is the one its author wrote, with its ``Annotated`` metadata, so a field's limits and choices stay
    on it. A ``TypedDict``'s ``Required`` and ``ReadOnly`` are taken off. A field a value may leave out, written
    ``NotRequired`` or in a class declared ``total=False``, is always given as ``NotRequired[...]``, so one test
    finds it. A ``ClassVar`` is no field.

    Raises:
        NodeDeclarationError: ``type_not_supported`` for a field written as text that names nothing Python can
            find.
    """
    origin, arguments = get_origin(hint) or hint, get_args(hint)
    # pydantic makes a class of its own for Box[int], which keeps the generic class and its arguments
    generic = (
        vars(origin).get("__pydantic_generic_metadata__") if isinstance(origin, type) else None
    )
    if generic is not None and generic["origin"] is not None:
        origin, arguments = generic["origin"], generic["args"]
    if not declares_fields(origin):
        return {}
    try:
        written = typing.get_type_hints(origin, include_extras=True)
    except NameError:
        raise NodeDeclarationError(
            declaration_problems.type_not_supported(type=origin.__qualname__)
        ) from None
    bound = dict(zip(getattr(origin, "__parameters__", ()), arguments, strict=False))
    # a TypedDict says which of its keys must be there, but not for a qualifier written as text, which only the
    # resolved hint shows
    required_keys = getattr(origin, "__required_keys__", written.keys())
    fields = {}
    for name, field in written.items():
        if get_origin(field) is ClassVar:
            continue
        qualifiers = set()
        while get_origin(field) in _QUALIFIERS:
            qualifiers.add(get_origin(field))
            field = get_args(field)[0]
        field = substituted(field, bound)
        left_out = NotRequired in qualifiers or (
            Required not in qualifiers and name not in required_keys
        )
        fields[name] = NotRequired[field] if left_out else field
    return fields


def types_and_left_out(fields: Mapping[str, Any]) -> dict[str, tuple[Any, bool]]:
    """Returns each of ``field_hints``' fields as its type, with whether a value may leave it out.

    ``NotRequired[int]`` gives ``(int, True)``, and ``int`` gives ``(int, False)``.
    """
    return {
        name: (get_args(hint)[0], True) if get_origin(hint) is NotRequired else (hint, False)
        for name, hint in fields.items()
    }


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
    if (is_alias(hint) or is_typed_dict(hint)) and id(origin) not in walked:
        walked |= {id(origin)}
        if is_alias(hint):
            parts.append(hint.__value__)
        else:
            parts += field_hints(hint).values()
    for part in parts:
        yield from hint_parts(part, walked)
