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
    Discriminator,
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


def has_own_config(cls: Any) -> bool:
    """Whether ``cls`` reads its fields by a pydantic config of its own: a model, a pydantic dataclass, or a class
    pydantic's ``with_config`` gave one, such as a ``TypedDict`` with ``__pydantic_config__``. ``False`` for
    anything that is no class, such as ``None``."""
    return isinstance(cls, type) and (
        issubclass(cls, BaseModel) or getattr(cls, "__pydantic_config__", None) is not None
    )


def allows_extra(cls: Any) -> bool:
    """Whether ``cls`` takes keys it doesn't declare, by a pydantic config of its own (``has_own_config``) whose
    ``extra`` is ``"allow"``: a model, a pydantic dataclass, or a ``TypedDict`` ``with_config`` gave one. ``False``
    for any other class, and for anything that is no class."""
    return has_own_config(cls) and _own_config(cls).get("extra") == "allow"


def _own_config(cls: type) -> Any:
    """The pydantic config a class with one of its own reads its fields by (``has_own_config``)."""
    return cls.model_config if issubclass(cls, BaseModel) else getattr(cls, "__pydantic_config__")


def revalidates_instances(cls: Any) -> bool:
    """Whether pydantic checks again a value that is already an instance of ``cls``, a model or a pydantic dataclass
    whose config's ``revalidate_instances`` isn't ``"never"``. It then reads a subclass's instance by ``cls``'s own
    config, so the value may change. ``False`` for any other class, and for anything that is no class."""
    if not has_own_config(cls):
        return False
    return _own_config(cls).get("revalidate_instances", "never") != "never"


def field_hints(hint: Any) -> dict[str, Any]:
    """Returns the type hint of each field ``hint`` declares, by name. It is empty for a hint that declares none.

    A pydantic model, a dataclass and a ``TypedDict`` declare their fields. This is the one place they are read.

    A model's field is what its ``model_fields`` holds: the type pydantic checks it as, with the ``Annotated``
    metadata pydantic keeps. So a generic model has its type arguments filled in: for ``class Box[T](BaseModel)``,
    the field ``value: T`` is ``int`` in ``Box[int]``, and in ``class IntBox(Box[int])`` too. A ``Field(ge=0)`` is
    its check, ``Ge(ge=0)``, a ``Field(discriminator="kind")`` is ``Discriminator("kind")``, and the ``Field``'s
    title and description, which check nothing, are left out.

    A dataclass's or a ``TypedDict``'s field is the hint its author wrote, with its ``Annotated`` metadata, so a
    ``Field`` written there stays as it is. A generic one has its type arguments filled in when they are given:
    ``Row[int]``'s ``value: T`` is ``int``. A subclass of one, like ``class IntRow(Row[int])``, leaves them open,
    as pydantic does. A ``TypedDict``'s ``Required`` and ``ReadOnly`` are taken off. A field a value may leave
    out, written ``NotRequired`` or in a ``TypedDict`` declared ``total=False``, is always given as
    ``NotRequired[...]``, so one test finds it (``types_and_left_out``). A type variable left open becomes its
    bound, or ``Any``.

    Only what the class declares as a field is one: a model's ``model_fields``, a dataclass's ``dataclasses.fields``
    and every key of a ``TypedDict``. So a model's private attribute, a dataclass's ``InitVar`` and a ``ClassVar``
    are left out, and a dataclass's ``field(init=False)``, which every value has, is kept.

    Raises:
        NodeDeclarationError: ``type_not_supported`` for a field written as text that names nothing Python can
            find.
    """
    origin, arguments = get_origin(hint) or hint, get_args(hint)
    if not declares_fields(origin):
        return {}
    if issubclass(origin, BaseModel):
        # pydantic finds a field written as text when the model is first used, so it is asked to do that now, in
        # the namespaces of the model's own module and of where it was defined, never this function's; it
        # returns False when a name can't be found
        if origin.model_rebuild(raise_errors=False, _parent_namespace_depth=0) is False:
            raise NodeDeclarationError(
                declaration_problems.type_not_supported(type=origin.__qualname__)
            )
        # pydantic keeps a field's Annotated metadata apart from its type, and its discriminator apart from both,
        # so they are put back together
        fields = {}
        for name, field in origin.model_fields.items():
            metadata = list(field.metadata)
            if isinstance(field.discriminator, str):
                metadata.append(Discriminator(field.discriminator))
            elif field.discriminator is not None:
                metadata.append(field.discriminator)
            read = Annotated[(field.annotation, *metadata)] if metadata else field.annotation
            fields[name] = substituted(read, {})
        return fields
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
    # a dataclass's annotations hold its InitVars and ClassVars too; its fields say which names are fields
    declared = (
        {f.name for f in dataclasses.fields(origin)}
        if dataclasses.is_dataclass(origin)
        else written.keys()
    )
    fields = {}
    for name, field in written.items():
        if name not in declared:
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


def type_fields(hint: Any) -> dict[str, Any]:
    """Returns the type of each field a pydantic model, a dataclass or a ``TypedDict`` declares, by name.

    It is empty for any other type, such as ``str`` or ``dict[str, int]``. A ``compute_outputs`` hook calls it on a
    type arriving on an input: ``type_fields(Contact)`` is ``{"name": str, "email": str | None}``. Conductor reads a
    type's fields the same way. It is provisional.

    A model's field is what its ``model_fields`` holds, the type pydantic checks it as: a generic model has its
    type arguments filled in, in ``Box[int]`` and in a subclass such as ``class IntBox(Box[int])``, and a
    ``Field(ge=0)`` is given as its check, ``Annotated[int, Ge(ge=0)]``. A dataclass's or a ``TypedDict``'s field
    is the hint as its author wrote it, with its ``Annotated`` limits, and a generic one has its arguments filled
    in when they are given: for ``class Row[T](TypedDict)``, ``Row[int]``'s field ``value: T`` is ``int``. A type
    variable left open, as in a bare ``Row``, becomes its bound, or ``Any``. A field a value may leave out is given
    as its type, without ``NotRequired``.

    Only what the class declares as a field is given. A model's private attribute, like ``_secret``, a
    dataclass's ``InitVar`` and a ``ClassVar`` are left out. A dataclass's ``field(init=False)`` is given, since
    every value has it.

    Raises:
        NodeDeclarationError: ``type_not_supported`` for a field written as text that names nothing Python can
            find.
    """
    return {name: field for name, (field, _) in types_and_left_out(field_hints(hint)).items()}


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
