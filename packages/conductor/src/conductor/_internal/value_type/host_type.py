"""``@host_type``: how a host declares a class of its own as a type conductor can name.

Example::

    @host_type("money", title="Money")
    class Money(BaseModel):
        amount: float

A field typed ``Money`` then has the type id ``"money"`` in ``describe()``, and an editor shows "Money".
The decorator checks the declaration when the class statement ends and stores it on the class as
``__conductor_type__``. ``Type`` reads it from there with ``declaration_of``.
"""

from __future__ import annotations

import datetime
import decimal
import uuid
from collections.abc import Callable, Iterable, Mapping
from typing import Annotated, Any

from pydantic import Field as PydanticField
from pydantic import TypeAdapter, ValidationError, field_validator
from pydantic.dataclasses import dataclass
from pydantic.errors import PydanticSchemaGenerationError
from pydantic_core import core_schema

from conductor._internal.problem import NodeDeclarationError

#: The id and title of each class conductor itself names. ``Series`` and ``Table`` name themselves in their own
#: packages, and every other class a field has is declared with ``@host_type``.
NAMES: Mapping[Any, tuple[str, str]] = {
    Any: ("Any", "Any value"), object: ("object", "Any value"), type(None): ("None", "Nothing"), bool: ("bool", "Yes or no"),
    int: ("int", "Whole number"), float: ("float", "Number"), decimal.Decimal: ("Decimal", "Decimal number"), str: ("str", "Text"),
    bytes: ("bytes", "Bytes"), datetime.date: ("date", "Date"), datetime.datetime: ("datetime", "Date and time"),
    datetime.time: ("time", "Time of day"), datetime.timedelta: ("timedelta", "Duration"), uuid.UUID: ("UUID", "UUID"),
    dict: ("dict", "Mapping"), list: ("list", "List"), tuple: ("tuple", "Tuple"), set: ("set", "Set"), frozenset: ("frozenset", "Set"),
}

#: The ids a host can't declare, as they are conductor's own. ``Type`` adds the id of each generic class of its own
#: (``Series``, ``Table``) when that class's package is imported.
OWN_IDS: set[str] = {own_id for own_id, _ in NAMES.values()} | {"record"}


@dataclass(frozen=True, config={"arbitrary_types_allowed": True, "strict": True})
class TypeDeclaration:
    """What ``@host_type`` stores on a class as ``__conductor_type__``. It holds only these three values; a type
    that needs behaviour of its own is a subclass of ``Type``, as ``Series`` and ``Table`` have.

    Attributes:
        id: The name an editor and a stored graph know the type by, such as ``"money"``. A word with none of
            ``[ ] | , :`` or a space, since the id of ``list[Money]`` is written ``list[money]``.
        title: What a person reads for the type, such as ``"Money"``.
        accepts: Other classes whose values an edge may pass to an input of this type.
    """

    id: Annotated[str, PydanticField(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$")]
    title: Annotated[str, PydanticField(min_length=1)]
    accepts: tuple[type, ...] = ()

    @field_validator("id")
    @classmethod
    def _not_one_of_conductors_own(cls, given: str) -> str:
        if given in OWN_IDS:
            raise ValueError(f"{given!r} is the id of one of conductor's own types")
        return given


def host_type[C: type](id: str, *, title: str, accepts: Iterable[type] = ()) -> Callable[[C], C]:  # noqa: A002  the word a type's id has in describe()
    """Declares a class as a type of the host's. Write it above the class, and above ``@dataclass`` when there is
    one::

        @host_type("text", title="Text")
        class Text(str): ...

        host_type("ip", title="IP address")(ipaddress.IPv4Address)   # a library's class

    A subclass of ``str``, ``int`` or another class conductor names needs nothing more: a value is validated as
    the parent class and then made the subclass. A subclass does not inherit the declaration; declare it too.

    Args:
        id: The name an editor and a stored graph know the type by.
        title: What a person reads for the type.
        accepts: Other classes whose values an edge may pass to an input of this type.

    Returns:
        A decorator that returns the class it is given, unchanged but for the declaration stored on it.

    Raises:
        NodeDeclarationError: When the class statement ends: the id is not a word or is one of conductor's own,
            the title is empty, ``accepts`` has something that is not a class, or the class can't be written on
            (``type_declaration_invalid``); or pydantic can't validate the class (``type_not_carried``).
    """
    try:
        declaration = TypeDeclaration(id=id, title=title, accepts=tuple(accepts))
    except ValidationError as error:
        said = "; ".join(f"{fault['loc'][0]}: {fault['msg']}" for fault in error.errors())
        raise NodeDeclarationError(
            f"@host_type({id!r}): {said}; give an id that is one word and not one of conductor's own, a title, and classes in accepts",
            code="type_declaration_invalid",
        ) from None

    def declare(cls: C) -> C:
        try:
            TypeAdapter(cls)
        except PydanticSchemaGenerationError:
            parent = next((base for base in cls.__mro__[1:-1] if base in NAMES), None)
            if parent is None:
                raise NodeDeclarationError(
                    f"@host_type({id!r}) on {cls.__qualname__}: pydantic can't validate it or write it as JSON; make it a pydantic model or a "
                    "dataclass (with @host_type above @dataclass), or give it its own __get_pydantic_core_schema__",
                    code="type_not_carried",
                ) from None
            validated_as_parent = classmethod(lambda made, source, handler: core_schema.no_info_after_validator_function(made, handler(parent)))
            cls.__get_pydantic_core_schema__ = validated_as_parent  # type: ignore[attr-defined]  # the hook pydantic looks for on a class
        try:
            cls.__conductor_type__ = declaration  # type: ignore[attr-defined]  # read back by declaration_of
        except TypeError:  # a class written in C, such as int
            raise NodeDeclarationError(
                f"@host_type({id!r}) on {cls.__qualname__}: Python lets nobody write on this class; put its value in a class of your own and declare that",
                code="type_declaration_invalid",
            ) from None
        return cls

    return declare


def declaration_of(cls: type) -> TypeDeclaration:
    """Returns what ``@host_type`` stored on ``cls`` itself. A parent's declaration does not count.

    Raises:
        NodeDeclarationError: ``cls`` has none (``type_not_named``).
    """
    declared: TypeDeclaration | None = vars(cls).get("__conductor_type__")
    if declared is None:
        parent = next((base for base in cls.__mro__[1:] if "__conductor_type__" in vars(base)), None)
        why = f"; {parent.__qualname__}'s declaration is its own and is not inherited" if parent else ""
        raise NodeDeclarationError(
            f'{cls.__module__}.{cls.__qualname__} is the type of a field and is not declared; write @host_type("...", title="...") above the class{why}',
            code="type_not_named",
        )
    return declared


def accepted_classes(cls: type) -> tuple[type, ...]:
    """Returns the other classes whose values an edge may pass to an input of the class ``cls``: what its
    ``@host_type(accepts=...)`` lists, and none for a class with no declaration, such as ``int``."""
    declared: TypeDeclaration | None = vars(cls).get("__conductor_type__")
    return () if declared is None else declared.accepts
