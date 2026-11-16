"""Names for types: conductor's own, and the ones a host declares with ``@host_type``.

Every type on an input or output needs an id, which an editor and a graph record know it by, and a title, which a
person sees. Conductor names the common Python types itself (``NAMES``): an ``int`` is ``"int"``, titled "Whole
number". A host names a class of its own with ``@host_type``::

    @host_type("money", title="Money")
    class Money(BaseModel):
        amount: float

A field typed ``Money`` then has the id ``"money"`` in ``describe()``, and an editor shows "Money". The decorator
checks the declaration right away and stores it on the class, where ``declaration_of`` finds it.
"""

from __future__ import annotations

import datetime
import re
import typing
from collections.abc import Callable, Mapping
from decimal import Decimal
from typing import Annotated, Any
from uuid import UUID

from pydantic import Field as PydanticField
from pydantic import TypeAdapter, ValidationError, field_validator
from pydantic.dataclasses import dataclass
from pydantic.errors import PydanticSchemaGenerationError
from pydantic_core import core_schema

from conductor._internal.problem import NodeDeclarationError, declaration_problems
from conductor._internal.utils import qualified_name

#: The id and title of each class conductor names itself. ``Series`` and ``Table`` are named in their own
#: packages. Any other class on a field must be declared with ``@host_type``.
NAMES: Mapping[Any, tuple[str, str]] = {
    Any: ("Any", "Any value"),
    object: ("object", "Any value"),
    type(None): ("None", "Nothing"),
    bool: ("bool", "Yes or no"),
    int: ("int", "Whole number"),
    float: ("float", "Number"),
    Decimal: ("Decimal", "Decimal number"),
    str: ("str", "Text"),
    bytes: ("bytes", "Bytes"),
    datetime.date: ("date", "Date"),
    datetime.datetime: ("datetime", "Date and time"),
    datetime.time: ("time", "Time of day"),
    datetime.timedelta: ("timedelta", "Duration"),
    UUID: ("UUID", "UUID"),
    dict: ("dict", "Mapping"),
    list: ("list", "List"),
    tuple: ("tuple", "Tuple"),
    set: ("set", "Set"),
    frozenset: ("frozenset", "Set"),
}

#: The ids a host can't declare, because conductor uses them. ``Type`` adds ``Series`` and ``Table`` here when
#: their packages are imported.
OWN_IDS: set[str] = {own_id for own_id, _ in NAMES.values()} | {"record"}


@dataclass(frozen=True, config={"arbitrary_types_allowed": True, "strict": True})
class TypeDeclaration:
    """What ``@host_type`` stores on a class, as ``__conductor_type__``.

    It is only these three values. A type that needs behaviour of its own is a subclass of ``Type`` instead, like
    the ones for ``Series`` and ``Table``.

    Attributes:
        id: The name an editor and a graph record know the type by, such as ``"money"``. It is one word, with no
            ``[ ] | , :`` or space, since it is written inside other ids: ``list[Money]`` is ``list[money]``.
        title: What a person sees for the type, such as ``"Money"``.
        accepts: Other classes whose values an edge may pass to an input of this type.
    """

    id: Annotated[str, PydanticField(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$")]
    title: Annotated[str, PydanticField(pattern=r"\S")]
    accepts: tuple[type, ...] = ()

    @field_validator("id")
    @classmethod
    def _not_one_of_conductors_own(cls, given: str) -> str:
        if given in OWN_IDS:
            raise ValueError(f"{given!r} is the id of one of conductor's own types")
        return given


def host_type[C: type](
    id: str, *, title: str, accepts: list[type] | tuple[type, ...] = ()
) -> Callable[[C], C]:  # noqa: A002  "id" is what describe() calls it
    """Gives a class of your own the id and title conductor shows it by.

    Write it above the class, and above ``@dataclass`` if there is one. For a class from a library, call it
    directly::

        @host_type("text", title="Text")
        class Text(str): ...

        host_type("ip", title="IP address")(ipaddress.IPv4Address)   # a library's class

    A subclass of ``str``, ``int`` or another class conductor names needs nothing more. A value is checked as the
    parent class, then made into the subclass. A subclass doesn't inherit its parent's declaration, so declare it
    too.

    Args:
        id: The name an editor and a graph record know the type by.
        title: What a person sees for the type.
        accepts: Other classes whose values an edge may pass to an input of this type.

    Returns:
        A decorator that stores the declaration on the class and returns the class.

    Raises:
        NodeDeclarationError: Right away, when the class is defined. ``type_declaration_invalid`` if the id isn't
            one word or is one of conductor's own, the title is empty, ``accepts`` has something that isn't a
            class, or the class can't take a new attribute. ``type_not_supported`` if pydantic can't check values of
            the class.
    """
    try:
        stored = TypeDeclaration(
            id=id, title=title, accepts=tuple(accepts) if isinstance(accepts, list) else accepts
        )
    except ValidationError as error:
        faults = "; ".join(f"{fault['loc'][0]}: {fault['msg']}" for fault in error.errors())
        raise NodeDeclarationError(
            declaration_problems.type_declaration_invalid(id=id, faults=faults)
        ) from None

    def declare(cls: C) -> C:
        if not isinstance(cls, type) or typing.is_typeddict(cls):
            raise NodeDeclarationError(
                declaration_problems.type_declaration_not_on_a_class(id=id, given=repr(cls))
            )
        try:
            TypeAdapter(cls)
        except PydanticSchemaGenerationError:
            parent = next((base for base in cls.__mro__[1:-1] if base in NAMES), None)
            if parent is None:
                raise NodeDeclarationError(
                    declaration_problems.type_not_supported_declared(id=id, cls=cls.__qualname__)
                ) from None
            validated_as_parent = classmethod(
                lambda made, source, handler: core_schema.no_info_after_validator_function(
                    made, handler(parent)
                )
            )
            cls.__get_pydantic_core_schema__ = validated_as_parent  # type: ignore[attr-defined]  # pydantic's hook for a class's schema
        try:
            cls.__conductor_type__ = stored  # type: ignore[attr-defined]  # declaration_of looks here
        except TypeError:  # a class written in C, such as int
            raise NodeDeclarationError(
                declaration_problems.type_declaration_not_writable(id=id, cls=cls.__qualname__)
            ) from None
        return cls

    return declare


def declaration_of(cls: type) -> TypeDeclaration:
    """Returns what ``@host_type`` stored on ``cls``.

    For a ``Money`` declared with ``@host_type("money", title="Money")``, ``declaration_of(Money).id`` is
    ``"money"``. Only a declaration on ``cls`` itself counts, not one on a parent class.

    Raises:
        NodeDeclarationError: ``type_not_named`` if ``cls`` has no declaration of its own.
    """
    declared: TypeDeclaration | None = vars(cls).get("__conductor_type__")
    if declared is None:
        parent = next(
            (base for base in cls.__mro__[1:] if "__conductor_type__" in vars(base)), None
        )
        suggested_id, suggested_title = _suggested_declaration(cls)
        raise NodeDeclarationError(
            declaration_problems.type_not_named(
                cls=qualified_name(cls),
                suggested_id=suggested_id,
                suggested_title=suggested_title,
                parent=parent.__qualname__ if parent else None,
            )
        )
    return declared


def _suggested_declaration(cls: type) -> tuple[str, str]:
    """Returns an id and a title for a class with no ``@host_type``, made from the class's name.

    ``type_not_named`` suggests them, so the developer can paste a declaration that works. The id is the name's
    words in lower case with ``-`` between them, and the title is the words as a phrase. ``InvoiceLine`` gets
    ``"invoice-line"`` and "Invoice line", and ``IPAddress`` gets ``"ip-address"`` and "IP address". An id that is
    one of conductor's own gets ``-type`` after it, so a class ``Record`` gets ``"record-type"``.
    """
    words = re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+", cls.__name__) or [cls.__name__]
    suggested_id = "-".join(word.lower() for word in words)
    if suggested_id in OWN_IDS:
        suggested_id = f"{suggested_id}-type"
    first, *rest = words
    title_words = [first[:1].upper() + first[1:]]
    # An acronym such as "IP" keeps its capitals, and any other word is written in lower case.
    title_words += [word if word.isupper() else word.lower() for word in rest]
    return suggested_id, " ".join(title_words)


def accepted_classes(cls: type) -> tuple[type, ...]:
    """Returns the other classes whose values an edge may pass to an input of type ``cls``.

    These are the classes ``@host_type(accepts=...)`` lists. For a ``Temperature`` declared with
    ``accepts=[Celsius]`` it is ``(Celsius,)``. For a class with no declaration, like ``int``, it is ``()``.
    """
    declared: TypeDeclaration | None = vars(cls).get("__conductor_type__")
    return () if declared is None else declared.accepts
