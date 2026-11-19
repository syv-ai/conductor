"""Names for types: conductor's own, and the ones a host declares with ``@host_type``.

Every type on an input or output needs an id, which an editor and a graph record know it by, and a title, which a
person sees. Conductor names Python's common types, the standard library's classes pydantic checks and pydantic's own
classes itself (``NAMES``): an ``int`` is ``"int"``, titled "Whole number", and a ``pathlib.Path`` is ``"Path"``. A
host names a class of its own with ``@host_type``::

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
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from decimal import Decimal
from fractions import Fraction
from ipaddress import (
    IPv4Address,
    IPv4Interface,
    IPv4Network,
    IPv6Address,
    IPv6Interface,
    IPv6Network,
)
from pathlib import Path
from re import Pattern
from typing import Any, Literal
from uuid import UUID
from zoneinfo import ZoneInfo

from pydantic import (
    AmqpDsn,
    AnyHttpUrl,
    AnyUrl,
    AnyWebsocketUrl,
    ByteSize,
    ClickHouseDsn,
    CockroachDsn,
    FileUrl,
    FtpUrl,
    HttpUrl,
    KafkaDsn,
    MariaDBDsn,
    MongoDsn,
    MySQLDsn,
    NameEmail,
    NatsDsn,
    PostgresDsn,
    RedisDsn,
    SnowflakeDsn,
    TypeAdapter,
    ValidationError,
    WebsocketUrl,
)
from pydantic.dataclasses import dataclass
from pydantic.errors import PydanticSchemaGenerationError
from pydantic_core import MultiHostUrl, Url, core_schema

from conductor._internal.problem import (
    NodeDeclarationError,
    conductor_error,
    declaration_problems,
    shape_error,
)
from conductor._internal.utils import qualified_name

from .hint import READ_AS

#: The id and title of each of Python's own classes that conductor names: the builtins, and the standard library's
#: classes pydantic checks. Each id is the class's name as a host imports it. Every value of one of these is a value of
#: its type, however it was made, so a validator keeps a value of exactly one of them as it is
#: (``Type.exact_classes``).
PYTHON_NAMES: Mapping[Any, tuple[str, str]] = {
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
    complex: ("complex", "Complex number"),
    Fraction: ("Fraction", "Fraction"),
    Path: ("Path", "File path"),
    Pattern: ("Pattern", "Regular expression"),
    deque: ("deque", "Queue"),
    Sequence: ("Sequence", "Sequence"),
    Mapping: ("Mapping", "Mapping"),
    ZoneInfo: ("ZoneInfo", "Time zone"),
    IPv4Address: ("IPv4Address", "IPv4 address"),
    IPv6Address: ("IPv6Address", "IPv6 address"),
    IPv4Interface: ("IPv4Interface", "IPv4 interface"),
    IPv6Interface: ("IPv6Interface", "IPv6 interface"),
    IPv4Network: ("IPv4Network", "IPv4 network"),
    IPv6Network: ("IPv6Network", "IPv6 network"),
}

#: The id and title of each class conductor names itself: Python's own (``PYTHON_NAMES``) and pydantic's. A value of
#: one of pydantic's may be made that its own check refuses, like ``ByteSize(-5)``, so it is always checked.
#: ``Series`` and ``Table`` are named in their own packages, and a class whose values are of another class, such as
#: pydantic's ``EmailStr``, is read as that class (``READ_AS``). Any other class on a field must be declared with
#: ``@host_type``.
NAMES: Mapping[Any, tuple[str, str]] = {
    **PYTHON_NAMES,
    Url: ("Url", "URL"),
    MultiHostUrl: ("MultiHostUrl", "URL with several hosts"),
    AnyUrl: ("AnyUrl", "URL"),
    AnyHttpUrl: ("AnyHttpUrl", "HTTP URL"),
    HttpUrl: ("HttpUrl", "Web address"),
    FileUrl: ("FileUrl", "File URL"),
    FtpUrl: ("FtpUrl", "FTP URL"),
    WebsocketUrl: ("WebsocketUrl", "WebSocket URL"),
    AnyWebsocketUrl: ("AnyWebsocketUrl", "WebSocket URL"),
    PostgresDsn: ("PostgresDsn", "PostgreSQL URL"),
    CockroachDsn: ("CockroachDsn", "CockroachDB URL"),
    AmqpDsn: ("AmqpDsn", "AMQP URL"),
    RedisDsn: ("RedisDsn", "Redis URL"),
    MongoDsn: ("MongoDsn", "MongoDB URL"),
    KafkaDsn: ("KafkaDsn", "Kafka URL"),
    NatsDsn: ("NatsDsn", "NATS URL"),
    MySQLDsn: ("MySQLDsn", "MySQL URL"),
    MariaDBDsn: ("MariaDBDsn", "MariaDB URL"),
    ClickHouseDsn: ("ClickHouseDsn", "ClickHouse URL"),
    SnowflakeDsn: ("SnowflakeDsn", "Snowflake URL"),
    NameEmail: ("NameEmail", "Name and email address"),
    ByteSize: ("ByteSize", "Size in bytes"),
}

#: The ids a host can't declare, because conductor uses them. ``Type`` adds ``Series`` and ``Table`` here when
#: their packages are imported.
OWN_IDS: set[str] = {own_id for own_id, _ in NAMES.values()} | {"record"}

#: The attribute ``@host_type`` stores a class's ``TypeDeclaration`` in, read only by ``_own_declaration``.
_DECLARATION = "__conductor_type__"


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

    id: str
    title: str
    accepts: tuple[type, ...] = ()


#: The pattern of a type's id: one word of letters, digits, ``_``, ``.`` and ``-``, starting with a letter or ``_``.
_TYPE_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def check_not_blank(
    declared_with: str,
    id: str,
    shown: Mapping[Literal["title", "description"], str],
    *place: str,
    version: int | None = None,
) -> None:
    """Checks that the title, and a node's description, are not blank: the one rule for the text a person reads about a
    node or a type, which ``@node``, ``subgraph`` and ``@host_type`` each call.

    Args:
        declared_with: ``"@node"``, ``"subgraph"`` or ``"@host_type"``.
        id: The node's or the type's id.
        shown: The texts, by which one each is.
        place: Where the fault is placed, as ``NodeDeclarationError`` takes it: the node's id, or nothing for a type.
        version: The node's version, as ``NodeDeclarationError`` takes it.

    Raises:
        NodeDeclarationError: ``empty_title_or_description``, for the first text that is empty or only spaces.
    """
    for empty, text in shown.items():
        if not text.strip():
            raise NodeDeclarationError(
                declaration_problems.empty_title_or_description(
                    declared_with=declared_with, id=id, empty=empty
                ),
                *place,
                version=version,
            )


def host_type[C: type](
    id: str, *, title: str, accepts: list[type] | tuple[type, ...] = ()
) -> Callable[[C], C]:  # noqa: A002  "id" is what describe() calls it
    """Gives a class of your own the id and title conductor shows it by.

    Write it above the class, and above ``@dataclass`` if there is one. For a class from a library, call it
    directly::

        @host_type("text", title="Text")
        class Text(str): ...

        host_type("http-status", title="HTTP status")(http.HTTPStatus)   # a library's class

    A class conductor names itself, such as ``HttpUrl`` or ``EmailStr``, can't be declared, since the declaration is
    stored on the class and would rename it for every registry. Declare a subclass of it instead, which is a type of
    your own.

    A subclass of ``str``, ``int`` or another class conductor names needs nothing more. A value is checked as the
    parent class, then made into the subclass. A subclass doesn't inherit its parent's declaration, so declare it
    too.

    A declared class is one value. A node that returns a declared dataclass has one output, ``result``, where a
    dataclass with no declaration would give one output per field.

    Args:
        id: The name an editor and a graph record know the type by.
        title: What a person sees for the type.
        accepts: Other classes whose values an edge may pass to an input of this type.

    Returns:
        A decorator that stores the declaration on the class and returns the class.

    Raises:
        NodeDeclarationError: Right away, when the class is defined. ``invalid_declaration_argument`` if the id or
            the title isn't text, ``accepts`` has something that isn't a class, or the class is one conductor names
            itself. ``invalid_node_id`` if the id isn't one word or is one of conductor's own, and
            ``empty_title_or_description`` if the title is blank. ``type_declaration_not_on_a_class`` if it isn't put
            above a class it can name, ``type_declaration_not_writable`` if the class can't take a new attribute, and
            ``host_type_not_supported`` if pydantic can't check values of the class.
    """
    try:
        stored = TypeDeclaration(
            id=id, title=title, accepts=tuple(accepts) if isinstance(accepts, list) else accepts
        )
    except ValidationError as error:
        faults = [shape_error(fault, fault["loc"]) for fault in error.errors()]
        raise NodeDeclarationError(
            declaration_problems.invalid_declaration_argument(
                declared_with="@host_type",
                reason=f"{faults[0]['loc'][0]}: {faults[0]['msg']}",
                errors=faults,
            )
        ) from None
    if not _TYPE_ID.match(id) or id in OWN_IDS:
        raise NodeDeclarationError(
            declaration_problems.invalid_node_id(declared_with="@host_type", id=id)
        )
    check_not_blank("@host_type", id, {"title": title})

    def declare(cls: C) -> C:
        if not isinstance(cls, type) or typing.is_typeddict(cls):
            raise NodeDeclarationError(
                declaration_problems.type_declaration_not_on_a_class(id=id, given=repr(cls))
            )
        # a declaration is stored on the class, so on one of conductor's own it would rename it for every registry
        if cls in NAMES or cls in READ_AS:
            own = shape_error(
                conductor_error(
                    "conductors_own_type",
                    "{cls} is one of conductor's own types; declare a subclass of it to give it an id of your own",
                    {"cls": cls.__qualname__},
                ),
                ["cls"],
            )
            raise NodeDeclarationError(
                declaration_problems.invalid_declaration_argument(
                    declared_with="@host_type", reason=f"cls: {own['msg']}", errors=[own]
                )
            )
        try:
            TypeAdapter(cls)
        except PydanticSchemaGenerationError:
            parent = next((base for base in cls.__mro__[1:-1] if base in NAMES), None)
            if parent is None:
                raise NodeDeclarationError(
                    declaration_problems.host_type_not_supported(id=id, cls=cls.__qualname__)
                ) from None
            validated_as_parent = classmethod(
                lambda made, source, handler: core_schema.no_info_after_validator_function(
                    made, handler(parent)
                )
            )
            cls.__get_pydantic_core_schema__ = validated_as_parent  # type: ignore[attr-defined]  # pydantic's hook for a class's schema
        try:
            setattr(cls, _DECLARATION, stored)  # _own_declaration reads it
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
    declared = _own_declaration(cls)
    if declared is None:
        parent = next((base for base in cls.__mro__[1:] if _own_declaration(base)), None)
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


def is_declared(cls: type) -> bool:
    """Whether ``@host_type`` declared ``cls`` or any parent class of it.

    ``@node`` asks it of a returned dataclass: one with no declaration anywhere lists the node's outputs, and any
    other is one value. So an undeclared subclass of a declared class is a type, and ``declaration_of`` refuses it
    as ``type_not_named``, on a return as on an input.
    """
    return any(_own_declaration(base) for base in cls.__mro__)


def _own_declaration(cls: type) -> TypeDeclaration | None:
    """Returns what ``@host_type`` stored on ``cls`` itself, or ``None``: a parent's declaration doesn't count."""
    declared: TypeDeclaration | None = vars(cls).get(_DECLARATION)
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
    declared = _own_declaration(cls)
    return () if declared is None else declared.accepts
