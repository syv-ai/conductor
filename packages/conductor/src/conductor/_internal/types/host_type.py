"""Names for types: conductor's own, and the ones a host declares with ``@host_type``.

Every type on an input or output needs an id, which an editor and a graph record know it by, and a title, which a
person sees. Conductor names Python's common types, the standard library's classes pydantic checks and pydantic's own
classes itself (``NAMES``): an ``int`` is ``"int"``, titled "Whole number", and a ``pathlib.Path`` is ``"Path"``. A
host names a class of its own with ``@host_type``::

    @host_type("money", title="Money")
    class Money(BaseModel):
        model_config = ConfigDict(frozen=True)
        amount: float

A field typed ``Money`` then has the id ``"money"`` in ``describe()``, and an editor shows "Money". The decorator
checks the declaration right away and stores it on the class, where ``declaration_of`` finds it. A dataclass or a
pydantic model is declared frozen, since a node never changes a value it is given.
"""

from __future__ import annotations

import dataclasses
import datetime
import re
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
from typing import Annotated, Any
from uuid import UUID
from zoneinfo import ZoneInfo

import typing_extensions
from pydantic import (
    AfterValidator,
    AmqpDsn,
    AnyHttpUrl,
    AnyUrl,
    AnyWebsocketUrl,
    BaseModel,
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
from pydantic_core import MultiHostUrl, PydanticCustomError, Url, core_schema

from conductor._internal.problem import (
    NodeDeclarationError,
    arguments_reason,
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
OWN_IDS: set[str] = {own_id for own_id, _ in NAMES.values()} | {"row"}

#: The attribute ``@host_type`` stores a class's ``TypeDeclaration`` in, read only by ``_own_declaration``.
_DECLARATION = "__conductor_type__"

#: The pattern of a type's id: one word of letters, digits, ``_``, ``.`` and ``-``, starting with a letter or ``_``.
_TYPE_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*$")


def _not_blank(text: str) -> str:
    """Returns ``text`` if it has a character other than whitespace, so that pydantic can use it as a check.

    Raises:
        PydanticCustomError: ``blank_text``, for text that is empty or only spaces, tabs, newlines or other
            whitespace.
    """
    if not text.strip():
        raise conductor_error(
            "blank_text", "the text is empty or only whitespace, so a person reads nothing"
        )
    return text


#: A title or description a person reads about a node or a type in a palette, so it isn't empty or only whitespace.
#: ``@node``, ``subgraph`` and ``@host_type`` declare their title with it, and a node its description.
NotBlank = Annotated[str, AfterValidator(_not_blank)]


def _a_type_id(text: str) -> str:
    """Returns ``text`` if a host can declare it as a type's id, so that pydantic can use it as a check.

    Raises:
        PydanticCustomError: ``type_id_not_one_word`` if it isn't one word, since a type's id is written inside other
            ids, as in ``list[money]``. ``conductors_own_type_id`` if it is the id of one of conductor's own types,
            such as ``str``.
    """
    if not _TYPE_ID.match(text):
        raise conductor_error(
            "type_id_not_one_word",
            "'{text}' isn't one word of letters, digits, '_', '.' or '-' that starts with a letter or '_', as a "
            "type's id is, since it is written inside other ids",
            {"text": text},
        )
    if text in OWN_IDS:
        raise conductor_error(
            "conductors_own_type_id",
            "'{text}' is the id of one of conductor's own types",
            {"text": text},
        )
    return text


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

    id: Annotated[str, AfterValidator(_a_type_id)]
    title: NotBlank
    accepts: tuple[type, ...] = ()


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

    A dataclass is declared with ``@dataclass(frozen=True)``, and a pydantic model with
    ``model_config = ConfigDict(frozen=True)``, since a node never changes a value it is given. A node that needs a
    changed value makes one with ``dataclasses.replace`` or ``model_copy(update=...)``.

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
        NodeDeclarationError: Right away, when the class is defined. ``invalid_declaration_argument``, with every
            fault in ``ctx.errors``, if the id isn't one word (``type_id_not_one_word``) or is one of conductor's own
            (``conductors_own_type_id``), the title is blank (``blank_text``), either isn't text, or ``accepts`` has something that isn't a
            class. Then, once the class is given, ``invalid_declaration_argument`` if it is one conductor names
            itself (``conductors_own_type``) or a dataclass or pydantic model that isn't frozen (``type_not_frozen``),
            ``type_declaration_not_on_a_class`` if it isn't a class it can name,
            ``type_declaration_not_writable`` if it can't take a new attribute, and ``host_type_not_supported`` if
            pydantic can't check its values.
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
                reason=arguments_reason(faults),
                errors=faults,
            )
        ) from None

    def declare(cls: C) -> C:
        if not isinstance(cls, type) or typing_extensions.is_typeddict(cls):
            raise NodeDeclarationError(
                declaration_problems.type_declaration_not_on_a_class(id=id, given=repr(cls))
            )
        fault = _class_fault(cls)
        if fault is not None:
            refused = shape_error(fault, ["cls"])
            raise NodeDeclarationError(
                declaration_problems.invalid_declaration_argument(
                    declared_with="@host_type", reason=arguments_reason([refused]), errors=[refused]
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


def _class_fault(cls: type) -> PydanticCustomError | None:
    """Returns why ``@host_type`` can't be put on ``cls`` itself, or ``None`` if it can.

    A class conductor names itself is ``conductors_own_type``. The declaration is stored on the class, so it would
    rename the class for every registry.

    A dataclass that isn't ``frozen=True``, or a pydantic model whose config isn't, is ``type_not_frozen``. Every
    node that reads a value gets the same object, and so does the run that keeps it, so a node never changes a value
    it is given. A frozen class refuses a new value for a field. It is shallow: a list in a field can still change.
    """
    if cls in NAMES or cls in READ_AS:
        return conductor_error(
            "conductors_own_type",
            "{cls} is one of conductor's own types; declare a subclass of it to give it an id of your own",
            {"cls": cls.__qualname__},
        )
    if dataclasses.is_dataclass(cls):
        # every dataclass has it, and a subclass that isn't a dataclass inherits its parent's
        frozen: bool = cls.__dataclass_params__.frozen  # type: ignore[attr-defined]
    elif issubclass(cls, BaseModel):
        frozen = cls.model_config.get("frozen") is True
    else:
        return None
    if frozen:
        return None
    return conductor_error(
        "type_not_frozen",
        "{cls} isn't frozen, so a node could change a value that every node reading it shares; declare it with "
        "@dataclass(frozen=True), or with model_config = ConfigDict(frozen=True) on a pydantic model, and make a "
        "changed value with dataclasses.replace or model_copy(update=...)",
        {"cls": cls.__qualname__},
    )


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
    one of conductor's own gets ``-type`` after it, so a class ``Row`` gets ``"row-type"``.
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
