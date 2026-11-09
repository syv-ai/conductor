"""``DType`` — a value's type, as the engine sees it.

A ``DType`` is a real Python class, usually built on a builtin::

    class Text(DType, str):
        id = "text"

    class Number(DType, float):
        id = "number"

so ``Text("hello")`` is both a ``str`` and a ``Text``, a type checker sees
``Text`` where ``Text`` is meant, and a pydantic model with a ``Text``
field gives back a ``Text`` (see ``__get_pydantic_core_schema__``). ``id``
is the stable name the persisted graph and the frontend use; ``display_title``
is what a person reads and defaults to the class name.

A type answers one question about edges — ``target.accepts(source)``:
may a value of type ``source`` land on an input declared as ``target``?
The default is ``issubclass``, so a subtype is accepted wherever its
parent is. It is asked about one value of each side::

    class Integer(Number):
        id = "integer"

    Number.accepts(Integer)   # True
    Integer.accepts(Number)   # False

Three things a ``DType`` deliberately does not do:

* **Convert.** A value arrives at a node as the type the edge carried.
  Where a conversion seems needed, the answer is a subtype, a node that
  does the work, or an input declared with the widest type the node
  handles.
* **Pick a widget.** The same ``Text`` may be a textarea, a single line or
  a dropdown; every input declares its own widget.
* **Format itself beyond text.** ``as_text`` is the one rendering hook:
  override it when a value should read differently from ``str(value)``.

``Any`` is what an author writes for "whatever arrives"; a record
holds a private ``DType`` in its place, so every field's type is a
``DType``. ``AsItArrives`` marks an open input received whole,
``**inputs: AsItArrives``. A plain Python type (``str``, ``int``, ``float``,
``bool``, ``date``, a pydantic model) is a ``DType`` the same way, in one
place, ``dtype_of``: nothing past it knows a plain type from one written with
``DType``. A value of such a type stays plain. So ``class Text(DType, str)``
is for a type of your own, one that restricts which edges land on it, or
carries a title for an editor; a ``str`` is what a node otherwise writes. A
type no edge can carry (``list[Branch]``) closes its input.

``Series[X]`` is not a type of its own: it says a field holds many values of
``X``, and the field records ``X`` and that it is many (``read_field_annotation``).

Conductor defines no concrete ``DType``. Which types
exist is the host application's decision, made on its ``NodeRegistry``:
the types its nodes declare plus the ``types`` a host names are that
registry's types, and nothing is recorded process-wide. Declaring a
type is therefore free of side effects — a notebook cell can declare the
same ``Text`` twice — and one id claimed by two classes raises where
the second reaches a registry, never at class definition.
"""

from __future__ import annotations

import datetime
from abc import ABC, ABCMeta
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from types import NoneType, UnionType
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Union, get_args, get_origin

from pydantic import BaseModel, PlainSerializer, PlainValidator, WithJsonSchema
from pydantic_core import core_schema

from conductor._internal.errors import NodeDeclarationError

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler

class _DTypeMeta(ABCMeta):
    """A type prints as its name, ``Text``, the way an author writes it."""

    def __repr__(cls) -> str:
        return cls.__name__


class DType(ABC, metaclass=_DTypeMeta):
    """Base class for every type a value on an edge can have.

    Subclass it together with the builtin the type is built on and declare
    ``id`` in the class body; ``display_title`` defaults to the class name. The
    base itself has no ``id`` and is never on an edge.

    A subclass is a type of its own and names itself: a class that leaves
    ``id`` to its parent raises at definition, since a type that quietly
    travels under its parent's ``number`` is the mistake nobody sees.
    """

    #: Stable identifier, used by the persisted graph and the frontend.
    id: ClassVar[str]
    #: The name a person reads; the class name unless the class says otherwise.
    display_title: ClassVar[str]
    #: The Polars type a table column of these values is stored as, by name
    #: (``"Boolean"``). ``None`` reads it off the builtin the type is built
    #: on (``conductor._internal.table.polars_type_of``); a type sets it only where that
    #: base would be wrong, as a flag built on ``int`` does.
    polars_column_type: ClassVar[str | None] = None
    #: The type this one carries a detail of (``Table[Contact]`` of ``Table``),
    #: and shares its id with; ``None`` for a type of its own.
    _generic_base: ClassVar[type["DType"] | None] = None

    def __init_subclass__(cls, *, _generic_base: type["DType"] | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        cls._generic_base = _generic_base
        if _generic_base is not None:
            return
        if "id" not in vars(cls):
            raise NodeDeclarationError(
                f"{cls.__name__} declares no 'id' of its own. A subclass is a type of its own and names itself",
                code="type_without_id",
            )
        if "display_title" not in vars(cls):
            cls.display_title = cls.__name__

    # -- the description --------------------------------------------------

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """This type as a JSON-ready record, for the frontend: ``{"id": ...}``.

        An object rather than a string, so nothing downstream parses a
        type. Where a
        value of this type may land — every type whose ``accepts`` admits
        it — is a question about a set of types, so it is the registry's
        answer (``NodeRegistry.accepted_as``), served once per type in
        ``NodeRegistry.describe()`` rather than on every field's record.
        """
        return {"id": cls.id}

    # -- text for a person --------------------------------------------------

    @classmethod
    def as_text(cls, value: Any) -> str:
        """``value`` rendered as text for a person.

        Every place a value becomes user-facing text goes through this, so
        a type whose values should not read as ``str(value)`` (a date, a
        number with a locale) overrides it once, here.
        """
        return str(value)

    # -- the one question ---------------------------------------------------

    @classmethod
    def accepts(cls, source: Any) -> bool:
        """May a value of type ``source`` land on an input declared as this type?

        The target decides, because an input is where a node states what
        it needs. Default: ``issubclass(source, cls)``. Both sides are one
        value's type. Bare ``DType`` as a target raises: it would accept anything.
        """
        if cls is DType:
            raise TypeError(
                "an input must declare a concrete DType, not the base — "
                "bare DType as a target would accept anything"
            )
        return issubclass(source, cls)

    # -- pydantic ------------------------------------------------------------

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: "GetCoreSchemaHandler"
    ) -> core_schema.CoreSchema:
        """Make pydantic validate a ``Text`` field into a ``Text``, not a bare ``str``.

        pydantic validates a subclass of ``str`` as plain ``str`` unless
        told otherwise, which would make every ``isinstance(value, Text)``
        downstream false. This validates as the builtin the type is built
        on and wraps the result in the subclass. A type built on no builtin
        is validated by ``isinstance``.
        """
        from conductor._internal.codec import float_as_json

        builtin = cls._builtin_base()
        if builtin is None:
            return core_schema.is_instance_schema(cls)
        return core_schema.no_info_after_validator_function(
            lambda value: value if isinstance(value, cls) else cls(value),
            handler.generate_schema(builtin),
            serialization=core_schema.plain_serializer_function_ser_schema(float_as_json, when_used="json") if issubclass(builtin, float) else None,
        )

    # -- internals -----------------------------------------------------------

    @classmethod
    def _instance_type(cls) -> type:
        """The class a value of this type is an instance of."""
        return cls

    @classmethod
    def _builtin_base(cls) -> type | None:
        """The ``str`` / ``float`` / ``date`` this type is built on, or ``None``.

        Walks the MRO and skips ``DType`` classes, ``ABC``, ``object`` and
        the ``abc`` / ``collections.abc`` / ``typing`` helpers, so a type
        built on an unusual base (``pathlib.Path``, ``decimal.Decimal``)
        still works.
        """
        for base in cls.__mro__[1:]:
            if issubclass(base, DType) or base in (ABC, object):
                continue
            if base.__module__ in ("abc", "collections.abc", "typing"):
                continue
            return base
        return None


class _UntypedValue(DType):
    """What a record holds where an author wrote ``Any``: whatever arrives,
    until compile types it from the edge. A host asking "is this untyped?"
    reads ``dtype.id == "any"``, the id on the wire."""

    id = "any"
    display_title = "Any"

    @classmethod
    def accepts(cls, source: Any) -> bool:
        return True

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: "GetCoreSchemaHandler") -> core_schema.CoreSchema:
        return core_schema.any_schema()


class _ReceivedAsItArrives(_UntypedValue):
    """The type of an input made from an edge into ``**inputs: AsItArrives``: never run per row, a column received as a column."""

    id = "as_it_arrives"


@dataclass(frozen=True)
class _AsItArrivesMark:
    """What makes ``AsItArrives`` mean more than ``Any``."""


#: Written on an open input, ``**inputs: AsItArrives``: every edge into the
#: node is an input named on the edge, typed by what arrives, and received
#: as it is, so a column arrives as a column and the node never runs per row.
AsItArrives = Annotated[Any, _AsItArrivesMark()]


class _PlainPythonType(DType):
    """A Python type that is not a ``DType``, as one: ``str``, a pydantic model, a list of branches.

    Its values are instances of the plain type, unwrapped, and its schema is
    the plain type's. ``plain_python_dtype`` builds one per type, and an author never
    sees it. One an edge can carry is named after the type (``{"id": "str"}``)
    and is a type a registry holds; any other is a setting, set
    on a closed input and never on an edge (``{"id": "setting", "of": "list"}``)."""

    id = "setting"
    #: The Python type this one stands for.
    wrapped_type: ClassVar[Any] = None
    travels_on_edges: ClassVar[bool] = False

    def __new__(cls, value: Any) -> Any:
        """A value of a plain type stays plain: making "one of these" from a cell of its own type is the cell."""
        return value

    @classmethod
    def describe(cls) -> dict[str, Any]:
        return {"id": cls.id} if cls.travels_on_edges else {"id": cls.id, "of": getattr(cls.wrapped_type, "__name__", str(cls.wrapped_type))}

    @classmethod
    def accepts(cls, source: Any) -> bool:
        """A value of its own type, or a subtype of it, ``DType`` or plain. An ``int`` lands on a ``float``, as a type checker says."""
        carried = source._builtin_base()
        return carried is not None and issubclass(carried, (float, int) if cls.wrapped_type is float else cls.wrapped_type)

    @classmethod
    def _instance_type(cls) -> type:
        return cls.wrapped_type

    @classmethod
    def _builtin_base(cls) -> type | None:
        return cls.wrapped_type if isinstance(cls.wrapped_type, type) else None

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: "GetCoreSchemaHandler") -> core_schema.CoreSchema:
        return handler.generate_schema(cls.wrapped_type)


def _an_edge_can_carry(python_type: Any) -> bool:
    return isinstance(python_type, type) and issubclass(python_type, (str, int, float, bool, datetime.date, BaseModel))


#: One per plain type, built once.
_PLAIN_PYTHON_DTYPES: dict[Any, type[_PlainPythonType]] = {}


def plain_python_dtype(python_type: Any) -> type[_PlainPythonType]:
    """The ``DType`` a plain Python type is."""
    dtype = _PLAIN_PYTHON_DTYPES.get(python_type)
    if dtype is not None:
        return dtype
    name = getattr(python_type, "__name__", str(python_type))
    if _an_edge_can_carry(python_type):
        dtype = type(name, (_PlainPythonType,), {"id": name, "display_title": name, "wrapped_type": python_type, "travels_on_edges": True})
    else:
        dtype = type(f"Setting[{name}]", (_PlainPythonType,), {"wrapped_type": python_type}, _generic_base=_PlainPythonType)
    _PLAIN_PYTHON_DTYPES[python_type] = dtype
    return dtype


@cache
def is_untyped(dtype: Any) -> bool:
    """Has this field no type of its own yet?"""
    return issubclass(dtype, _UntypedValue)


def split_off_none(annotation: Any) -> tuple[Any, bool]:
    """``annotation`` without its ``| None``, and whether it said ``| None``.

    ``| None`` is how a field says it takes a blank: nothing else about it
    changes, so an edge is typed by the bare type."""
    if get_origin(annotation) in (Union, UnionType) and NoneType in get_args(annotation):
        members = tuple(member for member in get_args(annotation) if member is not NoneType)
        return (members[0] if len(members) == 1 else Union[members]), True  # noqa: UP007  (the members are runtime values)
    return annotation, False


#: Written on a field that holds many values of one type: an input that receives them
#: all as one list, an output that makes a row of each one it returns. A type
#: checker reads a ``Sequence``. ``list[X]`` stays free to mean one value that is a list.
type Series[X] = Sequence[X]


def read_field_annotation(annotation: Any) -> tuple[Any, bool, bool]:
    """What a field's annotation says: the annotation of one value, whether the field
    holds many, and whether it takes a blank.

    ``X``, ``X | None``, ``Series[X]``, ``Series[X | None]`` and ``Series[X] | None``
    are read here and nowhere else. Bare ``Series`` is many of whatever arrives."""
    written, says_blank = split_off_none(annotation)
    if written is not Series and get_origin(written) is not Series:
        return written, False, says_blank
    element, element_says_blank = split_off_none(get_args(written)[0] if written is not Series else Any)
    if element is Series or get_origin(element) is Series:
        raise NodeDeclarationError("Series[Series[...]] does not exist — a nested structure is a Table cell, opened one level by a node", code="nested_series")
    if dtype_of(element) is None:
        raise NodeDeclarationError(f"Series[{element!r}]: a series holds values an edge can carry, not {element!r}", code="series_element_invalid")
    return element, True, says_blank or element_says_blank


def dtype_of(annotation: Any) -> type[DType] | None:
    """The ``DType`` an annotation declares, or ``None`` if it declares none.

    ``Annotated[...]`` is unwrapped first, since an input's widget and an
    output's ``Result`` travel there. ``Any`` declares the private untyped
    type; a plain type an edge can
    carry declares the ``DType`` it is; the one place that decides::

        dtype_of(Annotated[Text, Result(title="Summary")])   # Text
        dtype_of(Any)                                         # the untyped type
        dtype_of(str)                                         # str, as a DType
        dtype_of(list[int])                                   # None
    """
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if annotation is Any:
        return _UntypedValue
    if isinstance(annotation, type) and issubclass(annotation, DType):
        return annotation
    return plain_python_dtype(annotation) if _an_edge_can_carry(annotation) else None


def type_name_for_sentence(declared: Any, many: bool = False) -> str:
    """A declared type named for an English sentence: ``text``, ``a series of
    text``, ``any`` — by its id, which is what a host also keys on."""
    return f"a series of {declared.id}" if many else declared.id


def _validate_dtype_field(value: Any) -> type[DType]:
    """Validator for a ``DTypeField`` field: whatever ``dtype_of`` makes a type; anything else raises."""
    declared = dtype_of(value)
    if declared is None:
        raise ValueError(f"dtype must be a DType, a type an edge can carry or Any, not {value!r}")
    return declared


#: The JSON schema of ``describe()``'s record.
_DESCRIPTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"id": {"type": "string"}, "of": {}, "columns": {"type": "array"}},
    "required": ["id"],
}

#: The type of a record's ``dtype`` field. In Python it holds a ``DType``
#: class; dumped it is ``describe()``'s record, so nothing downstream parses a type.
DTypeField = Annotated[
    Any,
    PlainValidator(_validate_dtype_field),
    PlainSerializer(lambda dtype: dtype.describe(), return_type=dict),
    WithJsonSchema(_DESCRIPTION_SCHEMA),
]
