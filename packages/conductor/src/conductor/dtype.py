"""``DType`` — a value's type, as the engine sees it.

A ``DType`` is a real Python class, usually built on a builtin::

    class Text(DType, str):
        id = "text"
        title = "Text"

    class Number(DType, float):
        id = "number"
        title = "Number"

so ``Text("hello")`` is both a ``str`` and a ``Text``, a type checker sees
``Text`` where ``Text`` is meant, and a pydantic model with a ``Text``
field gives back a ``Text`` (see ``__get_pydantic_core_schema__``). ``id``
is the stable name the persisted graph and the frontend use; ``title`` is
what a person reads.

A type answers one question about edges — ``target.accepts(source)``:
may a value of type ``source`` land on an input declared as ``target``?
The default is ``issubclass``, so a subtype is accepted wherever its
parent is, and a ``Series`` of something is judged by its element::

    class Integer(Number):
        id = "integer"
        title = "Integer"

    Number.accepts(Integer)   # True
    Integer.accepts(Number)   # False
    Number.accepts(Series[Number])   # True — the node then runs once per row

Three things a ``DType`` deliberately does not do:

* **Convert.** A value arrives at a node as the type the edge carried.
  Where a conversion seems needed, the answer is a subtype, a node that
  does the work, or an input declared with the widest type the node
  handles.
* **Pick a widget.** The same ``Text`` may be a textarea, a single line or
  a dropdown; every input declares its own widget.
* **Format itself beyond text.** ``as_text`` is the one rendering hook:
  override it when a value should read differently from ``str(value)``.

An input that only routes a value it never reads is annotated ``Any``
instead of a type, and the type of what actually arrives is recorded when
the graph is compiled. ``Single`` marks an open interface, ``**inputs: Single``:
every connected name becomes an input of that node.

Conductor defines no concrete ``DType`` except ``Series``. Which types
exist is the host application's decision, made on its ``NodeRegistry``:
the types its nodes declare plus ``registry.add_types(...)`` are that
registry's vocabulary, and nothing is recorded process-wide. Declaring a
type is therefore free of side effects — a notebook cell can declare the
same ``Text`` twice — and one id claimed by two classes is refused where
the second reaches a registry, never at class definition.
"""

from __future__ import annotations

from abc import ABC, ABCMeta
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, get_args, get_origin

from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler

class _DTypeMeta(ABCMeta):
    """A type prints as its name, ``Text`` or ``Series[Text]``, the way an author writes it."""

    def __repr__(cls) -> str:
        return cls.__name__


class DType(ABC, metaclass=_DTypeMeta):
    """Base class for every type a value on an edge can have.

    Subclass it together with the builtin the type is built on and declare
    ``id`` and ``title`` in the class body. The base itself has no ``id``
    and is never on an edge.

    A subclass is a type of its own and names itself: a class that leaves
    ``id`` to its parent is refused at definition, since a ``Whole`` that
    quietly travels as ``number`` is the mistake nobody sees. The one
    exception is a **parameterisation** — ``Series[Text]``, a host's
    ``Table[columns]`` — a subclass built to carry a detail of its base's
    type and rightly sharing the base's id. A factory marks one with the
    class keyword ``parameterises=Base``; the registry then files it under
    its base (or, for a series, its element) rather than as a word of its
    own.
    """

    #: Stable identifier, used by the persisted graph and the frontend.
    id: ClassVar[str]
    #: Human-readable name, in the host's language.
    title: ClassVar[str]
    #: For a collection, the type of its elements; ``None`` for a scalar.
    #: Declared on the base so ``accepts`` can recognise a series without
    #: importing ``conductor.series``.
    element: ClassVar[Any] = None
    #: May a person type a value of this type in directly (into a cell, a
    #: form, a schema field)? ``False`` unless the type says otherwise;
    #: most values are carried on edges rather than typed in.
    authorable: ClassVar[bool] = False
    #: For a parameterisation (``Series[Text]``, ``Table[columns]``), the
    #: type it parameterises and whose ``id`` it carries; ``None`` for a
    #: type of its own.
    parameterises: ClassVar[type["DType"] | None] = None

    @classmethod
    def refuses_whole(cls) -> None:
        """Raise ``conductor.errors.Refuses`` when a node may not receive a value of this type as a whole.

        Returns quietly for nearly every type, and by default. Only a type
        that can be declared incompletely (a table whose columns nobody
        stated) overrides it, raising the same ``Refuses(code, message)`` a
        node's hook raises.

        The compiler asks it of a source type connected into an open interface
        (``**inputs: Single``), where the node will read the value; a
        refusal becomes a fatal problem on the field with that code and
        message. A value only routed through an ``Any`` input is never asked.
        """

    def __init_subclass__(cls, *, parameterises: type["DType"] | None = None, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if parameterises is not None:
            if not (
                isinstance(parameterises, type)
                and issubclass(parameterises, DType)
                and parameterises is not DType
                and issubclass(cls, parameterises)
            ):
                raise TypeError(
                    f"{cls.__name__} parameterises {parameterises!r}, which is not a concrete DType among its bases"
                )
            if "id" in vars(cls):
                raise TypeError(
                    f"{cls.__name__} parameterises {parameterises.__name__} and so carries its id; "
                    "a type with an id of its own is not a parameterisation"
                )
            cls.parameterises = parameterises
        else:
            if "id" not in vars(cls):
                raise TypeError(
                    f"{cls.__name__} declares no 'id' of its own. A subclass is a type of its own and "
                    "names itself; a subclass that only carries a detail of its base is declared with "
                    "parameterises=<base> and keeps the base's id"
                )
            # Set on every type of its own, so a self-named subclass of a
            # parameterisation does not inherit the mark.
            cls.parameterises = None
        if not getattr(cls, "title", None):
            raise TypeError(f"{cls.__name__} must declare a class-level 'title'")

    # -- the description --------------------------------------------------

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """This type as a JSON-ready record, for the frontend: ``{"id": ...}``.

        An object rather than a string, so nothing downstream parses a
        type. ``Series`` overrides this to nest its element type. Where a
        value of this type may land — every type whose ``accepts`` admits
        it — is a question about a vocabulary, so it is the registry's
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
        it needs. Default: ``issubclass(source, cls)``. A ``Series`` on the
        source side is judged by its element type — the compiler then runs
        the node once per row. Bare ``DType`` as a target raises: it would
        accept anything.
        """
        if cls is DType:
            raise TypeError(
                "an input must declare a concrete DType, not the base — "
                "bare DType as a target would accept anything"
            )
        if source.element is not None:
            return cls.accepts(source.element)
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
        builtin = cls._builtin_base()
        if builtin is None:
            return core_schema.is_instance_schema(cls)
        return core_schema.no_info_after_validator_function(
            lambda value: value if isinstance(value, cls) else cls(value),
            handler.generate_schema(builtin),
        )

    # -- internals -----------------------------------------------------------

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


class Single:
    """Marker for an open interface: ``def run(self, **inputs: Single)``.

    Every name connected into such a node becomes an input, typed by its edge,
    and each is received as one value — a series arrives as a whole series.
    The marker is only meaningful on ``**inputs``; the registry reads it
    when it derives a node's interface from its signature.
    """


def dtype_of(annotation: Any) -> Any:
    """The ``DType`` an annotation declares, ``Any`` if it declares that,
    or ``None`` if it declares neither.

    ``Annotated[...]`` is unwrapped first, since an input's widget and an
    output's ``Result`` travel there::

        dtype_of(Annotated[Text, Result(title="Summary")])   # Text
        dtype_of(Any)                                         # Any
        dtype_of(str)                                         # None
    """
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if annotation is Any:
        return Any
    if isinstance(annotation, type) and issubclass(annotation, DType):
        return annotation
    return None
