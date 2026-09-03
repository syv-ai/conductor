"""``DType`` — a value's type, as the flow engine sees it.

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

A type answers one question about wiring — ``target.accepts(source)``:
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

* **Convert.** A value arrives at a node as the type the wire carried.
  Where a conversion seems needed, the answer is a subtype, a node that
  does the work, or an input declared with the widest type the node
  handles.
* **Pick a widget.** The same ``Text`` may be a textarea, a single line or
  a dropdown; every input declares its own widget.
* **Format itself beyond text.** ``as_text`` is the one rendering hook:
  override it when a value should read differently from ``str(value)``.

An input that only routes a value it never reads is annotated ``Any``
instead of a type, and the type of what actually arrives is recorded when
the flow is compiled. ``Single`` marks an open roster, ``**inputs: Single``:
every wired name becomes an input of that node.

Conductor defines no concrete ``DType`` except ``Series``. Which types
exist is the host application's decision.
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, get_args, get_origin

from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler

class DType(ABC):
    """Base class for every type a value on a wire can have.

    Subclass it together with the builtin the type is built on and declare
    ``id`` and ``title``; the class is registered on definition. The base
    itself has no ``id`` and is never on a wire.
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
    #: most values are carried on wires rather than typed in.
    authorable: ClassVar[bool] = False

    #: Every declared type by its ``id``; filled by ``__init_subclass__``,
    #: read by ``registered_dtypes``.
    _by_id: ClassVar[dict[str, type["DType"]]] = {}

    @classmethod
    def refuses_whole(cls) -> tuple[str, str] | None:
        """Why a node may not receive a value of this type as a whole, if
        there is a reason.

        Returns a problem ``(code, message)``, or ``None`` — the answer for
        nearly every type, and the default. Only a type that can be declared
        incompletely (a table whose columns nobody stated) overrides it.

        The compiler asks it of a source type wired into an open roster
        (``**inputs: Single``), where the node will read the value; a
        ``None`` lets the wire through, a pair becomes a fatal problem on
        the field with that code and message. A value only routed through
        an ``Any`` input is never asked.
        """
        return None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # A subclass must say what it is. A parameterised ``Series[Text]``
        # inherits ``Series``'s id and passes.
        if getattr(cls, "id", None) is None or not getattr(cls, "title", None):
            raise TypeError(
                f"{cls.__name__} must declare a class-level 'id' and 'title'"
            )
        existing = DType._by_id.get(cls.id)
        if existing is not None and existing is not cls:
            if issubclass(cls, existing):
                # A parameterisation or a narrowing, not a collision:
                # `Series[Text]` is a subclass of `Series` and shares its
                # id. The declaring type keeps the registry entry, so
                # `registered_dtypes()` lists each type once.
                return
            raise ValueError(
                f"dtype id {cls.id!r} is already declared by {existing.__name__}"
            )
        DType._by_id[cls.id] = cls

    # -- the description --------------------------------------------------

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """This type as a JSON-ready record, for the frontend.

        ``{"id": ..., "accepted_as": [...]}`` — an object rather than a
        string, so nothing downstream parses a type. ``accepted_as`` lists
        the ids of every registered type whose ``accepts`` admits this one
        (its own included), which is what an editor needs to know where a
        value may be dropped. It is derived from ``accepts``, so widening a
        type's welcome updates it automatically. ``Series`` overrides this
        to nest its element type.
        """
        return {"id": cls.id, "accepted_as": [d.id for d in registered_dtypes() if d._admits(cls)]}

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

    @classmethod
    def _admits(cls, source: type[DType]) -> bool:
        """``accepts(source)``, treating "refuses to be a target" as ``False``.

        ``describe()`` asks this of every registered type. Bare ``Series``
        raises rather than answer, and for ``accepted_as`` that refusal
        simply means "not here".
        """
        try:
            return cls.accepts(source)
        except TypeError:
            return False


class Single:
    """Marker for an open roster: ``def run(self, **inputs: Single)``.

    Every name wired into such a node becomes an input, typed by its wire,
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


def registered_dtypes() -> tuple[type[DType], ...]:
    """Every ``DType`` declared so far — everything that can travel on a wire."""
    return tuple(DType._by_id.values())
