"""What a value *is*, not just what Python holds it in.

A ``DType`` is a real Python type — subclass ``str``, ``float``, ``date`` or
nothing at all — that additionally declares its wire id, its human title,
and one question: ``accepts(source)``.

It deliberately does **not** declare a widget. ``Text`` is legitimately a
textarea, a single-line field, a template editor or a dropdown of choices,
so a type that named one would be guessing on behalf of every input that
uses it. Widgets are declared per input, and an input without one is an
error.

And it does **not** convert. A value arrives at a node as the type the wire
carried. "May this edge connect?" is the whole of what a type is asked
about wiring, and ``accepts`` answers it once, in the type, at compile.

What a value reads like when it becomes text for a person is the type's to
say, once: ``as_text(value)``, default ``str``. Every user-facing rendering
calls it, so a type whose values read differently overrides it here and
nowhere else.

A node that does not read its value declares no type: it annotates the
input ``Any``, compile records what actually arrives there, and the
node's ``compute_outputs`` types its outputs from ``arriving`` — so
``accepts`` is never asked about an unconstrained input, and nothing on
a wire ever carries "could not say". ``Single`` is the open roster's
spelling on ``**inputs``: every wired name an input, typed by its wire.

Conductor defines no concrete ``DType``. What counts as a domain type is a
domain question.
"""

from __future__ import annotations

from abc import ABC
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    get_args,
    get_origin,
)

from pydantic import PlainSerializer, PlainValidator, WithJsonSchema
from pydantic_core import core_schema

if TYPE_CHECKING:
    from pydantic import GetCoreSchemaHandler

__all__ = [
    "DType",
    "DTypeRef",
    "Single",
    "description_of",
    "dtype_of",
    "registered_dtypes",
]

_BY_ID: dict[str, type["DType"]] = {}


class DType(ABC):
    """Base for every domain type.

    The base has no ``id`` and no ``title`` of its own: it is never on a
    wire, and a subclass that forgets its id must not inherit one.
    """

    #: Stable id. What the persisted graph and the frontend key on.
    id: ClassVar[str]
    #: Human name, in the host's language. Presentation only.
    title: ClassVar[str]
    #: The element type of a collection; ``None`` for a scalar. Declared on
    #: the base so ``accepts`` can tell a series from a scalar without
    #: importing the module that defines the series — "a series into a
    #: scalar input lifts" is a rule of the type system, not of ``Series``.
    element: ClassVar[Any] = None
    #: May a person author a value of this type directly — type it into a
    #: cell, a form answer, a schema field? ``False`` unless the type says
    #: so: most types are carried on wires, not typed in. The host's
    #: derived choice lists read this and nothing else.
    authorable: ClassVar[bool] = False

    @classmethod
    def refuses_whole(cls) -> tuple[str, str] | None:
        """Why a value of this type may not be **received whole** by a node
        that reads it — as a problem ``(code, message)`` — or ``None``, the
        answer for nearly every type. On an ``Any`` input a value is
        routed, not read, and nothing here is asked; on an open roster
        the node will read it, so a type that can be *stated
        incompletely* — a host's table whose columns nobody said — answers
        here and compile refuses the wire on the field, naming the fix.
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
        existing = _BY_ID.get(cls.id)
        if existing is not None and existing is not cls:
            if issubclass(cls, existing):
                # A parameterisation or a narrowing, not a collision:
                # `Series[Text]` is a subclass of `Series` and means the
                # same wire id. The declaring type keeps the entry, so
                # `registered_dtypes()` lists each word once.
                return
            raise ValueError(
                f"dtype id {cls.id!r} is already declared by {existing.__name__}"
            )
        _BY_ID[cls.id] = cls

    # -- the wire ------------------------------------------------------

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """This type, as the frontend reads it.

        An object rather than a string, so nothing downstream parses a type.
        ``accepted_as`` answers the editor's one applicability question —
        where may this value land? — as the ids of every registered type
        whose ``accepts`` admits this one, its own among them. Derived off
        ``accepts`` itself, one writer, so a type that widens its welcome
        (a number admitting its whole-number kin) is captured with no
        catalog in the browser. ``Series`` overrides the method to nest
        its element type.
        """
        return {"id": cls.id, "accepted_as": [d.id for d in registered_dtypes() if _admits(d, cls)]}

    # -- text for a person -----------------------------------------------

    @classmethod
    def as_text(cls, value: Any) -> str:
        """``value``, as a person reads it.

        Every place a value becomes user-facing text renders through this,
        so a type whose values read differently from ``str`` says so once,
        here, and no rendering site grows its own formatting. The default
        is ``str``: most types read as they are.
        """
        return str(value)

    # -- the one question ------------------------------------------------

    @classmethod
    def accepts(cls, source: Any) -> bool:
        """May a value of ``source`` land on an input of this type?

        The target decides, because an input is where a node states what it
        needs. ``issubclass`` for a scalar; a series arriving here is judged
        by its element — that it lifts the node is compile's reading of the
        shapes, not the type's. ``Series`` overrides this for the
        other direction. An ``Any`` input never reaches here: nothing
        arriving on it is judged, and what arrived is recorded instead.
        """
        if cls is DType:
            raise TypeError(
                "an input must declare a concrete DType, not the base — "
                "bare DType as a target would accept anything"
            )
        if source.element is not None:
            return cls.accepts(source.element)
        return issubclass(source, cls)

    # -- pydantic ---------------------------------------------------------

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: "GetCoreSchemaHandler"
    ) -> core_schema.CoreSchema:
        """Validate as the builtin this type is built on, then wrap.

        pydantic does not know a subclass of ``str``. Without this, a
        ``Text`` field validates to a bare ``str`` and every
        ``isinstance(value, Text)`` downstream is false. A type built on
        nothing is validated by instance.
        """
        builtin = _builtin_base(cls)
        if builtin is None:
            return core_schema.is_instance_schema(cls)
        return core_schema.no_info_after_validator_function(
            lambda value: value if isinstance(value, cls) else cls(value),
            handler.generate_schema(builtin),
        )


def _builtin_base(cls: type[DType]) -> type | None:
    """The ``str`` / ``float`` / ``date`` a type is built on, if any."""
    for base in cls.__mro__[1:]:
        if issubclass(base, DType) or base in (ABC, object):
            continue
        if base.__module__ in ("abc", "collections.abc", "typing"):
            continue
        return base
    return None


class Single:
    """``**inputs: Single`` — an open roster.

    The whole declaration: every wired name is an input, received as one
    value — the *carried* shape, series and all, so a series arrives whole
    and the loop is the node author's to write. It is spelled on
    ``**inputs`` and nowhere else; compile makes one ``Input`` per wire,
    typed by it. ``Interface.of`` reads it; nothing here does.
    """


def description_of(declared: Any) -> dict[str, Any] | None:
    """A declared type's wire form: a ``DType``'s ``describe()``, an
    unconstrained input's ``{"id": "any"}``, and ``None`` for a type
    nothing travels on — the static type of a handle-less input. The one
    place the three are told apart, so ``DTypeRef`` and ``Series.describe``
    agree."""
    if declared is Any:
        return {"id": "any"}
    if isinstance(declared, type) and issubclass(declared, DType):
        return declared.describe()
    return None


def _declared(value: Any) -> Any:
    """The validator for a ``dtype`` slot: whatever was declared, unchanged.

    A record's ``dtype`` is a declaration read off a signature, never a
    value read off the wire, so there is nothing to coerce — and ``Any``
    is not a type pydantic could validate a class against."""
    return value


#: How a record carries a declared type on the wire: the class — or
#: ``Any`` — in Python, ``description_of`` in JSON, and a JSON schema
#: a response model can publish. Two levels deep and no deeper, because
#: ``Series[Series[...]]`` does not exist — so the schema needs no
#: recursion. The ``null`` arm is a handle-less input's static type:
#: nothing travels on it, so it has no wire form.
_SCALAR_WIRE_ID: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "accepted_as": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["id"],
}
DTypeRef = Annotated[
    Any,
    PlainValidator(_declared),
    PlainSerializer(description_of, return_type=dict | None),
    WithJsonSchema(
        {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {**_SCALAR_WIRE_ID["properties"], "of": _SCALAR_WIRE_ID},
                    "required": ["id"],
                },
                {"type": "null"},
            ]
        }
    ),
]


def dtype_of(annotation: Any) -> Any:
    """The ``DType`` — or ``Any``, the unconstrained marker — an annotation
    declares, or ``None`` if it declares neither.

    Unwraps ``Annotated[...]``, since a param's widget travels there.
    """
    if get_origin(annotation) is Annotated:
        annotation = get_args(annotation)[0]
    if annotation is Any:
        return Any
    if isinstance(annotation, type) and issubclass(annotation, DType):
        return annotation
    return None


def registered_dtypes() -> tuple[type[DType], ...]:
    """Every declared type, for a host reporting its own vocabulary — what
    can travel on a wire, and nothing else."""
    return tuple(_BY_ID.values())


def _admits(target: type[DType], source: type[DType]) -> bool:
    """``target.accepts(source)``, where the question is well-posed.

    ``describe()`` asks it of every registered type; a type that refuses to
    be a target at all — bare ``Series`` raises, by its own rule — admits
    nothing, and that refusal is the answer, not an error here."""
    try:
        return target.accepts(source)
    except TypeError:
        return False
