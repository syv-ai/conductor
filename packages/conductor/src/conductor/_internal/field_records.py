"""The records that describe a node's fields — what an author writes, and what the node carries.

An author says what a person should read about a field inside
``Annotated``: ``Param`` on a parameter, ``Result`` on the return. A
parameter or a return with neither is titled by its name::

    @node("summarise")
    def summarise(
        text: Annotated[Text, Param(title="Text", widget=Textarea())],
        rows: Series[Text],
    ) -> Annotated[Text, Result(title="Summary")]:
        ...

The node ends up with an ``Input`` per parameter and an ``Output`` per
declared result. A ``compute_inputs`` or ``compute_outputs`` hook builds
more of them, written the way a signature is: ``Input("amount", Number)``.

A ``Param`` with a widget is edited in place and reachable by an edge; a
``Param`` with no widget is filled by an edge only; ``connectable=False``
closes the input to edges. A parameter of a type no edge can carry
(``list[Branch]``) is closed by itself.

These records are their own schema: ``dtype`` is a ``DTypeField``, so a
dump gives the type's ``describe()``, and what an editor offers is these records
dumped. A dump does not read back into one of these records: ``dtype``
is a ``DType`` class, never a description, and the class itself lives on
the signature.
"""

from __future__ import annotations

from typing import Any, Final

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.dtype import (
    DTypeField,
    _UntypedValue,
    dtype_of,
    is_untyped,
    plain_python_dtype,
    read_field_annotation,
)
from conductor._internal.errors import NodeDeclarationError
from conductor.widgets import AnyWidget


class _NoDefault:
    """The default of an input that has none."""

    def __repr__(self) -> str:
        return "REQUIRED"


_NO_DEFAULT: Final = _NoDefault()


class Param(ConductorModel):
    """What the author says about an input, inside ``Annotated`` on a parameter.

    ``title`` falls back to the parameter's name. ``widget`` is the control a
    person edits it with, or ``None`` for an input only an edge fills.
    ``connectable`` is whether an edge can reach the input.
    """

    title: str | None = None
    description: str | None = None
    widget: AnyWidget | None = None
    connectable: bool = True


class Result(ConductorModel):
    """What the author says about an output, inside ``Annotated`` on the return or on a field of the returned record.

    ``name`` is what other nodes edge to, for a one-output return; it
    defaults to ``"result"``, and a record's field is named by the field.
    ``title`` falls back to the name. Outputs of one node that share a
    ``choice`` are exclusive alternatives, exactly one of them produced per
    run, as with the two branches of an if/else node::

        if_true: Annotated[Text, Result(title="If true", choice="branches")]
        if_false: Annotated[Text, Result(title="If false", choice="branches")]
    """

    name: str | None = None
    title: str | None = None
    description: str | None = None
    choice: str | None = None


class Input(ConductorModel):
    """One input of a node, as the node carries it.

    ``name`` is what edges and bindings refer to (the ``field`` half of a
    ``Ref``). ``dtype`` is always a ``DType``, whatever was written
    (``dtype_of``), and a type no edge can carry (a schema, a list of
    branches) closes the input. A signature gives one per parameter; a
    ``compute_inputs`` hook builds more::

        Input("amount", Number, title="Amount", widget=TextWidget())

    With no ``default`` the input is ``required``. ``takes_blank`` is what
    the type said: ``Text | None``, or a type of no value of its own (``Any``).
    A blank reaching an input that doesn't take one means the node isn't
    called for that row. ``many`` is ``True`` for ``Series[X]``: ``dtype`` is
    ``X``, and a call receives a plain list of them.
    """

    name: str
    dtype: DTypeField
    title: str
    description: str | None = None
    widget: AnyWidget | None = None
    connectable: bool = True
    #: What the value is when nothing binds the input; ``None`` on a required one.
    default: Any = None
    required: bool = True
    takes_blank: bool = False
    many: bool = False

    def __init__(
        self,
        name: str,
        dtype: Any,
        /,
        *,
        title: str | None = None,
        description: str | None = None,
        widget: AnyWidget | None = None,
        connectable: bool = True,
        default: Any = _NO_DEFAULT,
    ) -> None:
        one_value, many, says_blank = read_field_annotation(dtype)
        dtype = dtype_of(one_value)
        if dtype is None or (not connectable and dtype is _UntypedValue):
            # Nothing can travel here, so a static is read as the plain type.
            dtype, connectable = plain_python_dtype(one_value), False
        takes_blank = says_blank or one_value is Any or is_untyped(dtype)
        required = default is _NO_DEFAULT
        if default is None and not takes_blank:
            raise NodeDeclarationError(f"input {name!r} defaults to None, which is a blank; write its type with | None", code="none_default_needs_blank")
        super().__init__(
            name=name, dtype=dtype, title=title or name, description=description, widget=widget,
            connectable=connectable, default=None if required else default, required=required, takes_blank=takes_blank, many=many,
        )


class Output(ConductorModel):
    """One output of a node, as the node carries it.

    A signature gives one per declared result; a ``compute_outputs`` hook
    builds them for a node whose outputs depend on its values or on what
    arrives. ``dtype`` is a ``DType`` or ``Any`` (typed by a hook once it
    knows what arrives). It carries no widget or default: nobody supplies a
    result. ``takes_blank`` is ``True`` when the type says ``| None``: a
    return of ``None`` is valid there and nowhere else. ``many`` is ``True``
    for ``Series[X]``: ``dtype`` is ``X``, and the node returns a sequence of
    them, each a row. ``choice`` is read by the compiler and the editor, never by the
    engine, which only propagates the skip.
    """

    name: str
    dtype: DTypeField
    title: str
    description: str | None = None
    choice: str | None = None
    takes_blank: bool = False
    many: bool = False

    def __init__(
        self,
        name: str,
        dtype: Any,
        /,
        *,
        title: str | None = None,
        description: str | None = None,
        choice: str | None = None,
    ) -> None:
        one_value, many, says_blank = read_field_annotation(dtype)
        super().__init__(name=name, dtype=one_value, title=title or name, description=description, choice=choice, takes_blank=says_blank, many=many)
