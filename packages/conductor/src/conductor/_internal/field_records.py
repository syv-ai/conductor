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
declared result. A ``compute_added_inputs`` or ``compute_outputs`` hook builds
more of them, written the way a signature is: ``Input("amount", Number)``.

A ``Param`` with a widget is edited in place and reachable by an edge; a
``Param`` with no widget is filled by an edge only; ``connectable=False``
closes the input to edges. A parameter of a type no edge can carry
(``list[Branch]``) is closed by itself. Its limits are pydantic's and FastAPI's
words, and every value the input receives is held to them::

    temperature: Annotated[float, Param(ge=0, le=2, widget=Range())] = 1.0

A field nobody titled reads as its name in everyday words: ``case_sensitive``
is "Case sensitive".

These records are their own schema: ``dtype`` is a ``DTypeField``, so a
dump gives the type's ``describe()``, and what an editor offers is these records
dumped. A dump does not read back into one of these records: ``dtype``
is a ``DType`` class, never a description, and the class itself lives on
the signature.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from functools import cache
from typing import Annotated, Any, Final, TypedDict, Unpack

from pydantic import (
    ConfigDict,
    Field,
    PlainSerializer,
    PydanticUserError,
    SerializeAsAny,
    TypeAdapter,
    ValidationError,
    with_config,
)

from conductor._internal.codec import to_jsonable
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
from conductor.widgets import Widget


class _NoDefault:
    """The default of an input that has none."""

    def __repr__(self) -> str:
        return "REQUIRED"


_NO_DEFAULT: Final = _NoDefault()


@with_config(ConfigDict(extra="forbid"))
class ValueBounds(TypedDict, total=False):
    """The limits an input's value must keep to, named as pydantic and FastAPI name them."""

    ge: float | date
    gt: float | date
    le: float | date
    lt: float | date
    min_length: int
    max_length: int
    pattern: str


_BOUNDS_ADAPTER: Final = TypeAdapter(ValueBounds)


def _checked_bounds(bounds: Mapping[str, Any]) -> dict[str, Any]:
    """``bounds``, or the mistake in them where the input is written."""
    unknown = sorted(bounds.keys() - ValueBounds.__annotations__.keys())
    if unknown:
        raise NodeDeclarationError(f"an input has no argument {unknown[0]!r}; it takes title, description, widget, connectable and the limits {sorted(ValueBounds.__annotations__)}", code="bound_invalid")
    try:
        return dict(_BOUNDS_ADAPTER.validate_python(bounds))
    except ValidationError as invalid:
        raise NodeDeclarationError(f"an input's limits are {sorted(ValueBounds.__annotations__)}; {invalid.errors()[0]['msg']} at {invalid.errors()[0]['loc']}", code="bound_invalid") from invalid


def _type_of_values(dtype: Any, bounds: tuple[tuple[str, Any], ...], many: bool, takes_blank: bool, whole_may_be_blank: bool) -> Any:
    """What a value of an input is read as: one value or a list of them, each ``None`` where the input takes a blank, and held to its bounds; a blank is held to none."""
    limits = dict(bounds)
    if not many:
        bounded = Annotated[dtype, Field(**limits)] if limits else dtype
        return bounded | None if takes_blank else bounded
    of_many = list[dtype | None if takes_blank else dtype]
    of_many = Annotated[of_many, Field(**limits)] if limits else of_many
    return of_many | None if whole_may_be_blank else of_many


@cache
def _adapter_of_values(dtype: Any, bounds: tuple[tuple[str, Any], ...], many: bool, takes_blank: bool, whole_may_be_blank: bool) -> TypeAdapter[Any]:
    return TypeAdapter(_type_of_values(dtype, bounds, many, takes_blank, whole_may_be_blank))


def _what_decides_how_input_reads(input_: Input) -> tuple[Any, tuple[tuple[str, Any], ...], bool, bool, bool]:
    return input_.dtype, tuple(input_.bounds.items()), input_.many, input_.takes_blank, input_.takes_blank and not input_.required and input_.default is None


def type_read_by_input(input_: Input) -> Any:
    """The one type every value of ``input_`` is read as: a static, a default or what a call receives."""
    return _type_of_values(*_what_decides_how_input_reads(input_))


def adapter_of_input(input_: Input) -> TypeAdapter[Any]:
    """A validator of ``type_read_by_input``, built once per kind of input."""
    return _adapter_of_values(*_what_decides_how_input_reads(input_))


@cache
def humanised(name: str) -> str:
    """A field or node name in everyday words: ``case_sensitive`` is ``Case sensitive``."""
    return name.replace("-", " ").replace("_", " ").strip().capitalize()


def was_titled_by_its_author(field: Input | Output) -> bool:
    """Did someone write this field's title, rather than it being its name in everyday words?"""
    return field.title != humanised(field.name)


def _default_as_written(default: Any) -> Any:
    """A default as JSON, or ``None`` for a host's own value that has no JSON form: an editor has nothing to show of it, and the signature still holds it."""
    try:
        return to_jsonable(default)
    except (TypeError, ValueError, PydanticUserError):
        return None


class Param(ConductorModel):
    """What the author says about an input, inside ``Annotated`` on a parameter.

    ``title`` falls back to the parameter's name in everyday words. ``widget`` is
    the control a person edits it with, or ``None`` for an input only an edge
    fills. ``connectable`` is whether an edge can reach the input. The rest are
    the limits a value must keep to (``ge``, ``le``, ``max_length``, ...).
    """

    title: str | None = None
    description: str | None = None
    widget: SerializeAsAny[Widget] | None = None
    connectable: bool = True
    bounds: Mapping[str, Any] = Field(default_factory=dict)

    def __init__(
        self, *, title: str | None = None, description: str | None = None, widget: Widget | None = None, connectable: bool = True, **bounds: Unpack[ValueBounds],
    ) -> None:
        fields: dict[str, Any] = {"title": title, "description": description, "widget": widget, "connectable": connectable, "bounds": _checked_bounds(bounds)}
        super().__init__(**fields)


class Result(ConductorModel):
    """What the author says about an output, inside ``Annotated`` on the return or on a field of the returned record.

    ``name`` is what other nodes edge to, for a one-output return; it
    defaults to ``"result"``, and a record's field is named by the field.
    ``title`` falls back to the name in everyday words. Outputs of one node that
    share an ``exclusive_group`` are alternatives, exactly one of them produced
    per run, as with the two branches of an if/else node::

        if_true: Annotated[Text, Result(exclusive_group="branches")]
        if_false: Annotated[Text, Result(exclusive_group="branches")]
    """

    name: str | None = None
    title: str | None = None
    description: str | None = None
    exclusive_group: str | None = None


class Input(ConductorModel):
    """One input of a node, as the node carries it.

    ``name`` is what edges and bindings refer to (the ``field`` half of a
    ``Ref``). ``dtype`` is always a ``DType``, whatever was written
    (``dtype_of``), and a type no edge can carry (a schema, a list of
    branches) closes the input. A signature gives one per parameter; a
    ``compute_added_inputs`` hook builds more::

        Input("amount", Number, title="Amount", widget=TextWidget())

    With no ``default`` the input is ``required``. ``takes_blank`` is what
    the type said: ``Text | None``, or a type of no value of its own (``Any``).
    A blank reaching an input that doesn't take one means the node isn't
    called for that row. ``many`` is ``True`` for ``Series[X]``: ``dtype`` is
    ``X``, and a call receives a plain list of them. ``bounds`` holds the limits that are set, under
    the words ``Param`` takes them by.
    """

    name: str
    dtype: DTypeField
    title: str
    description: str | None = None
    widget: SerializeAsAny[Widget] | None = None
    connectable: bool = True
    bounds: Mapping[str, Any] = Field(default_factory=dict)
    #: What the value is when nothing binds the input; ``None`` on a required one.
    default: Annotated[Any, PlainSerializer(_default_as_written, return_type=Any, when_used="json")] = None
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
        widget: Widget | None = None,
        connectable: bool = True,
        default: Any = _NO_DEFAULT,
        **bounds: Unpack[ValueBounds],
    ) -> None:
        one_value, many, says_blank = read_field_annotation(dtype)
        dtype = dtype_of(one_value)
        if dtype is None or (not connectable and dtype is _UntypedValue):
            # Nothing can travel here, so a static is read as the plain type.
            dtype, connectable = plain_python_dtype(one_value), False
        takes_blank = says_blank or one_value is Any or is_untyped(dtype)
        if many and bounds.keys() - {"min_length", "max_length"}:
            raise NodeDeclarationError(f"input {name!r} takes many values, so it is held to min_length and max_length only; a limit on each value goes on its own node", code="bound_invalid")
        required = default is _NO_DEFAULT
        if default is None and not takes_blank:
            raise NodeDeclarationError(f"input {name!r} defaults to None, which is a blank; write its type with | None", code="none_default_needs_blank")
        checked_bounds = _checked_bounds(bounds)
        if not required and default is not None:
            try:
                default = _adapter_of_values(dtype, tuple(checked_bounds.items()), many, takes_blank, False).validate_python(default)
            except ValidationError as invalid:
                raise NodeDeclarationError(f"input {name!r} defaults to {default!r}, which its type can't read: {invalid.errors()[0]['msg']}", code="default_invalid") from invalid
        super().__init__(
            name=name, dtype=dtype, title=title or humanised(name), description=description, widget=widget,
            connectable=connectable, bounds=checked_bounds, default=None if required else default, required=required, takes_blank=takes_blank, many=many,
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
    them, each a row. ``exclusive_group`` is read by the compiler and the editor, never by the
    engine, which only propagates the skip.
    """

    name: str
    dtype: DTypeField
    title: str
    description: str | None = None
    exclusive_group: str | None = None
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
        exclusive_group: str | None = None,
    ) -> None:
        one_value, many, says_blank = read_field_annotation(dtype)
        super().__init__(name=name, dtype=one_value, title=title or humanised(name), description=description, exclusive_group=exclusive_group, takes_blank=says_blank, many=many)
