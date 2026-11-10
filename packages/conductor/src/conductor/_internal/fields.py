"""What a node author writes in a signature, and the one record each input and output becomes.

A type is any class pydantic can check and write as JSON. A host's own type
may declare ``id`` (the name a stored graph and an editor know it by),
``display_title``, and a classmethod ``accepts(source)`` that lets values of
another type land on it; otherwise a subclass lands where its parent is
asked for. A subclass of a builtin (``class Text(str)``) is checked as the
builtin, then made one.

A field has one of three shapes: ``one`` value, ``many`` values
(``Series[X]``, a plain list to the node), or a ``table`` (``Table[R]``, a
``polars.DataFrame``). The shape decides how the field is wired and where its
node runs; the type decides what an edge may carry and how a value is
checked and written as JSON.
"""

from __future__ import annotations

import datetime
import functools
import types
import typing
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, Literal, Union, get_args, get_origin

import polars as pl
from pydantic import AfterValidator, ConfigDict, TypeAdapter
from pydantic import Field as PydanticField
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic_core import to_jsonable_python

from conductor._internal.errors import NodeDeclarationError


class _Required:
    def __repr__(self) -> str:
        return "REQUIRED"


#: The default of an input nothing defaults.
REQUIRED: Any = _Required()


class _Skipped:
    """The type of ``SKIPPED``: one falsy value whose repr is ``SKIPPED``."""

    _instance: _Skipped | None = None

    def __new__(cls) -> _Skipped:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "SKIPPED"

    def __reduce__(self) -> str:
        return "SKIPPED"

    def __bool__(self) -> bool:
        return False


#: The value of an output on a branch its node did not take. Typed ``Any``
#: for a checker: it goes where a value would, on any output.
SKIPPED: Any = _Skipped()


class Ask(BaseException):
    """Raised by a node to stop on its row and ask a person; the node's return type says what they answer.

    ``prompt`` is what the person reads. ``choices`` limits the answer to
    these values (a list, or a mapping of value to title), ``default``
    proposes one, and ``widget`` asks for a control of the host's. Nothing
    runs for the row until an answer arrives; the answer is the node's
    outputs, and the node is not called again for that row.

    A ``BaseException``, so an ``except Exception`` in the node lets it through.
    """

    def __init__(self, prompt: str, *, choices: Any = None, default: Any = None, widget: Any = None) -> None:
        super().__init__(prompt)
        self.prompt, self.choices, self.default, self.widget = prompt, choices, default, widget


#: The limits ``Param`` takes, by the words pydantic's ``Field`` uses, and the
#: types each kind applies to.
_NUMERIC = frozenset({"ge", "gt", "le", "lt", "multiple_of"})
_SIZED = frozenset({"min_length", "max_length", "pattern"})


@dataclass(frozen=True, init=False)
class Param:
    """What a person reads about a field, and the limits a value must keep, inside ``Annotated``::

        times: Annotated[int, Param(title="Times", ge=0)] = 1

    ``title`` falls back to the name in everyday words. ``widget`` is the
    control a person edits the input with, carried to the host and never read
    here. Outputs of one node that share an ``exclusive_group`` are
    alternatives: on each row the node gives a value on one of them and
    ``SKIPPED`` on the rest. The limits are pydantic's: ``ge``, ``gt``,
    ``le``, ``lt``, ``multiple_of``, ``min_length``, ``max_length``, ``pattern``.
    """

    title: str | None = None
    description: str | None = None
    widget: Any = None
    exclusive_group: str | None = None
    constraints: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self, *, title: str | None = None, description: str | None = None, widget: Any = None, exclusive_group: str | None = None, **constraints: Any,
    ) -> None:
        unknown = sorted(constraints.keys() - _NUMERIC - _SIZED)
        if unknown:
            raise NodeDeclarationError(
                f"Param has no argument {unknown[0]!r}; it takes title, description, widget, exclusive_group and the limits {sorted(_NUMERIC | _SIZED)}",
                code="bound_invalid",
            )
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "widget", widget)
        object.__setattr__(self, "exclusive_group", exclusive_group)
        object.__setattr__(self, "constraints", constraints)


class _ManyMarker:
    pass


@dataclass(frozen=True)
class _TableMarker:
    row: Any


class _FromRunMarker:
    pass


if TYPE_CHECKING:
    from builtins import list as Series

    FromRun = Annotated
else:

    class FromRun:
        """``FromRun[X]``: a value the host hands the run by its type (a session, who is running), never an input.

        It takes no edge, is not stored in the record, and is no part of what
        makes a row's result reusable. The run is given one with
        ``from_run={X: value}``.
        """

        def __class_getitem__(cls, item: Any) -> Any:
            return Annotated[item, _FromRunMarker()]

    class Series:
        """``Series[X]``: many values of ``X``, one per row; a plain list to the node."""

        def __class_getitem__(cls, item: Any) -> Any:
            return Annotated[list[item], _ManyMarker()]


class Table:
    """``Table[R]``: a whole table whose columns the ``TypedDict`` ``R`` names; a ``polars.DataFrame`` to the node.

    A bare ``Table`` is a table of any columns.
    """

    def __class_getitem__(cls, row: Any) -> Any:
        return Annotated[pl.DataFrame, _TableMarker(row)]


@functools.cache
def humanised(name: str) -> str:
    """A field or node name in everyday words: ``case_sensitive`` is ``Case sensitive``."""
    return name.replace("-", " ").replace("_", " ").strip().capitalize()


@dataclass(frozen=True)
class FieldSpec:
    """One input or output: its shape, the type of one value, and how a value is checked."""

    name: str
    shape: Literal["one", "many", "table"]
    type: Any
    optional: bool = False
    default: Any = REQUIRED
    param: Param = field(default_factory=Param)
    #: A table's columns by name, from its row type; ``None`` for a table of any columns.
    columns: dict[str, Any] | None = None

    @property
    def required(self) -> bool:
        return self.default is REQUIRED

    @property
    def title(self) -> str:
        return self.param.title or humanised(self.name)

    @functools.cached_property
    def adapter(self) -> TypeAdapter[Any]:
        """Checks one value of this field, held to its limits."""
        carried = _carried(self.type)
        one: Any = Annotated[carried, PydanticField(**self.param.constraints)] if self.param.constraints else carried
        return TypeAdapter(_list_of(one) if self.shape == "many" else one)

    @functools.cached_property
    def column_adapter(self) -> TypeAdapter[Any]:
        """Checks and writes one output's values over many units, blanks included."""
        carried = _carried(self.type)
        one: Any = _list_of(carried) if self.shape == "many" else carried
        return TypeAdapter(_list_of(one | None), config=_FLOATS_AS_TEXT)

    def describe(self) -> dict[str, Any]:
        """The field as JSON: its shape, its type's ``id`` and JSON schema, and what a person reads about it."""
        out: dict[str, Any] = {"name": self.name, "title": self.title, "shape": self.shape, "required": self.required}
        if self.shape != "table":
            out["type"] = type_id(without_none(self.type))
            out["schema"] = _schema(self.type)
        else:
            out["columns"] = {c: _schema(t) for c, t in (self.columns or {}).items()}
        if not self.required:
            out["default"] = to_jsonable_python(self.default, fallback=lambda _: None)
        for key in ("description", "widget", "exclusive_group"):
            if getattr(self.param, key) is not None:
                out[key] = to_jsonable_python(getattr(self.param, key), fallback=str)
        if self.param.constraints:
            out["constraints"] = to_jsonable_python(self.param.constraints, fallback=str)
        return out


def _schema(hint: Any) -> dict[str, Any]:
    """The JSON schema of a type; ``{}`` for one that writes none, which an editor offers no control for."""
    try:
        return TypeAdapter(_carried(hint)).json_schema()
    except PydanticInvalidForJsonSchema:
        return {}


def read_field(name: str, hint: Any, default: Any = REQUIRED, where: str = "") -> FieldSpec:
    """Read one annotation into a ``FieldSpec``; a hint conductor can't carry raises ``NodeDeclarationError``."""
    param = Param()
    if get_origin(hint) is Annotated:
        base, *meta = get_args(hint)
        params = [m for m in meta if isinstance(m, Param)]
        param = params[0] if params else param
        markers = [m for m in meta if isinstance(m, (_ManyMarker, _TableMarker))]
        if markers and isinstance(markers[0], _ManyMarker):
            item = get_args(base)[0]
            return _checked(FieldSpec(name, "many", item, _is_optional(item), default, param), where)
        if markers and isinstance(markers[0], _TableMarker):
            return _checked(FieldSpec(name, "table", pl.DataFrame, False, default, param, _columns_of(markers[0].row)), where)
        hint = base
    if hint is pl.DataFrame or hint is Table:
        return _checked(FieldSpec(name, "table", pl.DataFrame, False, default, param, None), where)
    return _checked(FieldSpec(name, "one", hint, _is_optional(hint), default, param), where)


def _checked(spec: FieldSpec, where: str) -> FieldSpec:
    """``spec``, once its limits fit its type and pydantic can carry it."""
    base = without_none(spec.type)
    limits = set(spec.param.constraints)
    if spec.shape == "table" and limits:
        raise NodeDeclarationError(f"{where}{spec.name}: a table takes no limits", code="bound_invalid")
    numeric = base in (int, float, datetime.date, datetime.datetime) or (isinstance(base, type) and issubclass(base, (int, float)))
    sized = base in (str, bytes) or (isinstance(base, type) and issubclass(base, (str, bytes)))
    if (limits & _NUMERIC and not numeric) or (limits & _SIZED and not sized):
        raise NodeDeclarationError(f"{where}{spec.name}: {sorted(limits)} can't apply to {getattr(base, '__name__', base)}", code="bound_invalid")
    if spec.shape == "table":
        return spec
    try:
        spec.adapter  # noqa: B018
    except Exception as error:  # noqa: BLE001  pydantic can't build a schema for this type
        raise NodeDeclarationError(f"{where}{spec.name}: conductor can't carry {spec.type!r}: {error}", code="type_not_carried") from None
    if not spec.required and spec.default is not None and spec.default is not SKIPPED:
        try:
            spec.adapter.validate_python(spec.default)
        except Exception as error:  # noqa: BLE001  pydantic's own error says why
            raise NodeDeclarationError(f"{where}{spec.name}: the default {spec.default!r} is not a {getattr(base, '__name__', base)}: {error}", code="default_invalid") from None
    return spec


_BUILTINS = (bool, int, float, str, bytes, datetime.datetime, datetime.date)


def _carried(hint: Any) -> Any:
    """``hint`` as pydantic reads it: a host's subclass of a builtin is checked as the builtin, then made one."""
    if get_origin(hint) in (Union, types.UnionType):
        return Union[tuple(_carried(a) for a in get_args(hint))]  # noqa: UP007
    if isinstance(hint, type) and hint not in _BUILTINS and not hasattr(hint, "__get_pydantic_core_schema__"):
        base = next((b for b in _BUILTINS if issubclass(hint, b)), None)
        if base is not None:
            return Annotated[base, AfterValidator(hint)]
    return hint


def carried_list(hint: Any) -> TypeAdapter[Any]:
    """Checks a list of ``hint``, as a caller gives many values on one input, and writes it as JSON."""
    return TypeAdapter(_list_of(_carried(hint)), config=_FLOATS_AS_TEXT)


#: A float that isn't finite is written as the text ``"NaN"``, ``"Infinity"`` or ``"-Infinity"``, which reads back.
_FLOATS_AS_TEXT = ConfigDict(ser_json_inf_nan="strings")


def _list_of(item: Any) -> Any:
    """``list[item]`` for an ``item`` known only at run time."""
    return types.GenericAlias(list, (item,))


def _columns_of(row: Any) -> dict[str, Any] | None:
    if row is Any or row is None:
        return None
    return typing.get_type_hints(row)


def _is_optional(hint: Any) -> bool:
    return get_origin(hint) in (Union, types.UnionType) and type(None) in get_args(hint)


def without_none(hint: Any) -> Any:
    """``hint`` with ``| None`` taken off."""
    if get_origin(hint) in (Union, types.UnionType):
        rest = [a for a in get_args(hint) if a is not type(None)]
        return rest[0] if len(rest) == 1 else Union[tuple(rest)]  # noqa: UP007
    return hint


def accepts(target: Any, source: Any) -> bool:
    """Whether a value of type ``source`` may land on an input of type ``target``."""
    if target is Any or source is Any or target == source:
        return True
    targets = _options(target)
    return all(any(_one_accepts(t, s) for t in targets) for s in _options(source) if s is not type(None))


def _options(hint: Any) -> list[Any]:
    return list(get_args(hint)) if get_origin(hint) in (Union, types.UnionType) else [hint]


def _one_accepts(target: Any, source: Any) -> bool:
    if target is Any or target == source:
        return True
    if get_origin(source) is Literal:
        return all(_one_accepts(target, type(v)) for v in get_args(source))
    if isinstance(target, type) and isinstance(source, type):
        own = getattr(target, "accepts", None)  # a host's type may widen what lands on it
        return issubclass(source, target) or (target is float and issubclass(source, int)) or (callable(own) and bool(own(source)))
    return False


def type_id(hint: Any) -> str:
    """The name an editor and a stored graph know a type by: its ``id`` if it declares one, else its name."""
    own = getattr(hint, "id", None)
    return own if isinstance(own, str) else getattr(hint, "__name__", None) or str(hint)


def type_title(hint: Any) -> str:
    """What a person reads for a type: its ``display_title`` if it declares one, else its name."""
    own = getattr(hint, "display_title", None)
    return own if isinstance(own, str) else getattr(hint, "__name__", None) or str(hint)
