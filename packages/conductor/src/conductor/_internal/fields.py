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

import builtins
import dataclasses
import datetime
import functools
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Literal,
    TypedDict,
    TypeVar,
    Union,
    Unpack,
    get_args,
    get_origin,
    overload,
)

import polars as pl
from pydantic import (
    AfterValidator,
    BeforeValidator,
    ConfigDict,
    JsonValue,
    TypeAdapter,
    ValidationError,
)
from pydantic import Field as PydanticField
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic_core import PydanticCustomError, to_jsonable_python

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import NodeDeclarationError, value_free_errors

if TYPE_CHECKING:
    from conductor.widgets import Widget


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


#: The limits ``Param`` takes, by the words pydantic's ``Field`` uses, and the
#: types each kind applies to.
_NUMERIC = frozenset({"ge", "gt", "le", "lt", "multiple_of"})
_SIZED = frozenset({"min_length", "max_length", "pattern"})


class Limits(TypedDict, total=False):
    """The limits ``Param`` takes, by pydantic's words: a number's or a date's bounds, a text's length and pattern."""

    ge: float | datetime.date
    gt: float | datetime.date
    le: float | datetime.date
    lt: float | datetime.date
    multiple_of: float
    min_length: int
    max_length: int
    pattern: str


@dataclass(frozen=True, init=False)
class Param:
    """What a person reads about a field, and the limits a value must keep, inside ``Annotated``::

        times: Annotated[int, Param(title="Times", ge=0)] = 1

    ``title`` falls back to the name in everyday words. ``widget`` is the
    control a person edits the field with, anything with a JSON form (a
    ``conductor.widgets`` control, say), carried to the host and never read
    here; on an output it is the control a person answers it with. Outputs of one node that share an ``exclusive_group`` are
    alternatives: on each row the node gives a value on one of them and
    ``SKIPPED`` on the rest. An input that is a ``setting`` (a model, a
    temperature) is the graph author's to set, by a value in the record or its
    own default; it is never one of the graph's inputs, so no caller gives it.
    The limits are pydantic's: ``ge``, ``gt``, ``le``, ``lt``,
    ``multiple_of``, ``min_length``, ``max_length``, ``pattern``.
    """

    title: str | None = None
    description: str | None = None
    widget: Widget | Mapping[str, JsonValue] | None = None
    exclusive_group: str | None = None
    setting: bool = False
    constraints: dict[str, Any] = field(default_factory=dict)

    def __init__(
        self, *, title: str | None = None, description: str | None = None, widget: Widget | Mapping[str, JsonValue] | None = None,
        exclusive_group: str | None = None, setting: bool = False, **constraints: Unpack[Limits],
    ) -> None:
        unknown = sorted(constraints.keys() - _NUMERIC - _SIZED)  # a caller a type checker didn't see
        if unknown:
            raise NodeDeclarationError(
                f"Param has no argument {unknown[0]!r}; it takes title, description, widget, exclusive_group, setting and the limits {sorted(_NUMERIC | _SIZED)}",
                code="bound_invalid",
            )
        if widget is not None:
            try:
                to_jsonable_python(widget)
            except Exception:  # noqa: BLE001  a value with no JSON form, such as a class passed bare
                raise NodeDeclarationError(f"Param's widget {widget!r} has no JSON form; give an instance, such as Textarea()", code="widget_not_json") from None
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "widget", widget)
        object.__setattr__(self, "exclusive_group", exclusive_group)
        object.__setattr__(self, "setting", setting)
        object.__setattr__(self, "constraints", constraints)


@dataclass(frozen=True)
class Named:
    """The id and title of a type that has no class of its own to carry them, inside ``Annotated``::

        YesNo = Annotated[bool, Named("yes-no", title="Ja/Nej")]

    A field typed ``YesNo`` is a ``bool`` to a node, a type checker and every check, and an editor and a stored graph
    know its type as ``yes-no``, titled ``Ja/Nej``; a host's own class says the same with ``id`` and ``display_title``.
    """

    id: str
    title: str | None = None


def _unnamed(hint: Any) -> Any:
    """``hint`` without its ``Named``: the type every check reads."""
    return get_args(hint)[0] if _marked(hint, Named) else hint


class _ManyMarker:
    pass


@dataclass(frozen=True)
class _TableMarker:
    row: Any


class _FromRunMarker:
    pass


_T = TypeVar("_T")
_Record = TypeVar("_Record", bound="FieldDescription")

#: ``FromRun[X]``: a value the host hands the run by its type (a session, who is running), never an input. It takes
#: no edge, is not stored in the record, and is no part of what makes a row's result reusable. The run is given one
#: with ``from_run={X: value}``. To a type checker the parameter is an ``X``.
FromRun = Annotated[_T, _FromRunMarker()]

#: ``Series[X]``: many values of ``X``, one per row; a plain ``list[X]`` to the node and to a type checker.
Series = Annotated[list[_T], _ManyMarker()]


class Table:
    """``Table[R]``: a whole table whose columns the ``TypedDict`` ``R`` names; a ``polars.DataFrame`` to the node.

    A bare ``Table`` is a table of any columns. It is a type only: a node returns a ``polars.DataFrame``.
    """

    def __new__(cls, *args: Any, **kwargs: Any) -> Table:
        raise TypeError("Table is a type for a signature (Table[R]); a node takes and returns a polars.DataFrame")

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
    def plain_classes(self) -> tuple[type, ...]:
        """The classes a value of exactly which needs no reading: the type's own, ``None`` for a blank-taking one; none under limits or ``Any``."""
        options = _options(self.type)
        if self.param.constraints or any(option is Any for option in options):
            return ()
        return tuple(option for option in options if isinstance(option, type) and not typing.is_typeddict(option))

    @functools.cached_property
    def item_adapter(self) -> TypeAdapter[Any]:
        """Checks one item of a ``Series``, or the one value of any other field, held to its limits."""
        return dataclasses.replace(self, shape="one").adapter if self.shape == "many" else self.adapter

    @functools.cached_property
    def column_adapter(self) -> TypeAdapter[Any]:
        """Checks and writes one output's values over many units, blanks included."""
        carried = _carried(self.type)
        one: Any = _list_of(carried | None) if self.shape == "many" else carried  # a skipped item is written as null
        return TypeAdapter(_list_of(one | None), config=_FLOATS_AS_TEXT)

    @functools.cached_property
    def column_adapters(self) -> dict[str, TypeAdapter[Any]]:
        """For a table, a check and writer of each column its row type names."""
        return {column: carried_list(Union[hint, None]) for column, hint in (self.columns or {}).items()}  # noqa: UP007  a cell may be blank

    @overload
    def describe(self, record: None = None, /, **changed: Any) -> FieldDescription: ...

    @overload
    def describe(self, record: builtins.type[_Record], /, **changed: Any) -> _Record: ...  # the field named type hides the builtin

    def describe(self, record: builtins.type[FieldDescription] | None = None, /, **changed: Any) -> FieldDescription:
        """The field as an editor reads it: its shape, its type's ``id`` and JSON schema, and what a person reads about
        it; as a ``record`` that adds to it (a graph's input, an input placed in a graph), with what ``changed`` says."""
        return (record or FieldDescription).model_validate({**self.as_json(), **changed})

    def as_json(self) -> dict[str, Any]:
        """``describe()`` as JSON, with only the keys the field has: what a question is made from."""
        out: dict[str, Any] = {"name": self.name, "title": self.title, "shape": self.shape, "required": self.required}
        if self.shape != "table":
            out["type"] = type_id(without_none(self.type))
            out["schema"] = _schema(self.type)
        else:
            out["columns"] = {c: _schema(t) for c, t in (self.columns or {}).items()}
        if not self.required:
            out["default"] = to_jsonable_python(self.default, fallback=lambda _: None)
        offered = literal_values(self.type) if self.shape != "table" else None
        if offered is not None:
            out["choices"] = [{"id": to_jsonable_python(value), "title": str(value)} for value in offered]
        for key in ("description", "widget", "exclusive_group"):
            if getattr(self.param, key) is not None:
                out[key] = to_jsonable_python(getattr(self.param, key), fallback=str)
        if self.param.setting:
            out["setting"] = True
        if self.param.constraints:
            out["constraints"] = to_jsonable_python(self.param.constraints, fallback=str)
        return out


class FieldDescription(ConductorModel):
    """An input or output as an editor reads it, and a question a waiting row asks.

    ``type`` is its type's id and ``schema`` its JSON schema, for one value or a ``Series`` item; a table has
    ``columns`` instead, each column's schema. ``default`` is what it holds when it is not ``required``.
    ``choices`` are ``{"id", "title"}`` records, from a ``Literal`` or, on a question, the ``Ask`` (a host's
    ``Choice`` subclass adds its own keys). The rest is the ``Param``'s.
    """

    model_config = ConfigDict(frozen=True, extra="forbid", validate_by_name=True, validate_by_alias=True, serialize_by_alias=True)

    name: str
    title: str
    shape: Literal["one", "many", "table"]
    required: bool
    type: str | None = None
    json_schema: dict[str, Any] | None = PydanticField(default=None, alias="schema")
    columns: dict[str, dict[str, Any]] | None = None
    default: Any = None
    choices: list[dict[str, Any]] | None = None
    description: str | None = None
    widget: dict[str, Any] | None = None
    exclusive_group: str | None = None
    setting: bool = False
    constraints: dict[str, Any] | None = None


def literal_values(hint: Any) -> tuple[Any, ...] | None:
    """The values a ``Literal`` type (``| None`` aside) allows, the one place a field's choices are declared; ``None`` for any other type."""
    item = without_none(hint)
    return get_args(item) if get_origin(item) is Literal else None


def _schema(hint: Any) -> dict[str, Any]:
    """The JSON schema of a type; ``{}`` for one that writes none, which an editor offers no control for."""
    try:
        return TypeAdapter(_carried(hint)).json_schema()
    except PydanticInvalidForJsonSchema:
        return {}


def read_field(name: str, hint: Any, default: Any = REQUIRED, where: str = "") -> FieldSpec:
    """Read one annotation into a ``FieldSpec``; a hint conductor can't carry raises ``NodeDeclarationError``.

    ``Annotated[X, Param(...)] | None`` is read as ``X | None`` with that ``Param``, and a bare ``Series`` as many
    values of any type.
    """
    or_none = False
    if get_origin(hint) in (Union, types.UnionType):
        members = [a for a in get_args(hint) if a is not type(None)]
        if type(None) in get_args(hint) and len(members) == 1 and get_origin(members[0]) is Annotated:
            hint, or_none = members[0], True
    param = Param()
    if get_origin(hint) is Annotated:
        base, *meta = get_args(hint)
        unknown = [m for m in meta if not isinstance(m, (Param, Named, _ManyMarker, _TableMarker, _FromRunMarker))]
        if unknown:
            raise NodeDeclarationError(
                f"{where}{name}: Annotated holds {unknown[0]!r}, which conductor doesn't read; a widget, a title or a limit goes on Param(...)",
                code="annotated_extra_unknown",
            )
        params = [m for m in meta if isinstance(m, Param)]
        param = params[0] if params else param
        markers = [m for m in meta if isinstance(m, (_ManyMarker, _TableMarker))]
        if markers and isinstance(markers[0], _ManyMarker):
            item = get_args(base)[0]
            if isinstance(item, TypeVar):  # a bare Series
                item = Any
            if item is pl.DataFrame or item is Table or _marked(item, _TableMarker):
                raise NodeDeclarationError(f"{where}{name}: a Series of tables; a table already holds many rows, so take or return one table", code="series_of_tables")
            if _marked(item, _ManyMarker):
                raise NodeDeclarationError(f"{where}{name}: a Series of Series; rows are never nested in a field, so return one Series", code="nested_series")
            return _checked(FieldSpec(name, "many", item, _is_optional(item), default, param), where)
        if markers and isinstance(markers[0], _TableMarker):
            return _checked(FieldSpec(name, "table", pl.DataFrame, False, default, param, _columns_of(markers[0].row)), where)
        names = [m for m in meta if isinstance(m, Named)]
        hint = Annotated[base, names[0]] if names else base
    if hint is pl.DataFrame or hint is Table:
        return _checked(FieldSpec(name, "table", pl.DataFrame, False, default, param, None), where)
    if or_none:
        hint = Union[hint, None]  # noqa: UP007
    return _checked(FieldSpec(name, "one", hint, _is_optional(hint), default, param), where)


def _marked(hint: Any, marker: type) -> bool:
    return get_origin(hint) is Annotated and any(isinstance(m, marker) for m in get_args(hint)[1:])


def _checked(spec: FieldSpec, where: str) -> FieldSpec:
    """``spec``, once its limits fit its type and pydantic can carry it; a default it declares is made its type."""
    base = _unnamed(without_none(spec.type))
    limits = set(spec.param.constraints)
    if spec.shape == "table" and limits:
        raise NodeDeclarationError(f"{where}{spec.name}: a table takes no limits", code="bound_invalid")
    numeric = base in (int, float, datetime.date, datetime.datetime) or (isinstance(base, type) and issubclass(base, (int, float)))
    sized = base in (str, bytes) or (isinstance(base, type) and issubclass(base, (str, bytes)))
    if (limits & _NUMERIC and not numeric) or (limits & _SIZED and not sized):
        raise NodeDeclarationError(f"{where}{spec.name}: {sorted(limits)} can't apply to {getattr(base, '__name__', base)}; ge, gt, le, lt and multiple_of fit numbers and dates, min_length, max_length and pattern texts", code="bound_invalid")
    if spec.shape == "table":
        return spec
    try:
        spec.adapter  # noqa: B018
    except Exception as error:  # noqa: BLE001  pydantic can't build a schema for this type
        if limits:
            raise NodeDeclarationError(f"{where}{spec.name}: the limits {spec.param.constraints!r} don't fit {getattr(base, '__name__', base)}: {error}", code="bound_invalid") from None
        raise NodeDeclarationError(
            f"{where}{spec.name}: conductor can't check or write {getattr(base, '__name__', base)} as JSON; "
            "use a type pydantic knows, such as a pydantic model or a dataclass",
            code="type_not_carried",
        ) from None
    if getattr(spec.param.widget, "choices", None) and literal_values(spec.type) is not None:
        raise NodeDeclarationError(
            f"{where}{spec.name}: the choices are the Literal's values, so the widget declares none; write Dropdown()", code="choices_declared_twice",
        )
    if spec.default is None and spec.shape == "one" and not spec.optional:
        raise NodeDeclarationError(f"{where}{spec.name}: the default None is a blank; write its type with | None", code="default_invalid")
    if not spec.required and spec.default is not None and spec.default is not SKIPPED:
        try:
            made = spec.adapter.validate_python(spec.default)
        except Exception as error:  # noqa: BLE001  pydantic's own error says why
            raise NodeDeclarationError(f"{where}{spec.name}: the default {spec.default!r} is not a {getattr(base, '__name__', base)}: {error}", code="default_invalid") from None
        return dataclasses.replace(spec, default=made)
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


#: The classes whose values are plain in JSON: no keys to order, and read back by their type as they were.
SCALARS = frozenset({str, int, float, bool, type(None), datetime.date, datetime.datetime, datetime.time, datetime.timedelta})


def carried_list(hint: Any) -> TypeAdapter[Any]:
    """Checks a list of ``hint``, as a caller gives many values on one input, and writes it as JSON."""
    return TypeAdapter(_list_of(_carried(hint)), config=_FLOATS_AS_TEXT)


#: A float that isn't finite is written as the text ``"NaN"``, ``"Infinity"`` or ``"-Infinity"``, which reads back.
_FLOATS_AS_TEXT = ConfigDict(ser_json_inf_nan="strings")


def _list_of(item: Any) -> Any:
    """``list[item]`` for an ``item`` known only at run time; a set is refused, as its order differs between processes."""
    return Annotated[types.GenericAlias(list, (item,)), BeforeValidator(_not_a_set)]


def _not_a_set(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        raise PydanticCustomError("set_has_no_order", "a set has no order, so its rows would come in another order in another process; give a list")
    return value


def _columns_of(row: Any) -> dict[str, Any] | None:
    if row is Any or row is None:
        return None
    return typing.get_type_hints(row)


def _is_optional(hint: Any) -> bool:
    """Whether a blank may arrive: ``X | None``, or ``Any``, which says nothing against one."""
    return hint is Any or (get_origin(hint) in (Union, types.UnionType) and type(None) in get_args(hint))


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
    """The types a hint allows, each as every check reads it (a ``Named`` one as its type)."""
    return [_unnamed(option) for option in get_args(hint)] if get_origin(hint) in (Union, types.UnionType) else [_unnamed(hint)]


def _one_accepts(target: Any, source: Any) -> bool:
    if target is Any or target == source:
        return True
    if get_origin(source) is Literal:
        return all(_one_accepts(target, type(v)) for v in get_args(source))
    if isinstance(target, type) and isinstance(source, type):
        own = getattr(target, "accepts", None)  # a host's type may widen what lands on it
        if issubclass(source, bool) and target in (int, float):
            return False  # a yes or no is never a number, though Python counts True as 1
        return issubclass(source, target) or (target is float and issubclass(source, int)) or (callable(own) and bool(own(source)))
    return False


def type_id(hint: Any) -> str:
    """The name an editor and a stored graph know a type by: its ``id`` if it declares one, else its name.

    A ``Literal`` is known by the type of its values (``Literal["low", "high"]`` is ``str``), its values in the schema.
    """
    if get_origin(hint) is Literal:
        return " | ".join(dict.fromkeys(type_id(type(value)) for value in get_args(hint)))
    if get_origin(hint) in (Union, types.UnionType):  # each member by its own name: a host's Text | None is "Text | None"
        return " | ".join("None" if member is type(None) else type_id(member) for member in get_args(hint))
    if _marked(hint, Named):
        return next(m for m in get_args(hint)[1:] if isinstance(m, Named)).id
    own = getattr(hint, "id", None)
    return own if isinstance(own, str) else getattr(hint, "__name__", None) or str(hint)


def type_title(hint: Any) -> str:
    """What a person reads for a type: its ``display_title`` (or ``Named`` title) if it declares one, else its name."""
    if _marked(hint, Named):
        named = next(m for m in get_args(hint)[1:] if isinstance(m, Named))
        return named.title or named.id
    own = getattr(hint, "display_title", None)
    return own if isinstance(own, str) else getattr(hint, "__name__", None) or str(hint)


# -- reading a value as its field ---------------------------------------------------------------


class Refused(Exception):
    """A value its field can't take: ``reason`` says why and quotes nothing of it; ``ctx`` has pydantic's ``errors``
    without the value, the field's ``name`` and, for a table, the ``column``. ``code`` is the problem it is where it
    was met: ``invalid_output`` for a node's return, ``invalid_input`` for what an edge brought."""

    def __init__(self, reason: str, *, code: str = "", **ctx: Any) -> None:
        super().__init__(reason)
        self.reason, self.code, self.ctx = reason, code, ctx


def already(target: Any, source: Any) -> bool:
    """Whether every value of type ``source`` already is one ``target`` takes as it is, with nothing to read."""
    if target is Any or target == source:
        return True
    targets = [t for t in _options(target) if isinstance(t, type) and not typing.is_typeddict(t)]
    sources = [s for s in _options(source) if s is not type(None)]
    return all(isinstance(s, type) and not typing.is_typeddict(s) and any(issubclass(s, t) for t in targets) for s in sources)


def read_value(spec: FieldSpec, value: Any, *, strict: bool) -> Any:
    """``value`` as one value of ``spec``: as it is when already of its class, else read by its type, ``strict`` as
    pydantic's strict mode is. A ``SKIPPED`` item of a ``Series`` is a gap, read around."""
    if value is SKIPPED:
        return value
    if spec.shape == "table":
        return read_table(spec, value, strict=strict)
    if spec.shape == "one" and type(value) in spec.plain_classes:
        return value
    if spec.shape == "many" and value is None and spec.default is None:
        return None  # a Series that may be None
    if spec.shape == "many" and isinstance(value, (tuple, range)):
        value = list(value)  # any sequence a node returns is its rows
    elif spec.shape == "many" and isinstance(value, pl.Series):
        value = value.to_list()  # a column of a frame is its values, as a frame is a Table
    try:
        return spec.adapter.validate_python(value, strict=strict)
    except ValidationError as error:
        if spec.shape == "many" and isinstance(value, list) and any(item is SKIPPED for item in value):
            items = iter(read_value(spec, [item for item in value if item is not SKIPPED], strict=strict))
            return [SKIPPED if item is SKIPPED else next(items) for item in value]
        raise _refused(error) from None


def read_table(spec: FieldSpec, value: Any, *, strict: bool, given: bool = False) -> pl.DataFrame:
    """``value`` as a table of ``spec``: every column its row type names, each read by the column's type unless its
    dtype already holds that type; a missing one is refused naming it.

    A table ``given`` (typed in, or a caller's) may also be a list of records and keeps only the named columns, as
    an edge into the input would; a table a node returns or an edge brings keeps its other columns too.
    """
    if given and not isinstance(value, pl.DataFrame):
        try:
            records = _RECORDS.validate_python(value)
        except ValidationError:
            raise Refused("a table is a frame or a list of records, one object per row") from None
        names = list(spec.columns) if spec.columns is not None else list(dict.fromkeys(k for r in records for k in r))
        present = {k for r in records for k in r}
        try:
            value = pl.DataFrame({c: [r.get(c) for r in records] for c in names if c in present or not records}, strict=False)
        except Exception:  # noqa: BLE001  polars' own account may quote a cell
            raise Refused("the records don't make a table") from None
    if not isinstance(value, pl.DataFrame):
        raise Refused(f"a table is a polars DataFrame, not a {type(value).__name__}")
    if spec.columns is None:
        return value
    missing = [c for c in spec.columns if c not in value.columns and (value.height or not given)]
    if missing:
        raise Refused(f"the column '{missing[0]}' its row type declares is missing", column=missing[0])
    read = {}
    for column, hint in spec.columns.items():
        if column not in value.columns:
            read[column] = [None] * value.height
        elif not already(hint, _PYTHON_OF.get(value.schema[column].base_type(), object)):
            try:
                read[column] = spec.column_adapters[column].validate_python(value[column].to_list(), strict=strict)
            except ValidationError as error:
                refused = _refused(error)
                raise Refused(f"column '{column}': {refused.reason}", column=column, **refused.ctx) from None
    if given:
        return pl.DataFrame({c: read[c] if c in read else value[c] for c in spec.columns}, strict=False)
    return value.with_columns(pl.Series(c, cells, strict=False) for c, cells in read.items()) if read else value


def _refused(error: ValidationError) -> Refused:
    errors = value_free_errors(error)
    return Refused(errors[0]["msg"], errors=errors)


_RECORDS = TypeAdapter(list[dict[str, Any]])

#: The type a polars dtype's values already are, so a column of it needs no reading; ``Null`` holds only blanks.
_PYTHON_OF: dict[Any, Any] = {
    **dict.fromkeys((pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64), int),
    pl.Float32: float, pl.Float64: float, pl.String: str, pl.Boolean: bool, pl.Date: datetime.date, pl.Datetime: datetime.datetime,
    pl.Null: type(None),
}
