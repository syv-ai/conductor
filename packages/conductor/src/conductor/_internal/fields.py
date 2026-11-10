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
import copy
import dataclasses
import datetime
import decimal
import functools
import inspect
import types
import typing
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Literal,
    TypeVar,
    Union,
    get_args,
    get_origin,
    overload,
)

import polars as pl
from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    TypeAdapter,
    ValidateAs,
    ValidationError,
)
from pydantic import Field as PydanticField
from pydantic.errors import PydanticInvalidForJsonSchema
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import (
    CoreSchema,
    PydanticCustomError,
    PydanticSerializationError,
    core_schema,
    to_json,
    to_jsonable_python,
)

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


#: The limits ``Param`` takes, by the words pydantic's ``Field`` uses.
_LIMITS = ("ge", "gt", "le", "lt", "multiple_of", "min_length", "max_length", "pattern")

#: What a bound may be: a number's, a date's, a time's or a duration's.
_Bound = float | decimal.Decimal | datetime.date | datetime.time | datetime.timedelta


@dataclass(frozen=True, kw_only=True)
class Param:
    """What a person reads about a field, and the limits a value must keep, inside ``Annotated``::

        times: Annotated[int, Param(title="Times", ge=0)] = 1

    A field's record carries all of it but the limits, which hold wherever pydantic reads the type.

    Attributes:
        title: What a person reads for the field; its name in everyday words when not given.
        description: A longer text about the field, for its help.
        widget: The ``conductor.widgets`` control a person edits the field with (on an output, answers it with); a
            host's own is a subclass of ``Widget``. Carried to the host and never read here.
        exclusive_group: Outputs of one node that share it are alternatives: on each row the node gives a value on
            one of them and ``SKIPPED`` on the rest.
        choices: The values the field takes, each a value or a ``Choice`` to give it a title. On a ``Literal`` field
            they title its values, one each; on any other they limit what it takes. A ``Dropdown`` draws them.
        ge: The least value taken. ``gt``, ``le`` and ``lt`` are the other bounds, and ``multiple_of`` a step; each
            fits a number, a date, a time or a duration.
        min_length: The fewest letters, items or keys taken; ``max_length`` the most. Each fits a text, bytes or a
            collection.
        pattern: The regular expression a text matches.
    """

    title: str | None = None
    description: str | None = None
    widget: Widget | None = None
    exclusive_group: str | None = None
    choices: Sequence[Any] | None = None
    ge: _Bound | None = None
    gt: _Bound | None = None
    le: _Bound | None = None
    lt: _Bound | None = None
    multiple_of: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None

    def __post_init__(self) -> None:
        if self.choices is not None:
            object.__setattr__(self, "choices", tuple(self.choices))

    @functools.cached_property
    def constraints(self) -> dict[str, Any]:
        """The limits set, by pydantic's words, as ``pydantic.Field`` takes them."""
        return {key: value for key in _LIMITS if (value := getattr(self, key)) is not None}

    @property
    def holds_a_value(self) -> bool:
        """Whether a value of the field's class must still be read: a limit or a choice may refuse it."""
        return bool(self.constraints) or self.choices is not None

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """``source`` checked and held to these limits and choices, so ``Annotated[int, Param(ge=0)]`` is checked alike in any pydantic model.

        A limit pydantic doesn't hold on ``source``, or a choice that isn't a value of it, is refused here, as the
        schema is made: each choice's id is read as an answer is (``read_strictly``) by the limited type.
        """
        unheld = _unheld(source, tuple(self.constraints.items()))
        if unheld:
            raise NodeDeclarationError(
                f"{unheld} can't apply to {getattr(_bare(source), '__name__', source)}, as pydantic holds no such limit on it; take them off, or check the value in the node",
                code="bound_invalid",
            )
        limited = handler(Annotated[source, PydanticField(**self.constraints)] if self.constraints else source)
        if self.choices is None:
            return limited
        reader = TypeAdapter(Annotated[source, PydanticField(**self.constraints)] if self.constraints else source)
        fault, ids = "", []
        try:
            ids = [read_strictly(reader, choice_record(choice)["id"]) for choice in self.choices]
        except Refused as refused:
            fault = f"a choice is not a value of the field's type: {refused.reason}"
        offered = literal_values(source)
        if not fault and (len({repr(i) for i in ids}) != len(ids) or not ids):
            fault = "the choices hold an id twice, or none"
        elif not fault and offered is not None and {repr(i) for i in ids} != {repr(v) for v in offered}:
            fault = f"on a Literal the choices title its values, one each: {list(offered)}"
        if fault:
            raise NodeDeclarationError(f"{fault}; give each choice once, as a value of the field's type", code="choices_invalid")
        if offered is not None:  # a Literal's choices title its values
            return limited

        def one_of_the_choices(value: Any) -> Any:
            if value is not None and value not in ids:
                raise PydanticCustomError("not_a_choice", "the value is not one of the choices")
            return value

        return core_schema.no_info_after_validator_function(one_of_the_choices, limited)

    def __get_pydantic_json_schema__(self, schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        """The JSON schema of the checked type, with this ``title`` and ``description`` where a model's field would carry them."""
        written = handler(schema)
        for key in ("title", "description"):
            if getattr(self, key) is not None:
                written[key] = getattr(self, key)
        return written


class Choice(ConductorModel):
    """One option a person may pick: the ``id`` the answer holds and the ``title`` they read: ``Choice("ida", "Ida B.")``.

    A host subclasses it to carry more about an option; the fields it adds
    reach the question, and a dropdown, as written.
    """

    id: Any
    title: str

    def __init__(self, id: Any, title: str, **more: Any) -> None:  # noqa: A002  the word a host's option has
        fields: dict[str, Any] = {"id": id, "title": title, **more}  # the base's own signature names no fields
        super().__init__(**fields)


def choice_record(given: Any, *, mode: Literal["python", "json"] = "python") -> dict[str, Any]:
    """One choice as a record: a ``Choice`` (or a host's subclass of it) as its fields, any other value as itself titled by its text."""
    if isinstance(given, Choice):
        return given.model_dump(mode=mode)
    return {"id": given, "title": str(given)}


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
    """One input or output: its shape, the type of one value, and how a value is checked.

    ``optional`` says a blank (``None``) may arrive in place of the field's value: of one value, or of a ``Series``'
    whole list (``Series[X] | None``). Whether a ``Series``' items may be blank is its item type's own ``| None``.
    """

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
    def _limited(self) -> Any:
        """The type pydantic reads, held to the field's limits: what a value and each of its choices must be."""
        if self.param.constraints:  # a field's title and description are its record's own, not repeated in its schema
            return Annotated[_carried(self.type), PydanticField(**self.param.constraints)]
        return _carried(self.type)

    @functools.cached_property
    def _one(self) -> Any:
        """The type one value is checked by: the type pydantic reads, its limits, its choices, and no ``bool`` on a number.

        Read strictly, pydantic refuses a ``bool`` on an ``int`` or ``float`` itself, but not on a ``Literal`` of
        numbers; ``_lax_one`` adds the refusal for a value read laxly.
        """
        one: Any = Annotated[_carried(self.type), self.param] if self.param.holds_a_value else _carried(self.type)
        if self.takes_a_number and self._literal_option:  # True == 1, strict or not
            one = Annotated[one, BeforeValidator(_not_a_bool)]
        return one

    @functools.cached_property
    def _literal_option(self) -> bool:
        return any(get_origin(option) is Literal for option in _options(self.type))

    @functools.cached_property
    def _lax_one(self) -> Any:
        """``_one`` for a value read laxly (given, typed in, brought by an edge), where pydantic's int and float take True as 1."""
        if self.takes_a_number and not self._literal_option:
            return Annotated[self._one, BeforeValidator(_not_a_bool)]
        return self._one

    def _of_field(self, one: Any) -> TypeAdapter[Any]:
        if self.shape != "many":
            return TypeAdapter(one)
        many: Any = Union[_list_of(one), None] if self.optional else _list_of(one)  # noqa: UP007
        return TypeAdapter(many)

    @functools.cached_property
    def adapter(self) -> TypeAdapter[Any]:
        """Checks one value of this field (a ``Series``' list of them), held to its limits and choices, and writes it; lax unless called with ``strict=True``."""
        return self._of_field(self._one)

    @functools.cached_property
    def lax_adapter(self) -> TypeAdapter[Any]:
        """Checks one value of this field (a ``Series``' list of them) read laxly: given, typed in, or brought by an edge."""
        return self._of_field(self._lax_one)

    @functools.cached_property
    def list_adapter(self) -> TypeAdapter[Any]:
        """Checks a list given for a field that takes one value, each run as its own row, and writes it as JSON."""
        return TypeAdapter(_list_of(self._lax_one), config=_FLOATS_AS_TEXT)

    @functools.cached_property
    def plain(self) -> bool:
        """Whether its values are of plain classes (``SCALARS``): JSON with no keys to order, read back by the type as they were.

        Not when it takes both a ``float`` and a ``str``: a float that isn't finite is written as text, and only its
        mark tells it from the text it reads as.
        """
        classes = set(self.plain_classes)
        return bool(classes) and classes <= SCALARS and not {float, str} <= classes

    @functools.cached_property
    def per_item(self) -> bool:
        """Whether a list given here is many values, a row per item: the field takes one value, of a type that takes no list as one."""
        return self.shape == "one" and not _takes_a_list(self.type)

    @functools.cached_property
    def takes_a_number(self) -> bool:
        """Whether a ``bool`` must not land here (``refuses_a_bool``)."""
        return refuses_a_bool(self.type)

    @functools.cached_property
    def plain_classes(self) -> tuple[type, ...]:
        """The classes a value of exactly which needs no reading: the type's own, ``None`` for a blank-taking one.

        None when any option is not a class (``list[int]``, a ``Literal``, a ``TypedDict``, ``Any``) or the field has
        limits or choices: such a value is always read by the type.
        """
        options = _options(self.type)
        if self.param.holds_a_value:
            return ()
        if not all(isinstance(option, type) and option is not Any and not typing.is_typeddict(option) for option in options):
            return ()
        return tuple(options)

    @functools.cached_property
    def column_adapter(self) -> TypeAdapter[Any]:
        """Checks and writes one output's values over many units, blanks included."""
        carried = _carried(self.type)
        one: Any = _column_of(carried | None) if self.shape == "many" else carried  # a skipped item is written as null
        return TypeAdapter(_column_of(one | None), config=_FLOATS_AS_TEXT)

    @functools.cached_property
    def column_adapters(self) -> dict[str, TypeAdapter[Any]]:
        """For a table, a check and writer of each column its row type names."""
        return {column: TypeAdapter(_column_of(_carried(Union[_no_bool_on_a_number(hint), None])), config=_FLOATS_AS_TEXT) for column, hint in (self.columns or {}).items()}  # noqa: UP007  a cell may be blank

    @overload
    def describe(self, record: None = None, /, **changed: Any) -> FieldDescription: ...

    @overload
    def describe(self, record: builtins.type[_Record], /, **changed: Any) -> _Record: ...  # the field named type hides the builtin

    def describe(self, record: builtins.type[FieldDescription] | None = None, /, **changed: Any) -> FieldDescription:
        """The field as an editor reads it: its shape, its type's ``id`` and JSON schema, and what a person reads about
        it; as a ``record`` that adds to it (a graph's input, an input placed in a graph), with what ``changed`` says."""
        return (record or FieldDescription).model_validate({**self.as_json(), **changed})

    @functools.cached_property
    def _schemas(self) -> dict[str, Any]:
        """The JSON schema of one value, or of each column of a table, made once: pydantic builds one in tens of microseconds."""
        if self.shape == "table":
            return {"columns": {c: _schema(t) for c, t in (self.columns or {}).items()}}
        return {"schema": _schema(self._limited)}

    def as_json(self) -> dict[str, Any]:
        """``describe()`` as JSON, with only the keys the field has: what a question is made from."""
        out: dict[str, Any] = {"name": self.name, "title": self.title, "shape": self.shape, "required": self.required}
        if self.shape != "table":
            out["type"] = type_id(without_none(self.type))
        out.update(copy.deepcopy(self._schemas))  # the caller may change what it is given
        if not self.required:
            out["default"] = to_jsonable_python(self.default, fallback=lambda _: None)
        offered = literal_values(self.type) if self.shape != "table" else None
        if self.param.choices is not None:
            out["choices"] = [{**(record := choice_record(choice)), "id": to_jsonable_python(record["id"])} for choice in self.param.choices]
        elif offered is not None:
            out["choices"] = [{"id": to_jsonable_python(value), "title": str(value)} for value in offered]
        for key in ("description", "widget", "exclusive_group"):
            if getattr(self.param, key) is not None:
                out[key] = to_jsonable_python(getattr(self.param, key), fallback=str)
        return out


class FieldDescription(ConductorModel):
    """An input or output as an editor reads it, and a question a waiting row asks.

    ``type`` is its type's id and ``schema`` its JSON schema, for one value or a ``Series`` item; a table has
    ``columns`` instead, each column's schema. ``default`` is what it holds when it is not ``required``.
    ``choices`` are ``{"id", "title"}`` records, from a ``Literal`` or, on a question, the ``Ask`` (a host's
    ``Choice`` subclass adds its own keys). The rest is the ``Param``'s.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_by_name=True, validate_by_alias=True, serialize_by_alias=True, ser_json_inf_nan="strings",
    )

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
    if _holds_from_run(hint):
        raise NodeDeclarationError(
            f"{where}{name}: FromRun is a whole parameter the host always gives, name: FromRun[Session]; anywhere else, a caller would fill it",
            code="from_run_misplaced",
        )
    or_none = False
    if get_origin(hint) in (Union, types.UnionType):
        members = [a for a in get_args(hint) if a is not type(None)]
        if type(None) in get_args(hint) and len(members) == 1 and (get_origin(members[0]) is Annotated or members[0] in (pl.DataFrame, Table)):
            hint, or_none = members[0], True
    param = Param()
    if get_origin(hint) is Annotated:
        base, *meta = get_args(hint)
        unknown = [m for m in meta if not isinstance(m, (Param, _ManyMarker, _TableMarker))]
        if unknown:
            raise NodeDeclarationError(
                f"{where}{name}: Annotated holds {unknown[0]!r}, which conductor doesn't read; a widget, a title or a limit goes on Param(...)",
                code="annotated_extra_unknown",
            )
        params = [m for m in meta if isinstance(m, Param)]
        param = _stacked(params)
        markers = [m for m in meta if isinstance(m, (_ManyMarker, _TableMarker))]
        if markers and isinstance(markers[0], _ManyMarker):
            item = get_args(base)[0]
            if isinstance(item, TypeVar):  # a bare Series
                item = Any
            item, of_item = _without_params(item)  # an item's Param is the field's: its schema is one item's
            param = _stacked([*of_item, *params])
            if item is pl.DataFrame or item is Table or marked(item, _TableMarker):
                raise NodeDeclarationError(f"{where}{name}: a Series of tables; a table already holds many rows, so take or return one table", code="series_of_tables")
            if marked(item, _ManyMarker):
                raise NodeDeclarationError(f"{where}{name}: a Series of Series; rows are never nested in a field, so return one Series", code="nested_series")
            return _checked(FieldSpec(name, "many", item, or_none, default, param), where)
        if markers and isinstance(markers[0], _TableMarker):
            return _checked(FieldSpec(name, "table", pl.DataFrame, or_none, default, param, _columns_of(markers[0].row, name, where)), where)
        hint = base
    if hint is pl.DataFrame or hint is Table:
        return _checked(FieldSpec(name, "table", pl.DataFrame, or_none, default, param, None), where)
    if or_none:
        hint = Union[hint, None]  # noqa: UP007
    return _checked(FieldSpec(name, "one", hint, takes_a_blank(hint), default, param), where)


def _stacked(params: list[Param]) -> Param:
    """The ``Param``s of one ``Annotated`` as one, as pydantic merges stacked ``Field``s: each argument the last one
    given, so ``Annotated[Line, Param(min_length=1)]`` keeps ``Line``'s widget too."""
    if len(params) < 2:
        return params[0] if params else Param()
    merged: dict[str, Any] = {}
    for param in params:
        merged.update({f.name: value for f in dataclasses.fields(param) if (value := getattr(param, f.name)) is not None})
    return Param(**merged)


def _without_params(hint: Any) -> tuple[Any, list[Param]]:
    """``hint`` with its ``Param``s taken off, ``Annotated[X, Param(...)] | None`` as ``X | None``, and those ``Param``s."""
    if get_origin(hint) in (Union, types.UnionType) and type(None) in get_args(hint):
        rest = [a for a in get_args(hint) if a is not type(None)]
        if len(rest) == 1:
            bare, params = _without_params(rest[0])
            return Union[bare, None], params  # noqa: UP007
    if get_origin(hint) is Annotated and all(isinstance(m, Param) for m in get_args(hint)[1:]):
        return get_args(hint)[0], list(get_args(hint)[1:])
    return hint, []


def _holds_from_run(hint: Any) -> bool:
    """Whether a ``FromRun`` mark sits anywhere in ``hint``."""
    if get_origin(hint) is Annotated and any(isinstance(m, _FromRunMarker) for m in get_args(hint)[1:]):
        return True
    return any(_holds_from_run(arg) for arg in get_args(hint))


def marked(hint: Any, marker: type) -> bool:
    """Whether ``hint`` is ``Annotated`` with a ``marker``: the mark ``Series`` or ``Table`` leaves."""
    return get_origin(hint) is Annotated and any(isinstance(m, marker) for m in get_args(hint)[1:])


def _checked(spec: FieldSpec, where: str) -> FieldSpec:
    """``spec``, once its limits fit its type and pydantic can carry it; a default it declares is made its type."""
    base = without_none(spec.type)
    limits = set(spec.param.constraints)
    if spec.shape == "table" and limits:
        raise NodeDeclarationError(f"{where}{spec.name}: a table takes no limits; take them off", code="bound_invalid")
    if spec.shape == "table" and spec.param.choices is not None:
        raise NodeDeclarationError(f"{where}{spec.name}: a table takes no choices; take them off", code="choices_invalid")
    if spec.shape == "table":
        return spec
    try:
        spec.adapter  # noqa: B018
    except NodeDeclarationError as error:  # a Param that doesn't fit its type, the field's own or one in a model it holds
        error.message = f"{where}{spec.name}: {error.message}"
        error.args = (error.message,)
        raise
    except Exception as error:  # noqa: BLE001  pydantic can't build a schema for this type
        if limits:
            raise NodeDeclarationError(f"{where}{spec.name}: the limits {spec.param.constraints!r} don't fit {getattr(base, '__name__', base)}: {error}", code="bound_invalid") from None
        raise NodeDeclarationError(
            f"{where}{spec.name}: conductor can't check or write {getattr(base, '__name__', base)} as JSON; "
            "use a type pydantic knows, such as a pydantic model or a dataclass",
            code="type_not_carried",
        ) from None
    if spec.default is None and spec.shape != "table" and not spec.optional:
        raise NodeDeclarationError(f"{where}{spec.name}: the default None is a blank; write its type with | None", code="default_invalid")
    if not spec.required and spec.default is not None and spec.default is not SKIPPED:
        try:
            made = spec.lax_adapter.validate_python(spec.default)
        except Exception as error:  # noqa: BLE001  pydantic's own error says why
            raise NodeDeclarationError(f"{where}{spec.name}: the default {spec.default!r} is not a {getattr(base, '__name__', base)}: {error}", code="default_invalid") from None
        return dataclasses.replace(spec, default=made)
    return spec


_BUILTINS = (bool, int, float, str, bytes, datetime.datetime, datetime.date)


def _bare(hint: Any) -> Any:
    """The type pydantic applies a limit to: ``| None`` and a carried subclass's ``Annotated`` off, the subclass as its builtin."""
    bare = without_none(hint)
    bare = get_args(bare)[0] if get_origin(bare) is Annotated else bare
    return next((b for b in _BUILTINS if issubclass(bare, b)), bare) if isinstance(bare, type) else bare


def _unheld(hint: Any, constraints: tuple[tuple[str, Any], ...]) -> list[str]:
    """The limits pydantic's schema for ``hint`` doesn't carry, so it would only try them on a value, and fail."""
    if not constraints:
        return []
    try:
        return _unheld_on(hint, constraints)
    except TypeError:  # a hint holding a Param (``list[Annotated[int, Param(ge=0)]]``) can't key a cache
        return _unheld_on.__wrapped__(hint, constraints)


@functools.cache
def _unheld_on(hint: Any, constraints: tuple[tuple[str, Any], ...]) -> list[str]:
    limited: Any = Annotated[_bare(hint), PydanticField(**dict(constraints))]
    try:
        held = TypeAdapter(limited).core_schema
    except Exception:  # noqa: BLE001  a limit of the wrong kind (ge="a"): building the field itself says so
        return []
    return sorted(key for key, _ in constraints if key not in held)


def _carried(hint: Any) -> Any:
    """``hint`` as pydantic reads it: a host's subclass of a builtin is checked as the builtin, then made one."""
    if get_origin(hint) in (Union, types.UnionType):
        return Union[tuple(_carried(a) for a in get_args(hint))]  # noqa: UP007
    if isinstance(hint, type) and hint not in _BUILTINS and not hasattr(hint, "__get_pydantic_core_schema__"):
        base = next((b for b in _BUILTINS if issubclass(hint, b)), None)
        if base is not None:
            return Annotated[hint, ValidateAs(base, hint)]
    return hint


#: The classes whose values are plain in JSON: no keys to order, and read back by their type as they were.
SCALARS = frozenset({str, int, float, bool, type(None), datetime.date, datetime.datetime, datetime.time, datetime.timedelta})


def carried_list(hint: Any) -> TypeAdapter[Any]:
    """Checks a list of values of ``hint``, as a saved column is read back, and writes it as JSON."""
    return TypeAdapter(_list_of(_carried(hint)), config=_FLOATS_AS_TEXT)


#: A float that isn't finite is written as the text ``"NaN"``, ``"Infinity"`` or ``"-Infinity"``, which reads back.
_FLOATS_AS_TEXT = ConfigDict(ser_json_inf_nan="strings")


def _list_of(item: Any) -> Any:
    """``list[item]`` for an ``item`` known only at run time; a set is refused, as its order differs between processes."""
    return Annotated[types.GenericAlias(list, (item,)), BeforeValidator(_not_a_set)]


def _column_of(item: Any) -> Any:
    """``list[item]`` for values conductor wrote itself, a saved column or a table's column, which are never a set."""
    return types.GenericAlias(list, (item,))


#: Why a set is refused wherever a list is read or written: its order differs between processes.
SET_HAS_NO_ORDER = "a set has no order, so its rows would come in another order in another process; give a list"


def _not_a_set(value: Any) -> Any:
    if isinstance(value, (set, frozenset)):
        raise PydanticCustomError("set_has_no_order", SET_HAS_NO_ORDER)
    return value


def _columns_of(row: Any, name: str, where: str) -> dict[str, Any] | None:
    """A row type's columns by name; a column is read by its bare type, so one ``Annotated`` with anything is refused."""
    if row is Any or row is None:
        return None
    columns = typing.get_type_hints(row, include_extras=True)
    for column, hint in columns.items():
        if _holds_annotated(hint):
            raise NodeDeclarationError(
                f"{where}{name}: the column '{column}' is Annotated, which a table's column doesn't read; check its values in the node",
                code="annotated_extra_unknown",
            )
    return columns


def _holds_annotated(hint: Any) -> bool:
    return get_origin(hint) is Annotated or any(_holds_annotated(arg) for arg in get_args(hint))


def given_as_items(spec: FieldSpec, value: Any) -> bool:
    """Whether ``value``, given or typed into ``spec``, is many values, each run as its own row: a list on a ``per_item`` field.

    A set counts as one too, so it is refused for having no order rather than read as one value.
    """
    return spec.per_item and isinstance(value, (list, tuple, set, frozenset))


def _takes_a_list(hint: Any) -> bool:
    """Whether a list is one value of ``hint``: a list's, a tuple's, a set's or a sequence's, ``Any``'s, a union
    holding one, or any type whose JSON form may be an array (a pydantic ``RootModel`` of a tuple, a type alias such as
    pydantic's ``JsonValue``)."""
    if hint is Any or hint is object or isinstance(hint, TypeVar):
        return True
    origin = get_origin(hint)
    if origin in (Union, types.UnionType):
        return any(_takes_a_list(a) for a in get_args(hint))
    if origin is Annotated:
        return _takes_a_list(get_args(hint)[0])
    kind = origin or hint
    if isinstance(kind, type) and issubclass(kind, (str, bytes, bytearray, Mapping, *_BUILTINS)):
        return False
    if isinstance(kind, type) and (issubclass(kind, (list, tuple, set, frozenset)) or (kind.__module__ == "collections.abc" and issubclass(kind, Iterable))):
        return True
    kinds = _json_kinds(hint)
    return kinds is None or "array" in kinds


def _json_kinds(hint: Any) -> frozenset[str] | None:
    """The JSON types a value of ``hint`` may be written as (``"string"``, ``"array"``, ``"null"``, ...), by its JSON
    schema; ``None`` when the schema says nothing against any. A type with no JSON form may be none of them."""
    try:
        schema = TypeAdapter(_carried(hint)).json_schema()
    except Exception:  # noqa: BLE001  a type with no JSON form is never written as JSON
        return frozenset()
    return _schema_kinds(schema, schema.get("$defs", {}))


def _schema_kinds(schema: Mapping[str, Any], defs: Mapping[str, Any]) -> frozenset[str] | None:
    """The JSON types a schema allows: its type, its branches' or its values'; ``None`` for no constraint on the type."""
    if "$ref" in schema:
        return _schema_kinds(defs.get(schema["$ref"].rpartition("/")[2], {}), defs)
    branches = [*schema.get("anyOf", ()), *schema.get("oneOf", ()), *schema.get("allOf", ())[:1]]
    if branches:
        kinds = [_schema_kinds(branch, defs) for branch in branches]
        return None if None in kinds else frozenset().union(*kinds)  # type: ignore[arg-type]
    kind = schema.get("type")
    if kind is not None:
        return frozenset([kind] if isinstance(kind, str) else kind)
    if {"enum", "const"} & schema.keys():
        return frozenset(_JSON_KIND[type(v)] for v in schema.get("enum", [schema.get("const")]))
    return None


_JSON_KIND = {type(None): "null", bool: "boolean", int: "integer", float: "number", str: "string", list: "array", dict: "object"}


#: An alias's JSON types, read once per alias: an edge or a blank asks again for each.
_alias_kinds = functools.lru_cache(maxsize=None)(_json_kinds)


def _is_alias(hint: Any) -> bool:
    """Whether ``hint`` is a type alias (``type X = ...``, ``TypeAliasType``), which is read by its JSON schema."""
    return type(hint).__name__ == "TypeAliasType"  # typing's, or typing_extensions' as pydantic's JsonValue is


def takes_a_blank(hint: Any) -> bool:
    """Whether a blank may arrive: ``X | None``, ``Any``, which says nothing against one, or a type alias whose JSON
    form may be ``null`` (pydantic's ``JsonValue``)."""
    if hint is Any or (get_origin(hint) in (Union, types.UnionType) and type(None) in get_args(hint)):
        return True
    return _is_alias(hint) and ((kinds := _alias_kinds(hint)) is None or "null" in kinds)


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


def is_a_number(cls: Any, kind: type | tuple[type, ...] = (int, float)) -> bool:
    """Whether ``cls`` is a class of number of ``kind`` (``int`` or ``float``, or a class of theirs), never ``bool``: a
    yes or no is not a number, though Python counts ``True`` as 1. A value is one when ``is_a_number(type(value))``."""
    return isinstance(cls, type) and issubclass(cls, kind) and not issubclass(cls, bool)


def lands_as_it_is(target: type, source: type) -> bool:
    """Whether a value of class ``source`` is one of class ``target`` as it is: its own class or a class of it, a yes
    or no never on a number."""
    return issubclass(source, target) and not (issubclass(source, bool) and is_a_number(target))


def refuses_a_bool(hint: Any) -> bool:
    """Say whether a ``bool`` is refused on a type, the one rule for a field and a table's column.

    Python counts ``True`` as 1, and pydantic's ``int``, ``float`` and ``Literal`` take it so; this rule refuses it.

    Args:
        hint: A field's type or a column's.

    Returns:
        True when an option of ``hint`` is a number's (``is_a_number``, or a ``Literal`` of numbers) and no option
        takes a ``bool`` as it is (``bool``, ``Any``, ``object``, a ``Literal`` holding one).
    """
    options = [option for option in _options(hint) if option is not type(None)]
    return any(map(_a_number_type, options)) and not any(map(_takes_a_bool, options))


def _no_bool_on_a_number(hint: Any) -> Any:
    """A column's ``hint``, refusing a yes or no where ``refuses_a_bool`` says so."""
    return Annotated[hint, BeforeValidator(_not_a_bool)] if refuses_a_bool(hint) else hint


def _options(hint: Any) -> list[Any]:
    """The types a hint allows."""
    return list(get_args(hint)) if get_origin(hint) in (Union, types.UnionType) else [hint]


def _one_accepts(target: Any, source: Any) -> bool:
    if target is Any or target == source:
        return True
    if _is_alias(target):  # a value lands if its JSON form is one the alias's may be: a number on a JSON value
        kinds, given = _alias_kinds(target), _json_kinds(source)
        return kinds is None or (bool(given) and given is not None and all(k in kinds or (k == "integer" and "number" in kinds) for k in given))
    if get_origin(source) is Literal:
        if get_origin(target) is Literal:  # a narrower Literal lands on one holding every value it may be
            return {(type(v), v) for v in get_args(source)} <= {(type(v), v) for v in get_args(target)}
        return all(_one_accepts(target, type(v)) for v in get_args(source))
    if typing.is_typeddict(target):
        return _typeddict_accepts(target, source)
    if isinstance(target, type) and isinstance(source, type):
        own = getattr(target, "accepts", None)  # a host's type may widen what lands on it
        if issubclass(source, bool) and is_a_number(target):
            return False  # a yes or no is never a number, a host's own included
        return lands_as_it_is(target, source) or (target is float and issubclass(source, int)) or (callable(own) and bool(own(source)))
    return False


def _typeddict_accepts(target: Any, source: Any) -> bool:
    """A TypedDict lands on another whose keys it holds, each with a type that fits, and required wherever the target
    requires it: read by its keys, as a type checker reads one. Nothing else lands on a TypedDict."""
    if not typing.is_typeddict(source):
        return False
    wanted, given = typing.get_type_hints(target), typing.get_type_hints(source)
    return target.__required_keys__ <= source.__required_keys__ and all(key in given and accepts(hint, given[key]) for key, hint in wanted.items())


def type_id(hint: Any) -> str:
    """The name an editor and a stored graph know a type by: its ``id`` if it declares one, else its name.

    A ``Literal`` is known by the type of its values (``Literal["low", "high"]`` is ``str``), its values in the schema.
    """
    if get_origin(hint) is Literal:
        return " | ".join(dict.fromkeys(type_id(type(value)) for value in get_args(hint)))
    if get_origin(hint) in (Union, types.UnionType):  # each member by its own name: a host's Text | None is "Text | None"
        return " | ".join("None" if member is type(None) else type_id(member) for member in get_args(hint))
    own = _declared(hint, "id") if isinstance(hint, type) else None
    return own if own is not None else getattr(hint, "__name__", None) or str(hint)


def type_title(cls: type) -> str:
    """What a person reads for a type: its ``display_title`` if it declares one, else its name."""
    own = _declared(cls, "display_title")
    return own if own is not None else cls.__name__


@functools.cache
def _declared(cls: type, name: str) -> str | None:
    """The text a class sets as ``name`` for itself (``id = "money"``, or annotated ``ClassVar``); a field of that name
    is each value's own, its default never the class's."""
    own = getattr(cls, name, None)
    if not isinstance(own, str):
        return None
    for klass in cls.__mro__:
        annotation = inspect.get_annotations(klass).get(name)
        if annotation is not None:
            return own if get_origin(annotation) is ClassVar or str(annotation).startswith(("ClassVar", "typing.ClassVar")) else None
    return own


# -- reading a value as its field ---------------------------------------------------------------


class Refused(Exception):
    """A value its field can't take: ``reason`` says why and quotes nothing of it; ``ctx`` has pydantic's ``errors``
    without the value, the field's ``name`` and, for a table, the ``column``. ``code`` is the problem it is where it
    was met: ``invalid_output`` for a node's return, ``invalid_edge_value`` for what an edge brought."""

    def __init__(self, reason: str, *, code: str = "", **ctx: Any) -> None:
        super().__init__(reason)
        self.reason, self.code, self.ctx = reason, code, ctx


def already(target: Any, source: Any) -> bool:
    """Whether every value of type ``source`` already is one ``target`` takes as it is, with nothing to read."""
    if target is Any or target == source:
        return True
    targets = [t for t in _options(target) if isinstance(t, type) and not typing.is_typeddict(t)]
    sources = [s for s in _options(source) if s is not type(None)]
    return all(isinstance(s, type) and not typing.is_typeddict(s) and any(lands_as_it_is(t, s) for t in targets) for s in sources)


def read_strictly(adapter: TypeAdapter[Any], value: Any) -> Any:
    """``value`` read by ``adapter`` from its JSON form, strictly, as a person's answer and a choice offered are: a yes
    or no is ``true`` or ``false``, a number is a number, never a text that spells one; a date is still its ISO text,
    and an enum its value, as JSON writes them."""
    try:
        written = to_json(value, inf_nan_mode="constants")
    except PydanticSerializationError:
        raise Refused("the value has no JSON form") from None
    try:
        return adapter.validate_json(written, strict=True)
    except ValidationError as error:
        raise Refused(value_free_errors(error)[0]["msg"]) from None


def _a_number_type(option: Any) -> bool:
    """Whether ``option`` is a number's class (``is_a_number``) or a ``Literal`` of numbers."""
    if get_origin(option) is Literal:
        return all(is_a_number(type(value)) for value in get_args(option))
    return is_a_number(option)


def _takes_a_bool(option: Any) -> bool:
    if get_origin(option) is Literal:
        return any(isinstance(value, bool) for value in get_args(option))
    return option in (bool, Any, object)


def _not_a_bool(value: Any) -> Any:
    if isinstance(value, bool):
        raise PydanticCustomError("bool_is_not_a_number", "a yes or no is not a number; give a number")
    return value


def read_given(spec: FieldSpec, value: Any, *, items: bool) -> Any:
    """``value`` typed into ``spec`` or given for it by a caller, a list of its values with ``items``; ``Refused`` if
    it can't be. A table may be a list of records."""
    try:
        if items:
            return spec.list_adapter.validate_python(value)
        if spec.shape == "table":
            return None if value is None and spec.optional else read_table(spec, value, strict=False, given=True)
        return spec.lax_adapter.validate_python(value)
    except ValidationError as error:
        raise _refused(error, _list_of(spec.type) if items or spec.shape == "many" else spec.type) from None


def read_value(spec: FieldSpec, value: Any, *, strict: bool) -> Any:
    """``value`` as one value of ``spec``: as it is when already of its class, else read by its type, ``strict`` as
    pydantic's strict mode is. A ``SKIPPED`` item of a ``Series`` is a gap, read around."""
    if value is SKIPPED:
        return value
    if spec.shape == "table":
        return None if value is None and spec.optional else read_table(spec, value, strict=strict)
    if spec.shape == "one" and type(value) in spec.plain_classes:
        return value
    if spec.shape == "many" and isinstance(value, (tuple, range)):
        value = list(value)  # any sequence a node returns is its rows
    elif spec.shape == "many" and isinstance(value, pl.Series):
        value = value.to_list()  # a column of a frame is its values, as a frame is a Table
    try:
        return (spec.adapter if strict else spec.lax_adapter).validate_python(value, strict=strict)
    except ValidationError as error:
        if spec.shape == "many" and isinstance(value, list) and any(item is SKIPPED for item in value):
            items = iter(read_value(spec, [item for item in value if item is not SKIPPED], strict=strict))
            return [SKIPPED if item is SKIPPED else next(items) for item in value]
        raise _refused(error, _list_of(spec.type) if spec.shape == "many" else spec.type) from None


def read_table(spec: FieldSpec, value: Any, *, strict: bool, given: bool = False) -> pl.DataFrame:
    """``value`` as a table of ``spec``: every column its row type names, each read by the column's type unless its
    dtype already holds that type; a missing one is refused naming it.

    A table ``given`` (typed in, or a caller's) may also be a list of records and keeps only the named columns, as
    an edge into the input would; a table a node returns or an edge brings keeps its other columns too.
    """
    if given and not isinstance(value, pl.DataFrame):
        try:
            records = RECORDS.validate_python(value)
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
                refused = _refused(error, _list_of(hint))
                raise Refused(f"column '{column}': {refused.reason}", column=column, **refused.ctx) from None
    if given:
        return pl.DataFrame({c: read[c] if c in read else value[c] for c in spec.columns}, strict=False)
    return value.with_columns(pl.Series(c, cells, strict=False) for c, cells in read.items()) if read else value


def _refused(error: ValidationError, hint: Any) -> Refused:
    """``error`` of reading a value of ``hint``, its errors without the value: each ``loc`` only as far as ``hint`` declares."""
    errors = value_free_errors(error)
    for said, raw in zip(errors, error.errors(include_url=False, include_input=False), strict=True):
        said["loc"] = _declared_loc(hint, raw["loc"])
    return Refused(errors[0]["msg"], errors=errors)


def _declared_loc(hint: Any, loc: tuple[int | str, ...]) -> tuple[int | str, ...]:
    """The steps of pydantic's ``loc`` into a value of ``hint`` that the type itself names: an index into a list, a
    field of a model, dataclass or ``TypedDict``. It stops at the first other step, a key of a dict or a branch of a
    union, as a key can be the value's own."""
    kept: list[int | str] = []
    for step in loc:
        hint = _unannotated(hint)
        origin = get_origin(hint) or hint
        if isinstance(step, int) and isinstance(origin, type) and issubclass(origin, (list, tuple, set, frozenset)):
            args = [a for a in get_args(hint) if a is not Ellipsis]
            hint = args[min(step, len(args) - 1)] if args else Any
        elif isinstance(step, str) and step in (fields := _fields_of(hint)):
            hint = fields[step]
        else:
            break
        kept.append(step)
    return tuple(kept)


def _unannotated(hint: Any) -> Any:
    """``hint`` without ``Annotated`` or ``| None``."""
    while True:
        bare = without_none(hint)
        bare = get_args(bare)[0] if get_origin(bare) is Annotated else bare
        if bare is hint:
            return hint
        hint = bare


def _fields_of(hint: Any) -> Mapping[str, Any]:
    """The fields a model, dataclass or ``TypedDict`` declares, by name; none for any other type."""
    if not isinstance(hint, type) or not (issubclass(hint, BaseModel) or dataclasses.is_dataclass(hint) or typing.is_typeddict(hint)):
        return {}
    try:
        return typing.get_type_hints(hint)
    except Exception:  # noqa: BLE001  a hint that names what isn't defined: no field is named
        return {}


#: Reads a table given as a list of records, one object per row.
RECORDS = TypeAdapter(list[dict[str, Any]])

#: The type a polars dtype's values already are, so a column of it needs no reading; ``Null`` holds only blanks.
_PYTHON_OF: dict[Any, Any] = {
    **dict.fromkeys((pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64), int),
    pl.Float32: float, pl.Float64: float, pl.String: str, pl.Boolean: bool, pl.Date: datetime.date, pl.Datetime: datetime.datetime,
    pl.Null: type(None),
}
