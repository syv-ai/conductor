"""What a node author writes in a signature, and the one record each input and output becomes.

A type is any class pydantic can check and write as JSON. Conductor's own
types are named in ``CONDUCTOR_TYPES``; a host's own type is a ``NamedType``,
declaring its ``type_id`` and ``type_title``. A host's type may declare a
classmethod ``accepts(source)`` that lets values of another type land on it;
otherwise a subclass lands where its parent is asked for. A subclass of a builtin (``class Text(str)``) is checked as the
builtin, then made one.

A field has one of three shapes: ``one`` value, ``many`` values
(``Series[X]``, a plain list to the node), or a ``table`` (``Table[R]``, a
``polars.DataFrame``). ``many`` is the one shape a graph reads: an output of
many values is worked through one item at a time by an input that takes one,
and an input of many gathers them back. A table is one value, like any other;
its shape only says how a frame is checked and written as JSON.
"""

from __future__ import annotations

import builtins
import copy
import dataclasses
import datetime
import decimal
import functools
import types
import typing
import uuid
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Final,
    Literal,
    NamedTuple,
    Protocol,
    TypeAliasType,
    TypeGuard,
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
    JsonValue,
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
    """The type of ``REQUIRED``: one value whose repr is ``REQUIRED``, kept as itself by a copy or a pickle."""

    def __repr__(self) -> str:
        return "REQUIRED"

    def __reduce__(self) -> str:
        return "REQUIRED"


#: The default of an input nothing defaults.
REQUIRED: Any = _Required()


class _Skipped:
    """The type of ``SKIPPED``: one falsy value whose repr is ``SKIPPED``, kept as itself by a copy or a pickle."""

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
    choices: list[Any] | tuple[Any, ...] | None = None
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
        except FieldValueError as refused:
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


#: The mark of a table of any columns.
_ANY_TABLE = _TableMarker(Any)

_T = TypeVar("_T")
_Record = TypeVar("_Record", bound="FieldDescription")

#: ``FromRun[X]``: a value the host hands the run by its type (a session, who is running), never an input. It takes
#: no edge, is not stored in the record, and is no part of what makes a row's result reusable. The run is given one
#: with ``from_run={X: value}``. To a type checker the parameter is an ``X``.
FromRun = Annotated[_T, _FromRunMarker()]

#: ``Series[X]``: many values of ``X``, which a graph works through one at a time; a plain ``list[X]`` to the node and
#: to a type checker.
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
    """A field's name in everyday words, its label when ``Param(title=)`` gives none: ``case_sensitive`` is ``Case sensitive``."""
    return name.replace("-", " ").replace("_", " ").strip().capitalize()


@dataclass(frozen=True)
class FieldSpec:
    """One input or output of a node, as ``@node`` read it from the signature: its shape, the type of one value, and
    how a value is checked.

    Attributes:
        name: The parameter's name, or the output's.
        shape: ``"one"`` for one value, ``"many"`` for a ``Series`` (a list of items), ``"table"`` for a ``Table``.
        type: The type of one value: the hint without ``Series`` or its ``Param``s, with its ``| None`` kept; a
            ``Series``' item type; ``Table[R]`` for a table of ``R``'s columns, ``polars.DataFrame`` for one of any.
        optional: A blank (``None``) may arrive in place of the field's value: of one value, or of a ``Series``'
            whole list (``Series[X] | None``). Whether a ``Series``' items may be blank is its item type's own
            ``| None``.
        default: The signature's default: ``REQUIRED`` when it has none, ``SKIPPED`` for an output on a branch not
            taken, ``None`` for a blank, else the value.
        param: The field's ``Param``, the ``Param``s of its hint merged; an empty one when it has none.
        columns: A table's columns by name, from its row type; ``None`` for a table of any columns, and for a field
            that isn't a table.
        type_id: The id of one value's type, ``| None`` off, as an editor reads it; a table's names its columns.
        classes: The classes ``type_id`` names, each once: what a registry holding the node offers by id. A table
            names none.
        column_type_ids: The ids of a table's declared column types, by column.
        item: The type of each value a field of many gathers: a ``Series``' item, a table's row (``Any`` for a table
            of any columns). ``None`` for a field of one value.
    """

    name: str
    shape: Literal["one", "many", "table"]
    type: Any
    optional: bool = False
    default: Any = REQUIRED
    param: Param = field(default_factory=Param)
    columns: dict[str, Any] | None = None
    type_id: str = field(kw_only=True)
    classes: tuple[type, ...] = field(kw_only=True)
    column_type_ids: Mapping[str, str] = field(kw_only=True, default_factory=dict[str, str])
    item: Any = field(kw_only=True, default=None)
    #: The default in its JSON form, written once when the node is declared; ``None`` where the field has no value
    #: to propose (it is required, or its default is ``SKIPPED``).
    default_json: JsonValue = field(kw_only=True, default=None)

    @property
    def required(self) -> bool:
        """Whether the signature gives no default, so a value must come."""
        return self.default is REQUIRED

    @property
    def title(self) -> str:
        """What a person reads as the field's name: ``Param(title=)``, else its name in everyday words."""
        return self.param.title or humanised(self.name)

    def json_of(self, value: Any) -> Any:
        """``value``, one this field holds, in its JSON form as its type writes it; a table as its rows' records,
        and a blank table as ``None``."""
        if self.shape == "table":
            return None if value is None else to_jsonable_python(value.to_dicts())
        return self.adapter.dump_python(value, mode="json")

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
    def plain(self) -> bool:
        """Whether its values are of plain classes (``SCALARS``): JSON with no keys to order, read back by the type as they were.

        Not when it takes both a ``float`` and a ``str``: a float that isn't finite is written as text, and only its
        mark tells it from the text it reads as.
        """
        classes = set(self.plain_classes)
        return bool(classes) and classes <= SCALARS and not {float, str} <= classes

    @functools.cached_property
    def keeps_blanks(self) -> bool:
        """Whether a field that gathers keeps the place of an item with no value, as a blank: a ``Series`` whose
        item type takes one. A table never does, as a record is never blank."""
        return self.shape == "many" and takes_a_blank(self.type)

    @functools.cached_property
    def takes_a_number(self) -> bool:
        """Whether a ``bool`` must not land here (``refuses_a_bool``)."""
        return refuses_a_bool(self.type)

    @functools.cached_property
    def plain_classes(self) -> tuple[type, ...]:
        """The classes whose instances, of exactly that class, the field takes as they are, with no reading.

        Returns:
            Each option of the type, ``NoneType`` included for a blank-taking one. Empty when any option is not a
            class (``list[int]``, a ``Literal``, a ``TypedDict``, ``Any``) or the field has limits or choices: such a
            value is always read by the type.
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
        """For a table, a check and writer of each column its row type names: a cell is a value of the column's type,
        so blank only where that type takes a blank, and never a yes or no where ``refuses_a_bool`` says so."""
        adapters = {}
        for column, hint in (self.columns or {}).items():
            checked = Annotated[hint, BeforeValidator(_not_a_bool)] if refuses_a_bool(hint) else hint
            adapters[column] = TypeAdapter(_column_of(_carried(checked)), config=_FLOATS_AS_TEXT)
        return adapters

    @overload
    def describe(self, record: None = None, /, **changed: Any) -> FieldDescription: ...

    @overload
    def describe(self, record: builtins.type[_Record], /, **changed: Any) -> _Record: ...  # the field named type hides the builtin

    def describe(self, record: builtins.type[FieldDescription] | None = None, /, **changed: Any) -> FieldDescription:
        """The field as an editor reads it: its shape, its type's ``id`` and JSON schema, and what a person reads about
        it. Only the keys the field has are set.

        Args:
            record: A ``FieldDescription`` subclass that adds to it, such as a graph's input or an input placed in a
                graph; ``FieldDescription`` itself when not given.
            **changed: Keys of the record set or replaced, such as the address a graph knows the field by.

        Returns:
            The record, validated.
        """
        out: dict[str, Any] = {"name": self.name, "title": self.title, "shape": self.shape, "required": self.required}
        if self.shape != "table":
            out["type"] = self.type_id
        out.update(copy.deepcopy(self._schemas))  # the caller may change what it is given
        if not self.required:
            out["default"] = self.default_json
        if self._declared_choices is not None:
            out["choices"] = self._declared_choices  # the record is made with its own copy of each
        for key in ("description", "widget", "exclusive_group"):
            if getattr(self.param, key) is not None:
                out[key] = to_jsonable_python(getattr(self.param, key), fallback=str)
        return (record or FieldDescription).model_validate({**out, **changed})

    @functools.cached_property
    def _declared_choices(self) -> list[dict[str, Any]] | None:
        """The choices the type declares (``Param(choices=)``, else a ``Literal``'s values) as a question shows them."""
        if self.shape == "table":
            return None
        given = self.param.choices if self.param.choices is not None else literal_values(self.type)
        return None if given is None else shown_choices(self, given)

    @functools.cached_property
    def _schemas(self) -> dict[str, Any]:
        """The JSON schema of one value, or of each column of a table, made once: pydantic builds one in tens of microseconds."""
        if self.shape == "table":
            return {"columns": {c: _schema(t) for c, t in (self.columns or {}).items()}}
        limited = _carried(self.type)
        if self.param.constraints:  # a field's title and description are its record's own, not repeated in its schema
            limited = Annotated[limited, PydanticField(**self.param.constraints)]
        return {"schema": _schema(limited)}


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
            item_id, classes = named_by(without_none(item))
            return _checked(FieldSpec(name, "many", item, or_none, default, param, type_id=item_id, classes=classes, item=item), where)
        if markers and isinstance(markers[0], _TableMarker):
            columns = _columns_of(markers[0].row, name, where)
            column_type_ids = {column: type_id(hint) for column, hint in columns.items()} if columns is not None else {}
            table: Any = Annotated[pl.DataFrame, markers[0]] if columns is not None else pl.DataFrame
            row = markers[0].row if columns is not None else Any
            return _checked(FieldSpec(name, "table", table, or_none, default, param, columns, type_id=type_id(table), classes=(), column_type_ids=column_type_ids, item=row), where)
        hint = base
    if hint is pl.DataFrame or hint is Table:
        return _checked(FieldSpec(name, "table", pl.DataFrame, or_none, default, param, None, type_id="Table", classes=(), item=Any), where)
    if or_none:
        hint = Union[hint, None]  # noqa: UP007
    one_id, classes = named_by(without_none(hint))
    return _checked(FieldSpec(name, "one", hint, takes_a_blank(hint), default, param, type_id=one_id, classes=classes), where)


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
        return _with_default_json(spec, where)
    try:
        spec.adapter  # noqa: B018
    except NodeDeclarationError as error:  # a Param that doesn't fit its type, the field's own or one in a model it holds
        error.message = f"{where}{spec.name}: {error.message}"
        error.args = (error.message,)
        raise
    except Exception as error:  # noqa: BLE001  pydantic can't build a schema for this type
        if limits:
            raise NodeDeclarationError(f"{where}{spec.name}: the limits {spec.param.constraints!r} don't fit {getattr(base, '__name__', base)}, so change one: {error}", code="bound_invalid") from None
        raise NodeDeclarationError(
            f"{where}{spec.name}: conductor can't check or write {getattr(base, '__name__', base)} as JSON; "
            "use a type pydantic knows, such as a pydantic model or a dataclass",
            code="type_not_carried",
        ) from None
    if spec.default is None and not spec.optional:
        raise NodeDeclarationError(f"{where}{spec.name}: the default None is a blank; write its type with | None", code="default_invalid")
    if not spec.required and spec.default is not None and spec.default is not SKIPPED:
        try:
            made = spec.lax_adapter.validate_python(spec.default)
        except Exception as error:  # noqa: BLE001  pydantic's own error says why
            raise NodeDeclarationError(f"{where}{spec.name}: the default {spec.default!r} is not a {getattr(base, '__name__', base)}, so give one that is: {error}", code="default_invalid") from None
        spec = dataclasses.replace(spec, default=made)
    return _with_default_json(spec, where)


def _with_default_json(spec: FieldSpec, where: str) -> FieldSpec:
    """``spec`` holding its default's JSON form; a default its type can't write as JSON raises ``default_invalid``."""
    if spec.required or spec.default is SKIPPED:
        return spec
    try:
        written = spec.json_of(spec.default)
    except ValueError as error:  # pydantic's verdict: a PydanticSerializationError, or bytes that aren't UTF-8
        raise NodeDeclarationError(
            f"{where}{spec.name}: the default {spec.default!r} has no JSON form, so give one its type writes as JSON: {error}", code="default_invalid",
        ) from None
    return dataclasses.replace(spec, default_json=written)


_BUILTINS: tuple[type, ...] = (bool, int, float, str, bytes, datetime.datetime, datetime.date)


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
SET_HAS_NO_ORDER = "a set has no order, so its items would come in another order in another process; give a list"


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
        known = [k for k in kinds if k is not None]
        return frozenset[str]().union(*known) if len(known) == len(kinds) else None
    kind = schema.get("type")
    if kind is not None:
        return frozenset([kind] if isinstance(kind, str) else kind)
    if {"enum", "const"} & schema.keys():
        return frozenset(_JSON_KIND[type(v)] for v in schema.get("enum", [schema.get("const")]))
    return None


_JSON_KIND = {type(None): "null", bool: "boolean", int: "integer", float: "number", str: "string", list: "array", dict: "object"}


#: An alias's JSON types, read once per alias: an edge or a blank asks again for each.
_alias_kinds = functools.lru_cache(maxsize=None)(_json_kinds)


def _is_alias(hint: Any) -> TypeGuard[TypeAliasType]:
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
    """Whether a value of type ``source`` may land on an input of type ``target``.

    A table lands on a table that names no column the source is known to lack or to hold as another type; a table
    of any columns lands on every table, and is checked when it arrives.
    """
    if target is Any or source is Any or target == source:
        return True
    wanted, held = _table_marker(target), _table_marker(source)
    if wanted is not None or held is not None:
        return wanted is not None and held is not None and _columns_fit(wanted, held, accepts, unknown=True)
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


def _options(hint: Any) -> list[Any]:
    """The types a hint allows."""
    return list(get_args(hint)) if get_origin(hint) in (Union, types.UnionType) else [hint]


def _one_accepts(target: Any, source: Any) -> bool:
    if target is Any or target == source:
        return True
    if _is_alias(target):  # a value lands if its JSON form is one the alias's may be: a number on a JSON value
        kinds, given = _alias_kinds(target), _json_kinds(source)
        if kinds is None:
            return True
        if not given:  # a type with no JSON form, or none known
            return False
        return all(k in kinds or (k == "integer" and "number" in kinds) for k in given)
    if get_origin(source) is Literal:
        if get_origin(target) is Literal:  # a narrower Literal lands on one holding every value it may be
            return {(type(v), v) for v in get_args(source)} <= {(type(v), v) for v in get_args(target)}
        return all(_one_accepts(target, type(v)) for v in get_args(source))
    if typing.is_typeddict(target):  # read by its keys, as a type checker reads one: nothing else lands on it
        if not typing.is_typeddict(source):
            return False
        wanted, held = typing.get_type_hints(target), typing.get_type_hints(source)
        # it holds every key the target does, each a type that fits, required wherever the target requires it
        return target.__required_keys__ <= source.__required_keys__ and all(key in held and accepts(hint, held[key]) for key, hint in wanted.items())
    if isinstance(target, type) and isinstance(source, type):
        own = getattr(target, "accepts", None)  # a host's type may widen what lands on it
        if issubclass(source, bool) and is_a_number(target):
            return False  # a yes or no is never a number, a host's own included
        return lands_as_it_is(target, source) or (target is float and issubclass(source, int)) or (callable(own) and bool(own(source)))
    return False


def _table_marker(hint: Any) -> _TableMarker | None:
    """The mark of a table type, holding its row type: ``Table[R]``'s own, one of any columns for a bare
    ``polars.DataFrame``, and ``None`` for a type that is no table."""
    if hint is pl.DataFrame:
        return _ANY_TABLE
    if get_origin(hint) is Annotated and get_args(hint)[0] is pl.DataFrame:
        return next(m for m in get_args(hint)[1:] if isinstance(m, _TableMarker))
    return None


def row_type(table: Any) -> Any:
    """The row type of a table type: what a hook reads from the type arriving on a table input.

    Args:
        table: A type, as a hook's parameter annotated ``type`` receives one.

    Returns:
        ``R``, the ``TypedDict`` naming the columns, for ``Table[R]``; ``Any`` for a table of any columns; ``None``
        for a type that is no table.
    """
    marker = _table_marker(table)
    return None if marker is None else marker.row


def _columns_fit(wanted: _TableMarker, held: _TableMarker, fits: Callable[[Any, Any], bool], *, unknown: bool) -> bool:
    """Whether a table of ``held``'s columns holds every column ``wanted`` names, each of a type that ``fits``;
    ``unknown`` is the answer for a source of any columns, known only when it arrives."""
    if wanted.row is Any or wanted.row is None:
        return True
    if held.row is Any or held.row is None:
        return unknown
    columns, given = typing.get_type_hints(wanted.row), typing.get_type_hints(held.row)
    return all(column in given and fits(hint, given[column]) for column, hint in columns.items())


class NamedType(Protocol):
    """A host's own type, as people see it: the id an editor and a stored graph know it by, and the title a person
    reads for it. Every class of a host's that a node's field holds, and every class in ``NodeRegistry(types=)``,
    declares both.

    Attributes:
        type_id: The name an editor and a stored graph know the type by, such as ``"money"``.
        type_title: What a person reads for the type, such as ``"Money"``.
    """

    type_id: ClassVar[str]
    type_title: ClassVar[str]


class TypeName(NamedTuple):
    """A type's id and title, as ``NamedType`` declares them.

    Attributes:
        id: The name an editor and a stored graph know the type by.
        title: What a person reads for the type.
    """

    id: str
    title: str


#: The id and title of each type conductor itself knows. Every other class a field holds is a ``NamedType``.
CONDUCTOR_TYPES: Final[Mapping[object, TypeName]] = MappingProxyType({
    Any: TypeName("Any", "Any value"),
    object: TypeName("object", "Any value"),
    type(None): TypeName("None", "Nothing"),
    bool: TypeName("bool", "Yes or no"),
    int: TypeName("int", "Whole number"),
    float: TypeName("float", "Number"),
    decimal.Decimal: TypeName("Decimal", "Decimal number"),
    str: TypeName("str", "Text"),
    bytes: TypeName("bytes", "Bytes"),
    datetime.date: TypeName("date", "Date"),
    datetime.datetime: TypeName("datetime", "Date and time"),
    datetime.time: TypeName("time", "Time of day"),
    datetime.timedelta: TypeName("timedelta", "Duration"),
    uuid.UUID: TypeName("UUID", "UUID"),
    dict: TypeName("dict", "Mapping"),
    list: TypeName("list", "List"),
    tuple: TypeName("tuple", "Tuple"),
    set: TypeName("set", "Set"),
    frozenset: TypeName("frozenset", "Set"),
})


#: The id and title of every ``TypedDict``: a record, shown by its fields.
RECORD: Final = TypeName("record", "Record")


def type_name(cls: Any) -> TypeName:
    """A class's id and title: its entry in ``CONDUCTOR_TYPES`` for conductor's own, else the ``type_id`` and
    ``type_title`` it declares as a ``NamedType``."""
    if cls in CONDUCTOR_TYPES:
        return CONDUCTOR_TYPES[cls]
    return TypeName(cls.type_id, cls.type_title)


def type_id(hint: Any) -> str:
    """The name an editor and a stored graph know a hint by, as ``named_by`` gives it."""
    return named_by(hint)[0]


#: What a field may take that no value is made of: anything, and a blank.
_NAMES_NO_CLASS: Final = frozenset({Any, type(None)})


def named_by(hint: Any) -> tuple[str, tuple[type, ...]]:
    """The name an editor and a stored graph know a hint by, and the classes that name is made of, each once.

    A class is known by its ``type_name``. A ``Literal`` is known by the types of its values (``Literal["low",
    "high"]`` is ``str``), a union by its members (``Money | None`` is ``money | None``, naming ``Money``), a generic
    by its class (``dict[str, int]`` is ``dict``), a ``TypedDict`` as ``record``, and a type alias by the name
    it is declared under, which names no class. ``Any`` and ``None`` name no class either. A table is known by its
    columns.
    """
    table = _table_marker(hint)
    if table is not None:  # a table is known by its columns, and names no class
        if table.row is Any or table.row is None:
            return "Table", ()
        columns = typing.get_type_hints(table.row)
        return f"Table[{', '.join(f'{column}: {type_id(of)}' for column, of in columns.items())}]", ()
    match get_origin(hint):
        case typing.Literal:
            values: tuple[object, ...] = get_args(hint)
            return _joined([named_by(cls) for cls in dict.fromkeys(type(value) for value in values)])
        case typing.Union | types.UnionType:
            return _joined([named_by(member) for member in get_args(hint)])
        case typing.Annotated:
            return named_by(get_args(hint)[0])
        case None if _is_alias(hint):
            return hint.__name__, ()
        case None if typing.is_typeddict(hint):  # a record declares no name of its own
            return RECORD.id, ()
        case None:
            return type_name(hint).id, () if hint in _NAMES_NO_CLASS else (hint,)
        case generic:
            return named_by(generic)


def _joined(named: list[tuple[str, tuple[type, ...]]]) -> tuple[str, tuple[type, ...]]:
    """Members' names read as one: their ids joined by `` | ``, their classes each once."""
    return " | ".join(name for name, _ in named), tuple(dict.fromkeys(cls for _, classes in named for cls in classes))


# -- reading a value as its field ---------------------------------------------------------------


class FieldValueError(Exception):
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
    wanted, held = _table_marker(target), _table_marker(source)
    if wanted is not None and held is not None:  # a table of other columns is cut to the ones the input names
        return wanted.row is Any or wanted.row is None
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
        raise FieldValueError("the value has no JSON form") from None
    try:
        return adapter.validate_json(written, strict=True)
    except ValidationError as error:
        raise FieldValueError(value_free_errors(error)[0]["msg"]) from None


def shown_choices(spec: FieldSpec, choices: list[Any] | tuple[Any, ...]) -> list[dict[str, Any]]:
    """Choices as a question shows them, each id the JSON of the value an answer picking it reads as.

    Args:
        spec: The field the choices are for.
        choices: Each a value or a ``Choice``.

    Returns:
        One ``{"id", "title", ...}`` record per choice, in order.

    Raises:
        FieldValueError: A choice is not a value of the field's type.
    """
    ids = [choice_record(choice)["id"] for choice in choices]
    if spec.shape == "many":  # a pick of several is one value: the list of them all
        kept = spec.json_of(read_strictly(spec.adapter, ids))
    else:
        kept = [spec.json_of(read_strictly(spec.adapter, i)) for i in ids]
    return [{**choice_record(choice, mode="json"), "id": k} for choice, k in zip(choices, kept, strict=True)]


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


def read_given(spec: FieldSpec, value: Any) -> Any:
    """``value`` typed into ``spec`` or given for it by a caller; ``FieldValueError`` if it can't be. A table may be
    a list of records."""
    try:
        if spec.shape == "table":
            return None if value is None and spec.optional else read_table(spec, value, strict=False, given=True)
        return spec.lax_adapter.validate_python(value)
    except ValidationError as error:
        raise _refused(error, _list_of(spec.type) if spec.shape == "many" else spec.type) from None


def read_value(spec: FieldSpec, value: Any, *, strict: bool) -> Any:
    """``value`` as one value of ``spec``: as it is when already of its class, else read by its type, ``strict`` as
    pydantic's strict mode is. A ``SKIPPED`` item of a ``Series`` is a gap, read around."""
    if value is SKIPPED:
        return value
    if spec.shape == "table":  # an input's table comes as given: a frame cut to its columns, or the records it gathered
        return None if value is None and spec.optional else read_table(spec, value, strict=strict, given=not strict)
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

    A table ``given`` (typed in, a caller's, or a person's answer) may also be a list of records, whose cells are read
    as they came, and keeps only the named columns, as an edge into the input would; a table a node returns or an
    edge brings keeps its other columns too. A table both ``given`` and ``strict`` is an answer: each cell is read from
    its JSON form, strictly, as a one-value answer is (``read_strictly``).
    """
    records: dict[str, list[Any]] = {}  # a given list of records, by column, before polars makes a dtype of it
    held: Collection[str]
    if given and not isinstance(value, pl.DataFrame):
        try:
            rows = RECORDS.validate_python(value)
        except ValidationError:
            raise FieldValueError("a table is a frame or a list of records, one object per row") from None
        names = list(spec.columns) if spec.columns is not None else list(dict.fromkeys(k for r in rows for k in r))
        present = {k for r in rows for k in r}
        records = {c: [r.get(c) for r in rows] for c in names if c in present or not rows}
        if spec.columns is None:
            return _frame_of(records)
        height, held = len(rows), records.keys()
    elif not isinstance(value, pl.DataFrame):
        raise FieldValueError(f"a table is a polars DataFrame, not a {type(value).__name__}")
    elif spec.columns is None:
        return value
    else:
        height, held = value.height, value.columns
    missing = [c for c in spec.columns if c not in held and height]  # a table of no rows holds every column, empty
    if missing:
        raise FieldValueError(f"the column '{missing[0]}' its row type declares is missing", column=missing[0])
    if isinstance(value, pl.DataFrame):  # a frame's column says how many blanks it holds without reading a cell
        for column, hint in spec.columns.items():
            if column in held and value[column].null_count() and not takes_a_blank(hint):
                raise FieldValueError(f"column '{column}': a cell is blank, and the column's type takes no blank", column=column)
    read: dict[str, list[object]] = {}
    for column, hint in spec.columns.items():
        if column not in held:
            read[column] = [None] * height
        elif column in records or not already(hint, _PYTHON_OF.get(value.schema[column].base_type(), object)):
            cells, adapter = records[column] if column in records else value[column].to_list(), spec.column_adapters[column]
            try:
                if given and strict:
                    read[column] = adapter.validate_json(to_json(cells, inf_nan_mode="constants"), strict=True)
                else:
                    read[column] = adapter.validate_python(cells, strict=strict)
            except PydanticSerializationError:
                raise FieldValueError(f"column '{column}': the value has no JSON form", column=column) from None
            except ValidationError as error:
                refused = _refused(error, _list_of(hint))
                raise FieldValueError(f"column '{column}': {refused.reason}", column=column, **refused.ctx) from None
    if given:
        return _frame_of({c: read[c] if c in read else value[c] for c in spec.columns})
    return value.with_columns(pl.Series(c, cells, strict=False) for c, cells in read.items()) if read else value


def _frame_of(columns: Mapping[str, Any]) -> pl.DataFrame:
    """A frame of ``columns``; ``FieldValueError`` when polars can't make one of them."""
    try:
        return pl.DataFrame(columns, strict=False)
    except Exception:  # noqa: BLE001  polars' own account may quote a cell
        raise FieldValueError("the records don't make a table") from None


def _refused(error: ValidationError, hint: Any) -> FieldValueError:
    """``error`` of reading a value of ``hint``, its errors without the value: each ``loc`` only as far as ``hint`` declares."""
    errors = value_free_errors(error)
    for said, raw in zip(errors, error.errors(include_url=False, include_input=False), strict=True):
        said["loc"] = _declared_loc(hint, raw["loc"])
    return FieldValueError(errors[0]["msg"], errors=errors)


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
        elif isinstance(step, str) and step in (fields := record_fields(hint)):
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


def record_fields(hint: Any) -> Mapping[str, Any]:
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
