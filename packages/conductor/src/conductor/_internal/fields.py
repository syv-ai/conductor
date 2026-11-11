"""A type, a field, and the record an editor reads of a field.

A ``Type`` is one annotation as conductor reads it, one object per hint: it names itself, reads a value, writes
it as JSON, and says which types may land on it and what that binds its type variables to. A ``Field`` is
one input or output of a node: a name, a hint and a default. Both are plain objects the engine reads on every
call. What an editor reads is a ``FieldDescription``, the one pydantic record here.

``Series[T]`` and ``Table[R]`` are generic classes pydantic carries by their own schema, so they stand in a
union, in a record and in each other like any other type. Validation is pydantic's throughout. The one thing
pydantic doesn't answer is whether a value of one type may land on another, which ``Type.fits`` does.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import functools
import operator
import types
import typing
import uuid
import weakref
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypeGuard,
    TypeVar,
    Union,
    get_args,
    get_origin,
)

import polars as pl
from pydantic import (
    BaseModel,
    ConfigDict,
    GetCoreSchemaHandler,
    GetJsonSchemaHandler,
    JsonValue,
    TypeAdapter,
    ValidationError,
)
from pydantic import Field as PydanticField
from pydantic.errors import PydanticInvalidForJsonSchema, PydanticSchemaGenerationError
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import (
    CoreSchema,
    PydanticCustomError,
    PydanticSerializationError,
    SchemaError,
    core_schema,
    from_json,
    to_json,
)

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.errors import NodeDeclarationError, value_free_errors
from conductor.widgets import Widget


class _Sentinel:
    """One named value that is falsy and stays itself through a copy or a pickle: ``REQUIRED`` or ``SKIPPED``.

    Attributes:
        name: The module-level name the value is kept under, which is also its repr.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return self.name

    def __reduce__(self) -> str:
        return self.name

    def __bool__(self) -> bool:
        return False


#: The default of an input nothing defaults.
REQUIRED: Any = _Sentinel("REQUIRED")

#: The value of an output on a branch its node did not take. Typed ``Any``
#: for a checker: it goes where a value would, on any output.
SKIPPED: Any = _Sentinel("SKIPPED")

#: The limits ``Param`` takes, by the words pydantic's ``Field`` uses.
_LIMITS = ("ge", "gt", "le", "lt", "multiple_of", "min_length", "max_length", "pattern")

#: What a bound may be: a number's, a date's, a time's or a duration's.
_Bound = float | decimal.Decimal | datetime.date | datetime.time | datetime.timedelta

#: Why a set is refused wherever a list is read or written: its order differs between processes.
SET_HAS_NO_ORDER = "a set has no order, so its items would come in another order in another process; give a list"


class Choice(ConductorModel):
    """One option a person may pick: ``Choice("ida", "Ida B.")``.

    A host subclasses it to carry more about an option; the fields it adds reach the question, and a dropdown,
    as written.

    Attributes:
        id: The value an answer picking it holds.
        title: What a person reads for it.
    """

    id: Any
    title: str

    def __init__(self, id: Any, title: str, **more: Any) -> None:  # noqa: A002  the word a host's option has
        """Makes a choice from its id and title given in that order.

        Args:
            id: The value an answer picking it holds.
            title: What a person reads for it.
            **more: The fields a host's subclass adds.
        """
        fields: dict[str, Any] = {"id": id, "title": title, **more}  # the base's own signature names no fields
        super().__init__(**fields)


@dataclass(frozen=True, kw_only=True)
class Param:
    """What a person reads about a field, and the limits a value must keep, inside ``Annotated``::

        times: Annotated[int, Param(title="Times", ge=0)] = 1

    The limits and choices hold wherever pydantic reads the type, in a field or in a host's own model.

    Attributes:
        title: What a person reads for the field; its name in everyday words when not given.
        description: A longer text about the field, for its help.
        widget: The ``conductor.widgets`` control a person edits the field with (on an output, answers it with); a
            host's own is a subclass of ``Widget``. Carried to the host and never read here.
        exclusive_group: Outputs of one node that share it are alternatives: on each call the node gives a value on
            one of them and ``SKIPPED`` on the rest.
        choices: The values the field takes, each given as a value or as a ``Choice`` to title it, and held as a
            ``Choice``. On a ``Literal`` field they title its values, one each; on any other they limit what it
            takes. A ``Dropdown`` draws them.
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
        """Holds each choice as a ``Choice``, a bare value titled by its text."""
        if self.choices is not None:
            object.__setattr__(self, "choices", tuple(c if isinstance(c, Choice) else Choice(c, str(c)) for c in self.choices))

    @functools.cached_property
    def limits(self) -> dict[str, Any]:
        """The limits set, by pydantic's words, as ``pydantic.Field`` takes them."""
        return {key: value for key in _LIMITS if (value := getattr(self, key)) is not None}

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Gives the schema of ``source`` held to these limits and choices.

        Args:
            source: The type the ``Param`` annotates.
            handler: Pydantic's, which builds the schema of a type.

        Returns:
            The schema pydantic reads a value by, in a field of a node or of a host's own model.

        Raises:
            NodeDeclarationError: A limit pydantic holds no rule for on ``source`` (``bound_invalid``), or choices
                that aren't each one value of it, once (``choices_invalid``).
        """
        limited_hint: Any = Annotated[source, PydanticField(**self.limits)] if self.limits else source
        limited = handler(limited_hint)
        # A limit pydantic holds no rule for on a type becomes a function of its own round the type's schema, which
        # fails on every value. A further function is a host's own schema, whose values only the host knows.
        held, functions, layer = set[str](), 0, limited
        while True:
            held |= layer.keys()
            functions += layer["type"].startswith("function")
            if "schema" not in layer:
                break
            layer = layer["schema"]  # type: ignore[typeddict-item]  # every schema that wraps another names it so
        unheld = sorted(self.limits.keys() - held)
        if unheld and functions <= len(unheld):
            raise NodeDeclarationError(
                f"{unheld} can't apply to {Type(source).id}, as pydantic holds no such limit on it; take them off, or check the value in the node",
                code="bound_invalid",
            )
        if self.choices is None:
            return limited
        reader = TypeAdapter(limited_hint)
        try:
            ids = [reader.validate_json(to_json(choice.id, inf_nan_mode="constants"), strict=True) for choice in self.choices]
        except PydanticSerializationError:
            raise NodeDeclarationError("a choice has no JSON form; give each choice once, as a value of the field's type", code="choices_invalid") from None
        except ValidationError as refused:
            raise NodeDeclarationError(
                f"a choice is not a value of the field's type: {value_free_errors(refused)[0]['msg']}; give each choice once, as a value of the field's type",
                code="choices_invalid",
            ) from None
        literal = [value for option in _options(source) if get_origin(option) is Literal for value in get_args(option)]
        distinct = {repr(i) for i in ids}
        if not ids or len(distinct) != len(ids):
            raise NodeDeclarationError("the choices hold an id twice, or none; give each choice once, as a value of the field's type", code="choices_invalid")
        if literal and distinct != {repr(value) for value in literal}:
            raise NodeDeclarationError(
                f"on a Literal the choices title its values, one each: {literal}; give each choice once, as a value of the field's type", code="choices_invalid",
            )
        if literal:  # a Literal's choices title its values
            return limited

        def one_of_the_choices(value: Any) -> Any:
            if value is not None and value not in ids:
                raise PydanticCustomError("not_a_choice", "the value is not one of the choices")
            return value

        return core_schema.no_info_after_validator_function(one_of_the_choices, limited)

    def __get_pydantic_json_schema__(self, schema: CoreSchema, handler: GetJsonSchemaHandler) -> JsonSchemaValue:
        """Gives the JSON schema of the limited type, with this ``title`` and ``description`` where a model's field carries them."""
        written = handler(schema)
        for key in ("title", "description"):
            if getattr(self, key) is not None:
                written[key] = getattr(self, key)
        return written


_NO_PARAM = Param()
_T = TypeVar("_T")


class _FromRun:
    """The mark ``FromRun`` leaves on a parameter. ``@node`` takes it off a whole parameter; anywhere else in a
    hint it refuses to be read."""

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Refuses a ``FromRun`` inside a field's type.

        Raises:
            NodeDeclarationError: Always (``from_run_misplaced``).
        """
        raise NodeDeclarationError(
            "FromRun is a whole parameter the host always gives, name: FromRun[Session]; anywhere else, a caller would fill it",
            code="from_run_misplaced",
        )


#: ``FromRun[X]``: a value the host hands the run by its type (a session, who is running), never an input. It takes
#: no edge, is not stored in the record, and is no part of what makes a result reusable. The run is given one
#: with ``from_run={X: value}``. To a type checker the parameter is an ``X``.
FromRun = Annotated[_T, _FromRun()]


def _not_a_bool(value: Any) -> Any:
    """Refuses a ``bool`` before pydantic reads it as the number Python counts it as."""
    if isinstance(value, bool):
        raise PydanticCustomError("bool_is_not_a_number", "a yes or no is not a number; give a number")
    return value


@dataclass(frozen=True)
class _NotABool:
    """The one rule that a yes or no is not a number: refuses a ``bool`` where pydantic alone reads ``True`` as 1.

    A ``Type`` adds it to a hint after asking pydantic what it makes of ``True`` there.

    Attributes:
        strict_too: Whether pydantic takes ``True`` in strict mode as well (a ``Literal`` of numbers). When false,
            a strict read is pydantic's own, untouched.
    """

    strict_too: bool

    def __get_pydantic_core_schema__(self, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Gives the schema of ``source`` behind the refusal."""
        plain = handler(source)
        guarded = core_schema.no_info_before_validator_function(_not_a_bool, plain)
        return guarded if self.strict_too else core_schema.lax_or_strict_schema(lax_schema=guarded, strict_schema=plain)


class _Series(Generic[_T]):
    """``Series[T]``: many values of ``T``, which a graph works through one at a time; a plain ``list[T]`` to the
    node and to a type checker."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Gives the schema of a list of items.

        From Python any sequence but a set is taken, as is a column of a frame, and an item that is ``SKIPPED``
        is a gap kept where it is.
        """
        items = core_schema.list_schema(handler.generate_schema((get_args(source) or (Any,))[0]))

        def read(value: Any, read_items: Callable[[Any], list[Any]]) -> list[Any]:
            if isinstance(value, (set, frozenset)):
                raise PydanticCustomError("set_has_no_order", SET_HAS_NO_ORDER)
            if isinstance(value, (tuple, range)):
                value = list(value)
            elif isinstance(value, pl.Series):
                value = value.to_list()
            if isinstance(value, list) and any(item is SKIPPED for item in value):
                kept = iter(read_items([item for item in value if item is not SKIPPED]))
                return [SKIPPED if item is SKIPPED else next(kept) for item in value]
            return read_items(value)

        return core_schema.json_or_python_schema(json_schema=items, python_schema=core_schema.no_info_wrap_validator_function(read, items))


#: The classes a table's column may be declared or written as, each with the polars dtype that holds it and how
#: its cells are read back from JSON. A column of any other kind (a list, a struct) is written under the word
#: ``json`` and read back as the JSON it was written as.
_CELLS: Mapping[type, tuple[Any, CoreSchema]] = {
    bool: (pl.Boolean, core_schema.bool_schema()), int: (pl.Int64, core_schema.int_schema()), float: (pl.Float64, core_schema.float_schema()),
    str: (pl.String, core_schema.str_schema()), datetime.date: (pl.Date, core_schema.date_schema()),
    datetime.datetime: (pl.Datetime, core_schema.datetime_schema()), datetime.time: (pl.Time, core_schema.time_schema()),
    datetime.timedelta: (pl.Duration, core_schema.timedelta_schema()), decimal.Decimal: (pl.Decimal, core_schema.decimal_schema()),
}

#: The polars dtype of a written column, by the word it says its type with.
_DTYPES = {kind.__name__: dtype for kind, (dtype, _) in _CELLS.items()}

#: A table as JSON: ``{"columns": [{"name", "type", "values"}]}``, its columns in order in an array, since a store
#: may give an object's keys back in another order.
_WRITTEN_TABLE = core_schema.typed_dict_schema({
    "columns": core_schema.typed_dict_field(core_schema.list_schema(core_schema.tagged_union_schema(
        {
            word: core_schema.typed_dict_schema({
                "name": core_schema.typed_dict_field(core_schema.str_schema()),
                "type": core_schema.typed_dict_field(core_schema.literal_schema([word])),
                "values": core_schema.typed_dict_field(core_schema.list_schema(core_schema.nullable_schema(cells))),
            }, extra_behavior="forbid")
            for word, cells in [*((kind.__name__, cells) for kind, (_, cells) in _CELLS.items()), ("json", core_schema.any_schema())]
        },
        discriminator="type",
    ))),
}, extra_behavior="forbid")


def written_table(frame: Any) -> dict[str, Any]:
    """Gives a table as conductor writes one: each column's name, the word for the type of its cells, and the cells.

    It is also what pydantic's ``to_json`` is given as its ``fallback``, to write a frame among values of no
    declared type.

    Raises:
        TypeError: ``frame`` is no ``polars.DataFrame``, which is how a fallback says it can't write a value.
    """
    if not isinstance(frame, pl.DataFrame):
        raise TypeError(type(frame).__name__)
    columns = []
    for (name, cells), dtype in zip(frame.to_dict(as_series=False).items(), frame.dtypes, strict=True):
        kind = dtype.to_python()
        columns.append({"name": name, "type": kind.__name__ if kind in _CELLS else "json", "values": cells})
    return {"columns": columns}


def _frame_of(columns: Iterable[tuple[str, list[Any], Any]]) -> pl.DataFrame:
    """Makes a frame of ``columns``, each a name, its cells and its polars dtype (``None`` to let polars find one).

    Raises:
        PydanticCustomError: Polars can't make a table of them (``not_a_table``): a name twice, lengths that
            differ, or cells of no one type.
    """
    try:
        return pl.DataFrame([pl.Series(name, cells, dtype=dtype, strict=False) for name, cells, dtype in columns])
    except (pl.exceptions.PolarsError, TypeError):  # polars' own account may quote a cell
        raise PydanticCustomError("not_a_table", "the columns don't make a table: one name twice, lengths that differ, or cells of no one type") from None


def _table_form(value: Any) -> str | None:
    """Says which of a table's three forms ``value`` is: a frame, a list of records, or the columns conductor writes."""
    if isinstance(value, pl.DataFrame):
        return "frame"
    return "written" if isinstance(value, Mapping) else "records" if isinstance(value, list) else None


class _Table(Generic[_T]):
    """``Table[R]``: a whole table whose columns the ``TypedDict`` ``R`` names; a ``polars.DataFrame`` to the node
    and to a type checker.

    A bare ``Table`` has any columns. A table is read from a frame, from a list of records (how gathered records
    and a person's answer arrive) or from the JSON conductor writes one as. A ``Table[R]`` read holds exactly
    ``R``'s columns, as a ``TypedDict`` read drops other keys: each present, each cell a value of its column's
    type, and blank only where that type takes a blank.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Gives the schema that reads a table in any of its three forms and writes it as its columns."""
        row = (get_args(source) or (Any,))[0]
        declared = typing.is_typeddict(row)
        columns = {name: Type(hint) for name, hint in typing.get_type_hints(row, include_extras=True).items()} if declared else {}
        cells = {name: handler.generate_schema(column.carried) for name, column in columns.items()}
        dtypes = {name: _CELLS[column.options[0]][0] if column.options[0] in _CELLS and len(column.options) == 1 else None for name, column in columns.items()}

        def from_records(records: list[dict[str, Any]]) -> pl.DataFrame:
            names = columns if declared else dict.fromkeys(key for record in records for key in record)
            return _frame_of((name, [record.get(name) for record in records], dtypes.get(name)) for name in names)

        def from_frame(frame: pl.DataFrame, read_columns: Callable[[dict[str, list[Any]]], dict[str, list[Any]]]) -> pl.DataFrame:
            if not frame.height:  # a table of no rows holds every column, empty and of its declared type
                kept = {name for name, dtype in frame.schema.items() if dtype != pl.Null}
                return pl.DataFrame([frame[name] if name in kept else pl.Series(name, [], dtype=dtypes[name]) for name in columns])
            for name, column in columns.items():
                if name not in frame.columns:
                    raise PydanticCustomError("column_missing", "the column '{column}' its row type declares is missing", {"column": name})
                if not column.optional and frame[name].null_count():
                    raise PydanticCustomError("column_blank", "column '{column}': a cell is blank, and the column's type takes no blank", {"column": name})
            unread = [name for name, column in columns.items() if not column.holds(Type(frame.schema[name].to_python()))]
            if unread:
                read = read_columns({name: frame[name].to_list() for name in unread})
                frame = frame.with_columns(pl.Series(name, values, strict=False) for name, values in read.items())
            return frame if frame.columns == list(columns) else frame.select(list(columns))

        def json_schema(_: CoreSchema, write: GetJsonSchemaHandler) -> JsonSchemaValue:
            column = [
                {"type": "object", "properties": {"name": {"const": name}, "values": {"type": "array", "items": write(cell)}}}
                for name, cell in cells.items()
            ]
            return {"type": "object", "properties": {"columns": {"type": "array", "prefixItems": column} if declared else {"type": "array"}}}

        record = core_schema.typed_dict_schema({name: core_schema.typed_dict_field(cell) for name, cell in cells.items()}) if declared else core_schema.dict_schema(core_schema.str_schema())
        by_column = core_schema.typed_dict_schema({name: core_schema.typed_dict_field(core_schema.list_schema(cell), required=False) for name, cell in cells.items()})
        as_frame = core_schema.no_info_wrap_validator_function(from_frame, by_column) if declared else core_schema.is_instance_schema(pl.DataFrame)
        written = core_schema.no_info_after_validator_function(lambda table: _frame_of((c["name"], c["values"], _DTYPES.get(c["type"])) for c in table["columns"]), _WRITTEN_TABLE)
        return core_schema.tagged_union_schema(
            {
                "frame": as_frame,
                "records": core_schema.no_info_after_validator_function(from_records, core_schema.list_schema(record)),
                "written": core_schema.chain_schema([written, as_frame]),
            },
            discriminator=_table_form,
            custom_error_type="not_a_table", custom_error_message="a table is a frame or a list of records, one object per row",
            serialization=core_schema.plain_serializer_function_ser_schema(written_table, when_used="json"),
            metadata={"pydantic_js_functions": [json_schema]},
        )


if TYPE_CHECKING:  # to a type checker a Series is the list a node gets, and a Table the frame, which keeps its row type
    from conductor._internal.table_as_a_type_checker_reads_it import Table

    type Series[T] = list[T]
else:
    Series, Table = _Series, _Table


class NamedType(Protocol):
    """A host's own type, as people see it. Every class of a host's that a node's field holds, and every class in
    ``NodeRegistry(types=)``, declares its id and title.

    A type may also declare ``type_accepts: ClassVar[tuple[type, ...]]``, the other classes whose values may land
    where it is asked for; it reads each such value itself. It is no member here, as a class that widens nothing
    declares none.

    Attributes:
        type_id: The name an editor and a stored graph know the type by, such as ``"money"``.
        type_title: What a person reads for the type, such as ``"Money"``.
    """

    type_id: ClassVar[str]
    type_title: ClassVar[str]


#: The id and title of each type conductor itself names. Every other class a field holds is a ``NamedType``.
_NAMES: Mapping[Any, tuple[str, str]] = {
    Any: ("Any", "Any value"), object: ("object", "Any value"), type(None): ("None", "Nothing"), bool: ("bool", "Yes or no"),
    int: ("int", "Whole number"), float: ("float", "Number"), decimal.Decimal: ("Decimal", "Decimal number"), str: ("str", "Text"),
    bytes: ("bytes", "Bytes"), datetime.date: ("date", "Date"), datetime.datetime: ("datetime", "Date and time"),
    datetime.time: ("time", "Time of day"), datetime.timedelta: ("timedelta", "Duration"), uuid.UUID: ("UUID", "UUID"),
    dict: ("dict", "Mapping"), list: ("list", "List"), tuple: ("tuple", "Tuple"), set: ("set", "Set"), frozenset: ("frozenset", "Set"),
    _Series: ("Series", "Series"), _Table: ("Table", "Table"),
}

#: The classes whose values are plain in JSON: no keys to order, and read back by their type as they were.
_SCALARS = frozenset({str, int, float, bool, type(None), datetime.date, datetime.datetime, datetime.time, datetime.timedelta})

#: The generic classes whose arguments ``Type`` reads as types of their own, so its rules reach their items.
_CONTAINERS = (list, dict, tuple, set, frozenset, _Series)


def _options(hint: Any) -> tuple[Any, ...]:
    """Gives the members of a union, or the hint itself as the one member of any other type."""
    return get_args(hint) if get_origin(hint) in (Union, types.UnionType) else (hint,)


def _list_of(item: Any) -> Any:
    """Gives ``list[item]`` for an ``item`` known only when a hint is read."""
    return types.GenericAlias(list, (item,))


def _union(options: Iterable[Any]) -> Any:
    """Gives the union of ``options``, which is the option itself when there is one."""
    return functools.reduce(operator.or_, options)


def _holds_a_model(schema: Any) -> bool:
    """Says whether a core schema holds a pydantic model or a dataclass anywhere in it."""
    if isinstance(schema, dict):
        return schema.get("type") in ("model", "dataclass") or any(map(_holds_a_model, schema.values()))
    return isinstance(schema, list) and any(map(_holds_a_model, schema))


def _a_number(cls: type) -> bool:
    """Says whether ``cls`` is a class of number that a yes or no must not be read as: ``int`` or ``float`` or a
    class of theirs, never ``bool``, though Python counts ``True`` as 1."""
    return issubclass(cls, (int, float)) and not issubclass(cls, bool)


def _is_alias(hint: Any) -> TypeGuard[typing.TypeAliasType]:
    """Says whether ``hint`` is a named type alias (``type X = ...``): typing's, or the one pydantic's own aliases are."""
    return isinstance(hint, (typing.TypeAliasType, type(JsonValue)))


def _is_record(cls: Any) -> bool:
    """Says whether ``cls`` is a class that declares fields: a pydantic model, a dataclass or a ``TypedDict``."""
    return isinstance(cls, type) and (issubclass(cls, BaseModel) or dataclasses.is_dataclass(cls) or typing.is_typeddict(cls))


@functools.cache
def _aliased(alias: Any) -> Any:
    """Gives what a named type alias stands for, the names it holds as text resolved where it was written."""
    holder = type("_", (), {"__annotations__": {"value": alias.__value__}, "__module__": alias.__module__})
    return typing.get_type_hints(holder, include_extras=True)["value"]


def _written(hint: Any) -> str:
    """One bare hint as ``Type.reads_as`` writes it; an ``Annotated`` inside it is written as the type it is."""
    origin, args = get_origin(hint), get_args(hint)
    if origin is Annotated:
        return Type(hint).reads_as
    if origin is Literal:
        return f"Literal{[repr(value) for value in args]}"
    if _is_alias(hint):
        return f"{hint.__module__}.{hint.__name__}"
    if args:
        name = "Union" if origin in (Union, types.UnionType) else _written(origin)
        return f"{name}[{', '.join(map(_written, args))}]"
    return f"{hint.__module__}.{hint.__qualname__}" if isinstance(hint, type) else repr(hint)


def _split(hint: Any) -> tuple[Any, Param, tuple[Any, ...]]:
    """Takes a hint apart into its bare type, its ``Param`` and its other metadata.

    ``Annotated[X, Param(...)] | None`` is ``X | None`` with that ``Param``; several ``Param``s are one, each
    argument the last one given. A ``polars.DataFrame`` is read as the ``Table`` of any columns it is.

    Returns:
        The hint with every ``Annotated`` taken off its members, the ``Param``s as one (an empty one for none),
        and whatever else ``Annotated`` held, which pydantic reads.
    """
    if type(hint) is type and hint is not pl.DataFrame:  # a plain class has nothing to take off, and most hints are one
        return hint, _NO_PARAM, ()
    options, params, others = [], list[Param](), list[Any]()
    for option in _options(hint):
        if get_origin(option) is Annotated:
            option, *metadata = get_args(option)
            params += [m for m in metadata if isinstance(m, Param)]
            others += [m for m in metadata if not isinstance(m, Param)]
            inner, inner_param, inner_others = _split(option)
            option, params, others = inner, [inner_param, *params], [*inner_others, *others]
        options.append(_Table if option is pl.DataFrame else option)
    merged = {f.name: value for param in params for f in dataclasses.fields(param) if (value := getattr(param, f.name)) is not None}
    return _union(options), Param(**merged) if merged else _NO_PARAM, tuple(others)


def _fit(target: Any, source: Any, bound: dict[TypeVar, Any], exact: bool) -> bool:
    """Says whether a value of ``source`` may land on ``target``, binding ``target``'s type variables as it walks.

    Args:
        target: The hint asked for, which may hold type variables.
        source: The hint arriving, which holds none.
        bound: What each variable is bound to so far, added to here; two places sharing a variable join.
        exact: Ask instead whether every ``source`` value already is a ``target`` as it is, with nothing to read.
    """
    target, source = _split(target)[0], _split(source)[0]
    if target == source:
        return True
    if _is_alias(target) or _is_alias(source):  # an alias is read as what it stands for
        return _fit(_aliased(target) if _is_alias(target) else target, _aliased(source) if _is_alias(source) else source, bound, exact)
    if isinstance(target, TypeVar):
        if target.__bound__ is not None and not _fit(target.__bound__, source, {}, exact):
            return False
        bound[target] = bound[target] | source if target in bound else source
        return True
    if target is Any:
        return True
    if source is Any:
        return not exact
    targets, sources = _options(target), [option for option in _options(source) if option is not type(None)]
    if len(targets) > 1 or len(sources) != len(_options(source)):
        for one in sources:
            for option in targets:
                if _fit(option, one, trial := dict(bound), exact):
                    bound.update(trial)
                    break
            else:
                return False
        return True
    if get_origin(source) is Literal:
        if get_origin(target) is Literal:  # a narrower Literal lands on one holding every value it may be
            return {(type(v), v) for v in get_args(source)} <= {(type(v), v) for v in get_args(target)}
        return all(_fit(target, type(value), bound, exact) for value in get_args(source))
    target_class, source_class = get_origin(target) or target, get_origin(source) or source
    if typing.is_typeddict(source):
        held = typing.get_type_hints(source)
        if typing.is_typeddict(target):  # read by its keys: every key the target names, each of a type that fits
            wanted = typing.get_type_hints(target)
            same = not exact or wanted.keys() == held.keys()  # a read drops the keys the target doesn't name
            return same and target.__required_keys__ <= source.__required_keys__ and all(key in held and _fit(hint, held[key], bound, exact) for key, hint in wanted.items())
        if target_class is not dict:
            return False
        key, value = get_args(target) or (Any, Any)
        return _fit(key, str, bound, exact) and all(_fit(value, hint, bound, exact) for hint in held.values())
    if typing.is_typeddict(target) or not (isinstance(target_class, type) and isinstance(source_class, type)):
        return False
    if issubclass(source_class, bool) and _a_number(target_class):
        return False
    # a host's type may name other classes it takes; a class that names none has no such attribute
    widened = not exact and ((target_class is float and issubclass(source_class, int)) or issubclass(source_class, getattr(target_class, "type_accepts", ())))
    if not (issubclass(source_class, target_class) or widened):
        return False
    wanted_args, held_args = get_args(target), get_args(source)
    if not held_args:  # a bare generic arriving says nothing of its items
        return not (exact and wanted_args)
    return len(wanted_args) != len(held_args) or all(_fit(a, b, bound, exact) for a, b in zip(wanted_args, held_args, strict=True))


def _substituted(hint: Any, bound: Mapping[TypeVar, Any]) -> Any:
    """Gives ``hint`` with each type variable replaced by what it is bound to, else its own bound, else ``Any``."""
    if isinstance(hint, TypeVar):
        return bound[hint] if hint in bound else Any if hint.__bound__ is None else hint.__bound__
    if isinstance(hint, type):
        return hint
    variables = getattr(hint, "__parameters__", ())  # typing's own, on every generic hint
    return hint[tuple(_substituted(variable, bound) for variable in variables)] if variables else hint


class Type:
    """One annotation as conductor reads it. ``Type(hint)`` gives the one object of a hint, so what a hint costs
    to read is paid once, and nothing is built until it is asked for.

    The one object is kept for as long as something holds it (a field, a loaded graph), so a hint made while a
    graph loads is let go with the graph.

    Attributes:
        hint: The annotation, with its ``Annotated`` metadata and its type variables as written.
    """

    hint: Any
    _made: ClassVar[weakref.WeakValueDictionary[Any, Type]] = weakref.WeakValueDictionary()

    def __new__(cls, hint: Any) -> Type:
        """Gives the ``Type`` of ``hint``: the one in use for an equal hint, else a new one."""
        try:
            return cls._made[hint]
        except KeyError:
            made = cls._made[hint] = super().__new__(cls)
        except TypeError:  # a hint that can't be hashed (a Param whose Choice holds a dict) is read each time
            made = super().__new__(cls)
        made.hint = hint
        return made

    def __repr__(self) -> str:
        return f"Type({self.hint!r})"

    @functools.cached_property
    def closed(self) -> Any:
        """The hint with each type variable read as its bound, else ``Any``."""
        return _substituted(self.hint, {})

    @functools.cached_property
    def _parts(self) -> tuple[Any, Param, tuple[Any, ...]]:
        """The closed hint as ``_split`` takes it apart."""
        return _split(self.closed)

    @property
    def bare(self) -> Any:
        """The closed hint with ``Annotated`` taken off its members, which is what a hook is given as the type arriving."""
        return self._parts[0]

    @functools.cached_property
    def options(self) -> tuple[Any, ...]:
        """The members of the bare hint, ``| None`` taken off."""
        return tuple(option for option in _options(self.bare) if option is not type(None))

    @functools.cached_property
    def optional(self) -> bool:
        """Whether a blank (``None``) may arrive: ``X | None``, ``Any``, or a type alias that stands for either."""
        options = _options(self.bare)
        return Any in options or type(None) in options or any(_is_alias(option) and Type(_aliased(option)).optional for option in options)

    @functools.cached_property
    def limited(self) -> bool:
        """Whether a value of the right class may still be refused: the hint holds a limit, choices or other metadata."""
        _, param, others = self._parts
        return bool(others or param.limits) or param.choices is not None

    @functools.cached_property
    def reads_as(self) -> str:
        """A text that is the same in every process and differs between any two types that read a value differently.

        A class is written by its module and name, a generic by its arguments, a ``Literal`` by its values, a type
        alias by its name, and a limit or a choice by what it holds. A title, a description and a widget are left
        out, as they change nothing that is read. What a unit's result rests on about a field is this text.
        """
        bare, param, others = self._parts
        held = [f"{limit}={value!r}" for limit, value in sorted(param.limits.items())]
        if param.choices is not None:
            held.append(f"choices={[choice.id for choice in param.choices]!r}")
        held += [repr(other) for other in others]
        return f"{_written(bare)}{held}" if held else _written(bare)

    @functools.cached_property
    def variables(self) -> tuple[TypeVar, ...]:
        """The type variables the hint leaves open, which an edge binds."""
        if isinstance(self.hint, TypeVar):
            return (self.hint,)
        return () if isinstance(self.hint, type) else getattr(self.hint, "__parameters__", ())  # typing's own, on every generic hint

    @functools.cached_property
    def id(self) -> str:
        """The name an editor and a stored graph know the type by, ``| None`` off.

        A class is known by the id it declares, a union by its members' (``money | int``), a generic by its
        class and arguments (``list[int]``, a bare ``list`` for ``list[Any]``), a ``Literal`` by the types of its values, a ``TypedDict`` as
        ``record``, a table by its columns and a type alias by its own name. A type variable is read as its bound,
        else ``Any``.
        """
        return self._named[0]

    @property
    def text(self) -> str:
        """The id with its blank said (``str | None``): how a type inside another, and a type an edge brings, is named."""
        return f"{self.id} | None" if type(None) in _options(self.bare) and self.options else self.id

    @functools.cached_property
    def title(self) -> str:
        """What a person reads for the type, from the titles its classes declare."""
        return self._named[1]

    @functools.cached_property
    def classes(self) -> tuple[type, ...]:
        """The classes ``id`` names, each once: what a registry holding the type offers by id."""
        return self._named[2]

    @functools.cached_property
    def _named(self) -> tuple[str, str, tuple[type, ...]]:
        """The type's id, its title and the classes they name."""
        members = list(self.options) or [type(None)]
        if len(members) > 1:
            named = [Type(member)._named for member in members]
            return " | ".join(n[0] for n in named), " | ".join(n[1] for n in named), tuple(dict.fromkeys(c for n in named for c in n[2]))
        (one,) = members
        origin, args = get_origin(one) or one, get_args(one)
        if _is_alias(one):
            return one.__name__, one.__name__, ()
        if origin is Literal:
            return Type(_union(dict.fromkeys(type(value) for value in args)))._named
        if typing.is_typeddict(one):  # a record declares no name of its own
            return "record", "Record", ()
        own_id, title = _NAMES[origin] if origin in _NAMES else (origin.type_id, origin.type_title)
        own = () if origin in (Any, type(None), _Series, _Table) else (origin,)
        if origin is _Table:  # a table is known by its columns, and names no class
            columns = typing.get_type_hints(args[0]) if args and typing.is_typeddict(args[0]) else {}
            return f"Table[{', '.join(f'{column}: {Type(hint).text}' for column, hint in columns.items())}]" if columns else own_id, title, ()
        if all(arg is Any for arg in args):  # a generic that says nothing of its items is said bare
            return own_id, title, own
        inner = [Type(arg) for arg in args if arg is not Ellipsis]
        said = ", ".join("..." if arg is Ellipsis else Type(arg).text for arg in args)
        return f"{own_id}[{said}]", title, tuple(dict.fromkeys((*own, *(c for of in inner for c in of.classes))))

    @functools.cached_property
    def _built(self) -> tuple[Any, TypeAdapter[Any]]:
        """The hint pydantic reads for the type, and pydantic's adapter of it.

        A container's arguments are each read as a type of their own. A type that takes a number and nothing a
        ``bool`` already is refuses a ``bool``, which pydantic alone reads as 1.

        Raises:
            NodeDeclarationError: A ``Param`` that doesn't fit, a misplaced ``FromRun``, or a type pydantic has no
                schema for (``type_not_carried``).
        """
        bare, param, others = self._parts
        options = []
        for option in _options(bare):
            if get_origin(option) in _CONTAINERS:
                option = get_origin(option)[tuple(arg if arg is Ellipsis else Type(arg).carried for arg in get_args(option))]
            options.append(option)
        read: Any = _union(options)
        if metadata := (*others, *([param] if param.limits or param.choices is not None else [])):
            read = Annotated[(read, *metadata)]
        literals = [option for option in options if get_origin(option) is Literal]
        numbers = [o for o in options if isinstance(o, type) and _a_number(o)] + [o for o in literals if all(_a_number(type(v)) for v in get_args(o))]
        takes_a_bool = any(o in (bool, Any, object) for o in options) or any(isinstance(v, bool) for o in literals for v in get_args(o))
        if numbers and not takes_a_bool:
            read = Annotated[read, _NotABool(strict_too=any(o in literals for o in numbers))]
        try:  # a record is written by its own config; any other type writes a float that isn't finite as its text
            adapter = TypeAdapter(read) if len(options) == 1 and _is_record(options[0]) else TypeAdapter(read, config=ConfigDict(ser_json_inf_nan="strings"))
        except SchemaError as error:  # a limit of the wrong kind for the type, such as ge="a" on an int
            raise NodeDeclarationError(f"the limits {param.limits!r} don't fit {self.id}, so change one: {error}", code="bound_invalid") from None
        except PydanticSchemaGenerationError:
            raise NodeDeclarationError(
                f"conductor can't check or write {self.id} as JSON; use a type pydantic knows, such as a pydantic model or a dataclass, "
                "or give the class its own __get_pydantic_core_schema__",
                code="type_not_carried",
            ) from None
        return read, adapter

    @property
    def carried(self) -> Any:
        """The hint pydantic reads for the type, for a hint that holds it (a list's item, a table's column)."""
        return self._built[0]

    @property
    def adapter(self) -> TypeAdapter[Any]:
        """Pydantic's reader and writer of the type."""
        return self._built[1]

    @functools.cached_property
    def _validate(self) -> Callable[..., Any]:
        """Pydantic-core's own validator, which costs a quarter of the adapter's method on each call."""
        return self.adapter.validator.validate_python

    @functools.cached_property
    def json_schema(self) -> dict[str, Any]:
        """The JSON schema of one value; ``{}`` for a type that writes none, which an editor offers no control for."""
        try:
            return self.adapter.json_schema()
        except PydanticInvalidForJsonSchema:
            return {}

    @functools.cached_property
    def choices(self) -> tuple[Choice, ...] | None:
        """The values the type takes when it names them: ``Param(choices=)``, else a ``Literal``'s. Each id is the
        JSON an answer picking it reads as."""
        given = self._parts[1].choices
        if given is None and len(self.options) == 1 and get_origin(self.options[0]) is Literal:
            given = tuple(Choice(value, str(value)) for value in get_args(self.options[0]))
        if given is None:
            return None
        return tuple(choice.model_copy(update={"id": self.to_json(self.read_json(choice.id))}) for choice in given)

    @functools.cached_property
    def exact_classes(self) -> tuple[type, ...]:
        """The classes whose instances, of exactly that class, are taken as they are with no reading.

        Empty when a value must always be read: an option is no plain class (``list[int]``, a ``Literal``, a
        ``TypedDict``, ``Any``, a table), or the type is ``limited``.
        """
        options = _options(self.bare)
        plain = all(isinstance(option, type) and not typing.is_typeddict(option) and option not in (Any, _Table, _Series) for option in options)
        return options if plain and not self.limited else ()

    @functools.cached_property
    def hashes_plainly(self) -> bool:
        """Whether its values are plain in JSON: no keys to order, and no float beside a text it could be read as. So
        what a unit received hashes as it is written, and a float that isn't finite reads back from its text."""
        classes = {option for option in _options(self.bare) if (get_origin(option) or option) is not _Table}  # a table is written by column, in order
        return classes <= _SCALARS and not {float, str} <= classes

    @functools.cached_property
    def record(self) -> Type | None:
        """The type of one record of a table (``| None`` aside), which is what a table input gathers; ``Any`` for a
        table of any columns, and ``None`` for a type that is no table."""
        if len(self.options) != 1 or (get_origin(self.options[0]) or self.options[0]) is not _Table:
            return None
        return Type((get_args(self.options[0]) or (Any,))[0])

    @functools.cached_property
    def fields(self) -> Mapping[str, Any]:
        """The hint of each field a record declares, by name: a model's, a dataclass's, a ``TypedDict``'s, and a
        table's columns. None for any other type."""
        one = self.options[0] if len(self.options) == 1 else None
        if get_origin(one) is _Table:
            one = get_args(one)[0]
        return {name: hint for name, hint in typing.get_type_hints(one).items() if get_origin(hint) is not ClassVar} if _is_record(one) else {}

    def read(self, value: Any, *, strict: bool = False) -> Any:
        """Reads ``value`` as the type holds it: as it is when exactly of a plain class, else by pydantic.

        Args:
            value: What a node returned (``strict``), or what a caller gave, a person typed in or an edge brought.
            strict: Pydantic's strict mode: a value already of its type, never ``"12"`` for an ``int``.

        Raises:
            FieldValueError: The type refuses the value.
        """
        if type(value) in self.exact_classes:
            return value
        try:
            return self._validate(value, strict=strict)
        except ValidationError as error:
            raise self._refused(error) from None

    def read_json(self, value: Any) -> Any:
        """Reads ``value`` strictly from its JSON form, as a person's answer and a declared choice are.

        A yes or no is ``true`` or ``false`` and a number is a number, never a text that spells one; a date is
        still its ISO text, as JSON writes it.

        Raises:
            FieldValueError: The value has no JSON form, or the type refuses it.
        """
        try:
            return self.adapter.validator.validate_json(to_json(value, inf_nan_mode="constants", fallback=written_table), strict=True)
        except PydanticSerializationError:
            raise FieldValueError("the value has no JSON form") from None
        except ValidationError as error:
            raise self._refused(error) from None

    def to_json(self, value: Any) -> JsonValue:
        """Writes ``value``, one the type holds, in its JSON form; a float that isn't finite as its text.

        Raises:
            PydanticSerializationError: The value has no JSON form.
        """
        written: JsonValue = from_json(to_json(self.adapter.serializer.to_python(value, mode="json"), inf_nan_mode="strings"))
        return written

    @functools.cached_property
    def saved_schema(self) -> CoreSchema:
        """The schema a saved run reads and writes a value of the type by: pydantic's own of the type.

        A float that isn't finite is written as its text, which a ``float`` reads back as the float and a type
        that takes a text keeps as the text. A pydantic model and a dataclass write by their own config whatever
        stands around them, and by default write such a float as ``null``, which a ``float`` field doesn't read
        back. So a type that holds one is written as ``to_json`` writes it: first as Python's JSON values, which
        keep the float.
        """
        schema = self.adapter.core_schema
        if not _holds_a_model(schema):
            return schema
        as_json = core_schema.plain_serializer_function_ser_schema(functools.partial(self.adapter.serializer.to_python, mode="json"))
        if schema["type"] == "definitions":  # what pydantic defined beside the type stays beside it
            return {**schema, "schema": {**schema["schema"], "serialization": as_json}}  # type: ignore[typeddict-unknown-key]
        return {**schema, "serialization": as_json}  # type: ignore[typeddict-unknown-key]

    def fits(self, source: Type) -> dict[TypeVar, Any] | None:
        """Says whether a value of ``source`` may land on this type, and what that binds its type variables to.

        Returns:
            What each variable binds to (``{}`` for a type that has none); ``None`` when the value may not land.
            Two places sharing a variable join, and a source outside a variable's bound does not fit.
        """
        bound: dict[TypeVar, Any] = {}
        return bound if _fit(self.hint, source.closed, bound, False) else None

    def holds(self, source: Type) -> bool:
        """Says whether every value of ``source`` already is one of this type as it is, so nothing is read per call."""
        return not self.limited and _fit(self.closed, source.closed, {}, True)

    def bind(self, bound: Mapping[TypeVar, Any]) -> Type:
        """Gives this type with its variables replaced by what ``bound`` binds them to."""
        return Type(_substituted(self.hint, bound))

    def _refused(self, error: ValidationError) -> FieldValueError:
        """Makes pydantic's ``error`` one that quotes nothing of the value.

        Each ``loc`` is kept only as far as the type itself names: an index into a list, a field of a record, a
        record and a column of a table. It stops at the first other step, such as a key of a dict, as a key is
        the value's own. A fault in a table's column says which column.
        """
        errors = value_free_errors(error)
        columns: list[str] = []
        for said, raw in zip(errors, error.errors(include_url=False, include_input=False), strict=True):
            at, kept, column = self, list[int | str](), None
            for step in raw["loc"]:
                origin = (get_origin(at.options[0]) or at.options[0]) if len(at.options) == 1 else None
                if origin is _Table and step in ("frame", "records", "written", "columns"):
                    continue  # the form the table came in, which is no place in it
                if origin is _Table and isinstance(step, str) and step in at.fields:
                    at, column = Type(at.fields[step]), step
                elif origin is _Table and isinstance(step, int):
                    pass  # a record's place; its columns are the table's
                elif isinstance(step, int) and isinstance(origin, type) and issubclass(origin, (list, tuple, set, frozenset, _Series)):
                    args = [arg for arg in get_args(at.options[0]) if arg is not Ellipsis]
                    at = Type(args[min(step, len(args) - 1)] if args else Any)
                elif isinstance(step, str) and step in at.fields:
                    at = Type(at.fields[step])
                else:
                    break
                kept.append(step)
            said["loc"] = tuple(kept)
            if column is not None:
                columns.append(column)
                said["msg"] = f"the column '{column}' its row type declares is missing" if raw["type"] == "missing" else f"column '{column}': {said['msg']}"
            elif whole := raw.get("ctx", {}).get("column"):  # a fault of a whole column of a frame
                columns.append(whole)
                said["loc"] = (whole,)
        return FieldValueError(errors[0]["msg"], errors=errors, **({"column": columns[0]} if columns else {}))


class FieldValueError(Exception):
    """A value its type refuses.

    Attributes:
        reason: Why, quoting nothing of the value.
        code: The problem it is where it was met: ``invalid_output`` for a node's return, ``invalid_edge_value``
            for what an edge brought.
        ctx: What a problem says beside the reason: pydantic's ``errors`` without the value, the field's ``name``,
            and the ``column`` of a table the first fault is in.
    """

    def __init__(self, reason: str, *, code: str = "", **ctx: Any) -> None:
        super().__init__(reason)
        self.reason, self.code, self.ctx = reason, code, ctx


class FieldDescription(ConductorModel):
    """An input or output as an editor reads it, and a question a waiting call asks. It reads back what it wrote.

    Attributes:
        name: The parameter's name, or the output's.
        title: What a person reads as its name.
        description: A longer text for its help.
        many: Whether it is a ``Series``: an output that makes items, an input that gathers them.
        type: The id of one value's type, a ``Series``' item's; a table's names its columns.
        type_variables: The names of the type variables its hint leaves open, which an edge binds; two fields that
            share a name have the same type once it is bound.
        json_schema: The JSON schema of one value, written as ``schema``; a table's columns are in it.
        required: Whether nothing defaults it.
        default: The default as JSON; ``None`` when ``required``.
        choices: The values it takes when it names them, each ``{"id", "title"}`` with whatever a host's ``Choice``
            subclass adds: from a ``Literal``, ``Param(choices=)`` or, on a question, the ``Ask``.
        widget: The control a person edits it with, as JSON.
        exclusive_group: The group of alternative outputs it belongs to.
    """

    model_config = ConfigDict(
        frozen=True, extra="forbid", validate_by_name=True, validate_by_alias=True, serialize_by_alias=True, ser_json_inf_nan="strings",
    )

    name: str
    title: str
    description: str | None = None
    many: bool
    type: str
    type_variables: list[str] = []
    json_schema: dict[str, Any] = PydanticField(alias="schema")
    required: bool
    default: JsonValue = None
    choices: list[dict[str, Any]] | None = None
    widget: dict[str, Any] | None = None
    exclusive_group: str | None = None


@dataclass(frozen=True)
class Field:
    """One input or output of a node: ``Field("times", Annotated[int, Param(ge=0)], 1)``.

    The constructor is the only way to a field: one renamed, given another default or bound to a type variable
    is a new ``Field`` of the changed ``name``, ``default`` or ``hint``, so nothing it holds can go stale.
    What the hint costs to read is kept per hint, so a second field of it is cheap.

    Attributes:
        name: The parameter's name, or the output's.
        hint: The annotation as written.
        default: The default as its type holds it; ``REQUIRED`` for none, ``SKIPPED`` for an output on a branch
            not taken, ``None`` for a blank.
        param: The ``Param``s of the hint as one; an empty one when it has none.
        type: The type of one value: a ``Series``' item, else ``whole``.
        whole: The type of the field's whole value, which reads it and writes it: a ``Series``' list.
        many: Whether it is a ``Series``: an output that makes items, an input that gathers them.
        default_json: The default as JSON; ``None`` where there is none to propose.
    """

    name: str
    hint: Any = field(repr=False)
    default: Any = REQUIRED
    param: Param = field(init=False, compare=False, repr=False)
    type: Type = field(init=False, compare=False)
    whole: Type = field(init=False, compare=False, repr=False)
    many: bool = field(init=False, compare=False)
    default_json: JsonValue = field(init=False, compare=False, repr=False, default=None)

    def __post_init__(self) -> None:
        """Reads the hint and the default once, refusing what conductor can't carry.

        Raises:
            NodeDeclarationError: Naming the field: a type pydantic has no schema for, a ``Series`` of ``Series``,
                a ``Param`` that doesn't fit, or a default its type doesn't take or can't write as JSON.
            PydanticSerializationError: The widget has no JSON form.
        """
        keep = functools.partial(object.__setattr__, self)
        try:
            param, whole, one = _field_parts(self.hint)
            keep("param", param)
            keep("whole", whole)
            keep("type", one)
            keep("many", one is not whole)
            whole.adapter  # noqa: B018  a type pydantic can't carry fails here
            whole.id  # noqa: B018  a class that declares no type_id fails here, as Python's own AttributeError
            if self.default is None and not whole.optional:
                raise NodeDeclarationError("the default None is a blank; write its type with | None", code="default_invalid")
            if self.default is not REQUIRED and self.default is not SKIPPED and self.default is not None:
                try:
                    keep("default", whole.read(self.default))
                    keep("default_json", whole.to_json(self.default))
                except (FieldValueError, PydanticSerializationError, UnicodeDecodeError) as error:
                    raise NodeDeclarationError(
                        f"the default {self.default!r} is not a value of {whole.id} that it writes as JSON, so give one that is: {error}", code="default_invalid",
                    ) from None
        except NodeDeclarationError as error:
            raise NodeDeclarationError(f"{self.name}: {error.message}", code=error.code) from None
        if self.param.widget is not None:
            self.description  # noqa: B018  a widget with no JSON form fails here

    def bind(self, bound: Mapping[TypeVar, Any]) -> Field:
        """Gives this field with its type variables replaced by what ``bound`` binds them to, one nothing binds by
        its bound, else ``Any``; the field itself when its hint holds none."""
        return Field(self.name, Type(self.hint).bind(bound).hint, self.default) if self.whole.variables else self

    @functools.cached_property
    def keeps_blanks(self) -> bool:
        """Whether a field that gathers keeps the place of an item with no value, as a blank: a ``Series`` whose
        item type takes one. A table never does, as a record is never blank."""
        return self.many and self.type.optional

    @functools.cached_property
    def column(self) -> Type:
        """The type of the field's values over many units, as a saved run holds them: a list in which a unit that
        gave none, and a ``Series``' item on a branch not taken, are blank."""
        one = _list_of(_union((self.type.hint, type(None)))) if self.many else self.whole.hint
        return Type(_list_of(_union((one, type(None)))))

    @functools.cached_property
    def description(self) -> FieldDescription:
        """The field as an editor reads it, made once."""
        choices, widget = self.type.choices, self.param.widget
        return FieldDescription(
            name=self.name, title=self.param.title or self.name.replace("-", " ").replace("_", " ").strip().capitalize(),
            description=self.param.description, many=self.many, type=self.type.id, type_variables=[variable.__name__ for variable in self.whole.variables],
            schema=self.type.json_schema, required=self.default is REQUIRED, default=self.default_json,
            choices=None if choices is None else [choice.model_dump(mode="json") for choice in choices],
            widget=None if widget is None else widget.model_dump(mode="json"), exclusive_group=self.param.exclusive_group,
        )


def _field_parts(hint: Any) -> tuple[Param, Type, Type]:
    """Reads a field's hint: its ``Param``, the type of its whole value and the type of one value.

    A type is made of the hint and what a value must keep (limits, choices, other metadata), so fields that
    differ only in what a person reads share one. A ``Series``' limits and choices are its items', wherever in
    the hint they were written.

    Raises:
        NodeDeclarationError: A ``Series`` of ``Series`` (``nested_series``), or a widget in ``Annotated`` that is
            not on a ``Param`` (``widget_outside_param``).
    """
    bare, param, others = _split(hint)
    beside_param = others  # what Annotated holds that is no Param, on the field and on a Series' item
    series = [option for option in _options(bare) if (get_origin(option) or option) is _Series]
    optional = type(None) in _options(bare)  # of the whole Series, when it is one
    if series:
        bare, of_item, others = _split((get_args(series[0]) or (Any,))[0])
        beside_param = (*beside_param, *others)
        if any((get_origin(option) or option) is _Series for option in _options(bare)):
            raise NodeDeclarationError("a Series of Series; a field holds one level of items, so return one Series", code="nested_series")
        param = dataclasses.replace(of_item, **{f.name: value for f in dataclasses.fields(param) if (value := getattr(param, f.name)) is not None})
    if any(isinstance(mark, Widget) for mark in beside_param):  # pydantic would pass it by, and the field would show no control
        raise NodeDeclarationError("a widget in Annotated goes on Param: Annotated[..., Param(widget=...)]", code="widget_outside_param")
    kept = {**param.limits, **({} if param.choices is None else {"choices": param.choices})}
    metadata = (*others, *([Param(**kept)] if kept else []))
    one = Type(Annotated[(bare, *metadata)] if metadata else bare)
    if not series:
        return param, one, one
    series_of: Any = _Series
    whole: Any = series_of[one.hint]
    return param, Type(_union((whole, type(None))) if optional else whole), one
