"""``Series``, ``Table``, and ``Field``: one input or output of a node."""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import functools
import typing
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Generic, TypeVar, get_args, get_origin

import polars as pl
from pydantic import ConfigDict, GetCoreSchemaHandler, GetJsonSchemaHandler, JsonValue
from pydantic import Field as PydanticField
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, PydanticCustomError, PydanticSerializationError, core_schema

from conductor._internal.conductor_model import ConductorModel
from conductor._internal.problem import NodeDeclarationError
from conductor._internal.sentinels import REQUIRED, SKIPPED
from conductor._internal.value_type import (
    FieldValueError,
    Param,
    Type,
    list_of,
    split_hint,
    union_members,
    union_of,
)
from conductor.widgets import Widget

_T = TypeVar("_T")




#: Why a set is refused wherever a list is read or written: its order differs between processes.
SET_HAS_NO_ORDER = "a set has no order, so its items would come in another order in another process; give a list"


#: How a value of a class that is no list is made the list of items it holds, by that class.
_AS_LIST: dict[type, Callable[[Any], list[Any]]] = {tuple: list, range: list}


def read_as_list(cls: type, as_list: Callable[[Any], list[Any]]) -> None:
    """Declares that a value of ``cls`` given for a ``Series`` is the list ``as_list`` makes of it."""
    _AS_LIST[cls] = as_list


class _Series(Generic[_T]):
    """``Series[T]``: many values of ``T``, which a graph works through one at a time; a plain ``list[T]`` to the
    node and to a type checker."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Gives the schema of a list of items.

        From Python any sequence but a set is taken, as is a value of a class declared with ``read_as_list``, and
        an item that is ``SKIPPED`` is a gap kept where it is.
        """
        items = core_schema.list_schema(handler.generate_schema((get_args(source) or (Any,))[0]))

        def read(value: Any, read_items: Callable[[Any], list[Any]]) -> list[Any]:
            if isinstance(value, (set, frozenset)):
                raise PydanticCustomError("set_has_no_order", SET_HAS_NO_ORDER)
            for cls, as_list in _AS_LIST.items():
                if isinstance(value, cls):
                    value = as_list(value)
                    break
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


def written_table(frame: pl.DataFrame) -> dict[str, Any]:
    """Gives a table as conductor writes one: each column's name, the word for the type of its cells, and the cells.

    """
    return _written_columns(frame, frame.to_dict(as_series=False))


def _written_columns(frame: pl.DataFrame, cells: Mapping[str, list[Any]]) -> dict[str, Any]:
    """Gives ``frame`` as conductor writes a table, from the ``cells`` of each of its columns by name."""
    columns = []
    for (name, values), dtype in zip(cells.items(), frame.dtypes, strict=True):
        kind = dtype.to_python()
        columns.append({"name": name, "type": kind.__name__ if kind in _CELLS else "json", "values": values})
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


class HeldSeries(ABC):
    """A way a run holds the items of a ``Series`` other than as the list a node returned, declared by subclassing::

        class Records(HeldSeries, returned_as=Frame): ...

    A node that returns a ``returned_as`` for a ``Series`` output has it held as the subclass, where the items'
    type allows it.
    """

    __slots__ = ()

    _by_class_returned: ClassVar[dict[type, type[HeldSeries]]] = {}

    def __init_subclass__(cls, *, returned_as: type) -> None:
        """Keeps the subclass as how a ``returned_as`` returned for a ``Series`` is held."""
        super().__init_subclass__()
        HeldSeries._by_class_returned[returned_as] = cls

    @classmethod
    @abstractmethod
    def hint_of(cls, item: Type) -> Any | None:
        """Gives the hint a saved run reads and writes a ``Series`` of ``item`` held this way by; ``None`` when a
        ``Series`` of such items can't be held this way."""

    @classmethod
    @abstractmethod
    def holding(cls, returned: Any, item: Type) -> HeldSeries:
        """Holds what a node ``returned`` for a ``Series`` of ``item``, read strictly.

        Raises:
            FieldValueError: The items' type refuses what was returned.
        """


class Records(HeldSeries, Generic[_T], returned_as=pl.DataFrame):
    """A ``Series`` of records held by column: the table a node returned for it, read one record at a time.

    It is internal to a run. A node that takes a record, or the whole ``Series``, receives plain records, and
    an edge that names a field reads the column's cell with no record built. ``Records[R]`` is the type a saved
    run reads and writes one by: the table's own JSON form, where a ``Series`` held as a list is an array.

    Attributes:
        frame: The table whose rows are the records.
        columns: The cells of each column as a list, by name.
    """

    __slots__ = ("columns", "frame")

    def __init__(self, frame: pl.DataFrame) -> None:
        """Holds ``frame``, a table already read as the type of the records' table."""
        self.frame, self.columns = frame, frame.to_dict(as_series=False)

    @classmethod
    def hint_of(cls, item: Type) -> Any | None:
        """Gives ``Records[R]`` for items of a ``TypedDict`` ``R`` or of ``Any``; ``None`` for any other items,
        those that carry a limit included."""
        if item.limited or not (item.bare is Any or typing.is_typeddict(item.bare)):
            return None
        records: Any = cls
        return records[item.bare]

    @classmethod
    def holding(cls, returned: Any, item: Type) -> Records[Any]:
        """Holds the frame a node ``returned`` for a ``Series`` of ``item``, read as the table of such records."""
        table: Any = _Table
        return cls(Type(table[item.bare]).read(returned, strict=True))

    def __len__(self) -> int:
        return self.frame.height

    def __getitem__(self, place: int) -> dict[str, Any]:
        """Gives the record at ``place``."""
        return {name: cells[place] for name, cells in self.columns.items()}

    def records(self) -> list[dict[str, Any]]:
        """Gives every record, in order, as a node receives a ``Series`` of them."""
        names = list(self.columns)
        return [dict(zip(names, cells, strict=True)) for cells in zip(*self.columns.values(), strict=True)]

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Gives the schema a saved run reads and writes the records by: their table's."""
        table: Any = _Table
        return core_schema.json_or_python_schema(
            json_schema=core_schema.no_info_after_validator_function(cls, handler.generate_schema(table[get_args(source)[0]])),
            python_schema=core_schema.is_instance_schema(cls),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda held: _written_columns(held.frame, held.columns)),
        )


if TYPE_CHECKING:  # to a type checker a Series is the list a node gets, and a Table the frame, which keeps its row type
    from conductor._internal.table_as_a_type_checker_reads_it import Table

    type Series[T] = list[T]
else:
    Series, Table = _Series, _Table


class SeriesType(Type, of=_Series):
    """The type of a ``Series[T]``, with or without ``| None``."""

    @functools.cached_property
    def item(self) -> Type:
        """The type of one item."""
        return Type((get_args(self.options[0]) or (Any,))[0])

    @functools.cached_property
    def _named(self) -> tuple[str, str, tuple[type, ...]]:
        """The Series' id, which says its items' type unless that is ``Any``, its title, and the classes its
        items name."""
        return ("Series" if self.item.hint is Any else f"Series[{self.item.text}]"), "Series", self.item.classes

    @functools.cached_property
    def held_hint(self) -> Any:
        """The hint of a Series as a unit holds it: a list whose item on a branch not taken is blank, or one of
        the other ways a Series of its items may be held."""
        other_ways = [hint for holder in HeldSeries._by_class_returned.values() if (hint := holder.hint_of(self.item)) is not None]
        return union_of((list_of(union_of((self.item.hint, type(None)))), *other_ways))

    def read_returned(self, value: Any) -> Any:
        """Reads what a node returned for the Series, strictly: a list of its items, or, for a value of a class
        held another way, what holds it.

        Raises:
            FieldValueError: The type refuses the value.
        """
        for returned_as, holder in HeldSeries._by_class_returned.items():
            if isinstance(value, returned_as) and holder.hint_of(self.item) is not None:
                return holder.holding(value, self.item)
        return self.read(value, strict=True)


#: The words pydantic's ``loc`` names the form a table came in with, which are no place in the table.
_TABLE_FORMS = ("frame", "records", "written", "columns")


class TableType(Type, of=_Table, value_class=pl.DataFrame):
    """The type of a ``Table[R]``, with or without ``| None``. A ``polars.DataFrame`` written as a hint is the
    table of any columns it is."""

    keys_in_order: ClassVar[bool] = True  # a table is written by column, in order
    typed_in_advice: ClassVar[str] = "; give a table as a list of records"

    @functools.cached_property
    def _row(self) -> Any:
        """The row type ``R``; ``Any`` for a table of any columns."""
        return (get_args(self.options[0]) or (Any,))[0]

    @functools.cached_property
    def fields(self) -> Mapping[str, Any]:
        """The hint of each column the row type declares, by name; none for a table of any columns."""
        return Type(self._row).fields

    @functools.cached_property
    def _named(self) -> tuple[str, str, tuple[type, ...]]:
        """The table's id, which names its columns, its title, and no class."""
        columns = typing.get_type_hints(self._row) if typing.is_typeddict(self._row) else {}
        return f"Table[{', '.join(f'{column}: {Type(hint).text}' for column, hint in columns.items())}]" if columns else "Table", "Table", ()

    @functools.cached_property
    def record(self) -> Type:
        """The type of one record, which is what a table input gathers: the row type, and ``dict`` for a table of
        any columns, whose records are mappings of any keys."""
        return Type(dict if self._row is Any else self._row)

    def _walked(self, step: int | str) -> tuple[Type, bool, str | None] | None:
        """Follows one step of pydantic's ``loc`` into the table: the form it came in is no place, a record's
        place keeps the table's columns, and a column is named."""
        if step in _TABLE_FORMS:
            return self, False, None
        if isinstance(step, int):
            return self, True, None
        return (Type(self.fields[step]), True, step) if step in self.fields else None

    def _fault_in(self, part: str, kind: str, msg: str) -> str:
        """Says a fault in the column ``part``."""
        return f"the column '{part}' its row type declares is missing" if kind == "missing" else f"column '{part}': {msg}"

    @staticmethod
    def untyped_as_json(value: Any) -> Any:
        """Gives a frame met among values of no declared type as conductor writes a table."""
        return written_table(value)

    @staticmethod
    def untyped_equal(one: Any, other: Any) -> bool:
        """Says whether two frames hold the same cells; a frame has no plain ``==``."""
        return isinstance(other, pl.DataFrame) and one.equals(other)


read_as_list(pl.Series, pl.Series.to_list)  # a column of a frame given for a Series is the list of its cells


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
    def slot(self) -> Type:
        """The type of the field's value on one unit, as a saved run and a run's written outputs hold it: blank
        where the unit gave none, and otherwise as its type says a unit holds one."""
        return Type(union_of((self.whole.held_hint, type(None))))

    @functools.cached_property
    def column(self) -> Type:
        """The type of the field's values over many units: a list of what ``slot`` is the type of."""
        return Type(list_of(self.slot.hint))

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
    bare, param, others = split_hint(hint)
    beside_param = others  # what Annotated holds that is no Param, on the field and on a Series' item
    written, of_items = bare, Type(bare).item is not None
    if of_items:  # the item as written, its type variables still open
        (items,) = (option for option in union_members(written) if option is not type(None))
        bare, of_item, others = split_hint((get_args(items) or (Any,))[0])
        beside_param = (*beside_param, *others)
        if Type(bare).item is not None:
            raise NodeDeclarationError("a Series of Series; a field holds one level of items, so return one Series", code="nested_series")
        param = dataclasses.replace(of_item, **{f.name: value for f in dataclasses.fields(param) if (value := getattr(param, f.name)) is not None})
    if any(isinstance(mark, Widget) for mark in beside_param):  # pydantic would pass it by, and the field would show no control
        raise NodeDeclarationError("a widget in Annotated goes on Param: Annotated[..., Param(widget=...)]", code="widget_outside_param")
    kept = {**param.limits, **({} if param.choices is None else {"choices": param.choices})}
    metadata = (*others, *([Param(**kept)] if kept else []))
    one = Type(Annotated[(bare, *metadata)] if metadata else bare)
    if not of_items:
        return param, one, one
    whole = union_of((get_origin(option) or option)[one.hint] if option is not type(None) else option for option in union_members(written))
    return param, Type(whole), one
