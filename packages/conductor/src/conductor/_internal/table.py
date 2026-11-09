"""``Table`` — many rows of named, typed columns, as one value.

Every level conductor runs over is a table: the rows of an index are its
rows and the fields on it its columns. A ``Table`` is that same thing held
as one value, so a node can take a whole table and hand one back: a CSV
read in, a filter, a sort. Inside it is a Polars ``DataFrame``; a node
reads and returns frames.

A table type names its columns, each with a scalar ``DType``::

    class Contact(TypedDict):
        name: Text
        email: Text

    Table[Contact]                                    # the columns, written as a type
    Table.with_columns({"name": Text, "email": Text})  # the same type, for a hook

Bare ``Table`` is a table whose columns are known only when it arrives (a
CSV nobody described). A column's Polars type comes from what its ``DType``
is built on — ``str`` a string column, ``float``, ``int``, ``bool`` and
``datetime.date`` theirs — and a type says it itself (``DType.column``)
only where that base would be wrong, as a flag built on ``int`` does.

Rows keep who they are. A table the engine stores carries a hidden column,
``__row``, holding each row's global id: the index it was made on and its
row there (``contacts:0``, ``structured.invoices:0.1``). Polars keeps the
column through a filter, a sort and a join, so a node that only narrows or
reorders a table hands its rows back with their ids, and a later merge can
match them. A row that comes back with no id, or with one an earlier row
already holds, is a new row and gets a fresh id; the others keep theirs
(``with_row_ids``).

Polars is imported the first time a table is built, never at
``import conductor``, so a host that runs no tables never pays for it.
"""

from __future__ import annotations

import datetime
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from itertools import count
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, get_type_hints, is_typeddict

from pydantic import TypeAdapter
from pydantic_core import PydanticSerializationError, core_schema, to_json

from conductor._internal.dtype import DType, dtype_of
from conductor._internal.errors import NodeDeclarationError
from conductor._internal.series import Index, Row

if TYPE_CHECKING:
    import polars as pl
    from pydantic import GetCoreSchemaHandler

#: The hidden column of global row ids every stored table carries.
ROW_ID = "__row"
#: While a merge runs, the row above each row of a source whose index repeats an id.
_UNDER_ROW = "__under_row"

#: The Polars column type of a value built on each builtin, by the dtype's name.
_COLUMN_OF_BASE: dict[type, str] = {
    str: "String", float: "Float64", bool: "Boolean", int: "Int64", datetime.datetime: "Datetime", datetime.date: "Date",
}

Column = tuple[str, type[DType]]


def column_type(dtype: type[DType]) -> pl.DataType | None:
    """The Polars type a column of ``dtype`` values is stored as, or ``None`` when it has none.

    Read off the builtin ``dtype`` is built on, unless the type says it
    itself (``DType.column``) because that base would be wrong.
    """
    import polars as pl

    base = dtype._builtin_base()
    name = dtype.column or (None if base is None else next((_COLUMN_OF_BASE[kind] for kind in base.__mro__ if kind in _COLUMN_OF_BASE), None))
    return None if name is None else getattr(pl, name)()


@cache
def holds_tables(dtype: Any) -> bool:
    """Is a field of this type a table, one or one per row?"""
    dtype = getattr(dtype, "element", None) or dtype
    return isinstance(dtype, type) and issubclass(dtype, Table)


@cache
def columns_of(dtype: Any) -> tuple[Column, ...]:
    """The columns a table type declares, each offered as a field of its own
    (``rows.email``); ``()`` for bare ``Table`` and for any other type."""
    return dtype.columns if isinstance(dtype, type) and issubclass(dtype, Table) else ()


def row_id(index: Index, row: Row) -> str:
    """The global id of the row at ``row`` on ``index``: ``contacts:0``, ``structured.invoices:0.1``.

    The one place a row id is spelled; nothing parses one."""
    return f"{index.id}:{'.'.join(str(n) for n in row)}"


def with_row_ids(frame: pl.DataFrame, index: Index, parent: Row) -> pl.DataFrame:
    """``frame`` with an id on every row, each row's own if it has one.

    A row with no id, or with an id an earlier row holds, is a new row: a
    person added it, or a node copied one. It gets the first id spelled
    from ``parent + (n,)`` on ``index`` that no row holds, so a row added
    to a table leaves every other row's id where it was."""
    if ROW_ID not in frame.columns:
        return _with_ids([row_id(index, (*parent, n)) for n in range(frame.height)], frame)
    ids = frame[ROW_ID]
    if ids.null_count() == 0 and ids.n_unique() == frame.height:
        return frame
    written = ids.to_list()
    taken = {value for value in written if value is not None}
    seen: set[str] = set()
    spare = (spelled for spelled in (row_id(index, (*parent, n)) for n in count()) if spelled not in taken)
    fresh = [next(spare) if value is None or value in seen or seen.add(value) else value for value in written]  # type: ignore[func-returns-value]
    return _with_ids(fresh, frame.drop(ROW_ID))


def _with_ids(ids: list[str], frame: pl.DataFrame) -> pl.DataFrame:
    """A new frame with ``ids`` as its first column: the frame given may be a node's own, or shared, and stays as it was."""
    import polars as pl

    return pl.DataFrame([pl.Series(ROW_ID, ids, dtype=pl.String), *frame.get_columns()])


@dataclass(frozen=True, slots=True)
class MergeSource:
    """One source of a merge, as a unit reads it: a frame with ``__row``, the
    index its rows are on, and each frame row's path on that index."""

    #: The node it comes from, which a clashing column takes on (``name_crm``).
    node_id: str
    frame: pl.DataFrame
    index: Index
    rows: Sequence[Row]


def column_frame(name: str, dtype: type[DType], ids: list[str], values: list[Any]) -> pl.DataFrame:
    """One column of values as a frame beside the ids of their rows."""
    import polars as pl

    return pl.DataFrame([
        pl.Series(ROW_ID, ids, dtype=pl.String),
        pl.Series(name, values, dtype=column_type(dtype), strict=False),
    ])


def frame_with_no_rows(dtype: Any) -> pl.DataFrame:
    """A frame of no rows with the columns a table type declares: a table a branch skipped."""
    import polars as pl

    declared = columns_of(getattr(dtype, "element", None) or dtype)
    return pl.DataFrame(schema={ROW_ID: pl.String, **{name: column_type(holds) for name, holds in declared}})


def no_row_kept_its_id(frame: pl.DataFrame) -> bool:
    """Has a frame with rows lost every row's id? No rows lost nothing."""
    return frame.height > 0 and (ROW_ID not in frame.columns or frame[ROW_ID].null_count() == frame.height)


def merge(sources: Sequence[MergeSource]) -> pl.DataFrame:
    """One table from several, matched on their rows' ids: the one place rows are matched.

    The rows are those of the deepest sources. A source with another source's
    rows under it brings no rows: its values are copied down onto the rows
    under each of its own, by a parent's id, and a parent row nothing sits
    under is not in the result. Every other source brings all its rows and is
    joined in full, so a row only one of them has is kept with blanks beside
    it: sources on one index first, matching id and parent row, then the
    indexes in wiring order, on the id alone. A column a source shares with what is already there
    keeps its name with the source's added (``name_crm``).
    """
    import polars as pl

    above = {s.index for s in sources for other in sources if other.index != s.index and s.index in other.index.lineage}
    ids_on: dict[Index, dict[Row, str]] = {}
    for source in sources:
        if source.index in above:
            ids_on.setdefault(source.index, {}).update(zip(source.rows, source.frame[ROW_ID].to_list(), strict=True))

    taken: set[str] = set()
    named: list[pl.DataFrame] = []
    for source in sources:  # names and column order follow the wiring
        renames = {}
        for name in source.frame.columns:
            if name == ROW_ID:
                continue
            if name in taken:
                renames[name] = _first_free_name(f"{name}_{source.node_id}", taken)
            taken.add(renames.get(name, name))
        named.append(source.frame.rename(renames))
    columns = [name for frame in named for name in frame.columns if name != ROW_ID]

    repeating_ids = {source.index for source in sources if source.frame[ROW_ID].n_unique() < source.frame.height}
    with_own_rows: dict[Index, list[pl.DataFrame]] = {}
    copied_down: list[tuple[Index, pl.DataFrame]] = []
    for source, frame in zip(sources, named, strict=True):
        if source.index in above:
            copied_down.append((source.index, frame))
            continue
        for parent in above.intersection(source.index.lineage):
            parent_ids = [ids_on[parent].get(row[: parent.depth]) for row in source.rows]
            frame = frame.with_columns(pl.Series(_parent_key(parent), parent_ids, dtype=pl.String))
        if source.index in repeating_ids:
            frame = frame.with_columns(pl.Series(_UNDER_ROW, [".".join(map(str, row[:-1])) for row in source.rows], dtype=pl.String))
        with_own_rows.setdefault(source.index, []).append(frame)

    def full_join(left: pl.DataFrame, right: pl.DataFrame, on: list[str]) -> pl.DataFrame:
        joined = left.join(right, on=on, how="full", coalesce=True, maintain_order="left_right")
        for key in (_parent_key(parent) for parent in above):  # two sources under one parent both hold its id
            if f"{key}_right" in joined.columns:
                joined = joined.with_columns(pl.coalesce(key, f"{key}_right").alias(key)).drop(f"{key}_right")
        return joined

    # Sources on one index match row by row, and a row there is its id under its parent:
    # one table passed through per row repeats its ids under each. Across indexes the id alone matches.
    per_index = []
    for frames in with_own_rows.values():
        on_one_index = frames[0]
        for frame in frames[1:]:
            on_one_index = full_join(on_one_index, frame, [ROW_ID, _UNDER_ROW] if _UNDER_ROW in frame.columns else [ROW_ID])
        per_index.append(on_one_index.drop(_UNDER_ROW, strict=False))
    merged = per_index[0]
    for frame in per_index[1:]:
        merged = full_join(merged, frame, [ROW_ID])
    for index, frame in copied_down:
        key = _parent_key(index)
        merged = merged.join(frame.rename({ROW_ID: key}), on=key, how="left", maintain_order="left")
    return merged.select(ROW_ID, *columns)


def _first_free_name(wanted: str, taken: set[str]) -> str:
    """``wanted``, or ``wanted_2``, ``wanted_3`` and on when an earlier column holds it."""
    name, n = wanted, 1
    while name in taken:
        n += 1
        name = f"{wanted}_{n}"
    return name


def _parent_key(index: Index) -> str:
    """The column that holds, on a row of a deeper source, the id of its row on ``index``."""
    return f"__{index.id}"


_PARAMETERISED: dict[tuple[Column, ...], type[Table]] = {}


#: The ``TypedDict`` naming a table's columns, for a type checker.
RowsOf = TypeVar("RowsOf")


class Table(DType, Generic[RowsOf]):
    """Many rows of named, typed columns, as one value: a Polars ``DataFrame`` and the ids of its rows.

    Build one from a ``DataFrame``, a list of records, another table or its
    JSON form; a table type with columns reads each declared column as its
    type's column, refusing a value that won't read and naming the column,
    and drops columns it doesn't declare. ``__row`` always passes through.
    """

    id = "table"
    display_title = "Table"
    #: The columns this type declares, each ``(name, scalar DType)``; ``()`` on bare ``Table``.
    columns: ClassVar[tuple[Column, ...]] = ()

    __slots__ = ("frame",)

    def __init__(self, value: Any) -> None:
        self.frame: pl.DataFrame = self._read(value)

    @classmethod
    def _holding(cls, frame: pl.DataFrame) -> Table:
        """A table of this type holding ``frame`` as it is, already read: the
        engine's way to store a table it has given row ids."""
        made = cls.__new__(cls)
        made.frame = frame
        return made

    def __class_getitem__(cls, item: Any) -> type[Table[Any]]:
        """``Table[Contact]``: the table type whose columns are the keys of a ``TypedDict``, each a scalar ``DType``."""
        if isinstance(item, TypeVar):
            return cls
        if not is_typeddict(item):
            raise NodeDeclarationError(
                f"Table[...] takes a TypedDict naming the columns, got {item!r}; for columns known only to a hook, Table.with_columns({{name: type}})",
                code="table_columns_not_a_typeddict",
            )
        return cls.with_columns(get_type_hints(item))

    @classmethod
    def with_columns(cls, columns: Mapping[str, Any]) -> type[Table[Any]]:
        """The table type with these columns, built once per set; ``Table[Contact]`` with the same columns is this type."""
        declared: list[Column] = []
        for name, written in columns.items():
            dtype = dtype_of(written)
            if not isinstance(name, str) or dtype is None or dtype.element is not None:
                raise NodeDeclarationError(f"a column is a name and a scalar type, got {name!r}: {written!r}", code="table_column_invalid")
            if name == ROW_ID:
                raise NodeDeclarationError(f"'{ROW_ID}' is the column of row ids every table carries; no column may be named it", code="table_column_reserved")
            if column_type(dtype) is None:
                raise NodeDeclarationError(
                    f"{dtype.__name__} can't be a table column: it is built on no builtin a column holds "
                    "and states no column type of its own (DType.column)",
                code="table_column_no_storage",
                )
            declared.append((name, dtype))
        key = tuple(declared)
        made = _PARAMETERISED.get(key)
        if made is None:
            label = ", ".join(f"{name}: {dtype.__name__}" for name, dtype in key)
            made = type(f"Table[{label}]", (cls,), {"columns": key, "__slots__": ()}, _parameterises=Table)
            _PARAMETERISED[key] = made
        return made

    # -- reading a value --------------------------------------------------------

    @classmethod
    def _read(cls, value: Any) -> pl.DataFrame:
        import polars as pl

        if isinstance(value, Table):
            frame = value.frame
        elif isinstance(value, pl.DataFrame):
            frame = value
        elif isinstance(value, Mapping) and "columns" in value and "data" in value:
            frame = _from_wire(value)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            try:
                frame = pl.DataFrame([dict(record) for record in value])
            except (TypeError, ValueError, pl.exceptions.PolarsError) as unreadable:
                raise ValueError(f"a list of records could not be read as a table: {unreadable}") from unreadable
        else:
            raise ValueError(f"a table is a DataFrame, a list of records or a table's JSON form, not a {type(value).__name__}")
        if ROW_ID in frame.columns and frame.schema[ROW_ID] != pl.String:
            frame = frame.with_columns(pl.col(ROW_ID).cast(pl.String))
        if not cls.columns:
            return frame
        if not frame.columns:
            return frame_with_no_rows(cls)
        read = []
        for name, dtype in cls.columns:
            if name not in frame.columns:
                raise ValueError(f"the table has no column '{name}'")
            read.append(_cast(frame[name], column_type(dtype), dtype.display_title))  # type: ignore[arg-type]
        if ROW_ID in frame.columns:
            read.insert(0, frame[ROW_ID])
        return pl.DataFrame(read)

    # -- the edge -------------------------------------------------------------

    @classmethod
    def accepts(cls, source: Any) -> bool:
        """A ``Table[...]`` input takes a table that has at least its columns,
        each of a type it accepts; bare ``Table`` takes any table. A table
        whose columns are known only when it arrives (bare) may land on
        either: its value is read through the input's type then."""
        if not (isinstance(source, type) and issubclass(source, Table)):
            return False
        if not cls.columns or not source.columns:
            return True
        theirs = dict(source.columns)
        return all(name in theirs and dtype.accepts(theirs[name]) for name, dtype in cls.columns)

    @classmethod
    def describe(cls) -> dict[str, Any]:
        """``{"id": "table"}``, plus the columns a table type declares, each with its type's own record."""
        described = super().describe()
        if cls.columns:
            described["columns"] = [{"name": name, "type": dtype.describe()} for name, dtype in cls.columns]
        return described

    @classmethod
    def as_text(cls, value: Any) -> str:
        return f"{value.frame.height} rows"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Table) and type(other).columns == type(self).columns and other.frame.equals(self.frame)

    def __hash__(self) -> int:
        return hash((type(self).columns, self.frame.height))

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.frame.height} rows)"

    # -- pydantic -------------------------------------------------------------

    @classmethod
    def __get_pydantic_json_schema__(cls, schema: core_schema.CoreSchema, handler: Any) -> dict[str, Any]:
        """A list of records, each column under its name: the form a person's answer is sent in.
        A record may carry its row's ``__row``, so a client editing a table keeps its rows; one without is a new row."""
        record: dict[str, Any] = {"type": "object", "properties": {ROW_ID: {"type": "string"}}}
        if cls.columns:
            record["properties"].update({name: TypeAdapter(dtype).json_schema() for name, dtype in cls.columns})
            record["required"] = [name for name, _ in cls.columns]
        return {"type": "array", "items": record}

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler) -> core_schema.CoreSchema:
        """In: a table, a frame, records or the JSON form, read as this type. Out, as JSON:
        ``{"columns": [{"name", "dtype"}], "data": {name: [...]}, "rows": [...]}``, where
        ``rows`` holds the ``__row`` ids (``null`` on a table that has none), ``columns``
        never lists ``__row``, ``dtype`` is the column's Polars type (a name, or an object
        with its parameters) and a blank cell is ``null``. A column that can't be written
        without changing it raises here."""

        def read(value: Any) -> Table:
            return value if type(value) is cls else cls(value)

        return core_schema.no_info_plain_validator_function(
            read, serialization=core_schema.plain_serializer_function_ser_schema(_to_wire, when_used="json"),
        )


def _to_wire(table: Table) -> dict[str, Any]:
    frame = table.frame
    names = [name for name in frame.columns if name != ROW_ID]
    return {
        "columns": [{"name": name, "dtype": _dtype_to_wire(frame[name].dtype)} for name in names],
        "data": {name: _column_to_wire(frame[name]) for name in names},
        "rows": frame[ROW_ID].to_list() if ROW_ID in frame.columns else None,
    }


def _from_wire(value: Mapping[str, Any]) -> pl.DataFrame:
    import polars as pl

    try:
        columns = [
            _column_from_wire(column["name"], value["data"][column["name"]], _dtype_from_wire(column["dtype"]))
            for column in value["columns"]
        ]
        if value.get("rows") is not None:
            columns.insert(0, pl.Series(ROW_ID, value["rows"], dtype=pl.String))
        return pl.DataFrame(columns)
    except (KeyError, TypeError, AttributeError, pl.exceptions.PolarsError) as malformed:
        raise ValueError(f"a table's JSON form does not line up: {malformed}") from malformed


def _column_to_wire(column: pl.Series) -> list[Any]:
    """A column's values as JSON: a type JSON holds exactly as it is; any other as JSON
    spells it, and only if reading that back gives this very column, so the form never
    changes a type or a value. A ``Duration`` travels as its count of the column's unit,
    and a float that isn't finite as ``"NaN"``, ``"Infinity"`` or ``"-Infinity"``."""
    import polars as pl

    dtype = column.dtype
    if dtype in (pl.String, pl.Boolean) or dtype.is_integer() or (dtype.is_float() and column.is_finite().all()):
        return column.to_list()
    try:
        written = json.loads(to_json((column.to_physical() if dtype == pl.Duration else column).to_list(), inf_nan_mode="strings"))
        kept = _column_from_wire(column.name, written, dtype).equals(column, check_dtypes=True)
    except (ValueError, TypeError, PydanticSerializationError, pl.exceptions.PolarsError):
        kept = False
    if not kept:
        raise ValueError(f"column '{column.name}' of type {dtype} can't be written as JSON without changing it")
    return written


def _column_from_wire(name: str, values: list[Any], dtype: pl.DataType) -> pl.Series:
    """One column read back as the type its JSON form names."""
    import polars as pl

    if dtype.is_nested():
        return pl.Series(name, values, dtype=dtype, strict=False)
    return _cast(pl.Series(name, values, strict=False), dtype, str(dtype))


#: A column type's parameters, by the name of the type; one with none isn't here.
_DTYPE_PARAMETERS: dict[str, tuple[str, ...]] = {
    "Decimal": ("precision", "scale"),
    "Datetime": ("time_unit", "time_zone"),
    "Duration": ("time_unit",),
    "Enum": ("categories",),
    "List": ("inner",),
    "Array": ("inner", "shape"),
    "Struct": ("fields",),
}


def _dtype_to_wire(dtype: pl.DataType) -> str | dict[str, Any]:
    """A column type as JSON: its name, or ``{"type": name, ...its parameters}``."""
    name = type(dtype).__name__
    if name not in _DTYPE_PARAMETERS:
        return name
    described: dict[str, Any] = {"type": name}
    for parameter in _DTYPE_PARAMETERS[name]:
        value = getattr(dtype, parameter)
        if parameter == "inner":
            value = _dtype_to_wire(value)
        elif parameter == "fields":
            value = {field.name: _dtype_to_wire(field.dtype) for field in value}
        elif parameter == "categories":
            value = value.to_list()
        described[parameter] = value
    return described


def _dtype_from_wire(described: str | Mapping[str, Any]) -> pl.DataType:
    """The column type ``_dtype_to_wire`` wrote. A name that is no Polars column type is a ``TypeError``."""
    import polars as pl

    name, parameters = (described, {}) if isinstance(described, str) else (described["type"], {k: v for k, v in described.items() if k != "type"})
    made = getattr(pl, name, None)
    if not (isinstance(made, type) and issubclass(made, pl.DataType)):
        raise TypeError(f"{name!r} is not a column type")
    if "inner" in parameters:
        parameters["inner"] = _dtype_from_wire(parameters["inner"])
    if "fields" in parameters:
        parameters["fields"] = {field: _dtype_from_wire(inner) for field, inner in parameters["fields"].items()}
    return made(**parameters)


def _cast(column: pl.Series, target: pl.DataType, said: str) -> pl.Series:
    """``column`` as ``target``, a value that won't read refused, naming the column and the type it isn't.

    A column already of ``target``'s kind stays as it is, its parameters
    with it: a ``DType`` naming ``"Decimal"`` says nothing of the scale.
    Text is parsed into a date, a time or a moment, since they travel as
    text in JSON and in most files; everything else is Polars' own cast."""
    import polars as pl

    if column.dtype == type(target):
        return column
    try:
        if column.dtype == pl.String and isinstance(target, pl.Date):
            cast = column.str.to_date(strict=False)
        elif column.dtype == pl.String and isinstance(target, pl.Time):
            cast = column.str.to_time(strict=False)
        elif column.dtype == pl.String and isinstance(target, pl.Datetime):
            cast = column.str.to_datetime(time_unit=target.time_unit, time_zone=target.time_zone, strict=False)
        else:
            cast = column.cast(target, strict=False)
    except (pl.exceptions.InvalidOperationError, pl.exceptions.ComputeError):
        cast = None
    if cast is None or cast.null_count() > column.null_count():
        raise ValueError(f"column '{column.name}' has a value that is not a {said}")
    return cast
