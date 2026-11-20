"""``Table[R]``, the hint a node author writes, and ``TableType``, how conductor reads that hint.

``Table`` itself does very little. It exists so pydantic can find the table's schema, which ``read.py`` builds.
``TableType`` is the ``Type`` subclass for a table: it gives the table's id, the type of one row, and the small
rules where a table differs from other values.
"""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, get_args, get_origin

import polars as pl
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema

from conductor._internal.types import (
    Type,
    is_typed_dict,
    json_by_its_class,
    union_members,
    union_of,
)

from .read import as_saved, check_table_row, column_missing, table_schema
from .write import (
    table_digest,
    written_table,
)


class _Table[T]:
    """A whole table passed as one value, with its columns named by a ``TypedDict``.

    Write ``Table[Invoice]`` for a table whose columns are the fields of ``Invoice``, or a bare ``Table`` for a
    table with any columns. The node gets a plain ``polars.DataFrame``, and a type checker sees one too.

    A table can be given as a frame, as a list of rows (``[{"customer": "Ann", "amount": 12.5}]``) or as the
    JSON conductor writes for one. A list of rows is how gathered rows and a person's answer arrive. Either way, a
    ``Table[Invoice]`` comes out with exactly ``Invoice``'s columns. Every column must be there, each cell must be a
    value of its column's type, and a cell may be blank only if that type allows ``None``. Extra columns are dropped,
    the same way a ``TypedDict`` drops extra keys.
    """

    # Named as a host imports it. The hash that decides whether a call can be reused includes this name, so it
    # must not change when an internal file moves.
    __module__, __qualname__ = "conductor", "Table"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Returns the pydantic schema of ``Table[R]``, which ``read.py`` builds."""
        return table_schema(source, handler)


class _RowsChecked[T]:
    """A ``Table[R]`` whose rows are already ``R``'s values, which a run checks a table of rows from outputs that
    declare ``R`` by (``with_rows_checked``). A list of rows is made into a frame without its cells being checked
    again, and a frame with exactly the polars types ``R``'s own frame has is kept. No host writes it."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Returns the pydantic schema of ``Table[R]`` for rows already ``R``'s values."""
        return table_schema(source, handler, rows_checked=True)


def with_rows_checked(hint: Any) -> Any:
    """Returns ``hint`` with each ``Table[R]`` in it read as rows already ``R``'s values (``_RowsChecked``).

    The rest of the hint is kept, so a host's own check on the table, like
    ``Annotated[Table[R], AfterValidator(look)]``, still runs, and is called as for any table.
    """
    origin = get_origin(hint)
    if origin is Annotated:
        inner, *metadata = get_args(hint)
        return Annotated[with_rows_checked(inner), *metadata]
    if origin is _Table:
        return _RowsChecked[get_args(hint)]  # type: ignore[misc]
    members = union_members(hint)
    return union_of(with_rows_checked(member) for member in members) if len(members) > 1 else hint


if TYPE_CHECKING:  # a type checker sees a Table as the frame a node gets, with its row type kept
    from .table_as_a_type_checker_reads_it import Table
else:
    Table = _Table


#: The words pydantic puts in an error's ``loc`` for the form a table was given in. They name no row or column,
#: so they are left out of the ``loc`` a host is shown.
_TABLE_FORMS = ("frame", "rows", "written", "columns")


class TableType(Type, of=_Table, value_class=pl.DataFrame):
    """The ``Type`` of a ``Table[R]`` hint, with or without ``| None``.

    A plain ``polars.DataFrame`` written as a hint is read as a bare ``Table``: a table with any columns.
    """

    @functools.cached_property
    def _row(self) -> Any:
        """The row type ``R``, or ``Any`` for a bare ``Table``."""
        return (get_args(self.options[0]) or (Any,))[0]

    @functools.cached_property
    def fields(self) -> Mapping[str, Any]:
        """The type hint of each column, by name: the fields of the ``TypedDict`` ``R``. A bare ``Table`` has none,
        and so has a table whose row a graph filled in with any other type, which is read as a bare table."""
        return Type(self._row).fields if is_typed_dict(self._row) else {}

    @functools.cached_property
    def _named(self) -> tuple[str, str, tuple[type, ...]]:
        """The table's id, its title and the classes it names.

        The id lists the columns, like ``Table[customer: str, amount: float]``, or is just ``Table`` for a bare
        table. The title is always ``Table``, and a table names no class.
        """
        columns = self.fields
        return (
            f"Table[{', '.join(f'{column}: {Type(hint).text}' for column, hint in columns.items())}]"
            if columns
            else "Table",
            "Table",
            (),
        )

    @functools.cached_property
    def row(self) -> Type:
        """The type of one row: the ``TypedDict`` ``R``, or ``dict`` for a bare ``Table``."""
        return Type(dict if self._row is Any else self._row)

    def _check_own_arguments(self) -> None:
        """Refuses a row type that names no columns, as ``Type._check_own_arguments`` says (``check_table_row``).

        The row is read as written, so a type variable a graph fills in is let through.
        """
        check_table_row(*get_args(self.hint))

    def _walked(self, step: int | str) -> tuple[Type, bool, str | None] | None:
        """Follows one step of a pydantic error's ``loc`` inside a table.

        A word for the form the table was given in, like ``"rows"``, is dropped. A row index is kept and stays on
        the table. A column name leads to that column's type and is returned as the column. Any other step is not
        part of the table.
        """
        if step in _TABLE_FORMS:
            return self, False, None
        if isinstance(step, int):
            return self, True, None
        return (Type(self.fields[step]), True, step) if step in self.fields else None

    def _fault_in(
        self, part: str, kind: str, msg: str, in_cell: tuple[int | str, ...]
    ) -> tuple[str, str]:
        """Returns the error's type and message with the column ``part`` named in it.

        Rows that lack a column get pydantic's ``missing``, whose "Field required" doesn't say it is a column. It
        becomes ``column_missing``, the error a frame without the column gets. A field missing from a value inside
        a cell stays ``missing``, and its message names the field, which pydantic's doesn't: "column 'price': the
        field 'currency' is required".
        """
        if kind == "missing" and not in_cell:
            missing = column_missing(part)
            return missing.type, missing.message()
        if kind == "missing" and isinstance(field := in_cell[-1], str):
            return kind, f"column '{part}': the field '{field}' is required"
        return kind, f"column '{part}': {msg}"


def json_fallback(value: Any) -> Any:
    """Writes an object pydantic can't write by itself as JSON, where no type says what the value is.

    That happens for a person's answer a host queues for later (``graph.answers_as_json``), and for the inputs a waiting
    call shows. A frame is written as a table, its columns first converted the way a run stores them (``as_saved``).
    Any other object is written by its own class's schema (``json_by_its_class``). Pydantic's ``to_json`` calls it as
    its ``fallback``.

    Raises:
        TypeError: The object can't be written as JSON. Pydantic's ``to_json`` expects ``fallback`` to raise it.
        PydanticCustomError: ``column_not_saved`` if a column would come back changed from a saved run.
    """
    if not isinstance(value, pl.DataFrame):
        return json_by_its_class(value)
    return written_table(as_saved(value))


def hash_fallback(value: Any) -> Any:
    """Writes an object pydantic can't write by itself, for a call's arguments to be hashed.

    A frame is written as ``{"table": digest}``. This keeps a large table from being written as JSON for every call
    that gets it, and ``table_digest`` works out each frame's digest once. Any other object is written by its own
    class's schema (``json_by_its_class``), as everywhere else. Pydantic's ``to_json`` calls it as its ``fallback``.

    Raises:
        TypeError: The object can't be written as JSON, so the call can't be hashed.
    """
    if not isinstance(value, pl.DataFrame):
        return json_by_its_class(value)
    return {"table": table_digest(value)}
