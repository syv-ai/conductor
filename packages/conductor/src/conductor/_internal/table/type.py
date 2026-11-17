"""``Table[R]``, the hint a node author writes, and ``TableType``, how conductor reads that hint.

``Table`` itself does very little. It exists so pydantic can find the table's schema, which ``read.py`` builds.
``TableType`` is the ``Type`` subclass for a table: it gives the table's id, the type of one row, and the small
rules where a table differs from other values.
"""

from __future__ import annotations

import functools
import typing
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, get_args

import polars as pl
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema

from conductor._internal.types import Type

from .read import as_saved, table_schema
from .write import (
    table_digest,
    written_table,
)


class _Table[T]:
    """A whole table passed as one value, with its columns named by a ``TypedDict``.

    Write ``Table[Invoice]`` for a table whose columns are the fields of ``Invoice``, or a bare ``Table`` for a
    table with any columns. The node gets a plain ``polars.DataFrame``, and a type checker sees one too.

    A table can be given as a frame, as a list of records (``[{"customer": "Ann", "amount": 12.5}]``) or as the
    JSON conductor writes for one. Records are how gathered rows and a person's answer arrive. Either way, a
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


if TYPE_CHECKING:  # a type checker sees a Table as the frame a node gets, with its row type kept
    from .table_as_a_type_checker_reads_it import Table
else:
    Table = _Table


#: The words pydantic puts in an error's ``loc`` for the form a table was given in. They name no row or column,
#: so they are left out of the ``loc`` a host is shown.
_TABLE_FORMS = ("frame", "records", "written", "columns")


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
        """The type hint of each column, by name. A bare ``Table`` has none."""
        return Type(self._row).fields

    @functools.cached_property
    def _named(self) -> tuple[str, str, tuple[type, ...]]:
        """The table's id, its title and the classes it names.

        The id lists the columns, like ``Table[customer: str, amount: float]``, or is just ``Table`` for a bare
        table. The title is always ``Table``, and a table names no class.
        """
        columns = typing.get_type_hints(self._row) if typing.is_typeddict(self._row) else {}
        return (
            f"Table[{', '.join(f'{column}: {Type(hint).text}' for column, hint in columns.items())}]"
            if columns
            else "Table",
            "Table",
            (),
        )

    @functools.cached_property
    def record(self) -> Type:
        """The type of one row: the ``TypedDict`` ``R``, or ``dict`` for a bare ``Table``."""
        return Type(dict if self._row is Any else self._row)

    def _walked(self, step: int | str) -> tuple[Type, bool, str | None] | None:
        """Follows one step of a pydantic error's ``loc`` inside a table.

        A word for the form the table was given in, like ``"records"``, is dropped. A row index is kept and stays on
        the table. A column name leads to that column's type and is returned as the column. Any other step is not
        part of the table.
        """
        if step in _TABLE_FORMS:
            return self, False, None
        if isinstance(step, int):
            return self, True, None
        return (Type(self.fields[step]), True, step) if step in self.fields else None

    def _fault_in(self, part: str, kind: str, msg: str, ends_at_column: bool) -> str:
        """Returns the error message ``msg`` with the column ``part`` named in it.

        A missing column gets a message of its own, since pydantic's "Field required" doesn't say it is a column.
        A field missing inside a cell, such as a record's, keeps pydantic's message after the column's name.
        """
        return (
            f"the column '{part}' is missing, and the row type needs it"
            if kind == "missing" and ends_at_column
            else f"column '{part}': {msg}"
        )


def frame_as_json(value: Any) -> Any:
    """Converts a frame to JSON where no type says it is a table.

    That happens for a frame inside an ``Any``, or in a person's answer given from Python. The columns are first
    converted the way a run stores them (``as_saved``). Pydantic's ``to_json`` calls it as its ``fallback`` for a
    value it can't write itself.

    Raises:
        TypeError: ``value`` is not a frame. Pydantic's ``to_json`` expects ``fallback`` to raise it for a value it
            can't write.
        PydanticCustomError: ``column_not_saved`` if a column would come back changed from a saved run.
    """
    if not isinstance(value, pl.DataFrame):
        raise TypeError(type(value).__name__)
    return written_table(as_saved(value))


def frame_to_hash(value: Any) -> Any:
    """Returns ``{"table": digest}`` to hash in place of a frame a call received.

    It is passed to ``to_json`` as ``fallback`` when a call's arguments are hashed. This keeps a large table from
    being written as JSON for every call that gets it. ``table_digest`` works out each frame's digest once.

    Raises:
        TypeError: ``value`` is not a frame, so it can't be written as JSON.
    """
    if not isinstance(value, pl.DataFrame):
        raise TypeError(type(value).__name__)
    return {"table": table_digest(value)}
