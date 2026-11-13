"""``Table[R]``, the hint a node author writes, and ``TableType``, the ``Type`` of such a hint."""

from __future__ import annotations

import functools
import typing
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar, get_args

import polars as pl
from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema

from conductor._internal.types import Type

from .read import table_schema
from .write import (
    table_digest,
    written_table,
)


class _Table[T]:
    """``Table[R]``: a whole table whose columns the ``TypedDict`` ``R`` names; a ``polars.DataFrame`` to the node
    and to a type checker.

    A bare ``Table`` has any columns. A table is read from a frame, from a list of records (how gathered records
    and a person's answer arrive) or from the JSON conductor writes one as. A ``Table[R]`` read holds exactly
    ``R``'s columns, as a ``TypedDict`` read drops other keys: each present, each cell a value of its column's
    type, and blank only where that type takes a blank.
    """

    # The class is named as a host imports it, so the text a reuse digest is made of names no internal file.
    __module__, __qualname__ = "conductor", "Table"

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        """Returns the schema of ``Table[R]``, built in ``read.py``."""
        return table_schema(source, handler)


if TYPE_CHECKING:  # to a type checker a Table is the frame a node gets, which keeps its row type
    from .table_as_a_type_checker_reads_it import Table
else:
    Table = _Table


#: Words pydantic puts in an error's ``loc`` to say which form the table was given in. They are dropped, as they
#: name no row or column.
_TABLE_FORMS = ("frame", "records", "written", "columns")


class TableType(Type, of=_Table, value_class=pl.DataFrame):
    """The type of a ``Table[R]``, with or without ``| None``. A ``polars.DataFrame`` written as a hint is the
    table of any columns it is."""

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
        """The type of one row as a record: the ``TypedDict`` ``R``, or ``dict`` for a table of any columns."""
        return Type(dict if self._row is Any else self._row)

    def written_once(self, stored: Any) -> Any | None:
        """Returns the frame itself: a table that two outputs both have is written once in a save."""
        return stored

    def _walked(self, step: int | str) -> tuple[Type, bool, str | None] | None:
        """Takes one step of a pydantic error's ``loc`` inside a table. A word for the form the table was given in
        (``"records"``) is dropped, a row index leads to the same table, and a column name leads to the column's
        type and is returned as the column."""
        if step in _TABLE_FORMS:
            return self, False, None
        if isinstance(step, int):
            return self, True, None
        return (Type(self.fields[step]), True, step) if step in self.fields else None

    def _fault_in(self, part: str, kind: str, msg: str) -> str:
        """Returns the error message ``msg`` with the column ``part`` named in it."""
        return f"the column '{part}' its row type declares is missing" if kind == "missing" else f"column '{part}': {msg}"

    @staticmethod
    def untyped_as_json(value: Any) -> Any:
        """Returns the JSON form of a frame found where no type is declared, such as inside an ``Any``."""
        return written_table(value)

    @staticmethod
    def untyped_to_hash(value: Any) -> Any:
        """Returns ``{"table": digest}`` for a frame in what a call received, where the digest is computed once
        per frame."""
        return {"table": table_digest(value)}

    @staticmethod
    def untyped_equal(one: Any, other: Any) -> bool:
        """Returns whether two frames have equal cells. ``==`` on frames returns a frame, not a bool."""
        return isinstance(other, pl.DataFrame) and one.equals(other)
