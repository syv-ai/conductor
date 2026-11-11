"""``Table[R]``: a whole table as one value, a ``polars.DataFrame`` to a node. Nothing outside this package imports polars."""

from conductor._internal.table.held import Records
from conductor._internal.table.type import Table, TableType

__all__ = ["Records", "Table", "TableType"]
