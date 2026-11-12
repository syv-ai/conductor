"""``Table[R]``: a whole table as one value, a ``polars.DataFrame`` to a node. Nothing outside this package imports polars."""

from .stored import Records
from .type import Table, TableType

__all__ = ["Records", "Table", "TableType"]
