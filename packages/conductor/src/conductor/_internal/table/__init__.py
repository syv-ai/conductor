"""``Table[R]``: a whole table as one value, a ``polars.DataFrame`` to a node."""

from .type import Table, TableType
from .write import written_columns

__all__ = ["Table", "TableType", "written_columns"]
