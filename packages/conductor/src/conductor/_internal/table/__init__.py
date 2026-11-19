"""Tables: a whole table passed along an edge as one value.

A node that works on many rows at once declares a ``Table`` input or output. Inside the node it is a plain
``polars.DataFrame``. ``Table[R]`` names its columns with a ``TypedDict`` ``R``, and a bare ``Table`` takes any
columns::

    class Invoice(TypedDict):
        customer: str
        amount: float

    @node("total", title="Total", description="Adds up the invoices.")
    def total(invoices: Table[Invoice]) -> float:
        return invoices["amount"].sum()

A table can arrive as a frame, as a list of rows (one dict per row) or as the JSON conductor writes for one.
It always reaches the node as a frame with exactly the columns ``R`` names. The full rules are in the reference:
``python -m conductor.about types``.
"""

from .read import is_bare_table_row
from .type import Table, TableType, hash_fallback, json_fallback
from .write import written_columns

__all__ = [
    "Table",
    "TableType",
    "hash_fallback",
    "is_bare_table_row",
    "json_fallback",
    "written_columns",
]
