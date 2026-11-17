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

A table can arrive as a frame, as a list of records (one dict per row) or as the JSON conductor writes for one.
It always reaches the node as a frame with exactly the columns ``R`` names. The full rules are in the reference:
``python -m conductor.about types``.
"""

from .type import Table, TableType, frame_as_json, frame_to_hash, is_bare_table_row
from .write import written_columns

__all__ = [
    "Table",
    "TableType",
    "frame_as_json",
    "frame_to_hash",
    "is_bare_table_row",
    "written_columns",
]
