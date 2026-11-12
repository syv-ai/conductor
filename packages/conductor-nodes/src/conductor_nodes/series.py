"""Nodes for working through a list one item at a time: ``series-split``, ``series-collect`` and ``series-stack``.

``series-split`` makes a ``Series`` of a table's records or a list's items, so
the nodes after it are called once for each. Its type variable is the record or
the item: a table gives its records typed as its row type, a list gives its
items, and anything else is a type mismatch on the edge.

``series-collect`` is the way back: it is called once and returns the value of
every item as one list. A node that is itself called once per item gets only
that item's value in a ``Series`` input; to give each call every item's value,
collect them here first and wire the list in.

``series-stack`` puts the tables made once per item into one table, row after
row.
"""

import typing

import polars as pl
from conductor import NodeRegistry, Series, Table

registry = NodeRegistry(category="series")


@registry.node("series-split", title="Split", description="Takes a table or a list apart, so what follows runs once for each record or item.")
def split[T](value: Table[T] | list[T]) -> Series[T]:
    """Returns a table's records, or a list's items, as a Series."""
    return typing.cast("list[T]", value)  # a table returned for a Series of records is the Series of its records


@registry.node("series-collect", title="Collect", description="Collects the result made for each item into one list, so a later step can use all of them at once.")
def collect[T](values: Series[T]) -> Series[T]:
    """Returns every item's value as one list, in the items' order."""
    return values


@registry.node("series-stack", title="Stack", description="Puts the tables made for each item into one table, one under the other.")
def stack[R](tables: Series[Table[R]]) -> Table[R]:
    """Returns one table of every table's rows, in the items' order."""
    return pl.concat(tables, how="vertical_relaxed") if tables else pl.DataFrame()
