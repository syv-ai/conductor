"""Nodes that start and end a loop: ``series-split``, ``series-collect`` and ``series-stack``.

A ``Series[T]`` is a list that a graph loops over: each node downstream of it is called once per item, until a node
whose input takes a ``Series`` or a ``Table`` collects the results back into one list or table. In
``series-split -> text-uppercase -> series-collect`` on ``["a", "b"]``, ``text-uppercase`` is called twice and
``series-collect`` once, with ``["A", "B"]``.

``series-split`` starts a loop over a table's records or a list's items. Its type variable ``T`` is the type of one of
them: a ``Table[Contact]`` gives ``Contact`` records, a ``list[int]`` gives ``int`` items, and any other type wired to
it is a ``type_mismatch`` when the graph is compiled.

``series-collect`` ends a loop: it is called once and returns every item's value as one list. It is needed where a
node inside the loop must see every item. Such a node is called once per item, so a ``Series`` input wired straight
from the loop gets a list of that one item's value: ``['A']`` on the call for ``"a"``. Wired from ``series-collect``
instead, the input gets ``['A', 'B']`` on every call.

``series-stack`` ends a loop whose nodes return a table per item: it puts those tables into one table, row after row.
"""

import typing

import polars as pl
from conductor import NodeRegistry, Series, Table

registry = NodeRegistry(category="series")


@registry.node("series-split", title="Split", description="Takes a table or a list apart, so what follows runs once for each record or item.")
def split[T](value: Table[T] | list[T]) -> Series[T]:
    """Returns a table's records, or a list's items, as a ``Series``: ``["a", "b"]`` gives the items ``"a"`` and ``"b"``."""
    return typing.cast("list[T]", value)  # the engine reads a table returned for a Series as the Series of its records


@registry.node("series-collect", title="Collect", description="Collects the result made for each item into one list, so a later step can use all of them at once.")
def collect[T](values: Series[T]) -> Series[T]:
    """Returns the values of every item, in the order of the items, as one list (a ``Series``, which is a list to the
    function): ``["A", "B"]``."""
    return values


@registry.node("series-stack", title="Stack", description="Puts the tables made for each item into one table, one under the other.")
def stack[R](tables: Series[Table[R]]) -> Table[R]:
    """Returns one table of the rows of every table, in the order of the items. No tables give a table with no columns."""
    return pl.concat(tables, how="vertical_relaxed") if tables else pl.DataFrame()
