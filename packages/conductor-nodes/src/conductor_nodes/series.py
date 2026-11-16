"""Nodes that start and end a loop: ``series-split``, ``series-collect`` and ``series-stack``.

A ``Series[T]`` is a list that a graph loops over. Wire it into an input that takes one ``T``, and that node is
called once for each item. Wire it into an input that takes the whole ``Series``, and that node is called once,
with every item in one list. That ends the loop.

Here is a small loop on ``["a", "b"]``::

    series-split -> text-uppercase -> series-collect

``text-uppercase`` is called twice, once for ``"a"`` and once for ``"b"``. ``series-collect`` is called once,
with ``["A", "B"]``. How loops work in full: ``python -m conductor.about one-or-many``.

``series-split`` starts a loop over a table's records or a list's items. ``series-collect`` ends one and gives
back a list. ``series-stack`` ends a loop whose nodes return a table for each item, and puts those tables into one.
"""

import typing

import polars as pl
from conductor import NodeRegistry, Series, Table

registry = NodeRegistry(category="series")


@registry.node(
    "series-split",
    title="Split",
    description="Takes a table or a list apart, so what follows runs once for each record or item.",
)
def split[T](value: Table[T] | list[T]) -> Series[T]:
    """Returns a table's records, or a list's items, as a ``Series``.

    ``["a", "b"]`` gives the items ``"a"`` and ``"b"``. ``T`` is the type of one of them, so a ``Table[Contact]``
    gives ``Contact`` records and a ``list[int]`` gives ``int`` items. Any other type wired in is a
    ``type_mismatch`` when the graph is compiled.
    """
    return typing.cast(
        "list[T]", value
    )  # a table returned here is read as the Series of its records, so it needs no converting


@registry.node(
    "series-collect",
    title="Collect",
    description="Puts the result of each item into one list, so the next node gets all of them at once.",
)
def collect[T](values: Series[T]) -> Series[T]:
    """Returns the value of every item as one list, in the order of the items, such as ``["A", "B"]``.

    The result is a ``Series`` too, so you can wire it into an input that takes one item to start a new loop.

    You also need this node for a node inside a loop that must see every item. Say ``check(line, all_lines)`` is
    called once for each line. Wired straight from the loop, ``all_lines`` would get one value per call, which
    compiling refuses. Wire ``all_lines`` from ``series-collect`` instead, and every call gets ``["A", "B"]``.
    """
    return values


@registry.node(
    "series-stack",
    title="Stack",
    description="Puts the table of each item into one table, one under the other.",
)
def stack[R](tables: Series[Table[R]]) -> Table[R]:
    """Puts the tables of every item into one table, one under the other, in the order of the items.

    With no tables at all, it returns an empty table with no columns.
    """
    return pl.concat(tables, how="vertical_relaxed") if tables else pl.DataFrame()
