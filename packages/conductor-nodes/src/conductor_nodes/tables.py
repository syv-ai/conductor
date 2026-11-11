"""Nodes that take a table or a list apart and put records back into a table: ``split`` and ``table-collect``.

``split`` makes a ``Series`` of a table's records or a list's items, so what
follows runs once for each. Its ``compute_outputs`` hook reads the type
arriving on ``value``: a table gives its records, typed as its row type (a
``dict`` of any fields for a table of any columns), and a list gives its
items. The input is typed ``Any``,
as a table can't stand in a union with a list, so the hook is what refuses a
type that is neither.

``table-collect`` gathers records into one table and adds columns beside
them. ``columns`` names the columns made one value per record, each a
``Series`` input that keeps a blank where a record got no value;
``constant_columns`` names the columns holding one value on every record,
each an input of one value. Its ``compute_outputs`` hook types the table by
the records and the values arriving.
"""

import typing
from typing import Annotated, Any, TypedDict

import polars as pl
from conductor import NodeError, NodeRegistry, Param, Series, Table, row_type
from conductor.widgets import ListInput

registry = NodeRegistry(category="table")

Names = Annotated[list[str], Param(widget=ListInput())]

#: ``Series`` and ``Table`` of a type known only when a graph loads, which a type checker can't follow.
_Series: Any = Series
_Table: Any = Table


def _items(value: type) -> dict[str, Any]:
    """A ``Series`` of what arrives on ``value``, one item at a time: a table's records or a list's items; ``Any``
    items until something is wired."""
    row = row_type(value)
    if row is not None:
        return {"result": _Series[dict if row is Any else row]}
    if value is Any or value is list:
        return {"result": _Series[Any]}
    if typing.get_origin(value) is list:
        return {"result": _Series[typing.get_args(value)[0]]}
    raise NodeError("'value' takes a table or a list, and what is wired to it is neither; wire a table or a list, or use Split text for a text", code="type_mismatch")


@registry.node("split", title="Split", description="Takes a table or a list apart, so what follows runs once for each record or item.", compute_outputs=_items)
def split(value: Any) -> dict[str, Any]:
    """Gives a table's records, or a list's items, as a Series."""
    return {"result": value.to_dicts() if isinstance(value, pl.DataFrame) else value}


def _column_inputs(columns: list[str], constant_columns: list[str]) -> dict[str, Any]:
    """An input for each new column: a ``Series`` that keeps blanks for one of ``columns``, one value for one of
    ``constant_columns``."""
    names = [*columns, *constant_columns]
    twice = [name for name in names if names.count(name) > 1]
    if twice:
        raise NodeError(f"The column '{twice[0]}' is named twice; name each new column once", code="column_named_twice", ctx={"column": twice[0]})
    return {**{name: Series[Any] for name in columns}, **{name: Any for name in constant_columns}}


def _table(records: type, columns: list[str], **added: type) -> dict[str, Any]:
    """The table collected: the records' columns, then each new one, typed by what arrives on its input. Records of
    unknown columns make a table of any columns."""
    if not typing.is_typeddict(records):
        return {"result": Table}
    known = typing.get_type_hints(records)
    taken = [name for name in added if name in known]
    if taken:
        raise NodeError(f"The new column '{taken[0]}' is already a column of the records; give it another name", code="column_taken", ctx={"column": taken[0]})
    new = {name: arriving | None if name in columns else arriving for name, arriving in added.items()}
    return {"result": _Table[TypedDict("Collected", {**known, **new})]}  # type: ignore[operator]  # columns known only at load


@registry.node(
    "table-collect", title="Collect into a table", description="Puts records back into one table, with a new column for each result made beside them.",
    compute_added_inputs=_column_inputs, compute_outputs=_table,
)
def collect(records: Series[dict], columns: Names = [], constant_columns: Names = [], **values: Any) -> dict[str, Any]:
    """Gives the records with each new column's value added, which the output reads as a table, so a table of no
    records still has every column."""
    collected = [
        {**record, **{name: values[name][place] for name in columns}, **{name: values[name] for name in constant_columns}}
        for place, record in enumerate(records)
    ]
    return {"result": collected}
