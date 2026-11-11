"""Nodes that take a table or a list apart and put tables back together: ``split``, ``stack`` and ``table-collect``.

``split`` makes a ``Series`` of a table's records or a list's items, so what
follows runs once for each. Its type variable is the record or the item: a
table gives its records typed as its row type, a list gives its items, and
anything else is a type mismatch on the edge.

``stack`` puts the tables made once per item into one table, row after row.

``table-collect`` gathers records into one table and adds columns beside
them. ``columns`` names the columns made one value per record, each a
``Series`` input that keeps a blank where a record got no value, and one
that doesn't hold a value for each record is refused when the node runs
(``columns_not_aligned``);
``constant_columns`` names the columns holding one value on every record,
each an input of one value. Its ``compute_outputs`` hook types the table by
the records and the values arriving.
"""

import typing
from typing import Annotated, Any, TypedDict

import polars as pl
from conductor import NodeError, NodeRegistry, Param, Series, Table
from conductor.widgets import ListInput

registry = NodeRegistry(category="table")

Names = Annotated[list[str], Param(widget=ListInput())]

#: ``Table`` of a row type known only when a graph loads, which a type checker can't follow.
_Table: Any = Table


@registry.node("split", title="Split", description="Takes a table or a list apart, so what follows runs once for each record or item.")
def split[T](value: Table[T] | list[T]) -> Series[T]:
    """Gives a table's records, or a list's items, as a Series."""
    return typing.cast("list[T]", value)  # a table returned for a Series of records is the Series of its records


@registry.node("stack", title="Stack", description="Puts the tables made for each item into one table, one under the other.")
def stack[R](tables: Series[Table[R]]) -> Table[R]:
    """Gives one table of every table's rows, in the items' order."""
    return pl.concat(tables, how="vertical_relaxed") if tables else pl.DataFrame()


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
    """Gives a table of the records, each with its new columns' values added.

    Raises:
        NodeError: A column doesn't hold one value for each record (``columns_not_aligned``).
    """
    for name in columns:
        held = len(values[name])
        if held != len(records):
            made_once = "; if it is one value for every record, name it in 'constant_columns', which writes it on each" if held == 1 else ""
            raise NodeError(
                f"The column '{name}' holds {held} value{'s' if held != 1 else ''} for {len(records)} record{'s' if len(records) != 1 else ''}; "
                "wire a column from a node that runs once per record "
                f"of the same Series as 'records'{made_once}",
                code="columns_not_aligned", ctx={"column": name, "records": len(records), "values": held},
            )
    constants = {name: values[name] for name in constant_columns}
    collected = [
        {**record, **dict(zip(columns, cells, strict=True)), **constants}
        for record, *cells in zip(records, *(values[name] for name in columns), strict=True)
    ]
    return {"result": pl.DataFrame(collected, infer_schema_length=None)}
