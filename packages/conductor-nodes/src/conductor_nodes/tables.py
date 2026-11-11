"""``table-collect``: gather records into one table and add columns beside them.

``columns`` names the columns made one value per record, each a ``Series``
input that keeps a blank where a record got no value, and one that doesn't
hold a value for each record is refused when the node runs
(``columns_not_aligned``); ``constant_columns`` names the columns holding one
value on every record, each an input of one value. Its ``compute_outputs``
hook types the table by the records and the values arriving.

Taking a table apart (``series-split``) and stacking tables (``series-stack``)
are in ``series.py``.
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
    unknown columns make a table of any columns. ``records`` is the type wired to the input: the record type of a
    ``Series``, or a whole ``Table[R]``, whose records are ``R``."""
    if typing.get_origin(records) is Table:
        records = typing.get_args(records)[0]
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
def collect(records: Table, columns: Names = [], constant_columns: Names = [], **values: Any) -> dict[str, Any]:
    """Returns the records' table with each new column added after its own columns.

    Raises:
        NodeError: A column doesn't hold one value for each record (``columns_not_aligned``).
    """
    for name in columns:
        held = len(values[name])
        if held != records.height:
            made_once = "; if it is one value for every record, name it in 'constant_columns', which writes it on each" if held == 1 else ""
            raise NodeError(
                f"The column '{name}' holds {held} value{'s' if held != 1 else ''} for {records.height} record{'s' if records.height != 1 else ''}; "
                "wire a column from a node that runs once per record "
                f"of the same Series as 'records'{made_once}",
                code="columns_not_aligned", ctx={"column": name, "records": records.height, "values": held},
            )
    made = [pl.Series(name, values[name], strict=False) for name in columns]
    constant = [pl.Series(name, [values[name]] * records.height, strict=False) for name in constant_columns]
    return {"result": records.with_columns(*made, *constant)}
