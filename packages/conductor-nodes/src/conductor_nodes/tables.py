"""``table-collect``: put the records of a loop back into one table, with new columns beside them.

A graph takes a table of contacts apart with ``series-split``, greets each contact, and puts the greetings back beside
the contacts::

    read-contacts -> series-split -> greet -> table-collect
    {"name": ["Ida", "Bo"]}  ->  {"name": ["Ida", "Bo"], "greeting": ["Hello Ida", "Hello Bo"]}

``records`` is a ``Table`` input. Wired to ``series-split.result``, a ``Series`` of records, it collects them: the node
is called once, with every record in one table. Each name in ``columns`` and ``constant_columns`` adds an input of that
name and a column:

- A name in ``columns`` is a ``Series`` input with one value per record, from a node called once for each record, such
  as ``greet``. A record for which that node was skipped, or not called because an argument was ``None``, gets ``None``. A list with another number of values than there are
  records fails the node with ``columns_not_aligned``.
- A name in ``constant_columns`` is an input of one value, written on every record: the file's name on each line.

Taking a table apart (``series-split``) and stacking tables (``series-stack``) are in ``series.py``.
"""

import typing
from typing import Annotated, Any, TypedDict

import polars as pl
from conductor import NodeError, NodeRegistry, Param, Series, Table
from conductor.widgets import ListInput

registry = NodeRegistry(category="table")

Names = Annotated[list[str], Param(widget=ListInput())]

#: ``Table`` typed as ``Any``, so that ``_table`` can write ``Table[...]`` of a row type it builds when the graph is
#: compiled, which a type checker can't follow.
_Table: Any = Table


def _column_inputs(columns: list[str], constant_columns: list[str]) -> dict[str, Any]:
    """Returns one input per new column: ``Series[Any]`` for a name in ``columns``, ``Any`` for one in
    ``constant_columns``.

    Raises:
        NodeError: A name is written twice, in one list or across the two (``column_named_twice``).
    """
    names = [*columns, *constant_columns]
    twice = [name for name in names if names.count(name) > 1]
    if twice:
        raise NodeError(
            f"The column '{twice[0]}' is named twice. Name each new column once, in 'columns' or in 'constant_columns'.",
            code="column_named_twice", ctx={"column": twice[0]},
        )
    return {**{name: Series[Any] for name in columns}, **{name: Any for name in constant_columns}}


def _table(records: type, columns: list[str], **added: type) -> dict[str, Any]:
    """Returns the type of the table ``collect`` returns: the records' columns, then each new column.

    ``records`` is the type of what is wired to ``records``: a record type ``R`` (from a ``Series[R]``) or a whole
    ``Table[R]``. ``added`` is the type of what is wired to each new column's input. With ``R`` a ``TypedDict``
    ``Contact`` of ``name: str``, and ``greeting`` in ``columns`` wired from a ``str``, the row type has ``name: str``
    and ``greeting: str | None``. A record type that is not a ``TypedDict`` gives a ``Table`` of any columns.

    Raises:
        NodeError: A new column has the name of a column the records already have (``column_taken``).
    """
    if typing.get_origin(records) is Table:
        records = typing.get_args(records)[0]
    if not typing.is_typeddict(records):
        return {"result": Table}
    known = typing.get_type_hints(records)
    taken = [name for name in added if name in known]
    if taken:
        raise NodeError(
            f"The records already have a column '{taken[0]}'. Give the new column another name.", code="column_taken", ctx={"column": taken[0]},
        )
    new = {name: arriving | None if name in columns else arriving for name, arriving in added.items()}
    return {"result": _Table[TypedDict("Collected", {**known, **new})]}  # type: ignore[operator]  # a row type built at compile


@registry.node(
    "table-collect", title="Collect into a table", description="Puts the records back into one table, with a new column for each result worked out per record.",
    compute_added_inputs=_column_inputs, compute_outputs=_table,
)
def collect(records: Table, columns: Names = [], constant_columns: Names = [], **values: Any) -> dict[str, Any]:
    """Returns the records' table with the new columns after its own: those of ``columns``, then of ``constant_columns``.

    ``values`` has one argument per new column, under its name: a list of one value per record for a name in
    ``columns``, one value for a name in ``constant_columns``.

    Raises:
        NodeError: A list of ``columns`` has another number of values than there are records (``columns_not_aligned``).
    """
    for name in columns:
        count = len(values[name])
        if count != records.height:
            constant_hint = " If it is one value for every record, name it in 'constant_columns' instead." if count == 1 else ""
            raise NodeError(
                f"The column '{name}' has {count} value{'s' if count != 1 else ''} for {records.height} record{'s' if records.height != 1 else ''}. "
                f"Wire it from a node that runs once for each of the same records.{constant_hint}",
                code="columns_not_aligned", ctx={"column": name, "records": records.height, "values": count},
            )
    per_record = [pl.Series(name, values[name], strict=False) for name in columns]
    constant = [pl.Series(name, [values[name]] * records.height, strict=False) for name in constant_columns]
    return {"result": records.with_columns(*per_record, *constant)}
