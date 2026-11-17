"""``table-collect``, a node that puts the records of a loop back into one table, with new columns beside them.

Say a graph takes a table of contacts apart with ``series-split`` and greets each contact. ``table-collect`` puts the
greetings back beside the contacts::

    read-contacts -> series-split -> greet -> table-collect
    {"name": ["Ida", "Bo"]}  ->  {"name": ["Ida", "Bo"], "greeting": ["Hello Ida", "Hello Bo"]}

Wire ``series-split.result`` into ``records``. The node is then called once, with every record in one table. Each
name you list in ``columns`` or ``constant_columns`` adds an input with that name, and a column with it too:

- A name in ``columns`` gets one value per record, from a node that runs once for each record, such as ``greet``.
  A record gets ``None`` if that node was skipped for it, or wasn't called because an argument was ``None``. If the
  number of values doesn't match the number of records, the node fails with ``columns_not_aligned``.
- A name in ``constant_columns`` gets one value, and every record gets the same one. Think of a file's name, written
  on each of its lines.

``series-split`` and ``series-stack``, the nodes that take a table apart and stack tables, are in ``series.py``.
"""

import typing
from typing import Any, TypedDict

import polars as pl
from conductor import NodeError, NodeRegistry, Policy, Series, Table

registry = NodeRegistry(category="table")

#: ``Table``, typed as ``Any``. ``_table`` builds a row type when the graph is compiled and writes ``Table[...]`` of
#: it, which a type checker can't follow.
_Table: Any = Table


def _column_inputs(columns: list[str], constant_columns: list[str]) -> dict[str, Any]:
    """Works out the inputs ``table-collect`` adds: one for each new column.

    A name in ``columns`` gets a ``Series[Any]`` input, one value per record. A name in ``constant_columns`` gets
    an ``Any`` input, one value for all of them.

    Raises:
        NodeError: ``column_named_twice`` if a name is given twice, in one list or across the two.
    """
    names = [*columns, *constant_columns]
    twice = [name for name in names if names.count(name) > 1]
    if twice:
        raise NodeError(
            f"The column '{twice[0]}' is named twice. Name each new column once, in 'columns' or in 'constant_columns'.",
            code="column_named_twice",
            ctx={"column": twice[0]},
        )
    return {**{name: Series[Any] for name in columns}, **{name: Any for name in constant_columns}}


def _refuse_taken(new_columns: list[str], columns_of_records: list[str]) -> None:
    """Checks that no new column has the name of a column the records already have.

    ``_table`` calls it as the graph is compiled, when the records' row type names their columns. ``collect`` calls
    it again when the node runs, since a ``Table`` of any columns only shows its columns then.

    Raises:
        NodeError: ``column_taken`` if a new column's name is taken.
    """
    taken = [name for name in new_columns if name in columns_of_records]
    if taken:
        raise NodeError(
            f"The records already have a column '{taken[0]}'. Give the new column another name.",
            code="column_taken",
            ctx={"column": taken[0]},
        )


def _table(records: type, columns: list[str], **added: type) -> dict[str, Any]:
    """Works out the type of the table ``table-collect`` returns: the records' columns, then the new ones.

    Conductor calls this when the graph is compiled. ``records`` is the type of what is wired to ``records``: a
    record type ``R``, from a ``Series[R]``, or a whole ``Table[R]``. ``added`` has the type of what is wired to each
    new column's input.

    Say ``R`` is a ``TypedDict`` ``Contact`` with ``name: str``, and ``greeting`` is in ``columns``, wired from a
    ``str``. The table's rows then have ``name: str`` and ``greeting: str | None``. A ``greeting`` may be ``None``,
    since a record gets ``None`` when ``greet`` wasn't called for it. If ``R`` isn't a ``TypedDict``, the result is
    a ``Table`` of any columns.

    Raises:
        NodeError: ``column_taken`` if a new column has the name of a column the records already have.
    """
    if typing.get_origin(records) is Table:
        records = typing.get_args(records)[0]
    if not typing.is_typeddict(records):
        return {"result": Table}
    known = typing.get_type_hints(records)
    _refuse_taken(list(added), list(known))
    new = {
        name: arriving | None if name in columns else arriving for name, arriving in added.items()
    }
    return {"result": _Table[TypedDict("Collected", {**known, **new})]}  # type: ignore[operator]  # a row type built as the graph is compiled


@registry.node(
    "table-collect",
    title="Collect into a table",
    description="Puts the records back into one table, with a new column for each result worked out per record.",
    compute_added_inputs=_column_inputs,
    compute_outputs=_table,
    policy=Policy(costs_nothing=True),
)
def collect(
    records: Table, columns: list[str] = [], constant_columns: list[str] = [], **values: Any
) -> dict[str, Any]:
    """Returns the records' table with the new columns added after its own.

    The columns of ``columns`` come first, then those of ``constant_columns``. ``values`` has one argument for each
    new column, under its name. For a name in ``columns`` it is a list with one value per record. For a name in
    ``constant_columns`` it is one value. A column wired from a value returned once, the same for every record, goes
    in ``constant_columns``: under ``columns`` it is ``type_mismatch`` (``one_value``) when the graph is compiled.

    Raises:
        NodeError: ``column_taken`` if a new column has the name of a column the records already have.
            ``columns_not_aligned`` if a column in ``columns`` doesn't have one value for each record.
    """
    _refuse_taken([*columns, *constant_columns], records.columns)
    for name in columns:
        count = len(values[name])
        if count != records.height:
            raise NodeError(
                f"The column '{name}' has {count} value{'s' if count != 1 else ''} for {records.height} record{'s' if records.height != 1 else ''}. "
                "Wire it from a node that runs once for each of the same records.",
                code="columns_not_aligned",
                ctx={"column": name, "records": records.height, "values": count},
            )
    per_record = [pl.Series(name, values[name], strict=False) for name in columns]
    constant = [
        pl.Series(name, [values[name]] * records.height, strict=False) for name in constant_columns
    ]
    return {"result": records.with_columns(*per_record, *constant)}
