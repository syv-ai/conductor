"""A table as JSON: ``{"columns": [{"name", "type", "values"}]}``, one entry per column, in order."""

from __future__ import annotations

import datetime
import decimal
from collections.abc import Mapping
from typing import Any

import polars as pl
from pydantic_core import CoreSchema, core_schema

#: The classes a table's column may be declared or written as, each with the polars dtype that holds it and how
#: its cells are read back from JSON. A column of any other kind (a list, a struct) is written under the word
#: ``json`` and read back as the JSON it was written as.
_CELLS: Mapping[type, tuple[Any, CoreSchema]] = {
    bool: (pl.Boolean, core_schema.bool_schema()), int: (pl.Int64, core_schema.int_schema()), float: (pl.Float64, core_schema.float_schema()),
    str: (pl.String, core_schema.str_schema()), datetime.date: (pl.Date, core_schema.date_schema()),
    datetime.datetime: (pl.Datetime, core_schema.datetime_schema()), datetime.time: (pl.Time, core_schema.time_schema()),
    datetime.timedelta: (pl.Duration, core_schema.timedelta_schema()), decimal.Decimal: (pl.Decimal, core_schema.decimal_schema()),
}

#: The polars dtype of a written column, by the word it says its type with.
_DTYPES = {kind.__name__: dtype for kind, (dtype, _) in _CELLS.items()}

#: A table as JSON: ``{"columns": [{"name", "type", "values"}]}``, its columns in order in an array, since a store
#: may give an object's keys back in another order.
_WRITTEN_TABLE = core_schema.typed_dict_schema({
    "columns": core_schema.typed_dict_field(core_schema.list_schema(core_schema.tagged_union_schema(
        {
            word: core_schema.typed_dict_schema({
                "name": core_schema.typed_dict_field(core_schema.str_schema()),
                "type": core_schema.typed_dict_field(core_schema.literal_schema([word])),
                "values": core_schema.typed_dict_field(core_schema.list_schema(core_schema.nullable_schema(cells))),
            }, extra_behavior="forbid")
            for word, cells in [*((kind.__name__, cells) for kind, (_, cells) in _CELLS.items()), ("json", core_schema.any_schema())]
        },
        discriminator="type",
    ))),
}, extra_behavior="forbid")


def written_table(frame: pl.DataFrame) -> dict[str, Any]:
    """Gives a table as conductor writes one: each column's name, the word for the type of its cells, and the cells.

    """
    return _written_columns(frame, frame.to_dict(as_series=False))


def _written_columns(frame: pl.DataFrame, cells: Mapping[str, list[Any]]) -> dict[str, Any]:
    """Gives ``frame`` as conductor writes a table, from the ``cells`` of each of its columns by name."""
    columns = []
    for (name, values), dtype in zip(cells.items(), frame.dtypes, strict=True):
        kind = dtype.to_python()
        columns.append({"name": name, "type": kind.__name__ if kind in _CELLS else "json", "values": values})
    return {"columns": columns}
