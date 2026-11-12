"""A table as JSON: ``{"columns": [{"name", "type", "values"}]}``, one entry per column, in order."""

from __future__ import annotations

import datetime
import hashlib
import weakref
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

import polars as pl
from pydantic_core import CoreSchema, core_schema, to_json

#: The classes a table's column may be declared or written as, each with the polars dtype that holds it and how
#: its cells are read back from JSON. A column of any other kind (a list, a struct) is written under the word
#: ``json`` and read back as the JSON it was written as.
CELLS: Mapping[type, tuple[Any, CoreSchema]] = {
    bool: (pl.Boolean, core_schema.bool_schema()), int: (pl.Int64, core_schema.int_schema()), float: (pl.Float64, core_schema.float_schema()),
    str: (pl.String, core_schema.str_schema()), datetime.date: (pl.Date, core_schema.date_schema()),
    datetime.datetime: (pl.Datetime, core_schema.datetime_schema()), datetime.time: (pl.Time, core_schema.time_schema()),
    datetime.timedelta: (pl.Duration, core_schema.timedelta_schema()), Decimal: (pl.Decimal, core_schema.decimal_schema()),
}

#: The polars dtype of a written column, by the word it says its type with.
DTYPES = {kind.__name__: dtype for kind, (dtype, _) in CELLS.items()}

#: A table as JSON: ``{"columns": [{"name", "type", "values"}]}``, its columns in order in an array, since a store
#: may give an object's keys back in another order.
WRITTEN_TABLE = core_schema.typed_dict_schema({
    "columns": core_schema.typed_dict_field(core_schema.list_schema(core_schema.tagged_union_schema(
        {
            word: core_schema.typed_dict_schema({
                "name": core_schema.typed_dict_field(core_schema.str_schema()),
                "type": core_schema.typed_dict_field(core_schema.literal_schema([word])),
                "values": core_schema.typed_dict_field(core_schema.list_schema(core_schema.nullable_schema(cells))),
            }, extra_behavior="forbid")
            for word, cells in [*((kind.__name__, cells) for kind, (_, cells) in CELLS.items()), ("json", core_schema.any_schema())]
        },
        discriminator="type",
    ))),
}, extra_behavior="forbid")


def written_table(frame: pl.DataFrame) -> dict[str, Any]:
    """Gives a table as conductor writes one: each column's name, the word for the type of its cells, and the cells.

    """
    return written_columns(frame, frame.to_dict(as_series=False))


def written_columns(frame: pl.DataFrame, cells: Mapping[str, list[Any]]) -> dict[str, Any]:
    """Gives ``frame`` as conductor writes a table, from the ``cells`` of each of its columns by name."""
    columns = []
    for (name, values), dtype in zip(cells.items(), frame.dtypes, strict=True):
        kind = dtype.to_python()
        columns.append({"name": name, "type": kind.__name__ if kind in CELLS else "json", "values": values})
    return {"columns": columns}


#: The digest of each frame ``table_digest`` was asked for, by ``id(frame)``, with a weak reference that removes the
#: entry when the frame is freed. A frame can't be a dict key, as it has no hash.
_DIGESTS: dict[int, tuple[weakref.ref[pl.DataFrame], str]] = {}


def table_digest(frame: pl.DataFrame) -> str:
    """Returns a hash of the table's JSON (column names, cell types and cells), computed the first time it is
    asked for a frame and reused for that same frame object after that.

    Example: a 2,000-row table that 2,000 calls each received is written as JSON once, not 2,000 times, when
    ``run.again`` works out which calls it can reuse. Two frames with equal cells have equal digests. A frame
    changed in place after its digest was computed keeps the old digest, so a node must not change a table it
    was given.
    """
    kept = _DIGESTS.get(id(frame))
    if kept is not None and kept[0]() is frame:
        return kept[1]
    digest = hashlib.blake2b(to_json(written_table(frame)), digest_size=16).hexdigest()
    key = id(frame)
    _DIGESTS[key] = (weakref.ref(frame, lambda _: _DIGESTS.pop(key, None)), digest)
    return digest
