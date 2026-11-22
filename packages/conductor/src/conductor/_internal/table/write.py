"""How conductor writes a table as JSON.

A table is written as a list of its columns, in order. Each column has a name, a word for the type of its cells,
and the cells themselves::

    {"columns": [{"name": "customer", "type": "str", "values": ["Ann", "Bo"]},
                 {"name": "amount", "type": "float", "values": [12.5, 3.0]}]}

The columns are a list and not an object keyed by name, because a store may give an object's keys back in
another order. A ``bytes`` column's cells are written as base64 text, as all bytes are (``JSON_CONFIG``):
``{"name": "data", "type": "bytes", "values": ["__4="]}``.
"""

from __future__ import annotations

import copy
import datetime
import weakref
from collections.abc import Mapping
from decimal import Decimal
from typing import Any, cast

import polars as pl
from pydantic_core import CoreSchema, core_schema

from conductor._internal.base import json_hash, json_typed_dict_schema

#: The cell types a column is written with by name, each with its polars dtype and the schema that converts its
#: cells back from JSON. A column of any other kind, such as a list or a struct, is written with the word ``json``
#: and comes back as the JSON it was written as.
CELLS: Mapping[type, tuple[Any, CoreSchema]] = {
    bool: (pl.Boolean, core_schema.bool_schema()),
    int: (pl.Int64, core_schema.int_schema()),
    float: (pl.Float64, core_schema.float_schema()),
    str: (pl.String, core_schema.str_schema()),
    bytes: (pl.Binary, core_schema.bytes_schema()),
    datetime.date: (pl.Date, core_schema.date_schema()),
    datetime.datetime: (pl.Datetime, core_schema.datetime_schema()),
    datetime.time: (pl.Time, core_schema.time_schema()),
    datetime.timedelta: (pl.Duration, core_schema.timedelta_schema()),
    Decimal: (pl.Decimal, core_schema.decimal_schema()),
}

#: The polars dtype for each word a written column gives as its type, such as ``"int"`` for ``pl.Int64``.
DTYPES = {kind.__name__: dtype for kind, (dtype, _) in CELLS.items()}

#: The pydantic schema of a table's JSON, as the module docstring shows it. Each column's ``type`` word picks the
#: schema its cells are checked by. Each column is read by conductor's config (``json_typed_dict_schema``), so a
#: ``bytes`` column reads its cells from base64 text, as every ``bytes`` value does.
WRITTEN_TABLE = core_schema.typed_dict_schema(
    {
        "columns": core_schema.typed_dict_field(
            core_schema.list_schema(
                core_schema.tagged_union_schema(
                    {
                        word: json_typed_dict_schema(
                            {
                                "name": core_schema.typed_dict_field(core_schema.str_schema()),
                                "type": core_schema.typed_dict_field(
                                    core_schema.literal_schema([word])
                                ),
                                "values": core_schema.typed_dict_field(
                                    core_schema.list_schema(core_schema.nullable_schema(cells))
                                ),
                            },
                            extra_behavior="forbid",
                        )
                        for word, cells in [
                            *((kind.__name__, cells) for kind, (_, cells) in CELLS.items()),
                            ("json", core_schema.any_schema()),
                        ]
                    },
                    discriminator="type",
                )
            )
        ),
    },
    extra_behavior="forbid",
)


def written_table(frame: pl.DataFrame) -> dict[str, Any]:
    """Converts a frame to the JSON conductor writes for a table."""
    return written_columns(frame, frame.to_dict(as_series=False))


def written_columns(frame: pl.DataFrame, cells: Mapping[str, list[Any]]) -> dict[str, Any]:
    """Writes a table as JSON from cells that are already converted.

    The column types come from ``frame``, and the values from ``cells``, a list of cells per column name. A caller
    that has already turned the cells into JSON values passes them here, so they are not converted twice.
    """
    columns = []
    for (name, values), dtype in zip(cells.items(), frame.dtypes, strict=True):
        kind = dtype.to_python()
        columns.append(
            {"name": name, "type": kind.__name__ if kind in CELLS else "json", "values": values}
        )
    return {"columns": columns}


#: The digests ``table_digest`` has worked out, by ``id(frame)``. A frame has no hash, so it can't be a dict key
#: itself. Each entry keeps a weak reference to its frame, which removes the entry when the frame is freed.
_DIGESTS: dict[int, tuple[weakref.ref[pl.DataFrame], str]] = {}


def table_digest(frame: pl.DataFrame) -> str:
    """Returns a short hash of a table's JSON: its column names, cell types and cells.

    The hash is worked out the first time a frame is asked about, and the same frame object gets the same answer
    after that. Say 2,000 calls each receive one 2,000-row table. When a new run checks which calls it can reuse,
    the table is written as JSON once, not 2,000 times.

    Two frames with equal cells get equal hashes. A frame changed in place after its hash was worked out keeps the
    old hash, so a node must not change a table it was given.
    """
    kept = _DIGESTS.get(id(frame))
    if kept is not None and kept[0]() is frame:
        return kept[1]
    digest = json_hash(written_table(frame), 16)
    _keep_digest(frame, digest)
    return digest


def own_copy[V](value: V) -> V:
    """Returns a copy of ``value`` that shares nothing a node can change with it.

    A run gets its own copy of its graph's constants and of the defaults of its graph inputs, once, when it is made.
    So a node that changes a value it was given, which it must not do, can't change a later run of the same graph.

    A frame is copied with ``clone()``. The clone shares the frame's memory until one of them is changed, so it costs
    almost nothing, and it gets the frame's digest, so ``table_digest`` doesn't hash the table again for every run.
    Any other value is copied with ``copy.deepcopy``, which keeps its exact class. Its cost grows with the value's
    size: a few microseconds for a small value, about 10 ms for a 10k-row list of dicts. A frame inside another value
    is cloned too, but its digest is worked out again. A value ``copy.deepcopy`` can't copy raises its ``TypeError``,
    so the run is never made.

    Raises:
        TypeError: ``copy.deepcopy`` can't copy the value.
    """
    if not isinstance(value, pl.DataFrame):
        return copy.deepcopy(value)
    digest = table_digest(value)
    clone = value.clone()
    _keep_digest(clone, digest)
    return cast("V", clone)


def _keep_digest(frame: pl.DataFrame, digest: str) -> None:
    """Stores ``digest`` as the digest of ``frame``, until the frame is freed."""
    key = id(frame)
    _DIGESTS[key] = (weakref.ref(frame, lambda _: _DIGESTS.pop(key, None)), digest)
