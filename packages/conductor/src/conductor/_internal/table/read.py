"""Helpers for reading a table: build a frame from columns, and tell which of the three accepted forms a value is
(a frame, a list of records, or the JSON form).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import polars as pl
from pydantic_core import PydanticCustomError


def _frame_of(columns: Iterable[tuple[str, list[Any], Any]]) -> pl.DataFrame:
    """Makes a frame of ``columns``, each a name, its cells and its polars dtype (``None`` to let polars find one).

    Raises:
        PydanticCustomError: Polars can't make a table of them (``not_a_table``): a name twice, lengths that
            differ, or cells of no one type.
    """
    try:
        return pl.DataFrame([pl.Series(name, cells, dtype=dtype, strict=False) for name, cells, dtype in columns])
    except (pl.exceptions.PolarsError, TypeError):  # polars' own account may quote a cell
        raise PydanticCustomError("not_a_table", "the columns don't make a table: one name twice, lengths that differ, or cells of no one type") from None


def _table_form(value: Any) -> str | None:
    """Says which of a table's three forms ``value`` is: a frame, a list of records, or the columns conductor writes."""
    if isinstance(value, pl.DataFrame):
        return "frame"
    return "written" if isinstance(value, Mapping) else "records" if isinstance(value, list) else None
