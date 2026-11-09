"""Conductor: declare nodes, put them in a graph, compile it, run it.

This package is the main public door. With ``conductor.widgets`` (the
input controls) and ``conductor.events`` (what a run emits) it is
everything a caller outside the library imports; ``__all__`` lists it.
Everything in ``conductor._internal`` is the library's own and may move.
"""

from typing import TYPE_CHECKING

from conductor._internal.codec import to_jsonable
from conductor._internal.dtype import AsItArrives, DType, Series
from conductor._internal.errors import (
    PROBLEM_CODES,
    ComputeHookError,
    ExternalFailure,
    InvalidCallError,
    NodeDeclarationError,
    NodeError,
    Problem,
)
from conductor._internal.index import ValuesOnRows
from conductor._internal.ref import Ref
from conductor._internal.table import ROW_ID_COLUMN, table_with_columns

if TYPE_CHECKING:
    # A node takes and returns frames; a type checker reads ``Table[R]`` as one.
    from conductor._internal.table_as_a_type_checker_reads_it import Table
else:
    from conductor._internal.table import Table

__all__ = [
    "PROBLEM_CODES",
    "ROW_ID_COLUMN",
    "AsItArrives",
    "ComputeHookError",
    "DType",
    "ExternalFailure",
    "InvalidCallError",
    "NodeDeclarationError",
    "NodeError",
    "Problem",
    "Ref",
    "Series",
    "Table",
    "ValuesOnRows",
    "table_with_columns",
    "to_jsonable",
]
