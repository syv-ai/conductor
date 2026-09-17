"""``RunRecord`` — what a run has produced so far, as a host stores it between legs.

A run takes several legs when a node waits on a person: one call of
``execute`` is a leg, and the next leg has to start from everything the
earlier ones produced. This record is that everything, in a form a host
can keep in a row and hand back: every cell of the ledger in JSON form,
the rows born on each index, which indexes are sealed, where an index has
no rows, which units are done — and a fingerprint per node of how the
graph placed it.

Written by ``Ledger.cells()`` at the end of every leg and carried on the
ending event as ``record``; read by ``Ledger.restore`` when a host passes
it to ``execute(record=...)``. The fingerprints are what make a record
safe to restore into a graph the author has since edited: a node whose
fingerprint differs from the graph's — a static edited, a version bumped,
a binding moved — is dropped with everything downstream of it and runs
again, and so is a node the graph no longer has.

Not the results: ``results`` on the same events is what a leg produced by
node and output, as values; the record is the ledger's whole state, in
wire form, and a host never reads inside it.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from conductor.model import ConductorModel


class RunRecord(ConductorModel):
    """The ledger's whole state in wire form, plus a fingerprint per node.

    ``cells`` are the ledger's cells, each ``{"ref": [node, field], "row":
    [...] or null}`` with either ``"value"`` in JSON form or ``"skipped"``
    (the depth of the row the skip was written at). ``rows``, ``sealed``,
    ``no_rows_under`` and ``done`` are the ledger's row bookkeeping, keyed
    by index id. ``fingerprints`` is one hash per node of its placement in
    the graph, ``CompiledNode.fingerprint``. ``RunRecord()`` is the record
    of a run that has produced nothing.
    """

    cells: list[dict[str, Any]] = Field(default_factory=list)
    rows: dict[str, list[list[int]]] = Field(default_factory=dict)
    sealed: list[str] = Field(default_factory=list)
    no_rows_under: dict[str, list[list[int] | None]] = Field(default_factory=dict)
    done: list[tuple[str, list[int] | None]] = Field(default_factory=list)
    fingerprints: dict[str, str] = Field(default_factory=dict)

    def without(self, *node_ids: str) -> RunRecord:
        """This record as if these nodes had never run: a host's "run from here".

        Their cells, rows and done units go, and so do their fingerprints,
        which is what tells ``Ledger.restore`` to run them again with
        everything that reads them. A node id also covers the inner nodes of
        an embedded graph placed under it (``approve`` covers ``approve/check``).
        """
        def gone(node_id: str) -> bool:
            return any(node_id == dropped or node_id.startswith(f"{dropped}/") for dropped in node_ids)

        return self.model_copy(update={
            "cells": [cell for cell in self.cells if not gone(cell["ref"][0])],
            "rows": {index_id: rows for index_id, rows in self.rows.items() if not gone(index_id)},
            "sealed": [index_id for index_id in self.sealed if not gone(index_id)],
            "no_rows_under": {index_id: rows for index_id, rows in self.no_rows_under.items() if not gone(index_id)},
            "done": [(node_id, row) for node_id, row in self.done if not gone(node_id)],
            "fingerprints": {node_id: fp for node_id, fp in self.fingerprints.items() if not gone(node_id)},
        })
