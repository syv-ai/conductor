"""``RunState`` — a frozen snapshot of what a run has produced so far, as a host stores it between legs.

A run takes several legs when a node waits on a person: one call of
``execute`` is a leg, and the next leg has to start from everything the
earlier ones produced. The run's state is that everything, in a form a
host can keep in a row and hand back: every value the ledger holds, in
JSON form, the rows born on each index, which indexes are sealed, where an
index has no rows, which units are done — and a fingerprint per node of
how the graph placed it. It accumulates across legs: each leg starts from
the state the last one ended on and ends on a new one.

Written by ``Ledger.state()`` at the end of every leg and carried on the
ending event as ``state``; read by ``Ledger.restore`` when a host passes
it to ``execute(state=...)``. The fingerprints are what make a state safe
to restore into a graph the author has since edited: a node whose
fingerprint differs from the graph's — a static edited, a version bumped,
a binding moved — is dropped with everything downstream of it and runs
again, and so is a node the graph no longer has.

Not the results: ``compiled.results(state)`` is what the run produced by
node and output, as values; the state is the ledger's whole bookkeeping,
in wire form, and a host never reads inside it.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from conductor.model import ConductorModel


class RunState(ConductorModel):
    """A frozen snapshot of a run's state: the ledger in wire form, plus a fingerprint per node.

    A snapshot, not live state: the ledger a leg runs on is the live thing,
    and this is what it wrote down when the leg ended. Nothing changes it
    in place; ``without`` returns a new one.

    ``values`` are what each output holds at each row, each ``{"ref": [node,
    field], "row": [...] or null}`` with either ``"value"`` in JSON form or
    ``"skipped"`` (the depth of the row the skip was written at). The next four are the
    ledger's row bookkeeping, keyed by index id: ``rows_by_index`` is every
    row born on an index; ``sealed_indexes`` are the indexes that will
    never gain a row; ``childless_parent_rows`` are, per index, the parent
    rows under which it bore no row at all; ``done_units`` are the units
    that ran to completion, a node and the row it ran for (``null`` for a
    node that ran once). ``node_fingerprints`` is one hash per node of its
    placement in the graph, ``CompiledNode.fingerprint``. ``RunState()``
    is the state of a run that has produced nothing.
    """

    values: list[dict[str, Any]] = Field(default_factory=list)
    rows_by_index: dict[str, list[list[int]]] = Field(default_factory=dict)
    sealed_indexes: list[str] = Field(default_factory=list)
    childless_parent_rows: dict[str, list[list[int] | None]] = Field(default_factory=dict)
    done_units: list[tuple[str, list[int] | None]] = Field(default_factory=list)
    node_fingerprints: dict[str, str] = Field(default_factory=dict)

    def without(self, *node_ids: str) -> RunState:
        """This state as if these nodes had never run: a host's "run from here".

        Their values, rows and done units go, and so do their fingerprints,
        which is what tells ``Ledger.restore`` to run them again with
        everything that reads them. A node id also covers the inner nodes of
        an embedded graph placed under it (``approve`` covers ``approve/check``).
        """
        def gone(node_id: str) -> bool:
            return any(node_id == dropped or node_id.startswith(f"{dropped}/") for dropped in node_ids)

        return self.model_copy(update={
            "values": [entry for entry in self.values if not gone(entry["ref"][0])],
            "rows_by_index": {index_id: rows for index_id, rows in self.rows_by_index.items() if not gone(index_id)},
            "sealed_indexes": [index_id for index_id in self.sealed_indexes if not gone(index_id)],
            "childless_parent_rows": {index_id: rows for index_id, rows in self.childless_parent_rows.items() if not gone(index_id)},
            "done_units": [(node_id, row) for node_id, row in self.done_units if not gone(node_id)],
            "node_fingerprints": {node_id: fingerprint for node_id, fingerprint in self.node_fingerprints.items() if not gone(node_id)},
        })
