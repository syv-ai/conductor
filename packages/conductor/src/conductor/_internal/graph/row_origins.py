"""Which rows can share ids: what a merge may expect, and what it may warn about, read off the edges.

A row keeps its id as it passes through a node, so a table's rows can trace
back to the tables wired into the node that returned it. That one fact gives
two answers while compile walks forward: a merge of sources that can never
share an id is the warning ``no_rows_in_common``, and a merge against a table
a source was built from makes that source's table ``row_ids_must_survive``:
at run time, a return where no row kept its id is ``row_ids_lost``.

Each output's origins sit on its field (``_row_origins``), as index ids; a
table's rows' origins are also kept by index, for the columns and series
that sit on them. A list typed into a node is an index of its own, and a
graph embedded in another has its origins worked out by the same rule
where it stands, so nothing stops at an embedding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from conductor._internal.errors import Problem, problem
from conductor._internal.ref import Ref
from conductor._internal.table import holds_tables

if TYPE_CHECKING:
    from conductor._internal.graph.compiled_node import CompiledField
    from conductor._internal.series import Index


@dataclass
class RowOriginsSoFar:
    """What the walk has learned about row origins so far.

    ``of_table_rows``: per index id of a table's rows, the indexes those rows can trace back to, ``None`` when unknown.
    ``must_survive``: the table outputs a later merge matches against what they were built from.
    """

    of_table_rows: dict[str, frozenset[str] | None] = field(default_factory=dict)
    must_survive: set[Ref] = field(default_factory=set)

    def of_rows_on(self, index: Index | None) -> frozenset[str] | None:
        """Where the ids of rows on ``index`` can come from: the index itself, or for a table's rows what that table carried in too."""
        if index is None:
            return frozenset()
        return self.of_table_rows.get(index.id, frozenset({index.id}))

    def note_table(self, table_rows_index: Index, taken_in: frozenset[str] | None) -> frozenset[str] | None:
        """Record a table a node returns, its rows on ``table_rows_index``: they can carry the ids of what it took in. Returns its origins."""
        known = None if taken_in is None else taken_in | {table_rows_index.id}
        self.of_table_rows[table_rows_index.id] = known
        return known

    def note_merge(self, node_id: str, field_name: str, sources: list[tuple[Ref, CompiledField]]) -> list[Problem]:
        """A merge of ``sources``: mark the tables it needs to keep their ids, and say if a source can never match."""
        found = [source._row_origins for _, source in sources]
        known = [origins for origins in found if origins is not None]
        for origin in frozenset[str]().union(*known):
            built_from = (self.of_table_rows.get(origin) or frozenset()) - {origin}
            if any(built_from & other and origin not in other for other in known):
                self.must_survive.add(Ref(origin))
        if len(sources) < 2 or len(known) < len(found):
            return []
        source_indexes = [source._makes_rows or source._index for _, source in sources]
        above = [frozenset() if index is None else frozenset(step.id for step in index.lineage[:-1]) for index in source_indexes]

        def meet(n: int, m: int) -> bool:
            """Can a row of source ``n`` meet one of source ``m``: by one id, or as its parent copied down?"""
            return bool(known[n] & known[m] or known[n] & above[m] or known[m] & above[n])

        for n, (source_ref, _) in enumerate(sources):
            if not any(meet(n, m) for m in range(len(sources)) if m != n):
                return [problem("no_rows_in_common", node_id, field_name, source=str(source_ref))]
        return []


def sources_taken_in_whole(reader_type: Any, sources: Sequence[CompiledField]) -> list[CompiledField]:
    """The sources a node takes in as a whole table, whichever side says so: a table it returns may carry their rows' ids."""
    return list(sources) if holds_tables(reader_type) else [source for source in sources if holds_tables(source.type)]


def origins_taken_in(sources: list[CompiledField]) -> frozenset[str] | None:
    """The origins of every table or merged column a node took in: a table it returns may carry their rows' ids. ``None`` when any is unknown."""
    taken: set[str] = set()
    for source in sources:
        if source._row_origins is None:
            return None
        taken |= source._row_origins
    return frozenset(taken)
