"""Which rows can share ids: what a merge may expect, and what it refuses, read off the edges.

A row keeps its id as it passes through a node, so a table's rows can trace
back to the tables wired into the node that returned it. That one fact gives
two answers while compile walks forward: a merge of sources that can never
share an id is the fatal ``no_rows_in_common``, and a merge against a table
a source was built from makes that source's table ``row_ids_must_survive``:
at run time, a return where no row kept its id is ``row_ids_lost``.

Each output's origins sit on its field (``_row_origins``), as index ids; a
table's rows' origins are also kept by index, for the columns and series
that sit on them. A list on a node is an index of its own, and a
graph embedded in another has its origins worked out by the same rule
where it stands, so nothing stops at an embedding.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from conductor._internal.errors import Problem, make_problem
from conductor._internal.ref import Ref
from conductor._internal.table import is_table_type

if TYPE_CHECKING:
    from conductor._internal.graph.compiled_node import CompiledField
    from conductor._internal.index import Index


@dataclass
class RowOriginsSeen:
    """What the walk has learned about row origins so far.

    ``origins_by_table_index``: per index id of a table's rows, the indexes those rows can trace back to, ``None`` when unknown.
    ``must_survive``: the table outputs a later merge matches against what they were built from.
    """

    origins_by_table_index: dict[str, frozenset[str] | None] = field(default_factory=dict)
    must_survive: set[Ref] = field(default_factory=set)

    def origins_of_rows_on(self, index: Index | None) -> frozenset[str] | None:
        """Where the ids of rows on ``index`` can come from: the index itself, or for a table's rows what that table carried in too."""
        if index is None:
            return frozenset()
        return self.origins_by_table_index.get(index.id, frozenset({index.id}))

    def record_table(self, table_rows_index: Index, taken_in: frozenset[str] | None) -> frozenset[str] | None:
        """Record a table a node returns, its rows on ``table_rows_index``: they can carry the ids of what it took in. Returns its origins."""
        known = None if taken_in is None else taken_in | {table_rows_index.id}
        self.origins_by_table_index[table_rows_index.id] = known
        return known

    def check_merge(self, node_id: str, field_name: str, places: list[list[tuple[Ref, CompiledField]]]) -> list[Problem]:
        """A merge of these sources, one place per set of alternatives: mark the tables it needs to keep their ids, and refuse a source that can never match. One place is no merge."""
        if len(places) < 2:
            return []
        sources = [pair for place in places for pair in place]
        origins_each = [source._row_origins for _, source in sources]
        known = [origins for origins in origins_each if origins is not None]
        for origin in frozenset[str]().union(*known):
            built_from = (self.origins_by_table_index.get(origin) or frozenset()) - {origin}
            if any(built_from & other and origin not in other for other in known):
                self.must_survive.add(Ref(origin))
        if len(known) < len(origins_each):
            return []
        source_indexes = [source._held_as.rows_made or source._index for _, source in sources]
        parent_index_ids = [frozenset() if index is None else frozenset(step.id for step in index.lineage[:-1]) for index in source_indexes]

        def can_share_rows(left: int, right: int) -> bool:
            """Can a row of source ``left`` meet one of source ``right``: by one id, or as its parent copied down?"""
            return bool(known[left] & known[right] or known[left] & parent_index_ids[right] or known[right] & parent_index_ids[left])

        for left, (source_ref, _) in enumerate(sources):
            if not any(can_share_rows(left, right) for right in range(len(sources)) if right != left):
                return [make_problem("no_rows_in_common", node_id, field_name, source=str(source_ref))]
        return []


def sources_taken_in_whole(reader_type: Any, sources: Sequence[CompiledField]) -> list[CompiledField]:
    """The sources a node takes in as a whole table, whichever side says so: a table it returns may carry their rows' ids."""
    return list(sources) if is_table_type(reader_type) else [source for source in sources if is_table_type(source.type)]


def origins_taken_in(sources: list[CompiledField]) -> frozenset[str] | None:
    """The origins of every table or merged column a node took in: a table it returns may carry their rows' ids. ``None`` when any is unknown."""
    taken: set[str] = set()
    for source in sources:
        if source._row_origins is None:
            return None
        taken |= source._row_origins
    return frozenset(taken)
