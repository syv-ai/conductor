"""How an input receives its value: one record per input, what the run reads and what ``compiled.run_plan`` shows.

A node in a graph runs as units, one per row of its index, or one unit when it
runs once. For every input, the question is: *at a unit's row, which of the
input's values does the unit receive?* Compile answers it once, while it walks
the edges, and stores the answer on the field (``CompiledField._receive_mode``).
The run reads that record to decide when a unit is ready and what to hand it,
and never works the answer out again from the types and indexes.

The run tells apart only what it reads differently, by ``type``:

- ``Cell``: the value at the unit's row, cut to the depth of the field it reads;
- ``RowsUnder``: all the rows of a field under the unit's row cut to ``depth``,
  ``0`` being everything on the field;
- ``FixedValue``: a value compile already holds, a static or a default;
- ``Gather``: unrelated sources collected into one list;
- ``Merge``: one table of everything wired into a ``Table`` input, matched on
  its rows' ids.

``Gather`` and ``Merge`` hold one read per source, a ``Cell`` or a ``RowsUnder``
settled once the node's row is known, so each source is read by how it stands
to that row and a unit waits only for what its own reads cover.

``receive_mode`` is what a person reads about the same thing, in a word:

- ``value_at_row``: the value at the call's own row;
- ``same_value_every_row``: one value, copied to every call, from an edge or set on the node;
- ``all_rows_under_row``: every row of a column under the call's row, however deep, flat; none there is an empty list, a skip above them no call;
- ``entire_value``: everything on the field, once;
- ``collected``: unrelated sources gathered into one series;
- ``merged``: tables and columns matched on their rows' ids into one table.

These say how a value is *received*, not where it comes from: an edge's
source is on the record, a binding is a different record on the same field, and
an output has no receive mode at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import Field

from conductor._internal.index import Index
from conductor._internal.ref import Ref


@dataclass(frozen=True, slots=True)
class Cell:
    """The unit receives the value at its own row cut to ``depth``, from whichever source covers it.

    A scalar input fed a series on an index: the node runs once per row of it.
    Fed by a node that ran once, ``depth`` is ``0`` and it is the one value
    there is. A list set on a scalar input is a ``Cell`` of its own field, read
    the way an edge reads it.
    """

    receive_mode: Literal["value_at_row", "same_value_every_row"]
    #: The outputs wired to it, or for a list set on it the input itself.
    sources: tuple[Ref, ...]
    #: The field whose rows it reads; ``None`` when there is one value.
    rows_of: Ref | None
    depth: int
    type: Literal["cell"] = "cell"

    @property
    def _what_is_read(self) -> str:
        if self.rows_of is not None:
            return f"the value at my row ({' + '.join(self.sources)})"
        return f"the same value every row ({' + '.join(self.sources)})"

    def __str__(self) -> str:
        return self._what_is_read


@dataclass(frozen=True, slots=True)
class RowsUnder:
    """The unit receives the rows of its sources under its own row cut to ``depth``, flat.

    A ``Series[X]`` input fed a column on a child index reduces the rows
    under the node's row, however many steps down they sit; ``depth`` is that
    of the deepest index above them that the node's row sits on. ``0`` is
    everything on the field: a ``Series[X]`` fed one column on a root index.
    A source of a ``Gather`` or a ``Merge`` on the node's own row or a row above
    it is read to that row's depth: the one row it has there.
    """

    receive_mode: Literal["all_rows_under_row", "entire_value"]
    sources: tuple[Ref, ...]
    #: The field whose rows it reads; ``None`` when it reads them all.
    rows_of: Ref | None
    #: The index the rows are grouped under, which a call's row sits on.
    grouped_under: Ref | None
    depth: int
    type: Literal["rows_under"] = "rows_under"

    @property
    def _what_is_read(self) -> str:
        if self.rows_of is None:
            return f"all of {' + '.join(self.sources)}"
        if self.grouped_under == self.rows_of:
            return f"the value at my row ({' + '.join(self.sources)})"
        return f"all rows of {' + '.join(self.sources)} under my row (grouped under {self.grouped_under})"

    def __str__(self) -> str:
        if self.rows_of is None or self.grouped_under == self.rows_of:
            return self._what_is_read
        return f"{self._what_is_read}; none under it: called with an empty list; skipped above them: not called"


@dataclass(frozen=True, slots=True)
class FixedValue:
    """The unit receives a value compile already holds: a static or a default.

    One value, or for a ``Series[X]`` input a list. The value stays out of the run plan's written form: a host
    sees it on the graph, and a fingerprint hashes it by its own type.
    """

    receive_mode: Literal["same_value_every_row", "entire_value"]
    value: Annotated[Any, Field(exclude=True)] = field(default=None, repr=False, compare=False)
    type: Literal["fixed"] = "fixed"

    @property
    def sources(self) -> tuple[Ref, ...]:
        return ()

    @property
    def depth(self) -> int:
        return 0

    def __str__(self) -> str:
        return "the same value every row" if self.receive_mode == "same_value_every_row" else "the whole list set on the node"


#: How one source of a ``Gather`` or a ``Merge`` is read at a unit's row.
ReadOfSource = Annotated[Cell | RowsUnder, Field(discriminator="type")]


@dataclass(frozen=True, slots=True)
class Gather:
    """The unit receives its unrelated sources collected into one series.

    Several scalars, several series on indexes that share no rows, or the
    values of one run of the graph around the node, into one ``Series[X]``
    input. Each source is read by its own record. A skipped source is left out
    rather than skipping the unit, since gathering is how "whichever branch
    fired" is expressed.
    """

    #: One read per source, in wiring order.
    source_reads: tuple[ReadOfSource, ...]
    receive_mode: Literal["collected"] = "collected"
    type: Literal["gather"] = "gather"

    def __str__(self) -> str:
        return f"collected into one list: {' + '.join(read._what_is_read for read in self.source_reads)}"


@dataclass(frozen=True, slots=True)
class Merge:
    """The unit receives one table merged from every source, each read by its own record.

    Several tables or columns into one ``Table`` input: a table is a ``Cell``, the
    one at the unit's row or the one there is; a column is a ``RowsUnder``, its
    rows under the unit's row or all of them. The pieces are merged on their
    rows' ids: the rows are the deepest sources', a parent's value copied down
    onto each of them, and a parent row nothing sits under is not in the table.
    A ``Table`` input never makes its node run per row by itself, and its sources
    never have to line up: rows that match nothing are blanks. Tables that are
    alternatives, only one of them ever at a row, are one read of several
    sources; a table wired in alone, or one set of alternatives, is a ``Cell``.
    """

    #: One read per source, in wiring order.
    source_reads: tuple[ReadOfSource, ...]
    #: The sources whose rows sit above another source's: copied down onto those rows.
    copied_down: tuple[Ref, ...]
    #: Per source, the positions of the other sources whose columns its table was made from: a column of the same name is one column.
    carries_columns_of: tuple[tuple[int, ...], ...]
    receive_mode: Literal["merged"] = "merged"
    type: Literal["merge"] = "merge"

    def __str__(self) -> str:
        return "merged: " + " + ".join(f"{' or '.join(read.sources)}{' [copied down]' if read.sources[0] in self.copied_down else ''}" for read in self.source_reads)


ReceiveMode = Annotated[Cell | RowsUnder | FixedValue | Gather | Merge, Field(discriminator="type")]


def cell_at_the_row(sources: tuple[Ref, ...], index: Index | None) -> Cell:
    """The ``Cell`` of ``sources``, which sit on ``index``, or hold one value with none."""
    if index is None:
        return Cell("same_value_every_row", sources, None, 0)
    return Cell("value_at_row", sources, Ref(index.id), index.depth)


def rows_under_the_row(sources: tuple[Ref, ...], index: Index, depth: int) -> RowsUnder:
    """The ``RowsUnder`` of ``sources`` on ``index``, grouped under its step at ``depth``, ``0`` being all of them."""
    if depth == 0:
        return RowsUnder("entire_value", sources, None, None, 0)
    return RowsUnder("all_rows_under_row", sources, Ref(index.id), Ref(index.lineage[depth - 1].id), depth)


def reads_of_input(receive_mode: Cell | RowsUnder | FixedValue | Gather | Merge) -> list[tuple[Ref, int]]:
    """Every field this input reads and the depth of the unit's row it is read at, in wiring order."""
    reads = receive_mode.source_reads if isinstance(receive_mode, Gather | Merge) else (receive_mode,)
    return [(source, read.depth) for read in reads for source in read.sources]
