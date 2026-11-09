"""How an input receives its value: one record per input, what the run reads and what ``compiled.run_plan`` shows.

A node in a graph runs as units, one per row of its index, or one unit when it
runs once. For every input, the question is: *at a unit's row, which of the
input's values does the unit receive?* Compile answers it once, while it walks
the edges, and stores the answer on the field (``CompiledField._receive_mode``).
The run reads that record to decide when a unit is ready and what to hand it,
and never works the answer out again from the types and indexes.

The run tells apart only what it reads differently, by ``kind``:

- ``Cell``: the value at the unit's row, cut to the depth of the field it reads;
- ``RowsUnder``: all the rows of a field under the unit's row cut to ``depth``,
  ``0`` being everything on the field;
- ``FixedValue``: a value compile already holds, a static or a default;
- ``Gather``: unrelated sources collected onto an index of the input's own;
- ``Merge``: one table of everything wired into a ``Table`` input, matched on
  its rows' ids, each source read as a ``Cell`` or a ``RowsUnder``.

``receive_mode`` is what a person reads about the same thing, in a word:

- ``value_at_row``: the value at the call's own row;
- ``same_value_every_row``: one value, copied to every call, from an edge or set on the node;
- ``all_rows_under_row``: every row of a column under the call's row, however deep, flat;
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


@dataclass(frozen=True, init=False)
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
    #: A blank here calls the node with ``None``. If not, the row isn't called.
    takes_blank: bool
    kind: Literal["cell"] = "cell"

    def __init__(self, receive_mode: str, sources: tuple[Ref, ...], rows_of: Ref | None, depth: int, takes_blank: bool, kind: str = "cell") -> None:
        """Compile builds one per input of a graph, so this sets them in one step, not one at a time."""
        self.__dict__.update(receive_mode=receive_mode, sources=sources, rows_of=rows_of, depth=depth, takes_blank=takes_blank, kind=kind)

    def __str__(self) -> str:
        if self.rows_of is not None:
            return f"the value at my row ({self.rows_of})"
        return f"the same value every row ({' + '.join(self.sources)})"


@dataclass(frozen=True, slots=True)
class RowsUnder:
    """The unit receives the rows of its sources under its own row cut to ``depth``, flat.

    A ``Series[X]`` input fed a column on a child index reduces the rows
    under the node's row, however many steps down they sit; ``depth`` is that
    of the deepest index above them that the node's row sits on. ``0`` is
    everything on the field: a ``Series[X]`` fed one column on a root index.
    """

    receive_mode: Literal["all_rows_under_row", "entire_value"]
    sources: tuple[Ref, ...]
    #: The field whose rows it reads; ``None`` when it reads them all.
    rows_of: Ref | None
    #: The index the rows are grouped under, which a call's row sits on.
    grouped_under: Ref | None
    depth: int
    #: A blank here stays in the series. If not, its row is left out.
    takes_blank: bool
    kind: Literal["rows_under"] = "rows_under"

    def __str__(self) -> str:
        if self.rows_of is None:
            return f"all of {' + '.join(self.sources)}"
        grouped_under = "" if self.grouped_under == self.rows_of else f" (grouped under {self.grouped_under})"
        return f"all rows of {self.rows_of} under my row{grouped_under}; none under it: called with an empty list"


@dataclass(frozen=True, init=False)
class FixedValue:
    """The unit receives a value compile already holds: a static or a default.

    One value, or for a ``Series[X]`` input a list on an index of the
    input's own. The value stays out of the run plan's written form: a host
    sees it on the graph, and a fingerprint hashes it by its own type.
    """

    receive_mode: Literal["same_value_every_row", "entire_value"]
    value: Annotated[Any, Field(exclude=True)] = field(default=None, repr=False, compare=False)
    #: A blank here calls the node with ``None``. If not, the row isn't called.
    takes_blank: bool = False
    kind: Literal["fixed"] = "fixed"

    def __init__(self, receive_mode: str, value: Any = None, takes_blank: bool = False, kind: str = "fixed") -> None:
        """Compile builds one per input left to a value, so this sets them in one step, not one at a time."""
        self.__dict__.update(receive_mode=receive_mode, value=value, takes_blank=takes_blank, kind=kind)

    @property
    def sources(self) -> tuple[Ref, ...]:
        return ()

    @property
    def depth(self) -> int:
        return 0

    def __str__(self) -> str:
        return "the same value every row" if self.receive_mode == "same_value_every_row" else "the whole list set on the node"


@dataclass(frozen=True, slots=True)
class Gather:
    """The unit receives its unrelated sources collected into one series.

    Several scalars, or several series on indexes that share no rows, into one
    ``Series[X]`` input: each is read under the unit's row cut to ``depth``, ``0``
    being all of it, and laid out on ``collected_onto``, the input's own index. A
    skipped source is left out rather than skipping the unit, since gathering is
    how "whichever branch fired" is expressed.
    """

    sources: tuple[Ref, ...]
    collected_onto: Ref
    depth: int
    #: A blank here stays in the series. If not, its row is left out.
    takes_blank: bool
    receive_mode: Literal["collected"] = "collected"
    kind: Literal["gather"] = "gather"

    def __str__(self) -> str:
        return f"collected into one list: {' + '.join(self.sources)}"


#: How one source of a merge is read: the two ways a table or a column reaches a row.
SourceOfMerge = Annotated[Cell | RowsUnder, Field(discriminator="kind")]


@dataclass(frozen=True, slots=True)
class Merge:
    """The unit receives one table merged from every source, each read as its plan says.

    Several tables or columns into one ``Table`` input: a table at the unit's
    row or the one there is, a column's rows under the unit's row or all of
    them, and the pieces are merged on their rows' ids: the rows are the deepest
    sources', a parent's value copied down onto each of them, and a parent row
    nothing sits under is not in the table. A ``Table`` input never makes its
    node run per row by itself, and its sources never have to line up: rows that
    match nothing are blanks. A table wired in alone is received as a ``Cell``.
    """

    sources: tuple[Ref, ...]
    #: One record per source, in wiring order.
    source_plans: tuple[SourceOfMerge, ...]
    #: The sources whose rows sit above another source's: copied down onto those rows.
    copied_down: tuple[Ref, ...]
    #: A blank here stays in the table.
    takes_blank: bool
    receive_mode: Literal["merged"] = "merged"
    kind: Literal["merge"] = "merge"

    def __str__(self) -> str:
        return "merged: " + " + ".join(f"{source}{' [copied down]' if source in self.copied_down else ''}" for source in self.sources)


ReceiveMode = Annotated[Cell | RowsUnder | FixedValue | Gather | Merge, Field(discriminator="kind")]


def cell_at_the_row(sources: tuple[Ref, ...], index: Index | None, takes_blank: bool) -> Cell:
    """The ``Cell`` of ``sources``, which sit on ``index``, or hold one value with none."""
    if index is None:
        return Cell("same_value_every_row", sources, None, 0, takes_blank)
    return Cell("value_at_row", sources, Ref._from_checked_address(index.id), index.depth, takes_blank)


def rows_under_the_row(sources: tuple[Ref, ...], index: Index, depth: int, takes_blank: bool) -> RowsUnder:
    """The ``RowsUnder`` of ``sources`` on ``index``, grouped under its step at ``depth``, ``0`` being all of them."""
    if depth == 0:
        return RowsUnder("entire_value", sources, None, None, 0, takes_blank)
    return RowsUnder("all_rows_under_row", sources, Ref._from_checked_address(index.id), Ref._from_checked_address(index.lineage[depth - 1].id), depth, takes_blank)


def reads_of_input(receive_mode: Cell | RowsUnder | FixedValue | Gather | Merge) -> list[tuple[Ref, int]]:
    """Every field this input reads and the depth of the unit's row it is read at, in wiring order."""
    if isinstance(receive_mode, Merge):
        return [(source, part.depth) for part in receive_mode.source_plans for source in part.sources]
    return [(source, receive_mode.depth) for source in receive_mode.sources]
