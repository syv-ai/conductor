"""How an input receives its value: one row at a time, one value everywhere, the whole thing, the rows under a row, a gathering, or a merge.

A node in a graph runs as units, one per row of its index, or one unit when it runs once. For every input, the question this
module answers is: *at a unit's row, which of the input's values does the
unit receive?* Compile answers it once, while it walks the edges, and
stores the answer on the field (``CompiledField._receive_mode``). The
engine reads that record to decide when a unit is ready and what to hand
it, and never works the answer out again from the types and indexes.

Six records, one per input, named for what happens to the value on its
way in. ``PerRow`` takes a series one row at a time: a scalar input fed
a series, so the node runs once per row and each unit takes the value at
its own row. ``Broadcast`` copies one value to every unit: a scalar fed by a
node that ran once, a static, a default. ``EntireValue`` leaves the
value as it sits on the field: a ``Series[X]`` input fed one series on a
root index takes it entire, and a ``**inputs`` name takes whatever its
edge carries. ``RowsUnder`` takes all the rows under the unit's own row, flat: a
``Series[X]`` input fed a series on a child index. ``Gather`` collects unrelated sources — several scalars,
several series on different indexes — onto a fresh index that belongs
to the input. ``Merge`` makes one table of everything wired into a
``Table`` input, matched on the rows' ids, each source received by a
mode of its own.

These say how a value is *received*, not where it comes from: the
binding (an edge from another node's output, or a static) is a different
record on the same field, and an output has no receive mode at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from conductor._internal.ref import Ref
from conductor._internal.series import Index


@dataclass(frozen=True)
class PerRow:
    """The unit receives the value at its own row, projected onto ``index``.

    A scalar input fed a series on ``index``: the node runs once per row
    of it, and each unit reads the row it stands on. A list set on a
    scalar input is the same thing on an index of the input's own, one
    row per value set.
    """

    index: Index


@dataclass(frozen=True)
class Broadcast:
    """The unit receives the one value there is, the same at every unit.

    A scalar input fed by an edge from a node that ran once, a static,
    or a default. The node does not run per row because of
    this input; when it runs per row for another, this value is copied to
    every row.
    """


@dataclass(frozen=True)
class EntireValue:
    """The unit receives everything on the field, as it sits there.

    A ``**inputs`` parameter takes whatever its one edge carries, series
    and all; a ``Series[X]`` input fed one series on a root index takes
    that series entire; a ``Series[X]`` input with a list set on it or a
    default takes it as a series on its own index. The node does not run
    per row because of this input.
    """


@dataclass(frozen=True)
class RowsUnder:
    """The unit receives the rows of ``index`` under its own row cut to ``depth``.

    A ``Series[X]`` input fed a series on a child index reduces the rows
    under the node's row, however many steps down they sit, flat; ``depth``
    is that of the deepest index above them that the node's row sits on. Inside an embedded graph that runs once per outer row, a
    series that entered through a scalar field is reduced one row at a
    time — ``depth`` is then the index's own, and the group is one row.
    """

    index: Index
    depth: int


@dataclass(frozen=True)
class Gather:
    """The unit receives its unrelated sources collected onto ``index``, a fresh index of the input's own.

    Several scalars, or several series on indexes that share no rows,
    into one ``Series[X]`` input: the values are read whole and laid out on
    ``index``, which is the field's own index. A skipped source is left
    out rather than skipping the unit, since gathering is how "whichever
    branch fired" is expressed.
    """

    index: Index


@dataclass(frozen=True)
class Merge:
    """The unit receives one table merged from every source, each received as its ``mode`` says.

    Several tables or columns into one ``Table`` input: each source is read
    as its part says — a table value at the unit's row (``PerRow``) or the
    one there is (``Broadcast``), a column's rows under the unit's row
    (``RowsUnder``) or all of them (``EntireValue``) — and the pieces are merged on
    their rows' ids: the rows are the deepest sources', a parent's value
    copied down onto each of them, and a parent row nothing sits under is not
    in the table. A ``Table`` input never makes its node run per row by itself, and its
    sources never have to line up: rows that match nothing are blanks.
    """

    #: Per source, in wiring order: the source and how the unit receives it.
    receive_mode_by_source: tuple[tuple[Ref, PerRow | Broadcast | EntireValue | RowsUnder], ...]
    #: The sources whose rows sit above another source's: copied down onto those rows.
    copied_down: frozenset[Ref] = frozenset()


ReceiveMode = PerRow | Broadcast | EntireValue | RowsUnder | Gather | Merge


def group_depth_of_read(receive_mode: ReceiveMode) -> int:
    """The depth of the row a reader of many values is read under: a ``RowsUnder``'s, else ``0`` (everything on the field)."""
    return receive_mode.depth if isinstance(receive_mode, RowsUnder) else 0
