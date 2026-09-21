"""How an input receives its value: one value per row, the whole series, a group under a row, or a gathering.

A node in a graph runs as units, one per row of the index it iterates on
(or one unit when it runs once). For every input, the question this
module answers is: *at a unit's row, which of the input's cells does the
unit receive?* Compile answers it once, while it walks the edges, and
stores the answer on the field (``CompiledField.receives``). The engine's
ledger reads that record to decide when a unit is ready and what to hand
it, and never works the answer out again from the types and indexes.

Four records, one per input. ``PerRow`` is the ordinary case: a scalar
input takes the cell at the unit's own row, projected onto the index the
value sits on — or the one cell there is, when the value is a scalar.
``Whole`` is the entire series (or the one value), the same at every
unit. ``Reduce`` is a ``Series[X]`` input fed a series on a child index:
the unit receives the child rows under its own row, cut to ``depth``.
``Gather`` is a ``Series[X]`` input whose sources are unrelated — several
scalars, several series on different indexes — collected onto a fresh
index that belongs to the input.

These say how a value is *received*, not where it comes from: the
binding (an edge from another node's output, or a value the author
typed) is a different record on the same field, and an output has no
receive record at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from conductor.series import Index


@dataclass(frozen=True)
class PerRow:
    """The unit receives the cell at its own row, projected onto ``index``.

    A scalar input fed a series on ``index`` — the node runs once per row
    of it, and each unit reads the row it stands on. ``index`` is ``None``
    when the value is one cell for every unit: a scalar fed by an edge from
    a node that ran once, a value the author typed, a default. A typed-in
    list on a scalar input is ``PerRow`` on an index of the input's own,
    one row per value typed.
    """

    index: Index | None


@dataclass(frozen=True)
class Whole:
    """The unit receives everything on the field, the same at every unit.

    A ``**inputs`` parameter takes whatever its one edge carries, series
    and all; a ``Series[X]`` input fed one series on a root index takes
    that series entire; a ``Series[X]`` input with a typed-in list or a
    default takes it as a series on its own index. The node does not run
    per row because of this input.
    """


@dataclass(frozen=True)
class Reduce:
    """The unit receives the rows of ``index`` under its own row cut to ``depth``.

    A ``Series[X]`` input fed a series on a child index: the node runs once
    per parent row and reduces the child rows under it, so ``depth`` is the
    parent's. Inside an embedded graph that runs once per outer row, a
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


Receive = PerRow | Whole | Reduce | Gather
