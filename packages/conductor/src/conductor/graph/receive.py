"""How an input receives its value: one row at a time, one value everywhere, the whole thing, a group under a row, or a gathering.

A node in a graph runs as units, one per row of the index it iterates on
(or one unit when it runs once). For every input, the question this
module answers is: *at a unit's row, which of the input's values does the
unit receive?* Compile answers it once, while it walks the edges, and
stores the answer on the field (``CompiledField.receives``). The engine's
ledger reads that record to decide when a unit is ready and what to hand
it, and never works the answer out again from the types and indexes.

Five records, one per input, named for what happens to the value on its
way in. ``Iterate`` splits a series into rows: a scalar input fed a
series, so the node runs once per row and each unit takes the value at its
own row. ``Broadcast`` copies one value to every unit: a scalar fed by a
node that ran once, a typed-in value, a default. ``Whole`` leaves the
value as it sits on the field: a ``Series[X]`` input fed one series on a
root index takes it entire, and a ``**inputs`` name takes whatever its
edge carries. ``Group`` cuts a series to the rows under the unit's own
row: a ``Series[X]`` input fed a series on a child index, reduced once
per parent row. ``Gather`` collects unrelated sources — several scalars,
several series on different indexes — onto a fresh index that belongs
to the input.

These say how a value is *received*, not where it comes from: the
binding (an edge from another node's output, or a value the author
typed) is a different record on the same field, and an output has no
receive record at all.
"""

from __future__ import annotations

from dataclasses import dataclass

from conductor.series import Index


@dataclass(frozen=True)
class Iterate:
    """The unit receives the value at its own row, projected onto ``index``.

    A scalar input fed a series on ``index``: the node runs once per row
    of it, and each unit reads the row it stands on. A typed-in list on a
    scalar input is the same thing on an index of the input's own, one
    row per value typed.
    """

    index: Index


@dataclass(frozen=True)
class Broadcast:
    """The unit receives the one value there is, the same at every unit.

    A scalar input fed by an edge from a node that ran once, a value the
    author typed, or a default. The node does not run per row because of
    this input; when it runs per row for another, this value is copied to
    every row.
    """


@dataclass(frozen=True)
class Whole:
    """The unit receives everything on the field, as it sits there.

    A ``**inputs`` parameter takes whatever its one edge carries, series
    and all; a ``Series[X]`` input fed one series on a root index takes
    that series entire; a ``Series[X]`` input with a typed-in list or a
    default takes it as a series on its own index. The node does not run
    per row because of this input.
    """


@dataclass(frozen=True)
class Group:
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


Receive = Iterate | Broadcast | Whole | Group | Gather
