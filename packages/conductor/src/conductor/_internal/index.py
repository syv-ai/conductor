"""``Index`` and ``Row``: where the many values of a field sit.

Compile names an index for every field that holds many values, and the run
keeps which rows each one has. A value never carries either: a node receives a
plain list. Two fields may feed one node row by row only if they share an index.
A child index, ``Index("lines", parent=Index("docs"))``, is made where a node
makes rows under a row. Each row on it is a path, ``(i, j)`` for the j-th row
made under parent row ``(i,)``, so a row knows its parent by construction.
"""

from __future__ import annotations

from typing import Any, TypedDict

from conductor._internal.conductor_model import ConductorModel

#: A row's position on its index, as a path: ``(i,)`` on a root index,
#: ``(i, j)`` for the j-th row made under parent row ``(i,)``.
Row = tuple[int, ...]


class ValuesOnRows(TypedDict):
    """What a run reports for an output holding many values: ``rows[i]`` is where ``values[i]`` sits.

    The rows of a row skipped are left out. JSON writes a row as a list of ints."""

    rows: list[Row]
    values: list[Any]


class Index(ConductorModel):
    """Where a field's rows come from — the identity two fields must share to align.

    An index is an ``id`` and, for a child index, the ``parent`` it was
    opened from. Nothing else: no length, no row table. Equality and hashing
    use the ``id`` alone, so an index can be named at compile time, before
    any row exists, and rows are added to it later::

        docs = Index("docs")                     # a root
        lines = Index("lines", parent=docs)      # one level down
        lines.depth                              # 2
    """

    id: str
    #: The index this one was made under, or ``None`` for a root.
    parent: Index | None = None

    def __init__(self, id: str, parent: Index | None = None) -> None:
        """Positional, as an index reads: ``Index("lines", parent=docs)``."""
        super().__init__(id=id, parent=parent)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Index) and other.id == self.id

    def __repr__(self) -> str:
        """``Index('lines', parent=Index('docs'))``: the call that makes it."""
        parent = "" if self.parent is None else f", parent={self.parent!r}"
        return f"Index({self.id!r}{parent})"

    def __hash__(self) -> int:
        return hash(self.id)

    @property
    def depth(self) -> int:
        """The length of a row path on this index: 1 on a root, 2 on its child."""
        return 1 if self.parent is None else self.parent.depth + 1

    @property
    def lineage(self) -> tuple[Index, ...]:
        """This index and every index it was opened from, root first:
        ``(docs, lines)`` for ``lines``. Two indexes align only when one is
        in the other's lineage."""
        return (self,) if self.parent is None else (*self.parent.lineage, self)
