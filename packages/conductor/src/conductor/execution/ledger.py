"""The ledger: what a run has produced, cell by cell, and what that makes ready.

A ledger is the engine's record of one run — every value every node has
produced so far — and the one place the engine asks what may run next.

The engine's unit of work is ``(node, row)``. A *series* is a value with
many rows, and an *index* names where those rows come from; a node that
receives a series on a scalar input runs once per row of that index. A
node that runs once is one unit with row ``None``; a node that runs per
row of index ``L`` is one unit per row of ``L``. A row is a path —
``(i,)`` on a root index, ``(i, j)`` for a child row created under parent
row ``(i,)`` — so a unit on a deeper index finds its row on an ancestor
by taking a prefix. A row is *born* when the unit producing the series
writes it; until then nothing knows how many rows there will be.

**A skip has a depth.** ``SKIPPED`` written at row ``k`` on a field means
the field holds nothing at ``k`` and at every row under it. A unit that
returns ``SKIPPED`` on an output writes it at its own row. A reader looks
for a skip at its own row and at every shorter prefix, and re-emits
``SKIPPED`` at the depth it found it: a skip at the reader's own depth
that one row is missing; a skip above it means nothing beneath it runs.
A unit whose every series output is skipped births no rows, so a node
running per row of that index runs once at the shorter row instead —
that is how a skip keeps its reach through a chain.

The ledger holds:

* a **cell** per ``(field, row)``: the value, or ``SKIPPED``;
* the **rows born** on each index — a root's by the one unit of the node
  that produced the series, a child's by each unit of its parent node;
* which indexes are **sealed**: every row they will ever have is born;
* under which rows an index has **no rows**, because something above them was skipped;
* which units are **done**, and which are **pending** on a person.

It answers ``units`` (a node's units right now), ``ready`` and
``inputs_for`` (may this unit run, and with what), ``record`` (this unit
produced this, and these units are ready now), ``runnable``, ``pend``,
``complete`` and ``progress``, ``results``, ``pending`` and ``cells``.
Nothing here is asynchronous and nothing here schedules; the engine is a
loop over these calls.

**Readiness is kept, not recomputed.** Asking every unit whether it is
ready after every unit finishes costs the square of the rows. So
``record`` reports the units its write made ready, and only those are
asked: the readers of a cell, at the cell's row and the rows under it;
the units a birth or a skip creates; a reduction whose group the write
completed. A group remembers how many of its rows are already written, so
no row is read twice. ``ready`` stays the definition, and ``runnable``
asks it of every unit: the engine does that when a leg starts, and again
when it goes quiet, where a ready unit that nobody started is a bug.

**A reduction groups by depth.** A ``Series[X]`` input fed a series on a
child index receives, per unit, the rows under the unit's own row on the
parent index — the node runs once per parent row and reduces the child
rows under it. Inside an embedded graph (a node whose version is itself a
graph; its inner nodes run as nodes of this run), a series that entered
through a scalar field is grouped by its own row — a group of one. One
rule covers both, the rows whose path starts with the unit's row at the
grouping depth, and the index a node runs once per row of
(``CompiledGraph.node(node_id).iterates_on``) says which depth that is.

**Cells are the record.** ``cells`` is everything a leg produced, row by
row, and ``restore`` starts the next leg from it. A leg is one call of
``execute``; a run takes several when a node waits on a person in
between. Nothing is pruned, so a unit done in one leg stays done in the
next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

from conductor._sentinel import SKIPPED, is_skipped
from conductor.errors import ErrorCause, NodeExecutionError
from conductor.execution.events import PendingUnit
from conductor.graph.binding import Edges, Static
from conductor.graph.compiled import CompiledGraph
from conductor.metadata import Input
from conductor.ref import Ref
from conductor.series import Index, Row, Series

#: One run of one node: the node id and the row it runs on — ``None`` for
#: a node that runs once; a row shorter than the node's index for the one
#: unit standing in for rows a skip above them never let be born.
Unit = tuple[str, Row | None]

_SKIPPED_WIRE = "__skipped__"
_MISSING = object()


@dataclass(frozen=True, slots=True)
class Skip:
    """This unit does not run: ``SKIPPED`` was found at row ``at`` on something it reads.

    ``at`` is the unit's own row, or a shorter prefix when the skip sits
    above it. The unit re-emits ``SKIPPED`` at ``at``, so the skip keeps its
    reach down the chain.
    Written and read inside the ledger only. Not the same thing as
    ``SKIPPED``, which is the value a node returns and carries no row.
    """

    at: Row | None


def _depth(row: Row | None) -> int:
    return 0 if row is None else len(row)


def _prefixes(key: Row | None) -> Iterator[Row | None]:
    """``None``, then every prefix of ``key`` from shortest to ``key`` itself."""
    yield None
    if key is not None:
        for n in range(1, len(key) + 1):
            yield key[:n]


def _group(row: Row | None, depth: int | None) -> Row | None:
    """The group a unit at ``row`` reads a series by: its row cut to ``depth``,
    or ``None`` — the whole series — when there is no depth."""
    return None if depth is None else row[:depth]


def _order(unit: Unit) -> Row:
    """A unit's row as a sort key, with ``None`` first."""
    return () if unit[1] is None else unit[1]


class Ledger:
    """The record of one run over one compiled graph, and what it makes ready.

    Built empty by ``execute`` for a first leg, or from ``cells()`` of an
    earlier leg by ``restore``. The engine is the only caller.
    """

    def __init__(self, compiled: CompiledGraph) -> None:
        self._compiled = compiled
        self._cells: dict[Ref, dict[Row | None, Any]] = {}
        self._rows: dict[str, set[Row]] = {}
        #: Per index, its rows under each shorter row, in the order they were
        #: born; ``None`` holds them all. Finds the rows under a row without
        #: reading the whole index.
        self._rows_by_prefix: dict[str, dict[Row | None, list[Row]]] = {}
        self._sealed: set[str] = set()
        #: Per index, the rows beneath which it has no rows at all, because
        #: something above them was skipped.
        self._no_rows_under: dict[str, set[Row | None]] = {}
        self._done: set[Unit] = set()
        #: Per node that runs once per row, how many of its units are done:
        #: at a row of its index, and standing in at a shorter row.
        self._done_rows: dict[str, int] = {}
        self._done_standing_in: dict[str, int] = {}
        #: Units waiting on a person: their prompt and questions, named by address.
        self._pending: dict[Unit, tuple[str | None, tuple[Input, ...]]] = {}
        #: Per group a series is read by — a field and the row it is read
        #: under — its rows in order and how many from the first are written.
        #: Kept once every row of the group is born, so no row is read twice.
        self._written_rows: dict[tuple[Ref, Row | None], tuple[list[Row], int]] = {}
        #: Each field's index as compile stored it, looked up once.
        self._indexes: dict[Ref, Index | None] = {}
        #: Nodes that birth rows on an index: those with a series output.
        self._births = frozenset(
            node_id
            for node_id in compiled.execution_order()
            if any(o.dtype.element is not None for o in compiled.node(node_id).interface.outputs)
        )
        self._position = {node_id: n for n, node_id in enumerate(compiled.execution_order())}

        # Who reads what, from the edges, so a write asks only its readers.
        #: Per node, its connected inputs: the edges, how the input reads
        #: them, and for a reduction the depth it groups at.
        self._connected: dict[str, tuple[tuple[Edges, str, int | None], ...]] = {}
        #: Per output, the nodes reading it one row at a time.
        self._scalar_readers: dict[Ref, list[str]] = {}
        #: Per output, the nodes reading it as a series, and the depth they group at.
        self._series_readers: dict[Ref, list[tuple[str, int | None]]] = {}
        #: The same readers by the index of the output they read.
        self._series_readers_on: dict[str, list[tuple[Ref, str, int | None]]] = {}
        #: Per index, the nodes that run once per row of it.
        self._iterating_on: dict[str, list[str]] = {}
        #: Per index, the nodes running once per row of it that birth rows of their own.
        self._births_on: dict[str, list[str]] = {}
        typed: list[str] = []
        for node_id in compiled.execution_order():
            node = compiled.node(node_id)
            connected: list[tuple[Edges, str, int | None]] = []
            for inp in node.interface.inputs:
                own = Ref(node_id, inp.name)
                binding = compiled.field(own).binding
                index = compiled.field(own).index
                # A scalar input where the author typed many values holds a
                # series on an index of the input's own, and the node runs
                # once per value. Compile stored that index on the field
                # (``field(ref).index``). Its rows are known before anything
                # runs, so they are born and sealed here.
                if getattr(inp.dtype, "element", None) is None and index is not None and isinstance(binding, Static):
                    for n in range(len(node.statics[inp.name])):
                        self._born(index.id, (n,))
                    self._sealed.add(index.id)
                    typed.append(index.id)
                if not isinstance(binding, Edges):
                    continue
                reading = self._reading(inp, binding)
                depth = self._group_depth(node_id, self._index(binding.refs[0])) if reading == "reduction" else None
                connected.append((binding, reading, depth))
                for ref in binding.refs:
                    if reading == "scalar":
                        self._scalar_readers.setdefault(ref, []).append(node_id)
                        continue
                    self._series_readers.setdefault(ref, []).append((node_id, depth))
                    if self._index(ref) is not None:
                        self._series_readers_on.setdefault(self._index(ref).id, []).append((ref, node_id, depth))
            self._connected[node_id] = tuple(connected)
            if node.iterates_on is not None:
                self._iterating_on.setdefault(node.iterates_on.id, []).append(node_id)
                if node_id in self._births:
                    self._births_on.setdefault(node.iterates_on.id, []).append(node_id)
        self._seal([node_id for index_id in typed for node_id in self._births_on.get(index_id, ())])

    # -- units ---------------------------------------------------------------

    def units(self, node_id: str) -> list[Unit]:
        """This node's units right now: one, or — running per row of ``L`` —
        one per row born on ``L``, plus one at each shorter row where ``L`` has no rows."""
        iterate = self._compiled.node(node_id).iterates_on
        if iterate is None:
            return [(node_id, None)]
        barren = sorted(self._no_rows_under.get(iterate.id, ()), key=lambda r: () if r is None else r)
        rows = sorted(self._rows.get(iterate.id, ()))
        return [(node_id, row) for row in (*barren, *rows)]

    def is_done(self, unit: Unit) -> bool:
        return unit in self._done

    def complete(self, node_id: str) -> bool:
        """Every unit this node will ever have is done."""
        iterate = self._compiled.node(node_id).iterates_on
        if iterate is None:
            return (node_id, None) in self._done
        return (
            iterate.id in self._sealed
            and self._done_rows.get(node_id, 0) == len(self._rows.get(iterate.id, ()))
            and self._done_standing_in.get(node_id, 0) == len(self._no_rows_under.get(iterate.id, ()))
        )

    def progress(self, node_id: str) -> tuple[int, int | None]:
        """``(done, total)`` rows for a node; ``total`` is ``None`` until the
        node's index is sealed. Cover units stand in for rows that were never
        born, so they count in neither number."""
        iterate = self._compiled.node(node_id).iterates_on
        if iterate is None:
            return int((node_id, None) in self._done), 1
        total = len(self._rows.get(iterate.id, ())) if iterate.id in self._sealed else None
        return self._done_rows.get(node_id, 0), total

    def _units_under(self, node_id: str, row: Row | None) -> list[Unit]:
        """This node's units at ``row`` or beneath it, in no particular order —
        not counting a unit standing in at a shorter row, which is ready
        the moment it exists."""
        iterate = self._compiled.node(node_id).iterates_on
        if iterate is None:
            return [(node_id, None)] if row is None else []
        if row in self._rows.get(iterate.id, ()):
            return [(node_id, row)]
        return [(node_id, r) for r in self._rows_by_prefix.get(iterate.id, {}).get(row, ())]

    # -- readiness -----------------------------------------------------------

    def ready(self, unit: Unit) -> bool:
        """Every value this unit needs exists, or its producer is done."""
        node_id, row = unit
        iterate = self._compiled.node(node_id).iterates_on
        if iterate is not None and _depth(row) < iterate.depth:
            return True  # standing in at a shorter row: the skip above it is already known
        return all(self._input_ready(binding, reading, depth, row) for binding, reading, depth in self._connected[node_id])

    def runnable(self) -> list[Unit]:
        """Every unit that may start now: ready, and neither done nor waiting
        on a person. Asks ``ready`` of every unit, so the engine calls it
        where a leg starts and where it goes quiet, never per write."""
        return [
            unit
            for node_id in self._compiled.execution_order()
            if not self.complete(node_id)
            for unit in self.units(node_id)
            if unit not in self._done and unit not in self._pending and self.ready(unit)
        ]

    def _input_ready(self, binding: Edges, reading: str, depth: int | None, row: Row | None) -> bool:
        if reading == "scalar":
            return all(self._present(ref, self._key(ref, row)) for ref in binding.refs)
        if reading == "gather":
            return all(self._written(ref, None) for ref in binding.refs)
        group = _group(row, depth)
        return all(self._written(ref, group) for ref in binding.refs)

    def _group_depth(self, node_id: str, index: Index) -> int | None:
        """The depth a ``Series[X]`` input on ``index`` groups at, for a unit
        of ``node_id``: ``None`` (the whole series, once) for a root index;
        the parent's depth for a child index; the index's own depth when the
        node sits inside an embedded graph that this index entered through a
        scalar field — then each unit's group is its own row."""
        placement = self._compiled.node(node_id).embedded_in
        scope = None if placement is None else self._compiled.node(placement).iterates_on
        if scope is not None and index == scope:
            return index.depth
        return None if index.parent is None else index.parent.depth

    def _present(self, ref: Ref, key: Row | None) -> bool:
        return self._lookup(ref, key)[0] is not _MISSING

    def _index(self, ref: Ref) -> Index | None:
        """The index compile stored on ``ref``, asked of compile once."""
        if ref not in self._indexes:
            self._indexes[ref] = self._compiled.field(ref).index
        return self._indexes[ref]

    def _key(self, ref: Ref, row: Row | None) -> Row | None:
        """Where a unit at ``row`` reads ``ref`` as a scalar: its row on the ref's index, or ``None``."""
        index = self._index(ref)
        return None if index is None else row[: index.depth]

    def _written(self, ref: Ref, group: Row | None) -> bool:
        """Is every cell of ``ref`` a series reader at ``group`` receives written?
        The one cell of a field that carries one value, else the rows under ``group``."""
        index = self._index(ref)
        return self._present(ref, None) if index is None else self._all_written(ref, index, group)

    def _all_written(self, ref: Ref, index: Index, parent_row: Row | None) -> bool:
        """Is every cell of ``ref`` under ``parent_row`` written? True when the
        field is skipped there, or every row under it is born and produced."""
        if is_skipped(self._lookup(ref, parent_row)[0]):
            return True
        barren = self._no_rows_under.get(index.id, ())
        if any(prefix in barren for prefix in _prefixes(parent_row)):
            return False  # the unit standing in at that shorter row has not run yet
        if parent_row is None:
            born = index.id in self._sealed
        elif len(parent_row) == index.depth:
            born = parent_row in self._rows.get(index.id, ())  # the group is the row itself, so it exists once that row is born
        else:
            born = (index.id, parent_row) in self._done
        if not born:
            return False
        # A born group's rows are final, so how many are written from the
        # first is kept, and a row once found written is not read again.
        group = (ref, parent_row)
        if group not in self._written_rows:
            self._written_rows[group] = (self._rows_under(index.id, parent_row), 0)
        rows, written = self._written_rows[group]
        while written < len(rows) and self._present(ref, rows[written]):
            written += 1
        self._written_rows[group] = (rows, written)
        return written == len(rows)

    def _reading(self, inp: Input, binding: Edges) -> str:
        """How a connected input reads its sources: ``"scalar"`` (per row — one
        ref, or several on one index of which one covers each row),
        ``"reduction"`` (one series, received whole on a root index and once
        per parent row on a child one — ``_group_depth`` says which) or
        ``"gather"`` (unrelated sources collected into a fresh series)."""
        if getattr(inp.dtype, "element", None) is None:
            return "scalar"
        indexes = {self._index(ref) for ref in binding.refs}
        if None not in indexes and len(indexes) == 1:
            return "reduction"
        return "gather"

    # -- what a unit receives -------------------------------------------------

    def inputs_for(self, unit: Unit) -> dict[str, Any] | Skip:
        """The keyword arguments this unit runs with, or the ``Skip`` that stops it.

        A ``Skip`` comes from ``SKIPPED`` on a connected scalar input, on the
        series a reduction receives, or at a row above this
        unit's. A gather drops skipped sources instead — an empty gather
        is an empty series — because gathering is how "whichever branch
        fired" is expressed.
        """
        node_id, row = unit
        iterate = self._compiled.node(node_id).iterates_on
        if iterate is not None and _depth(row) < iterate.depth:
            return Skip(at=row)
        values: dict[str, Any] = {}
        for inp in self._compiled.node(node_id).interface.inputs:
            own = Ref(node_id, inp.name)
            binding = self._compiled.field(own).binding
            if binding is None:
                if getattr(inp.dtype, "element", None) is not None and inp.optional:
                    values[inp.name] = self._typed(inp, inp.default, own)
                continue
            if isinstance(binding, Static):
                values[inp.name] = self._typed(inp, self._compiled.node(node_id).statics[inp.name], own, row)
                continue
            reading = self._reading(inp, binding)
            if reading == "scalar":
                covering, skips = self._covering(node_id, binding.refs, row)
                if not covering:
                    return Skip(at=max(skips, key=_depth))
                (values[inp.name],) = covering
            elif reading == "reduction":
                index = self._compiled.field(binding.refs[0]).index
                depth = self._group_depth(node_id, index)
                parent_row = None if depth is None else row[:depth]
                found = [self._lookup(ref, parent_row) for ref in binding.refs]
                if all(is_skipped(value) for value, _ in found):
                    return Skip(at=max((at for _, at in found), key=_depth))
                values[inp.name] = self._series(node_id, binding.refs, index, self._rows_under(index.id, parent_row))
            else:
                gathered: list[Any] = []
                for ref in binding.refs:
                    index = self._compiled.field(ref).index
                    value, _ = self._lookup(ref, None)
                    if is_skipped(value):
                        continue
                    if index is None:
                        gathered.append(value)
                    else:
                        gathered.extend(self._series(node_id, (ref,), index, self._rows_under(index.id, None)).values)
                values[inp.name] = Series(self._compiled.field(own).index, gathered)
        return values

    def _covering(self, node_id: str, refs: tuple[Ref, ...], row: Row | None) -> tuple[list[Any], list[Row | None]]:
        """Of several refs read at one row: the values that cover it, and the
        rows at which the others were skipped. Two covering the same row
        means two edges both supplied that row, and the node fails with
        that row."""
        covering: list[Any] = []
        skips: list[Row | None] = []
        for ref in refs:
            value, at = self._lookup(ref, self._key(ref, row))
            if value is _MISSING:
                raise KeyError(f"{ref} has no value at {at} — the unit was not ready")
            if is_skipped(value):
                skips.append(at)
            else:
                covering.append(value)
        if len(covering) > 1:
            raise NodeExecutionError(
                f"'{node_id}' received the same row from {len(covering)} edges",
                node_id=node_id,
                cause=ErrorCause(
                    code="row_covered_twice",
                    message="The row comes from more than one edge; a merge may cover each row only once.",
                    details={"edges": len(covering)},
                    row=row,
                ),
            )
        return covering, skips

    def _typed(self, inp: Input, value: Any, own: Ref, row: Row | None = None) -> Any:
        """A value the author typed (or the declared default) as the unit receives it.

        A sequence on a ``Series[X]`` input becomes a series on the input's
        own index; a sequence on a scalar input is read at this unit's row
        on that index. Which index, and whether the scalar input holds a
        sequence at all, is what compile stored on the field (``field(ref).index``).
        """
        index = self._compiled.field(own).index
        if getattr(inp.dtype, "element", None) is None:
            return value if index is None else list(value)[row[index.depth - 1]]
        values = value.values if isinstance(value, Series) else list(value)
        return Series(index, values)

    def _lookup(self, ref: Ref, key: Row | None) -> tuple[Any, Row | None]:
        """``(value, row)`` for the cell at ``(ref, key)``: ``SKIPPED`` at the
        shortest prefix of ``key`` that holds one, else what is at ``key``
        (``_MISSING`` when nothing is)."""
        by_row = self._cells.get(ref, {})
        for prefix in _prefixes(key):
            if is_skipped(by_row.get(prefix)):
                return SKIPPED, prefix
        return by_row.get(key, _MISSING), key

    def _rows_under(self, index_id: str, parent_row: Row | None) -> list[Row]:
        """The rows of an index whose path starts with ``parent_row``: every
        row for ``None``, a child's rows under a parent row, or the row
        itself when ``parent_row`` is a row of this very index."""
        if parent_row in self._rows.get(index_id, ()):
            return [parent_row]
        return sorted(self._rows_by_prefix.get(index_id, {}).get(parent_row, ()))

    def _series(self, node_id: str, refs: tuple[Ref, ...], index: Index, rows: list[Row]) -> Series[Any]:
        """A series of the values on ``rows`` — from one ref, or from whichever
        of several covers each row — sparse where none does."""
        present: list[tuple[Row, Any]] = []
        for r in rows:
            covering, _ = self._covering(node_id, refs, r)
            if covering:
                present.append((r, covering[0]))
        return Series(index, [v for _, v in present], rows=[r for r, _ in present])

    # -- recording -----------------------------------------------------------------

    def record(self, unit: Unit, outputs: dict[str, Any] | Skip) -> list[Unit]:
        """Record that this unit produced ``outputs``, or a ``Skip`` because it
        did not run, and return the units that are ready because of it.

        A scalar output is one cell at the unit's row; so is ``SKIPPED`` on
        any output. A series output births rows under the unit's row —
        ``(j,)`` for a node that ran once, ``row + (j,)`` for one running
        per row — and every series output of one unit must birth the same
        number of rows, since they are columns of one table. A unit that
        skips every series output leaves its index with no rows under that row.

        A node returns plain sequences for series outputs, never an index:
        the node does not know where it was placed, so the ledger, not the
        node, says which index a series lives on.

        The units returned are the ones this write could have made ready and
        that answer ``ready`` now, in execution order; the engine starts
        those and asks nothing else.
        """
        node_id, row = unit
        interface = self._compiled.node(node_id).interface
        births = node_id in self._births
        written: list[tuple[Ref, Row | None]] = []
        born: list[Row] = []
        barren: list[Row | None] = []
        if births:
            self._rows.setdefault(node_id, set())
        if isinstance(outputs, Skip):
            for out in interface.outputs:
                self._write(Ref(node_id, out.name), outputs.at, SKIPPED, written)
            if births:
                barren.append(outputs.at)
        else:
            length: int | None = None
            for out in interface.outputs:
                ref = Ref(node_id, out.name)
                value = outputs[out.name]
                if out.dtype.element is None or is_skipped(value):
                    self._write(ref, row, value, written)
                    continue
                values = list(value)
                if length is None:
                    length = len(values)
                elif length != len(values):
                    raise ValueError(
                        f"{node_id}: its series outputs differ in length ({length} and {len(values)})"
                    )
                for j, item in enumerate(values):
                    key = (j,) if row is None else (*row, j)
                    if self._born(node_id, key):
                        born.append(key)
                    self._write(ref, key, item, written)
            if births and length is None:
                barren.append(row)
        for at in barren:
            self._no_rows_under.setdefault(node_id, set()).add(at)
        self._finish(unit)
        sealed = self._seal([node_id] if births else [])
        return self._woken(unit, written, born, barren, sealed)

    def _write(self, ref: Ref, key: Row | None, value: Any, written: list[tuple[Ref, Row | None]]) -> None:
        """Put ``value`` in the cell, noting the cell in ``written`` when it was empty."""
        by_row = self._cells.setdefault(ref, {})
        if key not in by_row:
            written.append((ref, key))
        by_row[key] = value

    def _born(self, index_id: str, row: Row) -> bool:
        """Birth ``row`` on the index; ``False`` when it already was."""
        rows = self._rows.setdefault(index_id, set())
        if row in rows:
            return False
        rows.add(row)
        by_prefix = self._rows_by_prefix.setdefault(index_id, {})
        for prefix in _prefixes(row[:-1]):
            by_prefix.setdefault(prefix, []).append(row)
        return True

    def _finish(self, unit: Unit) -> None:
        """Mark the unit done, and count it toward its node's rows."""
        if unit in self._done:
            return
        self._done.add(unit)
        node_id, row = unit
        iterate = self._compiled.node(node_id).iterates_on
        if iterate is not None:
            counts = self._done_rows if _depth(row) == iterate.depth else self._done_standing_in
            counts[node_id] = counts.get(node_id, 0) + 1

    def _seal(self, births: list[str]) -> list[str]:
        """Seal the index of each of these nodes whose rows are all born, and
        of every node that iterates on an index sealed that way, since a
        child index can only be sealed once its parent is. Returns the
        indexes sealed."""
        sealed: list[str] = []
        while births:
            node_id = births.pop()
            if node_id in self._sealed or not self.complete(node_id):
                continue
            self._sealed.add(node_id)
            sealed.append(node_id)
            births.extend(self._births_on.get(node_id, ()))
        return sealed

    def _woken(
        self,
        unit: Unit,
        written: list[tuple[Ref, Row | None]],
        born: list[Row],
        barren: list[Row | None],
        sealed: list[str],
    ) -> list[Unit]:
        """The units ``unit``'s record could have made ready, that are ready.

        A unit's readiness reads cells of its inputs, the rows born on the
        indexes those cells and the unit itself sit on, and which of those
        indexes are sealed. So the candidates are: a scalar reader of a
        written cell, at its row and under it; a series reader whose group
        the write completed, or everything under a skip above that group;
        the units a birth or an empty row creates; a series reader of this
        node's index whose group is born now that this unit is done or the
        index is sealed.
        """
        node_id, row = unit
        woken: set[Unit] = set()
        for ref, key in written:
            for reader in self._scalar_readers.get(ref, ()):
                woken.update(self._units_under(reader, key))
            for reader, depth in self._series_readers.get(ref, ()):
                if depth is not None and _depth(key) < depth:
                    woken.update(self._units_under(reader, key))
                else:
                    self._wake_group(ref, reader, _group(key, depth), woken)
        for key in born:
            woken.update((reader, key) for reader in self._iterating_on.get(node_id, ()))
        for at in barren:
            woken.update((reader, at) for reader in self._iterating_on.get(node_id, ()))
        for ref, reader, depth in self._series_readers_on.get(node_id, ()):
            for key in born:
                if depth == len(key):
                    self._wake_group(ref, reader, key, woken)
            if row is not None and depth == len(row):
                self._wake_group(ref, reader, row, woken)
        for index_id in sealed:
            for ref, reader, depth in self._series_readers_on.get(index_id, ()):
                if depth is None:
                    self._wake_group(ref, reader, None, woken)
        ready = [u for u in woken if u not in self._done and u not in self._pending and self.ready(u)]
        return sorted(ready, key=lambda u: (self._position[u[0]], _order(u)))

    def _wake_group(self, ref: Ref, reader: str, group: Row | None, woken: set[Unit]) -> None:
        """Add ``reader``'s units in ``group`` when every cell of ``ref`` they read is written."""
        if self._written(ref, group):
            woken.update(self._units_under(reader, group))

    def inject(self, node_id: str, outputs: dict[str, Any]) -> None:
        """Record ``outputs`` as this node's complete result without running it —
        a person's answers, or an earlier run's results. For a node running
        per row the outputs are series on its index and are recorded row by row."""
        iterate = self._compiled.node(node_id).iterates_on
        if iterate is None:
            self.record((node_id, None), outputs)
            return
        for _, row in self.units(node_id):
            if _depth(row) < iterate.depth:
                self.record((node_id, row), Skip(at=row))
                continue
            at_row: dict[str, Any] = {}
            for out in self._compiled.node(node_id).interface.outputs:
                series = outputs[out.name]
                by_row = dict(zip(series.rows, series.values, strict=True))
                at_row[out.name] = by_row.get(row, SKIPPED)
            self.record((node_id, row), Skip(at=row) if all(is_skipped(v) for v in at_row.values()) else at_row)

    # -- waiting on a person ---------------------------------------------

    def pend(self, unit: Unit, questions: tuple[Input, ...], prompt: str | None = None) -> None:
        """Park this unit: its node returned ``Asks``. It and everything that
        reads it wait; the questions are re-keyed by address (``node.field``)."""
        node_id, _ = unit
        self._pending[unit] = (prompt, tuple(q.model_copy(update={"name": Ref(node_id, q.name)}) for q in questions))

    def is_pending(self, unit: Unit) -> bool:
        return unit in self._pending

    def pending(self) -> list[PendingUnit]:
        """Every unit waiting on a person, as ``PendingUnit`` records."""
        return [
            PendingUnit(node_id=node_id, row=None if row is None else list(row), prompt=prompt, questions=questions)
            for (node_id, row), (prompt, questions) in self._pending.items()
        ]

    # -- what the run produced -------------------------------------------------------

    def result_of(self, node_id: str) -> dict[str, Any] | None:
        """This node's outputs by name, or ``None`` when the node did not run
        (every output is ``SKIPPED`` at the top).

        A scalar output of a node that ran once is its value; anything on an
        index is a ``Series``, sparse where rows were skipped — a node whose
        every row was skipped away gives an empty series.
        """
        outputs = self._compiled.node(node_id).interface.outputs
        if all(is_skipped(self._cells.get(Ref(node_id, out.name), {}).get(None)) for out in outputs):
            return None
        result: dict[str, Any] = {}
        for out in outputs:
            ref = Ref(node_id, out.name)
            index = self._compiled.field(ref).index
            value, _ = self._lookup(ref, None)
            if index is None or is_skipped(value):
                result[out.name] = value
            else:
                result[out.name] = self._series(node_id, (ref,), index, self._rows_under(index.id, None))
        return result

    def results(self) -> dict[str, dict[str, Any]]:
        """Every complete node's outputs, by node id. A node that did not run is absent."""
        produced: dict[str, dict[str, Any]] = {}
        for node_id in self._compiled.execution_order():
            if not self.complete(node_id):
                continue
            result = self.result_of(node_id)
            if result is not None:
                produced[node_id] = result
        return produced

    # -- the record ---------------------------------------------------------------

    def cells(self) -> dict[str, Any]:
        """Everything this leg has produced, as JSON-ready data.

        A host stores it as the run's record and hands it back to
        ``execute(cells=...)`` for the next leg, or for a new run seeded from
        this one. A waiting unit is not in it: it produced nothing and asks
        again next leg unless answered through ``cache``.
        """
        return {
            "cells": [
                {"ref": [ref.node_id, ref.field], "row": None if row is None else list(row),
                 "value": _SKIPPED_WIRE if is_skipped(value) else value}
                for ref, by_row in self._cells.items()
                for row, value in by_row.items()
            ],
            "rows": {index_id: [list(r) for r in sorted(rows)] for index_id, rows in self._rows.items()},
            "sealed": sorted(self._sealed),
            "no_rows_under": {
                index_id: [None if r is None else list(r) for r in sorted(rows, key=lambda r: () if r is None else r)]
                for index_id, rows in self._no_rows_under.items()
            },
            "done": [[node_id, None if row is None else list(row)] for node_id, row in self._done],
        }

    @classmethod
    def restore(cls, compiled: CompiledGraph, data: dict[str, Any]) -> Ledger:
        """A ledger holding what ``cells()`` of an earlier leg recorded, over the same compiled graph."""
        ledger = cls(compiled)
        for cell in data["cells"]:
            ref = Ref(*cell["ref"])
            row = None if cell["row"] is None else tuple(cell["row"])
            value = SKIPPED if cell["value"] == _SKIPPED_WIRE else cell["value"]
            ledger._cells.setdefault(ref, {})[row] = value
        ledger._rows, ledger._rows_by_prefix = {}, {}
        for index_id, rows in data["rows"].items():
            ledger._rows[index_id] = set()
            for r in rows:
                ledger._born(index_id, tuple(r))
        ledger._sealed = set(data["sealed"])
        ledger._no_rows_under = {
            index_id: {None if r is None else tuple(r) for r in rows} for index_id, rows in data["no_rows_under"].items()
        }
        for node_id, row in data["done"]:
            ledger._finish((node_id, None if row is None else tuple(row)))
        return ledger
