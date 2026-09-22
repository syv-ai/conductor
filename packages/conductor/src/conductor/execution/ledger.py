"""The ledger: what a run has produced so far, and what that lets run next.

The engine runs a graph as units. A unit is one node on one row. A node
that runs once is a single unit with row ``None``; a node fed a series on
a scalar input runs once per row of that series' index, one unit per row.
A row is a path, ``(i,)`` on a root index and ``(i, j)`` for the j-th row
born under ``(i,)``, so a unit deep in the tree finds its row on an
ancestor index by taking a prefix. Rows are born when the unit producing
the series writes it; until then nobody knows how many there will be.

The ledger keeps everything a run has written, one cell per field and
row, and answers the engine's two questions: is this unit ready, and
what does it run with. The engine is a loop over those calls. Nothing in
here is asynchronous and nothing in here schedules.

Readiness is kept rather than recomputed. Checking every unit after every
write would cost the square of the rows, so ``record`` returns the units
its write could have made ready, and only those are checked: the readers
of the written cell at its row and below, the units a new row creates,
and a reduction whose group is now complete. ``ready`` stays the
definition, and ``runnable`` asks it of every unit when a leg starts and
when it goes quiet, where a ready unit nobody started is a bug.

How a unit receives each input is compile's decision, read here. Every
input carries a receive record (``CompiledField.receives``, from
``conductor.graph.receive``). ``Iterate`` takes the cell at the unit's own
row, ``Broadcast`` the one cell there is, ``Whole`` everything on the
field, ``Group`` the rows under the unit's row cut to a depth, and
``Gather`` unrelated sources collected onto the input's own index. The
ledger switches on the record to say when a unit is ready and what to
hand it, and derives none of this from types or indexes itself.

A skip has a depth. ``SKIPPED`` written at row ``k`` means the field
holds nothing at ``k`` and at every row under it. A unit that returns
``SKIPPED`` writes it at its own row. A reader looks for a skip at its
own row and at every shorter prefix, and re-emits it at the depth it
found it. A unit whose every series output is skipped births no rows, so
a node that would run per row of that index runs once at the shorter row
instead. That is how a skip keeps its reach down a chain.

The cells are the record. ``cells`` is everything a leg produced, and
``restore`` starts the next leg from it. A leg is one call of
``execute``; a run takes several when a node waits on a person. Nothing
is pruned, so a unit done in one leg stays done in the next. Every value
crosses through the codec (``conductor.codec``) by the type compile gave
its field, and a skipped cell is marked ``{"skipped": <depth>}`` beside
its address, since a skip is not a value and has no type.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Iterator

from conductor._sentinel import SKIPPED, is_skipped
from conductor.codec import from_wire, to_wire
from conductor.errors import ErrorCause, NodeExecutionError, StartRefused
from conductor.execution.events import PendingUnit
from conductor.execution.record import RunRecord
from conductor.graph.binding import From, Static
from conductor.graph.compiled import CompiledGraph
from conductor.graph.receive import Broadcast, Gather, Group, Iterate, Receive, Whole
from conductor.metadata import Input
from conductor.ref import Ref
from conductor.series import Index, Row, Series

#: One run of one node: the node id and the row it runs on — ``None`` for
#: a node that runs once; a row shorter than the node's index for the one
#: unit standing in for rows a skip above them never let be born.
Unit = tuple[str, Row | None]

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


@dataclass(frozen=True, slots=True)
class _Reader:
    """One node reading one output of another node, and how it receives it.

    Filed when the ledger is built, three ways: under the reading node as
    what it reads, so ``ready`` asks each; under the output it reads and,
    for a series, under the index that series sits on, so a write wakes
    only its own readers. A restore reads the same records to drop the
    readers of a node that changed. A ``per_row`` reader is ready and
    wakes at the written cell's row; any other when the group it reads is
    complete, ``depth`` being the group's, or ``None`` for a reader that
    takes everything on the field. Both are decided once here, so neither
    the ready path nor the wake path asks the record's type again.
    """

    node_id: str
    ref: Ref
    received: Receive
    per_row: bool = field(init=False)
    depth: int | None = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "per_row", isinstance(self.received, (Iterate, Broadcast)))
        object.__setattr__(self, "depth", self.received.depth if isinstance(self.received, Group) else None)


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

        # What the graph says, read off compile once: who reads which output
        # and how, which nodes birth rows on which index, and where the
        # author typed a list into a scalar input.
        order = compiled.execution_order
        self._position = {node_id: n for n, node_id in enumerate(order)}
        #: Nodes that birth rows on an index: those with a series output.
        self._births = frozenset(
            node_id
            for node_id in order
            if any(o.dtype.element is not None for o in compiled.node(node_id).interface.outputs)
        )
        #: Per node, what it reads: one ``_Reader`` per edge into it.
        self._connected: dict[str, tuple[_Reader, ...]] = {}
        #: The same readers per output read, and, for a series, per index it
        #: sits on, for the writes that birth or seal rows there.
        self._readers: dict[Ref, list[_Reader]] = {}
        self._readers_on: dict[str, list[_Reader]] = {}
        #: Per index, the nodes that run once per row of it, and those among
        #: them that birth rows of their own.
        self._iterating_on: dict[str, list[str]] = {}
        self._births_on: dict[str, list[str]] = {}
        #: Typed-in lists. Under a parent index: per parent, each list's index
        #: and how many values it holds, its rows born with each parent row.
        #: On a root: born here from the graph as it is now, so a restore
        #: never takes them from a record.
        self._typed_under: dict[str, list[tuple[str, int]]] = {}
        self._typed_children: set[str] = set()
        self._typed_roots: set[str] = set()
        for node_id in order:
            self._map_readers(node_id)
        roots = [root for node_id in order for root in self._map_typed_lists(node_id)]
        self._birth_typed_roots(roots)

    def _map_readers(self, node_id: str) -> None:
        """File what this node reads and how: one ``_Reader`` per edge, under
        the output it reads and, for a series, under the index that series
        sits on. Also file the index the node itself iterates on."""
        node = self._compiled.node(node_id)
        connected: list[_Reader] = []
        for inp in node.interface.inputs:
            compiled_field = self._compiled.field(Ref(node_id, inp.name))
            if not isinstance(compiled_field.binding, From):
                continue
            for ref in compiled_field.binding.refs:
                reader = _Reader(node_id, ref, compiled_field.receives)
                connected.append(reader)
                self._readers.setdefault(ref, []).append(reader)
                index = self._index(ref)
                if not reader.per_row and index is not None:
                    self._readers_on.setdefault(index.id, []).append(reader)
        self._connected[node_id] = tuple(connected)
        if node.iterates_on is not None:
            self._iterating_on.setdefault(node.iterates_on.id, []).append(node_id)
            if node_id in self._births:
                self._births_on.setdefault(node.iterates_on.id, []).append(node_id)

    def _map_typed_lists(self, node_id: str) -> list[str]:
        """The lists the author typed into this node's scalar inputs. Compile
        says ``Iterate`` on an index of the input's own, one row per value,
        under a typed-in binding. A list under a parent index is filed to be
        born with each parent row. A list on a root has its rows born and
        sealed here, and its index is returned."""
        node = self._compiled.node(node_id)
        roots: list[str] = []
        for inp in node.interface.inputs:
            compiled_field = self._compiled.field(Ref(node_id, inp.name))
            if not (isinstance(compiled_field.binding, Static) and isinstance(compiled_field.receives, Iterate)):
                continue
            index, count = compiled_field.receives.index, len(node.statics[inp.name])
            if index.parent is not None:
                self._typed_under.setdefault(index.parent.id, []).append((index.id, count))
                self._typed_children.add(index.id)
                continue
            for n in range(count):
                self._born(index.id, (n,))
            self._sealed.add(index.id)
            self._typed_roots.add(index.id)
            roots.append(index.id)
        return roots

    def _birth_typed_roots(self, roots: list[str]) -> None:
        """Under every row of a typed-in list on a root, birth the typed-in
        lists whose index is its child, and seal them. Then seal the index of
        every node that births rows per row of a list complete from the start."""
        for index_id in roots:
            for row in sorted(self._rows.get(index_id, ())):
                self._born_typed(index_id, row)
            self._seal_typed(index_id, [])
        self._seal([node_id for index_id in roots for node_id in self._births_on.get(index_id, ())])

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

    def completed_nodes(self) -> set[str]:
        """The nodes ``complete`` holds for, among those with a unit done.

        A leg starting on a restored ledger reads it so it does not announce
        ``node_start`` again for a node an earlier leg finished. A node with
        no units at all — its index sealed with no rows — is not in it; it
        never starts, so there is nothing to announce."""
        return {node_id for node_id, _ in self._done if self.complete(node_id)}

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
        return all(self._read_ready(reader, row) for reader in self._connected[node_id])

    def runnable(self) -> list[Unit]:
        """Every unit that may start now: ready, and neither done nor waiting
        on a person. Asks ``ready`` of every unit, so the engine calls it
        where a leg starts and where it goes quiet, never per write."""
        return [
            unit
            for node_id in self._compiled.execution_order
            if not self.complete(node_id)
            for unit in self.units(node_id)
            if unit not in self._done and unit not in self._pending and self.ready(unit)
        ]

    def _read_ready(self, reader: _Reader, row: Row | None) -> bool:
        """Is what a unit at ``row`` reads through this reader written? Per
        row: the cell at its row. Grouped: every row of the group under it.
        Whole or gathered: everything on the ref."""
        if reader.per_row:
            return self._present(reader.ref, self._key(reader.ref, row))
        return self._written(reader.ref, _group(row, reader.depth))

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
        elif index.id in self._typed_children:
            born = parent_row in self._rows.get(index.parent.id, ())  # a typed-in list's rows are born with the parent row
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

    # -- what a unit receives -------------------------------------------------

    def inputs_for(self, unit: Unit) -> dict[str, Any] | Skip:
        """The keyword arguments this unit runs with, or the ``Skip`` that stops it.

        A ``Skip`` comes from ``SKIPPED`` on an input received per row, on
        the series an input receives whole or reduced, or at a row above
        this unit's. A gather drops skipped sources instead — an empty
        gather is an empty series — because gathering is how "whichever
        branch fired" is expressed.
        """
        node_id, row = unit
        iterate = self._compiled.node(node_id).iterates_on
        if iterate is not None and _depth(row) < iterate.depth:
            return Skip(at=row)
        values: dict[str, Any] = {}
        for inp in self._compiled.node(node_id).interface.inputs:
            own = Ref(node_id, inp.name)
            binding = self._compiled.field(own).binding
            received = self._compiled.field(own).receives
            if binding is None:
                if isinstance(received, Whole):
                    values[inp.name] = self._typed(received, inp.default, own)
                continue  # a scalar default is the node's own; the call applies it
            if isinstance(binding, Static):
                values[inp.name] = self._typed(received, self._compiled.node(node_id).statics[inp.name], own, row)
                continue
            if isinstance(received, (Iterate, Broadcast)):
                covering, skips = self._covering(node_id, binding.refs, row)
                if not covering:
                    return Skip(at=max(skips, key=_depth))
                (values[inp.name],) = covering
            elif isinstance(received, Gather):
                gathered: list[Any] = []
                for ref in binding.refs:
                    index = self._index(ref)
                    value, _ = self._lookup(ref, None)
                    if is_skipped(value):
                        continue
                    if index is None:
                        gathered.append(value)
                    else:
                        gathered.extend(self._series(node_id, (ref,), index, self._rows_under(index.id, None)).values)
                values[inp.name] = Series(received.index, gathered)
            else:
                group = row[: received.depth] if isinstance(received, Group) else None
                found = [self._lookup(ref, group) for ref in binding.refs]
                if all(is_skipped(value) for value, _ in found):
                    return Skip(at=max((at for _, at in found), key=_depth))
                index = self._index(binding.refs[0])
                if index is None:
                    # Received whole from a source that ran once: the one value.
                    ((values[inp.name], _),) = found
                else:
                    values[inp.name] = self._series(node_id, binding.refs, index, self._rows_under(index.id, group))
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

    def _typed(self, received: Receive, value: Any, own: Ref, row: Row | None = None) -> Any:
        """A value the author typed (or the declared default) as the unit receives it.

        Received whole, it is a series on the input's own index; iterated
        on an index of the input's own, it is many typed-in values and the
        unit takes the one at its row; broadcast, it is the one value.
        """
        if isinstance(received, Whole):
            return Series(self._index(own), value.values if isinstance(value, Series) else list(value))
        if isinstance(received, Iterate):
            return list(value)[row[received.index.depth - 1]]
        return value

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
            # Every output is checked before any cell is written, so the
            # ledger holds all of a unit's outputs or none of them.
            cells: dict[Ref, Any] = {}
            series: dict[Ref, list[Any]] = {}
            length: int | None = None
            for out in interface.outputs:
                ref = Ref(node_id, out.name)
                value = outputs[out.name]
                if out.dtype.element is None or is_skipped(value):
                    cells[ref] = value
                    continue
                if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                    raise ValueError(f"{node_id}: '{out.name}' is a series output and must be a sequence, not a {type(value).__name__}")
                series[ref] = list(value)
                if length is None:
                    length = len(series[ref])
                elif length != len(series[ref]):
                    raise ValueError(
                        f"{node_id}: its series outputs differ in length ({length} and {len(series[ref])})"
                    )
            for ref, value in cells.items():
                self._write(ref, row, value, written)
            for ref, values in series.items():
                for j, item in enumerate(values):
                    key = (j,) if row is None else (*row, j)
                    if self._born(node_id, key):
                        born.append(key)
                    self._write(ref, key, item, written)
            if births and length is None:
                barren.append(row)
        typed = [child for key in born for child in self._born_typed(node_id, key)]
        for at in barren:
            self._no_rows_under.setdefault(node_id, set()).add(at)
            typed.extend(self._barren_typed(node_id, at))
        self._finish(unit)
        sealed = self._seal([node_id] if births else [])
        return self._woken(unit, written, born, barren, sealed, typed)

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

    def _born_typed(self, index_id: str, row: Row) -> list[tuple[str, Row]]:
        """Birth, under a row just born on ``index_id``, the rows of every
        typed-in list whose index is its child, and theirs in turn; returns
        each ``(index, row)`` born."""
        born: list[tuple[str, Row]] = []
        for child, count in self._typed_under.get(index_id, ()):
            for n in range(count):
                if self._born(child, (*row, n)):
                    born.append((child, (*row, n)))
                    born.extend(self._born_typed(child, (*row, n)))
        return born

    def _barren_typed(self, index_id: str, at: Row | None) -> list[tuple[str, Row | None]]:
        """Where ``index_id`` has no rows under ``at``, neither has a typed-in
        list beneath it; returns each ``(index, row)`` marked."""
        marked: list[tuple[str, Row | None]] = []
        for child, _ in self._typed_under.get(index_id, ()):
            self._no_rows_under.setdefault(child, set()).add(at)
            marked.append((child, at))
            marked.extend(self._barren_typed(child, at))
        return marked

    def _seal_typed(self, index_id: str, sealed: list[str]) -> list[str]:
        """Seal the typed-in lists beneath a sealed index: every parent row
        is born, so every row under them is. Returns the nodes that birth on
        the indexes sealed, for ``_seal`` to try next."""
        births: list[str] = []
        for child, _ in self._typed_under.get(index_id, ()):
            if child in self._sealed:
                continue
            self._sealed.add(child)
            sealed.append(child)
            births.extend(self._births_on.get(child, ()))
            births.extend(self._seal_typed(child, sealed))
        return births

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
            births.extend(self._seal_typed(node_id, sealed))
        return sealed

    def _woken(
        self,
        unit: Unit,
        written: list[tuple[Ref, Row | None]],
        born: list[Row],
        barren: list[Row | None],
        sealed: list[str],
        typed: list[tuple[str, Row | None]],
    ) -> list[Unit]:
        """The units ``unit``'s record could have made ready, that are ready.

        A unit's readiness reads cells of its inputs, the rows born on the
        indexes those cells and the unit itself sit on, and which of those
        indexes are sealed. So the candidates are: a scalar reader of a
        written cell, at its row and under it; a series reader whose group
        the write completed, or everything under a skip above that group;
        the units a birth or an empty row creates, on this node's index or on
        a typed-in list beneath it; a series reader of this node's index whose
        group is born now that this unit is done or the index is sealed.
        """
        node_id, row = unit
        woken: set[Unit] = set()
        for ref, key in written:
            for reader in self._readers.get(ref, ()):
                if reader.per_row or (reader.depth is not None and _depth(key) < reader.depth):
                    woken.update(self._units_under(reader.node_id, key))
                else:
                    self._wake_group(ref, reader.node_id, _group(key, reader.depth), woken)
        for key in born:
            woken.update((reader, key) for reader in self._iterating_on.get(node_id, ()))
        for at in barren:
            woken.update((reader, at) for reader in self._iterating_on.get(node_id, ()))
        for index_id, key in typed:
            woken.update((reader, key) for reader in self._iterating_on.get(index_id, ()))
        for reader in self._readers_on.get(node_id, ()):
            for key in born:
                if reader.depth == len(key):
                    self._wake_group(reader.ref, reader.node_id, key, woken)
            if row is not None and reader.depth == len(row):
                self._wake_group(reader.ref, reader.node_id, row, woken)
        for index_id in sealed:
            for reader in self._readers_on.get(index_id, ()):
                if reader.depth is None:
                    self._wake_group(reader.ref, reader.node_id, None, woken)
        ready = [u for u in woken if u not in self._done and u not in self._pending and self.ready(u)]
        return sorted(ready, key=lambda u: (self._position[u[0]], _order(u)))

    def _wake_group(self, ref: Ref, reader: str, group: Row | None, woken: set[Unit]) -> None:
        """Add ``reader``'s units in ``group`` when every cell of ``ref`` they read is written."""
        if self._written(ref, group):
            woken.update(self._units_under(reader, group))

    def inject(self, node_id: str, outputs: Mapping[str, Any]) -> None:
        """Record ``outputs`` as what this node produced, without running it —
        a person's answers, or an earlier run's results.

        Each value is read through the codec as the type its output
        declares, so a host may hand in the typed value or its JSON form.
        For a node that runs once, ``outputs`` is its result. For a node
        running per row, each output is a series on the node's index — a
        ``Series``, ``{"rows": [...], "values": [...]}`` as it came over
        the wire, or a plain list with one value per row still to answer
        (born and not yet done), in row order — and only the rows it
        names are recorded:
        answering row ``(1,)`` leaves rows ``(0,)`` and ``(2,)`` as they
        were, done or still to run; every output answers the same rows.

        A node the graph does not have, an output the node does not have,
        an output left out, a unit that is already done, a row the run has
        not produced, and a row one output answers and another does not all
        raise ``StartRefused``, naming the node and the output.
        """
        try:
            node = self._compiled.node(node_id)
        except KeyError:
            raise StartRefused(f"'{node_id}' is not a node of this graph, so it cannot be given a result") from None
        declared = [out.name for out in node.interface.outputs]
        for name in outputs:
            if name not in declared:
                raise StartRefused(f"'{node_id}' has no output '{name}'")
        for name in declared:
            if name not in outputs:
                raise StartRefused(f"'{node_id}' was given no value for its output '{name}'")
        iterate = node.iterates_on
        if iterate is None:
            # The node's whole output at once: a series output is answered as the series.
            self._inject((node_id, None), {
                name: self._answer(node_id, name, value, self._compiled.field(Ref(node_id, name)).type) for name, value in outputs.items()
            })
            return
        born = self._rows.get(iterate.id, set())
        open_rows = [row for row in sorted(born) if (node_id, row) not in self._done]
        by_output = {name: self._answered_rows(node_id, name, value, open_rows) for name, value in outputs.items()}
        for row in sorted({row for rows in by_output.values() for row in rows}):
            if row not in born:
                raise StartRefused(f"'{node_id}' has no row {list(row)} to record: the run has not produced it")
            for name, answered in by_output.items():
                if row not in answered:
                    raise StartRefused(f"'{node_id}' answers '{name}' for no row {list(row)}: every output answers the same rows")
            at_row = {name: answered[row] for name, answered in by_output.items()}
            self._inject((node_id, row), Skip(at=row) if all(is_skipped(v) for v in at_row.values()) else at_row)

    def _answer(self, node_id: str, name: str, value: Any, dtype: Any | None = None) -> Any:
        """One answered value as ``dtype`` — the cell's type unless given; ``SKIPPED`` passes."""
        if is_skipped(value):
            return value
        declared = self._cell_type(Ref(node_id, name)) if dtype is None else dtype
        try:
            return from_wire(value, declared)
        except (TypeError, ValueError) as invalid:
            raise StartRefused(f"'{node_id}' was given a value for '{name}' that is not a {getattr(declared, '__name__', declared)}") from invalid

    def _answered_rows(self, node_id: str, name: str, value: Any, open_rows: list[Row]) -> dict[Row, Any]:
        """A per-row answer as ``{row: value}``: from a ``Series``, from
        ``rows`` and ``values`` as they came over the wire, or from a plain
        list answering every row in ``open_rows`` — the rows still to
        answer — in order."""
        if isinstance(value, Series):
            rows, values = value.rows, value.values
        elif isinstance(value, Mapping) and {"rows", "values"} <= set(value):
            rows, values = [tuple(row) for row in value["rows"]], value["values"]
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes):
            if len(value) != len(open_rows):
                raise StartRefused(f"'{node_id}': '{name}' answers {len(value)} values for {len(open_rows)} rows still open; a list answers every open row in order")
            rows, values = open_rows, list(value)
        else:
            raise StartRefused(f"'{node_id}' runs per row, so '{name}' is answered as a series, as rows and values, or as a list")
        if len(rows) != len(values):
            raise StartRefused(f"'{node_id}': '{name}' names {len(rows)} rows for {len(values)} values")
        if len(set(rows)) != len(rows):
            raise StartRefused(f"'{node_id}': '{name}' names a row twice")
        return {row: self._answer(node_id, name, item) for row, item in zip(rows, values, strict=True)}

    def _inject(self, unit: Unit, outputs: dict[str, Any] | Skip) -> None:
        if unit in self._done:
            node_id, row = unit
            where = "" if row is None else f" at row {list(row)}"
            raise StartRefused(f"'{node_id}'{where} is already done, so it cannot be given a result")
        self.record(unit, outputs)

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
            PendingUnit(node_id=node_id, row=row, prompt=prompt, questions=questions)
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
        for node_id in self._compiled.execution_order:
            if not self.complete(node_id):
                continue
            result = self.result_of(node_id)
            if result is not None:
                produced[node_id] = result
        return produced

    # -- the record ---------------------------------------------------------------

    def cells(self) -> RunRecord:
        """Everything this leg has produced, as the ``RunRecord`` a host stores.

        A host keeps it as the run's record and hands it back to
        ``execute(record=...)`` for the next leg, or for a new run seeded from
        this one. A waiting unit is not in it: it produced nothing and asks
        again next leg unless answered through ``cache``. Each value goes
        through the codec by its cell's type; a skipped cell carries
        ``skipped`` (the depth of its row) in place of ``value``; and every
        node's fingerprint is stored, so a restore can tell what changed.
        """
        return RunRecord(
            cells=[self._cell_wire(ref, row, value) for ref, by_row in self._cells.items() for row, value in by_row.items()],
            rows_by_index={index_id: [list(r) for r in sorted(rows)] for index_id, rows in self._rows.items()},
            sealed_indexes=sorted(self._sealed),
            childless_parent_rows={
                index_id: [None if r is None else list(r) for r in sorted(rows, key=lambda r: () if r is None else r)]
                for index_id, rows in self._no_rows_under.items()
            },
            done_units=[(node_id, None if row is None else list(row)) for node_id, row in self._done],
            node_fingerprints={node_id: self._compiled.node(node_id).fingerprint for node_id in self._compiled.execution_order},
        )

    def _cell_wire(self, ref: Ref, row: Row | None, value: Any) -> dict[str, Any]:
        """One cell as the record carries it: its address, and its value through the codec or its skip."""
        cell: dict[str, Any] = {"ref": [ref.node_id, ref.field], "row": None if row is None else list(row)}
        if is_skipped(value):
            cell["skipped"] = _depth(row)
            return cell
        try:
            cell["value"] = to_wire(value, self._cell_type(ref))
        except Exception as unwritable:
            raise TypeError(f"{ref} at row {row}: the value has no JSON form ({unwritable})") from unwritable
        return cell

    def _cell_type(self, ref: Ref) -> Any:
        """The type of one cell of ``ref``: the element of the series on a
        field with rows, the field's own type otherwise."""
        declared = self._compiled.field(ref).type
        element = getattr(declared, "element", None)
        return declared if self._compiled.field(ref).index is None or element is None else element

    @classmethod
    def restore(cls, compiled: CompiledGraph, record: RunRecord) -> Ledger:
        """A ledger holding what an earlier leg recorded, over the graph as it is now.

        A node the record fingerprints differently from ``compiled`` — or
        does not fingerprint at all, or that ``compiled`` no longer has — is
        left out together with everything that reads it, so those units run
        again; the rest is restored cell for cell, each value read back
        through the codec by its field's type. The rows of a typed-in list
        are never taken from the record: the graph as it is now says how
        many values the author typed, and a fresh ledger births them. A
        cell or a done unit for a node the graph does not have is left out
        too; a cell whose value does not read back as its field's type is a
        ``StartRefused``.
        """
        ledger = cls(compiled)
        #: A node the record names in a cell or a done unit but never fingerprinted, and the graph does not have.
        unknown = (
            {Ref(*cell["ref"]).node_id for cell in record.cells} | {node_id for node_id, _ in record.done_units}
        ) - set(compiled.execution_order)
        dropped = ledger._dropped(record.node_fingerprints) | unknown
        typed = ledger._typed_roots | ledger._typed_children
        for cell in record.cells:
            ref = Ref(*cell["ref"])
            if ref.node_id in dropped:
                continue
            row = None if cell["row"] is None else tuple(cell["row"])
            if "skipped" in cell:
                value = SKIPPED
            else:
                try:
                    value = from_wire(cell["value"], ledger._cell_type(ref))
                except (KeyError, TypeError, ValueError) as unreadable:
                    raise StartRefused(f"the record's value for {ref} does not read back: {unreadable}") from unreadable
            ledger._cells.setdefault(ref, {})[row] = value
        for index_id, rows in record.rows_by_index.items():
            if index_id in dropped or index_id in typed:
                continue
            ledger._rows.setdefault(index_id, set())
            for r in rows:
                ledger._born(index_id, tuple(r))
                ledger._born_typed(index_id, tuple(r))
        ledger._sealed |= set(record.sealed_indexes) - dropped - typed
        for index_id in sorted(ledger._sealed):
            ledger._seal_typed(index_id, [])
        ledger._no_rows_under.update({
            index_id: {None if r is None else tuple(r) for r in rows}
            for index_id, rows in record.childless_parent_rows.items()
            if index_id not in dropped and index_id not in typed
        })
        for node_id, row in record.done_units:
            if node_id not in dropped:
                ledger._finish((node_id, None if row is None else tuple(row)))
        return ledger

    def _dropped(self, stored: Mapping[str, str]) -> set[str]:
        """The nodes a restore leaves out, given the record's fingerprints:
        those the graph places differently (or the record has none for),
        those the graph no longer has, everything that reads any of them,
        and the typed-in lists whose rows are born under theirs."""
        compiled = self._compiled
        current = {node_id: compiled.node(node_id).fingerprint for node_id in compiled.execution_order}
        dropped = {node_id for node_id, fingerprint in current.items() if stored.get(node_id) != fingerprint}
        dropped.update(node_id for node_id in stored if node_id not in current)
        frontier = [node_id for node_id in dropped if node_id in current]
        while frontier:
            node_id = frontier.pop()
            for out in compiled.node(node_id).interface.outputs:
                ref = Ref(node_id, out.name)
                for reader in self._readers.get(ref, ()):
                    if reader.node_id not in dropped:
                        dropped.add(reader.node_id)
                        frontier.append(reader.node_id)
        for index_id in list(dropped):
            dropped.update(self._typed_below(index_id))
        return dropped

    def _typed_below(self, index_id: str) -> set[str]:
        """The indexes of every typed-in list born under rows of ``index_id``, and theirs in turn."""
        below: set[str] = set()
        for child, _ in self._typed_under.get(index_id, ()):
            below.add(child)
            below.update(self._typed_below(child))
        return below
