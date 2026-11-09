"""One leg's live table: what a run has produced so far, and what that lets run next.

The engine runs a graph as units. A unit is one node on one row. A node
that runs once is a single unit on the empty row ``()``; a node fed a series on
a scalar input runs once per row of that series' index, one unit per row.
A row is a path, ``(i,)`` on a root index and ``(i, j)`` for the j-th row
made under ``(i,)``, so a unit deep in the tree finds its row on an
ancestor index by taking a prefix. Rows are made when the unit producing
the series writes it; until then nobody knows how many there will be.

This table keeps everything a run has written, one value per field and
row, and answers the engine's two questions: is this unit ready, and
what does it run with. The engine is a loop over those calls. Nothing in
here is asynchronous and nothing in here schedules.

Readiness is kept rather than recomputed. Checking every unit after every
write would cost the square of the rows, so ``record`` returns the units
its write could have made ready, and only those are checked: the readers
of the written value at its row and below, the units a new row creates,
and a reduction whose group is now complete. ``ready`` stays the
definition, and ``runnable`` asks it of every unit when a leg starts and
when it goes quiet, where a ready unit nobody started is a bug.

How a unit receives each input, and who reads what, are compile's
decisions, read here: a node's ``_reads``, an output's ``_read_by``, and
per index who runs on it and what sits on it (on the field that makes the
index's rows, ``CompiledField._makes_rows`` and its kin). A node's ``_runs_per_row_of`` and ``_interface`` are read the
same way, past the check ``runs_per_row_of`` and ``interface`` make for other
callers: the engine runs only a graph whose every node is ready. Every
input carries a receive record (``CompiledField.receives``, from
``conductor.graph.receive``). ``PerRow`` takes the value at the unit's own
row, ``Broadcast`` the one value there is, ``Whole`` everything on the
field, ``RowsUnder`` the rows under the unit's row cut to a depth, and
``Gather`` unrelated sources collected onto the input's own index. The
this module switches on the record to say when a unit is ready and what to
hand it, and derives none of this from types or indexes itself.

A skip has a depth. ``SKIPPED`` written at row ``k`` means the field
holds nothing at ``k`` and at every row under it. A unit that returns
``SKIPPED`` writes it at its own row. A reader looks for a skip at its
own row and at every shorter prefix, and re-emits it at the depth it
found it. A unit whose every series output is skipped makes no rows, so
a node that would run per row of that index runs once at the shorter row
instead. That is how a skip keeps its reach down a chain.

What it holds is the run's state: ``state()`` writes that down when a leg
ends and ``restore`` starts the next leg from it, working the rows out again
from the values. Nothing is pruned, so a unit done in one leg stays done in
the next. A value written here has already passed its output's check
(``conductor.execution.validate_unit_outputs``); inside a leg it is trusted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterator

from pydantic import ValidationError

from conductor._sentinel import SKIPPED, is_skipped
from conductor.errors import ErrorCause, InvalidCallDetail, InvalidCallError, NodeExecutionError
from conductor.execution.run_state import OutputColumn, RunState, RunStatus, WaitingUnit
from conductor.graph.receive import Broadcast, Gather, Merge, PerRow, Receive, RowsUnder, Whole
from conductor.ref import Ref
from conductor.series import Index, Row, Series
from conductor.table import ROW_ID, Table, column_type, holds_tables, merge, row_id, with_row_ids

if TYPE_CHECKING:
    import polars as pl

    from conductor.graph.compiled import CompiledGraph
    from conductor.graph.compiled_node import CompiledField

#: One run of one node: the node id and the row it runs on — ``()`` for
#: a node that runs once; a row shorter than the index the node runs per row of, for the one
#: unit standing in for rows a skip above them never let be made.
Unit = tuple[str, Row]

_MISSING = object()
#: What a field nothing has been written to holds: no row.
_NO_VALUES: Mapping[Row, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class Skip:
    """This unit does not run: ``SKIPPED`` was found at row ``at`` on something it reads.

    ``at`` is the unit's own row, or a shorter prefix when the skip sits
    above it. The unit re-emits ``SKIPPED`` at ``at``, so the skip keeps its
    reach down the chain.
    Written and read inside this module only. Not the same thing as
    ``SKIPPED``, which is the value a node returns and carries no row.
    """

    at: Row


def _fits(row: Row, index: Index | None) -> bool:
    """Whether a stored row is one a field or node on ``index`` can hold: a
    row no deeper than ``index``, where ``()`` is the top (a node that runs
    once, or a skip from above standing in)."""
    return len(row) <= (0 if index is None else index.depth)


def _prefixes(key: Row) -> Iterator[Row]:
    """Every prefix of ``key`` from ``()`` to ``key`` itself."""
    for n in range(len(key) + 1):
        yield key[:n]


def _rows_under_depth(received: Receive) -> int:
    """The depth of the rows a reader receives: a ``RowsUnder``'s, else ``0`` (everything on the field)."""
    return received.depth if isinstance(received, RowsUnder) else 0


def _per_row(received: Receive) -> bool:
    """A reader that takes one value at its row, not a group or the whole field."""
    return isinstance(received, (PerRow, Broadcast))


class _FieldsOf(dict[Ref, "CompiledField"]):
    """The compiled field at each address, looked up on first use and kept.

    A field is read for every value it reads or writes, about a
    dozen per unit, so it asks ``CompiledGraph.field`` once per address and
    keeps the answer: a compiled graph never changes. Not a mirror of the
    graph: an address nothing asks for is never filled in."""

    def __init__(self, compiled: CompiledGraph) -> None:
        super().__init__()
        self._compiled = compiled

    def __missing__(self, ref: Ref) -> CompiledField:
        field = self[ref] = self._compiled.field(ref)
        return field


class LiveRunValues:
    """The live state of one run over one compiled graph, and what it makes ready.

    Built empty for a first leg, or from the ``state()`` of an earlier leg by
    ``restore``. The engine is the only caller.
    """

    def __init__(self, compiled: CompiledGraph) -> None:
        compiled._refuse_unless_runnable()
        self._compiled = compiled
        #: Each field read, by the address it was asked under:
        #: ``compiled.field`` once per address, a dict read after that.
        self._fields = _FieldsOf(compiled)
        #: Every column of a table output offered as a field: its values are the
        #: table's cells, read where the table sits (``_cell``).
        self._views = compiled._views
        self._values: dict[Ref, dict[Row, Any]] = {}
        #: The fields with a ``SKIPPED`` written at some row. Only these can
        #: hide a value under a skip above it, so a read of any other field is
        #: one lookup at its own row.
        self._skipped_on: set[Ref] = set()
        self._rows: dict[str, set[Row]] = {}
        #: Every ``(index, parent row)`` whose rows under that parent row are
        #: all made, and every ``(index, ())`` whose index is sealed. Rows
        #: under a row are always made in one go — a unit returns its whole
        #: list at once, a list set on a node is known before the run — so a
        #: group that exists is complete, whoever made it.
        self._groups: set[tuple[str, Row]] = set()
        #: Per index, its rows under each shorter row, in the order they were
        #: made; ``()`` holds them all. Finds the rows under a row without
        #: reading the whole index.
        self._rows_by_prefix: dict[str, dict[Row, list[Row]]] = {}
        self._sealed: set[str] = set()
        #: Per index, the rows beneath which it has no rows at all, because
        #: something above them was skipped.
        self._no_rows_under: dict[str, set[Row]] = {}
        self._done: set[Unit] = set()
        #: Per node that runs once per row, how many of its units are done:
        #: at a row of its index, and standing in at a shorter row.
        self._done_rows: dict[str, int] = {}
        self._done_standing_in: dict[str, int] = {}
        #: Units waiting on a person, each as the state carries it.
        self._waiting: dict[Unit, WaitingUnit] = {}
        #: Per group a series is read by — a field and the row it is read
        #: under — its rows in order and how many from the first are written.
        #: Kept once every row of the group is made, so no row is read twice.
        self._written_rows: dict[tuple[Ref, Row], tuple[list[Row], int]] = {}

        # Who reads what, and what runs on each index, is compile's read plan
        # (``CompiledField._makes_rows``, ``CompiledField._read_by`` and their
        # kin), looked up where a write needs it. Built here: each node's place
        # in the execution order, and the rows of every list set on a node
        # whose own index is a root, made from the graph as it is now so a
        # restore never takes them from a stored state.
        order = compiled.execution_order
        self._position = {node_id: n for n, node_id in enumerate(order)}
        roots: list[str] = []
        for node_id in order:
            node = compiled.node(node_id)
            for inp in node.interface.inputs:
                typed = compiled.field(Ref(node_id, inp.name))
                if not typed.listed or typed.index.parent is not None:
                    continue
                for n in range(len(node.statics[inp.name])):
                    self._made(typed.index.id, (n,))
                self._close(typed.index.id)
                roots.append(typed.index.id)
        self._make_root_list_rows(roots)

    def _make_root_list_rows(self, roots: list[str]) -> None:
        """Under every row of a list whose own index is a root, make the rows
        of every list whose index is its child, and seal them. Then seal the
        index of every node that makes rows per row of a list complete from
        the start."""
        for index_id in roots:
            for row in sorted(self._rows.get(index_id, ())):
                self._made_lists(index_id, row)
            self._seal_listed(index_id, [])
        self._seal([made for index_id in roots for made in self._making_rows_on(index_id)])

    # -- the read plan, as compile stored it -----------------------------------

    def _owner(self, index_id: str) -> CompiledField:
        """The field that makes an index's rows and carries its read plan: a
        series output, or an input holding a list set on its node, as compile
        recorded it (``CompiledGraph._index_owners``)."""
        return self._compiled._index_owners[index_id]

    def _making_rows_on(self, index_id: str) -> list[str]:
        """The indexes whose rows are made by nodes running once per row of this one."""
        return [made for node_id in self._owner(index_id)._run_per_row_by for made in self._indexes_made_by(node_id)]

    def _indexes_made_by(self, node_id: str) -> list[str]:
        """The indexes this node's series outputs make rows on, one per output."""
        fields = self._fields
        return [
            index.id for out in self._compiled.node(node_id)._interface.outputs
            if (index := fields[Ref(node_id, out.name)]._makes_rows) is not None
        ]

    def _grouped_on(self, index_id: str) -> Iterator[tuple[str, Ref, int]]:
        """``(reader node, output read, group depth)`` for every reader of a
        series on this index that receives it grouped or whole, not one row
        at a time."""
        for ref in self._owner(index_id)._carried_by:
            for inp, received in self._fields[ref]._read_by:
                if not _per_row(received):
                    yield inp.node_id, ref, _rows_under_depth(received)

    def _typed_under(self, index_id: str) -> Iterator[tuple[str, int]]:
        """``(index, how many values)`` for each list set on a node, made under every row of this index."""
        compiled = self._compiled
        for ref in self._owner(index_id)._listed_inputs:
            yield self._fields[ref].index.id, len(compiled.node(ref.node_id).statics[ref.field])

    # -- units ---------------------------------------------------------------

    def units(self, node_id: str) -> list[Unit]:
        """This node's units right now: one, or — running per row of ``L`` —
        one per row made on ``L``, plus one at each shorter row where ``L`` has no rows."""
        runs_per_row_of = self._compiled.node(node_id)._runs_per_row_of
        if runs_per_row_of is None:
            return [(node_id, ())]
        no_rows_under = sorted(self._no_rows_under.get(runs_per_row_of.id, ()))
        rows = sorted(self._rows.get(runs_per_row_of.id, ()))
        return [(node_id, row) for row in (*no_rows_under, *rows)]

    def is_done(self, unit: Unit) -> bool:
        return unit in self._done

    def complete(self, node_id: str) -> bool:
        """Every unit this node will ever have is done."""
        runs_per_row_of = self._compiled.node(node_id)._runs_per_row_of
        if runs_per_row_of is None:
            return (node_id, ()) in self._done
        return (
            runs_per_row_of.id in self._sealed
            and self._done_rows.get(node_id, 0) == len(self._rows.get(runs_per_row_of.id, ()))
            and self._done_standing_in.get(node_id, 0) == len(self._no_rows_under.get(runs_per_row_of.id, ()))
        )

    def completed_nodes(self) -> set[str]:
        """The nodes ``complete`` holds for, among those with a unit done.

        A leg starting from a state reads it so it does not announce
        ``node_start`` again for a node an earlier leg finished. A node with
        no units at all — its index sealed with no rows — is not in it; it
        never starts, so there is nothing to announce."""
        return {node_id for node_id, _ in self._done if self.complete(node_id)}

    def on_a_row(self, unit: Unit) -> bool:
        """Whether this unit is at a row of its node's index: one ``progress``
        counts. A unit of a node that runs once, or one standing in at a
        shorter row for rows never made, is not."""
        node_id, row = unit
        runs_per_row_of = self._compiled.node(node_id)._runs_per_row_of
        return runs_per_row_of is not None and len(row) == runs_per_row_of.depth

    def progress(self, node_id: str) -> tuple[int, int | None]:
        """``(done, total)`` rows for a node; ``total`` is ``None`` until the
        node's index is sealed. Cover units stand in for rows that were never
        made, so they count in neither number."""
        runs_per_row_of = self._compiled.node(node_id)._runs_per_row_of
        if runs_per_row_of is None:
            return int((node_id, ()) in self._done), 1
        total = len(self._rows.get(runs_per_row_of.id, ())) if runs_per_row_of.id in self._sealed else None
        return self._done_rows.get(node_id, 0), total

    def _units_under(self, node_id: str, row: Row) -> list[Unit]:
        """This node's units at ``row`` or beneath it, in no particular order —
        not counting a unit standing in at a shorter row, which is ready
        the moment it exists."""
        runs_per_row_of = self._compiled.node(node_id)._runs_per_row_of
        if runs_per_row_of is None:
            return [(node_id, ())] if row == () else []
        if row in self._rows.get(runs_per_row_of.id, ()):
            return [(node_id, row)]
        return [(node_id, r) for r in self._rows_by_prefix.get(runs_per_row_of.id, {}).get(row, ())]

    # -- readiness -----------------------------------------------------------

    def ready(self, unit: Unit) -> bool:
        """Every value this unit needs exists, or its producer is done."""
        node_id, row = unit
        runs_per_row_of = self._compiled.node(node_id)._runs_per_row_of
        if runs_per_row_of is not None and len(row) < runs_per_row_of.depth:
            return True  # standing in at a shorter row: the skip above it is already known
        return all(self._read_ready(ref, received, row) for ref, received in self._compiled.node(node_id)._reads)

    def runnable(self) -> list[Unit]:
        """Every unit that may start now: ready, and neither done nor waiting
        on a person. Asks ``ready`` of every unit, so the engine calls it
        where a leg starts and where it goes quiet, never per write."""
        return [
            unit
            for node_id in self._compiled.execution_order
            if not self.complete(node_id)
            for unit in self.units(node_id)
            if unit not in self._done and unit not in self._waiting and self.ready(unit)
        ]

    def _read_ready(self, ref: Ref, received: Receive, row: Row) -> bool:
        """Is what a unit at ``row`` reads of ``ref``, received as ``received``,
        written? Per row: the value at its row. Grouped: every row of the
        group under it. Whole or gathered: everything on the ref."""
        if _per_row(received):
            return self._present(ref, self._key(ref, row))
        return self._written(ref, row[: _rows_under_depth(received)])

    def _present(self, ref: Ref, key: Row) -> bool:
        return self._lookup(ref, key)[0] is not _MISSING

    def _key(self, ref: Ref, row: Row) -> Row:
        """Where a unit at ``row`` reads ``ref`` as a scalar: its row on the ref's index, or ``()``."""
        index = self._fields[ref].index
        return () if index is None else row[: index.depth]

    def _written(self, ref: Ref, group: Row) -> bool:
        """Is every value of ``ref`` a series reader at ``group`` receives written?
        The one value of a field that carries a single value, else the rows under ``group``."""
        index = self._fields[ref].index
        return self._present(ref, ()) if index is None else self._all_written(ref, index, group)

    def _all_written(self, ref: Ref, index: Index, parent_row: Row) -> bool:
        """Is every value of ``ref`` under ``parent_row`` written? True when the
        field is skipped there, or every row under it is made and produced."""
        if is_skipped(self._lookup(ref, parent_row)[0]):
            return True
        no_rows_under = self._no_rows_under.get(index.id, ())
        if any(prefix in no_rows_under for prefix in _prefixes(parent_row)):
            return False  # the unit standing in at that shorter row has not run yet
        if len(parent_row) == index.depth:
            made = parent_row in self._rows.get(index.id, ())  # the group is the row itself
        else:
            made = (index.id, parent_row) in self._groups
        if not made:
            return False
        # A made group's rows are final, so how many are written from the
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
        node = self._compiled.node(node_id)
        runs_per_row_of = node._runs_per_row_of
        if runs_per_row_of is not None and len(row) < runs_per_row_of.depth:
            return Skip(at=row)
        values: dict[str, Any] = {}
        statics = node._statics
        for inp in node._interface.inputs:
            own = Ref(node_id, inp.name)
            binding = self._fields[own].binding
            received = self._fields[own].receives
            if binding is None:
                if isinstance(received, Whole):
                    values[inp.name] = self._typed(received, inp.default, own)
                continue  # a scalar default is the node's own; the call applies it
            if inp.name in statics:
                values[inp.name] = self._typed(received, statics[inp.name], own, row)
                continue
            if isinstance(received, (PerRow, Broadcast)):
                covering, skips = self._covering(node_id, binding.refs, row)
                if not covering:
                    return Skip(at=max(skips, key=len))
                (values[inp.name],) = covering
            elif isinstance(received, Merge):
                merged = self._merged(received, row)
                if merged is None:
                    return Skip(at=row)
                # Read through the input's own type when the call is checked, so a
                # merged table it won't take fails this node, naming the column.
                values[inp.name] = Table._holding(merged)
            elif isinstance(received, Gather):
                gathered: list[Any] = []
                for ref in binding.refs:
                    index = self._fields[ref].index
                    value, _ = self._lookup(ref, ())
                    if is_skipped(value):
                        continue
                    if index is None:
                        gathered.append(value)
                    else:
                        gathered.extend(self._series(node_id, (ref,), index, self._rows_under(index.id, ())).values)
                values[inp.name] = Series(received.index, gathered)
            else:
                group = row[: _rows_under_depth(received)]
                found = [self._lookup(ref, group) for ref in binding.refs]
                if all(is_skipped(value) for value, _ in found):
                    return Skip(at=max((at for _, at in found), key=len))
                index = self._fields[binding.refs[0]].index
                if index is None:
                    # Received whole from a source that ran once: the one value.
                    ((values[inp.name], _),) = found
                else:
                    values[inp.name] = self._series(node_id, binding.refs, index, self._rows_under(index.id, group))
        return values

    def _merged(self, received: Merge, row: Row) -> pl.DataFrame | None:
        """The one table a unit at ``row`` receives from every source of a ``Table``
        input, or ``None`` when every source is skipped there.

        Each source becomes a frame of its rows with their ids: a table as it is
        stored, a column as ``__row`` beside its values (the ids a table's own
        column has are the table's). A source on rows above another's is a
        parent: its value is copied down onto each row under it.
        """
        import polars as pl

        pieces: list[tuple[str, pl.DataFrame, Index, list[Row]]] = []  # (source node, frame, rows' index, row paths)
        for source, part in received.parts:
            field = self._fields[source]
            if field._makes_rows is not None and field._view_of is None and isinstance(part, (PerRow, Broadcast)):
                at = () if isinstance(part, Broadcast) else row[: part.index.depth]
                stored, _ = self._lookup(source, at)
                if stored is _MISSING or is_skipped(stored):
                    continue
                pieces.append((source.node_id, stored.frame, field._makes_rows, [(*at, n) for n in range(stored.frame.height)]))
                continue
            index = field.index
            if index is None:
                raise RuntimeError(f"{source} is merged as a column but sits on no rows")
            if isinstance(part, PerRow):
                rows = [row[: index.depth]]
            else:
                rows = self._rows_under(index.id, row[: part.depth] if isinstance(part, RowsUnder) else ())
            present = [(at, value) for at in rows if not is_skipped(value := self._lookup(source, at)[0]) and value is not _MISSING]
            view = self._views.get(source)
            name = view[1] if view is not None else source.field
            frame = pl.DataFrame([
                pl.Series(ROW_ID, [self._row_id(index, at) for at, _ in present], dtype=pl.String),
                pl.Series(name, [value for _, value in present], dtype=column_type(self._fields[source]._cell_type), strict=False),
            ])
            pieces.append((source.node_id, frame, index, [at for at, _ in present]))
        if not pieces:
            return None
        parts: list[tuple[str, pl.DataFrame, str]] = []
        for name, frame, index, paths in pieces:
            above = [other for _, _, other, _ in pieces if other != index and index in other.lineage]
            if above:
                parts.append((name, frame, f"__{index.id}"))
                continue
            for _, _, parent, _ in pieces:
                if parent != index and parent in index.lineage:
                    frame = frame.with_columns(pl.Series(f"__{parent.id}", [self._row_id(parent, at[: parent.depth]) for at in paths], dtype=pl.String))
            parts.append((name, frame, ROW_ID))
        return merge(parts)

    def _row_id(self, index: Index, row: Row) -> str:
        """The id of the row at ``row`` on ``index``: on a table's rows, the id
        its stored table carries, which may have come in with the table; on
        any other index, the id spelled from the index and the row."""
        owner = self._compiled._index_owners.get(index.id)
        if owner is not None and owner._view_of is None and owner._output and holds_tables(owner._type):
            stored, _ = self._lookup(owner.ref, row[:-1])
            return stored.frame[ROW_ID][row[-1]]
        return row_id(index, row)

    def _covering(self, node_id: str, refs: tuple[Ref, ...], row: Row) -> tuple[list[Any], list[Row]]:
        """Of several refs read at one row: the values that cover it, and the
        rows at which the others were skipped. Two covering the same row
        means two edges both supplied that row, and the node fails with
        that row."""
        covering: list[Any] = []
        skips: list[Row] = []
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

    def _typed(self, received: Receive, value: Any, own: Ref, row: Row = ()) -> Any:
        """A value the author typed (or the declared default) as the unit receives it.

        Received whole, it is a series on the input's own index; received
        per row of an index of the input's own, it is a list set on the node
        and the unit takes the value at its row; broadcast, it is the one
        value.
        """
        if isinstance(received, Whole):
            return Series(self._fields[own].index, value.values if isinstance(value, Series) else list(value))
        if isinstance(received, PerRow):
            return list(value)[row[received.index.depth - 1]]
        return value

    def _lookup(self, ref: Ref, key: Row) -> tuple[Any, Row]:
        """``(value, row)`` for the value at ``(ref, key)``: ``SKIPPED`` at the
        shortest prefix of ``key`` that holds one, else what is at ``key``
        (``_MISSING`` when nothing is). A column of a table is read off the
        table at the row above (``_cell``)."""
        view = self._views.get(ref)
        if view is not None:
            return self._cell(ref, view, key)
        by_row = self._values.get(ref, _NO_VALUES)
        if ref in self._skipped_on:
            for prefix in _prefixes(key):
                if by_row.get(prefix) is SKIPPED:
                    return SKIPPED, prefix
        return by_row.get(key, _MISSING), key

    def _cell(self, ref: Ref, view: tuple[Ref, str], key: Row) -> tuple[Any, Row]:
        """The value of a table's column at ``key``: the cell in row ``key[-1]``
        of the table stored at ``key[:-1]``, as its column's type, a blank
        cell ``None``. A skip on the table is a skip here, at its depth."""
        table, column = view
        if len(key) < self._fields[ref].index.depth:
            # A group above the table's rows: only a skip on the table reaches it.
            stored, at = self._lookup(table, key)
            return (stored, at) if is_skipped(stored) else (_MISSING, key)
        stored, at = self._lookup(table, key[:-1])
        if stored is _MISSING or is_skipped(stored):
            return stored, at
        cells = stored.frame[column]
        if key[-1] >= len(cells):
            return _MISSING, key
        cell = cells[key[-1]]
        return (None if cell is None else self._fields[ref]._cell_type(cell)), key

    def _rows_under(self, index_id: str, parent_row: Row) -> list[Row]:
        """The rows of an index whose path starts with ``parent_row``: every
        row for ``()``, a child's rows under a parent row, or the row
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

        ``outputs`` are cells that passed their outputs' checks: one cell, a
        ``SKIPPED``, or a list of cells for an output that makes a row per
        value. A scalar output is one value at the unit's row; so is
        ``SKIPPED`` on any output. A many-valued output makes rows under the
        unit's row on an index of its own — ``(j,)`` for a node that ran once,
        ``row + (j,)`` for one running per row — so two of them from one unit
        may differ in length. A skipped one leaves its index with no rows
        under that row; an empty one makes no rows and leaves an empty group,
        so a reduction over it still runs. A table is one value, and makes
        rows on its columns' index only when something reads a column.

        The units returned are the ones this write could have made ready and
        that answer ``ready`` now, in execution order; the engine starts
        those and asks nothing else.
        """
        node_id, row = unit
        node = self._compiled.node(node_id)
        fields = node._fields
        written: list[tuple[Ref, Row]] = []
        made: dict[str, list[Row]] = {}
        no_rows_under: list[tuple[str, Row]] = []
        making: list[str] = []
        skip_at = outputs.at if isinstance(outputs, Skip) else None
        for out in node._interface.outputs:
            field = fields[out.name]
            ref = field.ref
            value = SKIPPED if skip_at is not None else outputs[out.name]  # type: ignore[index]
            at = row if skip_at is None else skip_at
            index = field._makes_rows
            if index is None:
                self._write(ref, at, value, written)
                continue
            making.append(index.id)
            self._rows.setdefault(index.id, set())
            if value is SKIPPED:
                self._write(ref, at, value, written)
                no_rows_under.append((index.id, at))
            elif isinstance(value, Table):
                self._write(ref, at, value, written)
                if self._columns_read(index.id):
                    keys = made.setdefault(index.id, [])
                    for j in range(value.frame.height):
                        if self._made(index.id, (*at, j)):
                            keys.append((*at, j))
                self._groups.add((index.id, at))
            else:
                keys = made.setdefault(index.id, [])
                for j, item in enumerate(value):
                    key = (*at, j)
                    if self._made(index.id, key):
                        keys.append(key)
                    self._write(ref, key, item, written)
                self._groups.add((index.id, at))
        lists = [child for index_id, keys in made.items() for key in keys for child in self._made_lists(index_id, key)]
        for index_id, at in no_rows_under:
            self._no_rows_under.setdefault(index_id, set()).add(at)
            lists.extend(self._no_rows_under_lists(index_id, at))
        self._finish(unit)
        sealed = self._seal(list(making))
        return self._woken(unit, written, made, no_rows_under, making, sealed, lists)

    def _columns_read(self, index_id: str) -> bool:
        """Does anything read a column of the table that owns this index? If
        nothing does, a table passing through makes no rows at all."""
        owner = self._owner(index_id)
        return bool(owner._run_per_row_by) or any(self._fields[view]._read_by for view in owner._carried_by)

    def _write(self, ref: Ref, key: Row, value: Any, written: list[tuple[Ref, Row]]) -> None:
        """Put ``value`` at ``(ref, key)``, noting the address in ``written`` when nothing was there."""
        by_row = self._values.setdefault(ref, {})
        if key not in by_row:
            written.append((ref, key))
        by_row[key] = value
        if value is SKIPPED:
            self._skipped_on.add(ref)

    def _made(self, index_id: str, row: Row) -> bool:
        """Make ``row`` on the index; ``False`` when it already was made."""
        rows = self._rows.setdefault(index_id, set())
        if row in rows:
            return False
        rows.add(row)
        by_prefix = self._rows_by_prefix.setdefault(index_id, {})
        for prefix in _prefixes(row[:-1]):
            by_prefix.setdefault(prefix, []).append(row)
        return True

    def _made_lists(self, index_id: str, row: Row) -> list[tuple[str, Row]]:
        """Make, under a row just made on ``index_id``, the rows of every
        list set on a node whose index is its child, and theirs in turn;
        returns each ``(index, row)`` made."""
        made: list[tuple[str, Row]] = []
        for child, count in self._typed_under(index_id):
            for n in range(count):
                if self._made(child, (*row, n)):
                    made.append((child, (*row, n)))
                    made.extend(self._made_lists(child, (*row, n)))
            self._groups.add((child, row))
        return made

    def _no_rows_under_lists(self, index_id: str, at: Row) -> list[tuple[str, Row]]:
        """Where ``index_id`` has no rows under ``at``, neither has a list set
        on a node beneath it; returns each ``(index, row)`` marked."""
        marked: list[tuple[str, Row]] = []
        for child, _ in self._typed_under(index_id):
            self._no_rows_under.setdefault(child, set()).add(at)
            marked.append((child, at))
            marked.extend(self._no_rows_under_lists(child, at))
        return marked

    def _seal_listed(self, index_id: str, sealed: list[str]) -> list[str]:
        """Seal the lists set on nodes beneath a sealed index: every parent
        row is made, so every row under them is. Returns the indexes made by
        nodes running per row of the ones sealed, for ``_seal`` to try next."""
        next_up: list[str] = []
        for child, _ in self._typed_under(index_id):
            if child in self._sealed:
                continue
            self._close(child)
            sealed.append(child)
            next_up.extend(self._making_rows_on(child))
            next_up.extend(self._seal_listed(child, sealed))
        return next_up

    def _close(self, index_id: str) -> None:
        """Seal an index: no row will ever be added to it, so the whole of it is one made group."""
        self._sealed.add(index_id)
        self._groups.add((index_id, ()))

    def _finish(self, unit: Unit) -> None:
        """Mark the unit done, and count it toward its node's rows."""
        if unit in self._done:
            return
        self._done.add(unit)
        node_id, _ = unit
        if self._compiled.node(node_id)._runs_per_row_of is not None:
            counts = self._done_rows if self.on_a_row(unit) else self._done_standing_in
            counts[node_id] = counts.get(node_id, 0) + 1

    def _seal(self, index_ids: list[str]) -> list[str]:
        """Seal each of these indexes whose maker's node is complete, and every
        index made by a node running per row of one sealed that way, since a
        child index can only be sealed once its parent is. Returns the
        indexes sealed."""
        sealed: list[str] = []
        while index_ids:
            index_id = index_ids.pop()
            if index_id in self._sealed or not self.complete(self._owner(index_id).ref.node_id):
                continue
            self._close(index_id)
            sealed.append(index_id)
            index_ids.extend(self._making_rows_on(index_id))
            index_ids.extend(self._seal_listed(index_id, sealed))
        return sealed

    def _woken(
        self,
        unit: Unit,
        written: list[tuple[Ref, Row]],
        made: Mapping[str, list[Row]],
        no_rows_under: list[tuple[str, Row]],
        making: list[str],
        sealed: list[str],
        lists: list[tuple[str, Row]],
    ) -> list[Unit]:
        """The units ``unit``'s record could have made ready, that are ready.

        A unit's readiness reads values of its inputs, the rows made on the
        indexes those values and the unit itself sit on, and which of those
        indexes are sealed. So the candidates are: a scalar reader of a
        written value, at its row and under it; a series reader whose group
        the write completed, or everything under a skip above that group;
        the units new rows or an empty row create, on an index this node's
        outputs make or on a list set on a node beneath it; a series reader of
        such an index whose group is made now that this unit is done or the
        index is sealed.
        """
        _, row = unit
        woken: set[Unit] = set()
        for ref, key in written:
            for inp, received in self._fields[ref]._read_by:
                depth = _rows_under_depth(received)
                if _per_row(received) or len(key) < depth:
                    woken.update(self._units_under(inp.node_id, key))
                else:
                    self._wake_group(ref, inp.node_id, key[:depth], woken)
        for index_id, keys in made.items():
            readers = self._owner(index_id)._run_per_row_by
            for key in keys:
                woken.update((reader, key) for reader in readers)
        for index_id, key in (*no_rows_under, *lists):
            woken.update((reader, key) for reader in self._owner(index_id)._run_per_row_by)
        for index_id in making:
            keys = made.get(index_id, [])
            for reader, ref, depth in self._grouped_on(index_id):
                for key in keys:
                    if depth == len(key):
                        self._wake_group(ref, reader, key, woken)
                if 0 < depth == len(row):  # a whole read (depth 0) waits for the seal below
                    self._wake_group(ref, reader, row, woken)
        for index_id in sealed:
            for reader, ref, depth in self._grouped_on(index_id):
                if depth == 0:
                    self._wake_group(ref, reader, (), woken)
        ready = [u for u in woken if u not in self._done and u not in self._waiting and self.ready(u)]
        return sorted(ready, key=lambda u: (self._position[u[0]], u[1]))

    def _wake_group(self, ref: Ref, reader: str, group: Row, woken: set[Unit]) -> None:
        """Add ``reader``'s units in ``group`` when every value of ``ref`` they read is written."""
        if self._written(ref, group):
            woken.update(self._units_under(reader, group))

    # -- waiting on a person ---------------------------------------------

    def mark_waiting(self, unit: Unit, asked: WaitingUnit) -> None:
        """Park this unit: its node asked a person. It and whatever reads it wait."""
        self._waiting[unit] = asked

    def is_waiting(self, unit: Unit) -> bool:
        return unit in self._waiting

    def waiting(self) -> tuple[WaitingUnit, ...]:
        """Every unit waiting on a person, in execution order and row order
        within a node: not the order their concurrent calls happened to ask in."""
        return tuple(
            asked for _, asked in sorted(self._waiting.items(), key=lambda item: (self._position[item[0][0]], item[0][1]))
        )

    # -- what the run produced -------------------------------------------------------

    def result_of(self, node_id: str) -> dict[str, Any] | None:
        """This node's outputs by name, or ``None`` when the node did not run
        (every output is ``SKIPPED`` at the top).

        A result holds only what the node produced: an output the node
        skipped (the branch not taken) is absent, never ``SKIPPED``. A scalar
        output of a node that ran once is its value; anything on an index is
        a ``Series``, sparse where rows were skipped — a node whose every row
        was skipped away gives an empty series.
        """
        outputs = self._compiled.node(node_id)._interface.outputs
        if all(is_skipped(self._lookup(Ref(node_id, out.name), ())[0]) for out in outputs):
            return None
        result: dict[str, Any] = {}
        for out in outputs:
            ref = Ref(node_id, out.name)
            index = self._fields[ref].index
            value, _ = self._lookup(ref, ())
            if is_skipped(value):
                continue
            if index is None:
                result[out.name] = value
            else:
                written = [row for row in self._rows_under(index.id, ()) if self._present(ref, row)]
                result[out.name] = self._series(node_id, (ref,), index, written)
        return result

    def results(self) -> dict[str, dict[str, Any]]:
        """Every node's outputs so far, by node id: a node with a finished
        unit or none to run, its per-row outputs holding the rows written so far.
        A node that did not run is absent."""
        started = {node_id for node_id, _ in self._done}
        produced: dict[str, dict[str, Any]] = {}
        for node_id in self._compiled.execution_order:
            if node_id not in started and not self.complete(node_id):
                continue
            result = self.result_of(node_id)
            if result is not None:
                produced[node_id] = result
        return produced

    # -- the state ----------------------------------------------------------------

    def state(self, status: RunStatus, failure: ErrorCause | None = None) -> RunState:
        """Everything this leg has produced, as the ``RunState`` a host stores.

        Built without validation from values that already passed, and holding
        them typed: the next read costs nothing to decode, and a dump writes
        each one by its own type."""
        columns: dict[Ref, OutputColumn] = {}
        for ref, by_row in self._values.items():
            rows: list[Row] = []
            values: list[Any] = []
            skipped: list[Row] = []
            for row, value in by_row.items():
                if value is SKIPPED:
                    skipped.append(row)
                else:
                    rows.append(row)
                    values.append(value)
            columns[ref] = OutputColumn.model_construct(rows=rows, values=values, skipped=skipped)
        finished: dict[str, list[Row]] = {}
        for node_id, row in self._done:
            finished.setdefault(node_id, []).append(row)
        compiled = self._compiled
        return RunState.model_construct(
            status=status,
            failure=failure,
            waiting=self.waiting(),
            output_columns=columns,
            finished_rows=finished,
            node_fingerprints={node_id: compiled.node(node_id).fingerprint for node_id in compiled.execution_order},
        )

    @classmethod
    def restore(cls, compiled: CompiledGraph, state: RunState) -> LiveRunValues:
        """Live values holding what an earlier leg recorded, over the graph as it is now.

        Every node ``compiled.nodes_changed_since(state)`` names is left out,
        so its units run again; a node gone from the graph drops with them.
        The rest is restored, each value through its output's check. The rows
        are not stored: each finished unit's values make them again, as its
        write did (``_remake_rows``), and the indexes whose producers are
        complete are sealed again. The rows of a list set on a node come from
        the graph as it is now, as for a fresh run.

        A stored value that does not read as its output's type, a value or a
        unit at a row its field or node has no place for, and a value for a
        field the graph does not have are all reported at once, as
        ``InvalidCallError`` with the code ``state_unreadable``. A ``None``
        passes on any output: a blank is a legal value in a state.
        """
        restored = cls(compiled)
        dropped = set(compiled.nodes_changed_since(state))
        faults: list[InvalidCallDetail] = []
        for ref, column in state.output_columns.items():
            if ref.node_id not in dropped:
                restored._restore_column(ref, column, faults)
        done: list[Unit] = []
        for node_id, rows in state.finished_rows.items():
            if node_id in dropped:
                continue
            for row in rows:
                if restored._fits_node(node_id, row, faults, ("state", node_id, *row)):
                    done.append((node_id, row))
        for asked in state.waiting:
            if asked.node_id not in dropped and restored._fits_node(asked.node_id, asked.row, faults, ("state", asked.id)):
                restored._waiting[(asked.node_id, asked.row)] = asked
        if faults:
            raise InvalidCallError(faults)
        restored._remake_rows(done)
        for unit in done:
            restored._finish(unit)
        restored._seal([
            index_id for node_id in reversed(compiled.execution_order) if node_id not in dropped
            for index_id in restored._indexes_made_by(node_id)
        ])
        return restored

    def _restore_column(self, ref: Ref, column: OutputColumn, faults: list[InvalidCallDetail]) -> None:
        """One output's stored rows and cells written in, or the faults that stop them."""
        try:
            field = self._compiled.field(ref)
        except KeyError:
            faults.append(_unreadable(("state", ref), f"the state has a value for {ref}, which the graph does not have"))
            return
        depth = 0 if field.index is None else field.index.depth
        misplaced = [row for row in (*column.rows, *column.skipped) if len(row) > depth]
        for row in misplaced:
            faults.append(_unreadable(("state", ref, *row), f"the state's value for {ref} is at row {list(row)}, which {ref} has no place for"))
        if misplaced:
            return
        by_row = self._values.setdefault(ref, {})
        read = field._read_cell
        for row, stored in zip(column.rows, column.values, strict=True):
            try:
                by_row[row] = _restored_cell(field, row, stored, read)
            except (TypeError, ValueError) as unreadable:
                faults.append(_unreadable(("state", ref, *row), f"the state's value for {ref} at row {list(row)} does not read back: {_reason(unreadable)}"))
        if column.skipped:
            self._skipped_on.add(ref)
            for row in column.skipped:
                by_row[row] = SKIPPED

    def _fits_node(self, node_id: str, row: Row, faults: list[InvalidCallDetail], loc: tuple[str | int, ...]) -> bool:
        """Whether a stored unit sits at a row its node can have; the fault that says it does not, if not."""
        if node_id not in self._compiled._nodes:
            return False
        if _fits(row, self._compiled.node(node_id).runs_per_row_of):
            return True
        faults.append(_unreadable(loc, f"the state has {node_id!r} at row {list(row)}, which {node_id!r} has no place for"))
        return False

    def _remake_rows(self, done: list[Unit]) -> None:
        """Make again the rows the done units' writes made, and mark again
        where they left an index with no rows: what ``record`` did as each
        unit was written, read back off its values.

        Per series output of a done unit: a skip at the unit's row, or at a
        shorter row its skip reached from, left that output's index with no
        rows under the row; otherwise the output made the rows one level under
        the unit's row that it holds values at, none for a series that came
        back empty, and that group is made. Rows are made in order, and each
        row made makes the rows of the lists set on nodes under it, as
        ``record`` does.
        """
        made: dict[str, set[Row]] = {}
        for node_id, row in done:
            for out in self._compiled.node(node_id)._interface.outputs:
                ref = Ref(node_id, out.name)
                index = self._fields[ref]._makes_rows
                if index is None:
                    continue
                self._rows.setdefault(index.id, set())
                by_row = self._values.get(ref, {})
                at = next((key for key in _prefixes(row) if is_skipped(by_row.get(key))), None)
                if at is not None:
                    self._no_rows_under.setdefault(index.id, set()).add(at)
                    self._no_rows_under_lists(index.id, at)
                    continue
                stored = by_row.get(row)
                if isinstance(stored, Table):
                    if self._columns_read(index.id):
                        made.setdefault(index.id, set()).update((*row, j) for j in range(stored.frame.height))
                else:
                    made.setdefault(index.id, set()).update(key for key in by_row if len(key) == len(row) + 1 and key[:-1] == row)
                self._groups.add((index.id, row))
        for node_id in self._compiled.execution_order:
            for index_id in self._indexes_made_by(node_id):
                for key in sorted(made.get(index_id, ())):
                    self._made(index_id, key)
                    self._made_lists(index_id, key)


def _restored_cell(field: CompiledField, row: Row, stored: Any, read: Any) -> Any:
    """A stored value as its cell. A value already of its type came from a
    leg in this process and is as that leg left it; a ``None`` is a blank. Any
    other goes through the output's check, and a table from JSON gets the ids
    it may lack, as a returned one does."""
    if stored is None or type(stored) is field._cell_type:
        return stored
    cell = read(stored)
    if field._makes_rows is not None and isinstance(cell, Table):
        return type(cell)._holding(with_row_ids(cell.frame, field._makes_rows, row))
    return cell


def _unreadable(loc: tuple[str | int, ...], message: str) -> InvalidCallDetail:
    return {"type": "state_unreadable", "loc": loc, "msg": message, "ctx": {}}


def _reason(unreadable: Exception) -> str:
    """Why a value did not read: pydantic's first message when pydantic said, never the value."""
    if isinstance(unreadable, ValidationError):
        return unreadable.errors(include_url=False, include_input=False, include_context=False)[0]["msg"]
    return str(unreadable)
