"""``RunLedger``: what a run has produced so far, and what that lets run next.

The engine runs a graph as units. A unit is one node on one row. A node
that runs once is a single unit on the empty row ``()``; a node fed a series on
a scalar input runs once per row of that series' index, one unit per row.
A row is a path, ``(i,)`` on a root index and ``(i, j)`` for the j-th row
made under ``(i,)``, so a unit deep in the tree finds its row on a parent
index by cutting its path short. Rows are made when the unit producing them
writes; until then nobody knows how many there will be.

The ledger keeps one value per output and unit, written at the unit's own row:
a cell, ``SKIPPED``, a list of cells on an output that makes rows, or a table.
The list and the table hold the rows' cells; a reader finds its cell in them by
its position. What the values made, every index's rows, is kept per index
(``RowsOfIndex``). From those two the ledger answers the engine's questions:
what does this unit still need (``first_thing_unit_still_needs``), and what
does it run with (``inputs_for``). The engine is a loop over those calls and
nothing in here is asynchronous.

A skip belongs to a row. A unit that didn't run or a choice not taken writes
``SKIPPED``: at the unit's row, or, when the skip came from a shorter row of an
input, at that row, since the unit's whole group is then skipped. A row-making
output that is skipped gives its index no rows under the row, and every index
under that one none either, so a reader asks the index tree by prefix.

Compile decided how each unit receives each input and who reads what; both
are read here from ``EngineLookups``, never worked out again from types or
indexes. What the ledger holds is the run's state: ``state`` writes it down
when a leg ends or a checkpoint is due, and ``restore`` starts the next leg by
playing the finished units back through the writes that made them.

A node that changed since that state runs its units again, and a unit whose
inputs match one an earlier unit of the node received is not called: its
result is written through ``record`` at this unit's row, as if the node had
just returned it (``claim_earlier_result``). Nothing maps an old row to a new one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from operator import itemgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import ValidationError

from conductor._internal.errors import (
    PROBLEM_CATALOGUE,
    InvalidCallError,
    NodeError,
    Problem,
    make_problem,
)
from conductor._internal.execution.input_hash import DigestsOfValues, hash_of_inputs
from conductor._internal.execution.run_state import (
    EarlierWork,
    OutputColumn,
    RunState,
    RunStatus,
    WaitingUnit,
    id_of_waiting_unit,
)
from conductor._internal.graph.receive_modes import (
    Cell,
    FixedValue,
    Gather,
    Merge,
    ReceiveMode,
    RowsUnder,
)
from conductor._internal.index import Index, Row, ValuesOnRows
from conductor._internal.node_definition import NodeVersion
from conductor._internal.ref import Ref
from conductor._internal.skip_and_ask import SKIPPED
from conductor._internal.table import (
    ROW_ID_COLUMN,
    MergeSource,
    frame_with_no_rows,
    global_row_id,
    is_frame,
    is_table_type,
    merge_tables_on_row_ids,
    one_column_frame_with_row_ids,
    with_ids_spelled_at_row,
)

if TYPE_CHECKING:
    import polars as pl

    from conductor._internal.field_records import Output
    from conductor._internal.graph.compiled_graph import CompiledGraph
    from conductor._internal.graph.compiled_node import CompiledField
    from conductor._internal.graph.engine_lookups import EngineLookups

#: One run of one node: the node id and the row it runs on, ``()`` for a node that runs once.
Unit = tuple[str, Row]

_MISSING = object()
#: How a field's cell is held when it isn't by itself at its row.
_AT_ITS_OWN_ROW = object()
#: What a field nothing has been written to holds: no row.
_NO_VALUES: Mapping[Row, Any] = MappingProxyType({})
#: Where ``RunState.earlier_work`` starts numbering its rows: past any row a graph makes,
#: so they stay apart from the rows of units that ran, in the same pool of a later leg.
_ROWS_OF_EARLIER_WORK_START = 1 << 40


@dataclass(frozen=True, slots=True)
class SkippedAtRow:
    """This unit does not run: ``SKIPPED`` was found at row ``row`` on something it reads.

    A skip has a depth. ``SKIPPED`` written at a row means the field holds
    nothing at that row or at any row under it. A reader looks for a skip at
    its own row and at every shorter row, and writes it again at the row it
    found it, so the skip keeps its reach down the chain. ``row`` is the
    unit's own row or that shorter one.

    Written and read inside this module only. Not ``SKIPPED`` itself, the value
    a node returns, which carries no row.
    """

    row: Row


@dataclass(frozen=True, slots=True)
class BlankInputsStopCall:
    """This unit does not run: a blank reached these inputs, which take none.

    A blank is a value: ``None`` in a table's cell or written by a node. Its
    outputs are blank at that row, a ``choice`` output is skipped (no branch
    was chosen), and an output that makes rows has no rows there. A
    ``Series[X]`` input leaves blanks out row by row. A unit is checked in
    this order: not ready, skip, blank.
    """

    blank_inputs: tuple[str, ...]


class UnitRecorded(NamedTuple):
    """What one ``record`` changed for the engine: the units ready because of it,
    in execution order, and the nodes it finished, its own and any others that
    have no units left to run, each once."""

    units_ready: list[Unit]
    nodes_completed: list[str]


class RowsOfIndex:
    """One index's rows: how many under each row, and where it has none because of a skip.

    A row's position is the next one under its parent, so ``count_under`` says
    everything about which rows exist. ``count_under[()]`` is the whole index.
    Where a skip leaves the index with no rows, ``no_rows_under`` holds that
    row, and the same row for every index beneath this one. Holds no values
    and decides nothing: the ledger asks.
    """

    __slots__ = ("depth", "parent", "count_under", "no_rows_under", "child_indexes")

    def __init__(self, depth: int) -> None:
        self.depth = depth
        self.parent: RowsOfIndex | None = None
        self.count_under: dict[Row, int] = {}
        self.no_rows_under: set[Row] = set()
        self.child_indexes: list[RowsOfIndex] = []

    def make_rows_under(self, parent_row: Row, count: int) -> None:
        """Make the rows ``parent_row + (0..count)``, all at once: a unit's whole list."""
        for depth in range(len(parent_row) + 1):
            row_on_path = parent_row[:depth]
            self.count_under[row_on_path] = self.count_under.get(row_on_path, 0) + count

    def mark_no_rows_under(self, row: Row) -> None:
        """Record that this index and those under it have no rows beneath ``row``. Idempotent."""
        if row not in self.no_rows_under:
            self.no_rows_under.add(row)
            for child in self.child_indexes:
                child.mark_no_rows_under(row)

    def has_row(self, row: Row) -> bool:
        return 0 <= row[-1] < self.count_under.get(row[:-1], 0)

    def rows_under(self, group_row: Row) -> list[Row]:
        """The rows of this index whose path starts with ``group_row``, in order;
        the row itself when ``group_row`` is one."""
        if len(group_row) >= self.depth:
            row = group_row[: self.depth]
            return [row] if self.has_row(row) else []
        if len(group_row) == self.depth - 1:
            return [(*group_row, position) for position in range(self.count_under.get(group_row, 0))]
        assert self.parent is not None  # an index shallower than the group row always has parents
        return [row for parent_row in self.parent.rows_under(group_row) for row in self.rows_under(parent_row)]


class RunLedger:
    """The record of one run over one compiled graph, and what it makes ready.

    Built empty for a first leg, or from the ``state()`` of an earlier leg by
    ``restore``. The engine is the only caller.
    """

    def __init__(self, compiled: CompiledGraph) -> None:
        compiled._raise_unless_runnable()
        self._compiled = compiled
        #: Every lookup the engine makes, in one place (``conductor._internal.graph.engine_lookups``).
        lookups = self._lookups = compiled._engine_lookups
        self._fields = lookups.fields
        self._runs_per_row_of = lookups.runs_per_row_of
        #: Per unit a node reads, ``(the writer, the depth of the row it waits at)``, in reading order.
        self._reads_wait_for = lookups.what_each_read_waits_for
        self._counted_depths = lookups.depths_finished_units_are_counted_at
        self._waiting_on_rows_made_by = lookups.nodes_finished_when_rows_made_by
        self._maker_node_of_index = lookups.node_making_rows_of_index
        self._table_holding_row_ids = lookups.table_holding_row_ids_of_index
        self._values: dict[Ref, dict[Row, Any]] = {}
        #: The fields that may hold ``SKIPPED`` at a row, or sit on an index with
        #: no rows beneath some row. Only these can have a skip above a value,
        #: so a read of any other field is one lookup at its own row.
        self._fields_with_a_skip: set[Ref] = set()
        #: Per index id, its rows so far.
        self._rows: dict[str, RowsOfIndex] = {}
        for index_id, field in lookups.field_making_rows_of_index.items():
            index = field._makes_rows or field._index
            assert index is not None  # a field owns an index
            self._rows[index_id] = RowsOfIndex(index.depth)
        for index_id, rows in self._rows.items():
            rows.child_indexes = [self._rows[child] for child in lookups.indexes_under_index[index_id]]
            for child in rows.child_indexes:
                child.parent = rows
        #: The hash of what each finished unit received, ``None`` for one not called.
        #: Kept in the order units finished, which ``state`` then has little to sort.
        self._done: dict[Unit, str | None] = {}
        #: Per (node, shorter row), how many of its units under it are done, at the depths the node has readers for.
        self._units_done_under: dict[Unit, int] = {}
        #: Units waiting on a person, each as the state carries it.
        self._waiting: dict[Unit, WaitingUnit] = {}
        #: Per thing a unit still needs, the units waiting for it. A unit is under the one thing it found first.
        self._parked: dict[Unit, list[Unit]] = {}
        #: Per node, how many rows it wasn't called on, per input that was blank there.
        self._rows_not_called: dict[str, dict[str, int]] = {}
        #: Units a person answered.
        self._answered: set[Unit] = set()
        #: Per changed node, what a later unit may still reuse.
        self.earlier_units: dict[str, _EarlierUnitsOfNode] = {}
        #: Per node, how many units this leg took from earlier work.
        self._reused_units: dict[str, int] = {}
        self._digests_of_values: DigestsOfValues = {}
        #: Where a field's cell lives when it isn't at its own row: in the list at the row above (``None``), or the column of the table there.
        self._cell_is_held_in = lookups.cell_is_held_in
        self._list_cells = lookups.cells_of_list_on_node
        self._depth_of_field = lookups.depth_of_field
        order = compiled.execution_order
        self._execution_position = {node_id: position for position, node_id in enumerate(order)}
        for ref in self._list_cells:
            index = self._fields[ref]._index
            assert index is not None
            if index.parent is None:
                self._write_list_on_node_under(ref, (), [])

    # -- what the graph holds and how a field's cell is found -------------------

    def _write_list_on_node_under(self, ref: Ref, parent_row: Row, rows_made: list[tuple[str, Row, int]]) -> None:
        """Make, under ``parent_row``, the rows of the list set on a node at ``ref``, and write it there."""
        index = self._fields[ref]._index
        assert index is not None
        cells = self._list_cells[ref]
        self._values.setdefault(ref, {})[parent_row] = cells
        self._make_rows_under(index.id, parent_row, len(cells), rows_made)

    def _make_rows_under(self, index_id: str, parent_row: Row, count: int, rows_made: list[tuple[str, Row, int]]) -> None:
        """Make these rows, and under each the rows of every list on a node whose index is this one's child."""
        self._rows[index_id].make_rows_under(parent_row, count)
        rows_made.append((index_id, parent_row, count))
        for ref in self._lookups.lists_on_nodes_under_index[index_id]:
            for position in range(count):
                self._write_list_on_node_under(ref, (*parent_row, position), rows_made)

    def _mark_no_rows_under(self, index_id: str, row: Row) -> None:
        """A skip above left this index no rows under ``row``. Everything on or under the index holds nothing there."""
        rows = self._rows[index_id]
        if not rows.no_rows_under:
            self._note_fields_with_a_skip_on(index_id)
        rows.mark_no_rows_under(row)

    def _note_fields_with_a_skip_on(self, index_id: str) -> None:
        self._fields_with_a_skip.update(self._lookups.outputs_on_index[index_id])
        for child in self._lookups.indexes_under_index[index_id]:
            self._note_fields_with_a_skip_on(child)

    # -- units ---------------------------------------------------------------

    def units(self, node_id: str) -> list[Unit]:
        """This node's units right now: one, or one per row made on the index it runs per row of."""
        index = self._runs_per_row_of[node_id]
        if index is None:
            return [(node_id, ())]
        return [(node_id, row) for row in self._rows[index.id].rows_under(())]

    def is_done(self, unit: Unit) -> bool:
        return unit in self._done

    def node_is_complete(self, node_id: str) -> bool:
        """Every unit this node will ever have is done."""
        return self._finished_at_or_under(node_id, ())

    def completed_nodes(self) -> set[str]:
        """The nodes ``node_is_complete`` holds for, among those that had a unit run or reached
        no rows through a skip. A leg starting from a state reads it so it does not announce
        ``node_start`` again for a node an earlier leg finished. A node with no units at all,
        its index sealed empty with no skip, is not in it: it never starts, so there
        is nothing to announce."""
        return {node_id for node_id in self._compiled.execution_order if self._has_run_or_skipped(node_id) and self.node_is_complete(node_id)}

    def _has_run_or_skipped(self, node_id: str) -> bool:
        index = self._runs_per_row_of[node_id]
        if index is None:
            return (node_id, ()) in self._done
        return (node_id, ()) in self._units_done_under or bool(self._rows[index.id].no_rows_under)

    def node_runs_per_row(self, node_id: str) -> bool:
        return self._runs_per_row_of[node_id] is not None

    def progress(self, node_id: str) -> tuple[int, int | None]:
        """``(done, total)`` rows for a node; ``total`` is ``None`` until all the
        rows of its index are made."""
        index = self._runs_per_row_of[node_id]
        if index is None:
            return int((node_id, ()) in self._done), 1
        maker = self._maker_node_of_index[index.id]
        total = self._rows[index.id].count_under.get((), 0) if maker is None or self._finished_at_or_under(maker, ()) else None
        return self._units_done_under.get((node_id, ()), 0), total

    # -- readiness -----------------------------------------------------------

    def first_thing_unit_still_needs(self, unit: Unit) -> Unit | None:
        """The first of what this unit reads that isn't finished, as ``(writer, row)``, or ``None`` when all is.

        The writer's units at that row finished writing, or, at a shorter row, its units
        under it, which needs that row's rows of its own index all made. The one
        rule of what makes a unit ready: nothing about a unit's readiness is decided anywhere else."""
        node_id, row = unit
        done = self._done
        for writer, depth in self._reads_wait_for[node_id]:
            needed = (writer, row[:depth])
            if needed not in done and not self._finished_at_or_under(writer, needed[1]):
                return needed
        return None

    def units_ready_to_start(self) -> list[Unit]:
        """Every unit that may start now: not done, not waiting on a person, nothing left it needs.

        Asks every unit and leaves each not ready parked on the first thing it
        needs, so the engine calls it once, where a leg starts, and each
        ``record`` after that returns whatever it released. Calling it again is safe."""
        ready: list[Unit] = []
        for node_id in self._compiled.execution_order:
            if self.node_is_complete(node_id):
                continue
            for unit in self.units(node_id):
                if unit not in self._done and unit not in self._waiting:
                    self._ask(unit, ready)
        return ready

    def _ask(self, unit: Unit, ready: list[Unit]) -> None:
        """Ask this unit what it needs: ready, or parked on that one thing."""
        needed = self.first_thing_unit_still_needs(unit)
        if needed is None:
            ready.append(unit)
        else:
            self._parked.setdefault(needed, []).append(unit)

    def _finished_at_or_under(self, node_id: str, row: Row) -> bool:
        """The units of this node at ``row`` or under it are all done: the unit at
        the row cut to the node's depth, or, under a shorter row, every row of its
        index under that row made, and each done."""
        index = self._runs_per_row_of[node_id]
        if index is None:
            return (node_id, ()) in self._done
        rows = self._rows[index.id]
        if len(row) >= rows.depth:
            return (node_id, row[: rows.depth]) in self._done
        if self._units_done_under.get((node_id, row), 0) != rows.count_under.get(row, 0):
            return False
        maker = self._maker_node_of_index[index.id]
        return maker is None or self._finished_at_or_under(maker, row)

    def _wake_if_finished(self, node_id: str, row: Row, woken: list[Unit], completed: list[str]) -> None:
        """Hand every unit parked on ``(node, row)`` to ``woken`` when that thing is finished, and
        pass on to the nodes running on the rows it made: their readers may wait on the same row."""
        if not self._finished_at_or_under(node_id, row):
            return
        parked = self._parked.pop((node_id, row), None)
        if parked is not None:
            woken.extend(parked)
        if not row and self._has_run_or_skipped(node_id):
            completed.append(node_id)
        for node_on_rows in self._waiting_on_rows_made_by[node_id]:
            self._wake_if_finished(node_on_rows, row, woken, completed)

    # -- what a unit receives -------------------------------------------------

    def inputs_for(self, unit: Unit) -> dict[str, Any] | SkippedAtRow | BlankInputsStopCall:
        """The keyword arguments this unit runs with, or what stops it.

        A ``SkippedAtRow`` comes from ``SKIPPED`` on an input received per row, on
        the series an input receives whole or reduced, or at a row above
        this unit's. A gather drops skipped sources instead, an empty
        gather being an empty series, because gathering is how "whichever
        branch fired" is expressed.
        """
        node_id = unit[0]
        values: dict[str, Any] = {}
        for name, receive_mode in self._lookups.receive_modes_of_node[node_id]:
            value = self.read_input(receive_mode, unit)
            if value.__class__ is SkippedAtRow:
                return value
            values[name] = value
        node = self._compiled.node(node_id)
        blank = tuple(name for name in node._inputs_rejecting_a_blank if values[name] is None)
        if blank:
            return BlankInputsStopCall(blank)
        if node._series_inputs_leaving_blanks_out:
            self._leave_out_blank_rows(unit, values, node._series_inputs_leaving_blanks_out)
        return values

    def read_input(self, receive_mode: ReceiveMode, unit: Unit) -> Any | SkippedAtRow:
        """What this unit receives on an input that receives ``receive_mode``, or a ``SkippedAtRow``:
        the one match over receive modes a run needs. What it waits for first is ``reads_of_input``, once, at compile."""
        node_id, row = unit
        match receive_mode:
            case Cell(sources=sources, depth=depth):
                if len(sources) == 1:
                    value, value_row = self._value_or_skip_at(sources[0], row[:depth])
                    if value is _MISSING:
                        raise KeyError(f"{sources[0]} has no value at {value_row} — the unit was not ready")
                    return SkippedAtRow(row=value_row) if value is SKIPPED else value
                covering, skips = self._values_covering_row(node_id, sources, row[:depth])
                return covering[0] if covering else SkippedAtRow(row=max(skips, key=len))
            case FixedValue(value=value):
                return value
            case RowsUnder(sources=sources, depth=depth):
                found = [self._value_or_skip_at(ref, row[:depth]) for ref in sources]
                if all(value is SKIPPED for value, _ in found):
                    return SkippedAtRow(row=max((value_row for _, value_row in found), key=len))
                return self._rows_and_values_on(node_id, sources, row[:depth])[1]
            case Gather():
                return self._gathered_for_unit(receive_mode, node_id, row)
            case Merge():
                merged = self._merged_table_for_unit(receive_mode, row)
                # Read through the input's own type when the call is checked, so a
                # merged table it won't take fails this node, naming the column.
                return SkippedAtRow(row=row) if merged is None else merged

    def _gathered_for_unit(self, receive_mode: Gather, node_id: str, row: Row) -> list[Any]:
        """The values a unit at ``row`` gathers from its sources, left out where skipped."""
        group_row = row[: receive_mode.depth]
        gathered: list[Any] = []
        for ref in receive_mode.sources:
            value, _ = self._value_or_skip_at(ref, group_row)
            if value is SKIPPED:
                continue
            index = self._fields[ref]._index
            if index is None or index.depth <= len(group_row):
                gathered.append(value)
            else:
                gathered.extend(self._rows_and_values_on(node_id, (ref,), group_row)[1])
        return gathered

    def _merged_table_for_unit(self, receive_mode: Merge, row: Row) -> pl.DataFrame | None:
        """The one table a unit at ``row`` receives from every source of a ``Table``
        input, or ``None`` when every source is skipped there. Reads each source's
        rows; ``conductor._internal.table.merge_tables_on_row_ids`` matches them. A skipped source has no rows
        and still counts as a source."""
        sources: list[MergeSource] = []
        skipped = 0
        for plan in receive_mode.source_plans:
            (source,) = plan.sources
            field = self._fields[source]
            is_table = is_table_type(field.type)
            index: Index = field._makes_rows if is_table else field._index  # type: ignore[assignment]
            group_row = row[: plan.depth]
            copied_down = source in receive_mode.copied_down
            stored, _ = self._value_or_skip_at(source, group_row)
            is_skip = stored is SKIPPED
            skipped += is_skip
            if is_table:
                frame = frame_with_no_rows(field.type) if is_skip else stored
                sources.append(MergeSource(source.node_id, frame, index, [(*group_row, position) for position in range(frame.height)], copied_down))
                continue
            present = [
                (row, value) for row in self._rows[index.id].rows_under(group_row)
                if (value := self._value_or_skip_at(source, row)[0]) is not SKIPPED and value is not _MISSING
            ]
            frame = one_column_frame_with_row_ids(field._title, field.type, [self._row_id(index, row) for row, _ in present], [value for _, value in present])
            sources.append(MergeSource(source.node_id, frame, index, [row for row, _ in present], copied_down))
        return None if skipped == len(sources) else merge_tables_on_row_ids(sources)

    def _row_id(self, index: Index, row: Row) -> str:
        """The id of the row at ``row`` on ``index``: on a table's rows, the id
        its stored table carries, which may have come in with the table; on
        any other index, the id spelled from the index and the row."""
        table = self._table_holding_row_ids.get(index.id)
        if table is None:
            return global_row_id(index, row)
        stored, _ = self._value_or_skip_at(table, row[:-1])
        return stored[ROW_ID_COLUMN][row[-1]]

    def _values_covering_row(self, node_id: str, refs: tuple[Ref, ...], row: Row) -> tuple[list[Any], list[Row]]:
        """Of several refs read at one row: the values that cover it, and the
        rows at which the others were skipped. Two covering the same row
        means two edges both supplied that row, and the node fails with
        that row, named in the failure's ``ctx``."""
        covering: list[Any] = []
        skips: list[Row] = []
        for ref in refs:
            value, value_row = self._value_or_skip_at(ref, row)
            if value is _MISSING:
                raise KeyError(f"{ref} has no value at {value_row} — the unit was not ready")
            if value is SKIPPED:
                skips.append(value_row)
            else:
                covering.append(value)
        if len(covering) > 1:
            raise NodeError(PROBLEM_CATALOGUE["row_covered_twice"].message, node_id=node_id, code="row_covered_twice", ctx={"edges": len(covering), "row": list(row)})
        return covering, skips

    def _value_or_skip_at(self, ref: Ref, row: Row) -> tuple[Any, Row]:
        """``(value, row)`` for the value at ``(ref, row)``: ``SKIPPED`` at the
        shortest prefix of ``row`` the field is skipped at or has no rows under,
        else what is at ``row`` (``_MISSING`` when nothing is): a cell held inside a
        list or a table at the row above is read off it by position."""
        if ref in self._fields_with_a_skip:
            skip_row = self._row_a_skip_reaches(ref, row)
            if skip_row is not None:
                return SKIPPED, skip_row
        where = self._cell_is_held_in.get(ref, _AT_ITS_OWN_ROW)
        if where is _AT_ITS_OWN_ROW:
            return self._values.get(ref, _NO_VALUES).get(row, _MISSING), row
        if where is None:
            return self._list_cell(ref, row)
        return self._table_column_cell(ref, where, row)

    def _row_a_skip_reaches(self, ref: Ref, row: Row) -> Row | None:
        """The shortest prefix of ``row`` where this field holds nothing because of a skip, if any."""
        by_row = self._values.get(ref, _NO_VALUES)
        index = self._fields[ref]._index
        no_rows_under = self._rows[index.id].no_rows_under if index is not None else ()
        for depth in range(len(row) + 1):
            row_on_path = row[:depth]
            if by_row.get(row_on_path) is SKIPPED or row_on_path in no_rows_under:
                return row_on_path
        return None

    def _list_cell(self, ref: Ref, row: Row) -> tuple[Any, Row]:
        """The value at ``row`` of a row-making output or a list on a node: by position in the list at the row above."""
        if len(row) < self._depth_of_field[ref]:
            return _MISSING, row
        cells = self._values.get(ref, _NO_VALUES).get(row[:-1], _MISSING)
        if cells is _MISSING or row[-1] >= len(cells):
            return _MISSING, row
        return cells[row[-1]], row

    def _table_column_cell(self, ref: Ref, column: str, row: Row) -> tuple[Any, Row]:
        """The value of a table's column at ``row``: the cell in row ``row[-1]``
        of the table stored at ``row[:-1]``, as its column's type, a blank
        cell ``None``."""
        if len(row) < self._depth_of_field[ref]:
            return _MISSING, row
        table = self._fields[ref]._column_of_table
        assert table is not None
        stored = self._values.get(table[0], _NO_VALUES).get(row[:-1], _MISSING)
        if stored is _MISSING:
            return _MISSING, row
        cells = stored[column]
        if row[-1] >= len(cells):
            return _MISSING, row
        cell = cells[row[-1]]
        return (None if cell is None else self._fields[ref].type(cell)), row

    def _rows_and_values_on(self, node_id: str, refs: tuple[Ref, ...], under: Row) -> tuple[list[Row], list[Any]]:
        """The rows of ``refs``' index under ``under`` that hold a value and those values,
        from one ref or from whichever of several covers each row."""
        index = self._fields[refs[0]]._index
        assert index is not None  # what is read as rows has an index
        rows: list[Row] = []
        values: list[Any] = []
        for row in self._rows[index.id].rows_under(under):
            covering, _ = self._values_covering_row(node_id, refs, row)
            if covering:
                rows.append(row)
                values.append(covering[0])
        return rows, values

    def _leave_out_blank_rows(self, unit: Unit, values: dict[str, Any], names: tuple[str, ...]) -> None:
        """Leave a row out of every input reading an index when one of ``names`` is blank there.

        ``names`` are the inputs of many values that take no blank. A row leaves
        all of them on its index, or a node zipping them would pair wrong values.
        An input gathered or set on the node has rows of its own, so its blanks leave only it."""
        if not any(cell is None for name in names for cell in values[name]):  # by identity: a frame refuses ``==``
            return
        node_id, row = unit
        read_as_rows: dict[str, tuple[Index, list[Row], list[Any]]] = {}
        for name, receive_mode in self._lookups.receive_modes_of_node[node_id]:
            if isinstance(receive_mode, RowsUnder):
                index = self._fields[receive_mode.sources[0]]._index
                assert index is not None  # what is read as rows has an index
                read_as_rows[name] = (index, *self._rows_and_values_on(node_id, receive_mode.sources, row[: receive_mode.depth]))
        blank_rows: dict[Index, set[Row]] = {}
        for name in names:
            if name in read_as_rows:
                index, rows, cells = read_as_rows[name]
                blank_rows.setdefault(index, set()).update(on_row for on_row, cell in zip(rows, cells, strict=True) if cell is None)
            else:
                values[name] = [cell for cell in values[name] if cell is not None]
        for name, (index, rows, cells) in read_as_rows.items():
            if blank_rows.get(index):
                values[name] = [cell for on_row, cell in zip(rows, cells, strict=True) if on_row not in blank_rows[index]]

    # -- recording -----------------------------------------------------------------

    def record(self, unit: Unit, outputs: dict[str, Any] | SkippedAtRow | BlankInputsStopCall, input_hash: str | None = None) -> UnitRecorded:
        """Record what this unit produced, and hand back what that makes ready.

        ``outputs`` are cells that passed their outputs' checks, per output: one
        cell, ``SKIPPED``, a list of cells on an output that makes rows, or a
        table; or a ``SkippedAtRow`` or a ``BlankInputsStopCall`` when the unit didn't run,
        its outputs ``SKIPPED`` at that row or blank (``_blank_of``). An output that makes
        rows makes them under the unit's row, one per list item or table
        row, each child of ``row``; a skipped one leaves its index with no rows
        under the row, an empty one leaves an empty group, so a reduction over
        it still runs. A unit may finish with a skip reaching rows above it; such a
        reduction may then have run first on an empty group, and what it
        returned is hidden by the skip, so a run's results don't depend on
        which came first, though the number of calls can.

        ``input_hash`` is what the unit received, which a later leg reuses by;
        ``None`` for a unit that wasn't called.

        The units returned are the ones whose first need this write finished,
        asked again; the engine starts those and asks nothing else.
        """
        node_id, row = unit
        node = self._compiled.node(node_id)
        assert node.wiring is not None  # the engine runs only a runnable graph
        fields = node.wiring.fields
        rows_made: list[tuple[str, Row, int]] = []
        skip_row = outputs.row if isinstance(outputs, SkippedAtRow) else None
        not_called = outputs if isinstance(outputs, BlankInputsStopCall) else None
        if not_called is not None:
            counts = self._rows_not_called.setdefault(node_id, {})
            for name in not_called.blank_inputs:
                counts[name] = counts.get(name, 0) + 1
        for out in self._lookups.interfaces[node_id].outputs:
            field = fields[out.name]
            if skip_row is not None:
                value = SKIPPED
            elif not_called is not None:
                value = _blank_of(field, out)
            else:
                value = outputs[out.name]  # type: ignore[index]
            write_row = row if skip_row is None else skip_row
            self._values.setdefault(field.ref, {})[write_row] = value
            self._note_rows_made_by_write(field, write_row, value, rows_made)
        if self._waiting:
            self._waiting.pop(unit, None)
        self._mark_done(unit, input_hash)
        return self._what_this_write_makes_ready(unit, rows_made)

    def rows_not_called(self, node_id: str) -> dict[str, int]:
        """How many rows this node wasn't called on, per input that was blank there."""
        return self._rows_not_called.get(node_id, {})

    def _note_rows_made_by_write(self, field: CompiledField, row: Row, value: Any, rows_made: list[tuple[str, Row, int]]) -> None:
        """What a value written at ``row`` does to the rows: ``SKIPPED`` leaves its field's
        index with nothing there and, on a row-making output, under it; a list
        or a table makes one row per item."""
        index = field._makes_rows
        if value is SKIPPED:
            self._fields_with_a_skip.add(field.ref)
            if index is not None:
                self._mark_no_rows_under(index.id, row)
        elif index is not None:
            self._make_rows_under(index.id, row, len(value), rows_made)

    def _mark_done(self, unit: Unit, input_hash: str | None) -> None:
        """Mark the unit done, and count it under each shorter row its node has readers at."""
        self._done[unit] = input_hash
        node_id, row = unit
        if self._runs_per_row_of[node_id] is None:
            return
        counts = self._units_done_under
        for depth in self._counted_depths[node_id]:
            key = (node_id, row[:depth])
            counts[key] = counts.get(key, 0) + 1

    def _what_this_write_makes_ready(self, unit: Unit, rows_made: list[tuple[str, Row, int]]) -> UnitRecorded:
        """Wake what this unit finished, ask those again, and ask the units on the rows it made."""
        node_id, row = unit
        woken = list(self._parked.pop(unit, ()))
        completed: list[str] = []
        for node_on_rows in self._waiting_on_rows_made_by[node_id]:
            self._wake_if_finished(node_on_rows, row, woken, completed)
        index = self._runs_per_row_of[node_id]
        if index is None:
            completed.append(node_id)
        else:
            counts, rows = self._units_done_under, self._rows[index.id]
            for depth in self._counted_depths[node_id]:
                group_row = row[:depth]
                if counts[(node_id, group_row)] == rows.count_under.get(group_row, 0):
                    self._wake_if_finished(node_id, group_row, woken, completed)
        ready: list[Unit] = []
        for parked_unit in woken:
            if parked_unit not in self._done and parked_unit not in self._waiting:
                self._ask(parked_unit, ready)
        for index_id, parent_row, count in rows_made:
            readers = self._lookups.nodes_running_per_row_of_index[index_id]
            if readers:
                for position in range(count):
                    new_row = (*parent_row, position)
                    for reader in readers:
                        self._ask((reader, new_row), ready)
        position_of = self._execution_position
        if len(ready) > 1:
            ready.sort(key=lambda ready_unit: (position_of[ready_unit[0]], ready_unit[1]))
        if len(completed) > 1:
            completed.sort(key=position_of.__getitem__)
        return UnitRecorded(ready, completed)

    # -- waiting on a person ---------------------------------------------

    def mark_waiting(self, unit: Unit, waiting_unit: WaitingUnit) -> None:
        """Hold this unit as waiting: its node asked a person. It and whatever reads it wait."""
        self._waiting[unit] = waiting_unit

    def is_waiting(self, unit: Unit) -> bool:
        return unit in self._waiting

    def mark_answered(self, unit: Unit) -> None:
        """Note that a person answered this unit: an answer is only taken over by its own row
        when the inputs alone can't say which of several rows it belongs to."""
        self._answered.add(unit)

    def waiting(self) -> tuple[WaitingUnit, ...]:
        """Every unit waiting on a person, in execution order and row order
        within a node: not the order their concurrent calls happened to ask in."""
        position_of = self._execution_position
        return tuple(waiting_unit for _, waiting_unit in sorted(self._waiting.items(), key=lambda item: (position_of[item[0][0]], item[0][1])))

    # -- work from an earlier state ----------------------------------------------------

    def input_hash_of(self, seed: bytes, inputs: dict[str, Any]) -> str:
        """The hash of what a unit of the node seeded by ``seed`` is about to receive (``execution.input_hash``)."""
        return hash_of_inputs(seed, inputs, self._digests_of_values)

    def claim_earlier_result(self, unit: Unit, input_hash: str) -> dict[str, Any] | WaitingUnit | None:
        """An earlier result of this node for a unit that received ``input_hash``, taken so no
        other unit can have it: the cells to ``record`` at this unit's row, or, for
        one that was waiting on a person, that unit moved to this row, or ``None``
        when there is none and the unit must run."""
        node_id, row = unit
        earlier = self.earlier_units.get(node_id)
        old_row = None if earlier is None else earlier.claim(row, input_hash)
        if earlier is None or old_row is None:
            return None
        waiting_unit = earlier.waiting_at_row.get(old_row)
        if waiting_unit is not None:
            self._reused_units[node_id] = self._reused_units.get(node_id, 0) + 1
            return waiting_unit if old_row == row else waiting_unit.model_copy(update={"row": row, "id": id_of_waiting_unit(node_id, row, input_hash)})
        cells = self._cells_of_earlier_unit(earlier, node_id, old_row, row)
        if cells is None:
            return None
        self._reused_units[node_id] = self._reused_units.get(node_id, 0) + 1
        if old_row in earlier.answered_rows:
            self._answered.add(unit)
        return cells

    def _cells_of_earlier_unit(self, earlier: _EarlierUnitsOfNode, node_id: str, old_row: Row, row: Row) -> dict[str, Any] | None:
        """What an earlier unit wrote, one value per output as ``record`` takes it, ids its tables minted
        re-spelled at ``row``; ``None`` when the state has lost a value, and the unit runs."""
        cells: dict[str, Any] = {}
        for out in self._lookups.interfaces[node_id].outputs:
            ref = Ref(node_id, out.name)
            value = earlier.values.get(ref, _NO_VALUES).get(old_row, _MISSING)
            if value is _MISSING:
                return None
            makes_rows = self._fields[ref]._makes_rows
            if makes_rows is not None and old_row != row and is_frame(value):
                value = with_ids_spelled_at_row(value, makes_rows, old_row, row)
            cells[out.name] = value
        return cells

    def reused_units(self, node_id: str) -> int:
        return self._reused_units.get(node_id, 0)

    def _earlier_work_still_unclaimed(self) -> EarlierWork | None:
        """What is left of the earlier work of the nodes this leg hasn't finished, for a leg going on from this one."""
        left = {node_id: earlier for node_id, earlier in self.earlier_units.items() if earlier.hash_of_row and not self.node_is_complete(node_id)}
        if not left:
            return None
        finished: dict[str, list[Row]] = {}
        hashes: dict[str, list[str | None]] = {}
        answered: dict[str, list[Row]] = {}
        columns: dict[Ref, OutputColumn] = {}
        waiting: list[WaitingUnit] = []
        for node_id, earlier in left.items():
            rows = sorted(row for row in earlier.hash_of_row if row not in earlier.waiting_at_row)
            # Written apart from any row a unit will have, so a later leg's pool can hold them beside this leg's own work.
            written_at = {row: (_ROWS_OF_EARLIER_WORK_START + position, *row[1:]) if row else row for position, row in enumerate(sorted(earlier.hash_of_row))}
            if rows:
                finished[node_id] = [written_at[row] for row in rows]
                hashes[node_id] = [earlier.hash_of_row[row] for row in rows]
                if kept := sorted(written_at[row] for row in earlier.answered_rows & set(rows)):
                    answered[node_id] = kept
            waiting += [unit.model_copy(update={"row": written_at[row]}) for row, unit in sorted(earlier.waiting_at_row.items()) if row in earlier.hash_of_row]
            for ref, by_row in earlier.values.items():
                columns[ref] = _output_column_of({written_at[row]: by_row[row] for row in rows if row in by_row}, is_many_cells=self._fields[ref]._unit_writes_many_cells)
        return EarlierWork.model_construct(finished_rows=finished, input_hashes=hashes, answered_rows=answered, output_columns=columns, waiting=tuple(waiting))

    # -- what the run produced -------------------------------------------------------

    def result_of(self, node_id: str) -> dict[str, Any] | None:
        """This node's outputs by name, or ``None`` when the node did not run
        (every output is skipped at the top).

        A result holds only what the node produced: an output the node
        skipped (the branch not taken) is absent, never ``SKIPPED``. A scalar
        output of a node that ran once is its value; anything on an index is
        ``ValuesOnRows``, without the rows that were skipped.
        """
        return node_result(self._lookups, node_id, self._values)

    # -- the state ----------------------------------------------------------------

    def state(self, status: RunStatus, failure: Problem | None = None) -> RunState:
        """Everything this leg has produced, as the ``RunState`` a host stores.

        Built without validation from values that already passed, and holding
        them typed: the next read costs nothing to decode, and a dump writes
        each one by its own type."""
        columns: dict[Ref, OutputColumn] = {}
        for ref in sorted(self._values):
            field = self._fields[ref]
            if field._is_list_on_node:  # the graph holds a list set on a node, never a state
                continue
            columns[ref] = _output_column_of(self._values[ref], is_many_cells=field._unit_writes_many_cells)
        compiled = self._compiled
        order = compiled.execution_order
        finished_by_node: dict[str, list[Row]] = {}
        for node_id, row in self._done:
            finished_by_node.setdefault(node_id, []).append(row)
        finished_rows = {node_id: sorted(finished_by_node.get(node_id, ())) for node_id in order if node_id in finished_by_node or self.node_is_complete(node_id)}
        input_hash_of = self._done
        answered: dict[str, list[Row]] = {}
        for node_id, row in sorted(self._answered):
            answered.setdefault(node_id, []).append(row)
        return RunState.model_construct(
            status=status,
            failure=failure,
            waiting=self.waiting(),
            output_columns=columns,
            # A node complete on no rows is here with none: the state says it finished.
            finished_rows=finished_rows,
            input_hashes={node_id: [input_hash_of[(node_id, row)] for row in rows] for node_id, rows in finished_rows.items() if rows},
            answered_rows=answered,
            reused_units=dict(self._reused_units),
            earlier_work=self._earlier_work_still_unclaimed() if self.earlier_units else None,
            node_fingerprints={node_id: compiled.node(node_id).fingerprint for node_id in order},
            rows_not_called={node_id: dict(sorted(self._rows_not_called[node_id].items())) for node_id in order if node_id in self._rows_not_called},
        )

    @classmethod
    def restore(cls, compiled: CompiledGraph, state: RunState) -> RunLedger:
        """A ledger holding what an earlier leg recorded, over the graph being run.

        Every node ``compiled.nodes_changed_since(state)`` names is re-planned:
        its units run again, each one reusing an earlier result whose inputs
        matched, what the state holds of them read here; a node gone from the
        graph drops with them. The rest is restored, each value through its
        output's check, and every
        finished unit is played back through the writes ``record`` makes, in
        execution order, which makes the rows and the skips again. The rows of
        a list on a node come from the graph being run, as for a fresh run.

        A stored value that does not read as its output's type, a value or a
        unit at a row its field or node has no place for, and a value for a
        field the graph does not have are all reported at once, as
        ``InvalidCallError`` with the code ``state_unreadable``, the columns'
        problems first and then the units'. A ``None`` passes on any output: a
        blank is a legal value in a state.
        """
        restored = cls(compiled)
        changed = set(compiled.nodes_changed_since(state))
        values, problems = read_output_columns(compiled, state, changed)
        if problems:
            raise InvalidCallError(problems)
        for ref, by_row in values.items():
            restored._values.setdefault(ref, {}).update(by_row)
        restored._waiting = {(waiting_unit.node_id, waiting_unit.row): waiting_unit for waiting_unit in state.waiting if waiting_unit.node_id not in changed}
        restored._rows_not_called = {node_id: dict(counts) for node_id, counts in state.rows_not_called.items() if node_id not in changed}
        restored.earlier_units = _earlier_units_of_changed_nodes(compiled, state, changed)
        finished = {node_id: rows for node_id, rows in state.finished_rows.items() if node_id not in changed}
        problems = restored._replay_finished_units(finished, state.input_hashes)
        problems.extend(restored._problems_with_waiting_units())
        if problems:
            raise InvalidCallError(problems)
        restored._answered = {(node_id, row) for node_id, rows in state.answered_rows.items() if node_id not in changed for row in rows}
        return restored

    def _replay_finished_units(self, finished: Mapping[str, list[Row]], input_hashes: Mapping[str, list[str | None]]) -> list[Problem]:
        """Write every finished unit, in execution order, as its record did, and
        report one at a row that doesn't exist or with no value at its row, or a
        node whose hashes don't line up with its rows."""
        problems: list[Problem] = []
        interfaces = self._lookups.interfaces
        for node_id in self._compiled.execution_order:
            rows_of_node = finished.get(node_id)
            if rows_of_node is None:
                continue
            index = self._runs_per_row_of[node_id]
            hashes = input_hashes.get(node_id)
            if hashes is None:
                hashes = [None] * len(rows_of_node)
            elif len(hashes) != len(rows_of_node):
                problems.append(make_problem("state_unreadable", "state", node_id, reason=f"the state has {len(rows_of_node)} finished rows for {node_id!r} and {len(hashes)} input hashes"))
                continue
            for row, input_hash in sorted(zip(rows_of_node, hashes, strict=True), key=itemgetter(0)):
                if not self._row_exists_for_node(index, row):
                    problems.append(make_problem("state_unreadable", "state", node_id, *row, reason=f"the state has {node_id!r} at row {list(row)}, which {node_id!r} has no place for"))
                    continue
                for out in interfaces[node_id].outputs:
                    field = self._fields[Ref(node_id, out.name)]
                    value, write_row = _stored_value_of_unit(self._values.get(field.ref, _NO_VALUES), row)
                    if value is _MISSING:
                        problems.append(make_problem("state_unreadable", "state", field.ref, *row, reason=f"the state has {node_id!r} finished at row {list(row)}, with no value for {field.ref}"))
                        continue
                    self._note_rows_made_by_write(field, write_row, value, [])
                self._mark_done((node_id, row), input_hash)
        return problems

    def _row_exists_for_node(self, index: Index | None, row: Row) -> bool:
        if index is None:
            return row == ()
        return len(row) == self._rows[index.id].depth and self._rows[index.id].has_row(row)

    def _problems_with_waiting_units(self) -> list[Problem]:
        """A waiting unit on a row that was never made."""
        problems: list[Problem] = []
        for node_id, row in self._waiting:
            if not self._row_exists_for_node(self._runs_per_row_of[node_id], row):
                problems.append(make_problem("state_unreadable", "state", node_id, *row, reason=f"the state has {node_id!r} at row {list(row)}, which {node_id!r} has no place for"))
        return problems


class _EarlierUnitsOfNode:
    """The units a changed node finished or had waiting in the state a leg starts from, held to be claimed.

    Unclaimed units by row, the hash of what each received; ``claim`` hands
    each out once. A claim takes the unit's own old row when that row received
    the same, else the lowest unclaimed row with the hash: identical inputs
    are a multiset, so a third identical row gets a new call and never shares
    a result. A hash answered by a person on more than one row is taken only
    by its own row: the inputs can't say whose answer it was, and a person's
    answer has no second chance.
    """

    __slots__ = ("hash_of_row", "rows_with_hash_lowest_last", "hashes_whose_answers_can_not_be_told_apart", "answered_rows", "waiting_at_row", "values")

    def __init__(self, hash_of_row: dict[Row, str], answered_rows: set[Row], waiting_at_row: dict[Row, WaitingUnit], values: dict[Ref, dict[Row, Any]]) -> None:
        self.hash_of_row = hash_of_row
        self.answered_rows = answered_rows
        self.waiting_at_row = waiting_at_row
        self.values = values
        self.rows_with_hash_lowest_last: dict[str, list[Row]] = {}
        for row in sorted(hash_of_row, reverse=True):
            self.rows_with_hash_lowest_last.setdefault(hash_of_row[row], []).append(row)
        tied = (input_hash for input_hash, rows in self.rows_with_hash_lowest_last.items() if len(rows) > 1 and not answered_rows.isdisjoint(rows))
        self.hashes_whose_answers_can_not_be_told_apart = frozenset(tied)

    def claim(self, row: Row, input_hash: str) -> Row | None:
        """Take the earlier unit this unit's inputs match, gone from the pool, or ``None``."""
        if self.hash_of_row.get(row) == input_hash:
            del self.hash_of_row[row]
            return row
        rows = self.rows_with_hash_lowest_last.get(input_hash)
        if rows is None or input_hash in self.hashes_whose_answers_can_not_be_told_apart:
            return None
        while rows:
            candidate = rows.pop()
            if self.hash_of_row.get(candidate) == input_hash:  # not taken yet, or by its own row
                del self.hash_of_row[candidate]
                return candidate
        return None


def _earlier_units_of_changed_nodes(compiled: CompiledGraph, state: RunState, changed: set[str]) -> dict[str, _EarlierUnitsOfNode]:
    """Per node to re-plan and per node with work carried over from a leg that stopped short, what it can reuse."""
    earlier_work = state.earlier_work
    held_in: list[tuple[RunState | EarlierWork, set[str]]] = [(state, changed)]
    if earlier_work is not None:
        held_in.append((earlier_work, set(earlier_work.finished_rows) | {unit.node_id for unit in earlier_work.waiting}))
    units: dict[str, _EarlierUnitsOfNode] = {}
    for holder, node_ids in held_in:
        for node_id in node_ids:
            read = _earlier_units_in(compiled, holder, node_id)
            if read is None:
                continue
            already = units.get(node_id)
            if already is None:
                units[node_id] = read
            else:  # work carried over from a leg that stopped short sits at rows of its own
                for ref, by_row in read.values.items():
                    already.values[ref] = {**by_row, **already.values.get(ref, {})}
                units[node_id] = _EarlierUnitsOfNode({**read.hash_of_row, **already.hash_of_row}, already.answered_rows | read.answered_rows, {**read.waiting_at_row, **already.waiting_at_row}, already.values)
    return units


def _earlier_units_in(compiled: CompiledGraph, holder: RunState | EarlierWork, node_id: str) -> _EarlierUnitsOfNode | None:
    """What ``holder`` keeps of a re-planned node's units, or ``None`` when it can't reuse any: gone from the
    graph, a node that's a graph, opted out, nothing with a hash, or outputs that no longer read as its types."""
    node = compiled._nodes.get(node_id)
    if node is None or not isinstance(node.version, NodeVersion) or node.interface is None or not node.version.policy.reuse_when_inputs_match:
        return None
    hash_of_row: dict[Row, str] = {row: input_hash for row, input_hash in zip(holder.finished_rows.get(node_id, ()), holder.input_hashes.get(node_id, ()), strict=False) if input_hash is not None}
    waiting_at_row = {unit.row: unit for unit in holder.waiting if unit.node_id == node_id and unit.input_hash}
    hash_of_row.update((row, unit.input_hash) for row, unit in waiting_at_row.items())
    if not hash_of_row:
        return None
    values: dict[Ref, dict[Row, Any]] = {}
    problems: list[Problem] = []
    for out in node.interface.outputs:
        ref = Ref(node_id, out.name)
        column = holder.output_columns.get(ref)
        read = None if column is None else _read_output_column(compiled, ref, column, problems)
        if read is not None:
            values[ref] = read
    return None if problems else _EarlierUnitsOfNode(hash_of_row, set(holder.answered_rows.get(node_id, ())), waiting_at_row, values)


def _stored_value_of_unit(by_row: Mapping[Row, Any], row: Row) -> tuple[Any, Row]:
    """What a finished unit at ``row`` wrote: the value at its row, or a skip at a shorter one, else missing."""
    value = by_row.get(row, _MISSING)
    if value is not _MISSING:
        return value, row
    for depth in range(len(row)):
        if by_row.get(row[:depth]) is SKIPPED:
            return SKIPPED, row[:depth]
    return _MISSING, row


def _blank_of(field: CompiledField, out: Output) -> Any:
    """What a node that was not called writes on an output: a ``choice`` output
    is skipped, since no branch was chosen; an output that makes rows has none
    there; any other is blank."""
    if out.choice is not None:
        return SKIPPED
    if field._unit_writes_many_cells:
        return []
    if field._makes_rows is not None:
        return frame_with_no_rows(field.type)
    return None



def _output_column_of(by_row: Mapping[Row, Any], is_many_cells: bool) -> OutputColumn:
    """One output as a state writes it. On an output that makes rows, a row's
    value is the list of cells under it; a cell skipped alone is listed in ``skipped`` and a ``None`` stands in its place."""
    rows: list[Row] = []
    values: list[Any] = []
    skipped: list[Row] = []
    for row, value in sorted(by_row.items(), key=itemgetter(0)):
        if value is SKIPPED:
            skipped.append(row)
            continue
        rows.append(row)
        if is_many_cells:
            holes = [position for position, cell in enumerate(value) if cell is SKIPPED]
            if holes:
                value = list(value)
                for position in holes:
                    value[position] = None
                    skipped.append((*row, position))
        values.append(value)
    skipped.sort()
    return OutputColumn.model_construct(rows=rows, values=values, skipped=skipped)


def read_output_columns(compiled: CompiledGraph, state: RunState, dropped: set[str]) -> tuple[dict[Ref, dict[Row, Any]], list[Problem]]:
    """What ``state`` holds on every output of the graph, by row, each value
    read through its output's check, and every problem found.

    The one reader of ``state.output_columns``: a leg restores from it and
    ``compiled.results`` reads from it. ``dropped`` (``nodes_changed_since``)
    is left out."""
    values: dict[Ref, dict[Row, Any]] = {}
    problems: list[Problem] = []
    for ref, column in state.output_columns.items():
        if ref.node_id not in dropped:
            read = _read_output_column(compiled, ref, column, problems)
            if read is not None:
                values[ref] = read
    return values, problems


def _read_output_column(compiled: CompiledGraph, ref: Ref, column: OutputColumn, problems: list[Problem]) -> dict[Row, Any] | None:
    """One output's stored rows and cells, or ``None`` and the problems that stop them."""
    try:
        field = compiled.field(ref)
    except KeyError:
        problems.append(make_problem("state_unreadable", "state", ref, reason=f"the state has a value for {ref}, which the graph does not have"))
        return None
    depth = 0 if field._index is None else field._index.depth
    many_cells = field._unit_writes_many_cells
    depth_of_written_row = depth - 1 if many_cells else depth
    stored_rows = (*column.rows, *column.skipped)
    misplaced: list[Row] = []
    if max(map(len, column.rows), default=0) > depth_of_written_row or max(map(len, column.skipped), default=0) > depth or min(chain.from_iterable(stored_rows), default=0) < 0:
        misplaced = [row for row in column.rows if len(row) > depth_of_written_row or min(row, default=0) < 0]
        misplaced += [row for row in column.skipped if len(row) > depth or min(row, default=0) < 0]
    for row in misplaced:
        problems.append(make_problem("state_unreadable", "state", ref, *row, reason=f"the state's value for {ref} is at row {list(row)}, which {ref} has no place for"))
    if misplaced:
        return None
    by_row: dict[Row, Any] = {}
    read = field._validate_cell
    cell_type = field.type
    for row, stored in zip(column.rows, column.values, strict=True):
        try:
            by_row[row] = _restored_list_of_cells(field, stored, read) if many_cells else stored if stored is None or type(stored) is cell_type else _restored_cell(field, stored, read)
        except (TypeError, ValueError) as unreadable:
            problems.append(make_problem("state_unreadable", "state", ref, *row, reason=f"the state's value for {ref} at row {list(row)} does not read back: {_read_failure_message(unreadable)}"))
    for row in column.skipped:
        if many_cells and len(row) == depth:
            cells = by_row.get(row[:-1])
            if not isinstance(cells, list) or row[-1] >= len(cells):
                problems.append(make_problem("state_unreadable", "state", ref, *row, reason=f"the state's skip for {ref} at row {list(row)} has no list to be in"))
                continue
            cells[row[-1]] = SKIPPED
        else:
            by_row[row] = SKIPPED
    return by_row


def _restored_list_of_cells(field: CompiledField, stored: Any, read: Any) -> list[Any]:
    """The cells a many-valued output wrote at one row, each read as its output's cell."""
    if not isinstance(stored, list):
        raise TypeError(f"a list of values, not a {type(stored).__name__}")
    cell_type = field.type
    return [cell if cell is None or type(cell) is cell_type else _restored_cell(field, cell, read) for cell in stored]


def _restored_cell(field: CompiledField, stored: Any, read: Any) -> Any:
    """A stored value not already of its type as its cell, through the output's check:
    one of its type came from a leg in this process, and a ``None`` is a blank, so
    the caller keeps those as they are. A table has to have kept its row ids."""
    cell = read(stored)
    if field._makes_rows is not None and not field._unit_writes_many_cells and ROW_ID_COLUMN not in cell.columns:
        raise ValueError("a stored table has lost the ids of its rows")
    return cell


def node_result(lookups: EngineLookups, node_id: str, values: Mapping[Ref, Mapping[Row, Any]]) -> dict[str, Any] | None:
    """A node's outputs by name, read off what each holds, or ``None`` when
    the node did not run (every output is skipped at the top).

    A result holds only what the node produced: an output the node skipped
    (the branch not taken) is absent, never ``SKIPPED``. A scalar output of a
    node that ran once is its value; anything on an index is ``ValuesOnRows``,
    the rows written so far and a value each, without the rows that were skipped."""
    outputs = lookups.interfaces[node_id].outputs
    by_output = {out.name: values.get(Ref(node_id, out.name), _NO_VALUES) for out in outputs}
    skipped_at_top = {name: _skipped_at_the_top(lookups, Ref(node_id, name), values) for name in by_output}
    if outputs and all(skipped_at_top.values()):
        return None
    result: dict[str, Any] = {}
    for name, by_row in by_output.items():
        if skipped_at_top[name]:
            continue
        field = lookups.fields[Ref(node_id, name)]
        if field._index is not None:
            rows, cells = _rows_and_cells_with_a_value(by_row, field._unit_writes_many_cells)
            result[name] = ValuesOnRows(rows=rows, values=cells)
        elif (at_top := by_row.get((), _MISSING)) is not _MISSING:
            result[name] = at_top
    return result


def _skipped_at_the_top(lookups: EngineLookups, ref: Ref, values: Mapping[Ref, Mapping[Row, Any]]) -> bool:
    """``SKIPPED`` at ``()`` on this output, or an index above it that got no rows at all from a skip."""
    if values.get(ref, _NO_VALUES).get(()) is SKIPPED:
        return True
    index = lookups.fields[ref]._index
    while index is not None:
        maker = lookups.field_making_rows_of_index[index.id]
        if not maker._is_list_on_node and values.get(maker.ref, _NO_VALUES).get(()) is SKIPPED:
            return True
        index = index.parent
    return False


def _rows_and_cells_with_a_value(by_row: Mapping[Row, Any], is_many_cells: bool) -> tuple[list[Row], list[Any]]:
    """A field's rows and cells in order, leaving out skipped rows and those under a skip."""
    rows = sorted(by_row)
    skips = {row for row, value in by_row.items() if value is SKIPPED}
    if skips:
        rows = [row for row in rows if row not in skips and not any(row[:depth] in skips for depth in range(len(row)))]
    if not is_many_cells:
        return rows, [by_row[row] for row in rows]
    cell_rows: list[Row] = []
    cells: list[Any] = []
    for row in rows:
        for position, cell in enumerate(by_row[row]):
            if cell is not SKIPPED:
                cell_rows.append((*row, position))
                cells.append(cell)
    return cell_rows, cells


def _read_failure_message(unreadable: Exception) -> str:
    """Why a value did not read: pydantic's first message when pydantic said, never the value."""
    if isinstance(unreadable, ValidationError):
        return unreadable.errors(include_url=False, include_input=False, include_context=False)[0]["msg"]
    return str(unreadable)
