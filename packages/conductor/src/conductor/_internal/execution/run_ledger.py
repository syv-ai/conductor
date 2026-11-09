"""``RunLedger``: what a run has produced so far, and what that lets run next.

The engine runs a graph as units. A unit is one node on one row. A node
that runs once is a single unit on the empty row ``()``; a node fed a series on
a scalar input runs once per row of that series' index, one unit per row.
A row is a path, ``(i,)`` on a root index and ``(i, j)`` for the j-th row
made under ``(i,)``, so a unit deep in the tree finds its row on a parent
index by cutting its path short. Rows are made when the unit producing them
writes; until then nobody knows how many there will be.

The ledger keeps what each output holds through ``ValuesOfRun``, the module
that knows how a field's values are held; every index's rows are kept there
beside them (``RowsOfIndex``). From those two the ledger answers the engine's questions:
what does this unit still need (``first_thing_unit_still_needs``), and what
becomes of it when its turn comes (``start_unit``). The engine is a loop over
those calls and nothing in here is asynchronous.

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
just returned it (``start_unit``). Nothing maps an old row to a new one. Units
with the same inputs are taken in row order, the k-th new one the k-th old one,
and a table it returned takes on the ids of the rows it received now.

A person's answer belongs to more than the inputs. A question is bound to the
row it was asked on: the row read whole, the rows above it, the row its id came
from, and which of how many identical rows it is (``_identity_of_row_for_a_question``). The unit's
``reuse_key`` folds that in, so only the same row can take the answer, or the
question while it waits, and a form opened for another row is refused.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from hashlib import blake2b
from itertools import chain, groupby
from operator import attrgetter, itemgetter
from typing import TYPE_CHECKING, Any, NamedTuple

from conductor._internal.errors import (
    PROBLEM_CATALOGUE,
    InvalidCallError,
    NodeError,
    Problem,
    make_problem,
)
from conductor._internal.execution.events import NodeComplete, NodeSkipped
from conductor._internal.execution.input_hash import (
    DigestsOfValues,
    hash_of_inputs,
    hash_of_row_cells,
)
from conductor._internal.execution.run_state import (
    EarlierUnit,
    FinishedUnit,
    RunState,
    RunStatus,
    WaitingUnit,
    WorkOfNode,
    id_of_waiting_unit,
)
from conductor._internal.execution.values_of_run import (
    MISSING,
    TableAtRow,
    ValuesOfRun,
    blank_of,
    names_in_a_merge,
    output_column_of,
    read_output_column,
    value_for_another_row,
)
from conductor._internal.graph.receive_modes import (
    Cell,
    FixedValue,
    Gather,
    Merge,
    ReceiveMode,
    RowsUnder,
)
from conductor._internal.index import Index, Row
from conductor._internal.ref import Ref
from conductor._internal.skip_and_ask import SKIPPED
from conductor._internal.table import (
    HOLDS_NO_TABLE,
    ROW_ID_COLUMN,
    MergeSource,
    is_frame,
    merge_tables_on_row_ids,
    one_column_frame_with_row_ids,
    row_of_global_id,
    without_row_id_columns,
)

if TYPE_CHECKING:
    import polars as pl

    from conductor._internal.graph.compiled_graph import CompiledGraph

#: One run of one node: the node id and the row it runs on, ``()`` for a node that runs once.
Unit = tuple[str, Row]

#: An earlier unit of a re-planned node: the row it had and its reuse key.
_EarlierUnitKey = tuple[Row, str]


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
    outputs are blank at that row, an ``exclusive_group`` output is skipped (no branch
    was chosen), and an output that makes rows has no rows there. A
    ``Series[X]`` input leaves blanks out row by row. A unit is checked in
    this order: not ready, skip, blank.
    """

    blank_inputs: tuple[str, ...]


class UnitRecorded(NamedTuple):
    """What one ``record`` changed for the engine: the units ready because of it,
    in execution order, the event of each node it finished, its own and any
    others that have no units left to run, each once, and whether the node took
    part at all: a unit skipped doesn't start its node."""

    units_ready: list[Unit]
    node_events: list[NodeComplete | NodeSkipped]
    node_started: bool


class CallTheNode(NamedTuple):
    """This unit runs: call the node on ``inputs``."""

    inputs: dict[str, Any]


#: What comes of a unit's turn. Call the node; or it needed no call and is recorded; or an earlier leg's question
#: to a person, whose inputs match, is this unit's and is held as waiting; or its inputs can't be read and it fails.
UnitStart = CallTheNode | UnitRecorded | WaitingUnit | Problem


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
        #: What every field holds, by row, and the rows of every index.
        values = self._values = ValuesOfRun(lookups)
        self._value_or_skip_at = values.value_or_skip_at
        self._rows = values.rows
        #: Per unit handed out to be called, the hash of the inputs it was handed.
        self._input_hash_of_call: dict[Unit, str] = {}
        #: Each finished unit's reuse key, ``None`` for one not called.
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
        #: The tables each unit of a table-returning node received, for moving its ids when it is reused.
        self._tables_received: dict[Unit, list[pl.DataFrame]] = {}
        self._nodes_returning_a_table = frozenset(field.ref.node_id for field in lookups.field_making_rows_of_index.values() if field._held_as.__class__ is TableAtRow)
        #: Per (index id, row above), what each row under it reads: worked out once, for a unit a person is asked about.
        self._identities_of_rows_for_a_question: dict[tuple[str, Row], list[str]] = {}
        self._nodes_with_series_inputs_on_one_index = frozenset(node_id for node_id, sources in lookups.sources_of_series_inputs_on_one_index.items() if sources)
        #: Per changed node, what a later unit may still reuse.
        self._earlier_units: dict[str, _EarlierUnitsOfNode] = {}
        #: Per node, how many units this leg took from earlier work.
        self._reused_units: dict[str, int] = {}
        self._digests_of_values: DigestsOfValues = {}
        order = compiled.execution_order
        self._execution_position = {node_id: position for position, node_id in enumerate(order)}

    # -- units ---------------------------------------------------------------

    def units(self, node_id: str) -> list[Unit]:
        """This node's units right now: one, or one per row made on the index it runs per row of."""
        index = self._runs_per_row_of[node_id]
        if index is None:
            return [(node_id, ())]
        return [(node_id, row) for row in self._rows[index.id].rows_under(())]

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
        """This unit is ready, or parked on the first thing it reads that isn't finished, as ``(writer, row)``.

        The writer's units at that row finished writing, or, at a shorter row, its units
        under it, which needs that row's rows of its own index all made. The one
        rule of what makes a unit ready: nothing about a unit's readiness is decided anywhere else."""
        node_id, row = unit
        done = self._done
        for writer, depth in self._reads_wait_for[node_id]:
            needed = (writer, row[:depth])
            if needed not in done and not self._finished_at_or_under(writer, needed[1]):
                self._parked.setdefault(needed, []).append(unit)
                return
        ready.append(unit)

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

    # -- a unit's turn -----------------------------------------------------------

    def start_unit(self, unit: Unit) -> UnitStart:
        """What becomes of this unit now its turn has come. The one place that knows the order:
        its inputs, whether a skip or a blank stops it, what it hashes to, whether earlier work
        answers it. Anything that needs no call is recorded here."""
        node_id, row = unit
        try:
            inputs = self.inputs_for(unit)
        except NodeError as failure:
            return failure._as_problem_at(node_id, row)
        if isinstance(inputs, (SkippedAtRow, BlankInputsStopCall)):
            return self.record(unit, inputs)
        input_hash = hash_of_inputs(self._compiled.node(node_id)._input_hash_seed, inputs, self._digests_of_values)
        if node_id in self._nodes_returning_a_table:
            self._tables_received[unit] = self._tables_received_by(node_id, inputs)
        if self._earlier_units:
            earlier = self._claim_earlier_result(unit, input_hash)
            if isinstance(earlier, WaitingUnit):
                self._waiting[unit] = earlier
                return earlier
            if earlier is not None:
                cells, self._input_hash_of_call[unit] = earlier
                return self.record(unit, cells)
        self._input_hash_of_call[unit] = input_hash
        return CallTheNode(inputs)

    def _tables_received_by(self, node_id: str, inputs: dict[str, Any]) -> list[pl.DataFrame]:
        """The stored tables this unit received whole, as they are, and of a merged one its ids: what its own table's ids can have come from."""
        received: list[pl.DataFrame] = []
        for name, receive_mode in self._lookups.receive_modes_of_node[node_id]:
            value = inputs[name]
            if receive_mode.__class__ is Cell and is_frame(value) and ROW_ID_COLUMN in value.columns:
                received.append(value)
            elif receive_mode.__class__ is Merge:
                received.append(value.select(ROW_ID_COLUMN))
        return received

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
        if node._series_inputs_leaving_blanks_out or node_id in self._nodes_with_series_inputs_on_one_index:
            self._line_up_series_inputs(unit, values, node._series_inputs_leaving_blanks_out)
        return values

    def read_input(self, receive_mode: ReceiveMode, unit: Unit) -> Any | SkippedAtRow:
        """What this unit receives on an input that receives ``receive_mode``, or a ``SkippedAtRow``:
        the one match over receive modes a run needs. What it waits for first is ``reads_of_input``, once, at compile."""
        row = unit[1]
        match receive_mode:
            case Cell(sources=sources, depth=depth):
                if len(sources) == 1:
                    value, value_row = self._value_or_skip_at(sources[0], row[:depth])
                    if value is MISSING:
                        raise KeyError(f"{sources[0]} has no value at {value_row} — the unit was not ready")
                    return SkippedAtRow(row=value_row) if value is SKIPPED else value
                covering, skips = self._values_covering_row(sources, row[:depth])
                return covering[0] if covering else SkippedAtRow(row=max(skips, key=len))
            case FixedValue(value=value):
                return value
            case RowsUnder(sources=sources, depth=depth):
                found = [self._value_or_skip_at(ref, row[:depth]) for ref in sources]
                if all(value is SKIPPED for value, _ in found):
                    return SkippedAtRow(row=max((value_row for _, value_row in found), key=len))
                return self._rows_and_values_on(sources, row[:depth])[1]
            case Gather():
                return self._gathered_for_unit(receive_mode, unit)
            case Merge():
                merged = self._merged_table_for_unit(receive_mode, unit)
                # Read through the input's own type when the call is checked, so a
                # merged table it won't take fails this node, naming the column.
                return SkippedAtRow(row=row) if merged is None else merged

    def _gathered_for_unit(self, receive_mode: Gather, unit: Unit) -> list[Any]:
        """The values this unit gathers, each source read as compile settled, left out where skipped."""
        gathered: list[Any] = []
        for read in receive_mode.source_reads:
            value = self.read_input(read, unit)
            if isinstance(value, SkippedAtRow):
                continue
            if read.__class__ is Cell:
                gathered.append(value)
            else:
                gathered.extend(value)
        return gathered

    def _merged_table_for_unit(self, receive_mode: Merge, unit: Unit) -> pl.DataFrame | None:
        """The one table this unit receives from every source of a ``Table`` input, or
        ``None`` when every source is skipped there. Reads each source by its own record;
        ``conductor._internal.table.merge_tables_on_row_ids`` matches them. A skipped source
        has no rows and still counts as a source, and of tables that are alternatives the one that ran is read."""
        row = unit[1]
        sources: list[MergeSource] = []
        skipped = 0
        for read, carried in zip(receive_mode.source_reads, receive_mode.carries_columns_of, strict=True):
            group_row = row[: read.depth]
            source = next((ref for ref in read.sources if self._value_or_skip_at(ref, group_row)[0] is not SKIPPED), read.sources[0])
            field = self._fields[source]
            output_name = names_in_a_merge(self._fields[read.sources[0]])[1]
            copied_down = source in receive_mode.copied_down
            stored, _ = self._value_or_skip_at(source, group_row)
            is_skip = stored is SKIPPED
            skipped += is_skip
            if read.__class__ is Cell:  # a table: its rows are the ones it holds
                frame = blank_of(field) if is_skip else stored
                sources.append(MergeSource(output_name, frame, field._held_as.rows_made, [(*group_row, position) for position in range(frame.height)], copied_down, carried))  # type: ignore[arg-type]
                continue
            index: Index = field._index  # type: ignore[assignment]
            rows, values = self._rows_and_values_on(read.sources, group_row)
            row_id_at = self._values.row_id
            frame = one_column_frame_with_row_ids(names_in_a_merge(field)[0], field.type, [row_id_at(index, row) for row in rows], values)
            sources.append(MergeSource(output_name, frame, index, rows, copied_down, carried))
        return None if skipped == len(sources) else merge_tables_on_row_ids(sources)

    def _values_covering_row(self, refs: tuple[Ref, ...], row: Row) -> tuple[list[Any], list[Row]]:
        """Of several refs read at one row: the values that cover it, and the
        rows at which the others were skipped. Two covering the same row
        means two edges both supplied that row, and the node fails with
        that row, named in the failure's ``ctx``."""
        covering: list[Any] = []
        skips: list[Row] = []
        for ref in refs:
            value, value_row = self._value_or_skip_at(ref, row)
            if value is MISSING:
                raise KeyError(f"{ref} has no value at {value_row} — the unit was not ready")
            if value is SKIPPED:
                skips.append(value_row)
            else:
                covering.append(value)
        if len(covering) > 1:
            raise NodeError(PROBLEM_CATALOGUE["row_covered_twice"].message, code="row_covered_twice", ctx={"edges": len(covering), "row": list(row)})
        return covering, skips

    def _rows_and_values_on(self, refs: tuple[Ref, ...], under: Row) -> tuple[list[Row], list[Any]]:
        """The rows of ``refs``' index under ``under`` that hold a value and those values,
        from one ref or from whichever of several covers each row."""
        index = self._fields[refs[0]]._index
        assert index is not None  # what is read as rows has an index
        rows: list[Row] = []
        values: list[Any] = []
        for row in self._rows[index.id].rows_under(under):
            covering, _ = self._values_covering_row(refs, row)
            if covering:
                rows.append(row)
                values.append(covering[0])
        return rows, values

    def _line_up_series_inputs(self, unit: Unit, values: dict[str, Any], leaving_blanks_out: tuple[str, ...]) -> None:
        """Keep the inputs of many values that read one index on the same rows, whatever a node does with them.

        A row one column skipped is blank in that column, the row kept in the others;
        a row blank on an input in ``leaving_blanks_out`` (they take no blank) then leaves
        all of them on its index, or a node zipping them would pair wrong values.
        An input gathered or set on the node has rows of its own, so its blanks leave only it."""
        node_id, row = unit
        may_be_out_of_step = any(map(self._values.may_hold_a_skip, self._lookups.sources_of_series_inputs_on_one_index[node_id]))
        if not may_be_out_of_step and not any(cell is None for name in leaving_blanks_out for cell in values[name]):  # by identity: a frame refuses ``==``
            return
        read_as_rows: dict[str, tuple[Index, dict[Row, Any]]] = {}
        rows_of_index: dict[Index, set[Row]] = {}
        for name, receive_mode in self._lookups.receive_modes_of_node[node_id]:
            if isinstance(receive_mode, RowsUnder):
                index = self._fields[receive_mode.sources[0]]._index
                assert index is not None  # what is read as rows has an index
                rows, cells = self._rows_and_values_on(receive_mode.sources, row[: receive_mode.depth])
                read_as_rows[name] = (index, dict(zip(rows, cells, strict=True)))
                rows_of_index.setdefault(index, set()).update(rows)
        rows_leaving: dict[Index, set[Row]] = {}
        for name in leaving_blanks_out:
            if name in read_as_rows:
                index, cell_at = read_as_rows[name]
                rows_leaving.setdefault(index, set()).update(on_row for on_row in rows_of_index[index] if cell_at.get(on_row) is None)
            else:
                values[name] = [cell for cell in values[name] if cell is not None]
        for name, (index, cell_at) in read_as_rows.items():
            leaving = rows_leaving.get(index, ())
            values[name] = [cell_at.get(on_row) for on_row in sorted(rows_of_index[index]) if on_row not in leaving]

    # -- recording -----------------------------------------------------------------

    def record(self, unit: Unit, outputs: dict[str, Any] | SkippedAtRow | BlankInputsStopCall) -> UnitRecorded:
        """Record what this unit produced, and hand back what that makes ready.

        ``outputs`` are cells that passed their outputs' checks, per output: one
        cell, ``SKIPPED``, a list of cells on an output that makes rows, or a
        table; or a ``SkippedAtRow`` or a ``BlankInputsStopCall`` when the unit didn't run,
        its outputs ``SKIPPED`` at that row or blank (``values_of_run.blank_of``). An output that makes
        rows makes them under the unit's row, one per list item or table
        row, each child of ``row``; a skipped one leaves its index with no rows
        under the row, an empty one leaves an empty group, so a reduction over
        it still runs. A unit may finish with a skip reaching rows above it; such a
        reduction may then have run first on an empty group, and what it
        returned is hidden by the skip, so a run's results don't depend on
        which came first, though the number of calls can.

        The units returned are the ones whose first need this write finished,
        asked again; the engine starts those and asks nothing else.
        """
        node_id, row = unit
        node = self._compiled.node(node_id)
        assert node._wiring is not None  # the engine runs only a runnable graph
        fields = node._wiring.fields
        write = self._values.write
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
                value = SKIPPED if out.exclusive_group is not None else blank_of(field)  # no branch was chosen
            else:
                value = outputs[out.name]  # type: ignore[index]
            write(field, row if skip_row is None else skip_row, value, rows_made)
        was_waiting = self._waiting.pop(unit, None)
        input_hash = self._input_hash_of_call.pop(unit, None)  # an attempt that failed left one, and this one may be a skip
        self._mark_done(unit, was_waiting.reuse_key if was_waiting is not None else None if skip_row is not None or not_called is not None else input_hash)
        return self._what_this_write_makes_ready(unit, rows_made, skip_row is None)

    def _mark_done(self, unit: Unit, reuse_key: str | None) -> None:
        """Mark the unit done, and count it under each shorter row its node has readers at."""
        self._done[unit] = reuse_key
        node_id, row = unit
        if self._runs_per_row_of[node_id] is None:
            return
        counts = self._units_done_under
        for depth in self._counted_depths[node_id]:
            key = (node_id, row[:depth])
            counts[key] = counts.get(key, 0) + 1

    def _what_this_write_makes_ready(self, unit: Unit, rows_made: list[tuple[str, Row, int]], node_started: bool) -> UnitRecorded:
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
        return UnitRecorded(ready, [self._event_of_completed_node(completed_node_id) for completed_node_id in completed], node_started)

    def _event_of_completed_node(self, node_id: str) -> NodeComplete | NodeSkipped:
        """What the engine says of a node with no units left: skipped when it did nothing, else its result."""
        result = self._values.result_of(node_id)
        rows_not_called = self._rows_not_called.get(node_id, {})
        if result is None and not rows_not_called:
            return NodeSkipped(node_id=node_id)
        return NodeComplete(node_id=node_id, outputs=result or {}, rows_not_called=rows_not_called, reused_units=self._reused_units.get(node_id, 0))

    # -- waiting on a person ---------------------------------------------

    def mark_waiting(self, unit: Unit, waiting_unit: WaitingUnit) -> WaitingUnit:
        """Hold this unit as waiting: its node asked a person, on this row. It and whatever reads it
        wait. Hands back the unit as held, its question bound to the row it was asked on and known by that."""
        reuse_key = self._reuse_key_of_question(unit, self._input_hash_of_call.pop(unit))
        held = self._waiting[unit] = waiting_unit.model_copy(update={"reuse_key": reuse_key, "id": id_of_waiting_unit(unit[0], unit[1], reuse_key)})
        return held

    def record_answer(self, unit: Unit, cells: dict[str, Any] | None) -> None:
        """A person answered this waiting unit: record it as if its node had returned ``cells``; ``None`` is no value on its row."""
        self.record(unit, SkippedAtRow(unit[1]) if cells is None else cells)
        self._answered.add(unit)

    def _reuse_key_of_question(self, unit: Unit, input_hash: str) -> str:
        """The reuse key of a question this unit puts to a person: ``input_hash``, and on a row the row it is asked on."""
        index = self._runs_per_row_of[unit[0]]
        return input_hash if index is None else blake2b(f"{input_hash}|{self._identity_of_row_for_a_question(index, unit[1])}".encode(), digest_size=16).hexdigest()

    def _identity_of_row_for_a_question(self, index: Index, row: Row) -> str:
        """All a row is: what it holds, the row above it, the row its id came from, and which of how many identical rows it is."""
        group = row[:-1]
        identities = self._identities_of_rows_for_a_question.get((index.id, group))
        if identities is None:
            identities = self._identities_of_rows_for_a_question[(index.id, group)] = self._identities_of_rows_under(index, group)
        return identities[row[-1]]

    def _identities_of_rows_under(self, index: Index, group: Row) -> list[str]:
        """Every row of ``index`` under ``group``, as ``_identity_of_row_for_a_question`` has it, in order."""
        holds = self._what_rows_under_hold(index, group)
        above = "" if index.parent is None else self._identity_of_row_for_a_question(index.parent, group)
        how_many_alike = Counter(holds)
        seen: dict[str, int] = {}
        identities: list[str] = []
        for held in holds:
            nth = seen.get(held, 0)
            seen[held] = nth + 1
            identities.append(f"{above}|{held}|{nth}/{how_many_alike[held]}")
        return identities

    def _what_rows_under_hold(self, index: Index, group: Row) -> list[str]:
        """What each row of ``index`` under ``group`` holds; a row of a table whose id another index spelled holds that row too."""
        maker = self._lookups.field_making_rows_of_index[index.id]
        if maker._held_as.__class__ is not TableAtRow:
            cells = self._values.cells_under(maker.ref, group)
            return [repr(cell) if isinstance(cell, HOLDS_NO_TABLE) else hash_of_row_cells((cell,), self._digests_of_values) for cell in cells]
        stored, _ = self._value_or_skip_at(maker.ref, group)
        values = without_row_id_columns(stored)
        held = list(map(repr, values.iter_rows())) if values.width else [""] * stored.height  # a table's cells are plain, and their repr is exact
        return [digest if row_id.rpartition(":")[0] == index.id else digest + self._identity_of_the_row_its_id_came_from(row_id) for digest, row_id in zip(held, stored[ROW_ID_COLUMN].to_list(), strict=True)]

    def _identity_of_the_row_its_id_came_from(self, row_id: str) -> str:
        """The identity of the row a carried id names, ``""`` when it names none this run has: a node may hand on any text."""
        spelled = row_of_global_id(row_id)
        field = None if spelled is None else self._lookups.field_making_rows_of_index.get(spelled[0])
        if spelled is None or field is None:
            return ""
        index_id, row = spelled
        rows = self._rows[index_id]
        if len(row) != rows.depth or not rows.has_row(row):
            return ""
        index = field._held_as.rows_made
        assert index is not None  # a field that makes rows has their index
        return self._identity_of_row_for_a_question(index, row)

    def waiting(self) -> tuple[WaitingUnit, ...]:
        """Every unit waiting on a person, in execution order and row order
        within a node: not the order their concurrent calls happened to ask in."""
        position_of = self._execution_position
        return tuple(waiting_unit for _, waiting_unit in sorted(self._waiting.items(), key=lambda item: (position_of[item[0][0]], item[0][1])))

    # -- work from an earlier state ----------------------------------------------------

    def _claim_earlier_result(self, unit: Unit, input_hash: str) -> tuple[dict[str, Any], str] | WaitingUnit | None:
        """An earlier result of this node for a unit that received ``input_hash``, taken so no
        other unit can have it: the cells to ``record`` at this unit's row and its reuse key, or, for
        one that was waiting on a person, that unit moved to this row, or ``None``
        when there is none and the unit must run."""
        node_id, row = unit
        earlier = self._earlier_units.get(node_id)
        if earlier is None:
            return None
        claimed = earlier.claim(input_hash, self._reuse_key_of_question(unit, input_hash) if earlier.a_person_was_asked else None)
        if claimed is None:
            return None
        if claimed.waiting is not None:
            self._reused_units[node_id] = self._reused_units.get(node_id, 0) + 1
            return claimed.waiting if claimed.row == row else claimed.waiting.model_copy(update={"row": row, "id": id_of_waiting_unit(node_id, row, claimed.reuse_key)})
        cells = self._cells_of_earlier_unit(claimed, unit)
        if cells is None:
            return None
        self._reused_units[node_id] = self._reused_units.get(node_id, 0) + 1
        if claimed.answered:
            self._answered.add(unit)
        return cells, claimed.reuse_key

    def _cells_of_earlier_unit(self, claimed: _EarlierUnit, unit: Unit) -> dict[str, Any] | None:
        """What an earlier unit wrote, one value per output as ``record`` takes it: ids its tables minted
        re-spelled at this unit's row, ids they took in moved to what this unit received;
        ``None`` when the state has lost a value, and the unit runs."""
        node_id, row = unit
        cells: dict[str, Any] = {}
        for out in self._lookups.interfaces[node_id].outputs:
            value = claimed.cells.get(out.name, MISSING)
            if value is MISSING:
                return None
            cells[out.name] = value_for_another_row(self._fields[Ref(node_id, out.name)], value, claimed.row, row, claimed.tables_received, self._tables_received.get(unit, []))
        return cells

    def _earlier_units_still_unclaimed(self, node_id: str) -> list[EarlierUnit]:
        """What is left of this node's earlier work when the node isn't finished, for a leg going on from this one."""
        earlier = self._earlier_units.get(node_id)
        if earlier is None or self.node_is_complete(node_id):
            return []
        return [
            EarlierUnit.model_construct(
                row=unit.row, reuse_key=unit.reuse_key, answered=unit.answered, waiting=unit.waiting, tables_received=unit.tables_received,
                outputs={name: output_column_of({unit.row: cell}, self._fields[Ref(node_id, name)]) for name, cell in unit.cells.items()},
            )
            for unit in earlier.unclaimed()
        ]

    # -- the state ----------------------------------------------------------------

    def state(self, status: RunStatus, failure: Problem | None = None) -> RunState:
        """Everything this leg has produced, as the ``RunState`` a host stores.

        Built without validation from values that already passed, and holding
        them typed: the next read costs nothing to decode, and a dump writes
        each one by its own type."""
        compiled = self._compiled
        finished_units: dict[str, list[FinishedUnit]] = {}
        for (node_id, row), reuse_key in self._done.items():
            finished_units.setdefault(node_id, []).append((row, reuse_key))
        answered: dict[str, list[Row]] = {}
        for node_id, row in sorted(self._answered):
            answered.setdefault(node_id, []).append(row)
        waiting = {node_id: tuple(units) for node_id, units in groupby(self.waiting(), key=attrgetter("node_id"))}
        tables_received: dict[str, list[tuple[Row, list[pl.DataFrame]]]] = {}
        for (node_id, row), tables in self._tables_received.items():
            if tables and (node_id, row) in self._done:
                tables_received.setdefault(node_id, []).append((row, tables))
        nodes: dict[str, WorkOfNode] = {}
        for node_id in compiled.execution_order:
            units = finished_units.get(node_id)
            if units is not None:
                units.sort()  # no two units of a node share a row, so the hash is never compared
            elif self.node_is_complete(node_id):
                units = []  # a node complete on no rows: the state says it finished
            nodes[node_id] = WorkOfNode._built_from_checked_fields(
                fingerprint=compiled.node(node_id).fingerprint,
                finished_units=units,
                answered_rows=answered.get(node_id, []),
                tables_received=sorted(tables_received[node_id], key=itemgetter(0)) if node_id in tables_received else [],
                outputs=self._values.output_columns_of(node_id),
                waiting=waiting.get(node_id, ()),
                rows_not_called=dict(sorted(self._rows_not_called[node_id].items())) if node_id in self._rows_not_called else {},
                reused_units=self._reused_units.get(node_id, 0),
                earlier_units=self._earlier_units_still_unclaimed(node_id) if self._earlier_units else [],
            )
        return RunState.model_construct(status=status, failure=failure, nodes=nodes)

    @classmethod
    def restore(cls, compiled: CompiledGraph, state: RunState) -> RunLedger:
        """A ledger holding what an earlier leg recorded, over the graph being run.

        Every node ``compiled._nodes_a_leg_runs_again(state)`` names is re-planned:
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
        changed = set(compiled._nodes_a_leg_runs_again(state))
        problems = restored._values.read_state(compiled, state, changed)
        if problems:
            raise InvalidCallError(problems)
        restored._earlier_units = {node_id: pool for node_id, work in state.nodes.items() if (pool := _earlier_units_of_node(compiled, node_id, work, read_its_own_units=node_id in changed))}
        kept = {node_id: work for node_id, work in state.nodes.items() if node_id not in changed}
        restored._waiting = {(waiting_unit.node_id, waiting_unit.row): waiting_unit for work in kept.values() for waiting_unit in work.waiting}
        restored._tables_received = {(node_id, row): tables for node_id, work in kept.items() for row, tables in work.tables_received}
        restored._rows_not_called = {node_id: dict(work.rows_not_called) for node_id, work in kept.items() if work.rows_not_called}
        problems = restored._replay_finished_units(kept)
        problems.extend(restored._problems_with_waiting_units())
        if problems:
            raise InvalidCallError(problems)
        restored._answered = {(node_id, row) for node_id, work in kept.items() for row in work.answered_rows}
        return restored

    def _replay_finished_units(self, kept: Mapping[str, WorkOfNode]) -> list[Problem]:
        """Write every finished unit, in execution order, as its record did, and
        report one at a row that doesn't exist or with no value at its row."""
        problems: list[Problem] = []
        interfaces = self._lookups.interfaces
        for node_id in self._compiled.execution_order:
            work = kept.get(node_id)
            if work is None or work.finished_units is None:
                continue
            index = self._runs_per_row_of[node_id]
            rows_seen: set[Row] = set()
            for row, reuse_key in sorted(work.finished_units, key=itemgetter(0)):
                if not self._row_exists_for_node(index, row):
                    problems.append(make_problem("state_unreadable", "state", node_id, *row, reason=f"the state has {node_id!r} at row {list(row)}, which {node_id!r} has no place for"))
                    continue
                if row in rows_seen:
                    problems.append(make_problem("state_unreadable", "state", node_id, *row, reason=f"the state lists {node_id!r} as finished at row {list(row)} twice"))
                    continue
                rows_seen.add(row)
                for out in interfaces[node_id].outputs:
                    field = self._fields[Ref(node_id, out.name)]
                    if not self._values.make_rows_again_from_stored_output(field, row):
                        problems.append(make_problem("state_unreadable", "state", field.ref, *row, reason=f"the state has {node_id!r} finished at row {list(row)}, with no value for {field.ref}"))
                self._mark_done((node_id, row), reuse_key)
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


class _EarlierUnit(NamedTuple):
    """A unit a re-planned node left behind, with what it wrote read back as ``record`` takes it: ``cells``, per output."""

    row: Row
    reuse_key: str
    answered: bool
    waiting: WaitingUnit | None
    tables_received: list[pl.DataFrame]
    cells: dict[str, Any]


class _EarlierUnitsOfNode:
    """The units a re-planned node left in the state a leg starts from, held to be claimed.

    Units of several legs sit side by side, two of them at one row. ``claim``
    hands each out once, the lowest row first: identical inputs are a multiset,
    so a third identical row gets a new call and never shares a result, and the
    k-th identical row takes the k-th. A unit a person was asked about has its
    row in its key, so only the same row finds it.
    """

    __slots__ = ("units_with_reuse_key_lowest_row_last", "a_person_was_asked")

    def __init__(self, units: Iterable[_EarlierUnit]) -> None:
        self.a_person_was_asked = False
        self.units_with_reuse_key_lowest_row_last: dict[str, list[_EarlierUnit]] = {}
        for unit in sorted(units, key=itemgetter(0, 1), reverse=True):
            self.a_person_was_asked |= unit.answered or unit.waiting is not None
            self.units_with_reuse_key_lowest_row_last.setdefault(unit.reuse_key, []).append(unit)

    def claim(self, input_hash: str, key_if_a_person_was_asked: str | None) -> _EarlierUnit | None:
        """Take the earlier unit bound to this row or, failing that, one that only took these inputs; gone from the pool, or ``None``."""
        for reuse_key in (key_if_a_person_was_asked, input_hash):
            units = None if reuse_key is None else self.units_with_reuse_key_lowest_row_last.get(reuse_key)
            if units:
                return units.pop()
        return None

    def unclaimed(self) -> list[_EarlierUnit]:
        return sorted(chain.from_iterable(self.units_with_reuse_key_lowest_row_last.values()), key=itemgetter(0, 1))


def _earlier_units_of_node(compiled: CompiledGraph, node_id: str, work: WorkOfNode, read_its_own_units: bool) -> _EarlierUnitsOfNode | None:
    """What this node can reuse: its own finished and waiting units, when it is re-planned, and those carried
    from earlier legs; or ``None`` when it can't reuse any: gone from the graph, a node that's a graph, opted out, nothing with a hash,
    or outputs that no longer read as its types."""
    node = compiled._nodes.get(node_id)
    if node is None or node.embedded_graph is not None or node.interface is None or not node.version.policy.reuse_when_inputs_match:  # type: ignore[union-attr]  # an interface comes with a version
        return None
    outputs = [(out.name, Ref(node_id, out.name)) for out in node.interface.outputs]
    units: dict[_EarlierUnitKey, _EarlierUnit] = {}
    problems: list[Problem] = []
    if read_its_own_units:
        answered = set(work.answered_rows)
        tables_received = dict(work.tables_received)
        finished = {row: _EarlierUnit(row, reuse_key, row in answered, None, tables_received.get(row, []), {}) for row, reuse_key in work.finished_units or () if reuse_key is not None}
        units.update(((unit.row, unit.reuse_key), unit) for unit in finished.values())
        units.update(((unit.row, unit.reuse_key), _EarlierUnit(unit.row, unit.reuse_key, False, unit, [], {})) for unit in work.waiting if unit.reuse_key)
        for name, ref in outputs:
            column = work.outputs.get(name)
            read = None if column is None else read_output_column(compiled, ref, column, problems)
            if read is not None:
                for row, unit in finished.items():
                    if row in read:
                        unit.cells[name] = read[row]
    for stored in work.earlier_units:
        key = (stored.row, stored.reuse_key)
        if key in units:  # a unit of this very leg at that row with that key has the newer result
            continue
        unit = units[key] = _EarlierUnit(stored.row, stored.reuse_key, stored.answered, stored.waiting, stored.tables_received, {})
        for name, ref in outputs:
            column = stored.outputs.get(name)
            read = None if column is None else read_output_column(compiled, ref, column, problems)
            if read is not None and stored.row in read:
                unit.cells[name] = read[stored.row]
    return None if problems or not units else _EarlierUnitsOfNode(units.values())
