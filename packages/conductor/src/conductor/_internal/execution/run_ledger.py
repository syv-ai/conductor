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
just returned it (``start_unit``). Nothing maps an old row to a new one.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import groupby
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
from conductor._internal.execution.input_hash import DigestsOfValues, hash_of_inputs
from conductor._internal.execution.run_state import (
    EarlierUnit,
    FinishedUnit,
    OutputColumn,
    RunState,
    RunStatus,
    WaitingUnit,
    WorkOfNode,
    id_of_waiting_unit,
)
from conductor._internal.execution.values_of_run import (
    MISSING,
    NO_VALUES,
    ValuesOfRun,
    blank_of,
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
from conductor._internal.node_definition import NodeVersion
from conductor._internal.ref import Ref
from conductor._internal.skip_and_ask import SKIPPED
from conductor._internal.table import (
    MergeSource,
    merge_tables_on_row_ids,
    one_column_frame_with_row_ids,
)

if TYPE_CHECKING:
    import polars as pl

    from conductor._internal.graph.compiled_graph import CompiledGraph

#: One run of one node: the node id and the row it runs on, ``()`` for a node that runs once.
Unit = tuple[str, Row]

#: An earlier unit of a re-planned node: the row it had and the hash of what it received.
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
    outputs are blank at that row, a ``choice`` output is skipped (no branch
    was chosen), and an output that makes rows has no rows there. A
    ``Series[X]`` input leaves blanks out row by row. A unit is checked in
    this order: not ready, skip, blank.
    """

    blank_inputs: tuple[str, ...]


class UnitRecorded(NamedTuple):
    """What one ``record`` changed for the engine: the units ready because of it,
    in execution order, and the event of each node it finished, its own and any
    others that have no units left to run, each once."""

    units_ready: list[Unit]
    node_events: list[NodeComplete | NodeSkipped]


class CallTheNode(NamedTuple):
    """This unit runs: call the node on ``inputs``. ``input_hash`` is what a later leg reuses it by."""

    inputs: dict[str, Any]
    input_hash: str


class FinishedWithoutACall(NamedTuple):
    """This unit needed no call and is recorded: an earlier result was reused, or a blank reached an input that takes none."""

    recorded: UnitRecorded


class SkippedWithoutACall(NamedTuple):
    """This unit needed no call and is recorded skipped; its node may never have started."""

    recorded: UnitRecorded


class WaitsOnAPersonAgain(NamedTuple):
    """An earlier leg's question to a person, whose inputs match, is this unit's: held as waiting, already."""

    waiting_unit: WaitingUnit


class CouldNotReadInputs(NamedTuple):
    """The unit's inputs can't be read (two edges into one input both supplied this row): the unit fails."""

    problem: Problem


UnitStart = CallTheNode | FinishedWithoutACall | SkippedWithoutACall | WaitsOnAPersonAgain | CouldNotReadInputs


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

    # -- a unit's turn -----------------------------------------------------------

    def start_unit(self, unit: Unit) -> UnitStart:
        """What becomes of this unit now its turn has come. The one place that knows the order:
        its inputs, whether a skip or a blank stops it, what it hashes to, whether earlier work
        answers it. Anything that needs no call is recorded here."""
        node_id, row = unit
        try:
            inputs = self.inputs_for(unit)
        except NodeError as failure:
            return CouldNotReadInputs(failure._as_problem_at(node_id, row))
        if isinstance(inputs, SkippedAtRow):
            return SkippedWithoutACall(self.record(unit, inputs))
        if isinstance(inputs, BlankInputsStopCall):
            return FinishedWithoutACall(self.record(unit, inputs))
        input_hash = hash_of_inputs(self._compiled.node(node_id)._input_hash_seed, inputs, self._digests_of_values)
        if self._earlier_units:
            earlier = self._claim_earlier_result(unit, input_hash)
            if isinstance(earlier, WaitingUnit):
                self.mark_waiting(unit, earlier)
                return WaitsOnAPersonAgain(earlier)
            if earlier is not None:
                return FinishedWithoutACall(self.record(unit, earlier, input_hash))
        return CallTheNode(inputs, input_hash)

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
                    if value is MISSING:
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
        has no rows and still counts as a source."""
        row = unit[1]
        sources: list[MergeSource] = []
        skipped = 0
        for read in receive_mode.source_reads:
            (source,) = read.sources
            field = self._fields[source]
            group_row = row[: read.depth]
            copied_down = source in receive_mode.copied_down
            stored, _ = self._value_or_skip_at(source, group_row)
            is_skip = stored is SKIPPED
            skipped += is_skip
            if read.__class__ is Cell:  # a table: its rows are the ones it holds
                frame = blank_of(field) if is_skip else stored
                sources.append(MergeSource(source.node_id, frame, field._held_as.rows_made, [(*group_row, position) for position in range(frame.height)], copied_down))  # type: ignore[arg-type]
                continue
            index: Index = field._index  # type: ignore[assignment]
            rows, values = self._rows_and_values_on(unit[0], read.sources, group_row)
            row_id_at = self._values.row_id
            frame = one_column_frame_with_row_ids(field._title, field.type, [row_id_at(index, row) for row in rows], values)
            sources.append(MergeSource(source.node_id, frame, index, rows, copied_down))
        return None if skipped == len(sources) else merge_tables_on_row_ids(sources)

    def _values_covering_row(self, node_id: str, refs: tuple[Ref, ...], row: Row) -> tuple[list[Any], list[Row]]:
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
            raise NodeError(PROBLEM_CATALOGUE["row_covered_twice"].message, node_id=node_id, code="row_covered_twice", ctx={"edges": len(covering), "row": list(row)})
        return covering, skips

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
        its outputs ``SKIPPED`` at that row or blank (``values_of_run.blank_of``). An output that makes
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
        if self._waiting:
            self._waiting.pop(unit, None)
        self._mark_done(unit, input_hash)
        return self._what_this_write_makes_ready(unit, rows_made)

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
        return UnitRecorded(ready, [self._event_of_completed_node(completed_node_id) for completed_node_id in completed])

    def _event_of_completed_node(self, node_id: str) -> NodeComplete | NodeSkipped:
        """What the engine says of a node with no units left: skipped when it did nothing, else its result."""
        result = self.result_of(node_id)
        rows_not_called = self._rows_not_called.get(node_id, {})
        if result is None and not rows_not_called:
            return NodeSkipped(node_id=node_id)
        return NodeComplete(node_id=node_id, result=result or {}, rows_not_called=rows_not_called, reused_units=self._reused_units.get(node_id, 0))

    # -- waiting on a person ---------------------------------------------

    def mark_waiting(self, unit: Unit, waiting_unit: WaitingUnit) -> None:
        """Hold this unit as waiting: its node asked a person. It and whatever reads it wait."""
        self._waiting[unit] = waiting_unit

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

    def _claim_earlier_result(self, unit: Unit, input_hash: str) -> dict[str, Any] | WaitingUnit | None:
        """An earlier result of this node for a unit that received ``input_hash``, taken so no
        other unit can have it: the cells to ``record`` at this unit's row, or, for
        one that was waiting on a person, that unit moved to this row, or ``None``
        when there is none and the unit must run."""
        node_id, row = unit
        earlier = self._earlier_units.get(node_id)
        claimed = None if earlier is None else earlier.claim(row, input_hash)
        if earlier is None or claimed is None:
            return None
        old_row = claimed[0]
        waiting_unit = earlier.waiting.get(claimed)
        if waiting_unit is not None:
            self._reused_units[node_id] = self._reused_units.get(node_id, 0) + 1
            return waiting_unit if old_row == row else waiting_unit.model_copy(update={"row": row, "id": id_of_waiting_unit(node_id, row, input_hash)})
        cells = self._cells_of_earlier_unit(earlier, node_id, claimed, row)
        if cells is None:
            return None
        self._reused_units[node_id] = self._reused_units.get(node_id, 0) + 1
        if claimed in earlier.answered:
            self._answered.add(unit)
        return cells

    def _cells_of_earlier_unit(self, earlier: _EarlierUnitsOfNode, node_id: str, claimed: _EarlierUnitKey, row: Row) -> dict[str, Any] | None:
        """What an earlier unit wrote, one value per output as ``record`` takes it, ids its tables minted
        re-spelled at ``row``; ``None`` when the state has lost a value, and the unit runs."""
        cells: dict[str, Any] = {}
        old_row = claimed[0]
        for out in self._lookups.interfaces[node_id].outputs:
            ref = Ref(node_id, out.name)
            value = earlier.cells.get(ref, NO_VALUES).get(claimed, MISSING)
            if value is MISSING:
                return None
            cells[out.name] = value_for_another_row(self._fields[ref], value, old_row, row)
        return cells

    def _earlier_units_still_unclaimed(self, node_id: str) -> list[EarlierUnit]:
        """What is left of this node's earlier work when the node isn't finished, for a leg going on from this one."""
        earlier = self._earlier_units.get(node_id)
        if earlier is None or not earlier.unclaimed or self.node_is_complete(node_id):
            return []
        outputs = [(out.name, Ref(node_id, out.name)) for out in self._lookups.interfaces[node_id].outputs]
        units: list[EarlierUnit] = []
        for key in sorted(earlier.unclaimed):
            row, input_hash = key
            columns: dict[str, OutputColumn] = {}
            for name, ref in outputs:
                by_key = earlier.cells.get(ref, NO_VALUES)
                if key in by_key:
                    columns[name] = output_column_of({row: by_key[key]}, self._fields[ref])
            units.append(EarlierUnit.model_construct(
                row=row, input_hash=input_hash, answered=key in earlier.answered,
                only_its_own_row_may_claim=input_hash in earlier.hashes_only_their_own_rows_may_claim,
                waiting=earlier.waiting.get(key), outputs=columns,
            ))
        return units

    # -- what the run produced -------------------------------------------------------

    def result_of(self, node_id: str) -> dict[str, Any] | None:
        """This node's outputs by name, or ``None`` when the node did not run
        (every output is skipped at the top).

        A result holds only what the node produced: an output the node
        skipped (the branch not taken) is absent, never ``SKIPPED``. A scalar
        output of a node that ran once is its value; anything on an index is
        ``ValuesOnRows``, without the rows that were skipped.
        """
        return self._values.result_of(node_id)

    # -- the state ----------------------------------------------------------------

    def state(self, status: RunStatus, failure: Problem | None = None) -> RunState:
        """Everything this leg has produced, as the ``RunState`` a host stores.

        Built without validation from values that already passed, and holding
        them typed: the next read costs nothing to decode, and a dump writes
        each one by its own type."""
        compiled = self._compiled
        finished_units: dict[str, list[FinishedUnit]] = {}
        for (node_id, row), input_hash in self._done.items():
            finished_units.setdefault(node_id, []).append((row, input_hash))
        answered: dict[str, list[Row]] = {}
        for node_id, row in sorted(self._answered):
            answered.setdefault(node_id, []).append(row)
        waiting = {node_id: tuple(units) for node_id, units in groupby(self.waiting(), key=attrgetter("node_id"))}
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

        Every node ``compiled._nodes_changed_since(state)`` names is re-planned:
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
        changed = set(compiled._nodes_changed_since(state))
        problems = restored._values.read_state(compiled, state, changed)
        if problems:
            raise InvalidCallError(problems)
        restored._earlier_units = _earlier_units_of_nodes(compiled, state, changed)
        kept = {node_id: work for node_id, work in state.nodes.items() if node_id not in changed}
        restored._waiting = {(waiting_unit.node_id, waiting_unit.row): waiting_unit for work in kept.values() for waiting_unit in work.waiting}
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
            for row, input_hash in sorted(work.finished_units, key=itemgetter(0)):
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
    """The units a re-planned node left in the state a leg starts from, held to be claimed.

    Each is keyed by its row and hash, so units of several legs sit side by
    side, two of them at one row. ``claim`` hands each out once. A claim
    takes the unit at the claimer's own row with the same hash, else the lowest
    unclaimed row with the hash: identical inputs are a multiset, so a third
    identical row gets a new call and never shares a result. A hash answered
    by a person on more than one row is taken only by its own row: the inputs
    can't say whose answer it was, and a person's answer has no second chance.
    ``cells`` holds what each output wrote, per unit.
    """

    __slots__ = ("unclaimed", "units_with_hash_lowest_row_last", "hashes_only_their_own_rows_may_claim", "answered", "waiting", "cells")

    def __init__(
        self, unclaimed: set[_EarlierUnitKey], answered: set[_EarlierUnitKey], hashes_only_their_own_rows_may_claim: set[str],
        waiting: dict[_EarlierUnitKey, WaitingUnit], cells: dict[Ref, dict[_EarlierUnitKey, Any]],
    ) -> None:
        self.unclaimed = unclaimed
        self.answered = answered
        self.waiting = waiting
        self.cells = cells
        self.units_with_hash_lowest_row_last: dict[str, list[_EarlierUnitKey]] = {}
        for key in sorted(unclaimed, reverse=True):
            self.units_with_hash_lowest_row_last.setdefault(key[1], []).append(key)
        tied = (input_hash for input_hash, keys in self.units_with_hash_lowest_row_last.items() if len(keys) > 1 and not answered.isdisjoint(keys))
        self.hashes_only_their_own_rows_may_claim = frozenset((*hashes_only_their_own_rows_may_claim, *tied))

    def claim(self, row: Row, input_hash: str) -> _EarlierUnitKey | None:
        """Take the earlier unit this unit's inputs match, gone from the pool, or ``None``."""
        own = (row, input_hash)
        if own in self.unclaimed:
            self.unclaimed.remove(own)
            return own
        keys = self.units_with_hash_lowest_row_last.get(input_hash)
        if keys is None or input_hash in self.hashes_only_their_own_rows_may_claim:
            return None
        while keys:
            candidate = keys.pop()
            if candidate in self.unclaimed:  # not taken yet, or by its own row
                self.unclaimed.remove(candidate)
                return candidate
        return None


def _earlier_units_of_nodes(compiled: CompiledGraph, state: RunState, changed: set[str]) -> dict[str, _EarlierUnitsOfNode]:
    """Per node to re-plan and per node carrying work from a leg that stopped short, what it can reuse."""
    pools: dict[str, _EarlierUnitsOfNode] = {}
    for node_id, work in state.nodes.items():
        if node_id in changed or work.earlier_units:
            pool = _earlier_units_of_node(compiled, node_id, work, read_its_own_units=node_id in changed)
            if pool is not None:
                pools[node_id] = pool
    return pools


def _earlier_units_of_node(compiled: CompiledGraph, node_id: str, work: WorkOfNode, read_its_own_units: bool) -> _EarlierUnitsOfNode | None:
    """What this node can reuse: its own finished and waiting units, when it is re-planned, and those carried
    from earlier legs; or ``None`` when it can't reuse any: gone from the graph, a node that's a graph, opted out, nothing with a hash,
    or outputs that no longer read as its types."""
    node = compiled._nodes.get(node_id)
    if node is None or not isinstance(node.version, NodeVersion) or node.interface is None or not node.version.policy.reuse_when_inputs_match:
        return None
    outputs = [(out.name, Ref(node_id, out.name)) for out in node.interface.outputs]
    unclaimed: set[_EarlierUnitKey] = set()
    answered: set[_EarlierUnitKey] = set()
    only_their_own_rows: set[str] = set()
    waiting: dict[_EarlierUnitKey, WaitingUnit] = {}
    cells: dict[Ref, dict[_EarlierUnitKey, Any]] = {}
    problems: list[Problem] = []
    if read_its_own_units:
        hash_of_row = {row: input_hash for row, input_hash in work.finished_units or () if input_hash is not None}
        waiting.update(((unit.row, unit.input_hash), unit) for unit in work.waiting if unit.input_hash)
        unclaimed.update(hash_of_row.items(), waiting)
        answered.update((row, hash_of_row[row]) for row in work.answered_rows if row in hash_of_row)
        for name, ref in outputs:
            column = work.outputs.get(name)
            read = None if column is None else read_output_column(compiled, ref, column, problems)
            if read is not None:
                cells[ref] = {(row, input_hash): read[row] for row, input_hash in hash_of_row.items() if row in read}
    for unit in work.earlier_units:
        key = (unit.row, unit.input_hash)
        if key in unclaimed:  # a unit of this very leg at that row with that hash has the newer result
            continue
        unclaimed.add(key)
        if unit.answered:
            answered.add(key)
        if unit.only_its_own_row_may_claim:
            only_their_own_rows.add(unit.input_hash)
        if unit.waiting is not None:
            waiting[key] = unit.waiting
        for name, ref in outputs:
            column = unit.outputs.get(name)
            read = None if column is None else read_output_column(compiled, ref, column, problems)
            if read is not None and unit.row in read:
                cells.setdefault(ref, {})[key] = read[unit.row]
    if problems or not unclaimed:
        return None
    return _EarlierUnitsOfNode(unclaimed, answered, only_their_own_rows, waiting, cells)
