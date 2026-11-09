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
index's rows, ``CompiledField._makes_rows`` and its kin). A node's ``_runs_per_row_of`` is read the
same way, for other
callers: the engine runs only a graph whose every node is ready. Every
input carries a receive record (``CompiledField.receives``, from
``conductor._internal.graph.receive``). ``PerRow`` takes the value at the unit's own
row, ``Broadcast`` the one value there is, ``EntireValue`` everything on the
field, ``RowsUnder`` the rows under the unit's row cut to a depth, and
``Gather`` unrelated sources collected onto the input's own index. The
this module switches on the record to say when a unit is ready and what to
hand it, and derives none of this from types or indexes itself.

A blank is a value: ``None`` in a table's cell or written by a node. A
unit is not called when a blank reaches an input that does not say
``| None``: its outputs are blank at that row, a ``choice`` output is
skipped (no branch was chosen), and an output that makes rows has no rows
there. ``Series[X]`` leaves blanks out row by row. Order at a unit: not
ready, skip, blank.

A skip has a depth. ``SKIPPED`` written at row ``k`` means the field
holds nothing at ``k`` and at every row under it. A unit that returns
``SKIPPED`` writes it at its own row. A reader looks for a skip at its
own row and at every shorter prefix, and re-emits it at the depth it
found it. A unit whose every series output is skipped makes no rows, so
a node that would run per row of that index runs once at the shorter row
instead. That is how a skip keeps its reach down a chain.

A reduction over a group with no rows may be called with an empty series
just before a skip from a sibling row reaches above that group. What it
returned is then hidden by the skip, so a run's results don't depend on
which came first, though the number of calls can.

What it holds is the run's state: ``state()`` writes that down when a leg
ends or a checkpoint is due, and ``restore`` starts the next leg from it, working the rows out again
from the values. Nothing is pruned, so a unit done in one leg stays done in
the next. A value written here has already passed its output's check
(``conductor._internal.execution.validate_unit_outputs``); inside a leg it is trusted.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain
from operator import itemgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterator

from pydantic import ValidationError

from conductor._internal.errors import (
    CATALOGUE,
    InvalidCallError,
    NodeExecutionError,
    Problem,
    problem,
)
from conductor._internal.execution.run_state import OutputColumn, RunState, RunStatus, WaitingUnit
from conductor._internal.graph.receive import (
    Broadcast,
    EntireValue,
    Gather,
    Merge,
    PerRow,
    Receive,
    RowsUnder,
)
from conductor._internal.ref import Ref
from conductor._internal.series import Index, Row, Series
from conductor._internal.skip_and_ask import SKIPPED
from conductor._internal.table import (
    ROW_ID,
    MergeSource,
    column_frame,
    frame_with_no_rows,
    holds_tables,
    is_frame,
    merge,
    row_id,
)

if TYPE_CHECKING:
    import polars as pl

    from conductor._internal.field_records import Output
    from conductor._internal.graph.compiled import CompiledGraph
    from conductor._internal.graph.compiled_node import CompiledField
    from conductor._internal.graph.engine_read_plan import EngineReadPlan

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


@dataclass(frozen=True, slots=True)
class NotCalled:
    """This unit does not run: a blank reached these inputs, which take none."""

    blank_inputs: tuple[str, ...]


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


class LiveRunValues:
    """The live state of one run over one compiled graph, and what it makes ready.

    Built empty for a first leg, or from the ``state()`` of an earlier leg by
    ``restore``. The engine is the only caller.
    """

    def __init__(self, compiled: CompiledGraph) -> None:
        compiled._refuse_unless_runnable()
        self._compiled = compiled
        #: Every lookup the engine makes, in one place (``conductor._internal.graph.engine_read_plan``).
        self._plan = compiled._read_plan
        self._fields = self._plan.fields
        #: Every column of a table output offered as a field: its values are the
        #: table's cells, read where the table sits (``_cell``).
        self._views = self._plan.views
        self._values: dict[Ref, dict[Row, Any]] = {}
        #: The fields with a ``SKIPPED`` written at some row. Only these can
        #: hide a value under a skip above it, so a read of any other field is
        #: one lookup at its own row.
        self._skipped_on: set[Ref] = set()
        self._rows: dict[str, set[Row]] = {}
        #: Every ``(index, row)`` whose rows under that row are all made, and
        #: every ``(index, ())`` whose index is sealed. Rows under their
        #: immediate parent are always made in one go — a unit returns its
        #: whole list at once, a list set on a node is known before the run —
        #: so such a group that exists is complete, whoever made it. A row
        #: further up joins once every group under it has (``_groups_under_made``).
        self._groups: set[tuple[str, Row]] = set()
        #: Per index, its rows under each shorter row, in the order they were
        #: made; ``()`` holds them all. Finds the rows under a row without
        #: reading the whole index.
        self._rows_by_prefix: dict[str, dict[Row, list[Row]]] = {}
        self._sealed: set[str] = set()
        #: Per index, the rows beneath which it has no rows at all, because
        #: something above them was skipped.
        self._no_rows_under: dict[str, set[Row]] = {}
        #: Kept in the order units finished, which ``state`` then has little to sort.
        self._done: dict[Unit, None] = {}
        #: Per node that runs once per row, how many of its units are done:
        #: at a row of its index, and standing in at a shorter row.
        self._done_rows: dict[str, int] = {}
        self._done_standing_in: dict[str, int] = {}
        #: Units waiting on a person, each as the state carries it.
        self._waiting: dict[Unit, WaitingUnit] = {}
        #: Per node, how many rows it wasn't called on, per input that was blank there.
        self._rows_not_called: dict[str, dict[str, int]] = {}
        #: Per group a series is read by — a field and the row it is read
        #: under — its rows in order and how many from the first are written.
        #: Kept once every row of the group is made, so no row is read twice.
        self._written_rows: dict[tuple[Ref, Row], tuple[list[Row], int]] = {}
        #: Per group more than one level above its index's rows that is still
        #: being made, the parent rows under it and how many from the first are made.
        self._parent_rows_made: dict[tuple[str, Row], tuple[list[Row], int]] = {}

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
            for inp in self._plan.interfaces[node_id].inputs:
                typed = compiled.field(Ref(node_id, inp.name))
                if not typed._listed or typed._index.parent is not None:
                    continue
                for n, value in enumerate(node.statics[inp.name]):
                    self._made(typed._index.id, (n,))
                    self._values.setdefault(typed.ref, {})[(n,)] = value
                self._close(typed._index.id)
                roots.append(typed._index.id)
        self._make_root_list_rows(roots)

    def _make_root_list_rows(self, roots: list[str]) -> None:
        """Under every row of a list whose own index is a root, make the rows
        of every list whose index is its child, and seal them. Then seal the
        index of every node that makes rows per row of a list complete from
        the start."""
        next_up: list[str] = []
        for index_id in roots:
            for row in sorted(self._rows.get(index_id, ())):
                self._made_lists(index_id, row)
            next_up.extend(self._making_rows_on(index_id))
            next_up.extend(self._seal_listed(index_id, []))
        self._seal(next_up)

    # -- the read plan, as compile stored it -----------------------------------

    def _owner(self, index_id: str) -> CompiledField:
        """The field that makes an index's rows and carries its read plan: a
        series output, or an input holding a list set on its node, as compile
        recorded it."""
        return self._plan.owners[index_id]

    def _making_rows_on(self, index_id: str) -> list[str]:
        """The indexes whose rows are made by nodes running once per row of this one."""
        return [made for node_id in self._plan.run_per_row_by[index_id] for made in self._indexes_made_by(node_id)]

    def _indexes_made_by(self, node_id: str) -> list[str]:
        """The indexes this node's series outputs make rows on, one per output."""
        fields = self._fields
        return [
            index.id for out in self._plan.interfaces[node_id].outputs
            if (index := fields[Ref(node_id, out.name)]._makes_rows) is not None
        ]

    def _grouped_on(self, index_id: str) -> Iterator[tuple[str, Ref, int]]:
        """``(reader node, output read, group depth)`` for every reader of a
        series on this index that receives it grouped or whole, not one row
        at a time."""
        for ref in self._plan.carried_by[index_id]:
            for inp, received in self._plan.read_by[ref]:
                if not _per_row(received):
                    yield inp.node_id, ref, _rows_under_depth(received)

    def _typed_under(self, index_id: str) -> Iterator[tuple[Ref, str]]:
        """``(the input, its index)`` for each list set on a node, made under every row of this index."""
        for ref in self._plan.listed_inputs[index_id]:
            yield ref, self._fields[ref]._index.id

    # -- units ---------------------------------------------------------------

    def units(self, node_id: str) -> list[Unit]:
        """This node's units right now: one, or — running per row of ``L`` —
        one per row made on ``L``, plus one at each shorter row where ``L`` has no rows."""
        runs_per_row_of = self._plan.runs_per_row_of[node_id]
        if runs_per_row_of is None:
            return [(node_id, ())]
        no_rows_under = sorted(self._no_rows_under.get(runs_per_row_of.id, ()))
        rows = sorted(self._rows.get(runs_per_row_of.id, ()))
        return [(node_id, row) for row in (*no_rows_under, *rows)]

    def is_done(self, unit: Unit) -> bool:
        return unit in self._done

    def complete(self, node_id: str) -> bool:
        """Every unit this node will ever have is done."""
        runs_per_row_of = self._plan.runs_per_row_of[node_id]
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
        runs_per_row_of = self._plan.runs_per_row_of[node_id]
        return runs_per_row_of is not None and len(row) == runs_per_row_of.depth

    def progress(self, node_id: str) -> tuple[int, int | None]:
        """``(done, total)`` rows for a node; ``total`` is ``None`` until the
        node's index is sealed. Cover units stand in for rows that were never
        made, so they count in neither number."""
        runs_per_row_of = self._plan.runs_per_row_of[node_id]
        if runs_per_row_of is None:
            return int((node_id, ()) in self._done), 1
        total = len(self._rows.get(runs_per_row_of.id, ())) if runs_per_row_of.id in self._sealed else None
        return self._done_rows.get(node_id, 0), total

    def _units_under(self, node_id: str, row: Row) -> list[Unit]:
        """This node's units at ``row`` or beneath it, in no particular order —
        not counting a unit standing in at a shorter row, which is ready
        the moment it exists."""
        runs_per_row_of = self._plan.runs_per_row_of[node_id]
        if runs_per_row_of is None:
            return [(node_id, ())] if row == () else []
        if row in self._rows.get(runs_per_row_of.id, ()):
            return [(node_id, row)]
        return [(node_id, r) for r in self._rows_by_prefix.get(runs_per_row_of.id, {}).get(row, ())]

    # -- readiness -----------------------------------------------------------

    def ready(self, unit: Unit) -> bool:
        """Every value this unit needs exists, or its producer is done."""
        node_id, row = unit
        runs_per_row_of = self._plan.runs_per_row_of[node_id]
        if runs_per_row_of is not None and len(row) < runs_per_row_of.depth:
            return True  # standing in at a shorter row: the skip above it is already known
        return all(self._read_ready(ref, received, row) for ref, received in self._plan.reads[node_id])

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
        group under it. EntireValue or gathered: everything on the ref."""
        if _per_row(received):
            return self._present(ref, self._key(ref, row))
        return self._written(ref, row[: _rows_under_depth(received)])

    def _present(self, ref: Ref, key: Row) -> bool:
        return self._lookup(ref, key)[0] is not _MISSING

    def _key(self, ref: Ref, row: Row) -> Row:
        """Where a unit at ``row`` reads ``ref`` as a scalar: its row on the ref's index, or ``()``."""
        index = self._fields[ref]._index
        return () if index is None else row[: index.depth]

    def _written(self, ref: Ref, group: Row) -> bool:
        """Is every value of ``ref`` a series reader at ``group`` receives written?
        The one value of a field that carries a single value, else the rows under ``group``."""
        index = self._fields[ref]._index
        return self._present(ref, ()) if index is None else self._all_written(ref, index, group)

    def _all_written(self, ref: Ref, index: Index, parent_row: Row) -> bool:
        """Is every value of ``ref`` under ``parent_row`` written? True when the
        field is skipped there, or every row under it is made and produced."""
        if self._lookup(ref, parent_row)[0] is SKIPPED:
            return True
        no_rows_under = self._no_rows_under.get(index.id, ())
        if any(prefix in no_rows_under for prefix in _prefixes(parent_row)):
            return False  # the unit standing in at that shorter row has not run yet
        if len(parent_row) == index.depth:
            made = parent_row in self._rows.get(index.id, ())  # the group is the row itself
        else:
            made = (index.id, parent_row) in self._groups or self._groups_under_made(index, parent_row)
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

    def _groups_under_made(self, index: Index, row: Row) -> bool:
        """For ``row`` above ``index``'s immediate parent: are the rows under it
        all made? They are once the parent's rows under ``row`` are, and the
        rows under each of those. A parent row found made stays made, so how
        many are, from the first, is kept and none is looked at twice; the
        answer stays true, so it is kept as well."""
        parent = index.parent
        if parent is None or len(row) >= parent.depth:
            return False
        if (parent.id, row) not in self._groups and not self._groups_under_made(parent, row):
            return False
        group = (index.id, row)
        if group not in self._parent_rows_made:
            self._parent_rows_made[group] = (self._rows_under(parent.id, row), 0)
        parent_rows, made = self._parent_rows_made[group]
        no_rows_under = self._no_rows_under.get(index.id, ())
        while made < len(parent_rows) and (
            (index.id, parent_rows[made]) in self._groups or any(prefix in no_rows_under for prefix in _prefixes(parent_rows[made]))
        ):
            made += 1
        if made < len(parent_rows):
            self._parent_rows_made[group] = (parent_rows, made)
            return False
        del self._parent_rows_made[group]
        self._groups.add(group)
        return True

    # -- what a unit receives -------------------------------------------------

    def inputs_for(self, unit: Unit) -> dict[str, Any] | Skip | NotCalled:
        """The keyword arguments this unit runs with, or what stops it.

        A ``Skip`` comes from ``SKIPPED`` on an input received per row, on
        the series an input receives whole or reduced, or at a row above
        this unit's. A gather drops skipped sources instead — an empty
        gather is an empty series — because gathering is how "whichever
        branch fired" is expressed.
        """
        node_id, row = unit
        node = self._compiled.node(node_id)
        runs_per_row_of = self._plan.runs_per_row_of[node_id]
        if runs_per_row_of is not None and len(row) < runs_per_row_of.depth:
            return Skip(at=row)
        values: dict[str, Any] = {}
        statics = node.statics
        for inp in self._plan.interfaces[node_id].inputs:
            own = Ref(node_id, inp.name)
            binding = self._fields[own].binding
            received = self._fields[own]._receives
            if binding is None:
                if isinstance(received, EntireValue):
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
                values[inp.name] = merged
            elif isinstance(received, Gather):
                gathered: list[Any] = []
                for ref in binding.refs:
                    index = self._fields[ref]._index
                    value, _ = self._lookup(ref, ())
                    if value is SKIPPED:
                        continue
                    if index is None:
                        gathered.append(value)
                    else:
                        gathered.extend(self._series(node_id, (ref,), index, self._rows_under(index.id, ())).values)
                values[inp.name] = Series(received.index, gathered)
            else:
                group = row[: _rows_under_depth(received)]
                found = [self._lookup(ref, group) for ref in binding.refs]
                if all(value is SKIPPED for value, _ in found):
                    return Skip(at=max((at for _, at in found), key=len))
                index = self._fields[binding.refs[0]]._index
                if index is None:
                    # Received whole from a source that ran once: the one value.
                    ((values[inp.name], _),) = found
                else:
                    values[inp.name] = self._series(node_id, binding.refs, index, self._rows_under(index.id, group))
        blank = tuple(name for name in node._inputs_refusing_a_blank if values.get(name, _MISSING) is None)
        if blank:
            return NotCalled(blank)
        if node._series_inputs_leaving_blanks_out:
            _leave_out_blank_rows(values, node._series_inputs_leaving_blanks_out)
        return values

    def _merged(self, received: Merge, row: Row) -> pl.DataFrame | None:
        """The one table a unit at ``row`` receives from every source of a ``Table``
        input, or ``None`` when every source is skipped there. Reads each source's
        rows; ``conductor._internal.table.merge`` matches them. A skipped source has no rows
        and still counts as a source."""
        sources: list[MergeSource] = []
        skipped = 0
        for source, part in received.parts:
            field = self._fields[source]
            is_table = holds_tables(field.type)
            index: Index = field._makes_rows if is_table else field._index  # type: ignore[assignment]
            group = row[: part.index.depth] if isinstance(part, PerRow) else row[: _rows_under_depth(part)]
            stored, _ = self._lookup(source, group)
            is_skip = stored is SKIPPED
            skipped += is_skip
            if is_table:
                frame = frame_with_no_rows(field.type) if is_skip else stored
                sources.append(MergeSource(source.node_id, frame, index, [(*group, n) for n in range(frame.height)]))
                continue
            present = [
                (at, value) for at in self._rows_under(index.id, group)
                if (value := self._lookup(source, at)[0]) is not SKIPPED and value is not _MISSING
            ]
            frame = column_frame(field._title, field._cell_type, [self._row_id(index, at) for at, _ in present], [value for _, value in present])
            sources.append(MergeSource(source.node_id, frame, index, [at for at, _ in present]))
        return None if skipped == len(sources) else merge(sources)

    def _row_id(self, index: Index, row: Row) -> str:
        """The id of the row at ``row`` on ``index``: on a table's rows, the id
        its stored table carries, which may have come in with the table; on
        any other index, the id spelled from the index and the row."""
        owner = self._plan.owners.get(index.id)
        if owner is not None and owner._view_of is None and owner._is_output and holds_tables(owner.type):
            stored, _ = self._lookup(owner.ref, row[:-1])
            return stored[ROW_ID][row[-1]]
        return row_id(index, row)

    def _covering(self, node_id: str, refs: tuple[Ref, ...], row: Row) -> tuple[list[Any], list[Row]]:
        """Of several refs read at one row: the values that cover it, and the
        rows at which the others were skipped. Two covering the same row
        means two edges both supplied that row, and the node fails with
        that row, named in the failure's ``ctx``."""
        covering: list[Any] = []
        skips: list[Row] = []
        for ref in refs:
            value, at = self._lookup(ref, self._key(ref, row))
            if value is _MISSING:
                raise KeyError(f"{ref} has no value at {at} — the unit was not ready")
            if value is SKIPPED:
                skips.append(at)
            else:
                covering.append(value)
        if len(covering) > 1:
            raise NodeExecutionError(CATALOGUE["row_covered_twice"][0], node_id=node_id, code="row_covered_twice", ctx={"edges": len(covering), "row": list(row)})
        return covering, skips

    def _typed(self, received: Receive, value: Any, own: Ref, row: Row = ()) -> Any:
        """A value the author typed (or the declared default) as the unit receives it.

        Received whole, it is a series on the input's own index; received
        per row of an index of the input's own, it is a list set on the node
        and the unit takes the value at its row; broadcast, it is the one
        value.
        """
        if isinstance(received, EntireValue):
            return Series(self._fields[own]._index, value.values if isinstance(value, Series) else list(value))
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
        if len(key) < self._fields[ref]._index.depth:
            # A group above the table's rows: only a skip on the table reaches it.
            stored, at = self._lookup(table, key)
            return (stored, at) if stored is SKIPPED else (_MISSING, key)
        stored, at = self._lookup(table, key[:-1])
        if stored is _MISSING or stored is SKIPPED:
            return stored, at
        cells = stored[column]
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

    def record(self, unit: Unit, outputs: dict[str, Any] | Skip | NotCalled) -> list[Unit]:
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
        rows on its columns' index only when something reads a column. For a
        ``NotCalled`` unit each output is its blank (``_blank_of``).

        The units returned are the ones this write could have made ready and
        that answer ``ready`` now, in execution order; the engine starts
        those and asks nothing else.
        """
        node_id, row = unit
        node = self._compiled.node(node_id)
        assert node.wiring is not None  # the engine runs only a runnable graph
        fields = node.wiring.fields
        written: list[tuple[Ref, Row]] = []
        made: dict[str, list[Row]] = {}
        no_rows_under: list[tuple[str, Row]] = []
        making: list[str] = []
        skip_at = outputs.at if isinstance(outputs, Skip) else None
        not_called = outputs if isinstance(outputs, NotCalled) else None
        if not_called is not None:
            counts = self._rows_not_called.setdefault(node_id, {})
            for name in not_called.blank_inputs:
                counts[name] = counts.get(name, 0) + 1
        for out in self._plan.interfaces[node_id].outputs:
            field = fields[out.name]
            ref = field.ref
            if skip_at is not None:
                value = SKIPPED
            elif not_called is not None:
                value = _blank_of(field, out)
            else:
                value = outputs[out.name]  # type: ignore[index]
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
            elif not field._holds_many:
                self._write(ref, at, value, written)
                if self._columns_read(index.id):
                    keys = made.setdefault(index.id, [])
                    for j in range(value.height):
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

    def rows_not_called(self, node_id: str) -> dict[str, int]:
        """How many rows this node wasn't called on, per input that was blank there."""
        return self._rows_not_called.get(node_id, {})

    def _columns_read(self, index_id: str) -> bool:
        """Does anything read a column of the table that owns this index? If
        nothing does, a table passing through makes no rows at all."""
        plan = self._plan
        return bool(plan.run_per_row_by[index_id]) or any(plan.read_by[view] for view in plan.carried_by[index_id])

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
        for ref, child in self._typed_under(index_id):
            cells = self._values.setdefault(ref, {})
            for n, value in enumerate(self._compiled.node(ref.node_id).statics[ref.field]):
                if self._made(child, (*row, n)):
                    made.append((child, (*row, n)))
                    made.extend(self._made_lists(child, (*row, n)))
                cells[(*row, n)] = value
            self._groups.add((child, row))
        return made

    def _no_rows_under_lists(self, index_id: str, at: Row) -> list[tuple[str, Row]]:
        """Where ``index_id`` has no rows under ``at``, neither has a list set
        on a node beneath it; returns each ``(index, row)`` marked."""
        marked: list[tuple[str, Row]] = []
        for _, child in self._typed_under(index_id):
            self._no_rows_under.setdefault(child, set()).add(at)
            marked.append((child, at))
            marked.extend(self._no_rows_under_lists(child, at))
        return marked

    def _seal_listed(self, index_id: str, sealed: list[str]) -> list[str]:
        """Seal the lists set on nodes beneath a sealed index: every parent
        row is made, so every row under them is. Returns the indexes made by
        nodes running per row of the ones sealed, for ``_seal`` to try next."""
        next_up: list[str] = []
        for _, child in self._typed_under(index_id):
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
        self._done[unit] = None
        node_id, _ = unit
        if self._plan.runs_per_row_of[node_id] is not None:
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
            for inp, received in self._plan.read_by[ref]:
                depth = _rows_under_depth(received)
                if _per_row(received) or len(key) < depth:
                    woken.update(self._units_under(inp.node_id, key))
                else:
                    self._wake_group(ref, inp.node_id, key[:depth], woken)
        for index_id, keys in made.items():
            readers = self._plan.run_per_row_by[index_id]
            for key in keys:
                woken.update((reader, key) for reader in readers)
        for index_id, key in (*no_rows_under, *lists):
            woken.update((reader, key) for reader in self._plan.run_per_row_by[index_id])
        for index_id, key in lists:
            for reader, ref, depth in self._grouped_on(index_id):
                if 0 < depth <= len(key):
                    self._wake_group(ref, reader, key[:depth], woken)
        for index_id in making:
            keys = made.get(index_id, [])
            for reader, ref, depth in self._grouped_on(index_id):
                for key in keys:
                    if depth == len(key):
                        self._wake_group(ref, reader, key, woken)
                if 0 < depth <= len(row):  # a whole read (depth 0) waits for the seal below
                    self._wake_group(ref, reader, row[:depth], woken)
            if not keys:  # no rows here means none further down, so nothing below will write
                self._wake_readers_further_down(index_id, row, woken)
        for index_id in sealed:
            for reader, ref, depth in self._grouped_on(index_id):
                if depth == 0:
                    self._wake_group(ref, reader, (), woken)
        ready = [u for u in woken if u not in self._done and u not in self._waiting and self.ready(u)]
        return sorted(ready, key=lambda u: (self._position[u[0]], u[1]))

    def _wake_readers_further_down(self, index_id: str, row: Row, woken: set[Unit]) -> None:
        """Wake the readers of every index under ``index_id`` whose group sits at or above ``row``."""
        for child in self._making_rows_on(index_id):
            for reader, ref, depth in self._grouped_on(child):
                if 0 < depth <= len(row):
                    self._wake_group(ref, reader, row[:depth], woken)
            self._wake_readers_further_down(child, row, woken)

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
        return node_result(self._plan, node_id, self._values)

    # -- the state ----------------------------------------------------------------

    def state(self, status: RunStatus, failure: Problem | None = None) -> RunState:
        """Everything this leg has produced, as the ``RunState`` a host stores.

        Built without validation from values that already passed, and holding
        them typed: the next read costs nothing to decode, and a dump writes
        each one by its own type."""
        columns: dict[Ref, OutputColumn] = {}
        for ref in sorted(self._values):
            if self._fields[ref]._listed:  # the graph holds a list set on a node, never a state
                continue
            by_row = self._values[ref]
            rows: list[Row] = []
            values: list[Any] = []
            skipped: list[Row] = []
            for row, value in sorted(by_row.items(), key=itemgetter(0)):
                if value is SKIPPED:
                    skipped.append(row)
                else:
                    rows.append(row)
                    values.append(value)
            columns[ref] = OutputColumn.model_construct(rows=rows, values=values, skipped=skipped)
        compiled = self._compiled
        order = compiled.execution_order
        finished_by_node: dict[str, list[Row]] = {}
        for node_id, row in self._done:
            finished_by_node.setdefault(node_id, []).append(row)
        return RunState.model_construct(
            status=status,
            failure=failure,
            waiting=self.waiting(),
            output_columns=columns,
            # A node complete on no rows is here with none: the state says it finished.
            finished_rows={node_id: sorted(finished_by_node.get(node_id, ())) for node_id in order if node_id in finished_by_node or self.complete(node_id)},
            node_fingerprints={node_id: compiled.node(node_id).fingerprint for node_id in order},
            rows_not_called={node_id: dict(sorted(self._rows_not_called[node_id].items())) for node_id in order if node_id in self._rows_not_called},
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
        ``InvalidCallError`` with the code ``state_unreadable``, the columns'
        faults first and then the units'. A ``None`` passes on any output: a
        blank is a legal value in a state.
        """
        restored = cls(compiled)
        dropped = set(compiled.nodes_changed_since(state))
        values, faults = read_output_columns(compiled, state, dropped)
        restored._values.update(values)
        if faults:
            raise InvalidCallError(faults)
        restored._skipped_on = {ref for ref in values if state.output_columns[ref].skipped}
        finished = {node_id: rows for node_id, rows in state.finished_rows.items() if node_id not in dropped}
        done = [(node_id, row) for node_id, rows in finished.items() for row in rows]
        restored._waiting = {(asked.node_id, asked.row): asked for asked in state.waiting if asked.node_id not in dropped}
        restored._remake_rows(done)
        restored._rows_not_called = {node_id: dict(counts) for node_id, counts in state.rows_not_called.items() if node_id not in dropped}
        faults = restored._faults_in_units(finished)
        if faults:
            raise InvalidCallError(faults)
        for unit in done:
            restored._finish(unit)
        restored._seal([
            index_id for node_id in reversed(compiled.execution_order) if node_id not in dropped
            for index_id in restored._indexes_made_by(node_id)
        ])
        return restored

    def _faults_in_units(self, finished: Mapping[str, list[Row]]) -> list[Problem]:
        """What is wrong with the stored units once the rows are made again: a
        unit, finished or waiting, on a row that was never made or kept empty, or
        a finished unit one of whose one-value outputs holds nothing at its row."""
        faults: list[Problem] = []
        by_node = {node_id: set(rows) for node_id, rows in finished.items()}
        stored = {node_id: set(rows) for node_id, rows in by_node.items()}
        for node_id, row in self._waiting:
            stored.setdefault(node_id, set()).add(row)
        for node_id, rows in stored.items():
            index = self._plan.runs_per_row_of[node_id]
            stray = rows - {()} if index is None else rows.difference(self._rows.get(index.id, ()), self._no_rows_under.get(index.id, ()))
            faults.extend(
                problem("state_unreadable", "state", node_id, *row, reason=f"the state has {node_id!r} at row {list(row)}, which {node_id!r} has no place for")
                for row in sorted(stray)
            )
        for node_id, rows in by_node.items():
            for out in self._plan.interfaces[node_id].outputs:
                ref = Ref(node_id, out.name)
                if self._fields[ref]._holds_many:
                    continue
                faults.extend(
                    problem("state_unreadable", "state", ref, *row, reason=f"the state has {node_id!r} finished at row {list(row)}, with no value for {ref}")
                    for row in sorted(rows - self._values.get(ref, _NO_VALUES).keys())
                    if self._lookup(ref, row)[0] is _MISSING
                )
        return faults

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
        rows_under_parent: dict[Ref, dict[Row, list[Row]]] = {}
        for node_id, row in done:
            for out in self._plan.interfaces[node_id].outputs:
                ref = Ref(node_id, out.name)
                index = self._fields[ref]._makes_rows
                if index is None:
                    continue
                self._rows.setdefault(index.id, set())
                by_row = self._values.get(ref, {})
                at = next((key for key in _prefixes(row) if by_row.get(key) is SKIPPED), None)
                if at is not None:
                    self._no_rows_under.setdefault(index.id, set()).add(at)
                    self._no_rows_under_lists(index.id, at)
                    continue
                stored = by_row.get(row)
                if is_frame(stored):
                    if self._columns_read(index.id):
                        made.setdefault(index.id, set()).update((*row, j) for j in range(stored.height))
                else:
                    if ref not in rows_under_parent:
                        rows_under_parent[ref] = _rows_by_parent(by_row)
                    made.setdefault(index.id, set()).update(rows_under_parent[ref].get(row, ()))
                self._groups.add((index.id, row))
        for node_id in self._compiled.execution_order:
            for index_id in self._indexes_made_by(node_id):
                for key in sorted(made.get(index_id, ())):
                    self._made(index_id, key)
                    self._made_lists(index_id, key)


def _blank_of(field: CompiledField, out: Output) -> Any:
    """What a node that was not called writes on an output: a ``choice`` output
    is skipped, since no branch was chosen; an output that makes rows has none
    there; any other is blank."""
    if out.choice is not None:
        return SKIPPED
    if field._holds_many:
        return []
    if field._makes_rows is not None:
        return frame_with_no_rows(field._cell_type)
    return None


def _leave_out_blank_rows(values: dict[str, Any], names: tuple[str, ...]) -> None:
    """Leave a row out of every series input on an index when one of ``names`` is blank there.

    ``names`` are the series inputs that take no blank. A row leaves all of
    the series on its index, or a node zipping them would pair wrong values."""
    blank_rows: dict[Index, set[Row]] = {}
    for name in names:
        series = values.get(name)
        if isinstance(series, Series) and None in series.values:
            blank_rows.setdefault(series.index, set()).update(row for row, value in zip(series.rows, series.values, strict=True) if value is None)
    for name, value in values.items():
        if isinstance(value, Series) and value.index in blank_rows:
            gone = blank_rows[value.index]
            kept = [(row, cell) for row, cell in zip(value.rows, value.values, strict=True) if row not in gone]
            values[name] = Series(value.index, [cell for _, cell in kept], rows=[row for row, _ in kept])


def _rows_by_parent(by_row: Mapping[Row, Any]) -> dict[Row, list[Row]]:
    """A column's rows grouped under the row one level above each."""
    grouped: dict[Row, list[Row]] = {}
    for row in by_row:
        if row:
            grouped.setdefault(row[:-1], []).append(row)
    return grouped


def read_output_columns(compiled: CompiledGraph, state: RunState, dropped: set[str]) -> tuple[dict[Ref, dict[Row, Any]], list[Problem]]:
    """What ``state`` holds on every output of the graph, by row, each value
    read through its output's check, and every fault found.

    The one reader of ``state.output_columns``: a leg restores from it and
    ``compiled.results`` reads from it. ``dropped`` (``nodes_changed_since``)
    is left out."""
    values: dict[Ref, dict[Row, Any]] = {}
    faults: list[Problem] = []
    for ref, column in state.output_columns.items():
        if ref.node_id not in dropped:
            read = _read_output_column(compiled, ref, column, faults)
            if read is not None:
                values[ref] = read
    return values, faults


def _read_output_column(compiled: CompiledGraph, ref: Ref, column: OutputColumn, faults: list[Problem]) -> dict[Row, Any] | None:
    """One output's stored rows and cells, or ``None`` and the faults that stop them."""
    try:
        field = compiled.field(ref)
    except KeyError:
        faults.append(problem("state_unreadable", "state", ref, reason=f"the state has a value for {ref}, which the graph does not have"))
        return None
    depth = 0 if field._index is None else field._index.depth
    stored_rows = (*column.rows, *column.skipped)
    misplaced: list[Row] = []
    if max(map(len, stored_rows), default=0) > depth or min(chain.from_iterable(stored_rows), default=0) < 0:
        misplaced = [row for row in stored_rows if len(row) > depth or min(row, default=0) < 0]
    for row in misplaced:
        faults.append(problem("state_unreadable", "state", ref, *row, reason=f"the state's value for {ref} is at row {list(row)}, which {ref} has no place for"))
    if misplaced:
        return None
    by_row: dict[Row, Any] = {}
    read = field._read_cell
    for row, stored in zip(column.rows, column.values, strict=True):
        try:
            by_row[row] = _restored_cell(field, stored, read)
        except (TypeError, ValueError) as unreadable:
            faults.append(problem("state_unreadable", "state", ref, *row, reason=f"the state's value for {ref} at row {list(row)} does not read back: {_reason(unreadable)}"))
    for row in column.skipped:
        by_row[row] = SKIPPED
    return by_row


def _restored_cell(field: CompiledField, stored: Any, read: Any) -> Any:
    """A stored value as its cell. A value already of its type came from a
    leg in this process and is as that leg left it; a ``None`` is a blank. Any
    other goes through the output's check, and a table has to have kept its row ids."""
    if stored is None or type(stored) is field._cell_type:
        return stored
    cell = read(stored)
    if field._makes_rows is not None and not field._holds_many and ROW_ID not in cell.columns:
        raise ValueError("a stored table has lost the ids of its rows")
    return cell


def node_result(plan: EngineReadPlan, node_id: str, values: Mapping[Ref, Mapping[Row, Any]]) -> dict[str, Any] | None:
    """A node's outputs by name, read off what each holds, or ``None`` when
    the node did not run (every output is ``SKIPPED`` at the top).

    A result holds only what the node produced: an output the node skipped
    (the branch not taken) is absent, never ``SKIPPED``. A scalar output of a
    node that ran once is its value; anything on an index is a ``Series`` of
    the rows written so far, sparse where rows were skipped."""
    outputs = plan.interfaces[node_id].outputs
    by_output = {out.name: values.get(Ref(node_id, out.name), _NO_VALUES) for out in outputs}
    if outputs and all(by_row.get(()) is SKIPPED for by_row in by_output.values()):
        return None
    result: dict[str, Any] = {}
    for name, by_row in by_output.items():
        at_top = by_row.get((), _MISSING)
        if at_top is SKIPPED:
            continue
        index = plan.fields[Ref(node_id, name)]._index
        if index is not None:
            rows = _rows_with_a_value(by_row)
            result[name] = Series(index, [by_row[row] for row in rows], rows=rows)
        elif at_top is not _MISSING:
            result[name] = at_top
    return result


def _rows_with_a_value(by_row: Mapping[Row, Any]) -> list[Row]:
    """A field's rows in order, leaving out the skipped ones and those under a skip."""
    skipped = {row for row, value in by_row.items() if value is SKIPPED}
    rows = sorted(by_row)
    return [row for row in rows if not any(prefix in skipped for prefix in _prefixes(row))] if skipped else rows


def _reason(unreadable: Exception) -> str:
    """Why a value did not read: pydantic's first message when pydantic said, never the value."""
    if isinstance(unreadable, ValidationError):
        return unreadable.errors(include_url=False, include_input=False, include_context=False)[0]["msg"]
    return str(unreadable)
