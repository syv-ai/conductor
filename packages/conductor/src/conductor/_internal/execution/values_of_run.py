"""How a run holds a field's values: the one module that knows.

Compile says once, per field, how its values are held (``CompiledField._held_as``):

- ``OneValue``: one value at the row of the unit that wrote it.
- ``CellsInAList``: the unit writes one list at its row, and each item is a row
  of the index the output makes.
- ``ListOnNode``: the same shape, from a list set on the node; no unit writes it.
- ``TableAtRow``: the unit writes one table at its row, and each table row is a
  row of the index the output makes.
- ``ColumnOfTable``: nothing of its own; each cell is read from the table above it.
- ``NotHeld``: a plain input, read from its sources and kept under no address.

Every ``match`` over these records is in this module, so what is written, read
back, stored in a state or read out of one is decided here for all of them.
A reader outside asks one thing: ``rows_made``, the index a field makes, or
``None``. Each record has it.

A skip belongs to a row: ``SKIPPED`` written at a row means the field holds
nothing at that row or at any row under it, and a skip on an output that makes
rows leaves its index, and each index under it, without rows there
(``RowsOfIndex.no_rows_under``). ``_row_a_skip_reaches`` is that rule, and
nothing else in the engine walks a row's prefixes looking for one.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from itertools import chain
from operator import itemgetter
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import ValidationError

from conductor._internal.errors import Problem, make_problem
from conductor._internal.execution.run_state import OutputColumn, RunState
from conductor._internal.index import Index, Row, ValuesOnRows
from conductor._internal.ref import Ref
from conductor._internal.skip_and_ask import SKIPPED
from conductor._internal.table import (
    ROW_ID_COLUMN,
    frame_with_no_rows,
    global_row_id,
    is_frame,
    no_row_kept_its_id,
    with_ids_spelled_at_row,
    with_received_ids_moved,
    with_row_ids,
)

if TYPE_CHECKING:
    import polars as pl

    from conductor._internal.graph.compiled_graph import CompiledGraph
    from conductor._internal.graph.compiled_node import CompiledField
    from conductor._internal.graph.engine_lookups import EngineLookups

#: What is at a row where nothing has been written.
MISSING = object()
#: What a field nothing has been written to holds: no row.
NO_VALUES: Mapping[Any, Any] = MappingProxyType({})


@dataclass(frozen=True, slots=True)
class OneValue:
    """One value at the row of the unit that wrote it: the row itself, or ``()`` for a node that runs once."""

    rows_made: ClassVar[None] = None


@dataclass(frozen=True, slots=True)
class CellsInAList:
    """A many-valued output: its unit writes a list at its row, one item per row of ``rows_made`` under that row."""

    rows_made: Index


@dataclass(frozen=True, slots=True)
class ListOnNode:
    """A list set on a node's input: ``cells`` are written under each row above them, and no unit writes it."""

    rows_made: Index
    cells: list[Any]


@dataclass(frozen=True, slots=True)
class TableAtRow:
    """A table output: its unit writes a table at its row, one table row per row of ``rows_made`` under that row."""

    rows_made: Index


@dataclass(frozen=True, slots=True)
class ColumnOfTable:
    """One column of the table output ``table``, offered as a field; each cell is read from the table above it."""

    table: Ref
    column: str
    rows_made: ClassVar[None] = None


@dataclass(frozen=True, slots=True)
class NotHeld:
    """A plain input: what a unit receives is read from its sources, and nothing is kept under this address."""

    rows_made: ClassVar[None] = None


HeldAs = OneValue | CellsInAList | ListOnNode | TableAtRow | ColumnOfTable | NotHeld


def is_held_at_its_own_row(field: CompiledField) -> bool:
    """A read of this field finds its value at the row it's read at, with no position to work out."""
    return field._held_as.__class__ in (OneValue, TableAtRow)


def is_list_on_node(field: CompiledField) -> bool:
    return field._held_as.__class__ is ListOnNode


def holds_values(field: CompiledField) -> bool:
    """Can an edge read this field? A plain input holds nothing to read."""
    return field._held_as.__class__ is not NotHeld


def blank_of(field: CompiledField) -> Any:
    """What this field holds where a blank is written on it: an output that makes rows makes none, any other holds ``None``."""
    match field._held_as:
        case CellsInAList():
            return []
        case TableAtRow():
            return frame_with_no_rows(field.type)
        case _:
            return None


def output_problem(output: str, message: str, *, errors: tuple[dict[str, Any], ...] = (), code: str = "invalid_output") -> Problem:
    """One output that does not fit: which, a sentence that never quotes the value, and pydantic's own errors when its check said why."""
    return Problem(type=code, loc=(output,), msg=message, ctx={"errors": errors})


def held_form_of_return(field: CompiledField, name: str, row: Row, value: Any, row_ids_must_survive: frozenset[Ref]) -> Any | Problem:
    """``value``, returned on this output by the unit at ``row``, as the run holds it, or the ``Problem`` it is.

    Raises pydantic's ``ValidationError`` where the output's own type refuses it."""
    held_as = field._held_as
    if value is None and field._takes_blank:
        return blank_of(field)
    if held_as.__class__ is OneValue:
        return field._validate_cell(value)
    match held_as:
        case CellsInAList():
            if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
                return output_problem(name, f"'{name}' is many values and must be a sequence, not a {type(value).__name__}")
            read = field._validate_cell
            return [item if item is SKIPPED else read(item) for item in value]
        case TableAtRow(rows_made=rows_made):
            table = field._validate_cell(value)
            if field.ref in row_ids_must_survive and no_row_kept_its_id(table):
                return output_problem(name, f"'{name}' lost the ids of its rows, so they can't be matched to the table it was built from", code="row_ids_lost")
            return with_row_ids(table, rows_made, row)
        case _:
            return field._validate_cell(value)


def value_for_another_row(field: CompiledField, value: Any, old_row: Row, row: Row, received_then: list[pl.DataFrame], received_now: list[pl.DataFrame]) -> Any:
    """What a unit that was at ``old_row`` wrote, as a unit at ``row`` writes it: a table's
    own row ids follow the row, and the ids it took in follow the tables this unit received."""
    match field._held_as:
        case TableAtRow(rows_made=rows_made) if is_frame(value):
            if old_row != row:
                value = with_ids_spelled_at_row(value, rows_made, old_row, row)
            return with_received_ids_moved(value, received_then, received_now) if received_then else value
        case _:
            return value


def output_column_of(by_row: Mapping[Row, Any], field: CompiledField) -> OutputColumn:
    """One output as a state writes it. On an output that makes rows, a row's
    value is the list of cells under it; a cell skipped alone is listed in ``skipped`` and a ``None`` stands in its place."""
    many_cells = field._held_as.__class__ is CellsInAList
    rows: list[Row] = []
    values: list[Any] = []
    skipped: list[Row] = []
    for row, value in sorted(by_row.items(), key=itemgetter(0)):
        if value is SKIPPED:
            skipped.append(row)
            continue
        rows.append(row)
        if many_cells:
            holes = [position for position, cell in enumerate(value) if cell is SKIPPED]
            if holes:
                value = list(value)
                for position in holes:
                    value[position] = None
                    skipped.append((*row, position))
        values.append(value)
    skipped.sort()
    return OutputColumn._built_from_checked_fields(rows=rows, values=values, skipped=skipped)


def read_output_column(compiled: CompiledGraph, ref: Ref, column: OutputColumn, problems: list[Problem]) -> dict[Row, Any] | None:
    """One output's stored rows and cells, or ``None`` and the problems that stop them.

    A ``None`` passes as a value (a blank is legal in a state) except where an
    output's table makes rows, which it can't do from a ``None``."""
    try:
        field = compiled.field(ref)
    except KeyError:
        problems.append(make_problem("state_unreadable", "state", ref, reason=f"the state has a value for {ref}, which the graph does not have"))
        return None
    depth = 0 if field._index is None else field._index.depth
    many_cells = field._held_as.__class__ is CellsInAList
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
    a_blank_passes = field._held_as.__class__ is not TableAtRow
    for row, stored in zip(column.rows, column.values, strict=True):
        try:
            by_row[row] = _restored_list_of_cells(field, stored, read) if many_cells else stored if (stored is None and a_blank_passes) or type(stored) is cell_type else _restored_cell(field, stored, read)
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
    the caller keeps those as they are. A table has to be one, with its row ids."""
    if field._held_as.__class__ is TableAtRow:
        if stored is None:
            raise TypeError("null, where a table makes rows")
        table = read(stored)
        if ROW_ID_COLUMN not in table.columns:
            raise ValueError("a stored table has lost the ids of its rows")
        return table
    return read(stored)


def _read_failure_message(unreadable: Exception) -> str:
    """Why a value did not read: pydantic's first message when pydantic said, never the value."""
    if isinstance(unreadable, ValidationError):
        return unreadable.errors(include_url=False, include_input=False, include_context=False)[0]["msg"]
    return str(unreadable)


def _row_a_skip_reaches(by_row: Mapping[Row, Any], no_rows_under: Collection[Row], row: Row) -> Row | None:
    """The shortest prefix of ``row``, ``row`` itself included, where a skip left this field nothing, if any."""
    for depth in range(len(row) + 1):
        row_on_path = row[:depth]
        if by_row.get(row_on_path) is SKIPPED or row_on_path in no_rows_under:
            return row_on_path
    return None


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


class ValuesOfRun:
    """What one run holds: each field's values by row, the rows of each index, and where a skip left nothing.

    The ledger writes through ``write`` and reads through ``value_or_skip_at``.
    It reads ``rows`` directly to know which rows exist. A state is read in
    with ``read_state``, and ``compiled.results`` goes through the same reader
    as a leg restoring from it.
    """

    __slots__ = ("rows", "_lookups", "_fields", "_fields_held_at_their_own_row", "_values", "_fields_with_a_skip")

    def __init__(self, lookups: EngineLookups) -> None:
        self._lookups = lookups
        self._fields = lookups.fields
        self._fields_held_at_their_own_row = lookups.fields_held_at_their_own_row
        self._values: dict[Ref, dict[Row, Any]] = {}
        #: The fields that may hold ``SKIPPED`` at a row, or sit on an index with
        #: no rows beneath some row. Only these can have a skip above a value,
        #: so a read of any other field is one lookup at its own row.
        self._fields_with_a_skip: set[Ref] = set()
        #: Per index id, its rows so far.
        self.rows: dict[str, RowsOfIndex] = {}
        for index_id, field in lookups.field_making_rows_of_index.items():
            index = field._held_as.rows_made
            assert index is not None  # a field owns an index
            self.rows[index_id] = RowsOfIndex(index.depth)
        for index_id, rows in self.rows.items():
            rows.child_indexes = [self.rows[child] for child in lookups.indexes_under_index[index_id]]
            for child in rows.child_indexes:
                child.parent = rows
        for ref in lookups.lists_on_nodes_at_the_top:
            self._write_list_on_node_under(ref, (), [])

    # -- writing -----------------------------------------------------------------

    def write(self, field: CompiledField, row: Row, value: Any, rows_made: list[tuple[str, Row, int]]) -> None:
        """Write ``value`` at ``row`` on this output, and note in ``rows_made`` what it made: ``SKIPPED``
        leaves the field, and on a row-making output its index, nothing there; a list or a table makes one row per item."""
        self._values.setdefault(field.ref, {})[row] = value
        if value is SKIPPED:
            self._note_skip_at(field, row)
        elif (index := field._held_as.rows_made) is not None:
            self._make_rows_under(index.id, row, len(value), rows_made)

    def _note_skip_at(self, field: CompiledField, row: Row) -> None:
        self._fields_with_a_skip.add(field.ref)
        index = field._held_as.rows_made
        if index is not None:
            rows = self.rows[index.id]
            if not rows.no_rows_under:
                self._note_fields_with_a_skip_on(index.id)
            rows.mark_no_rows_under(row)

    def _note_fields_with_a_skip_on(self, index_id: str) -> None:
        self._fields_with_a_skip.update(self._lookups.outputs_on_index[index_id])
        for child in self._lookups.indexes_under_index[index_id]:
            self._note_fields_with_a_skip_on(child)

    def _write_list_on_node_under(self, ref: Ref, parent_row: Row, rows_made: list[tuple[str, Row, int]]) -> None:
        """Make, under ``parent_row``, the rows of the list set on a node at ``ref``, and write it there."""
        held_as = self._fields[ref]._held_as
        assert isinstance(held_as, ListOnNode)
        self._values.setdefault(ref, {})[parent_row] = held_as.cells
        self._make_rows_under(held_as.rows_made.id, parent_row, len(held_as.cells), rows_made)

    def _make_rows_under(self, index_id: str, parent_row: Row, count: int, rows_made: list[tuple[str, Row, int]]) -> None:
        """Make these rows, and under each the rows of every list on a node whose index is this one's child."""
        self.rows[index_id].make_rows_under(parent_row, count)
        rows_made.append((index_id, parent_row, count))
        for ref in self._lookups.lists_on_nodes_under_index[index_id]:
            for position in range(count):
                self._write_list_on_node_under(ref, (*parent_row, position), rows_made)

    # -- reading ------------------------------------------------------------------

    def value_or_skip_at(self, ref: Ref, row: Row) -> tuple[Any, Row]:
        """``(value, row)`` for the value at ``(ref, row)``: ``SKIPPED`` at the
        shortest prefix of ``row`` the field is skipped at or has no rows under,
        else what is at ``row`` (``MISSING`` when nothing is): a cell held inside a
        list or a table at the row above is read off it by position."""
        if ref in self._fields_with_a_skip:
            skip_row = self._row_a_skip_reaches_on(ref, row)
            if skip_row is not None:
                return SKIPPED, skip_row
        if ref in self._fields_held_at_their_own_row:
            return self._values.get(ref, NO_VALUES).get(row, MISSING), row
        if not row:  # cells sit in a value one level above their rows: ``()`` has no position, and a shorter row finds no value there
            return MISSING, row
        field = self._fields[ref]
        held_as = field._held_as
        if held_as.__class__ is ColumnOfTable:
            return self._cell_of_column(field, held_as, row), row  # type: ignore[arg-type]
        cells = self._values.get(ref, NO_VALUES).get(row[:-1], MISSING)
        if cells is MISSING or row[-1] >= len(cells):
            return MISSING, row
        return cells[row[-1]], row

    def cells_under(self, ref: Ref, row: Row) -> list[Any]:
        """The list a row-making output holds at ``row``, whose items are the rows under it."""
        return self._values[ref][row]

    def _cell_of_column(self, field: CompiledField, held_as: ColumnOfTable, row: Row) -> Any:
        """The cell of a table's column at ``row``, as the column's type, a blank cell ``None``."""
        table = self._values.get(held_as.table, NO_VALUES).get(row[:-1], MISSING)
        if table is MISSING:
            return MISSING
        column = table[held_as.column]
        if row[-1] >= len(column):
            return MISSING
        cell = column[row[-1]]
        return None if cell is None else field.type(cell)

    def _row_a_skip_reaches_on(self, ref: Ref, row: Row) -> Row | None:
        index = self._fields[ref]._index
        return _row_a_skip_reaches(self._values.get(ref, NO_VALUES), self.rows[index.id].no_rows_under if index is not None else (), row)

    def row_id(self, index: Index, row: Row) -> str:
        """The id of the row at ``row`` on ``index``: on a table's rows, the id
        its stored table carries, which may have come in with the table; on
        any other index, the id spelled from the index and the row."""
        maker = self._lookups.field_making_rows_of_index[index.id]
        if maker._held_as.__class__ is not TableAtRow:
            return global_row_id(index, row)
        table, _ = self.value_or_skip_at(maker.ref, row[:-1])
        return table[ROW_ID_COLUMN][row[-1]]

    # -- what the run produced -----------------------------------------------------

    def result_of(self, node_id: str) -> dict[str, Any] | None:
        """This node's outputs by name, or ``None`` when the node did not run (every output is skipped at the top).

        A result holds only what the node produced: an output the node skipped
        (the branch not taken) is absent, never ``SKIPPED``. A scalar output of a
        node that ran once is its value; anything on an index is ``ValuesOnRows``,
        the rows written so far and a value each, without the rows that were skipped."""
        outputs = self._lookups.interfaces[node_id].outputs
        skipped_at_top = {out.name: self._row_a_skip_reaches_on(Ref(node_id, out.name), ()) is not None for out in outputs}
        if outputs and all(skipped_at_top.values()):
            return None
        result: dict[str, Any] = {}
        for out in outputs:
            if skipped_at_top[out.name]:
                continue
            field = self._fields[Ref(node_id, out.name)]
            by_row = self._values.get(field.ref, NO_VALUES)
            if field._index is not None:
                rows, cells = self._rows_and_cells_with_a_value(field, by_row)
                result[out.name] = ValuesOnRows(rows=rows, values=cells)
            elif (at_top := by_row.get((), MISSING)) is not MISSING:
                result[out.name] = at_top
        return result

    def _rows_and_cells_with_a_value(self, field: CompiledField, by_row: Mapping[Row, Any]) -> tuple[list[Row], list[Any]]:
        """A field's rows and cells in order, leaving out rows a skip reaches."""
        rows = sorted(by_row)
        if field.ref in self._fields_with_a_skip:
            no_rows_under = self.rows[field._index.id].no_rows_under  # type: ignore[union-attr]  # what is on rows has an index
            rows = [row for row in rows if _row_a_skip_reaches(by_row, no_rows_under, row) is None]
        if field._held_as.__class__ is not CellsInAList:
            return rows, [by_row[row] for row in rows]
        cell_rows: list[Row] = []
        cells: list[Any] = []
        for row in rows:
            for position, cell in enumerate(by_row[row]):
                if cell is not SKIPPED:
                    cell_rows.append((*row, position))
                    cells.append(cell)
        return cell_rows, cells

    # -- to and from a state --------------------------------------------------------

    def output_columns_of(self, node_id: str) -> dict[str, OutputColumn]:
        """What this node's outputs hold, as a state writes it. An output nothing was written to is left out."""
        columns: dict[str, OutputColumn] = {}
        for out in self._lookups.interfaces[node_id].outputs:
            ref = Ref._from_checked_parts(node_id, out.name)
            if ref in self._values:
                columns[out.name] = output_column_of(self._values[ref], self._fields[ref])
        return columns

    def read_state(self, compiled: CompiledGraph, state: RunState, dropped: set[str]) -> list[Problem]:
        """Take in what ``state`` holds on every output, each value read through its output's
        check, and mark the skips as they are read. ``dropped`` (``_nodes_changed_since``) is
        left out. Reports every problem found: the one reader of a state's output columns."""
        problems: list[Problem] = []
        for node_id, work in state.nodes.items():
            if node_id in dropped:
                continue
            for name, column in work.outputs.items():
                ref = Ref(node_id, name)
                read = read_output_column(compiled, ref, column, problems)
                if read is None:
                    continue
                self._values[ref] = read
                for row in column.skipped:
                    if read.get(row) is SKIPPED:
                        self._note_skip_at(self._fields[ref], row)
        return problems

    def make_rows_again_from_stored_output(self, field: CompiledField, row: Row) -> bool:
        """Make again the rows the unit at ``row`` made on this output, from what ``read_state``
        took in. ``False`` when nothing is stored for that unit at its row or as a skip above it."""
        by_row = self._values.get(field.ref, NO_VALUES)
        value = by_row.get(row, MISSING)
        if value is MISSING:
            return _row_a_skip_reaches(by_row, (), row) is not None
        if value is not SKIPPED and (index := field._held_as.rows_made) is not None:
            self._make_rows_under(index.id, row, len(value), [])
        return True
