"""The door into a run's state: what one unit produced, checked against its outputs.

A value enters the state three ways: a node returns it, a person answers with
it, or a stored state is read back. Each passes the one check its output has
(``CompiledField._read_cell``, built once per output from its type). Inside a
leg nothing checks a value again on the way into the state.

``validate_unit_outputs`` is the shared part for a return and an answer, which
are the same thing: one unit's outputs by name. It reads each into the cells
the state holds, a many-valued output into a list of cells and a table with its
rows' ids written on, and reports every output that does not fit instead of
stopping at the first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from conductor._sentinel import SKIPPED
from conductor.series import Row, Series
from conductor.table import Table, frame_with_no_rows, no_row_kept_its_id, with_row_ids

if TYPE_CHECKING:
    from conductor.graph.compiled_node import CompiledField, CompiledNode
    from conductor.ref import Ref


@dataclass(frozen=True)
class OutputFault:
    """One output that does not fit: which, a sentence that never quotes the
    value, and pydantic's own errors when its check said why."""

    output: str
    message: str
    errors: tuple[dict[str, Any], ...] = ()
    code: str = "invalid_output"


def validate_unit_outputs(node: CompiledNode, row: Row, named: Mapping[str, Any], row_ids_must_survive: frozenset[Ref]) -> tuple[dict[str, Any], list[OutputFault]]:
    """``named`` as the cells the state holds, and every output that did not fit.

    ``row`` is the unit's own: a table's new rows are named after it. ``SKIPPED``
    passes on any output and on any element of a many-valued one. Only outputs
    that fit are in the cells; a caller with faults writes none of them.
    """
    assert node.interface is not None and node.wiring is not None  # the engine runs only a runnable graph
    cells: dict[str, Any] = {}
    faults: list[OutputFault] = []
    for out in node.interface.outputs:
        name = out.name
        if name not in named:
            faults.append(OutputFault(name, f"no value for the output '{name}'"))
            continue
        value = named[name]
        if value is SKIPPED:
            cells[name] = value
            continue
        field = node.wiring.fields[name]
        try:
            cells[name] = _read_many(field, name, value) if field._holds_many else _read_one(field, row, value, row_ids_must_survive)
        except ValidationError as invalid:
            faults.append(OutputFault(
                name,
                f"'{name}' must be {_named(field)}, not a {type(value).__name__}",
                tuple(invalid.errors(include_url=False, include_input=False, include_context=False)),
            ))
        except _NotASequence as invalid:
            faults.append(OutputFault(name, str(invalid)))
        except _RowIdsLost:
            faults.append(OutputFault(name, f"'{name}' lost the ids of its rows, so they can't be matched to the table it was built from", code="row_ids_lost"))
    for name in named:
        if name not in node.wiring.fields or not node.wiring.fields[name]._is_output:
            faults.append(OutputFault(name, f"'{name}' is not one of this node's outputs"))
    return cells, faults


def _read_many(field: CompiledField, name: str, value: Any) -> list[Any]:
    if value is None and field._takes_blank:
        return []  # row-making outputs have no blank form: no rows
    if isinstance(value, Series):
        value = value.values
    elif not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _NotASequence(f"'{name}' is many values and must be a sequence, not a {type(value).__name__}")
    read = field._read_cell
    return [item if item is SKIPPED else read(item) for item in value]


def _read_one(field: CompiledField, row: Row, value: Any, row_ids_must_survive: frozenset[Ref]) -> Any:
    cell = field._read_cell(value)
    if cell is None and field._makes_rows is not None:  # row-making outputs have no blank form: no rows
        return Table._holding(frame_with_no_rows(field._cell_type))
    if field._makes_rows is not None and isinstance(cell, Table):
        if field.ref in row_ids_must_survive and no_row_kept_its_id(cell.frame):
            raise _RowIdsLost
        return type(cell)._holding(with_row_ids(cell.frame, field._makes_rows, row))
    return cell


def _named(field: CompiledField) -> str:
    return f"a {getattr(field._cell_type, '__name__', field._cell_type)}"


class _NotASequence(Exception):
    """A many-valued output was given one value."""


class _RowIdsLost(Exception):
    """A table a merge matches on its rows' ids came back with none of them."""
