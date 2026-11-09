"""What one unit produced, checked against its outputs before it enters a run's state.

A value enters the state three ways: a node returns it, a person answers with
it, or a stored state is read back. Each passes the one check its output has
(``CompiledField._validate_cell``, built once per output from its type). Inside a
leg nothing checks a value again on the way into the state.

``validate_unit_outputs`` is the shared part for a return and an answer, which
are the same thing: one unit's outputs by name. It reads each into the cells
the state holds, a many-valued output into a list of cells and a table with its
rows' ids written on, and reports every output that does not fit, not only the first.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from conductor._internal.errors import Problem
from conductor._internal.series import Row, Series
from conductor._internal.skip_and_ask import SKIPPED
from conductor._internal.table import frame_with_no_rows, is_frame, no_row_kept_its_id, with_row_ids

if TYPE_CHECKING:
    from conductor._internal.graph.compiled_node import CompiledField, CompiledNode
    from conductor._internal.ref import Ref


def validate_unit_outputs(node: CompiledNode, row: Row, value_by_output: Mapping[str, Any], row_ids_must_survive: frozenset[Ref] = frozenset()) -> tuple[dict[str, Any], list[Problem]]:
    """``value_by_output`` as the cells the state holds, and every output that did not fit.

    ``row`` is the unit's own: a table's new rows are named after it. ``SKIPPED``
    passes on any output and on any element of a many-valued one. Only outputs
    that fit are in the cells; a caller with problems writes none of them.
    ``row_ids_must_survive`` holds the outputs that had an id to keep: a unit given
    no rows, or a person's answer whose form has no place for one, passes none.
    """
    assert node.interface is not None and node.wiring is not None  # the engine runs only a runnable graph
    cells: dict[str, Any] = {}
    problems: list[Problem] = []
    for out in node.interface.outputs:
        name = out.name
        if name not in value_by_output:
            problems.append(_output_problem(name, f"no value for the output '{name}'"))
            continue
        value = value_by_output[name]
        if value is SKIPPED:
            cells[name] = value
            continue
        field = node.wiring.fields[name]
        try:
            cells[name] = _cells_of_many_valued_output(field, name, value) if field._unit_writes_many_cells else _cell_of_output(field, row, value, row_ids_must_survive)
        except ValidationError as invalid:
            problems.append(_output_problem(
                name, f"'{name}' must be {_type_name_for_error(field)}, not a {type(value).__name__}",
                errors=tuple(invalid.errors(include_url=False, include_input=False, include_context=False)),
            ))
        except _NotASequence as invalid:
            problems.append(_output_problem(name, str(invalid)))
        except _RowIdsLost:
            problems.append(_output_problem(name, f"'{name}' lost the ids of its rows, so they can't be matched to the table it was built from", code="row_ids_lost"))
    for name in value_by_output:
        if name not in node.wiring.fields or not node.wiring.fields[name]._is_output:
            problems.append(_output_problem(name, f"'{name}' is not one of this node's outputs"))
    return cells, problems


def _output_problem(output: str, message: str, *, errors: tuple[dict[str, Any], ...] = (), code: str = "invalid_output") -> Problem:
    """One output that does not fit: which, a sentence that never quotes the value, and pydantic's own errors when its check said why."""
    return Problem(type=code, loc=(output,), msg=message, ctx={"errors": errors})


def _cells_of_many_valued_output(field: CompiledField, name: str, value: Any) -> list[Any]:
    if value is None and field._takes_blank:
        return []  # row-making outputs have no blank form: no rows
    if isinstance(value, Series):
        value = value.values
    elif not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise _NotASequence(f"'{name}' is many values and must be a sequence, not a {type(value).__name__}")
    read = field._validate_cell
    return [item if item is SKIPPED else read(item) for item in value]


def _cell_of_output(field: CompiledField, row: Row, value: Any, row_ids_must_survive: frozenset[Ref]) -> Any:
    cell = field._validate_cell(value)
    if field._makes_rows is None:
        return cell
    if cell is None:  # row-making outputs have no blank form: no rows
        return frame_with_no_rows(field._cell_type)
    if field.ref in row_ids_must_survive and no_row_kept_its_id(cell):
        raise _RowIdsLost
    return with_row_ids(cell, field._makes_rows, row)


def took_a_row_in(inputs: Mapping[str, Any]) -> bool:
    """Did these inputs carry a row a returned table could keep the id of?"""
    return any((is_frame(value) or isinstance(value, Series)) and len(value) > 0 for value in inputs.values())


def _type_name_for_error(field: CompiledField) -> str:
    return f"a {getattr(field._cell_type, '__name__', field._cell_type)}"


class _NotASequence(Exception):
    """A many-valued output was given one value."""


class _RowIdsLost(Exception):
    """A table a merge matches on its rows' ids came back with none of them."""
