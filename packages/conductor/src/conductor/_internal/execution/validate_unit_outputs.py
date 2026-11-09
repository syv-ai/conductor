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

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from conductor._internal.errors import Problem
from conductor._internal.execution.values_of_run import held_form_of_return, output_problem
from conductor._internal.index import Row
from conductor._internal.skip_and_ask import SKIPPED

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
    assert node.interface is not None and node._wiring is not None  # the engine runs only a runnable graph
    cells: dict[str, Any] = {}
    problems: list[Problem] = []
    for out in node.interface.outputs:
        name = out.name
        if name not in value_by_output:
            problems.append(output_problem(name, f"no value for the output '{name}'"))
            continue
        value = value_by_output[name]
        if value is SKIPPED:
            cells[name] = value
            continue
        field = node._wiring.fields[name]
        try:
            cell = held_form_of_return(field, name, row, value, row_ids_must_survive)
        except ValidationError as invalid:
            problems.append(output_problem(
                name, f"'{name}' must be {_type_name_for_error(field)}, not a {type(value).__name__}",
                errors=tuple(invalid.errors(include_url=False, include_input=False, include_context=False)),
            ))
        else:
            if cell.__class__ is Problem:
                problems.append(cell)
            else:
                cells[name] = cell
    for name in value_by_output:
        if name not in node._wiring.fields or not node._wiring.fields[name]._is_output:
            problems.append(output_problem(name, f"'{name}' is not one of this node's outputs"))
    return cells, problems


def _type_name_for_error(field: CompiledField) -> str:
    return f"a {getattr(field.type, '__name__', field.type)}"
