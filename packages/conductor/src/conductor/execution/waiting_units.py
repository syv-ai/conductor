"""A unit waiting on a person: what it asks, and how an answer finishes it.

A node returns ``AskPerson`` and ``waiting_unit_for`` turns that into an entry
of ``RunState.waiting``. ``apply_answers`` is the other half, and the verb
``CompiledGraph.answer``: it checks every answer against what its unit asked,
refuses the call listing every fault, and otherwise writes the answered units
into a new state as finished, with the same cells a node's return would have
written. No node runs and nothing else in the state is read, so the cost
follows the answers. Whether the rest of the graph can then run is the next
leg's business.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from conductor.codec import json_schema_of, to_wire
from conductor.dtype import description_of
from conductor.errors import InvalidCallError, Problem, problem
from conductor.execution.run_state import Answers, OutputColumn, Question, RunState, WaitingUnit
from conductor.execution.validate_unit_outputs import validate_unit_outputs
from conductor.field_records import Output
from conductor.ref import Ref
from conductor.series import Row
from conductor.skip_and_ask import SKIPPED, AskPerson
from conductor.table import Table

if TYPE_CHECKING:
    from conductor.graph.compiled import CompiledGraph
    from conductor.graph.compiled_node import CompiledNode


def waiting_unit_for(node: CompiledNode, row: Row, asked: AskPerson) -> WaitingUnit:
    """The entry for this unit in ``RunState.waiting``. A ``ValueError`` when
    ``asked`` names an output the node does not have or proposes a default that
    is not its output's type: the node's own fault, not the person's."""
    assert node.interface is not None and node.wiring is not None  # the engine runs only a runnable graph
    outputs = {out.name: out for out in node.interface.outputs}
    for name in (*asked.defaults, *asked.widgets):
        if name not in outputs:
            raise ValueError(f"AskPerson names '{name}', which is not one of this node's outputs")
    defaults: dict[str, Any] = {}
    if asked.defaults:
        checked = {name: asked.defaults.get(name, SKIPPED) for name in outputs}
        cells, faults = validate_unit_outputs(node, row, checked, frozenset())
        if faults:
            raise ValueError("; ".join(fault.message for fault in faults))
        defaults = {name: _default_as_wire(cells[name], outputs[name]) for name in asked.defaults}
    return WaitingUnit(
        id=node.id if row == () else f"{node.id}@{','.join(str(n) for n in row)}",
        node_id=node.id,
        row=row,
        prompt=asked.prompt,
        questions=tuple(_question(out, defaults, asked) for out in outputs.values()),
    )


def _default_as_wire(cell: Any, out: Output) -> Any:
    if isinstance(cell, list):
        return [to_wire(item, type(item)) for item in cell]
    return to_wire(cell, type(cell))


def _question(out: Output, defaults: Mapping[str, Any], asked: AskPerson) -> Question:
    widget = asked.widgets.get(out.name)
    return Question(
        name=out.name,
        title=out.title,
        description=out.description,
        dtype=description_of(out.dtype),
        json_schema=json_schema_of(out.dtype),
        widget=None if widget is None else widget.model_dump(mode="json"),
        default=defaults.get(out.name),
        optional=out.name in defaults or out.takes_blank,
        choice=out.choice,
    )


def apply_answers(compiled: CompiledGraph, state: RunState, answers: Answers) -> RunState:
    """``state`` with these units answered; see the module docstring."""
    changed = set(compiled.nodes_changed_since(state))
    waiting = {asked.id: asked for asked in state.waiting}
    faults: list[Problem] = []
    finished: list[tuple[WaitingUnit, dict[str, Any] | None]] = []
    for waiting_id, given in answers.items():
        asked = waiting.get(waiting_id)
        if asked is None or asked.node_id in changed:
            faults.append(problem("not_waiting", "answers", waiting_id, waiting_id=waiting_id))
            continue
        cells = _answered_cells(compiled.node(asked.node_id), asked, given, faults, compiled._row_ids_must_survive)
        if cells is not ...:
            finished.append((asked, cells))
    if faults:
        raise InvalidCallError(faults)
    return _with_units_finished(compiled, state, finished)


def _answered_cells(node: CompiledNode, asked: WaitingUnit, given: Mapping[str, Any] | None, faults: list[Problem], row_ids_must_survive: frozenset[Ref]) -> Any:
    """One unit's answer as the cells it finishes with, ``None`` for a unit
    with no value on its row. Adds its faults and returns ``...`` when it has any."""
    if given is None:
        return None
    where = ("answers", asked.id)
    if not isinstance(given, Mapping):
        faults.append(problem("invalid_value", *where, reason=f"an answer is a value per output, not a {type(given).__name__}"))
        return ...
    before = len(faults)
    questions = {question.name: question for question in asked.questions}
    for name in given:
        if name not in questions:
            faults.append(problem("not_asked", *where, name, name=name))
    groups: dict[str, list[str]] = {}
    for question in asked.questions:
        if question.choice is not None:
            groups.setdefault(question.choice, []).append(question.name)
    values = _with_choices_and_defaults(questions, groups, given, where, faults)
    cells, output_faults = validate_unit_outputs(node, asked.row, {name: value for name, value in values.items() if name in questions}, row_ids_must_survive)
    for fault in output_faults:
        faults.append(problem("invalid_value", *where, fault.output, reason=fault.message, errors=list(fault.errors)))
    return ... if len(faults) > before else cells


def _with_choices_and_defaults(
    questions: Mapping[str, Question], groups: Mapping[str, list[str]], given: Mapping[str, Any],
    where: tuple[str, ...], faults: list[Problem],
) -> dict[str, Any]:
    """What was given, a choice not taken as ``SKIPPED``, a blank question its default."""
    values = dict(given)
    for choice, names in groups.items():
        taken = [name for name in names if name in given]
        if len(taken) != 1:
            faults.append(problem("choice_unanswered", *where, choice, alternatives=names, answered=len(taken)))
        for name in names:
            if name not in given:
                values[name] = SKIPPED
    for name, question in questions.items():
        if name in values:
            continue
        if question.optional:
            values[name] = question.default
        else:
            values[name] = SKIPPED  # refused below; a value in its place so the other questions are still checked
            faults.append(problem("missing", *where, name, name=name))
    return values


def _with_units_finished(compiled: CompiledGraph, state: RunState, finished: list[tuple[WaitingUnit, dict[str, Any] | None]]) -> RunState:
    """A new state holding these units' cells, each output copied once, however many units wrote to it."""
    columns = dict(state.output_columns)
    finished_rows = dict(state.finished_rows)
    copied: set[str] = set()
    answered: set[str] = set()

    def column_of(ref: Ref) -> OutputColumn:
        if ref not in copied:
            old = columns.get(ref)
            columns[ref] = OutputColumn.model_construct(
                rows=[*old.rows] if old else [], values=[*old.values] if old else [], skipped=[*old.skipped] if old else [],
            )
            copied.add(ref)
        return columns[ref]

    def finished_rows_of(node_id: str) -> list[Row]:
        if node_id not in copied:
            finished_rows[node_id] = [*finished_rows.get(node_id, ())]
            copied.add(node_id)
        return finished_rows[node_id]

    for asked, cells in finished:
        answered.add(asked.id)
        finished_rows_of(asked.node_id).append(asked.row)
        node = compiled.node(asked.node_id)
        assert node.interface is not None and node.wiring is not None  # the engine runs only a runnable graph
        for out in node.interface.outputs:
            ref = node.wiring.fields[out.name].ref
            value = SKIPPED if cells is None else cells[out.name]
            column = column_of(ref)
            if value is SKIPPED:
                column.skipped.append(asked.row)
            elif isinstance(value, list) and not isinstance(value, Table) and node.wiring.fields[out.name]._holds_many:
                for j, item in enumerate(value):
                    (column.skipped if item is SKIPPED else column.rows).append((*asked.row, j))
                    if item is not SKIPPED:
                        column.values.append(item)
            else:
                column.rows.append(asked.row)
                column.values.append(value)
    return state.model_copy(update={
        "waiting": tuple(unit for unit in state.waiting if unit.id not in answered),
        "output_columns": columns,
        "finished_rows": finished_rows,
    })
