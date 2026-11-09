"""A unit waiting on a person: what it asks, and how an answer finishes it.

A node returns ``AskPerson`` and ``waiting_unit_for`` turns that into an entry
of ``RunState.waiting``. ``apply_answers`` is the other half, and the verb
``CompiledGraph.answer``: it checks every answer against what its unit asked,
raises with every problem it found, and otherwise writes the answered units
into a new state as finished, with the same cells a node's return would have
written. No node runs: the state is read back into a ledger, each answer
recorded, and the state written again. Whether the rest of the graph can then
run is the next leg's business.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from conductor._internal.codec import to_wire
from conductor._internal.errors import InvalidCallError, Problem, make_problem
from conductor._internal.execution.run_ledger import RunLedger, SkippedAtRow
from conductor._internal.execution.run_state import (
    Answers,
    Question,
    RunState,
    WaitingUnit,
)
from conductor._internal.execution.validate_unit_outputs import validate_unit_outputs
from conductor._internal.series import Row
from conductor._internal.skip_and_ask import SKIPPED, AskPerson
from conductor.widgets import Widget

if TYPE_CHECKING:
    from conductor._internal.graph.compiled_graph import CompiledGraph
    from conductor._internal.graph.compiled_node import CompiledNode


def waiting_unit_for(node: CompiledNode, row: Row, ask_person: AskPerson, inputs: dict[str, Any]) -> WaitingUnit:
    """The entry for this unit in ``RunState.waiting``, with the ``inputs`` it was
    called with. A ``ValueError`` when ``ask_person`` names an output the node does
    not have or proposes a default that is not its output's type: the node's
    own doing, not the person's."""
    assert node.interface is not None and node.wiring is not None  # the engine runs only a runnable graph
    questions = node._questions
    if ask_person.defaults or ask_person.widgets:
        questions = _with_defaults_and_widgets(node, row, ask_person)
    return WaitingUnit(
        id=node.id if row == () else f"{node.id}@{','.join(str(n) for n in row)}",
        node_id=node.id,
        row=row,
        prompt=ask_person.prompt,
        questions=questions,
        inputs=inputs,
    )


def _with_defaults_and_widgets(node: CompiledNode, row: Row, ask_person: AskPerson) -> tuple[Question, ...]:
    """The node's questions with what this ``AskPerson`` adds, each a copy. A ``ValueError`` when it isn't sound."""
    names = {question.name for question in node._questions}
    for name in (*ask_person.defaults, *ask_person.widgets):
        if name not in names:
            raise ValueError(f"AskPerson names '{name}', which is not one of this node's outputs")
    for name, widget in ask_person.widgets.items():
        if not isinstance(widget, Widget):
            raise ValueError(f"AskPerson's widget for '{name}' is a {type(widget).__name__}, not a widget")
    cells, problems = validate_unit_outputs(node, row, {name: ask_person.defaults.get(name, SKIPPED) for name in names})
    if problems:
        raise ValueError("; ".join(problem.msg for problem in problems))
    defaults = {name: _default_that_has_json_form(cells[name], name) for name in ask_person.defaults}
    return tuple(
        question.model_copy(update={
            "default": defaults.get(question.name),
            "optional": question.choice is None and (question.name in defaults or question.optional),
            "widget": None if question.name not in ask_person.widgets else ask_person.widgets[question.name].model_dump(mode="json"),
        })
        for question in node._questions
    )


def _default_that_has_json_form(cell: Any, name: str) -> Any:
    """``cell``, held typed as a state's values are, refused here if a dump of the state would fail on it."""
    try:
        for item in cell if isinstance(cell, list) else (cell,):
            to_wire(item, type(item))
    except Exception as unwritable:
        raise ValueError(f"AskPerson's default for '{name}' has no JSON form: {unwritable}") from unwritable
    return cell


def apply_answers(compiled: CompiledGraph, state: RunState, answers: Answers) -> RunState:
    """``state`` with these units answered; see the module docstring."""
    changed = set(compiled.nodes_changed_since(state))
    waiting = {waiting_unit.id: waiting_unit for waiting_unit in state.waiting}
    problems: list[Problem] = []
    finished: list[tuple[WaitingUnit, dict[str, Any] | None]] = []
    for waiting_id, answer in answers.items():
        waiting_unit = waiting.get(waiting_id)
        if waiting_unit is None or waiting_unit.node_id in changed:
            problems.append(make_problem("not_waiting", "answers", waiting_id, waiting_id=waiting_id))
            continue
        cells, answer_problems = _answered_cells(compiled.node(waiting_unit.node_id), waiting_unit, answer)
        problems.extend(answer_problems)
        if not answer_problems:
            finished.append((waiting_unit, cells))
    if problems:
        raise InvalidCallError(problems)
    ledger = RunLedger.restore(compiled, state)
    for waiting_unit, cells in finished:
        ledger.record((waiting_unit.node_id, waiting_unit.row), SkippedAtRow(waiting_unit.row) if cells is None else cells)
    return ledger.state(state.status, state.failure)


def _answered_cells(node: CompiledNode, waiting_unit: WaitingUnit, answer: Mapping[str, Any] | None) -> tuple[dict[str, Any] | None, list[Problem]]:
    """One unit's answer as the cells it finishes with, ``None`` for a unit with
    no value on its row, and what is wrong with it."""
    if answer is None:
        return None, []
    problem_location = ("answers", waiting_unit.id)
    if not isinstance(answer, Mapping):
        return None, [make_problem("invalid_value", *problem_location, reason=f"an answer is a value per output, not a {type(answer).__name__}")]
    problems: list[Problem] = []
    questions = {question.name: question for question in waiting_unit.questions}
    for name in answer:
        if name not in questions:
            problems.append(make_problem("not_asked", *problem_location, name, name=name))
    groups: dict[str, list[str]] = {}
    for question in waiting_unit.questions:
        if question.choice is not None:
            groups.setdefault(question.choice, []).append(question.name)
    values = _with_choices_and_defaults(questions, groups, answer, problem_location, problems)
    cells, output_problems = validate_unit_outputs(node, waiting_unit.row, {name: value for name, value in values.items() if name in questions})
    for output_problem in output_problems:
        problems.append(make_problem("invalid_value", *problem_location, output_problem.loc[0], reason=output_problem.msg, errors=list(output_problem.ctx["errors"])))
    return cells, problems


def _with_choices_and_defaults(
    questions: Mapping[str, Question], groups: Mapping[str, list[str]], answer: Mapping[str, Any],
    problem_location: tuple[str, ...], problems: list[Problem],
) -> dict[str, Any]:
    """What was given, a choice not taken as ``SKIPPED``, a blank question its default."""
    values = dict(answer)
    for choice, names in groups.items():
        taken = [name for name in names if name in answer]
        if len(taken) != 1:
            problems.append(make_problem("choice_unanswered", *problem_location, choice, alternatives=names, answered=len(taken)))
        for name in names:
            if name not in answer:
                values[name] = SKIPPED
    for name, question in questions.items():
        if name in values:
            continue
        if question.optional:
            values[name] = question.default
        else:
            values[name] = SKIPPED  # the problem below stops the call; a value in its place keeps the other questions checked
            problems.append(make_problem("missing", *problem_location, name, name=name))
    return values
