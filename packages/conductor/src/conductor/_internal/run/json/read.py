"""Reads a run from the JSON that ``run.to_json()`` wrote.

``run_from_json`` is the door. The graph's codec validates the whole text: every key and every value. Then
``outcomes_from_saved`` and ``reusable_from_saved`` turn each node's saved lists into what a run keeps in memory, and
``run_from_json`` checks what only the lists together can show: that the lists of each node are as long as its calls,
that a table written once is found where its entry says, and that an answer is for a call that waits.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from pydantic import ValidationError

from conductor._internal.ask import Answer
from conductor._internal.base import SKIPPED, ItemIndexes, parse_node_path, place_text
from conductor._internal.compiled import (
    CompiledGraph,
    outputs_when_not_called,
    outputs_when_skipped,
)
from conductor._internal.field import Field
from conductor._internal.gate import output_validator_of
from conductor._internal.problem import InvalidCallError, call_problems, value_free_errors

from ..flat import FlatGraph, flat_graph_of
from ..outcome import Answered, Done, NoneInput, Outcome, Skipped, Unanswered
from ..progress import all_item_indexes
from ..reuse import ReusableOutcomes
from .codec import codec_of, where_in
from .type import (
    SAVED_DONE,
    SAVED_SKIPPED,
    SameAs,
    SavedOutcomes,
    SavedReusable,
)


def run_from_json[R](run_type: Callable[..., R], saved: str | bytes, graph: CompiledGraph) -> R:
    """Returns the run of ``graph`` that ``run_to_json`` wrote as ``saved``.

    For the README graph, ``run_from_json(Run, run.to_json(), graph)`` returns a run with the inputs, the
    outcomes and the state of ``run``: its ``results["shout"]["result"]`` is ``["A!", "B!", "C!"]`` again.

    The graph's codec validates the JSON text in one pass, and reads each value the way its type reads JSON: a
    date from its text, for example. Each node's saved lists are then turned into a list of outcomes in
    call-number order, because item indexes are not saved. They are worked out here, node by node in topological
    order: a node's item indexes follow from the lengths of the series upstream of it, and those are loaded by then.

    Args:
        run_type: The class ``Run``. It is passed in because ``run/type.py`` imports this package.
        saved: The JSON text.
        graph: The graph the run was made on.

    Raises:
        InvalidCallError: ``saved`` was written for another graph (``saved_run_of_another_graph``), or is not a
            saved run of this graph (``saved_run_unreadable``, whose ``where`` names the key, the node, the output
            and the call number, and which quotes no value).
    """
    codec = codec_of(graph)
    try:
        read = codec.model_validate_json(saved)
    except ValidationError as error:
        first = error.errors()[0]
        if first["type"] == "literal_error" and first["loc"] == ("graph",):  # the fingerprint under "graph" is not this graph's
            raise InvalidCallError([call_problems.saved_run_of_another_graph("saved_run", saved=first["input"], graph=graph.fingerprint)]) from None
        reason = value_free_errors(error)[0]["msg"]
        raise InvalidCallError([call_problems.saved_run_unreadable("saved_run", where=where_in(codec, first["loc"]), reason=reason)]) from None
    flat = flat_graph_of(graph)
    in_call_order: dict[str, list[Outcome | None]] = {}
    reused: dict[str, ReusableOutcomes] = {}
    try:
        for name, saved_outcomes in read.outcomes.items():
            where = f"outcomes.{name}"
            in_call_order[name] = outcomes_from_saved(flat.function_nodes[name].compiled_node.outputs, saved_outcomes)
        for name, saved_reusable in read.reusable.items():
            where = f"reusable.{name}.outcomes"
            reusable_in_call_order = outcomes_from_saved(flat.function_nodes[name].compiled_node.outputs, saved_reusable.outcomes)
            where = f"reusable.{name}"
            reused[name] = reusable_from_saved(saved_reusable, reusable_in_call_order)
    except ValueError as wrong:
        raise InvalidCallError([call_problems.saved_run_unreadable("saved_run", where=where, reason=str(wrong))]) from None
    _share_tables_written_once(flat, in_call_order, read.tables_written_once)
    outcomes: dict[str, dict[ItemIndexes, Outcome]] = {}
    for node in flat.function_nodes.values():
        outcomes[node.name] = {}
        if (of_node := in_call_order.get(node.name)) is not None:
            every_item_indexes, _ = all_item_indexes(outcomes, node.loops, ())
            if len(of_node) != len(every_item_indexes):
                reason = f"its lists have {len(of_node)} entries, but the saved series give the node {len(every_item_indexes)} calls"
                raise InvalidCallError([call_problems.saved_run_unreadable("saved_run", where=f"outcomes.{node.name}", reason=reason)])
            outcomes[node.name] = {
                item_indexes: outcome for item_indexes, outcome in zip(every_item_indexes, of_node, strict=True) if outcome is not None
            }
    answers = _answers_from_saved(flat, read.answers, outcomes)
    code_changed = {
        parse_node_path(name): fingerprint for name, fingerprint in read.code.items() if fingerprint != flat.function_nodes[name].compiled_node.code_fingerprint
    }
    # outcomes that other code produced are never reused
    reusable = {path: of_node for name, of_node in reused.items() if (path := parse_node_path(name)) not in code_changed}
    return run_type(
        graph, read.inputs, read.max_items, outcomes, reusable, _flat=flat, state=read.state, _code_changed=code_changed,
        _finished_nodes=frozenset(parse_node_path(name) for name in read.complete), _answers=answers,
    )


def _answers_from_saved(
    flat: FlatGraph, saved: Mapping[str, Mapping[int, Any]], outcomes: dict[str, dict[ItemIndexes, Outcome]],
) -> dict[str, dict[ItemIndexes, Answer]]:
    """Turns each waiting call that a saved run has an answer for into that answer, kept until the node is called.

    ``{"approve": {1: True}}`` with ``approve``'s second call waiting takes that ``Unanswered`` out of ``outcomes`` and
    returns ``{"approve": {(1,): Answer(its question, "ok", True)}}``.

    Args:
        flat: The flat graph of the run's graph.
        saved: The saved run's ``answers``, each read by its node's ``Asked`` parameter: the codec takes them only
            for such a node.
        outcomes: The outcomes read so far, by node name. Changed in place.

    Raises:
        InvalidCallError: An answer names a call that is not waiting (``saved_run_unreadable``).
    """
    answers: dict[str, dict[ItemIndexes, Answer]] = {}
    for name, by_call_number in saved.items():
        node = flat.function_nodes[name]
        (parameter,) = node.compiled_node.answer_fields
        every_item_indexes, _ = all_item_indexes(outcomes, node.loops, ())
        for call_number, value in by_call_number.items():
            item_indexes = every_item_indexes[call_number] if call_number < len(every_item_indexes) else None
            waiting = outcomes[name].get(item_indexes) if item_indexes is not None else None
            if type(waiting) is not Unanswered or item_indexes is None:
                reason = "it names a call that is not waiting for a person"
                raise InvalidCallError([call_problems.saved_run_unreadable("saved_run", where=f"answers.{name}.{call_number}", reason=reason)])
            del outcomes[name][item_indexes]
            answers.setdefault(name, {})[item_indexes] = Answer(waiting.question, parameter, value)
    return answers


def outcomes_from_saved(outputs: Mapping[str, Field], saved: SavedOutcomes) -> list[Outcome | None]:
    """Returns one node's outcomes from their saved lists, in call-number order; ``None`` for a call with no outcome.

    For ``shout`` not called for its second line::

        {"states": ["done", null, "done"], "outputs": {"result": ["A!", null, "C!"]}, "not_called_because": {"1": "line"}}

    it returns ``[Done({"result": "A!"}), NoneInput({"result": None}, "line"), Done({"result": "C!"})]``.

    Args:
        outputs: The outputs of the node, by name.
        saved: The saved lists, each value already validated by its output's type.

    Raises:
        ValueError: The list of an output is not as long as ``states``; or ``skipped``, ``skipped_items``,
            ``questions`` or ``not_called_because`` names a call that does not exist or that cannot have it.
    """
    states, values_by_output = saved.states, saved.outputs
    if any(len(values) != len(states) for values in values_by_output.values()):
        raise ValueError("the list of an output is not as long as 'states'")
    try:
        for name, call_numbers in saved.skipped.items():
            for call_number in call_numbers:
                values_by_output[name][call_number] = SKIPPED
        for name, by_call_number in saved.skipped_items.items():
            for call_number, indexes in by_call_number.items():
                for index in indexes:
                    values_by_output[name][call_number][index] = SKIPPED
        outputs_by_call = [{name: values[call_number] for name, values in values_by_output.items()} for call_number in range(len(states))]
        done, skipped = SAVED_DONE, SAVED_SKIPPED
        skipped_outcome = Skipped(outputs_when_skipped(outputs))
        outcomes: list[Outcome | None] = [
            Done(of_call) if state == done else skipped_outcome if state == skipped else None
            for state, of_call in zip(states, outputs_by_call, strict=True)
        ]
        for call_number, question in saved.questions.items():
            outcome = outcomes[call_number]
            if states[call_number] is None:
                outcomes[call_number] = Unanswered(question)
            elif type(outcome) is Done:
                outcomes[call_number] = Answered(outcome.outputs, question)
            else:
                raise LookupError(call_number)
        for call_number, input_name in saved.not_called_because.items():
            if outcomes[call_number] is not None:
                raise LookupError(call_number)
            outcomes[call_number] = NoneInput(outputs_when_not_called(outputs), input_name)
    except (LookupError, TypeError, AttributeError):  # a call number past the end of the lists, or a call of another kind
        message = "'skipped', 'skipped_items', 'questions' or 'not_called_because' names a call that does not exist or cannot have it"
        raise ValueError(message) from None
    return outcomes


def _share_tables_written_once(flat: FlatGraph, outcomes: Mapping[str, list[Outcome | None]], entries: list[SameAs]) -> None:
    """Fills in the outputs that a saved run wrote as ``null`` and listed under ``tables_written_once``.

    The saved run lists ``{"output": ["split", "result", 0], "same_as": ["people", "result", 0]}``. The table
    read for ``people.result`` is validated by the type of ``split.result`` and put there, so both outputs have
    the same table object again.

    Args:
        flat: The flat graph of the run's graph.
        outcomes: Each node's outcomes in call-number order, by name. Changed in place.
        entries: The saved run's ``tables_written_once``.

    Raises:
        InvalidCallError: An entry names an output or a call number the saved run does not have, an output with
            no table written in full, an output that is not ``null``, or a table the output's own type refuses
            (``saved_run_unreadable``).
    """
    for entry in entries:
        (node, output, call_number), (from_node, from_output, from_call_number) = entry.output, entry.same_as
        try:
            outcome, source = outcomes[node][call_number], outcomes[from_node][from_call_number]
            if not isinstance(outcome, (Done, Answered)) or not isinstance(source, (Done, Answered)) or outcome.outputs[output] is not None:
                raise LookupError(call_number)
            output_type = flat.function_nodes[node].compiled_node.outputs[output].whole
            table = flat.function_nodes[from_node].compiled_node.outputs[from_output].whole.written_once(source.outputs[from_output])
            if table is None:
                raise LookupError(from_call_number)
            outcome.outputs[output] = output_validator_of(output_type).validate_python(table, strict=True)
        except (LookupError, AttributeError, ValidationError) as wrong:
            reason = output_type.refusal(wrong).reason if isinstance(wrong, ValidationError) else "that is not an output written in full before it"
            where = f"{place_text(parse_node_path(node), output)}[{call_number}]"
            same_as = f"{place_text(parse_node_path(from_node), from_output)}[{from_call_number}]"
            reason = f"'{where}' can't be read as the table at '{same_as}': {reason}"
            raise InvalidCallError([call_problems.saved_run_unreadable("saved_run", where="tables_written_once", reason=reason)]) from None


def reusable_from_saved(saved: SavedReusable, in_call_order: list[Outcome | None]) -> ReusableOutcomes:
    """Returns the outcomes one node may reuse from the previous run, from their saved shape.

    The three saved outcomes of ``shout`` with ``"item_indexes": {"0": [0], "1": [1], "2": [2]}`` come back as a
    ``ReusableOutcomes`` that has each of them twice: by the item indexes its call had, and by the fingerprint
    of its call's arguments.

    Args:
        saved: The saved outcomes and what identifies each.
        in_call_order: ``saved.outcomes`` as ``outcomes_from_saved`` returned them.

    Raises:
        ValueError: ``fingerprints`` is not as long as the lists of ``outcomes``; or an outcome is not one a call
            produced: it is skipped, not called or missing.
    """
    if len(saved.fingerprints) != len(in_call_order):
        raise ValueError("'fingerprints' is not as long as the lists of 'outcomes'")
    item_indexes, outer_item_fingerprints = saved.item_indexes, saved.outer_item_fingerprints
    reusable = ReusableOutcomes()
    for place, (fingerprint, outcome) in enumerate(zip(saved.fingerprints, in_call_order, strict=True)):
        if type(outcome) is Done:
            reusable.by_fingerprint.setdefault(fingerprint, outcome)
            if place in item_indexes:
                reusable.by_item_indexes[item_indexes[place]] = (fingerprint, outcome)
        elif type(outcome) is Answered or type(outcome) is Unanswered:
            reusable.answers.setdefault((fingerprint, outer_item_fingerprints.get(place, "")), []).append(outcome)
        else:
            raise ValueError("an outcome is skipped, not called or missing, and only an outcome of a call is reused")
    return reusable
