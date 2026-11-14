"""Reads a run from the JSON that ``run.to_json()`` wrote.

``run_from_json`` is the door. The graph's codec validates the whole text: every key and every value. Then
``outcomes_from_saved`` and ``add_reusable_from_saved`` turn each node's saved lists into what a run keeps in memory, and
``run_from_json`` checks what only the lists together can show: that the lists of each node are as long as its calls.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, assert_never

from pydantic import ValidationError

from conductor._internal.base import SKIPPED, ItemIndexes, NodePath, parse_node_path, place_text
from conductor._internal.field import Field
from conductor._internal.graph import (
    Graph,
    outputs_when_not_called,
    outputs_when_skipped,
)
from conductor._internal.problem import InvalidCallError, call_problems, value_free_errors

from ..blueprint import BlueprintFunctionNode, RunBlueprint
from ..outcome import Done, NoneInput, Outcome, Skipped, Unanswered
from ..progress import all_item_indexes
from ..reuse import ReusableOutcomes
from .codec import where_in
from .type import (
    SAVED_DONE,
    SAVED_SKIPPED,
    SavedOutcomes,
    SavedReusable,
    SavedRun,
)


def run_from_json[R](run_type: Callable[..., R], saved: str | bytes, graph: Graph[Any], blueprint: RunBlueprint, codec: type[SavedRun]) -> R:
    """Returns the run of ``graph`` that ``run_to_json`` wrote as ``saved``.

    For the README graph, ``run_from_json(Run, run.to_json(), graph)`` returns a run with the inputs, the
    outcomes and the status of ``run``: its ``results["shout"]["result"]`` is ``["A!", "B!", "C!"]`` again.

    The graph's codec validates the JSON text in one pass, and reads each value the way its type reads JSON: a
    date from its text, for example. Each node's saved lists are then turned into a list of outcomes in
    call-number order, because item indexes are not saved. They are worked out here, node by node in topological
    order: a node's item indexes follow from the lengths of the series upstream of it, and those are loaded by then.

    Args:
        run_type: The class ``Run``. It is passed in because ``run/type.py`` imports this package.
        saved: The JSON text.
        graph: The graph the run was made on.
        blueprint: The graph's blueprint.
        codec: The graph's codec.

    Raises:
        InvalidCallError: ``saved`` was written for another graph (``saved_run_of_another_graph``), or is not a
            saved run of this graph (``saved_run_unreadable``, whose ``where`` names the key, the node, the output
            and the call number, and which quotes no value).
    """
    try:
        read = codec.model_validate_json(saved)
    except ValidationError as error:
        first = error.errors()[0]
        if first["type"] == "literal_error" and first["loc"] == ("graph",):  # the fingerprint under "graph" is not this graph's
            raise InvalidCallError([call_problems.saved_run_of_another_graph("saved_run", saved=first["input"], graph=graph.fingerprint)]) from None
        reason = value_free_errors(error)[0]["msg"]
        raise InvalidCallError([call_problems.saved_run_unreadable("saved_run", where=where_in(codec, first["loc"]), reason=reason)]) from None
    in_call_order: dict[NodePath, list[Outcome | None]] = {}
    code_changed = {
        path: fingerprint for name, fingerprint in read.code.items()
        if fingerprint != blueprint.function_nodes[path := parse_node_path(name)].compiled_node.code_fingerprint
    }
    reusable = ReusableOutcomes(code_changed=code_changed)
    try:
        for name, saved_outcomes in read.outcomes.items():
            where, path = f"outcomes.{name}", parse_node_path(name)
            in_call_order[path] = outcomes_from_saved(blueprint.function_nodes[path].compiled_node.outputs, saved_outcomes)
        for name, saved_reusable in read.reusable.items():
            where, path = f"reusable.{name}.outcomes", parse_node_path(name)
            reusable_in_call_order = outcomes_from_saved(blueprint.function_nodes[path].compiled_node.outputs, saved_reusable.outcomes)
            where = f"reusable.{name}"
            add_reusable_from_saved(reusable, blueprint.function_nodes[path], saved_reusable, reusable_in_call_order)
    except ValueError as wrong:
        raise InvalidCallError([call_problems.saved_run_unreadable("saved_run", where=where, reason=str(wrong))]) from None
    outcomes: dict[NodePath, dict[ItemIndexes, Outcome]] = {}
    for path, node in blueprint.function_nodes.items():
        outcomes[path] = {}
        if (of_node := in_call_order.get(path)) is not None:
            every_item_indexes, _ = all_item_indexes(outcomes, node.loops, ())
            if len(of_node) != len(every_item_indexes):
                reason = f"its lists have {len(of_node)} entries, but the saved series give the node {len(every_item_indexes)} calls"
                raise InvalidCallError([call_problems.saved_run_unreadable("saved_run", where=f"outcomes.{place_text(path)}", reason=reason)])
            outcomes[path] = {
                item_indexes: outcome for item_indexes, outcome in zip(every_item_indexes, of_node, strict=True) if outcome is not None
            }
    reusable.drop(code_changed)  # outcomes that other code produced are never reused
    return run_type(
        graph, read.inputs, read.max_items, outcomes, reusable, _last_resume=read.ended_early,
        _complete_nodes=frozenset(parse_node_path(name) for name in read.complete),
    )


def outcomes_from_saved(outputs: Mapping[str, Field], saved: SavedOutcomes) -> list[Outcome | None]:
    """Returns one node's outcomes from their saved lists, in call-number order; ``None`` for a call with no outcome.

    For ``shout`` not called for its second line::

        {"states": ["done", null, "done"], "outputs": {"result": ["A!", null, "C!"]}, "not_called_because": {"1": "line"}}

    it returns ``[Done({"result": "A!"}), NoneInput({"result": None}, "line"), Done({"result": "C!"})]``.

    Args:
        outputs: The outputs of the node, by name.
        saved: The saved lists, each value already validated by its output's type.

    Raises:
        ValueError: The list of an output is not as long as ``states``; a done call has ``null`` for an output
            whose type takes no ``None`` and that ``skipped`` does not list; or ``skipped``, ``skipped_items``,
            ``questions`` or ``not_called_because`` names a call that does not exist or that cannot have it.
    """
    states, values_by_output = saved.states, saved.outputs
    if any(len(values) != len(states) for values in values_by_output.values()):
        raise ValueError("the list of an output is not as long as 'states'")
    for name, values in values_by_output.items():
        if not outputs[name].whole.optional:
            skipped_calls = set(saved.skipped.get(name, ()))
            for call_number, (state, value) in enumerate(zip(states, values, strict=True)):
                if state == SAVED_DONE and value is None and call_number not in skipped_calls:
                    raise ValueError(f"call {call_number} is done and has null for '{name}', whose type takes no None")
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
        for call_number, waiting in saved.questions.items():
            if states[call_number] is not None:
                raise LookupError(call_number)
            outcomes[call_number] = waiting
        for call_number, input_name in saved.not_called_because.items():
            if outcomes[call_number] is not None:
                raise LookupError(call_number)
            outcomes[call_number] = NoneInput(outputs_when_not_called(outputs), input_name)
    except (LookupError, TypeError, AttributeError):  # a call number past the end of the lists, or a call of another kind
        message = "'skipped', 'skipped_items', 'questions' or 'not_called_because' names a call that does not exist or cannot have it"
        raise ValueError(message) from None
    return outcomes


def add_reusable_from_saved(reusable: ReusableOutcomes, node: BlueprintFunctionNode, saved: SavedReusable, in_call_order: list[Outcome | None]) -> None:
    """Adds the outcomes one node may reuse from the previous run to ``reusable``, from their saved shape.

    The three saved outcomes of ``shout`` with ``"item_indexes": {"0": [0], "1": [1], "2": [2]}`` are added each
    twice: by the item indexes its call had, and by the fingerprint of its call's arguments.

    Args:
        reusable: The run's reusable outcomes, added to.
        node: The node.
        saved: The saved outcomes and what identifies each.
        in_call_order: ``saved.outcomes`` as ``outcomes_from_saved`` returned them.

    Raises:
        ValueError: ``fingerprints`` is not as long as the lists of ``outcomes``; or an outcome is not one a call
            produced: it is skipped, not called or missing.
    """
    if len(saved.fingerprints) != len(in_call_order):
        raise ValueError("'fingerprints' is not as long as the lists of 'outcomes'")
    item_indexes, outer_item_fingerprints = saved.item_indexes, saved.outer_item_fingerprints
    for place, (fingerprint, outcome) in enumerate(zip(saved.fingerprints, in_call_order, strict=True)):
        match outcome:
            case Done() | Unanswered():
                reusable.add(node, fingerprint, outcome, item_indexes=item_indexes.get(place), outer_items=outer_item_fingerprints.get(place, ""))
            case Skipped() | NoneInput() | None:
                raise ValueError("an outcome is skipped, not called or missing, and only an outcome of a call is reused")
            case _:
                assert_never(outcome)
