"""Reads a run back from the JSON that ``run.to_json()`` wrote.

``run_from_json`` is where a saved run comes in. First the graph's codec checks the whole text: every key and
every value. Then ``outcomes_from_saved`` turns each node's saved lists back into the outcomes a run stores, and
``ReusableOutcomes.add`` keeps those of ``reusable`` that a call may still reuse. Last, ``run_from_json`` checks what only the lists together can show: that each
node's lists are as long as its number of calls.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from conductor._internal.base import SKIPPED, NodePath, parse_node_path, place_text
from conductor._internal.graph import CompiledFunctionNode, Graph
from conductor._internal.problem import (
    InvalidCallError,
    Problem,
    call_problems,
    conductor_error,
    shape_error,
    value_free_errors,
)
from conductor._internal.series import SeriesItems

from ..outcome import Done, NoneInput, Outcome, Skipped, Unanswered
from ..outcomes import RunOutcomes
from ..reuse import ReusableOutcomes
from .codec import where_in
from .type import SAVED_DONE, SAVED_SKIPPED, SavedOutcomes


def run_from_json[R](run_type: Callable[..., R], saved: str | bytes, graph: Graph[Any]) -> R:
    """Returns the run of ``graph`` that ``run_to_json`` saved as ``saved``.

    For the README graph, ``run_from_json(Run, run.to_json(), graph)`` returns a run with the inputs, the outcomes
    and the status of ``run``. Its ``results["shout"]["result"]`` is ``["A!", "B!", "C!"]`` again.

    The graph's codec checks the JSON text in one pass, and converts each value back by its type, so a date comes
    back from its text, for example. Item indexes aren't saved, so each node's saved lists come back as outcomes
    in call-number order. The item indexes are then worked out node by node, in the order a run tries them. A
    node's item indexes follow from the lengths of the series upstream of it, and those are read by then.

    The outcomes the saved run could still reuse go through ``ReusableOutcomes.add``, as a run made from a previous
    one does. So they are kept only for a node whose ``Policy`` says ``reuse_when_inputs_match`` now, and a node
    switched to ``reuse_when_inputs_match=False`` after the save is called again for every item. An outcome that
    other code made is kept, but no call reuses it, since its fingerprint keeps that code. A waiting question of a
    node that no longer asks is dropped, from the run's outcomes and from what it may reuse (``outcomes_from_saved``).
    So the next ``resume`` calls the node's code for it.

    Args:
        run_type: The class ``Run``. It is passed in because ``run/type.py`` imports this package.
        saved: The JSON text.
        graph: The graph the run was made on.

    Raises:
        InvalidCallError: ``saved_run_of_another_graph`` if ``saved`` was written for another graph.
            ``saved_run_unreadable`` if ``saved`` isn't a saved run of this graph. Its ``where`` names the key,
            node, output and call number, and it quotes no value.
    """
    blueprint, codec = graph._blueprint, graph._codec
    try:
        read = codec.model_validate_json(saved)
    except ValidationError as error:
        first = error.errors()[0]
        # the fingerprint under "graph" is not this graph's
        if first["type"] == "literal_error" and first["loc"] == ("graph",):
            saved_graph = first["input"]
            # quoted only when it has a fingerprint's form, since a hand-edited save may hold any value there
            is_fingerprint = (
                isinstance(saved_graph, str)
                and len(saved_graph) == len(graph.fingerprint)
                and re.fullmatch("[0-9a-f]+", saved_graph) is not None
            )
            raise InvalidCallError(
                [
                    call_problems.saved_run_of_another_graph(
                        "saved_run",
                        graph=graph.fingerprint,
                        saved=saved_graph if is_fingerprint else None,
                    )
                ]
            ) from None
        refused = value_free_errors(error)[0]
        raise InvalidCallError(
            [
                call_problems.saved_run_unreadable(
                    "saved_run",
                    where=where_in(codec, first["loc"]),
                    reason=refused["msg"],
                    errors=[refused],
                )
            ]
        ) from None
    in_call_order: dict[NodePath, list[Outcome | None]] = {}
    reusable = ReusableOutcomes()
    try:
        for name, saved_outcomes in read.outcomes.items():
            where, path = f"outcomes.{name}", parse_node_path(name)
            in_call_order[path] = outcomes_from_saved(
                blueprint.function_nodes[path].compiled_node, saved_outcomes
            )
        for name, saved_reusable in read.reusable.items():
            where, path = f"reusable.{name}", parse_node_path(name)
            node = blueprint.function_nodes[path]
            reusable_outcomes = outcomes_from_saved(node.compiled_node, saved_reusable)
            if len(saved_reusable.item_indexes) != len(reusable_outcomes):
                raise ValueError("'item_indexes' is not as long as 'states'")
            reusable.add(node, zip(saved_reusable.item_indexes, reusable_outcomes))
    except ValueError as wrong:
        raise InvalidCallError([_lists_disagree(where, str(wrong))]) from None
    outcomes = RunOutcomes(blueprint, read.inputs)
    for path, node in blueprint.function_nodes.items():
        if (of_node := in_call_order.get(path)) is not None:
            every_item_indexes, _ = outcomes.item_indexes(node.loops)
            if len(of_node) != len(every_item_indexes):
                reason = f"its lists have {len(of_node)} entries, but the saved series give the node {len(every_item_indexes)} calls"
                raise InvalidCallError([_lists_disagree(f"outcomes.{place_text(path)}", reason)])
            for item_indexes, outcome in zip(every_item_indexes, of_node, strict=True):
                if outcome is not None:
                    outcomes.record(node, item_indexes, outcome)
    return run_type(
        graph,
        read.max_items,
        outcomes,
        reusable,
        _last_resume=read.ended_early,
        _complete_nodes=frozenset(parse_node_path(name) for name in read.complete),
    )


def _lists_disagree(where: str, reason: str) -> Problem:
    """Returns the ``saved_run_unreadable`` problem for saved lists that don't agree with each other or the graph.

    Pydantic read each list, so the fault is one conductor found, and its error is ``saved_lists_disagree``.

    Args:
        where: Where in the saved run, as ``ctx.where`` says it.
        reason: What disagrees, naming only outputs, call numbers and counts.
    """
    error = shape_error(conductor_error("saved_lists_disagree", reason))
    return call_problems.saved_run_unreadable(
        "saved_run", where=where, reason=error["msg"], errors=[error]
    )


def outcomes_from_saved(node: CompiledFunctionNode, saved: SavedOutcomes) -> list[Outcome | None]:
    """Returns one node's outcomes from their saved lists, in call-number order.

    A call with no outcome is ``None`` in the list. Say ``shout`` wasn't called for its second line::

        {"states": ["done", null, "done"], "outputs": {"result": ["A!", null, "C!"]},
         "fingerprints": ["<fp a>", null, "<fp c>"], "none_inputs": {"1": "line"}}

    That gives ``[Done({"result": "A!"}), NoneInput({"result": None}, "line"), Done({"result": "C!"})]``, each
    ``Done`` with its fingerprint.

    No call of a node that never asks waits for a person. So when the node's code asked at the save and no longer
    does, each waiting call reads as a call not made.

    Args:
        node: The node, as compiled.
        saved: The saved lists. Each value has already been checked by its output's type.

    Raises:
        ValueError: An output's list, or ``fingerprints``, isn't as long as ``states``. Or a call that is neither
            done nor waiting has a fingerprint. Or a done call has ``null`` for an output whose
            type takes no ``None``, or for an item whose type takes none, and ``skipped`` or ``skipped_items``
            doesn't list it. Or ``skipped``, ``skipped_items``, ``questions`` or ``none_inputs`` names a
            call that doesn't exist or can't have it.
    """
    states, values_by_output, fingerprints = saved.states, saved.outputs, saved.fingerprints
    if any(len(values) != len(states) for values in values_by_output.values()):
        raise ValueError("the list of an output is not as long as 'states'")
    if len(fingerprints) != len(states):
        raise ValueError("'fingerprints' is not as long as 'states'")
    try:
        for name, call_numbers in saved.skipped.items():
            for call_number in call_numbers:
                values_by_output[name][call_number] = SKIPPED
        for name, by_call_number in saved.skipped_items.items():
            for call_number, indexes in by_call_number.items():
                for index in indexes:
                    values_by_output[name][call_number][index] = SKIPPED
        outputs_by_call = [
            {name: values[call_number] for name, values in values_by_output.items()}
            for call_number in range(len(states))
        ]
        skipped_outcome = Skipped(node.outputs_when_skipped)
        outcomes: list[Outcome | None] = []
        for state, of_call, fingerprint in zip(states, outputs_by_call, fingerprints, strict=True):
            if state == SAVED_DONE:
                outcomes.append(Done(of_call, fingerprint=fingerprint))
            elif state == SAVED_SKIPPED:
                outcomes.append(skipped_outcome)
            else:
                outcomes.append(None)
        for call_number, waiting in saved.questions.items():
            if states[call_number] is not None:
                raise LookupError(call_number)
            outcomes[call_number] = dataclasses.replace(
                waiting, fingerprint=fingerprints[call_number]
            )
        for call_number, input_name in saved.none_inputs.items():
            if outcomes[call_number] is not None:
                raise LookupError(call_number)
            outcomes[call_number] = NoneInput(node.outputs_when_none_input, input_name)
    except (
        LookupError,
        TypeError,
        AttributeError,
    ):  # a call number past the end of the lists, or a call of another kind
        message = "'skipped', 'skipped_items', 'questions' or 'none_inputs' names a call that does not exist or cannot have it"
        raise ValueError(message) from None
    if any(
        fingerprint is not None and not isinstance(outcome, (Done, Unanswered))
        for fingerprint, outcome in zip(fingerprints, outcomes, strict=True)
    ):
        raise ValueError("a call that is neither done nor waiting has a fingerprint")
    _refuse_null_where_none_is_refused(node, states, values_by_output)
    if not node.asks:
        # a question saved while the node's code asked: no call of a node that never asks waits for a person
        return [None if isinstance(outcome, Unanswered) else outcome for outcome in outcomes]
    return outcomes


def _refuse_null_where_none_is_refused(
    node: CompiledFunctionNode, states: list[Any], values_by_output: dict[str, list[Any]]
) -> None:
    """Refuses a ``null`` in a done call's output where the output's type takes no ``None``.

    A saved run writes ``SKIPPED`` as ``null``, so its codec takes ``null`` for every output and every item of a
    ``Series``. Once ``skipped`` and ``skipped_items`` have put ``SKIPPED`` back, a ``None`` left over is one the
    node returned. A node that returned it would have failed with ``invalid_output``, so a save is refused for it
    too. Say ``split`` returns ``Series[str]``, and its saved ``["a", null]`` lists no skipped item: it is refused.

    Raises:
        ValueError: A done call has ``None`` for an output, or for an item of a ``Series`` output, whose type takes
            no ``None``.
    """
    for name, values in values_by_output.items():
        field = node.outputs[name]
        takes_none, items_take_none = field.whole.optional, field.type.optional
        # an output whose value and items may be None has nothing to refuse
        if takes_none and items_take_none:
            continue
        # most values are not None, so a call's state is looked up only for a value that may be refused
        for call_number, value in enumerate(values):
            if value is None and not takes_none and states[call_number] == SAVED_DONE:
                raise ValueError(
                    f"call {call_number} is done and has null for '{name}', whose type takes no None"
                )
            if (
                not items_take_none
                and type(value) is SeriesItems
                and states[call_number] == SAVED_DONE
                and any(item is None for item in value)
            ):
                raise ValueError(
                    f"call {call_number} is done and has null for an item of '{name}', whose items take no None"
                )
