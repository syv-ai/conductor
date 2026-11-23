"""Reads a run back from the JSON that ``run.to_json()`` wrote.

``run_from_json`` is where a saved run comes in. First the graph's codec checks the whole text: every key and
every value, and each node's lists against each other (``SavedOutcomes``). Then ``outcomes_from_saved`` turns each
node's saved lists back into the outcomes a run stores, and ``ReusableOutcomes.add`` keeps those of ``reusable`` that
a call may still reuse. Last, ``run_from_json`` checks what only the graph can show: that each node's lists are as
long as its number of calls, and that a declined call is one the graph has.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from conductor._internal.base import READ_BY_NAME, SKIPPED, NodePath, parse_node_path, place_text
from conductor._internal.graph import CompiledFunctionNode, Graph
from conductor._internal.problem import (
    InvalidCallError,
    Problem,
    call_problems,
    conductor_error,
    shape_error,
    value_free_errors,
)

from ..outcome import Done, NoneInput, Outcome, Skipped
from ..outcomes import RunOutcomes
from ..reuse import ReusableOutcomes
from ..state import EndedEarly
from .codec import where_in
from .type import SAVED_DONE, SAVED_SKIPPED, SavedOutcomes, question_disagreement


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
    switched to ``reuse_when_inputs_match=False`` after the save is called again for every item. One that can't be
    reused, such as a waiting question of a node that no longer asks, is dropped. An outcome that other code made
    is kept, but no call reuses it, since its fingerprint keeps that code.

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
        # a field is saved under its name, never an alias (the codec's _SavedValue), so it is read by its name
        read = codec.model_validate_json(saved, **READ_BY_NAME)
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
    for name, saved_outcomes in read.outcomes.items():
        path = parse_node_path(name)
        in_call_order[path] = outcomes_from_saved(
            blueprint.function_nodes[path].compiled_node, saved_outcomes
        )
    reusable = ReusableOutcomes()
    for name, saved_reusable in read.reusable.items():
        node = blueprint.function_nodes[parse_node_path(name)]
        reusable_outcomes = outcomes_from_saved(node.compiled_node, saved_reusable)
        reusable.add(node, zip(saved_reusable.item_indexes, reusable_outcomes, strict=True))
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
    if read.ended_early is not None:
        try:
            ended_early = _checked_ending(read.ended_early, outcomes)
        except ValueError as wrong:
            raise InvalidCallError([_lists_disagree("ended_early", str(wrong))]) from None
    else:
        ended_early = None
    return run_type(
        graph,
        read.max_items,
        outcomes,
        reusable,
        _last_resume=ended_early,
        _complete_nodes=frozenset(parse_node_path(name) for name in read.complete),
    )


def _checked_ending(ended_early: EndedEarly, outcomes: RunOutcomes) -> EndedEarly:
    """Returns how a saved run ended early, once its declined call is one this graph's run could have.

    The codec has read an ``EndedEarly``, which holds its status, cause and declined call to one another. Here the
    declined call is held to the graph: it is a call the graph has, with no outcome of its own any more, and its
    question names only outputs of the node (``question_disagreement``). Whether the node asks isn't checked: a deploy may
    make it stop asking, and a run read back keeps what it did, as it keeps an answered call's question.

    Raises:
        ValueError: Any of that doesn't hold. The message names no value.
    """
    declined = ended_early.declined
    if declined is None:
        return ended_early
    node = outcomes.blueprint.function_nodes.get(declined.node_path)
    if node is None:
        raise ValueError("the declined call is not of a node of the graph")
    if declined.item_indexes not in outcomes.item_indexes(node.loops)[0]:
        raise ValueError("the node has no call at the declined call's item indexes")
    if outcomes.of(declined.node_path).get(declined.item_indexes) is not None:
        raise ValueError("the declined call has an outcome")
    if (wrong := question_disagreement(declined.question, node.compiled_node.outputs)) is not None:
        raise ValueError(wrong)
    return ended_early


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
    ``Done`` with its fingerprint. A question under ``questions`` is a waiting call's ``Unanswered`` where the call
    has no state, and the ``asked`` of its ``Done`` where a person answered it.

    Args:
        node: The node, as compiled.
        saved: The saved lists, as the codec read them: each value checked by its output's type, and the lists in
            agreement (``SavedOutcomes``).
    """
    values_by_output = saved.outputs
    for name, call_numbers in saved.skipped.items():
        for call_number in call_numbers:
            values_by_output[name][call_number] = SKIPPED
    for name, by_call_number in saved.skipped_items.items():
        for call_number, indexes in by_call_number.items():
            for index in indexes:
                values_by_output[name][call_number][index] = SKIPPED
    skipped_outcome = Skipped(node.outputs_when_skipped)
    outcomes: list[Outcome | None] = []
    for call_number, (state, fingerprint) in enumerate(
        zip(saved.states, saved.fingerprints, strict=True)
    ):
        if state == SAVED_DONE:
            of_call = {name: values[call_number] for name, values in values_by_output.items()}
            outcomes.append(Done(of_call, fingerprint=fingerprint))
        elif state == SAVED_SKIPPED:
            outcomes.append(skipped_outcome)
        else:
            outcomes.append(None)
    for call_number, saved_question in saved.questions.items():
        asked = dataclasses.replace(saved_question, fingerprint=saved.fingerprints[call_number])
        answered = outcomes[call_number]
        # a question a person answered, or one still waiting
        outcomes[call_number] = (
            dataclasses.replace(answered, asked=asked) if isinstance(answered, Done) else asked
        )
    for call_number, input_name in saved.none_inputs.items():
        outcomes[call_number] = NoneInput(node.outputs_when_none_input, input_name)
    return outcomes
