"""Writes a run as JSON (``run_to_json``), and a complete run's graph outputs as JSON (``outputs_to_json``).

``saved_run`` turns the run into an instance of its graph's codec, the subclass of ``SavedRun`` made for the graph,
which writes itself as text.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

import pydantic_core
from pydantic_core import PydanticSerializationError

from conductor._internal.ask import Question
from conductor._internal.base import SKIPPED, NodePath, place_text
from conductor._internal.field import Field
from conductor._internal.problem import InvalidCallError, call_problems
from conductor._internal.types import Type

from ..flat import FlatFunctionNode
from ..outcome import Answered, Done, NoneInput, Outcome, Skipped, Unanswered
from ..progress import all_item_indexes, has_every_outcome
from ..reuse import ReusableOutcomes
from .codec import saved_type
from .type import (
    SAVED_DONE,
    SAVED_SKIPPED,
    SameAs,
    SavedOutcomes,
    SavedOutcomeState,
    SavedReusable,
    SavedRun,
)

if TYPE_CHECKING:
    from ..type import Run

#: What ``states`` says for an outcome of each class; an outcome of any other class, and a call with no outcome,
#: is written as ``None``.
_STATE_OF: Mapping[type, SavedOutcomeState] = {Done: SAVED_DONE, Answered: SAVED_DONE, Skipped: SAVED_SKIPPED}

#: The outcome classes whose outputs are written: what a function returned, or what a person answered.
_HAS_OUTPUTS = (Done, Answered)

#: The classes whose question is written: a call answered, or waiting.
_HAS_QUESTION = (Answered, Unanswered)


def run_to_json(run: Run) -> str:
    r"""Returns the whole run as JSON text.

    For the finished run of the README graph on ``"a\nb\nc"`` the text starts
    ``{"graph":"<16 hex characters>","state":{"status":"complete"},"max_items":null,"inputs":{"text":"a\nb\nc"},``
    and goes on with every node's outcomes. ``SavedRun`` shows all of it.

    Raises:
        InvalidCallError: A value cannot be written as JSON (``run_unwritable``, naming the input or the output).
    """
    saved = saved_run(run)
    try:
        return saved.model_dump_json(exclude_defaults=True)
    except PydanticSerializationError:
        where = first_unwritable_value(saved, run._flat.function_nodes)
        raise InvalidCallError([call_problems.run_unwritable("saved_run", where, where=where, reason="a value has no JSON form")]) from None


def outputs_to_json(run: Run) -> str:
    """Returns the graph outputs of a complete run (``run.outputs``) as JSON text, by name.

    For the README run it returns ``{"result":"A!+B!+C!"}``. Each value is written by its output's type, as
    in a saved run: a table as its columns, ``inf`` and ``nan`` as text, a host type by its own schema.

    Raises:
        InvalidCallError: The run is not complete (``run_not_complete``), or a value cannot be written as JSON
            (``run_unwritable``, naming the output).
    """
    written = {}
    for text, value in run.outputs.items():
        source = run.graph._outputs[text]
        function, output = run._flat.nodes[source.node].source_of(source.output)
        hint: Any = saved_type(function.compiled_node.outputs[output]).hint
        for _ in function.loops:  # one list, or None in its place, for each loop the function node is inside
            hint = list[hint] | None
        try:
            written[text] = Type(hint).to_json(value)
        except PydanticSerializationError:
            raise InvalidCallError([call_problems.run_unwritable("outputs", text, where=text, reason="a value has no JSON form")]) from None
    return pydantic_core.to_json(written).decode()


def saved_run(run: Run) -> SavedRun:
    """Returns ``run`` as an instance of its graph's codec, which writes itself as JSON.

    For the README run, ``saved_run(run).outcomes["shout"]`` has ``states=["done", "done", "done"]`` and
    ``outputs={"result": ["A!", "B!", "C!"]}``, and is of the codec's class for ``shout``. Each part is of the
    codec's own class, so the run is the one the codec reads back from its JSON.

    Only function nodes are saved, since only they are called. A node inside a subgraph is saved like any
    other, under its node path as text (``"loud/shout"``). A node with no outcome and nothing to reuse has no
    key at all.
    """
    codec = run.graph._codec
    function_nodes = run._flat.function_nodes
    names = {path: place_text(path) for path in function_nodes}
    outcomes = {
        names[path]: saved_outcomes(codec.outcomes_classes[names[path]], node.compiled_node.outputs, _outcomes_by_call_number(run, node))
        for path, node in function_nodes.items() if run._outcomes[path]
    }
    reusable = {}
    for path, node in function_nodes.items():
        if path in run._reusable.done or path in run._reusable.answers:  # every outcome in done_by_call is in done too
            reusable[names[path]] = saved_reusable(codec.reusable_classes[names[path]], codec.outcomes_classes[names[path]], node.compiled_node.outputs, run._reusable, path)
    code = {
        names[path]: run._reusable.code_changed.get(path) or node.compiled_node.code_fingerprint
        for path, node in function_nodes.items() if names[path] in outcomes or names[path] in reusable
    }
    return codec.model_construct(
        graph=run.graph.fingerprint, state=run.state, max_items=run.max_items, inputs=run.inputs, outcomes=outcomes, reusable=reusable, code=code,
        complete=[names[path] for path in function_nodes if path in run._finished_nodes],
        tables_written_once=tables_written_once(function_nodes.values(), outcomes),
    )


def _outcomes_by_call_number(run: Run, node: FlatFunctionNode) -> list[Outcome | None]:
    """Returns the outcomes of ``node`` as a list with one entry per call, in the order of the calls' item indexes.

    For ``shout`` after its first two calls of three: ``[Done(...), Done(...), None]``. The place of an entry
    is the call number of its call; ``None`` is a call that has no outcome yet.
    """
    outcomes = run._outcomes[node.node_path]
    # speed only: the outcomes are in call order already, as the line below would list them. Without it to_json takes 1.3 to 1.9
    # times as long.
    if has_every_outcome(run._outcomes, node):
        return list(outcomes.values())
    return [outcomes.get(item_indexes) for item_indexes in all_item_indexes(run._outcomes, node.loops, ())[0]]


def saved_outcomes(saved_class: type[SavedOutcomes], outputs: Mapping[str, Field], outcomes: list[Outcome | None]) -> SavedOutcomes:
    """Returns the outcomes of one node as a saved run writes them: a list per output, with one entry per call.

    ``[Done({"result": "A!"}), NoneInput({"result": None}, "line"), Done({"result": "C!"})]`` becomes
    ``saved_class(states=["done", None, "done"], outputs={"result": ["A!", None, "C!"]}, not_called_because={1: "line"})``.

    Args:
        saved_class: The codec's class for the node's outcomes.
        outputs: The outputs of the node, by name.
        outcomes: The outcomes in call-number order; ``None`` for a call with no outcome.
    """
    classes = set(map(type, outcomes))
    values_by_output: dict[str, list[Any]] = {}
    skipped: dict[str, list[int]] = {}
    skipped_items: dict[str, dict[int, list[int]]] = {}
    questions: dict[int, Question] = {}
    not_called_because: dict[int, str] = {}
    if not classes.isdisjoint(_HAS_OUTPUTS):
        if classes.issubset(_HAS_OUTPUTS):  # speed only: every call has outputs, as in a node that is complete
            with_outputs = cast("list[Done | Answered]", outcomes)
            values_by_output = {name: [outcome.outputs[name] for outcome in with_outputs] for name in outputs}
        else:
            values_by_output = {
                name: [outcome.outputs[name] if isinstance(outcome, _HAS_OUTPUTS) else None for outcome in outcomes]
                for name in outputs
            }
        for name, output in outputs.items():
            values = values_by_output[name]
            skipped_calls = [call_number for call_number, value in enumerate(values) if value is SKIPPED]
            if skipped_calls:
                skipped[name] = skipped_calls
                for call_number in skipped_calls:
                    values[call_number] = None
            if output.is_series:
                for call_number, series in enumerate(values):
                    if series is not None:
                        values[call_number], skipped_indexes = series.for_saving()
                        if skipped_indexes:
                            skipped_items.setdefault(name, {})[call_number] = skipped_indexes
    else:
        values_by_output = {name: [None] * len(outcomes) for name in outputs}
    if not classes.isdisjoint(_HAS_QUESTION):
        questions = {
            call_number: outcome.question for call_number, outcome in enumerate(outcomes)
            if isinstance(outcome, _HAS_QUESTION)
        }
    if NoneInput in classes:
        not_called_because = {call_number: outcome.input for call_number, outcome in enumerate(outcomes) if type(outcome) is NoneInput}
    return saved_class.model_construct(
        states=list(map(_STATE_OF.get, map(type, outcomes))), outputs=values_by_output, questions=questions,
        not_called_because=not_called_because, skipped=skipped, skipped_items=skipped_items,
    )


def tables_written_once(function_nodes: Iterable[FlatFunctionNode], outcomes: Mapping[str, SavedOutcomes]) -> list[SameAs]:
    """Finds the outputs that have the very same table object as an output before them, and writes each such table once.

    ``read.result`` is a table and ``split.result`` is a series of records made of that same table. The entry
    of ``split.result`` in ``outcomes`` is replaced by ``None``, and the result is
    ``[SameAs(output=("split", "result", 0), same_as=("read", "result", 0))]``.

    Args:
        function_nodes: Every function node of the run's graph, in the order they are saved.
        outcomes: The saved outcomes of each node, as ``saved_outcomes`` returned them. Changed in place.

    Returns:
        One entry for each output replaced by ``None``.
    """
    first: dict[int, tuple[Any, tuple[str, str, int]]] = {}  # by id() of a table, which stays alive in ``outcomes`` during this call
    same: list[SameAs] = []
    for node in function_nodes:
        name = place_text(node.node_path)
        of_node = outcomes.get(name)
        if of_node is None:
            continue
        for output, field in node.compiled_node.outputs.items():
            values = of_node.outputs[output]
            for call_number, stored in enumerate(values):
                table = None if stored is None else field.whole.written_once(stored)
                if table is None:
                    continue
                before = first.setdefault(id(table), (table, (name, output, call_number)))
                if before[1] != (name, output, call_number) and before[0] is table:
                    values[call_number] = None
                    same.append(SameAs.model_construct(output=(name, output, call_number), same_as=before[1]))
    return same


def saved_reusable(
    saved_class: type[SavedReusable], outcomes_class: type[SavedOutcomes], outputs: Mapping[str, Field], reusable: ReusableOutcomes, path: NodePath,
) -> SavedReusable:
    """Returns the outcomes one node may reuse from the previous run as a saved run writes them, each outcome once.

    ``SavedReusable`` shows the JSON. The order is: the outcomes known by the item indexes their call had, then
    the outcomes known only by the fingerprint of their call's arguments, then the answers and the waiting
    questions.

    Args:
        saved_class: The codec's class for the node's reusable outcomes.
        outcomes_class: The codec's class for the node's outcomes.
        outputs: The outputs of the node, by name.
        reusable: The run's reusable outcomes.
        path: The node's path.
    """
    done_by_call = reusable.done_by_call.get(path, {})
    outcomes: list[Outcome | None] = [done for _, done in done_by_call.values()]
    fingerprints = [fingerprint for fingerprint, _ in done_by_call.values()]
    item_indexes = dict(enumerate(done_by_call))
    already_saved = set(fingerprints)  # ``done`` has these outcomes too: the first call that got those arguments
    for fingerprint, done in reusable.done.get(path, {}).items():
        if fingerprint not in already_saved:
            outcomes.append(done)
            fingerprints.append(fingerprint)
    outer_item_fingerprints: dict[int, str] = {}
    for (fingerprint, outer_items_fingerprint), answers in reusable.answers.get(path, {}).items():
        for outcome in answers:
            if outer_items_fingerprint:
                outer_item_fingerprints[len(outcomes)] = outer_items_fingerprint
            outcomes.append(outcome)
            fingerprints.append(fingerprint)
    return saved_class.model_construct(
        outcomes=saved_outcomes(outcomes_class, outputs, outcomes), fingerprints=fingerprints, item_indexes=item_indexes,
        outer_item_fingerprints=outer_item_fingerprints,
    )


def first_unwritable_value(saved: SavedRun, function_nodes: Iterable[NodePath]) -> str:
    """Returns where a run that could not be written has a value with no JSON form: ``"text"``, ``"loud/shout.result"``.

    It writes each graph input and each output's list on its own, through the graph's codec (``include=`` names the
    one value), and returns the first that fails. It is called only after pydantic refused the whole run, so a run
    that can be written pays nothing for it.

    Args:
        saved: The run as ``saved_run`` returned it.
        function_nodes: The path of every function node of the run's graph, in the order they are saved.

    Returns:
        The name of the graph input, or the output as ``place_text`` writes it; ``"the run"`` when each is
        written alone.
    """
    alone: list[tuple[str, Any]] = [(name, {"inputs": {name}}) for name in saved.inputs]
    for path in function_nodes:
        if (outcomes := saved.outcomes.get(name := place_text(path))) is not None:
            alone.extend((place_text(path, output), {"outcomes": {name: {"outputs": {output}}}}) for output in outcomes.outputs)
    for path in function_nodes:
        if (reusable := saved.reusable.get(name := place_text(path))) is not None:
            alone.extend((place_text(path, output), {"reusable": {name: {"outcomes": {"outputs": {output}}}}}) for output in reusable.outcomes.outputs)
    for where, include in alone:
        try:
            saved.model_dump_json(include=include)
        except PydanticSerializationError:
            return where
    return "the run"
