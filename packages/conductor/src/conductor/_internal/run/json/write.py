"""Writes a run as JSON (``run_to_json``), and a complete run's graph outputs as JSON (``outputs_to_json``).

``saved_run`` turns the run into the classes of ``type.py``, and the graph's codec writes those as text.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any, cast

import pydantic_core
from pydantic_core import PydanticSerializationError

from conductor._internal.address import node_path_text
from conductor._internal.ask import Question
from conductor._internal.field import Field
from conductor._internal.markers import SKIPPED
from conductor._internal.outcomes import (
    Answered,
    Done,
    NotCalled,
    Outcome,
    Skipped,
    Waiting,
    all_item_indexes,
    has_every_outcome,
)
from conductor._internal.problem import InvalidCallError
from conductor._internal.problem import catalogue as problem
from conductor._internal.run.json.type import (
    SAVED_DONE,
    SAVED_SKIPPED,
    SameAs,
    SavedOutcomes,
    SavedOutcomeState,
    SavedReusable,
    SavedRun,
)
from conductor._internal.run.reuse import ReusableOutcomes
from conductor._internal.value_type import Type

if TYPE_CHECKING:
    from conductor._internal.compiled import FlatFunctionNode
    from conductor._internal.run.type import Run

#: What ``states`` says for an outcome of each class; an outcome of any other class, and a call with no outcome,
#: is written as ``None``.
_STATE_OF: Mapping[type, SavedOutcomeState] = {Done: SAVED_DONE, Answered: SAVED_DONE, Skipped: SAVED_SKIPPED}

#: The outcome classes whose outputs are written: what a function returned, or what a person answered.
_HAS_OUTPUTS = frozenset({Done, Answered})


def run_to_json(run: Run) -> str:
    r"""Returns the whole run as JSON text.

    For the finished run of the README graph on ``"a\nb\nc"`` the text starts
    ``{"graph":"<16 hex characters>","state":{"status":"complete"},"max_items":null,"inputs":{"lines.text":"a\nb\nc"},``
    and goes on with every node's outcomes. ``SavedRun`` shows all of it.

    Raises:
        InvalidCallError: A value cannot be written as JSON (``run_unwritable``, naming the input or the output).
    """
    saved, codec = saved_run(run), run.graph._run_codec
    try:
        return codec.serializer.to_json(saved, exclude_defaults=True).decode()
    except PydanticSerializationError:
        where = first_unwritable_value(codec.inputs, codec.outputs, saved)
        raise InvalidCallError([problem.run_unwritable("saved_run", where, where=where, reason="a value has no JSON form")]) from None


def outputs_to_json(run: Run) -> str:
    """Returns the graph outputs of a complete run (``run.outputs``) as JSON text, by address.

    For the README run it returns ``{"join.result":"A!+B!+C!"}``. Each value is written by its output's type, as
    in a saved run: a table as its columns, ``inf`` and ``nan`` as text, a host type by its own schema.

    Raises:
        InvalidCallError: The run is not complete (``run_not_complete``), or a value cannot be written as JSON
            (``run_unwritable``, naming the output).
    """
    written = {}
    for address, value in run.outputs.items():
        function, output = run._function_of(address)
        hint: Any = function.compiled_node.outputs[output].saved_type.hint
        for _ in function.loops:  # one list, or None in its place, for each loop the function node is inside
            hint = list[hint] | None
        try:
            written[address] = Type(hint).to_json(value)
        except PydanticSerializationError:
            raise InvalidCallError([problem.run_unwritable("outputs", address, where=address, reason="a value has no JSON form")]) from None
    return pydantic_core.to_json(written).decode()


def saved_run(run: Run) -> SavedRun[SavedOutcomes, SavedReusable[SavedOutcomes]]:
    """Returns ``run`` as the ``SavedRun`` that its graph's codec writes as JSON.

    For the README run, ``saved_run(run).outcomes["shout"]`` is
    ``SavedOutcomes(states=["done", "done", "done"], outputs={"result": ["A!", "B!", "C!"]})``.

    Only function nodes are saved, since only they are called. A node inside a subgraph is saved like any
    other, under its node path as text (``"loud/shout"``). A node with no outcome and nothing to reuse has no
    key at all.
    """
    graph = run.graph
    functions, codec = graph._functions, graph._run_codec
    outcomes = {
        node.name: saved_outcomes(node.compiled_node.outputs, _outcomes_by_call_number(run, node))
        for node in graph._flat_function_nodes if run._outcomes[node.name]
    }
    reusable = {
        node_path_text(path): saved_reusable(functions[path].outputs, of_node)
        for path, of_node in run._reusable.items() if of_node.by_fingerprint or of_node.answers
    }
    code = {
        name: run._code_changed.get(path) or compiled_node.code_fingerprint
        for path, compiled_node in functions.items() if (name := node_path_text(path)) in outcomes or name in reusable
    }
    return SavedRun(
        graph=graph.fingerprint, state=run.state, max_items=run.max_items, inputs=run.inputs, outcomes=outcomes, reusable=reusable, code=code,
        complete=[node.name for node in graph._flat_function_nodes if node.node_path in run._finished_nodes],
        tables_written_once=tables_written_once(codec.outputs, codec.shareable, outcomes) if codec.shareable else None,
    )


def _outcomes_by_call_number(run: Run, node: FlatFunctionNode) -> list[Outcome | None]:
    """Returns the outcomes of ``node`` as a list with one entry per call, in the order of the calls' item indexes.

    For ``shout`` after its first two calls of three: ``[Done(...), Done(...), None]``. The place of an entry
    is the call number of its call; ``None`` is a call that has no outcome yet.
    """
    outcomes = run._outcomes[node.name]
    # speed only: the outcomes are in call order already, as the line below would list them. Without it to_json takes 1.3 to 1.9
    # times as long.
    if has_every_outcome(run._outcomes, node):
        return list(outcomes.values())
    return [outcomes.get(item_indexes) for item_indexes in all_item_indexes(run._outcomes, node.loops, ())[0]]


def saved_outcomes(outputs: Mapping[str, Field], outcomes: list[Outcome | None]) -> SavedOutcomes:
    """Returns the outcomes of one node as a saved run writes them: a list per output, with one entry per call.

    ``[Done({"result": "A!"}), NotCalled({"result": None}, "line"), Done({"result": "C!"})]`` becomes
    ``SavedOutcomes(states=["done", None, "done"], outputs={"result": ["A!", None, "C!"]}, not_called_because={1: "line"})``.

    Args:
        outputs: The outputs of the node, by name.
        outcomes: The outcomes in call-number order; ``None`` for a call with no outcome.
    """
    classes = set(map(type, outcomes))
    values_by_output: dict[str, list[Any]] = {}
    skipped: dict[str, list[int]] = {}
    skipped_items: dict[str, dict[int, list[int]]] = {}
    questions: dict[int, Question] | None = None
    not_called_because: dict[int, str] | None = None
    if classes & _HAS_OUTPUTS:
        if classes <= _HAS_OUTPUTS:  # speed only: every call has outputs, as in a node that is complete
            with_outputs = cast("list[Done | Answered]", outcomes)
            values_by_output = {name: [outcome.outputs[name] for outcome in with_outputs] for name in outputs}
        else:
            values_by_output = {
                name: [outcome.outputs[name] if type(outcome) is Done or type(outcome) is Answered else None for outcome in outcomes]
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
    if Answered in classes or Waiting in classes:
        questions = {
            call_number: outcome.question for call_number, outcome in enumerate(outcomes) if type(outcome) is Answered or type(outcome) is Waiting
        }
    if NotCalled in classes:
        not_called_because = {call_number: outcome.input for call_number, outcome in enumerate(outcomes) if type(outcome) is NotCalled}
    return SavedOutcomes(
        states=list(map(_STATE_OF.get, map(type, outcomes))), outputs=values_by_output, questions=questions,
        not_called_because=not_called_because, skipped=skipped or None, skipped_items=skipped_items or None,
    )


def tables_written_once(
    outputs: Mapping[str, Mapping[str, Field]], shareable: Iterable[tuple[str, str]], outcomes: Mapping[str, SavedOutcomes],
) -> list[SameAs] | None:
    """Finds the outputs that have the very same table object as an output before them, and writes each such table once.

    ``read.result`` is a table and ``split.result`` is a series of records made of that same table. The entry
    of ``split.result`` in ``outcomes`` is replaced by ``None``, and the result is
    ``[SameAs(output=("split", "result", 0), same_as=("read", "result", 0))]``.

    Args:
        outputs: The outputs of each node, by the node's name in a saved run.
        shareable: The ``(node, output)`` pairs whose type can have such a table, in the order they are saved.
        outcomes: The saved outcomes of each node, as ``saved_outcomes`` returned them. Changed in place.

    Returns:
        One entry for each output replaced by ``None``, or ``None`` when there is none.
    """
    first: dict[int, tuple[Any, tuple[str, str, int]]] = {}  # by id() of a table, which stays alive in ``outcomes`` during this call
    same: list[SameAs] = []
    for node, output in shareable:
        of_node = outcomes.get(node)
        if of_node is None:
            continue
        written_once, values = outputs[node][output].whole.written_once, of_node.outputs[output]
        for call_number, stored in enumerate(values):
            table = None if stored is None else written_once(stored)
            if table is None:
                continue
            before = first.setdefault(id(table), (table, (node, output, call_number)))
            if before[1] != (node, output, call_number) and before[0] is table:
                values[call_number] = None
                same.append(SameAs(output=(node, output, call_number), same_as=before[1]))
    return same or None


def saved_reusable(outputs: Mapping[str, Field], reusable: ReusableOutcomes) -> SavedReusable[SavedOutcomes]:
    """Returns the outcomes one node may reuse from the previous run as a saved run writes them, each outcome once.

    ``SavedReusable`` shows the JSON. The order is: the outcomes known by the item indexes their call had, then
    the outcomes known only by the fingerprint of their call's arguments, then the answers and the waiting
    questions.

    Args:
        outputs: The outputs of the node, by name.
        reusable: The outcomes the node may reuse.
    """
    outcomes: list[Outcome | None] = [done for _, done in reusable.by_item_indexes.values()]
    fingerprints = [fingerprint for fingerprint, _ in reusable.by_item_indexes.values()]
    item_indexes = dict(enumerate(reusable.by_item_indexes))
    already_saved = set(fingerprints)  # by_fingerprint has these outcomes too: the first call that got those arguments
    for fingerprint, done in reusable.by_fingerprint.items():
        if fingerprint not in already_saved:
            outcomes.append(done)
            fingerprints.append(fingerprint)
    outer_item_fingerprints: dict[int, str] = {}
    for (fingerprint, outer_items_fingerprint), answers in reusable.answers.items():
        for outcome in answers:
            if outer_items_fingerprint:
                outer_item_fingerprints[len(outcomes)] = outer_items_fingerprint
            outcomes.append(outcome)
            fingerprints.append(fingerprint)
    return SavedReusable(
        outcomes=saved_outcomes(outputs, outcomes), fingerprints=fingerprints, item_indexes=item_indexes or None,
        outer_item_fingerprints=outer_item_fingerprints or None,
    )


def first_unwritable_value(
    inputs: Mapping[str, Type], outputs: Mapping[str, Mapping[str, Field]], saved: SavedRun[SavedOutcomes, SavedReusable[SavedOutcomes]],
) -> str:
    """Returns where a run that could not be written has a value with no JSON form: ``"lines.text"``, ``"loud/shout.result"``.

    It writes each graph input and each output's list on its own and returns the first that fails. It is
    called only after pydantic refused the whole run, so a run that can be written pays nothing for it.

    Args:
        inputs: The type of each graph input, by address.
        outputs: The outputs of each function node, by the node's name in a saved run.
        saved: The run that could not be written.

    Returns:
        The address of the graph input, or the node's name in a saved run and the output; ``"the run"`` when
        each is written alone.
    """
    written_alone = {address: (inputs[address], value) for address, value in saved.inputs.items()}
    by_node = [*saved.outcomes.items(), *((name, reusable.outcomes) for name, reusable in saved.reusable.items())]
    for name, outcomes in by_node:
        written_alone.update({f"{name}.{output}": (outputs[name][output].saved_list_type, values) for output, values in outcomes.outputs.items()})
    for where, (written_by, value) in written_alone.items():
        try:
            written_by.adapter.serializer.to_json(value)
        except PydanticSerializationError:
            return where
    return "the run"
