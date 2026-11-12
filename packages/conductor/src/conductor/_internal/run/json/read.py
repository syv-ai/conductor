"""Reads a run from the JSON that ``run.to_json()`` wrote.

``run_from_json`` is the door: the graph's codec validates the whole text once, and what comes out is trusted.
The other three functions are called by the codec while it validates, to turn each saved shape into what a run
keeps in memory.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError
from pydantic_core import PydanticCustomError

from conductor._internal.address import ItemIndexes, parse_node_path
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
)
from conductor._internal.problem import InvalidCallError, value_free_errors
from conductor._internal.problem import catalogue as problem
from conductor._internal.run.json.type import (
    SAVED_DONE,
    SAVED_SKIPPED,
    SavedOutcomes,
    SavedReusable,
    SavedRun,
)
from conductor._internal.run.reuse import ReusableOutcomes
from conductor._internal.value_type import FieldValueError

if TYPE_CHECKING:
    from conductor._internal.graph import Graph
    from conductor._internal.run.type import Run


def run_from_json(run_type: type[Run], saved: str | bytes, graph: Graph) -> Run:
    """Returns the run of ``graph`` that ``run_to_json`` wrote as ``saved``.

    For the README graph, ``run_from_json(Run, run.to_json(), graph)`` returns a run with the inputs, the
    outcomes and the state of ``run``: its ``results["shout"]["result"]`` is ``["A!", "B!", "C!"]`` again.

    The graph's codec validates the JSON text in one pass, and reads each value the way its type reads JSON: a
    date from its text, for example. The codec returns each node's outcomes as a list in call-number order,
    because item indexes are not saved. They are worked out here, node by node in topological order: a node's
    item indexes follow from the lengths of the series upstream of it, and those are loaded by then.

    Args:
        run_type: The class ``Run``. It is passed in because ``run/type.py`` imports this package.
        saved: The JSON text.
        graph: The graph the run was made on.

    Raises:
        InvalidCallError: ``graph`` has problems (they are raised, and nothing of ``saved`` is read); ``saved``
            was written for another graph (``saved_run_of_another_graph``); or ``saved`` is not a saved run of
            this graph (``saved_run_unreadable``, whose ``where`` names the key, the node, the output and the
            call number, and which quotes no value).
    """
    if graph.problems:
        raise InvalidCallError(graph.problems)
    try:
        read: SavedRun[list[Outcome | None], ReusableOutcomes] = graph._run_codec.validator.validate_json(saved)
    except ValidationError as error:
        first = error.errors(include_url=False, include_input=False)[0]
        if first["type"] == "union_tag_invalid" and not first["loc"]:  # the fingerprint under "graph" is not this graph's
            raise InvalidCallError([problem.saved_run_of_another_graph("saved_run", saved=first["ctx"]["tag"], graph=graph.fingerprint)]) from None
        where = ".".join(str(step) for step in first["loc"][1:]) or "the run"  # pydantic's loc starts with the graph's fingerprint
        raise InvalidCallError([problem.saved_run_unreadable("saved_run", where=where, reason=value_free_errors(error)[0]["msg"])]) from None
    outcomes: dict[str, dict[ItemIndexes, Outcome]] = {}
    for node in graph._flat_function_nodes:
        outcomes[node.name] = {}
        if (in_call_order := read.outcomes.get(node.name)) is not None:
            every_item_indexes, _ = all_item_indexes(outcomes, node.loops, ())
            if len(in_call_order) != len(every_item_indexes):
                reason = f"its lists have {len(in_call_order)} entries, but the saved series give the node {len(every_item_indexes)} calls"
                raise InvalidCallError([problem.saved_run_unreadable("saved_run", where=f"outcomes.{node.name}", reason=reason)])
            outcomes[node.name] = {
                item_indexes: outcome for item_indexes, outcome in zip(every_item_indexes, in_call_order, strict=True) if outcome is not None
            }
    functions = graph._functions
    code_changed = {
        path: fingerprint for name, fingerprint in read.code.items() if fingerprint != functions[path := parse_node_path(name)].code_fingerprint
    }
    # outcomes that other code produced are never reused
    reusable = {path: of_node for name, of_node in read.reusable.items() if (path := parse_node_path(name)) not in code_changed}
    return run_type(
        graph, read.inputs, read.max_items, outcomes, reusable, state=read.state, _code_changed=code_changed,
        _finished_nodes=frozenset(parse_node_path(name) for name in read.complete),
    )


def outcomes_from_saved(skipped_outputs: dict[str, Any], not_called_outputs: dict[str, Any], saved: SavedOutcomes) -> list[Outcome | None]:
    """Returns one node's outcomes from their saved lists, in call-number order; ``None`` for a call with no outcome.

    For ``shout`` not called for its second line::

        {"states": ["done", null, "done"], "outputs": {"result": ["A!", null, "C!"]}, "not_called_because": {"1": "line"}}

    it returns ``[Done({"result": "A!"}), NotCalled({"result": None}, "line"), Done({"result": "C!"})]``.

    Args:
        skipped_outputs: The ``outputs`` a ``Skipped`` outcome of the node has.
        not_called_outputs: The ``outputs`` a ``NotCalled`` outcome of the node has.
        saved: The saved lists, each value already validated by its output's type.

    Raises:
        PydanticCustomError: The list of an output is not as long as ``states`` (``lengths_differ``); or
            ``skipped``, ``skipped_items``, ``questions`` or ``not_called_because`` names a call that does not
            exist or that cannot have it (``misplaced``).
    """
    states, outputs = saved.states, saved.outputs
    if any(len(values) != len(states) for values in outputs.values()):
        raise PydanticCustomError("lengths_differ", "the list of an output is not as long as 'states'")
    try:
        for name, call_numbers in (saved.skipped or {}).items():
            for call_number in call_numbers:
                outputs[name][call_number] = SKIPPED
        for name, by_call_number in (saved.skipped_items or {}).items():
            for call_number, indexes in by_call_number.items():
                for index in indexes:
                    outputs[name][call_number][index] = SKIPPED
        if len(outputs) == 1:  # speed only: most nodes have one output
            ((name, values),) = outputs.items()
            outputs_by_call = [{name: value} for value in values]
        elif outputs:
            names = tuple(outputs)
            outputs_by_call = [dict(zip(names, values, strict=True)) for values in zip(*outputs.values(), strict=True)]
        else:
            outputs_by_call = [{} for _ in states]
        done, skipped = SAVED_DONE, SAVED_SKIPPED
        skipped_outcome = Skipped(skipped_outputs)
        outcomes: list[Outcome | None] = [
            Done(of_call) if state == done else skipped_outcome if state == skipped else None
            for state, of_call in zip(states, outputs_by_call, strict=True)
        ]
        for call_number, question in (saved.questions or {}).items():
            outcome = outcomes[call_number]
            if states[call_number] is None:
                outcomes[call_number] = Waiting(question)
            elif type(outcome) is Done:
                outcomes[call_number] = Answered(outcome.outputs, question)
            else:
                raise LookupError(call_number)
        for call_number, input_name in (saved.not_called_because or {}).items():
            if outcomes[call_number] is not None:
                raise LookupError(call_number)
            outcomes[call_number] = NotCalled(not_called_outputs, input_name)
    except (LookupError, TypeError, AttributeError):  # a call number past the end of the lists, or a call of another kind
        message = "'skipped', 'skipped_items', 'questions' or 'not_called_because' names a call that does not exist or cannot have it"
        raise PydanticCustomError("misplaced", message) from None
    return outcomes


def with_shared_tables(
    outputs: Mapping[str, Mapping[str, Field]], saved: SavedRun[list[Outcome | None], ReusableOutcomes],
) -> SavedRun[list[Outcome | None], ReusableOutcomes]:
    """Fills in the outputs that a saved run wrote as ``null`` and listed under ``tables_written_once``.

    The saved run lists ``{"output": ["split", "result", 0], "same_as": ["people", "result", 0]}``. The table
    read for ``people.result`` is validated by the type of ``split.result`` and put there, so both outputs have
    the same table object again.

    Args:
        outputs: The outputs of each node, by the node's name in a saved run.
        saved: The saved run as read, with each node's outcomes in call-number order.

    Raises:
        PydanticCustomError: An entry names an output or a call number the saved run does not have, an output
            with no table written in full, an output that is not ``null``, or a table the output's own type
            refuses (``same_as_unreadable``).
    """
    for entry in saved.tables_written_once or ():
        (node, output, call_number), (from_node, from_output, from_call_number) = entry.output, entry.same_as
        try:
            outcome, source = saved.outcomes[node][call_number], saved.outcomes[from_node][from_call_number]
            if not isinstance(outcome, (Done, Answered)) or not isinstance(source, (Done, Answered)) or outcome.outputs[output] is not None:
                raise LookupError(call_number)
            table = outputs[from_node][from_output].whole.written_once(source.outputs[from_output])
            if table is None:
                raise LookupError(from_call_number)
            outcome.outputs[output] = outputs[node][output].whole.read_returned(table)
        except (LookupError, AttributeError, FieldValueError) as wrong:
            reason = wrong.reason if isinstance(wrong, FieldValueError) else "that is not an output written in full before it"
            raise PydanticCustomError(
                "same_as_unreadable", "tables_written_once: '{output}' can't be read as the table at '{same_as}': {reason}",
                {"output": f"{node}.{output}[{call_number}]", "same_as": f"{from_node}.{from_output}[{from_call_number}]", "reason": reason},
            ) from None
    return saved


def reusable_from_saved(saved: SavedReusable[list[Outcome | None]]) -> ReusableOutcomes:
    """Returns the outcomes one node may reuse from the previous run, from their saved shape.

    The three saved outcomes of ``shout`` with ``"item_indexes": {"0": [0], "1": [1], "2": [2]}`` come back as a
    ``ReusableOutcomes`` that has each of them twice: by the item indexes its call had, and by the fingerprint
    of its call's arguments.

    Raises:
        PydanticCustomError: ``fingerprints`` is not as long as the lists of ``outcomes`` (``lengths_differ``);
            or an outcome is not one a call produced: it is skipped, not called or missing (``not_reusable``).
    """
    if len(saved.fingerprints) != len(saved.outcomes):
        raise PydanticCustomError("lengths_differ", "'fingerprints' is not as long as the lists of 'outcomes'")
    item_indexes, outer_item_fingerprints = saved.item_indexes or {}, saved.outer_item_fingerprints or {}
    reusable = ReusableOutcomes()
    for place, (fingerprint, outcome) in enumerate(zip(saved.fingerprints, saved.outcomes, strict=True)):
        if type(outcome) is Done:
            reusable.by_fingerprint.setdefault(fingerprint, outcome)
            if place in item_indexes:
                reusable.by_item_indexes[item_indexes[place]] = (fingerprint, outcome)
        elif type(outcome) is Answered or type(outcome) is Waiting:
            reusable.answers.setdefault((fingerprint, outer_item_fingerprints.get(place, "")), []).append(outcome)
        else:
            raise PydanticCustomError("not_reusable", "an outcome is skipped, not called or missing, and only an outcome of a call is reused")
    return reusable
