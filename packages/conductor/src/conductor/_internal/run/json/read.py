"""Reading a save: the units and the earlier work of a node, from their columns."""

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
    """The run of ``graph`` that ``run.to_json()`` wrote as ``saved``: ``Run.from_json(run.to_json(), graph)`` is ``run`` again.

    A graph with problems takes no run, so they are what is refused, before anything of the save is read. The
    save is read once, by the graph's ``RunCodec``, from its JSON text, so each value is read as its type reads
    JSON: a date, and a float that isn't finite, from their text.

    Raises:
        InvalidCallError: The graph's problems; ``saved_run_of_another_graph``; or ``saved_run_unreadable`` at the place
            pydantic names (the section, the node, the output, the slot), quoting nothing of a value; or a node
            that doesn't hold one slot for each instance the saved run made of it (``saved_run_unreadable``).
    """
    if graph.problems:
        raise InvalidCallError(graph.problems)
    try:
        read: SavedRun[list[Outcome | None], ReusableOutcomes] = graph._run_codec.validator.validate_json(saved)
    except ValidationError as error:
        first = error.errors(include_url=False, include_input=False)[0]
        if first["type"] == "union_tag_invalid" and not first["loc"]:  # a save is read as one of this graph, which its fingerprint says it is
            raise InvalidCallError([problem.saved_run_of_another_graph("state", saved=first["ctx"]["tag"], graph=graph.fingerprint)]) from None
        at = ".".join(str(step) for step in first["loc"][1:]) or "the run"  # a place in the save comes after the fingerprint it was read under
        raise InvalidCallError([problem.saved_run_unreadable("state", where=at, reason=value_free_errors(error)[0]["msg"])]) from None
    outcomes: dict[str, dict[ItemIndexes, Outcome]] = {}
    for node in graph._flat_function_nodes:  # each after the nodes it reads, so the Series its instances are of are in place
        outcomes[node.name] = {}
        if (node_outcomes := read.outcomes.get(node.name)) is not None:
            every_item_indexes, _ = all_item_indexes(outcomes, node.loops, ())
            if len(node_outcomes) != len(every_item_indexes):
                at, reason = f"outcomes.{node.name}", f"it holds {len(node_outcomes)} slots for the {len(every_item_indexes)} instances the saved run made of the node"
                raise InvalidCallError([problem.saved_run_unreadable("state", where=at, reason=reason)])
            outcomes[node.name] = {item_indexes: outcome for item_indexes, outcome in zip(every_item_indexes, node_outcomes, strict=True) if outcome is not None}
    functions = graph._functions
    code_changed = {path: digest for name, digest in read.code.items() if digest != functions[path := parse_node_path(name)].code_fingerprint}
    reusable = {path: of_node for name, of_node in read.reusable.items() if (path := parse_node_path(name)) not in code_changed}  # work of other code is never taken
    return run_type(
        graph, read.inputs, read.max_items, outcomes, reusable, state=read.state, _code_changed=code_changed,
        _finished_nodes=frozenset(parse_node_path(name) for name in read.complete),
    )


def outcomes_from_saved(skipped_outputs: dict[str, Any], not_called: dict[str, Any], saved: SavedOutcomes) -> list[Outcome | None]:
    """Makes the units a node's saved columns hold, in the order of their slots; ``None`` for an instance that
    hasn't run.

    Args:
        skipped_outputs: What a skipped unit of the node holds for its outputs.
        not_called: What a unit of the node that wasn't called for a blank holds for its outputs.
        saved: The columns.

    Raises:
        PydanticCustomError: An output doesn't hold one slot for each state (``slots_differ``); or a skip, a
            question or a blank input is placed on a slot that can't hold one (``misplaced``).
    """
    states, outputs = saved.states, saved.outputs
    if any(len(column) != len(states) for column in outputs.values()):
        raise PydanticCustomError("slots_differ", "an output doesn't hold one slot for each state")
    try:
        for name, call_numbers in (saved.skipped or {}).items():
            for call_number in call_numbers:
                outputs[name][call_number] = SKIPPED
        for name, by_call_number in (saved.skipped_items or {}).items():
            for call_number, places in by_call_number.items():
                for place in places:
                    outputs[name][call_number][place] = SKIPPED
        if len(outputs) == 1:  # most nodes have one output
            ((name, column),) = outputs.items()
            outs = [{name: value} for value in column]
        else:
            names = tuple(outputs)
            outs = [dict(zip(names, values, strict=True)) for values in zip(*outputs.values(), strict=True)] if outputs else [{} for _ in states]
        done, skipped = SAVED_DONE, SAVED_SKIPPED
        skipped_outcome = Skipped(skipped_outputs)
        outcomes: list[Outcome | None] = [Done(out) if state == done else skipped_outcome if state == skipped else None for state, out in zip(states, outs, strict=True)]
        for call_number, question in (saved.questions or {}).items():
            answered = outcomes[call_number]
            if states[call_number] is None:
                outcomes[call_number] = Waiting(question)
            elif type(answered) is Done:
                outcomes[call_number] = Answered(answered.outputs, question)
            else:
                raise LookupError(call_number)
        for call_number, blank in (saved.not_called_because or {}).items():
            if outcomes[call_number] is not None:
                raise LookupError(call_number)
            outcomes[call_number] = NotCalled(not_called, blank)
    except (LookupError, TypeError, AttributeError):  # a slot the columns don't have, or one that holds another kind of unit
        raise PydanticCustomError("misplaced", "a skip, a question or a blank input is placed on a slot that can't hold one") from None
    return outcomes


def with_shared_tables(outputs: Mapping[str, Mapping[str, Field]], saved: SavedRun[list[Outcome | None], ReusableOutcomes]) -> SavedRun[list[Outcome | None], ReusableOutcomes]:
    """Fills in the outputs a save wrote as ``null`` and listed under ``tables_written_once``.

    Example: the save lists ``{"output": ["split", "result", 0], "same_as": ["people", "result", 0]}``. The table
    read for ``people.result`` is validated by the type of ``split.result`` and stored there, so both outputs have
    the same table object again.

    Args:
        outputs: The outputs of each node, by the node's name in a save.
        saved: The save as read, each node's units in slot order.

    Raises:
        PydanticCustomError: An entry names an output or slot the save does not have, one that has no table
            written in full, one whose own column is not ``null`` there, or a table its own type refuses
            (``same_as_unreadable``).
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
    """Makes the earlier work a node's save holds.

    Raises:
        PydanticCustomError: It doesn't hold one fingerprint for each unit (``slots_differ``), or it carries a
            unit that was never called (``not_carried``).
    """
    if len(saved.fingerprints) != len(saved.outcomes):
        raise PydanticCustomError("slots_differ", "it doesn't hold one fingerprint for each unit")
    item_indexes, outer_item_fingerprints = saved.item_indexes or {}, saved.outer_item_fingerprints or {}
    previous = ReusableOutcomes()
    for call_number, (fingerprint, outcome) in enumerate(zip(saved.fingerprints, saved.outcomes, strict=True)):
        if type(outcome) is Done:
            previous.by_fingerprint.setdefault(fingerprint, outcome)
            if call_number in item_indexes:
                previous.by_item_indexes[item_indexes[call_number]] = (fingerprint, outcome)
        elif type(outcome) is Answered or type(outcome) is Waiting:
            previous.answers.setdefault((fingerprint, outer_item_fingerprints.get(call_number, "")), []).append(outcome)
        else:
            raise PydanticCustomError("not_carried", "it carries a unit that was never called")
    return previous
