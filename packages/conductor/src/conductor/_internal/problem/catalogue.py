"""The problems conductor's core writes, one function each.

``PROBLEMS.problem`` says how a function's signature is read. A caller makes a problem by calling the function:
``not_a_chain(after_id, before=before_id, inputs=open_inputs, outputs=outputs)``.
"""

from __future__ import annotations

from typing import Any, Literal

from conductor._internal.problem.type import ProblemCatalogue, listed

#: The kinds of problem conductor's core writes.
PROBLEMS = ProblemCatalogue()


# -- what loading finds in a graph ---------------------------------------------


@PROBLEMS.problem()
def cycle(*loc: str | int) -> str:
    """A node is on a cycle of edges.

    Args:
        loc: The node.
    """
    return "The node is part of a cycle, so the graph cannot run; remove one of the edges that close it."


@PROBLEMS.problem()
def invalid_value(node_id: str, field: str, *, reason: str, errors: list[dict[str, Any]] | None = None, column: str | None = None) -> str:
    """A value in the record is not one its input's type takes.

    Args:
        node_id: The node.
        field: The input.
        reason: Why the value is refused, quoting nothing of it.
        errors: pydantic's account of the refusal, without the value.
        column: The column of a table the first fault is in.
    """
    return f"The value typed into '{field}' is not valid: {reason.rstrip('.')}. Type a value of the field's type."


@PROBLEMS.problem()
def type_mismatch(node_id: str, field: str, *, source: str, got: str, expected: str) -> str:
    """An edge brings a type its input doesn't take.

    Args:
        node_id: The node the edge goes into.
        field: The input.
        source: The address of the output the edge comes from.
        got: The type that output gives.
        expected: The id of the type the input takes.
    """
    return f"'{source}' gives {got}, but '{node_id}.{field}' takes {expected}; connect an output that gives {expected}, or convert it in a node between."


@PROBLEMS.problem("type_mismatch", hint="split_first")
def type_mismatch_split_first(node_id: str, field: str, *, source: str, got: str, expected: str, whole: Literal["table", "list"], fields: list[str]) -> str:
    """A whole table or list is wired into an input of one value.

    Args:
        node_id: The node the edge goes into.
        field: The input.
        source: The address of the output the edge comes from.
        got: The type that output gives.
        expected: The id of the type the input takes.
        whole: Which whole value the source is.
        fields: The fields of each record or item a Split would give; none when they aren't known.
    """
    return (
        f"'{source}' is a whole {whole} and '{node_id}.{field}' takes one {expected}. To run '{node_id}' once for each record or item "
        "in it, put a Split between them and wire Split's result, or one field of it."
    )


@PROBLEMS.problem("type_mismatch", hint="add_column")
def type_mismatch_add_column(node_id: str, field: str, *, source: str, got: str, expected: str, column: str, column_type: str, has: list[str]) -> str:
    """A table input is fed records without a column it names.

    Args:
        node_id: The node the edge goes into.
        field: The input.
        source: The address of the output the edge comes from.
        got: The type that output gives.
        expected: The id of the type the input takes.
        column: The first column the input names that the source lacks.
        column_type: The id of that column's type.
        has: The columns the source has.
    """
    return f"'{node_id}.{field}' needs the column '{column}' ({column_type}); '{source}' has {listed(has)}. Add the column in a node between, or connect a table that has it."


@PROBLEMS.problem()
def unknown_input(node_id: str, field: str, *, inputs: list[str]) -> str:
    """A value or an edge is for an input the node doesn't have.

    Args:
        node_id: The node.
        field: The name that is no input of it.
        inputs: The inputs the node has.
    """
    return f"'{node_id}' has no input '{field}'; it has {listed(inputs)}."


@PROBLEMS.problem()
def value_and_edge(node_id: str, field: str) -> str:
    """An input has an edge into it and a value in the record.

    Args:
        node_id: The node.
        field: The input.
    """
    return f"Field '{field}' has an edge and a typed-in value; keep one."


# -- what a call is refused for ------------------------------------------------


@PROBLEMS.problem()
def answer_stale(*loc: str | int, waiting_id: str, current: str, moved_to: str | None = None) -> str:
    """An answer names a question that another has replaced at the same place.

    Args:
        loc: The unit the answer is for.
        waiting_id: The id the answer was given for.
        current: The id of the question that waits there.
        moved_to: The id the answered question waits under, when it moved.
    """
    return f"The answer was given for {waiting_id!r}, and another question waits at that place by now, {current!r}; show that one and answer it."


@PROBLEMS.problem()
def cannot_move_to_version(*loc: str | int, node: str, version: int, target: int, versions: list[int]) -> str:
    """``registry.upgrade`` is asked to move a placed node to a version it can't reach from the one it is at.

    Args:
        loc: The placed node.
        node: The id of the node's definition.
        version: The version the node is placed at.
        target: The version asked for.
        versions: The versions the definition has.
    """
    return f"'{node}' cannot move from version {version} to {target}; it has {listed(versions)}."


@PROBLEMS.problem()
def from_run_missing(node_id: str, parameter: str, *, needs: str, given: list[str]) -> str:
    """A node takes a value from the run, and the call's ``from_run`` has none under that class.

    Args:
        node_id: The node.
        parameter: Its ``FromRun`` parameter.
        needs: The name of the class the parameter names.
        given: The names of the classes ``from_run`` has values for.
    """
    return f"'{node_id}' needs a {needs} for '{parameter}', and from_run gives none (it gives {listed(given)}); pass from_run={{{needs}: ...}}."


@PROBLEMS.problem()
def invalid_input(*loc: str | int, name: str, reason: str, errors: list[dict[str, Any]] | None = None, column: str | None = None) -> str:
    """A call gives an input a value its type doesn't take.

    Args:
        loc: ``"inputs"``, then the input's name.
        name: The input's name.
        reason: Why the value is refused, quoting nothing of it.
        errors: pydantic's account of the refusal, without the value.
        column: The column of a table the first fault is in.
    """
    return f"The value given for '{name}' is not valid: {reason.rstrip('.')}. Give a value of the type the input takes."


@PROBLEMS.problem("invalid_input", hint="make_a_series")
def invalid_input_make_a_series(*loc: str | int, name: str, expected: str, length: int) -> str:
    """A call gives a list to an input that takes one value.

    Args:
        loc: ``"inputs"``, then the input's name.
        name: The input's name.
        expected: The id of the type the input takes.
        length: How many items the list has.
    """
    return (
        f"'{name}' takes one {expected} and was given a list of {length}. A graph runs once per item only after a node that makes "
        "a Series, such as a list or an upload node: give the list to such a node, or give one value."
    )


@PROBLEMS.problem()
def invalid_record(*loc: str | int, at: str, reason: str, error: str) -> str:
    """What was given as a graph record is not one.

    Args:
        loc: ``"record"``, then where in the record the fault is.
        at: That place as one text, its steps joined by dots; ``the record`` for the record as a whole.
        reason: What is wrong there.
        error: The type of pydantic's error, such as ``extra_forbidden``.
    """
    return f"The record is not a graph record at '{at}': {reason.rstrip('.')}. Fix the record's JSON."


@PROBLEMS.problem()
def not_a_chain(node_id: str, *, before: str, inputs: list[str], outputs: list[str]) -> str:
    """``make_graph`` is given nodes without edges, and one can't be chained to the node before it.

    Args:
        node_id: The node that can't follow.
        before: The node before it.
        inputs: The required open inputs ``node_id`` has, where a chain takes exactly one.
        outputs: The outputs ``before`` has, where a chain takes exactly one.
    """
    return (
        f"'{node_id}' can't follow '{before}' in a chain: a chain takes its one required open input (it has {listed(inputs)}) "
        f"from the one output of the node before it (that has {listed(outputs)}); pass edges= to make_graph."
    )


@PROBLEMS.problem()
def not_offered(*loc: str | int, name: str, inputs: list[str]) -> str:
    """A call gives a value under a name that is no input of the graph.

    Args:
        loc: ``"inputs"``, then the name.
        name: The name the value was given under.
        inputs: The inputs the graph has.
    """
    return f"The graph has no input '{name}'; it has {listed(inputs)}."


@PROBLEMS.problem()
def not_waiting(*loc: str | int, waiting_id: str, status: str, moved_to: str | None = None) -> str:
    """An answer names a unit that isn't waiting for one.

    Args:
        loc: The unit the answer is for; the id's node alone when its place can't be read.
        waiting_id: The id the answer was given for.
        status: The run's status.
        moved_to: The id the answered question waits under, when it moved.
    """
    return f"Nothing is waiting on an answer as {waiting_id!r} in a run that is {status}; run.waiting lists what is."


# -- why a node or a run stopped -----------------------------------------------


@PROBLEMS.problem()
def execution_failed(*loc: str | int, error: str) -> str:
    """A node raised an exception that is no ``NodeError``; its text is in the ``conductor`` log.

    Args:
        loc: The unit that failed.
        error: The exception's class name.
    """
    return "The node failed; its developer finds why in the conductor log."


@PROBLEMS.problem()
def external_failed(*loc: str | int, error: str) -> str:
    """A node raised an exception its ``Policy(retry_on=)`` names, and no attempt is left.

    An ``ExternalError`` a node raises itself has this code too, with the node's own message and ``ctx``.

    Args:
        loc: The unit that failed.
        error: The exception's class name.
    """
    return "An outside service did not answer."


@PROBLEMS.problem()
def failed(*loc: str | int) -> str:
    """A node raised a ``NodeError`` and gave it no code of its own.

    The problem's message and ``ctx`` are the node's; this sentence is for a host that has none of its own for the code.

    Args:
        loc: The unit that failed.
    """
    return "The node reported a failure."


@PROBLEMS.problem()
def invalid_ask(*loc: str | int, reason: str) -> str:
    """A node raised an ``Ask`` its outputs can't be answered with.

    Args:
        loc: The unit that asked.
        reason: What is wrong with the question.
    """
    return f"The node asked a question its outputs can't be answered with: {reason.rstrip('.')}. Fix the node's Ask."


@PROBLEMS.problem()
def invalid_edge_value(
    *loc: str | int, name: str, reason: str, record_field: str | None = None, errors: list[dict[str, Any]] | None = None, column: str | None = None,
) -> str:
    """An edge brought a value its input doesn't take, met when the unit was to be called.

    Args:
        loc: The unit that was to be called.
        name: The input.
        reason: Why the value is refused, quoting nothing of it.
        record_field: The field of a record the edge reads, when the record has none by that name.
        errors: pydantic's account of the refusal, without the value.
        column: The column of a table the first fault is in.
    """
    return f"The value an edge brought to '{name}' is not one it takes: {reason.rstrip('.')}. Fix the node that made it, or connect an output of the type the input takes."


@PROBLEMS.problem()
def invalid_output(*loc: str | int, reason: str, name: str | None = None, errors: list[dict[str, Any]] | None = None, column: str | None = None) -> str:
    """A node returned what its return type doesn't declare.

    Args:
        loc: The unit that returned it.
        reason: Why the value is refused, quoting nothing of it.
        name: The output, when the fault is in one.
        errors: pydantic's account of the refusal, without the value.
        column: The column of a table the first fault is in.
    """
    return f"The node returned a value that is not what it declares: {reason.rstrip('.')}. Fix what the node returns."


@PROBLEMS.problem()
def item_ceiling(*loc: str | int, series: str, items: int, max_items: int) -> str:
    """A Series has more items than the run may work through, and the run stopped before any was started.

    Args:
        loc: The unit that made the Series.
        series: The address of the output that is the Series.
        items: How many items it has.
        max_items: The run's ceiling.
    """
    return f"{series} has {items} items, more than the {max_items} this run may work through; resume it with a larger max_items."


@PROBLEMS.problem()
def question_unwritable(*loc: str | int, reason: str, input: str) -> str:  # noqa: A002  the ctx key a host reads
    """A node asked a person, and an input it received has no JSON form to show them.

    Args:
        loc: The unit that asked.
        reason: Why the input can't be written.
        input: The input.
    """
    return f"The node asked a person, but what it received at '{input}' has no JSON form to show them: {reason.rstrip('.')}. Give that input a type pydantic writes as JSON."


@PROBLEMS.problem()
def run_cancelled() -> str:
    """The run's ``cancel`` was set while a leg ran."""
    return "The run was cancelled. Resume it to carry on, or start a new run."


@PROBLEMS.problem()
def run_timed_out(*, timeout: float) -> str:
    """A leg ran longer than the ``timeout`` it was given.

    Args:
        timeout: The seconds the leg was given.
    """
    return "The run took too long and was stopped; run.resume() carries on."


@PROBLEMS.problem()
def timeout(*loc: str | int, seconds: float) -> str:
    """A node's call took longer than its ``Policy(attempt_timeout=)``, and no attempt is left.

    Args:
        loc: The unit that failed.
        seconds: The policy's ``attempt_timeout``.
    """
    return "The node did not answer in time; its Policy(attempt_timeout=) sets how long it may take."
