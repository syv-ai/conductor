"""The problems conductor's core writes, one function each.

``@PROBLEMS.problem()`` makes the function under it a kind of problem: its name is the code, its positional
parameters are the places of ``loc``, its keyword-only parameters are the keys of ``ctx``, and what it returns is
the English. Calling it builds the ``Problem``, ``problem.unknown_input("upper", "txet", inputs=["text"])``, so a
call that leaves a key out or gives one the wrong type is a type error where it is written.
``ProblemCatalogue.problem`` has the rules in full.
"""

from __future__ import annotations

from typing import Any, Literal

from conductor._internal.problem.type import ProblemCatalogue, listed

#: The kinds of problem conductor's core writes.
PROBLEMS = ProblemCatalogue()


# -- what loading finds in a graph ---------------------------------------------


@PROBLEMS.problem()
def alternatives_in_different_series(node_id: str, field: str, *, series: list[str | None]) -> str:
    """An input takes alternatives that run per item of different Series.

    Args:
        node_id: The node.
        field: The input.
        series: The Series the alternatives run per item of, ``None`` for one that runs once.
    """
    said = listed(["none (it runs once)" if one is None else one for one in series])
    return (
        f"'{field}' takes alternatives that run per item of different Series, {said}, so a call for an item of one has no place "
        "for the other's value. Bring them to items of one Series first."
    )


@PROBLEMS.problem(of_a_hook=True)
def compute_hook_failed(*loc: str | int, reason: str, error: str, error_ctx: dict[str, Any] | None = None) -> str:
    """A node's ``compute_...`` hook raised, or returned what loading can't take.

    Args:
        loc: The node.
        reason: Why the hook failed.
        error: The code of the hook's ``NodeError``, or the class name of the exception it raised.
        error_ctx: The ``ctx`` of the hook's ``NodeError``, when it has one.
    """
    return f"The node could not work out its fields from what it was given: {reason.rstrip('.')}. Check what is typed into the node and what is wired to it."


@PROBLEMS.problem()
def cycle(*loc: str | int) -> str:
    """A node is on a cycle of edges.

    Args:
        loc: The node.
    """
    return "The node is part of a cycle, so the graph cannot run; remove one of the edges that close it."


@PROBLEMS.problem()
def default_on_later_node(node_id: str, field: str) -> str:
    """A record gives a default to an input that is not an open input of the graph.

    Args:
        node_id: The node.
        field: The input.
    """
    return f"'{field}' is not an open input of the graph, so a caller can't change its default; make it a value."


@PROBLEMS.problem()
def duplicate_node_id(node_id: str) -> str:
    """Two nodes of the record have the same id.

    Args:
        node_id: The id.
    """
    return f"Two nodes have the id '{node_id}'; rename one."


@PROBLEMS.problem()
def edge_names_missing_node(*loc: str | int, missing: str, source: str, target: str) -> str:
    """An edge names a node the record doesn't have.

    Args:
        loc: The node the edge goes into and its input.
        missing: The id of the node that is not there.
        source: The address the edge comes from.
        target: The address the edge goes into.
    """
    return f"The edge from '{source}' to '{target}' names '{missing}', which is not a node of the graph; add that node or remove the edge."


@PROBLEMS.problem()
def embeds_itself(*loc: str | int, node: str) -> str:
    """A node is a graph that contains itself.

    Args:
        loc: The node.
        node: The id of the graph it embeds.
    """
    return f"The node '{node}' is a graph that contains itself; embed a graph that does not contain this one."


@PROBLEMS.problem()
def hook_needs_value(node_id: str, field: str) -> str:
    """A ``compute_...`` hook reads an input that has no value in the record.

    Args:
        node_id: The node.
        field: The input.
    """
    return f"'{field}' must be typed in: the node works out its fields from it when the graph loads."


@PROBLEMS.problem()
def invalid_node_id(node_id: str) -> str:
    """A node's id holds a character that means something in an address or a waiting id.

    Args:
        node_id: The id.
    """
    return f"'{node_id}' cannot be a node id: '.', '/', '@' and '~' each mean something in an address or a waiting id; rename it without them."


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


@PROBLEMS.problem(of_a_hook=True)
def parameter_name_invalid(node_id: str, field: str) -> str:
    """A ``compute_...`` hook returns a field name that cannot be one.

    Args:
        node_id: The node.
        field: The name.
    """
    return f"'{field}' cannot be a field name: '.', '/' and '@' each mean something in an address, and a name can't be empty; rename the field."


@PROBLEMS.problem()
def repeated_edge(node_id: str, field: str, *, source: str) -> str:
    """The same edge is drawn twice.

    Args:
        node_id: The node the edge goes into.
        field: The input.
        source: The address the edge comes from.
    """
    return f"The edge from '{source}' into '{field}' is drawn twice; remove one."


@PROBLEMS.problem()
def series_inputs_never_share_an_item(*loc: str | int, inputs: list[str]) -> str:
    """Two inputs of a node read branches that never both have a value for an item.

    Args:
        loc: The node.
        inputs: The two inputs.
    """
    return (
        f"{listed(inputs)} read branches that never both have a value for an item, so no item would reach the node. "
        "Write Series[X | None] on them to keep every item, or take the branches through one input."
    )


@PROBLEMS.problem()
def series_not_nested(node_id: str, *, series: list[str], inputs: list[str], wire_from: str, wire_into: str) -> str:
    """A node reads items of two Series, and neither is made inside the other.

    Args:
        node_id: The node.
        series: The two Series, by the address of each one's output.
        inputs: The two inputs that read them.
        wire_from: The address of the output of the Series placed later.
        wire_into: The id of the node that makes the other Series.
    """
    return (
        f"'{node_id}' reads items of {listed(series)}, and neither is made inside the other, so nothing says which item goes with which. "
        f"Make one inside the other (wire '{wire_from}' into '{wire_into}', the node that makes the other), or give one of them whole to a Series input."
    )


@PROBLEMS.problem()
def several_edges_into_one_input(node_id: str, field: str, *, sources: list[str]) -> str:
    """More than one edge into an input can hold a value on the same call.

    Args:
        node_id: The node.
        field: The input.
        sources: The addresses the edges come from.
    """
    return (
        f"More than one of the edges into '{field}', from {listed(sources)}, can hold a value on the same call. "
        "Feed it from opposite branches of one decision, or put a node between that has an input for each."
    )


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
def unbound_on_later_node(node_id: str, field: str) -> str:
    """An input of a node that is not the first has no edge and no value.

    Args:
        node_id: The node.
        field: The input.
    """
    return f"Nothing feeds '{field}', and only a first node takes inputs from the caller; draw an edge into it or type a value in."


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
def unknown_node(*loc: str | int, node: str, choices: list[str]) -> str:
    """A node names a definition the registry doesn't have.

    Args:
        loc: The node.
        node: The id of the definition it names.
        choices: The ids the registry has.
    """
    return f"There is no node '{node}'; the nodes are {listed(choices)}."


@PROBLEMS.problem()
def unknown_output(*loc: str | int, source: str, outputs: list[str], record_field: str | None = None) -> str:
    """An edge reads an output, or a field of one, that its source doesn't give.

    Args:
        loc: The node the edge goes into and its input.
        source: The address the edge comes from.
        outputs: What the source gives.
        record_field: The field of a record the edge reads, when the record has none by that name.
    """
    return f"'{source}' is not something that node gives; it gives {listed(outputs)}."


@PROBLEMS.problem("unknown_output", hint="split_first")
def unknown_output_split_first(*loc: str | int, source: str, outputs: list[str], record_field: str, whole: Literal["table", "list"], fields: list[str]) -> str:
    """An edge reads a field of a whole table or list.

    Args:
        loc: The node the edge goes into and its input.
        source: The address the edge comes from.
        outputs: What the source gives.
        record_field: The field the edge reads.
        whole: Which whole value the source is.
        fields: The fields of each record or item a Split would give; none when they aren't known.
    """
    return (
        f"'{source}' names a field of a whole {whole}, which has no field to wire. Split it first: put a Split after it and wire "
        "the field from Split's result."
    )


@PROBLEMS.problem()
def unknown_version(*loc: str | int, node: str, version: int, newest: int) -> str:
    """A node is placed at a version its definition doesn't have.

    Args:
        loc: The node.
        node: The id of its definition.
        version: The version it is placed at.
        newest: The newest version the definition has.
    """
    return f"'{node}' has no version {version}; the newest is {newest}."


@PROBLEMS.problem()
def value_and_default(node_id: str, field: str) -> str:
    """An input has a value and a default in the record.

    Args:
        node_id: The node.
        field: The input.
    """
    return f"Field '{field}' has both a value and a default; keep one."


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
def input_given_twice(*loc: str | int, name: str, address: str) -> str:
    """A call gives an input by its name and by its address.

    Args:
        loc: ``"inputs"``, then the input's name.
        name: The input's name.
        address: The input's address.
    """
    return f"'{name}' is given twice, by its name and by its address '{address}'; give it once."


@PROBLEMS.problem()
def invalid_answer(*loc: str | int, waiting_id: str, reason: str, question: str | None = None) -> str:
    """An answer is not one the question it is for takes.

    Args:
        loc: The unit the answer is for.
        waiting_id: The id the answer was given for.
        reason: What is wrong with the answer.
        question: The question the answer is for, when the fault is about it.
    """
    return f"That is not a valid answer: {reason.rstrip('.')}"


@PROBLEMS.problem()
def invalid_checkpoint_every(*loc: str | int, checkpoint_every: float) -> str:
    """A call gives ``checkpoint_every`` as something other than a number of seconds above 0.

    Args:
        loc: ``"checkpoint_every"``.
        checkpoint_every: The value given.
    """
    return f"checkpoint_every is a number of seconds above 0; it was {checkpoint_every}."


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
def invalid_max_items(*loc: str | int, max_items: int) -> str:
    """A call gives ``max_items`` below 1.

    Args:
        loc: ``"max_items"``.
        max_items: The value given.
    """
    return f"max_items is at least 1; it was {max_items}."


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
def invalid_timeout(*loc: str | int, timeout: float) -> str:
    """A call gives ``timeout`` as something other than a number of seconds above 0.

    Args:
        loc: ``"timeout"``.
        timeout: The value given.
    """
    return f"timeout is a number of seconds above 0; it was {timeout}."


@PROBLEMS.problem()
def missing_input(*loc: str | int, name: str) -> str:
    """A call leaves out an input the graph requires.

    Args:
        loc: ``"inputs"``, then the input's name.
        name: The input's name.
    """
    return f"'{name}' is required; give it in the call."


@PROBLEMS.problem()
def node_not_in_graph(node_id: str, *, nodes: list[str]) -> str:
    """A call names a node the graph doesn't have.

    Args:
        node_id: The id given.
        nodes: The ids the graph has.
    """
    return f"'{node_id}' is not a node of this graph; it has {listed(nodes)}."


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


@PROBLEMS.problem()
def run_not_complete(*loc: str | int, status: str, reason: str) -> str:
    """``outputs`` is read of a run that isn't complete.

    Args:
        loc: ``"outputs"``.
        status: The run's status.
        reason: Why the run stopped, empty when it is still going.
    """
    return f"The run is {status}, so it has no outputs yet; run.results has what it did. {reason.rstrip('.')}".rstrip()


@PROBLEMS.problem()
def state_of_another_graph(*loc: str | int, saved: str, graph: str) -> str:
    """A saved run is of another graph than the one it is loaded into.

    Args:
        loc: ``"state"``.
        saved: The fingerprint the save names.
        graph: The fingerprint of the graph.
    """
    return f"The saved run is of another graph ({saved}) than this one ({graph}); an edited graph starts a new run."


@PROBLEMS.problem()
def state_unreadable(*loc: str | int, at: str, reason: str) -> str:
    """A saved run doesn't fit the graph it is loaded into.

    Args:
        loc: ``"state"``.
        at: The place in the save, its steps joined by dots; ``the run`` for the save as a whole.
        reason: What is wrong there.
    """
    return f"The saved run does not fit this graph at '{at}': {reason.rstrip('.')}. Load it into the graph it was saved from."


@PROBLEMS.problem()
def state_unwritable(*loc: str | int, at: str, reason: str) -> str:
    """A run holds a value that can't be written as JSON.

    Args:
        loc: ``"state"``, or ``"outputs"`` and the output's address.
        at: Where the value is.
        reason: Why it can't be written.
    """
    return f"The run can't be written as JSON at '{at}': {reason.rstrip('.')}. Give that field a type pydantic writes as JSON."


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


#: Every code a ``Problem`` conductor's core writes can have, for a host that translates by code to cover.
PROBLEM_CODES: frozenset[str] = PROBLEMS.codes
