"""The problems conductor's core writes, one function each.

``PROBLEMS.problem`` says how a function's signature is read. A caller makes a problem by calling the function:
``core_problems.not_a_chain(after_id, before=before_id, inputs=open_inputs, outputs=outputs)``.
"""

from __future__ import annotations

from conductor._internal.problem import ProblemCatalogue, listed

#: The kinds of problem conductor's core writes.
PROBLEMS = ProblemCatalogue()


# -- what a call is refused for ------------------------------------------------


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
