"""One call of a node, whole: its inputs checked, its function run, what it gave back checked.

``call_node`` is a plain function that runs in the leg's worker thread and
touches nothing the leg owns: given a compiled node and a unit's inputs, it
says what came of the call. An ordinary failure is a value, not an exception.
The engine's loop keeps what is about scheduling (slots, retries, a policy's
timeout, events) and asks one thing of the outcome: which of the three it is.
A table a node returned is checked and given its rows' ids here too, so a slow
one holds up its own thread and not the leg.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, NamedTuple

from pydantic import ValidationError

from conductor._internal.errors import (
    ExternalFailure,
    NodeError,
    Problem,
    make_problem,
    value_free_errors,
)
from conductor._internal.execution.run_state import WaitingUnit
from conductor._internal.execution.validate_unit_outputs import validate_unit_outputs
from conductor._internal.execution.waiting_units import waiting_unit_for
from conductor._internal.index import Row
from conductor._internal.node_signature import split_return_into_outputs
from conductor._internal.skip_and_ask import AskPerson
from conductor._internal.table import is_frame, missing_column_in

if TYPE_CHECKING:
    from conductor._internal.graph.compiled_node import CompiledNode
    from conductor._internal.ref import Ref

logger = logging.getLogger("conductor")


class CallFailed(NamedTuple):
    """A call that came to nothing: why, and whether the policy may try the unit again.

    ``retryable`` is set only for what the outside world caused: an
    ``ExternalFailure``, or a foreign exception the policy's ``retry_on`` names."""

    problem: Problem
    retryable: bool = False


def call_node(
    node: CompiledNode, row: Row, inputs: dict[str, Any], from_run: Mapping[type, Any],
    row_ids_must_survive: frozenset[Ref] = frozenset(),
) -> dict[str, Any] | WaitingUnit | CallFailed:
    """Call ``node`` on ``inputs`` at ``row``: the cells it wrote, what it asked a person, or why it failed.

    The inputs are checked against the node's interface rather than its
    declaration, so what a ``compute_added_inputs`` hook added is checked like the
    rest. A ``NodeError`` keeps the message the node chose. Anything else
    raised, ``SystemExit`` and a stray ``CancelledError`` included, is the
    unit's failure: ``external_failed`` when the policy's ``retry_on`` names
    its class, ``execution_failed`` otherwise, with the exception in the log,
    its class name in ``ctx`` and its text on no event. ``row_ids_must_survive`` holds the tables
    a merge matches on the ids of the rows they were built from.
    """
    version = node.version
    try:
        try:
            kwargs = node._validate_call_inputs(inputs)
        except ValidationError as invalid:
            return CallFailed(_problem_of_invalid_inputs(node, row, invalid))
        kwargs.update({name: from_run[needed] for name, needed in version.from_run.items()})
        returned = version.body(**kwargs)
    except NodeError as raised:
        return CallFailed(raised._as_problem_at(node.id, *row), retryable=isinstance(raised, ExternalFailure))
    except version.policy.retry_on as raised:
        logger.warning("node %r: an outside service failed", node.id, exc_info=raised)
        return CallFailed(make_problem("external_failed", node.id, *row, exception=type(raised).__name__), retryable=True)
    except BaseException as raised:
        logger.error("node %r failed", node.id, exc_info=raised)
        return CallFailed(make_problem("execution_failed", node.id, *row, exception=type(raised).__name__))

    if isinstance(returned, AskPerson):
        asked = waiting_unit_for(node, row, returned, inputs)
        return _failed_on_its_outputs(node, row, asked) if isinstance(asked, list) else asked
    if row_ids_must_survive and not _took_a_row_in(node, inputs):
        row_ids_must_survive = frozenset()
    value_by_output = split_return_into_outputs(version.return_shape, returned, node.interface.outputs, version.interface.outputs)
    if isinstance(value_by_output, list):
        return _failed_on_its_outputs(node, row, value_by_output)
    cells, problems = validate_unit_outputs(node, row, value_by_output, row_ids_must_survive)
    return _failed_on_its_outputs(node, row, problems) if problems else cells


def _took_a_row_in(node: CompiledNode, inputs: Mapping[str, Any]) -> bool:
    """Did these inputs carry a row a returned table could keep the id of? A table, or an input of many values, with something in it."""
    many = {input_.name for input_ in node._inputs if input_.many}
    return any(len(value) > 0 for name, value in inputs.items() if is_frame(value) or (name in many and value is not None))


def _problem_of_invalid_inputs(node: CompiledNode, row: Row, invalid: ValidationError) -> Problem:
    """Why the inputs were refused, one line per field by its title; a table lacking a column names it."""
    titles = {input_.name: input_.title for input_ in node.interface.inputs}
    parts = [f"{titles.get(name, name)}: {error['msg']}" for error in value_free_errors(invalid) for name in error["loc"] or ("?",)]
    reason = "Invalid input — " + "; ".join(parts)
    column = missing_column_in(invalid)
    if column is not None:
        return Problem(type="table_column_missing", msg=reason, loc=(node.id, *row), ctx={"column": column})
    return Problem(type="invalid_input", msg=reason, loc=(node.id, *row), ctx={"reason": reason})


def _failed_on_its_outputs(node: CompiledNode, row: Row, problems: list[Problem]) -> CallFailed:
    """The failure of a node that returned what it did not declare, every output that did not fit in one reason."""
    reason = "; ".join(problem.msg for problem in problems)
    errors = [error for problem in problems for error in problem.ctx.get("errors", ())]
    code = next((problem.type for problem in problems if problem.type != "invalid_output"), "invalid_output")
    return CallFailed(make_problem(code, node.id, *row, reason=reason, **({"errors": errors} if errors else {})))
