"""One call of a node's function, with what goes in and what comes out checked.

The scheduler calls a node through ``call``. On the way in, the node's edge validator checks the arguments the
graph's types can't vouch for (see ``validator.py``). Then the function is called. On the way out, the node's
return validator turns what the function returned into the node's outputs.

For example, ``call(shout, {"line": "b", "mark": "!"}, None)`` returns ``({"result": "B!"}, None)``. If ``shout``
raises, it returns ``(None, the exception)``. If it returns something that isn't a ``str``, it returns ``(None, a
Problem)``, here ``invalid_output``.
"""

from __future__ import annotations

from typing import Any

from pydantic_core import PydanticSerializationError

from conductor._internal.base import plain_json
from conductor._internal.problem import InvalidCallError, Problem, run_problems
from conductor._internal.types import untyped_as_json

from .blueprint import BlueprintFunctionNode


def call(
    node: BlueprintFunctionNode, arguments: dict[str, Any], context: Any
) -> tuple[Any, BaseException | Problem | None]:
    """Calls the function of ``node`` with ``arguments``, and checks what goes in and what comes out.

    The arguments are passed by name. If the node takes the run's context, ``context`` is passed first.

    Before the call, the node's edge validator checks the inputs whose type asks for more than their edges promise.
    An edge from an output of type ``Any`` is one example. After the call, the node's return validator turns what
    the function returned into its outputs.

    The function's own exceptions are returned, not raised, so the scheduler can decide what to do with them. Only
    ``KeyboardInterrupt`` and ``SystemExit`` are raised.

    Args:
        node: The node to call.
        arguments: The call's arguments by input name, as ``arguments_for`` returns them.
        context: The run's context, already checked when the run started. Only a node that takes it gets it.

    Returns:
        One of three pairs:

        - ``(outputs, None)``: the call worked. ``outputs`` are the node's outputs by name, as the run stores them.
        - ``(None, problem)``: a validator refused an argument (``invalid_edge_value``) or what the function
          returned (``invalid_output``). The problem has no loc yet. The scheduler adds the call's.
        - ``(None, error)``: what the function raised. That includes an ``Ask``, which the scheduler checks next
          (a node that didn't declare ``asks=True`` fails with ``ask_not_declared``). It also includes an
          ``InvalidCallError`` raised inside the function, which counts as the node's own failure.
    """
    compiled_node = node.compiled_node
    definition = compiled_node.definition
    try:
        if (edge_validator := node.edge_validator) is not None:
            arguments = edge_validator(arguments)
        try:
            returned = (
                definition.function(**arguments)
                if definition.context is None
                else definition.function(context, **arguments)
            )
        except BaseException as error:  # noqa: BLE001  the scheduler turns it into a problem, or a call that waits for a person
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
            return None, error
        return compiled_node.return_validator(returned), None
    # The edge validator or the return validator refused. Each reports only its first fault.
    except InvalidCallError as refused:
        (problem,) = refused.problems
        return None, problem


def jsonable(arguments: dict[str, Any]) -> Any:
    """Returns a call's arguments as JSON values, to show the person a node asks.

    For example, ``jsonable({"line": "b", "mark": "!"})`` returns the same dict. ``inf`` and ``nan`` become the
    strings ``"Infinity"`` and ``"NaN"``.

    Raises:
        InvalidCallError: ``question_unwritable`` if an argument can't be turned into JSON. The problem names the
            first input that can't.
    """
    try:
        return plain_json(arguments, fallback=untyped_as_json)
    except (TypeError, ValueError, PydanticSerializationError) as error:
        for name, value in arguments.items():
            try:
                plain_json(value, fallback=untyped_as_json)
            except (TypeError, ValueError, PydanticSerializationError) as unwritable:
                raise InvalidCallError(
                    [run_problems.question_unwritable(reason=str(unwritable), input=name)]
                ) from None
        raise error
