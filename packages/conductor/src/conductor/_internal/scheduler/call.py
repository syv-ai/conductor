"""One call of a node's function: the arguments read by the node's edge door where the graph's types leave a
doubt, the function called, and what it returns read into the node's outputs by its return door.

Example: ``call(shout, {"line": "b", "mark": "!"}, {})`` returns ``({"result": "B!"}, None)``. When ``shout`` raises,
or returns something that is not a ``str``, it returns ``(None, the exception)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic_core import PydanticSerializationError

from conductor._internal.compiled import FlatFunctionNode
from conductor._internal.gate import RefusedError
from conductor._internal.plain_json import plain_json
from conductor._internal.problem import catalogue as problem
from conductor._internal.value_type import untyped_as_json


def call(node: FlatFunctionNode, arguments: dict[str, Any], context: Mapping[type, Any]) -> tuple[Any, BaseException | None]:
    """Call the function of ``node`` with ``arguments``.

    Before the call, the node's edge door reads the inputs whose type demands more than their edges' sources
    declare (a source of ``Any``, say), each by its type. The ``Context`` values the function asks
    for are added to the arguments. After the call, the node's return door reads what the function returned into
    its outputs.

    Args:
        node: The node to call.
        arguments: The call's arguments by input name, as ``arguments_for`` returns them.
        context: The value of each ``Context`` type.

    Returns:
        ``(outputs, None)``: the node's outputs by name, as a run keeps them. Or ``(None, error)``: what the
        function raised (an ``Ask`` included), or ``RefusedError`` for an argument or an output that its type
        refuses. ``KeyboardInterrupt`` and ``SystemExit`` are raised, not returned.
    """
    compiled_node = node.compiled_node
    definition = compiled_node.definition
    try:
        if (edge_door := node.edge_door) is not None:
            arguments = edge_door(arguments)
        if definition.context:
            arguments = {**arguments, **{name: context[needed] for name, needed in definition.context.items()}}
        returned = definition.function(**arguments)
    except BaseException as error:  # noqa: BLE001  the scheduler turns it into a problem or a Waiting outcome
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return None, error
    try:
        return compiled_node.return_door(returned), None
    except RefusedError as refused:
        return None, refused


def jsonable(arguments: dict[str, Any]) -> Any:
    """A call's arguments as JSON values, to show a person who is asked a question.

    Example: ``jsonable({"line": "b", "mark": "!"})`` returns the same dict. ``inf`` and ``nan`` become the
    strings ``"Infinity"`` and ``"NaN"``.

    Raises:
        RefusedError: An argument has no JSON form (problem ``question_unwritable``, which names the first
            such input).
    """
    try:
        return plain_json(arguments, fallback=untyped_as_json)
    except (TypeError, ValueError, PydanticSerializationError) as error:
        for name, value in arguments.items():
            try:
                plain_json(value, fallback=untyped_as_json)
            except (TypeError, ValueError, PydanticSerializationError) as unwritable:
                raise RefusedError([problem.question_unwritable(reason=str(unwritable), input=name)]) from None
        raise error
