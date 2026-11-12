"""One call of a node's function: the arguments read by the node's edge door where the graph's types leave a
doubt, the function called, and what it returns read into the node's outputs by its return door.

Example: ``call(shout, {"line": "b", "mark": "!"}, None)`` returns ``({"result": "B!"}, None)``. When ``shout`` raises,
or returns something that is not a ``str``, it returns ``(None, the exception)``.

A node with an ``Asked`` parameter, ``approve(invoice, ok: Annotated[bool, Asked(approval)])``, runs the function the
flat graph chose for it (``AskedParameter.filling``): ``ok`` from the arguments when a person's answer is there, and
else from ``approval(invoice=...)``. When ``approval`` raises ``Ask``, ``approve`` is not called and the ``Ask`` is
returned.
"""

from __future__ import annotations

from typing import Any

from pydantic_core import PydanticSerializationError

from conductor._internal.compiled import FlatFunctionNode
from conductor._internal.gate import RefusedError
from conductor._internal.plain_json import plain_json
from conductor._internal.problem import catalogue as problem
from conductor._internal.value_type import untyped_as_json


def call(node: FlatFunctionNode, arguments: dict[str, Any], context: Any) -> tuple[Any, BaseException | None]:
    """Call the function of ``node`` with ``arguments``, by name, and ``context`` first when the node takes it.

    Before the call, the node's edge door reads the inputs whose type demands more than their edges' sources
    declare (a source of ``Any``, say), each by its type, and an ``Asked`` parameter not in ``arguments`` gets its
    function's return value. After the call, the node's return door reads what the function returned into its
    outputs.

    Args:
        node: The node to call.
        arguments: The call's arguments by input name, as ``arguments_for`` returns them, and the answer to the
            node's ``Asked`` parameter by the parameter's name when a person gave one.
        context: The run's context, as the context door let it in; passed only to a node that takes it.

    Returns:
        ``(outputs, None)``: the node's outputs by name, as a run keeps them. Or ``(None, error)``: what the
        function raised, an ``Ask`` included (the node's ``Ask`` door, in the scheduler, refuses one the node did
        not declare: ``ask_not_declared``); ``RefusedError`` for an argument or an output that its type refuses.
        ``KeyboardInterrupt`` and ``SystemExit`` are raised, not returned.
    """
    compiled_node = node.compiled_node
    definition = compiled_node.definition
    try:
        if (edge_door := node.edge_door) is not None:
            arguments = edge_door(arguments)
        returned = node.function(**arguments) if definition.context is None else node.function(context, **arguments)
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
