"""One call of a node's function: the arguments validated where the graph's types leave a doubt, the function
called, and what it returns validated against the node's outputs.

Example: ``call(shout, {"line": "b", "mark": "!"}, {})`` returns ``({"result": "B!"}, None)``. When ``shout`` raises,
or returns something that is not a ``str``, it returns ``(None, the exception)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError
from pydantic_core import PydanticSerializationError

from conductor._internal.compiled import FlatFunctionNode
from conductor._internal.field import Field
from conductor._internal.gate import RefusedError
from conductor._internal.markers import SKIPPED
from conductor._internal.node_definition import ReturnShape
from conductor._internal.plain_json import plain_json
from conductor._internal.problem import catalogue as problem
from conductor._internal.value_type import untyped_as_json


def call(node: FlatFunctionNode, arguments: dict[str, Any], context: Mapping[type, Any]) -> tuple[Any, BaseException | None]:
    """Call the function of ``node`` with ``arguments``.

    Before the call, each input in ``node.validated_inputs`` is validated against its type: these are the inputs
    whose edge does not guarantee the type (the source is ``Any``, say). The ``Context`` values the function asks
    for are added to the arguments. After the call, what the function returned is validated against the node's
    outputs (``checked_outputs``).

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
        for name in node.validated_inputs:
            arguments = {**arguments, name: _checked(compiled_node.inputs[name].field, name, arguments[name], returned=False)}
        if definition.context:
            arguments = {**arguments, **{name: context[needed] for name, needed in definition.context.items()}}
        returned = definition.function(**arguments)
    except BaseException as error:  # noqa: BLE001  the scheduler turns it into a problem or a Waiting outcome
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        return None, error
    try:
        return checked_outputs(definition.returns, compiled_node.outputs, returned), None
    except RefusedError as refused:
        return None, refused


def checked_outputs(returns: ReturnShape, outputs: Mapping[str, Field], returned: Any) -> dict[str, Any]:
    """What a function returned, as the node's outputs by name, each validated against its type.

    Example: ``shout`` returns ``"B!"``, and the result is ``{"result": "B!"}``. A function that returns
    ``SKIPPED`` gets ``SKIPPED`` for every output.

    Validation is strict: a value that is already of the output's class is kept as it is, a dict is read as a
    model, and ``"12"`` is not an ``int``. Of the outputs in one exclusive group at most one may have a value;
    the others are ``SKIPPED``.

    Args:
        returns: How the function returns its outputs: one value, a dataclass, a mapping, or nothing.
        outputs: The node's outputs, by name.
        returned: What the function returned.

    Raises:
        RefusedError: An output is not of its type, or two outputs of one exclusive group have a value
            (problem ``invalid_output``).
    """
    if returned is SKIPPED:
        return dict.fromkeys(outputs, SKIPPED)
    spread, needs_no_validation = returns.spread(returned, outputs)
    if needs_no_validation:
        return spread
    groups: dict[str, str] = {}
    for name, field in outputs.items():
        spread[name] = _checked(field, name, spread[name], returned=True)
        group = field.param.exclusive_group
        if group is not None and spread[name] is not SKIPPED:
            if group in groups:
                reason = f"'{groups[group]}' and '{name}' are alternatives of '{group}', and both hold a value"
                raise RefusedError([problem.invalid_output(reason=reason, name=name)])
            groups[group] = name
    return spread


def _checked(field: Field, name: str, value: Any, *, returned: bool) -> Any:
    """``value`` validated against the type of the input or output ``name``. ``SKIPPED`` is returned as it is.

    Args:
        field: The input or output.
        name: Its name, for the problem.
        value: An argument that came over an edge, or a value the function returned.
        returned: True for a value the function returned, which is validated strictly.

    Raises:
        RefusedError: The type refuses the value: problem ``invalid_output`` for a returned value, else
            ``invalid_edge_value``. The problem does not quote the value.
    """
    if value is SKIPPED:
        return value
    try:
        read = field.whole.read_returned(value) if returned else field.whole.read(value)
    except ValidationError as error:
        refused = field.whole.refusal(error.errors(include_url=False, include_input=False))
        if returned:
            wrong = problem.invalid_output(reason=refused.reason, name=name, errors=refused.errors, column=refused.column)
        else:
            wrong = problem.invalid_edge_value(name=name, reason=refused.reason, errors=refused.errors, column=refused.column)
        raise RefusedError([wrong]) from None
    exact = field.whole.exact_classes
    if returned and exact and not isinstance(read, exact):
        reason = f"its type's own check gave a {type(read).__name__}, not a {field.whole.id}"
        raise RefusedError([problem.invalid_output(reason=reason, name=name)])
    return read


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
